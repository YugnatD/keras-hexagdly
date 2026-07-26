#ifndef NNET_HEX_CONV_LINEBUFFER_3D_STREAM_H_
#define NNET_HEX_CONV_LINEBUFFER_3D_STREAM_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_hex_conv_linebuffer_3d.h"
#include "nnet_types.h"

// Depth-aware hexagonal conv, line-buffer io_stream kernel.
//
// The depth axis is line-buffered: only depth_size(+1 lag slot) input frames are
// kept resident in a circular frame buffer (vs the gather path, which streams
// D_in frames of the K x-inflated gathered tensor), and the K x gathered blow-up
// is removed entirely.
//
// This is the OPTIMAL memory point for streaming 3D conv, not a compromise.
// Computing output frame t needs input frames t .. t+depth_size-1; in a raster
// (d, h, w) stream those frames are N = H*W tokens apart, so having them
// simultaneously forces buffering >= (depth_size-1)*N tokens — an incompressible
// lower bound.  Unlike the 2D case (which reaches O(rows) because a pixel needs
// only consecutive rows), 3D cannot go below O(depth_size * N); this kernel sits
// exactly at that floor (depth_size full frames, +1 for pipelining -- see below).
// Reference note: stock hls4ml has no Conv3D at all — there is no upstream 3D
// line buffer to mirror.
//
// Fused, single-loop, per-pixel design (mirrors nnet_hex_conv_linebuffer_stream.h
// exactly, "row" -> "frame", "column" -> "spatial pixel", stride fixed at 1):
// output frame t_out is fully unlocked only once input frame t_out+depth_size-1
// has been ENTIRELY read (spatial neighbor taps can reach any pixel of that
// frame, not just the one at the same position, so partial-frame reads are not
// enough — this is the frame-level analogue of 2D's row-level atomicity). Its
// N spatial pixels are then emitted one-per-cycle, INTERLEAVED with reading the
// NEXT frame's N pixels (1 read + 1 emit per cycle, single PIPELINE region, one
// hardware instance of the MAC -- splitting emission into a separate call site
// per trigger, as the earlier per-frame-batch design did, gave each its own
// resource-bound region in Vitis and multiplied DSP usage for identical math).
// The circular frame buffer therefore needs depth_size+1 (not depth_size) slots:
// depth_size needed for taps + 1 for the frame currently streaming in during the
// lag. Any trailing trigger with no "next frame" to lag into (the tail, possibly
// entirely zero-padded) is covered by extending the read loop with virtual
// frames (r >= n_depth_in: no read, taps referencing them are already 0).
//
// Input  stream: one pixel per token (data_T::size == n_chan), n_depth_in*H*W
//                tokens, raster order (depth, then row, then col).
// Output stream: one pixel per token (res_T::size == n_filt), n_depth_out*H*W
//                tokens, same raster order.
//
// Border / parity / zero-padding identical to the 2D kernel; depth frames
// outside [0, n_depth_in) (padding) contribute zero.
//
// SYNTHESIS SCALING (previously a hard limitation, now addressed): this kernel
// used to unroll its ENTIRE per-pixel MAC -- depth_size*k*n_chan taps against
// n_filt filters -- in one pipelined region, so reuse_factor only set the
// pipeline II and never reduced the logic Vitis had to schedule. Past a few
// thousand unrolled products the scheduler slowed to a crawl (a modest 8x8,
// D=5, Cin=2, Cout=4, depth_size=2 config hung 15+ min at "Applying dataflow").
//
// That was originally written up as a 3D-specific quirk. It was not: the 2D
// sibling hit the identical wall once given a realistically sized layer (24x24,
// k=19, Cin=16, Cout=32 -> 9728 products). Depth only makes 3D reach the
// ceiling sooner. The same treatment is now applied here:
//   * hex_lb_pixel_mac_3d_reuse folds the products over reuse_factor cycles
//     (only ceil(n_in/RF)*n_filt exist in hardware, n_in = DK*rows*n_chan);
//   * the per-pixel loop is pipelined only under strategy='latency', so the
//     reuse loop survives under 'resource';
//   * an ALLOCATION cap stops Vitis re-replicating the folded multipliers.
// On the 2D layer this turned a synthesis that never finished into 3m11s at
// 17% LUT. The 3D path has NOT yet been re-benchmarked at scale -- the
// mechanism is shared and csim-verified, but treat the numbers as unmeasured.
//
// The io_parallel fallback is unaffected either way (it already caps
// multipliers via multiplier_limit).
//
// Corresponds to the HexConvLineBuffer3D Keras layer (hex_gather.py).
//
// KNOWN LIMITATION (RTL correctness, io_stream + Strategy='Latency'+RF=1 +
// a downstream repack_stream layer such as Flatten): see the detailed note
// in nnet_hex_conv_linebuffer_stream.h (the 2D sibling). Workaround:
// Strategy='Resource' with ReuseFactor>=2.

namespace nnet {

// Compute and write one output pixel (t_out, oh, ow). INLINE: must live inside
// the caller's single PIPELINE region (see caller / file header).
template <class res_T, class w_T, class b_T, typename CONFIG_T, class dval>
void hex_lb_emit_pixel_3d(
    // [depth_size+1][N*n_chan], circular over frames. A genuine 2D array (not a
    // flattened 1D index) so ARRAY_PARTITION can split the SMALL depth_size+1
    // axis explicitly (see caller): without this, Vitis's mux/crossbar inference
    // treats fslot*N+pix as one opaque address into one large bank, which is what
    // made synthesis pathologically slow for 3D (the 2D kernel's row_buf is much
    // smaller so it never hit this).
    const dval frame_buf[CONFIG_T::depth_size + 1][
        CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    int t_out, int oh, int ow,
    const w_T weights[CONFIG_T::depth_size * CONFIG_T::num_weight_rows *
                      CONFIG_T::n_chan * CONFIG_T::n_filt],
    const b_T bias[CONFIG_T::n_filt],
    hls::stream<res_T> &output
) {
    #pragma HLS INLINE
    constexpr int      H     = (int)CONFIG_T::in_height;
    constexpr int      W     = (int)CONFIG_T::in_width;
    constexpr unsigned Cin   = CONFIG_T::n_chan;
    constexpr unsigned Cout  = CONFIG_T::n_filt;
    constexpr unsigned DK    = CONFIG_T::depth_size;
    constexpr unsigned DKBUF = DK + 1;  // +1: frame currently streaming in during the lag
    constexpr unsigned N     = CONFIG_T::in_height * CONFIG_T::in_width;
    const bool odd = (ow & 1);

    // NOTE: an attempt to process depth taps sequentially (a real, non-unrolled
    // loop with a persistent accumulator) to shrink Vitis's per-cycle synthesis
    // problem was tried here and abandoned: Vitis auto-unrolls small constant-
    // trip-count loops nested in a PIPELINE'd region regardless of an explicit
    // `#pragma HLS UNROLL factor=1` (confirmed via vitis_hls.log: "Unrolling
    // loop 'DepthTap' ... completely with a factor of 2" even with that pragma
    // present). So this is the same fully-unrolled, single-shot per-pixel
    // gather+MAC as the rest of the linebuffer kernels -- see the file header
    // for the (still open) synthesis-time scaling caveat this implies for 3D.
    dval taps[DK * CONFIG_T::k * Cin];
    #pragma HLS ARRAY_PARTITION variable=taps complete
DepthTapGather:
    for (unsigned dd = 0; dd < DK; dd++) {
        #pragma HLS UNROLL
        const int t_in = t_out + (int)dd - CONFIG_T::pad_top;
        const bool frame_ok = (t_in >= 0) && (t_in < (int)CONFIG_T::n_depth_in);
        const unsigned fslot = ((unsigned)t_in) % DKBUF;
        for (unsigned ki = 0; ki < CONFIG_T::k; ki++) {
            #pragma HLS UNROLL
            const int dr = odd ? CONFIG_T::off_odd[ki][0] : CONFIG_T::off_even[ki][0];
            const int dc = odd ? CONFIG_T::off_odd[ki][1] : CONFIG_T::off_even[ki][1];
            const int ih = oh + dr;
            const int iw = ow + dc;
            const bool ok = frame_ok && (ih >= 0) && (ih < H) && (iw >= 0) && (iw < W);
            const unsigned pix = (unsigned)ih * (unsigned)W + (unsigned)iw;
            for (unsigned c = 0; c < Cin; c++) {
                #pragma HLS UNROLL
                taps[(dd * CONFIG_T::k + ki) * Cin + c] = ok
                    ? frame_buf[fslot][pix * Cin + c]
                    : (dval)0;
            }
        }
    }

    typename res_T::value_type res_out[Cout];
    #pragma HLS ARRAY_PARTITION variable=res_out complete
    hex_lb_pixel_mac_3d_reuse<CONFIG_T, dval, w_T, b_T, typename res_T::value_type>(
        taps, weights, bias, res_out);

    res_T tok;
    PRAGMA_DATA_PACK(tok)
    for (unsigned o = 0; o < Cout; o++) {
        #pragma HLS UNROLL
        tok[o] = res_out[o];
    }
    output.write(tok);
}

template <class data_T, class w_T, class b_T, class res_T, typename CONFIG_T>
void hex_conv_linebuffer_3d_stream(
    hls::stream<data_T> &input,
    w_T weights[CONFIG_T::depth_size * CONFIG_T::num_weight_rows *
                CONFIG_T::n_chan * CONFIG_T::n_filt],
    b_T bias[CONFIG_T::n_filt],
    hls::stream<res_T> &output
) {
    constexpr int      D_in  = (int)CONFIG_T::n_depth_in;
    constexpr int      D_out = (int)CONFIG_T::n_depth_out;
    constexpr int      W     = (int)CONFIG_T::in_width;
    constexpr int      DK1   = (int)CONFIG_T::depth_size - 1 - CONFIG_T::pad_top;  // trigger offset
    constexpr unsigned N     = CONFIG_T::in_height * CONFIG_T::in_width;
    constexpr unsigned Cin   = CONFIG_T::n_chan;
    constexpr unsigned DK    = CONFIG_T::depth_size;
    constexpr unsigned DKBUF = DK + 1;

    typedef typename data_T::value_type dval;

    // Multiplier budget: how many products the reuse loop issues per cycle.
    // Weight storage is left to Vitis (see the 2D kernel for why forcing a
    // reshaped BRAM ROM backfires); only RF==1 is worth full partitioning.
    constexpr unsigned MAC_N_IN   = CONFIG_T::depth_size * CONFIG_T::num_weight_rows * Cin;
    constexpr unsigned MAC_IN_PER = DIV_ROUNDUP(MAC_N_IN, CONFIG_T::reuse_factor);
    constexpr unsigned MAC_BLOCK  = MAC_IN_PER * CONFIG_T::n_filt;
    #pragma HLS ALLOCATION operation instances=mul limit=MAC_BLOCK
    if (CONFIG_T::reuse_factor == 1) {
        #pragma HLS ARRAY_PARTITION variable=weights complete
    }
    #pragma HLS ARRAY_PARTITION variable=bias    complete
    #pragma HLS FUNCTION_INSTANTIATE variable=weights,bias

    // Circular depth-frame buffer: DKBUF spatial planes resident. A genuine 2D
    // array, partitioned completely on the (small, DKBUF-sized) slot axis so
    // Vitis synthesizes DKBUF independent small memories instead of one huge
    // bank needing a wide runtime-indexed crossbar -- see hex_lb_emit_pixel_3d.
    dval frame_buf[DKBUF][N * Cin];
    #pragma HLS ARRAY_PARTITION variable=frame_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=frame_buf cyclic factor=Cin dim=2

    // The output frame currently being emitted (one spatial pixel per input-
    // pixel iteration), lagging one input frame behind its trigger.
    int cur_emit_t = -1;
    int cur_emit_p = 0;  // flattened spatial index into [0, N)

    // See nnet_hex_conv_linebuffer_stream.h for the full derivation (identical,
    // "row" -> "frame", "column" -> "spatial pixel", stride fixed at 1 here):
    // a single pipelined loop over R_TOTAL >= D_in frames (real frames read
    // normally; virtual trailing frames r >= D_in skip the read, their taps are
    // already 0) keeps the whole schedule -- including the zero-padded tail --
    // in ONE call site / ONE resource-binding region.
    constexpr int last_trigger_r = (D_out - 1) + DK1;
    constexpr int min_r_total    = last_trigger_r + 2;
    constexpr int R_TOTAL        = (min_r_total > D_in) ? min_r_total : D_in;

ReadFrames:
    for (int r = 0; r < R_TOTAL; r++) {
    ReadFramePixels:
        for (unsigned p = 0; p < N; p++) {
            // See the 2D kernel: pipelining here would swallow the MAC's
            // reuse loop and re-unroll every product, which is exactly what
            // made this kernel unschedulable at realistic sizes.
            if (CONFIG_T::strategy == nnet::latency) {
                #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
            }
            if (r < D_in) {
                data_T tok = input.read();
                const unsigned fslot = ((unsigned)r) % DKBUF;
                for (unsigned c = 0; c < Cin; c++) {
                    #pragma HLS UNROLL
                    frame_buf[fslot][p * Cin + c] = tok[c];
                }
            }
            if (cur_emit_t >= 0 && (int)cur_emit_p < (int)N) {
                const int oh = (int)cur_emit_p / W;
                const int ow = (int)cur_emit_p % W;
                hex_lb_emit_pixel_3d<res_T, w_T, b_T, CONFIG_T, dval>(
                    frame_buf, cur_emit_t, oh, ow, weights, bias, output);
                cur_emit_p++;
            }
        }
        // Frame r unlocks output frame t_ready(r) = r - DK1 (an `if` on a pure
        // function of the static loop index r -- see the 2D kernel for why this
        // resolves latency/II statically instead of "?"). Since depth_stride is
        // fixed at 1, t_ready advances by exactly 1 per frame once active.
        const int t_cur  = (r     >= DK1) ? r     - DK1 : -1;
        const int t_prev = (r - 1 >= DK1) ? r - 1 - DK1 : -1;
        if (t_cur != t_prev && t_cur >= 0 && t_cur < D_out) {
            cur_emit_t = t_cur;
            cur_emit_p = 0;
        }
    }
}

} // namespace nnet

#endif // NNET_HEX_CONV_LINEBUFFER_3D_STREAM_H_
