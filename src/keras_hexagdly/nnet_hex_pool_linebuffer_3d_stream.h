#ifndef NNET_HEX_POOL_LINEBUFFER_3D_STREAM_H_
#define NNET_HEX_POOL_LINEBUFFER_3D_STREAM_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_hex_pool_linebuffer_3d.h"
#include "nnet_types.h"

// Depth-aware hexagonal max-pool, line-buffer io_stream kernel.
//
// Same fused, single-loop, per-frame-pixel design as
// nnet_hex_conv_linebuffer_3d_stream.h ("row" -> "frame", stride = depth_stride,
// which — unlike Conv3d — may be > 1 for pooling): only depth_size+1 spatial
// planes resident (depth_size needed for the pool window + 1 for the frame
// currently streaming in during the one-frame emission lag). Reduces the max
// over the depth window and the K spatial taps; out-of-bounds spatial taps -> 0
// (hexagdly zero-pad). MaxPool has no depth padding (valid pooling only), so —
// unlike the conv path — no zero-padded tail exists; the virtual-frame loop
// extension still covers the case where the last trigger's lag frame would
// otherwise fall outside the real input range.
//
// Corresponds to the HexPoolLineBuffer3D Keras layer (hex_gather.py).
//
// KNOWN LIMITATION (RTL correctness, io_stream + Strategy='Latency'+RF=1 +
// a downstream repack_stream layer such as Flatten): see the detailed note
// in nnet_hex_conv_linebuffer_stream.h. Workaround: Strategy='Resource' with
// ReuseFactor>=2.

namespace nnet {

// Compute and write one output pixel (t_out, oh, ow). INLINE: must live inside
// the caller's single PIPELINE region (see file header).
//
// KNOWN LIMITATION (synthesis time, not correctness): see the header comment in
// nnet_hex_conv_linebuffer_3d_stream.h. A sequential (non-unrolled) depth-tap
// accumulation was tried here too and abandoned for the same reason (Vitis
// auto-unrolls regardless); this is the plain fully-unrolled, single-shot
// per-pixel gather+max.
template <class res_T, typename CONFIG_T, class dval>
void hex_lb_emit_pool_pixel_3d(
    // [depth_size+1][N*n_chan], circular over frames. A genuine 2D array (not a
    // flattened 1D index) so ARRAY_PARTITION can split the SMALL depth_size+1
    // axis explicitly -- see nnet_hex_conv_linebuffer_3d_stream.h.
    const dval frame_buf[CONFIG_T::depth_size + 1][
        CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    int t_out, int oh, int ow,
    hls::stream<res_T> &output
) {
    #pragma HLS INLINE
    constexpr int      H     = (int)CONFIG_T::in_height;
    constexpr int      W     = (int)CONFIG_T::in_width;
    constexpr unsigned C     = CONFIG_T::n_chan;
    constexpr unsigned DK    = CONFIG_T::depth_size;
    constexpr unsigned DKBUF = DK + 1;
    const bool odd = (ow & 1);

    dval taps[DK * CONFIG_T::k * C];
    #pragma HLS ARRAY_PARTITION variable=taps complete
    for (unsigned dd = 0; dd < DK; dd++) {
        #pragma HLS UNROLL
        const unsigned t_in = (unsigned)t_out * CONFIG_T::depth_stride + dd;
        const unsigned fslot = t_in % DKBUF;
        for (unsigned ki = 0; ki < CONFIG_T::k; ki++) {
            #pragma HLS UNROLL
            const int dr = odd ? CONFIG_T::off_odd[ki][0] : CONFIG_T::off_even[ki][0];
            const int dc = odd ? CONFIG_T::off_odd[ki][1] : CONFIG_T::off_even[ki][1];
            const int ih = oh + dr;
            const int iw = ow + dc;
            const bool ok = (ih >= 0) && (ih < H) && (iw >= 0) && (iw < W);
            const unsigned pix = (unsigned)ih * (unsigned)W + (unsigned)iw;
            for (unsigned c = 0; c < C; c++) {
                #pragma HLS UNROLL
                taps[(dd * CONFIG_T::k + ki) * C + c] =
                    ok ? frame_buf[fslot][pix * C + c] : (dval)0;
            }
        }
    }
    typename res_T::value_type res_out[C];
    #pragma HLS ARRAY_PARTITION variable=res_out complete
    hex_lb_pixel_max_3d<CONFIG_T, dval, typename res_T::value_type>(taps, res_out);

    res_T tok;
    PRAGMA_DATA_PACK(tok)
    for (unsigned c = 0; c < C; c++) {
        #pragma HLS UNROLL
        tok[c] = res_out[c];
    }
    output.write(tok);
}

template <class data_T, class res_T, typename CONFIG_T>
void hex_pool_linebuffer_3d_stream(hls::stream<data_T> &input, hls::stream<res_T> &output) {
    constexpr int      D_in  = (int)CONFIG_T::n_depth_in;
    constexpr int      D_out = (int)CONFIG_T::n_depth_out;
    constexpr int      W     = (int)CONFIG_T::in_width;
    constexpr int      DS    = (int)CONFIG_T::depth_stride;
    constexpr int      DK1   = (int)CONFIG_T::depth_size - 1;  // trigger offset (no depth padding)
    constexpr unsigned N     = CONFIG_T::in_height * CONFIG_T::in_width;
    constexpr unsigned C     = CONFIG_T::n_chan;
    constexpr unsigned DK    = CONFIG_T::depth_size;
    constexpr unsigned DKBUF = DK + 1;

    typedef typename data_T::value_type dval;

    dval frame_buf[DKBUF][N * C];
    #pragma HLS ARRAY_PARTITION variable=frame_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=frame_buf cyclic factor=C dim=2

    int cur_emit_t = -1;
    int cur_emit_p = 0;

    // See nnet_hex_conv_linebuffer_3d_stream.h for the full derivation
    // (identical single-loop, virtual-frame-extended, one-frame-lag design).
    constexpr int last_trigger_r = (D_out - 1) * DS + DK1;
    constexpr int min_r_total    = last_trigger_r + 2;
    constexpr int R_TOTAL        = (min_r_total > D_in) ? min_r_total : D_in;

ReadFrames:
    for (int r = 0; r < R_TOTAL; r++) {
    ReadFramePixels:
        for (unsigned p = 0; p < N; p++) {
            #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
            if (r < D_in) {
                data_T tok = input.read();
                const unsigned fslot = ((unsigned)r) % DKBUF;
                for (unsigned c = 0; c < C; c++) {
                    #pragma HLS UNROLL
                    frame_buf[fslot][p * C + c] = tok[c];
                }
            }
            if (cur_emit_t >= 0 && (int)cur_emit_p < (int)N) {
                const int oh = (int)cur_emit_p / W;
                const int ow = (int)cur_emit_p % W;
                hex_lb_emit_pool_pixel_3d<res_T, CONFIG_T, dval>(
                    frame_buf, cur_emit_t, oh, ow, output);
                cur_emit_p++;
            }
        }
        const int t_cur  = (r     >= DK1) ? (r     - DK1) / DS : -1;
        const int t_prev = (r - 1 >= DK1) ? (r - 1 - DK1) / DS : -1;
        if (t_cur != t_prev && t_cur >= 0 && t_cur < D_out) {
            cur_emit_t = t_cur;
            cur_emit_p = 0;
        }
    }
}

} // namespace nnet

#endif // NNET_HEX_POOL_LINEBUFFER_3D_STREAM_H_
