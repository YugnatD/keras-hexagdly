#ifndef NNET_HEX_CONV_LINEBUFFER_STREAM_H_
#define NNET_HEX_CONV_LINEBUFFER_STREAM_H_

#include "ap_shift_reg.h"
#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_hex_conv_linebuffer.h"
#include "nnet_types.h"

// Hexagonal conv, line-buffer io_stream kernel (keras-hexagdly "linebuffer").
//
// The memory win over the gather path: instead of materialising the whole frame
// AND the N_out*K*Cin gathered tensor in registers, we keep only rows_resident+1
// (= 2*kernel_size + 1) image rows resident in shift-register line buffers and
// slide a register window over the streamed image — the hex analogue of
// hls4ml's conv line buffer (ap_shift_reg + kernel window).
//
// Fused, per-pixel loop (mirrors hls4ml's own conv_2d_buffer_cl /
// compute_output_buffer_2d): a SINGLE loop over the H*W input tokens reads the
// incoming pixel AND emits one output pixel of whichever output row is
// currently "lagging" one input row behind. This differs from a naive
// per-row-batch design (read a whole row, THEN emit a whole row as a separate
// call): that design leaves the row-read loop unpipelined, so Vitis cannot
// resolve a fixed latency/II, and each emit call site gets its own multiplier
// bank. One loop, one call site, one hardware instance.
//
// How that loop is scheduled depends on CONFIG_T::strategy, exactly as in
// hls4ml's native kernel:
//   latency  -- the loop is pipelined at II=reuse_factor and the whole MAC is
//               unrolled inside it (max throughput, max area).
//   resource -- the loop is NOT pipelined, so the inlined emit body keeps its
//               MAC's own reuse loop and only n_in*n_out/reuse_factor
//               products exist in hardware. Pipelining here would swallow that
//               reuse loop and re-unroll everything, which is exactly how an
//               earlier revision of this kernel ended up asking Vitis to place
//               ~10k multipliers in one region and stalling its scheduler.
//
// Scheduling invariant (why one row of lag is always enough): row r "unlocks"
// output row oh_ready(r) = floor((r - dr_max) / stride) — since stride >= 1,
// oh_ready(r) - oh_ready(r-1) <= 1, so at most one new output row unlocks per
// input row. Because out_width <= in_width always, that newly-unlocked row's
// out_width output pixels always fit within the in_width column-iterations of
// the VERY NEXT input row — so "emit the lagging row's next column" interleaved
// with "read this row's next column" never falls behind, except for the row
// unlocked by the very last input row (H-1), which has no "next row" to lag
// into and is finished in a small, statically-bounded flush after the read loop.
//
// Input  stream: one pixel per token (data_T::size == n_chan), H*W tokens, raster
//                order (row-major over the (H, W) grid).
// Output stream: one pixel per token (res_T::size == n_filt), H*W tokens, raster.
//
// Border / zero-padding: each output pixel (oh, ow) reads its K neighbor taps at
// the parity-dependent compile-time offsets (dr, dc); a tap whose (oh+dr, ow+dc)
// falls outside [0,H)x[0,W) contributes 0 — exactly matching hexagdly's
// zero-padded border (the -1 sentinel in the flat neighbor table).
//
// Parity: the tap offsets differ between even and odd output columns (zig-zag
// half-pixel shift). CONFIG_T::off_even / off_odd hold the (dr, dc) pairs as
// compile-time constants; the parity is (ow & 1), so all indices constant-fold.
//
// Corresponds to the HexConvLineBuffer Keras layer (hex_gather.py).
//
// KNOWN LIMITATION (RTL correctness, io_stream only): when this kernel is
// followed by a shape-changing layer that hls4ml implements via
// nnet::repack_stream (e.g. Flatten), AND the model uses Strategy='Latency'
// with ReuseFactor=1 (fully unrolled, near-combinational per-pixel body),
// real RTL cosimulation (cosim_design, not just csim) can produce a WRONG,
// CONSTANT (bias-only) output for every pixel/image, even though csim is
// correct. Root cause: Vitis infers an "auto-rewind" pipeline for this
// kernel inside the DATAFLOW region (see warning HLS 200-656, "Deadlocks can
// occur ... ap_ctrl_none ... auto-rewind pipeline"), and at RF=1 the kernel's
// per-pixel latency is too short for Vitis's auto-generated dataflow control
// logic to correctly propagate "start" between this process and the
// downstream repack_stream process -- a real, longstanding Vitis/hls4ml
// dataflow-synchronization edge case (the same class of bug as
// fastmachinelearning/hls4ml#351, where the fix was to remove the offending
// pragma; here the auto-rewind inference is internal to Vitis and has no
// user-facing pragma or config_dataflow option to disable it -- confirmed by
// exhausting ap_ctrl_chain on the top function, config_dataflow
// -start_fifo_depth / -strict_stable_sync=true, forcing ap_ctrl_hs on this
// sub-process (rejected: Vitis disallows custom INTERFACE pragmas on
// DATAFLOW-region sub-processes), and forcing a longer II on this kernel
// alone -- none change the outcome).
//
// WORKAROUND (verified via real cosim_design, twice, on independent models):
// use Strategy='Resource' with ReuseFactor>=2 for any model that combines
// io_stream, this kernel (or the gather-path stream kernels -- the same
// symptom reproduces there too, so this is not specific to keras-hexagdly),
// and a repack_stream-triggering layer downstream. This is not a
// resource/latency-neutral change: Strategy='Latency'+RF=1 is a legitimate
// and common hls4ml configuration in its own right (not a "toy" mode), so
// this workaround is a real tradeoff (extra latency), not a free lunch, if
// RF=1 latency was actually required. Without Flatten/repack_stream
// downstream, or in io_parallel, this kernel is unaffected at any RF.

namespace nnet {

// Compute and write one output pixel (oh, ow) from the sliding window.
//
// INLINE (not INLINE OFF), and it matters in both strategies:
//   latency  -- the body must sit inside the caller's PIPELINE for the whole
//               per-pixel path to schedule at II=reuse_factor. Making it a
//               separate instance instead costs a measured 38x in latency and
//               45x in II on an 8x8/k=1 layer (94 -> 3608 cycles, II 80 ->
//               3600): each pixel then pays the emit function's full latency
//               serially instead of being absorbed by the pipeline.
//   resource -- the caller is NOT pipelined there, so inlining does NOT swallow
//               the MAC's own ReuseLoop; its `PIPELINE II=1` survives and the
//               products stay folded over reuse_factor cycles.
// Since there is exactly one call site (the virtual-row design below), inlining
// also cannot duplicate the multiplier bank, which is what an
// `allocation ... limit=1 function` cap would otherwise have been needed for.
template <class res_T, class w_T, class b_T, typename CONFIG_T, class dval>
void hex_lb_emit_pixel(
    // Sliding window in registers: win[i][j] holds the pixel at
    // (read_row - i, read_col - (col_span-1 - j)). Because the caller emits an
    // output pixel at a FIXED offset behind the read head (dr_max+1 rows,
    // dc_max columns), every tap lands at a compile-time (i, j) -- so the whole
    // gather is a static register pick, with no address arithmetic at all.
    const dval win[CONFIG_T::rows_resident + 2][CONFIG_T::col_span][CONFIG_T::n_chan],
    int oh, int ow,
    w_T weights[CONFIG_T::num_weight_rows * CONFIG_T::n_chan * CONFIG_T::n_filt],
    b_T bias[CONFIG_T::n_filt],
    hls::stream<res_T> &output
) {
    #pragma HLS INLINE
    constexpr int      H     = (int)CONFIG_T::in_height;
    constexpr int      W     = (int)CONFIG_T::in_width;
    constexpr int      S     = (int)CONFIG_T::stride;
    constexpr int      DRMAX = CONFIG_T::dr_max;
    constexpr int      DCMAX = CONFIG_T::dc_max;
    constexpr int      CSPAN = (int)CONFIG_T::col_span;
    constexpr unsigned Cin   = CONFIG_T::n_chan;
    constexpr unsigned Cout  = CONFIG_T::n_filt;
    const bool odd = (ow & 1);
    const int base_h = S * oh;
    const int base_w = S * ow;

    // Border validity, factored out of the tap loop. A tap's row validity
    // depends only on its dr and its column validity only on its dc -- and in
    // window coordinates dr <-> wi and dc <-> wj are bijections, so there are
    // only NR row flags and col_span column flags to compute, versus 4
    // comparisons per tap per pixel before (76 of them on a K=19 layer, all in
    // the critical path). hls4ml's native conv sidesteps this entirely by
    // asserting zero padding and letting an upstream ZeroPadding2D inject the
    // border; the hex border is parity-shaped so we cannot do that, but we can
    // at least stop recomputing it K times.
    constexpr unsigned NR = CONFIG_T::rows_resident + 2;
    bool row_ok[NR];
    #pragma HLS ARRAY_PARTITION variable=row_ok complete
    for (unsigned wi = 0; wi < NR; wi++) {
        #pragma HLS UNROLL
        const int ih = base_h + DRMAX + 1 - (int)wi;   // wi = DRMAX + 1 - dr
        row_ok[wi] = (ih >= 0) && (ih < H);
    }
    bool col_ok[CSPAN];
    #pragma HLS ARRAY_PARTITION variable=col_ok complete
    for (int wj = 0; wj < CSPAN; wj++) {
        #pragma HLS UNROLL
        const int iw = base_w + wj - (CSPAN - 1) + DCMAX;  // wj = CSPAN-1-DCMAX+dc
        col_ok[wj] = (iw >= 0) && (iw < W);
    }

    dval taps[CONFIG_T::k * Cin];
    #pragma HLS ARRAY_PARTITION variable=taps complete
TapGather:
    for (unsigned ki = 0; ki < CONFIG_T::k; ki++) {
        #pragma HLS UNROLL
        const int dr = odd ? CONFIG_T::off_odd[ki][0] : CONFIG_T::off_even[ki][0];
        const int dc = odd ? CONFIG_T::off_odd[ki][1] : CONFIG_T::off_even[ki][1];
        const int wi = DRMAX + 1 - dr;
        const int wj = CSPAN - 1 - DCMAX + dc;
        const bool in_bounds = row_ok[wi] && col_ok[wj];
        // Window coordinates. The caller guarantees read_row = base_h+DRMAX+1
        // and read_col = base_w+DCMAX at the moment of this call, so:
        //   i = read_row - ih = DRMAX + 1 - dr        (in [1, rows_resident+1])
        //   j = CSPAN-1 - (read_col - iw) = CSPAN-1 - DCMAX + dc   (in [0, CSPAN-1])
        // Both are constants of (parity, ki), so this is a 2-way select between
        // two fixed registers -- not a memory access.
        for (unsigned c = 0; c < Cin; c++) {
            #pragma HLS UNROLL
            taps[ki * Cin + c] = in_bounds ? win[wi][wj][c] : (dval)0;
        }
    }

    typename res_T::value_type res_out[Cout];
    #pragma HLS ARRAY_PARTITION variable=res_out complete
    hex_lb_pixel_mac_reuse<CONFIG_T, dval, w_T, b_T, typename res_T::value_type>(
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
void hex_conv_linebuffer_stream(
    hls::stream<data_T> &input,
    w_T weights[CONFIG_T::num_weight_rows * CONFIG_T::n_chan * CONFIG_T::n_filt],
    b_T bias[CONFIG_T::n_filt],
    hls::stream<res_T> &output
) {
    constexpr int      H     = (int)CONFIG_T::in_height;
    constexpr int      W     = (int)CONFIG_T::in_width;
    constexpr int      W_OUT = (int)CONFIG_T::out_width;
    constexpr int      S     = (int)CONFIG_T::stride;
    constexpr int      DRMAX = CONFIG_T::dr_max;
    constexpr int      OH    = (int)CONFIG_T::out_height;
    constexpr unsigned Cin   = CONFIG_T::n_chan;
    // +2 (not +1): a triggered output row's emission is lagged into the NEXT
    // input row's read iterations (see file header). While that row streams in,
    // the tap window still needs rows_resident+1 older rows resident, so
    // rows_resident+1(needed) + 1(currently incoming) = rows_resident+2 distinct
    // row indices must be simultaneously live without a circular-slot collision.
    constexpr unsigned NR    = CONFIG_T::rows_resident + 2;  // resident input rows

    typedef typename data_T::value_type dval;

    // One pixel per token is baked into the read/emit bookkeeping below (one
    // input.read() per (row, column) iteration, one output token per output
    // pixel). hls4ml's native conv has an "encoded" variant for multi-pixel
    // tokens; we have no such path, so make a mismatch a compile error instead
    // of a silent misread.
    static_assert(data_T::size == CONFIG_T::n_chan,
                  "hex_conv_linebuffer_stream: expects one pixel per input token "
                  "(data_T::size == n_chan)");
    static_assert(res_T::size == CONFIG_T::n_filt,
                  "hex_conv_linebuffer_stream: expects one pixel per output token "
                  "(res_T::size == n_filt)");

    #pragma HLS ARRAY_PARTITION variable=bias    complete
    #pragma HLS FUNCTION_INSTANTIATE variable=weights,bias

    // MAC_BLOCK = how many products the reuse loop issues per cycle. Capping
    // the multiplier count here is what stops Vitis re-replicating the products
    // that hex_lb_pixel_mac_reuse just folded over reuse_factor cycles.
    constexpr unsigned MAC_N_IN   = CONFIG_T::num_weight_rows * Cin;
    constexpr unsigned MAC_IN_PER = DIV_ROUNDUP(MAC_N_IN, CONFIG_T::reuse_factor);
    constexpr unsigned MAC_BLOCK  = MAC_IN_PER * CONFIG_T::n_filt;
    #pragma HLS ALLOCATION operation instances=mul limit=MAC_BLOCK

    // Weight storage is deliberately left to Vitis rather than forced into a
    // reshaped BRAM ROM the way nnet_dense_resource does. That rule
    // (ARRAY_RESHAPE block factor=<products per cycle> + ROM_nP_BRAM) only
    // behaves when the per-cycle slice is small; here it is MAC_BLOCK wide,
    // and forcing e.g. a 160-element slice produced 3840-bit memory words that
    // Vitis serviced with 17040 BRAMs (788% of the device) for an array whose
    // total content is ~233 Kb -- and at low reuse it degenerated further into
    // a 29184-bit part-select mux that collapsed Fmax to 24 MHz. The weights
    // are compile-time constants, so leaving them unconstrained lets Vitis
    // const-fold them into the (already capped) multipliers instead.
    if (CONFIG_T::reuse_factor == 1) {
        #pragma HLS ARRAY_PARTITION variable=weights complete
    }

    // Line buffer + sliding window, mirroring hls4ml's own conv line buffer.
    //
    // line_sr[i] delays the stream by exactly one iteration-row, so pushing the
    // incoming pixel down the chain yields, in one step, the whole VERTICAL
    // STRIP of pixels at the current column across the last NR rows. That strip
    // is shifted into `win`, a small register window holding the last col_span
    // strips. Every tap is then a compile-time index into `win`.
    //
    // This replaces an earlier circular row buffer that was re-read K times per
    // output pixel (K*n_chan dynamic addresses per cycle -- 304 of them on a
    // k=19, Cin=16 layer). No amount of banking makes that cheap; not doing it
    // at all is the actual fix, and it is why the native kernel maintains its
    // window incrementally rather than re-gathering. Storage is unchanged in
    // order: (NR-1)*n_chan shift registers of C_TOTAL entries is the same
    // O(rows) footprint the row buffer had.
    constexpr int      CSPAN   = (int)CONFIG_T::col_span;
    constexpr int      DCMAX   = CONFIG_T::dc_max;
    constexpr int      C_TOTAL = W + DCMAX;  // iterations per row (see below)

    static_assert(DRMAX + 1 - (DRMAX - (int)CONFIG_T::rows_resident) <= (int)NR - 1,
                  "hex_conv_linebuffer_stream: row window too small for the tap span");

    // Partition the CHANNEL axis only, exactly as hls4ml's own line buffer does.
    // `complete dim=0` also splits ap_shift_reg's internal storage, which Vitis
    // rejects outright ("must be a one dimensional array for valid memory shift
    // operation(s)"); the row axis stays whole and is indexed by an unrolled
    // loop, so its indices are compile-time constants anyway.
    ap_shift_reg<dval, C_TOTAL> line_sr[NR - 1][Cin];
    #pragma HLS ARRAY_PARTITION variable=line_sr complete dim=2

    dval win[NR][CSPAN][Cin];
    #pragma HLS ARRAY_PARTITION variable=win complete dim=0

    // The output row currently being emitted, lagging one input row behind its
    // trigger -- see file header. The output COLUMN is derived from the read
    // column instead of being counted, because the window fixes the lag at
    // exactly DCMAX columns.
    int cur_emit_oh  = -1;
    int cur_emit_col = 0;

    // The row loop runs R_TOTAL >= H iterations: real input rows (r < H, one
    // input.read() per column) plus, if needed, a handful of *virtual* trailing
    // rows (r >= H, no read -- taps referencing them are already 0 via the
    // in_bounds check in hex_lb_emit_pixel) so the LAST output row's trigger
    // (which needs one full row of lag to be consumed, like every other row --
    // see file header) always has somewhere to be emitted. This keeps emission
    // inside the SAME single pipelined loop / single call site as the reads:
    // splitting the tail into separate flush loops each got its own resource
    // binding in Vitis (one multiplier bank per #pragma HLS PIPELINE region,
    // even for tiny loops auto-pipelined by the tool), tripling DSP usage for
    // identical math. One loop, one call site, one hardware instance.
    constexpr int last_trigger_r = S * (OH - 1) + DRMAX;         // r that unlocks oh = OH-1
    constexpr int min_r_total    = last_trigger_r + 2;           // + 1 lag row to consume it
    constexpr int R_TOTAL        = (min_r_total > H) ? min_r_total : H;

    // Strategy dispatch, mirroring hls4ml's conv_2d_buffer_cl:
    //   latency  -> pipeline the per-pixel loop at II=reuse_factor; the inlined
    //               emit body (and its MAC) is unrolled into that pipeline.
    //   resource -> do NOT pipeline here, so the inlined MAC keeps its own
    //               ReuseLoop and only n_in*n_out/reuse_factor products exist.
    //               Pipelining here would swallow that loop and re-unroll it.

ReadRows:
    for (int r = 0; r < R_TOTAL; r++) {
        // C_TOTAL = W + DCMAX, not W: the last DCMAX iterations of every row are
        // *virtual columns* that read nothing. They exist because an output
        // pixel is emitted DCMAX columns after the read head passes it, so the
        // rightmost output columns of a row would otherwise have no iteration
        // left to be emitted in -- the column analogue of the virtual rows
        // above. Keeping the per-row iteration count fixed at C_TOTAL is also
        // what lets the shift registers be exactly C_TOTAL long: pushing a zero
        // during a virtual column keeps every row aligned to the same phase.
    ReadCols:
        for (int c = 0; c < C_TOTAL; c++) {
            if (CONFIG_T::strategy == nnet::latency) {
                #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
            }

            // 1. Pull the next pixel (zero on virtual rows/columns) and push it
            //    down the delay chain to obtain the vertical strip at this
            //    column: strip[i] is the pixel at (r - i, c).
            const bool real_px = (r < H) && (c < W);
            data_T tok;
            if (real_px) {
                tok = input.read();
            }
            dval strip[NR][Cin];
            #pragma HLS ARRAY_PARTITION variable=strip complete dim=0
            for (unsigned ic = 0; ic < Cin; ic++) {
                #pragma HLS UNROLL
                strip[0][ic] = real_px ? (dval)tok[ic] : (dval)0;
            }
            for (unsigned i = 1; i < NR; i++) {
                #pragma HLS UNROLL
                for (unsigned ic = 0; ic < Cin; ic++) {
                    #pragma HLS UNROLL
                    strip[i][ic] = line_sr[i - 1][ic].shift(strip[i - 1][ic]);
                }
            }

            // 2. Slide the window one column left and drop the new strip in at
            //    the right edge.
            for (unsigned i = 0; i < NR; i++) {
                #pragma HLS UNROLL
                for (int j = 0; j < CSPAN - 1; j++) {
                    #pragma HLS UNROLL
                    for (unsigned ic = 0; ic < Cin; ic++) {
                        #pragma HLS UNROLL
                        win[i][j][ic] = win[i][j + 1][ic];
                    }
                }
                for (unsigned ic = 0; ic < Cin; ic++) {
                    #pragma HLS UNROLL
                    win[i][CSPAN - 1][ic] = strip[i][ic];
                }
            }

            // 3. Emit the output pixel whose taps the window now holds. The
            //    read head is DCMAX columns past it, so its output column is
            //    (c - DCMAX)/stride -- only every stride-th iteration emits.
            //    cur_emit_col also acts as the "already emitted" guard: with
            //    stride > 1 the trigger below leaves cur_emit_oh unchanged for
            //    S consecutive input rows, so without a counter that saturates
            //    at out_width the same output row would be emitted S times.
            const int ce = c - DCMAX;
            if (cur_emit_oh >= 0 && ce >= 0 && (ce % S) == 0 && cur_emit_col < W_OUT) {
                hex_lb_emit_pixel<res_T, w_T, b_T, CONFIG_T, dval>(
                    win, cur_emit_oh, cur_emit_col, weights, bias, output);
                cur_emit_col++;
            }
        }
        // Row r unlocks output row oh_ready(r) = floor((r - DRMAX) / S) (an `if`
        // on a pure function of the static loop index r -- see file header for
        // why this, not a `while`, is what lets Vitis resolve a fixed latency/II).
        // This formula is agnostic to whether row r was real or virtual.
        const int oh_cur  = (r     >= DRMAX) ? (r     - DRMAX) / S : -1;
        const int oh_prev = (r - 1 >= DRMAX) ? (r - 1 - DRMAX) / S : -1;
        if (oh_cur != oh_prev && oh_cur >= 0 && oh_cur < OH) {
            cur_emit_oh  = oh_cur;
            cur_emit_col = 0;
        }
    }
}

} // namespace nnet

#endif // NNET_HEX_CONV_LINEBUFFER_STREAM_H_
