#ifndef NNET_HEX_POOL_LINEBUFFER_STREAM_H_
#define NNET_HEX_POOL_LINEBUFFER_STREAM_H_

#include "ap_shift_reg.h"
#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_hex_pool_linebuffer.h"
#include "nnet_types.h"

// Hexagonal neighbor max-pool, line-buffer io_stream kernel.
//
// Same circular-row-buffer front-end as the conv line-buffer kernel
// (nnet_hex_conv_linebuffer_stream.h) -- see that file's header for the full
// scheduling proof (this kernel uses the identical single-loop, virtual-row-
// extended, one-row-lag design). Only rows_resident+2 input rows resident,
// parity-dependent compile-time tap offsets, out-of-bounds taps -> 0 (hexagdly
// zero-padded MaxPool border). Reduces the max over the K taps instead of a MAC.
//
// Corresponds to the HexPoolLineBuffer Keras layer (hex_gather.py).
//
// KNOWN LIMITATION (RTL correctness, io_stream + Strategy='Latency'+RF=1 +
// a downstream repack_stream layer such as Flatten): see the detailed note
// in nnet_hex_conv_linebuffer_stream.h. Workaround: Strategy='Resource' with
// ReuseFactor>=2.

namespace nnet {

// Compute and write one output pixel (oh, ow) from the sliding window.
// INLINE: must live inside the caller's single PIPELINE region (see caller).
template <class res_T, typename CONFIG_T, class dval>
void hex_lb_emit_pool_pixel(
    // win[i][j] = pixel at (read_row - i, read_col - (col_span-1 - j)). The
    // caller keeps the read head a fixed (dr_max+1, dc_max) ahead of the pixel
    // being emitted, so every tap resolves to a compile-time (i, j) -- a static
    // register pick with no address arithmetic. See the conv sibling.
    const dval win[CONFIG_T::rows_resident + 2][CONFIG_T::col_span][CONFIG_T::n_chan],
    int oh, int ow,
    hls::stream<res_T> &output
) {
    #pragma HLS INLINE
    constexpr int      H     = (int)CONFIG_T::in_height;
    constexpr int      W     = (int)CONFIG_T::in_width;
    constexpr int      S     = (int)CONFIG_T::stride;
    constexpr int      DRMAX = CONFIG_T::dr_max;
    constexpr int      DCMAX = CONFIG_T::dc_max;
    constexpr int      CSPAN = (int)CONFIG_T::col_span;
    constexpr unsigned C     = CONFIG_T::n_chan;
    const bool odd = (ow & 1);
    const int base_h = S * oh;
    const int base_w = S * ow;

    // Border validity factored out of the tap loop -- see the conv sibling for
    // the rationale (row validity depends only on dr, column validity only on
    // dc, and both map bijectively to the window indices).
    constexpr unsigned NR = CONFIG_T::rows_resident + 2;
    bool row_ok[NR];
    #pragma HLS ARRAY_PARTITION variable=row_ok complete
    for (unsigned wi = 0; wi < NR; wi++) {
        #pragma HLS UNROLL
        const int ih = base_h + DRMAX + 1 - (int)wi;
        row_ok[wi] = (ih >= 0) && (ih < H);
    }
    bool col_ok[CSPAN];
    #pragma HLS ARRAY_PARTITION variable=col_ok complete
    for (int wj = 0; wj < CSPAN; wj++) {
        #pragma HLS UNROLL
        const int iw = base_w + wj - (CSPAN - 1) + DCMAX;
        col_ok[wj] = (iw >= 0) && (iw < W);
    }

    dval taps[CONFIG_T::k * C];
    #pragma HLS ARRAY_PARTITION variable=taps complete
    for (unsigned ki = 0; ki < CONFIG_T::k; ki++) {
        #pragma HLS UNROLL
        const int dr = odd ? CONFIG_T::off_odd[ki][0] : CONFIG_T::off_even[ki][0];
        const int dc = odd ? CONFIG_T::off_odd[ki][1] : CONFIG_T::off_even[ki][1];
        const int wi = DRMAX + 1 - dr;
        const int wj = CSPAN - 1 - DCMAX + dc;
        const bool in_bounds = row_ok[wi] && col_ok[wj];
        for (unsigned c = 0; c < C; c++) {
            #pragma HLS UNROLL
            taps[ki * C + c] = in_bounds ? win[wi][wj][c] : (dval)0;
        }
    }
    typename res_T::value_type res_out[C];
    #pragma HLS ARRAY_PARTITION variable=res_out complete
    hex_lb_pixel_max<CONFIG_T, dval, typename res_T::value_type>(taps, res_out);

    res_T tok;
    PRAGMA_DATA_PACK(tok)
    for (unsigned c = 0; c < C; c++) {
        #pragma HLS UNROLL
        tok[c] = res_out[c];
    }
    output.write(tok);
}

template <class data_T, class res_T, typename CONFIG_T>
void hex_pool_linebuffer_stream(hls::stream<data_T> &input, hls::stream<res_T> &output) {
    constexpr int      H     = (int)CONFIG_T::in_height;
    constexpr int      W     = (int)CONFIG_T::in_width;
    constexpr int      W_OUT = (int)CONFIG_T::out_width;
    constexpr int      S     = (int)CONFIG_T::stride;
    constexpr int      DRMAX = CONFIG_T::dr_max;
    constexpr int      OH    = (int)CONFIG_T::out_height;
    constexpr unsigned C     = CONFIG_T::n_chan;
    constexpr unsigned NR    = CONFIG_T::rows_resident + 2;

    typedef typename data_T::value_type dval;

    // One pixel per token is baked into the read/emit bookkeeping below; make a
    // mismatch a compile error rather than a silent misread (see the conv
    // sibling for the same guard).
    static_assert(data_T::size == CONFIG_T::n_chan,
                  "hex_pool_linebuffer_stream: expects one pixel per input token "
                  "(data_T::size == n_chan)");
    static_assert(res_T::size == CONFIG_T::n_chan,
                  "hex_pool_linebuffer_stream: expects one pixel per output token "
                  "(res_T::size == n_chan)");

    // Shift-register line buffers + register window, exactly as in the conv
    // sibling (see it for the derivation): the incoming pixel is pushed down a
    // chain of one-row delays to yield the vertical strip at the current column,
    // which is shifted into a small window. Replaces re-reading K taps per
    // output pixel. Note hls4ml's own max-pool stream kernel
    // (compute_pool_buffer_2d) is built the same way -- this brings the hex pool
    // back in line with it.
    constexpr int CSPAN   = (int)CONFIG_T::col_span;
    constexpr int DCMAX   = CONFIG_T::dc_max;
    constexpr int C_TOTAL = W + DCMAX;  // iterations per row (virtual columns included)

    // Channel axis only: `complete dim=0` would also split ap_shift_reg's
    // internal storage, which Vitis rejects for shift operations.
    ap_shift_reg<dval, C_TOTAL> line_sr[NR - 1][C];
    #pragma HLS ARRAY_PARTITION variable=line_sr complete dim=2

    dval win[NR][CSPAN][C];
    #pragma HLS ARRAY_PARTITION variable=win complete dim=0

    int cur_emit_oh  = -1;
    int cur_emit_col = 0;

    // See nnet_hex_conv_linebuffer_stream.h for the full derivation: a single
    // pipelined loop over R_TOTAL >= H rows (real rows read normally; any
    // virtual trailing rows r >= H skip the read -- their taps are already 0 via
    // in_bounds) keeps everything (including the tail) in ONE call site / ONE
    // resource-binding region, avoiding separate flush loops that each got their
    // own hardware instance in Vitis.
    constexpr int last_trigger_r = S * (OH - 1) + DRMAX;
    constexpr int min_r_total    = last_trigger_r + 2;
    constexpr int R_TOTAL        = (min_r_total > H) ? min_r_total : H;

ReadRows:
    for (int r = 0; r < R_TOTAL; r++) {
    ReadCols:
        for (int c = 0; c < C_TOTAL; c++) {
            // Unconditionally pipelined, unlike the conv sibling, which gates
            // this on strategy==latency. That gate exists there to protect the
            // MAC's reuse loop from being unrolled; max-pool has no multiplier
            // bank to time-share, so an unpipelined resource variant would only
            // cost throughput and save nothing.
            #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

            const bool real_px = (r < H) && (c < W);
            data_T tok;
            if (real_px) {
                tok = input.read();
            }
            dval strip[NR][C];
            #pragma HLS ARRAY_PARTITION variable=strip complete dim=0
            for (unsigned ic = 0; ic < C; ic++) {
                #pragma HLS UNROLL
                strip[0][ic] = real_px ? (dval)tok[ic] : (dval)0;
            }
            for (unsigned i = 1; i < NR; i++) {
                #pragma HLS UNROLL
                for (unsigned ic = 0; ic < C; ic++) {
                    #pragma HLS UNROLL
                    strip[i][ic] = line_sr[i - 1][ic].shift(strip[i - 1][ic]);
                }
            }
            for (unsigned i = 0; i < NR; i++) {
                #pragma HLS UNROLL
                for (int j = 0; j < CSPAN - 1; j++) {
                    #pragma HLS UNROLL
                    for (unsigned ic = 0; ic < C; ic++) {
                        #pragma HLS UNROLL
                        win[i][j][ic] = win[i][j + 1][ic];
                    }
                }
                for (unsigned ic = 0; ic < C; ic++) {
                    #pragma HLS UNROLL
                    win[i][CSPAN - 1][ic] = strip[i][ic];
                }
            }

            // cur_emit_col saturating at out_width is also the "already emitted"
            // guard: with stride > 1 the trigger below leaves cur_emit_oh
            // unchanged for S input rows.
            const int ce = c - DCMAX;
            if (cur_emit_oh >= 0 && ce >= 0 && (ce % S) == 0 && cur_emit_col < W_OUT) {
                hex_lb_emit_pool_pixel<res_T, CONFIG_T, dval>(
                    win, cur_emit_oh, cur_emit_col, output);
                cur_emit_col++;
            }
        }
        const int oh_cur  = (r     >= DRMAX) ? (r     - DRMAX) / S : -1;
        const int oh_prev = (r - 1 >= DRMAX) ? (r - 1 - DRMAX) / S : -1;
        if (oh_cur != oh_prev && oh_cur >= 0 && oh_cur < OH) {
            cur_emit_oh  = oh_cur;
            cur_emit_col = 0;
        }
    }
}

} // namespace nnet

#endif // NNET_HEX_POOL_LINEBUFFER_STREAM_H_
