#ifndef NNET_HEX_POOL_LINEBUFFER_STREAM_H_
#define NNET_HEX_POOL_LINEBUFFER_STREAM_H_

#include "ap_int.h"
#include "ap_shift_reg.h"
#include "hls_stream.h"
#include <type_traits>
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
// RTL correctness with Strategy='Latency' + ReuseFactor=1 on the hex convolutions
// and a downstream repack_stream layer such as Flatten: see the historical note in
// nnet_hex_conv_linebuffer_stream.h (not reproduced with the current kernels).
// This kernel itself has no strategy and ignores the reuse factor (always one
// pixel per cycle).

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

// One iteration (r, c) of the traversal: read a pixel, advance the line buffers and the window,
// emit the output pixel the window now holds (if any). INLINE: it is the body of the two loop
// forms below.
template <class data_T, class res_T, typename CONFIG_T>
void hex_pool_lb_step(
    int r, int c,
    hls::stream<data_T> &input,
    ap_shift_reg<typename data_T::value_type, (int)CONFIG_T::in_width + CONFIG_T::dc_max>
        line_sr[CONFIG_T::rows_resident + 1][CONFIG_T::n_chan],
    typename data_T::value_type win[CONFIG_T::rows_resident + 2][CONFIG_T::col_span][CONFIG_T::n_chan],
    hls::stream<res_T> &output
) {
    #pragma HLS INLINE
    constexpr int      H     = (int)CONFIG_T::in_height;
    constexpr int      W     = (int)CONFIG_T::in_width;
    constexpr int      DRMAX = CONFIG_T::dr_max;
    constexpr int      DCMAX = CONFIG_T::dc_max;
    constexpr int      CSPAN = (int)CONFIG_T::col_span;
    constexpr unsigned C     = CONFIG_T::n_chan;
    constexpr unsigned NR    = CONFIG_T::rows_resident + 2;
    typedef typename data_T::value_type dval;

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

    // The pixel emitted at (r, c) is a pure function of the loop indices, as in
    // the conv sibling: output row (r - 1 - DRMAX) / S, unlocked by the previous
    // input row, output column (c - DCMAX) / S. Nothing is carried from one
    // iteration to the next: the row/column counters this replaced were a
    // loop-carried 32-bit recurrence, and at II=1 they were the critical path
    // (estimated clock 4.57 ns with them, 3.57 ns without, on an 8x8x2 pool).
    const int ce = c - DCMAX;
    const int rr = r - 1 - DRMAX;
    int oh_emit, ow_emit;
    if (hex_lb_emit_index<CONFIG_T>(rr, ce, oh_emit, ow_emit)) {
        hex_lb_emit_pool_pixel<res_T, CONFIG_T, dval>(win, oh_emit, ow_emit, output);
    }
}

// Shift-register line buffers + register window, exactly as in the conv sibling (see it for the
// derivation): the incoming pixel is pushed down a chain of one-row delays to yield the vertical
// strip at the current column, which is shifted into a small window. hls4ml's own max-pool stream
// kernel (compute_pool_buffer_2d) is built the same way.
//
// The loop runs over R_TOTAL >= H rows (real rows read normally; any virtual trailing rows r >= H
// skip the read -- their taps are already 0 via in_bounds) and C_TOTAL = W + dc_max columns, so
// that everything, the tail included, stays in ONE loop: separate flush loops each got their own
// hardware instance in Vitis.
// (The channel axis only is partitioned: `complete dim=0` would also split ap_shift_reg's internal
// storage, which Vitis rejects for shift operations.)
#define HEX_POOL_LB_STATE                                                                 \
    constexpr int      C_TOTAL = (int)CONFIG_T::in_width + CONFIG_T::dc_max;              \
    constexpr unsigned NR      = CONFIG_T::rows_resident + 2;                             \
    constexpr int      R_NEED  = (int)CONFIG_T::stride * ((int)CONFIG_T::out_height - 1) + CONFIG_T::dr_max + 2; \
    constexpr int      R_TOTAL = (R_NEED > (int)CONFIG_T::in_height) ? R_NEED : (int)CONFIG_T::in_height; \
    typedef typename data_T::value_type dval;                                             \
    static_assert(data_T::size == CONFIG_T::n_chan,                                       \
                  "hex_pool_linebuffer_stream: expects one pixel per input token");       \
    static_assert(res_T::size == CONFIG_T::n_chan,                                        \
                  "hex_pool_linebuffer_stream: expects one pixel per output token");      \
    ap_shift_reg<dval, C_TOTAL> line_sr[NR - 1][CONFIG_T::n_chan];                        \
    dval win[NR][CONFIG_T::col_span][CONFIG_T::n_chan];

// 2D (one frame): the nested row / column loops.
//
// Unconditionally pipelined, unlike the conv sibling, which gates this on strategy==latency. That
// gate exists there to protect the MAC's reuse loop from being unrolled; max-pool has no
// multiplier bank to time-share, so an unpipelined resource variant would only cost throughput and
// save nothing. For the same reason the II is 1 and not reuse_factor: the pool inherits the
// model-level ReuseFactor, which is about multipliers, and it only slowed the pool down (228
// cycles instead of 62 on a 6x6x4 pool at ReuseFactor=4).
template <class data_T, class res_T, typename CONFIG_T>
void hex_pool_linebuffer_stream_frames(
    hls::stream<data_T> &input, hls::stream<res_T> &output, std::false_type
) {
    HEX_POOL_LB_STATE
    #pragma HLS ARRAY_PARTITION variable=line_sr complete dim=2
    #pragma HLS ARRAY_PARTITION variable=win complete dim=0
ReadRows:
    for (int r = 0; r < R_TOTAL; r++) {
    ReadCols:
        for (int c = 0; c < C_TOTAL; c++) {
            #pragma HLS PIPELINE II=1
            hex_pool_lb_step<data_T, res_T, CONFIG_T>(r, c, input, line_sr, win, output);
        }
    }
}

// 3D (n_frames > 1, behind a hex_depth_pool_stream): the same traversal repeated for every frame,
// as ONE flat loop with the row and column in counters just wide enough for them. A third nested
// loop around the two above was flattened by Vitis into a control recurrence that set the clock
// (4.78 ns estimated at a 4 ns target on 4 frames of 12x12); see hex_lb_window_stream in the conv
// sibling for the same measurement.
template <class data_T, class res_T, typename CONFIG_T>
void hex_pool_linebuffer_stream_frames(
    hls::stream<data_T> &input, hls::stream<res_T> &output, std::true_type
) {
    HEX_POOL_LB_STATE
    #pragma HLS ARRAY_PARTITION variable=line_sr complete dim=2
    #pragma HLS ARRAY_PARTITION variable=win complete dim=0
    constexpr int R_BITS = hex_lb_bits((unsigned)R_TOTAL);
    constexpr int C_BITS = hex_lb_bits((unsigned)C_TOTAL);
    ap_uint<R_BITS> r_count = 0;
    ap_uint<C_BITS> c_count = 0;
Traverse:
    for (int it = 0; it < (int)CONFIG_T::n_frames * R_TOTAL * C_TOTAL; it++) {
        #pragma HLS PIPELINE II=1
        hex_pool_lb_step<data_T, res_T, CONFIG_T>(
            (int)r_count, (int)c_count, input, line_sr, win, output);
        if (c_count == C_TOTAL - 1) {
            c_count = 0;
            if (r_count == R_TOTAL - 1) {
                r_count = 0;
            } else {
                r_count = r_count + 1;
            }
        } else {
            c_count = c_count + 1;
        }
    }
}
#undef HEX_POOL_LB_STATE

template <class data_T, class res_T, typename CONFIG_T>
void hex_pool_linebuffer_stream(hls::stream<data_T> &input, hls::stream<res_T> &output) {
    #pragma HLS INLINE
    hex_pool_linebuffer_stream_frames<data_T, res_T, CONFIG_T>(
        input, output, std::integral_constant<bool, (CONFIG_T::n_frames > 1)>());
}

} // namespace nnet

#endif // NNET_HEX_POOL_LINEBUFFER_STREAM_H_
