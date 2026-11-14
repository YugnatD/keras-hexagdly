#ifndef NNET_HEX_POOL_LINEBUFFER_H_
#define NNET_HEX_POOL_LINEBUFFER_H_

#include "nnet_common.h"
#include "nnet_helpers.h"

// Which output pixel, if any, the traversal emits at the iteration whose read head is
// rr = r - 1 - dr_max rows and ce = c - dc_max columns past it: (rr / S, ce / S) when both
// are multiples of the stride and inside the output grid. Shared by the conv and pool
// stream kernels.
//
// With a stride that is a power of two the division and the modulo are a shift and a mask.
// With any other stride (3...) Vitis built real dividers in the traversal loop: measured at
// stride 3 on a 9x9x2 window, a pipeline 39 stages deep with 4 DSP and 4972 FF; decoding
// rr == o*S against every output row o (compile-time constants) instead gives 5 stages,
// 0 DSP and 1020 FF, for the same values.
#ifndef NNET_HEX_LB_EMIT_INDEX_
#define NNET_HEX_LB_EMIT_INDEX_
namespace nnet {
template <typename CONFIG_T> bool hex_lb_emit_index(int rr, int ce, int &oh, int &ow) {
    #pragma HLS INLINE
    constexpr int S  = (int)CONFIG_T::stride;
    constexpr int OH = (int)CONFIG_T::out_height;
    constexpr int OW = (int)CONFIG_T::out_width;
    if ((S & (S - 1)) == 0) {
        oh = rr / S;
        ow = ce / S;
        return rr >= 0 && (rr % S) == 0 && oh < OH && ce >= 0 && (ce % S) == 0 && ow < OW;
    }
    bool row_hit = false, col_hit = false;
    oh = 0;
    ow = 0;
    for (int o = 0; o < OH; o++) {
        #pragma HLS UNROLL
        if (rr == o * S) {
            row_hit = true;
            oh = o;
        }
    }
    for (int o = 0; o < OW; o++) {
        #pragma HLS UNROLL
        if (ce == o * S) {
            col_hit = true;
            ow = o;
        }
    }
    return row_hit && col_hit;
}

// Bits needed to count from 0 to n - 1 (at least 1).
constexpr int hex_lb_bits(unsigned n) { return n <= 2 ? 1 : 1 + hex_lb_bits((n + 1) / 2); }
} // namespace nnet
#endif

// Fused hexagonal neighbor max-pool for the "linebuffer" export strategy —
// io_parallel variant + the shared per-pixel max body reused by the io_stream
// line-buffer kernel (nnet_hex_pool_linebuffer_stream.h).
//
// Reduces the max over the K neighbor taps of each output pixel.  Border slots
// (indices == -1, or out-of-bounds in the stream window) contribute 0, matching
// hexagdly's zero-padded MaxPool border.  No weights.  Any hex stride: output
// pixel (oh, ow) maps to input base (stride*oh, stride*ow).
//
// Corresponds to the HexPoolLineBuffer Keras layer (hex_gather.py).

namespace nnet {

struct hex_pool_linebuffer_config {
    static const unsigned in_height  = 8;
    static const unsigned in_width   = 8;
    static const unsigned out_height = 8;
    static const unsigned out_width  = 8;
    static const unsigned stride     = 1;
    // io_stream only: frames streamed back to back (see hex_conv_linebuffer_config).
    static const unsigned n_frames   = 1;
    static const int      dr_max     = 1;
    static const unsigned n_chan     = 1;   // channels pass through
    static const unsigned k          = 7;   // neighbor slots K
    static const unsigned rows_resident = 2;
    // Column extent of the tap footprint, max(dc)-min(dc)+1 over both parities;
    // used by the io_stream kernel to bank the row buffer's column axis.
    static const unsigned col_span      = 3;
    // Max tap column offset; the io_stream kernel emits an output pixel dc_max
    // input columns behind the read head so its register window already holds
    // every tap column (see nnet_hex_conv_linebuffer_stream.h).
    static const int      dc_max        = 1;

    static const unsigned reuse_factor = 1;

    static const int indices[out_height * out_width * k];
    static const int off_even[k][2];
    static const int off_odd[k][2];
};

// Shared per-output-pixel max reduction over the K taps (border taps are 0).
template <typename CONFIG_T, class dval, class rval>
void hex_lb_pixel_max(
    const dval taps[CONFIG_T::k * CONFIG_T::n_chan],
    rval       res_out[CONFIG_T::n_chan]
) {
    #pragma HLS INLINE
    constexpr unsigned C = CONFIG_T::n_chan;
    for (unsigned c = 0; c < C; c++) {
        #pragma HLS UNROLL
        // Balanced tree (hls4ml's nnet::reduce), not a running max: a chain of k - 1
        // compare-selects is k - 1 levels deep, and since the window moves every cycle each tap
        // still waiting for its comparison is carried in pipeline registers -- registers that
        // grow as k^2 / 2 (0.7.1, k = 7, 16 channels of 24 bits: 16 657 FF for a 4 608-bit window).
        dval column[CONFIG_T::k];
        #pragma HLS ARRAY_PARTITION variable=column complete
        for (unsigned ki = 0; ki < CONFIG_T::k; ki++) {
            #pragma HLS UNROLL
            column[ki] = taps[ki * C + c];
        }
        nnet::Op_max<dval> op_max;
        res_out[c] = (rval)nnet::reduce<dval, CONFIG_T::k, nnet::Op_max<dval>>(column, op_max);
    }
}

// io_parallel fused gather + max (fallback; same cost as strategy="gather").
template <class data_T, class res_T, typename CONFIG_T>
void hex_pool_linebuffer(
    data_T input  [CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  output [CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_chan]
) {
    constexpr unsigned N_OUT = CONFIG_T::out_height * CONFIG_T::out_width;
    constexpr unsigned C     = CONFIG_T::n_chan;

PoolPixel:
    for (unsigned n = 0; n < N_OUT; n++) {
        #pragma HLS PIPELINE II=1
        data_T taps[CONFIG_T::k * C];
        #pragma HLS ARRAY_PARTITION variable=taps complete
        for (unsigned ki = 0; ki < CONFIG_T::k; ki++) {
            #pragma HLS UNROLL
            const int idx = CONFIG_T::indices[n * CONFIG_T::k + ki];
            for (unsigned c = 0; c < C; c++) {
                #pragma HLS UNROLL
                taps[ki * C + c] = (idx >= 0) ? input[(unsigned)idx * C + c] : (data_T)0;
            }
        }
        res_T res_out[C];
        #pragma HLS ARRAY_PARTITION variable=res_out complete
        hex_lb_pixel_max<CONFIG_T, data_T, res_T>(taps, res_out);
        for (unsigned c = 0; c < C; c++) {
            #pragma HLS UNROLL
            output[n * C + c] = res_out[c];
        }
    }
}

} // namespace nnet

#endif // NNET_HEX_POOL_LINEBUFFER_H_
