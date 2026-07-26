#ifndef NNET_HEX_POOL_LINEBUFFER_H_
#define NNET_HEX_POOL_LINEBUFFER_H_

#include "nnet_common.h"
#include "nnet_helpers.h"

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
        dval acc = taps[c];  // slot 0
        for (unsigned ki = 1; ki < CONFIG_T::k; ki++) {
            #pragma HLS UNROLL
            const dval v = taps[ki * C + c];
            acc = (v > acc) ? v : acc;
        }
        res_out[c] = (rval)acc;
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
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
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
