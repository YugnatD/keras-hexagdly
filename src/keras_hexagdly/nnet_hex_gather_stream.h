#ifndef NNET_HEX_GATHER_STREAM_H_
#define NNET_HEX_GATHER_STREAM_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_types.h"

// Hexagonal neighbor gather — io_stream variant.
//
// hls4ml io_stream packs each variable with shape[-1] elements per token and
// depth = total_elements / shape[-1].  For our custom hex layers (1D flat
// shapes), this means the OUTPUT streams always have depth=1 (one big token),
// while the INPUT streams may have depth>1 when the upstream reshape emits
// one token per pixel.
//
// We derive the number of tokens to read/write from the stream element type
// sizes (data_T::size, res_T::size) rather than hard-coding config fields, so
// the same kernel works whether the upstream packs one pixel per token or the
// whole tensor per token.
//
//   in_tokens  = (n_in * n_chan)         / data_T::size
//   out_tokens = (n_out * k * n_chan)    / res_T::size   (= 1 in practice)

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void hex_gather_stream(hls::stream<data_T> &input, hls::stream<res_T> &output) {
    constexpr unsigned N_IN  = CONFIG_T::n_in  * CONFIG_T::n_chan;
    constexpr unsigned N_OUT = CONFIG_T::n_out * CONFIG_T::k * CONFIG_T::n_chan;
    constexpr unsigned IN_TOKENS  = N_IN  / data_T::size;
    constexpr unsigned OUT_TOKENS = N_OUT / res_T::size;

    typename data_T::value_type in_arr[N_IN];
    #pragma HLS ARRAY_PARTITION variable=in_arr complete

ReadInput:
    for (unsigned t = 0; t < IN_TOKENS; t++) {
        #pragma HLS PIPELINE II=1
        data_T tok = input.read();
        for (unsigned i = 0; i < data_T::size; i++) {
            #pragma HLS UNROLL
            in_arr[t * data_T::size + i] = tok[i];
        }
    }

    typename res_T::value_type out_arr[N_OUT];
    #pragma HLS ARRAY_PARTITION variable=out_arr complete

    nnet::hex_gather<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(in_arr, out_arr);

WriteOutput:
    for (unsigned t = 0; t < OUT_TOKENS; t++) {
        #pragma HLS PIPELINE II=1
        res_T tok;
        PRAGMA_DATA_PACK(tok)
        for (unsigned i = 0; i < res_T::size; i++) {
            #pragma HLS UNROLL
            tok[i] = out_arr[t * res_T::size + i];
        }
        output.write(tok);
    }
}

} // namespace nnet

#endif // NNET_HEX_GATHER_STREAM_H_
