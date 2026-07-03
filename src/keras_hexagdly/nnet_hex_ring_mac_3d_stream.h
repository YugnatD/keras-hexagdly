#ifndef NNET_HEX_RING_MAC_3D_STREAM_H_
#define NNET_HEX_RING_MAC_3D_STREAM_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_hex_ring_mac_3d.h"
#include "nnet_types.h"

// Depth-aware hexagonal ring-sharing MAC — io_stream variant.
// Token counts derived from element type sizes.

namespace nnet {

template <class data_T, class w_T, class b_T, class res_T, typename CONFIG_T>
void hex_ring_mac_3d_stream(
    hls::stream<data_T> &input,
    w_T weights[CONFIG_T::num_weight_rows * CONFIG_T::n_in_chan * CONFIG_T::n_out_chan],
    b_T bias[CONFIG_T::n_out_chan],
    hls::stream<res_T> &output) {
    #pragma HLS ARRAY_PARTITION variable=weights complete
    #pragma HLS ARRAY_PARTITION variable=bias    complete
    #pragma HLS FUNCTION_INSTANTIATE variable=weights,bias

    constexpr unsigned N_IN  = CONFIG_T::n_depth * CONFIG_T::n_out * CONFIG_T::k * CONFIG_T::n_in_chan;
    constexpr unsigned N_OUT = CONFIG_T::n_depth * CONFIG_T::n_out * CONFIG_T::n_out_chan;
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

    nnet::hex_ring_mac_3d<typename data_T::value_type, w_T, b_T,
                          typename res_T::value_type, CONFIG_T>(in_arr, weights, bias, out_arr);

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

#endif // NNET_HEX_RING_MAC_3D_STREAM_H_
