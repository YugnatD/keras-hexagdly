#ifndef NNET_HEX_MAX_POOL_H_
#define NNET_HEX_MAX_POOL_H_

#include "nnet_common.h"
#include "nnet_helpers.h"

// Hexagonal neighbor max pooling for keras-hexagdly hls4ml export.
//
// Takes the gathered tensor (N_out * K * n_chan) output of nnet_hex_gather
// and reduces each pixel's K neighbor values to a single max per channel.
//
// Border slots were zeroed by HexGather.  The accumulator is initialised to
// the first slot value so that a pure-negative input returns a negative max
// rather than 0.  Border (zero) slots still participate normally: if all K
// real neighbors are negative and at least one slot is a border (zero), the
// result is 0 — exactly matching hexagdly's zero-padding behavior.
//
// No weights — pure reduction.  Corresponds to HexMaxPool in hex_gather.py.

namespace nnet {

struct hex_max_pool_config {
    static const unsigned n_out  = 81;   // number of output pixels N_out
    static const unsigned k      = 7;    // number of neighbor slots K
    static const unsigned n_chan  = 1;   // channels C
};

template<class data_T, class res_T, typename CONFIG_T>
void hex_max_pool(
    data_T input [CONFIG_T::n_out * CONFIG_T::k * CONFIG_T::n_chan],
    res_T  output[CONFIG_T::n_out * CONFIG_T::n_chan]
) {
    #pragma HLS PIPELINE II=1
    #pragma HLS ARRAY_PARTITION variable=input  complete
    #pragma HLS ARRAY_PARTITION variable=output complete

PoolOut:
    for (unsigned n = 0; n < CONFIG_T::n_out; n++) {
        #pragma HLS UNROLL
    PoolChan:
        for (unsigned c = 0; c < CONFIG_T::n_chan; c++) {
            #pragma HLS UNROLL
            // Initialise from slot 0 so negative-only inputs return a negative max.
            // Border slots are 0 from HexGather, so they participate in the max
            // comparison — a pixel with all-negative real neighbors but >=1 border
            // slot will return 0, matching hexagdly's zero-padding behavior.
            res_T acc = input[n * CONFIG_T::k * CONFIG_T::n_chan + c];
        PoolSlot:
            for (unsigned ki = 1; ki < CONFIG_T::k; ki++) {
                #pragma HLS UNROLL
                data_T val = input[(n * CONFIG_T::k + ki) * CONFIG_T::n_chan + c];
                if (val > acc) acc = val;
            }
            output[n * CONFIG_T::n_chan + c] = acc;
        }
    }
}

} // namespace nnet

#endif // NNET_HEX_MAX_POOL_H_
