#ifndef NNET_HEX_GATHER_H_
#define NNET_HEX_GATHER_H_

#include "nnet_common.h"
#include "nnet_helpers.h"

// Hexagonal neighbor gather for keras-hexagdly hls4ml export.
//
// Reads a flat pixel tensor (N_in * n_chan) and gathers each pixel's K
// neighbors into an output tensor (N_out * K * n_chan).
//
// The neighbor index table is a compile-time constant (CONFIG_T::indices,
// generated out-of-line in parameters.h) - the same trick hls4ml uses for
// its im2col fill_buffer codegen: with the loops fully unrolled every
// indices[] access constant-folds, so the gather synthesizes to pure wiring
// plus literal zeros for border slots.  No index ROM, no runtime sign test,
// no input crossbar.
//
// Border slots (index == -1) produce zero output, matching hexagdly's
// zero-padding behavior at the camera edge.
//
// Corresponds to the HexGather Keras layer (hex_gather.py).

namespace nnet {

struct hex_gather_config {
    static const unsigned n_in   = 81;   // number of input pixels  N_in
    static const unsigned n_out  = 81;   // number of output pixels N_out
    static const unsigned k      = 7;    // number of neighbor slots K
    static const unsigned n_chan  = 1;   // number of channels C

    // Neighbor index table (N_out * K), -1 = border slot -> zero output.
    // Generated (out-of-line) in parameters.h per layer.
    static const int indices[n_out * k];
};

template<class data_T, class res_T, typename CONFIG_T>
void hex_gather(
    data_T  input  [CONFIG_T::n_in  * CONFIG_T::n_chan],
    res_T   output [CONFIG_T::n_out * CONFIG_T::k * CONFIG_T::n_chan]
) {
    #pragma HLS PIPELINE II=1
    #pragma HLS ARRAY_PARTITION variable=input   complete
    #pragma HLS ARRAY_PARTITION variable=output  complete

GatherOut:
    for (unsigned n = 0; n < CONFIG_T::n_out; n++) {
        #pragma HLS UNROLL
    GatherSlot:
        for (unsigned ki = 0; ki < CONFIG_T::k; ki++) {
            #pragma HLS UNROLL
            // Compile-time constant after unroll -> folds to a wire or a zero.
            const int idx = CONFIG_T::indices[n * CONFIG_T::k + ki];
        GatherChan:
            for (unsigned c = 0; c < CONFIG_T::n_chan; c++) {
                #pragma HLS UNROLL
                const unsigned out_idx = (n * CONFIG_T::k + ki) * CONFIG_T::n_chan + c;
                if (idx >= 0) {
                    output[out_idx] = input[(unsigned)idx * CONFIG_T::n_chan + c];
                } else {
                    // border slot: zero-pad (matches hexagdly grid border behavior)
                    output[out_idx] = res_T(0);
                }
            }
        }
    }
}

} // namespace nnet

#endif // NNET_HEX_GATHER_H_
