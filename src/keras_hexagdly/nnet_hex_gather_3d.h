#ifndef NNET_HEX_GATHER_3D_H_
#define NNET_HEX_GATHER_3D_H_

#include "nnet_common.h"
#include "nnet_helpers.h"

// Depth-aware hexagonal neighbor gather for keras-hexagdly Conv3d hls4ml export.
//
// Reads a flat tensor (n_depth * n_in * n_chan) and gathers each pixel's K
// neighbors into (n_depth * n_out * k * n_chan) using a precomputed integer
// index ROM.  The same spatial index table is applied to every one of the
// n_depth frames — the depth axis is a passthrough leading dimension.
//
// Border slots (index == -1) produce zero output, matching hexagdly's
// zero-padding behavior at the camera edge.
//
// Corresponds to the HexGather3D Keras layer (hex_gather.py).  This is the 2D
// nnet_hex_gather body wrapped in an outer depth loop with per-frame offsets.

namespace nnet {

struct hex_gather_3d_config {
    static const unsigned n_depth = 1;   // number of depth frames D
    static const unsigned n_in    = 81;  // input pixels  per frame  N_in
    static const unsigned n_out   = 81;  // output pixels per frame  N_out
    static const unsigned k       = 7;   // neighbor slots K
    static const unsigned n_chan  = 1;   // channels C

    typedef int indices_t;
};

template<class data_T, class idx_T, class res_T, typename CONFIG_T>
void hex_gather_3d(
    data_T  input  [CONFIG_T::n_depth * CONFIG_T::n_in  * CONFIG_T::n_chan],
    idx_T   indices[CONFIG_T::n_out * CONFIG_T::k],
    res_T   output [CONFIG_T::n_depth * CONFIG_T::n_out * CONFIG_T::k * CONFIG_T::n_chan]
) {
    #pragma HLS PIPELINE II=1
    #pragma HLS ARRAY_PARTITION variable=input   complete
    #pragma HLS ARRAY_PARTITION variable=indices complete
    #pragma HLS ARRAY_PARTITION variable=output  complete

GatherDepth:
    for (unsigned d = 0; d < CONFIG_T::n_depth; d++) {
        #pragma HLS UNROLL
        const unsigned in_base  = d * CONFIG_T::n_in  * CONFIG_T::n_chan;
        const unsigned out_base = d * CONFIG_T::n_out * CONFIG_T::k * CONFIG_T::n_chan;
    GatherOut:
        for (unsigned n = 0; n < CONFIG_T::n_out; n++) {
            #pragma HLS UNROLL
        GatherSlot:
            for (unsigned ki = 0; ki < CONFIG_T::k; ki++) {
                #pragma HLS UNROLL
                // Cast to int: hls4ml may assign ap_fixed precision to the index
                // table; an explicit int cast ensures correct sign comparison and
                // array addressing regardless of the storage type.
                int idx = (int)indices[n * CONFIG_T::k + ki];
            GatherChan:
                for (unsigned c = 0; c < CONFIG_T::n_chan; c++) {
                    #pragma HLS UNROLL
                    unsigned out_idx = out_base + (n * CONFIG_T::k + ki) * CONFIG_T::n_chan + c;
                    if (idx >= 0) {
                        output[out_idx] = input[in_base + (unsigned)idx * CONFIG_T::n_chan + c];
                    } else {
                        // border slot: zero-pad (matches hexagdly grid border behavior)
                        output[out_idx] = res_T(0);
                    }
                }
            }
        }
    }
}

} // namespace nnet

#endif // NNET_HEX_GATHER_3D_H_
