#ifndef NNET_HEX_MAX_POOL_3D_H_
#define NNET_HEX_MAX_POOL_3D_H_

#include "nnet_common.h"
#include "nnet_helpers.h"

// Depth-aware hexagonal max pooling for keras-hexagdly MaxPool3d hls4ml export.
//
// Takes the gathered tensor (n_depth_in * n_out * k * n_chan) output of
// nnet_hex_gather_3d and reduces, for each output frame, the max over both the
// depth pool window (depth_size taps at stride depth_stride) and the K spatial
// neighbor slots:
//
//   output[t, n, c] = max over d in [0, depth_size), ki in [0, k) of
//                     input[t*depth_stride + d, n, ki, c]
//
// n_depth_out = (n_depth_in - depth_size) / depth_stride + 1  (valid depth
// pooling; hex MaxPool has no depth padding).  Border slots were zeroed by
// HexGather3D, so a pixel whose real neighbors are all negative but has >=1
// border slot returns 0, matching hexagdly's zero-padding behavior.
//
// No weights.  Corresponds to the HexMaxPool3D Keras layer (hex_gather.py).
// This is the 2D nnet_hex_max_pool body with an added depth-window loop.

namespace nnet {

struct hex_max_pool_3d_config {
    static const unsigned n_depth_in   = 8;   // input frames  D_in
    static const unsigned n_depth_out  = 7;   // output frames D_out
    static const unsigned depth_size   = 2;   // depth pool window
    static const unsigned depth_stride = 1;   // depth pool stride
    static const unsigned n_out        = 81;  // output pixels per frame N_out
    static const unsigned k            = 7;   // neighbor slots K
    static const unsigned n_chan       = 1;   // channels C
};

template<class data_T, class res_T, typename CONFIG_T>
void hex_max_pool_3d(
    data_T input [CONFIG_T::n_depth_in  * CONFIG_T::n_out * CONFIG_T::k * CONFIG_T::n_chan],
    res_T  output[CONFIG_T::n_depth_out * CONFIG_T::n_out * CONFIG_T::n_chan]
) {
    #pragma HLS PIPELINE II=1
    #pragma HLS ARRAY_PARTITION variable=input  complete
    #pragma HLS ARRAY_PARTITION variable=output complete

    const unsigned frame_stride = CONFIG_T::n_out * CONFIG_T::k * CONFIG_T::n_chan;

PoolDepth:
    for (unsigned t = 0; t < CONFIG_T::n_depth_out; t++) {
        #pragma HLS UNROLL
        const unsigned t_base = t * CONFIG_T::depth_stride;
    PoolOut:
        for (unsigned n = 0; n < CONFIG_T::n_out; n++) {
            #pragma HLS UNROLL
        PoolChan:
            for (unsigned c = 0; c < CONFIG_T::n_chan; c++) {
                #pragma HLS UNROLL
                // Initialise from the first tap's slot 0 so a negative-only
                // window returns a negative max.  Border slots are 0 from
                // HexGather3D and participate in the comparison.
                unsigned first = t_base * frame_stride + (n * CONFIG_T::k) * CONFIG_T::n_chan + c;
                res_T acc = input[first];
            PoolTap:
                for (unsigned d = 0; d < CONFIG_T::depth_size; d++) {
                    #pragma HLS UNROLL
                    unsigned in_frame = (t_base + d) * frame_stride;
                PoolSlot:
                    for (unsigned ki = 0; ki < CONFIG_T::k; ki++) {
                        #pragma HLS UNROLL
                        data_T val = input[in_frame + (n * CONFIG_T::k + ki) * CONFIG_T::n_chan + c];
                        if (val > acc) acc = val;
                    }
                }
                output[(t * CONFIG_T::n_out + n) * CONFIG_T::n_chan + c] = acc;
            }
        }
    }
}

} // namespace nnet

#endif // NNET_HEX_MAX_POOL_3D_H_
