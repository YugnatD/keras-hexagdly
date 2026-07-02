#ifndef NNET_HEX_RING_MAC_3D_H_
#define NNET_HEX_RING_MAC_3D_H_

#include "nnet_common.h"
#include "nnet_helpers.h"

// Depth-aware hexagonal ring-sharing MAC for keras-hexagdly Conv3d hls4ml export.
//
// Takes the gathered tensor (n_depth * n_out * K * Cin) from hex_gather_3d and
// applies a single depth tap's convolution weights to produce
// (n_depth * n_out * Cout).  The same weight ROM is applied to every one of the
// n_depth frames — the depth axis is a passthrough leading dimension; summation
// across depth taps happens in the Keras graph (a binary Add tree over the
// per-tap HexRingMAC3D outputs), not here.
//
// Weight modes match nnet_hex_ring_mac.h (share_neighbors true/false).
//
// Corresponds to the HexRingMAC3D Keras layer (hex_gather.py).  This is the 2D
// nnet_hex_ring_mac body wrapped in an outer depth loop with per-frame offsets.

namespace nnet {

struct hex_ring_mac_3d_config {
    static const unsigned n_depth         = 1;     // depth frames D
    static const unsigned n_out           = 81;    // output pixels per frame N_out
    static const unsigned k               = 7;     // neighbor slots K
    static const unsigned n_in_chan       = 1;     // input channels Cin
    static const unsigned n_out_chan      = 4;     // output channels Cout
    static const unsigned num_weight_rows = 7;     // K (full) or num_rings (shared)
    static const bool     share_neighbors = false;

    typedef float weight_t;
    typedef int   ring_idx_t;
    typedef float bias_t;
    typedef float accum_t;
};

template<class data_T, class w_T, class ridx_T, class b_T, class res_T, typename CONFIG_T>
void hex_ring_mac_3d(
    data_T  input     [CONFIG_T::n_depth * CONFIG_T::n_out * CONFIG_T::k * CONFIG_T::n_in_chan],
    w_T     weights   [CONFIG_T::num_weight_rows * CONFIG_T::n_in_chan * CONFIG_T::n_out_chan],
    ridx_T  ring_idx  [CONFIG_T::k],
    b_T     bias      [CONFIG_T::n_out_chan],
    res_T   output    [CONFIG_T::n_depth * CONFIG_T::n_out * CONFIG_T::n_out_chan]
) {
    #pragma HLS PIPELINE II=1
    #pragma HLS ARRAY_PARTITION variable=input    complete
    #pragma HLS ARRAY_PARTITION variable=weights  complete
    #pragma HLS ARRAY_PARTITION variable=ring_idx complete
    #pragma HLS ARRAY_PARTITION variable=bias     complete
    #pragma HLS ARRAY_PARTITION variable=output   complete

DepthLoop:
    for (unsigned d = 0; d < CONFIG_T::n_depth; d++) {
        #pragma HLS UNROLL
        const unsigned in_base  = d * CONFIG_T::n_out * CONFIG_T::k * CONFIG_T::n_in_chan;
        const unsigned out_base = d * CONFIG_T::n_out * CONFIG_T::n_out_chan;

        typename CONFIG_T::accum_t acc[CONFIG_T::n_out * CONFIG_T::n_out_chan];
        #pragma HLS ARRAY_PARTITION variable=acc complete

    InitAccum:
        for (unsigned n = 0; n < CONFIG_T::n_out; n++) {
            #pragma HLS UNROLL
            for (unsigned o = 0; o < CONFIG_T::n_out_chan; o++) {
                #pragma HLS UNROLL
                // Seed with bias so it is added once per output pixel per frame.
                // The Conv3d export passes a real bias only on the final depth
                // tap (zeros on the others), so the total bias contribution is 1x.
                acc[n * CONFIG_T::n_out_chan + o] = typename CONFIG_T::accum_t(bias[o]);
            }
        }

    MacOut:
        for (unsigned n = 0; n < CONFIG_T::n_out; n++) {
            #pragma HLS UNROLL
        MacSlot:
            for (unsigned ki = 0; ki < CONFIG_T::k; ki++) {
                #pragma HLS UNROLL
                // Map slot ki to weight row: ring index when sharing, slot index otherwise.
                // Cast to int: ring_idx may be stored as ap_fixed by hls4ml.
                unsigned row = CONFIG_T::share_neighbors
                               ? (unsigned)(int)ring_idx[ki]
                               : ki;
            MacCin:
                for (unsigned c = 0; c < CONFIG_T::n_in_chan; c++) {
                    #pragma HLS UNROLL
                    data_T in_val = input[in_base + (n * CONFIG_T::k + ki) * CONFIG_T::n_in_chan + c];
                MacCout:
                    for (unsigned o = 0; o < CONFIG_T::n_out_chan; o++) {
                        #pragma HLS UNROLL
                        unsigned w_idx = (row * CONFIG_T::n_in_chan + c) * CONFIG_T::n_out_chan + o;
                        acc[n * CONFIG_T::n_out_chan + o] +=
                            typename CONFIG_T::accum_t(in_val * weights[w_idx]);
                    }
                }
            }
        }

    WriteOut:
        for (unsigned n = 0; n < CONFIG_T::n_out; n++) {
            #pragma HLS UNROLL
            for (unsigned o = 0; o < CONFIG_T::n_out_chan; o++) {
                #pragma HLS UNROLL
                output[out_base + n * CONFIG_T::n_out_chan + o] =
                    res_T(acc[n * CONFIG_T::n_out_chan + o]);
            }
        }
    }
}

} // namespace nnet

#endif // NNET_HEX_RING_MAC_3D_H_
