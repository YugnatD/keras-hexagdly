#ifndef NNET_HEX_RING_MAC_H_
#define NNET_HEX_RING_MAC_H_

#include "nnet_common.h"
#include "nnet_helpers.h"

// Hexagonal ring-sharing MAC for keras-hexagdly hls4ml export.
//
// Takes the gathered tensor (N_out * K * Cin) from hex_gather and applies
// the learned convolution weights to produce (N_out * Cout).
//
// Two modes controlled by CONFIG_T::share_neighbors:
//
//   share_neighbors = false:
//     weights has shape (K, Cin, Cout) — one independent weight per slot.
//     ring_idx is ignored (identity mapping).
//
//   share_neighbors = true:
//     weights has shape (num_rings, Cin, Cout) — one weight per hex ring.
//     ring_idx has shape (K,) — maps each slot to its ring index.
//     Slots in the same ring share the same weight vector.
//     At kernel_size=1: num_rings=2 (center + ring-1 neighbors),
//     ring_idx = [0, 1, 1, 1, 1, 1, 1], saving 5/7 weight rows.
//
// Corresponds to the HexRingMAC Keras layer (hex_gather.py).

namespace nnet {

struct hex_ring_mac_config {
    static const unsigned n_out          = 81;    // output pixels N_out
    static const unsigned k              = 7;     // neighbor slots K
    static const unsigned n_in_chan      = 1;     // input channels Cin
    static const unsigned n_out_chan     = 4;     // output channels Cout
    static const unsigned num_weight_rows = 7;    // K (full) or num_rings (shared)
    static const bool     share_neighbors = false;

    typedef float weight_t;
    typedef int   ring_idx_t;
    typedef float accum_t;
};

template<class data_T, class w_T, class ridx_T, class res_T, typename CONFIG_T>
void hex_ring_mac(
    data_T  input     [CONFIG_T::n_out * CONFIG_T::k * CONFIG_T::n_in_chan],
    w_T     weights   [CONFIG_T::num_weight_rows * CONFIG_T::n_in_chan * CONFIG_T::n_out_chan],
    ridx_T  ring_idx  [CONFIG_T::k],
    res_T   output    [CONFIG_T::n_out * CONFIG_T::n_out_chan]
) {
    #pragma HLS PIPELINE II=1
    #pragma HLS ARRAY_PARTITION variable=input    complete
    #pragma HLS ARRAY_PARTITION variable=weights  complete
    #pragma HLS ARRAY_PARTITION variable=ring_idx complete
    #pragma HLS ARRAY_PARTITION variable=output   complete

    typename CONFIG_T::accum_t acc[CONFIG_T::n_out * CONFIG_T::n_out_chan];
    #pragma HLS ARRAY_PARTITION variable=acc complete

// Initialize accumulators
InitAccum:
    for (unsigned n = 0; n < CONFIG_T::n_out; n++) {
        #pragma HLS UNROLL
        for (unsigned o = 0; o < CONFIG_T::n_out_chan; o++) {
            #pragma HLS UNROLL
            acc[n * CONFIG_T::n_out_chan + o] = typename CONFIG_T::accum_t(0);
        }
    }

// Accumulate over pixels, slots, and input channels
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
                data_T in_val = input[(n * CONFIG_T::k + ki) * CONFIG_T::n_in_chan + c];
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

// Write output
WriteOut:
    for (unsigned n = 0; n < CONFIG_T::n_out; n++) {
        #pragma HLS UNROLL
        for (unsigned o = 0; o < CONFIG_T::n_out_chan; o++) {
            #pragma HLS UNROLL
            output[n * CONFIG_T::n_out_chan + o] =
                res_T(acc[n * CONFIG_T::n_out_chan + o]);
        }
    }
}

} // namespace nnet

#endif // NNET_HEX_RING_MAC_H_
