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

    // Resource reuse: pipeline initiation interval = reuse_factor, and cap the
    // number of parallel multipliers at multiplier_limit = ceil(n_mult / reuse).
    static const unsigned reuse_factor    = 1;
    static const unsigned n_mult          = 81 * 7 * 1 * 4;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_mult, reuse_factor);

    typedef float weight_t;
    typedef int   ring_idx_t;
    typedef float bias_t;
    typedef float accum_t;
};

template<class data_T, class w_T, class ridx_T, class b_T, class res_T, typename CONFIG_T>
void hex_ring_mac(
    data_T  input     [CONFIG_T::n_out * CONFIG_T::k * CONFIG_T::n_in_chan],
    w_T     weights   [CONFIG_T::num_weight_rows * CONFIG_T::n_in_chan * CONFIG_T::n_out_chan],
    ridx_T  ring_idx  [CONFIG_T::k],
    b_T     bias      [CONFIG_T::n_out_chan],
    res_T   output    [CONFIG_T::n_out * CONFIG_T::n_out_chan]
) {
    // Resource-reuse structure modelled on hls4ml's dense_resource:
    //  - the total n_mult = n_out*k*n_in_chan*n_out_chan products are spread
    //    over reuse_factor cycles, block_factor = ceil(n_mult/reuse) per cycle;
    //  - the inner MultLoop is unrolled and pipelined at II=1, so HLS builds
    //    ~block_factor multipliers and cycles them reuse_factor times;
    //  - the strided index map (idx = ir + reuse*im) mirrors dense_resource so
    //    that same-cycle writes to one accumulator are minimised.
    // At reuse_factor=1 this is block_factor=n_mult in a single cycle -> fully
    // parallel, matching the original behaviour. Correctness is independent of
    // the mapping: every product is visited exactly once and added to acc[n,o].
    const unsigned reuse        = CONFIG_T::reuse_factor;
    const unsigned block_factor = DIV_ROUNDUP(CONFIG_T::n_mult, reuse);

    #pragma HLS ARRAY_PARTITION variable=ring_idx complete
    #pragma HLS ARRAY_PARTITION variable=bias     complete
    #pragma HLS ARRAY_PARTITION variable=output   complete
    // Reshape weights so the reuse loop can stream them from BRAM when reuse>1.
    #pragma HLS ARRAY_RESHAPE variable=weights block factor=block_factor
    #pragma HLS FUNCTION_INSTANTIATE variable=weights,bias

    typename CONFIG_T::accum_t acc[CONFIG_T::n_out * CONFIG_T::n_out_chan];
    #pragma HLS ARRAY_PARTITION variable=acc complete

// Initialize accumulators with the bias (added once per output pixel).
InitAccum:
    for (unsigned i = 0; i < CONFIG_T::n_out * CONFIG_T::n_out_chan; i++) {
        #pragma HLS UNROLL
        acc[i] = typename CONFIG_T::accum_t(bias[i % CONFIG_T::n_out_chan]);
    }

    const unsigned CO   = CONFIG_T::n_out_chan;
    const unsigned CICO = CONFIG_T::n_in_chan * CONFIG_T::n_out_chan;
    const unsigned KCICO = CONFIG_T::k * CICO;

ReuseLoop:
    for (unsigned ir = 0; ir < reuse; ir++) {
        #pragma HLS PIPELINE II=1 rewind
        #pragma HLS ALLOCATION operation instances=mul limit=block_factor
    MultLoop:
        for (unsigned im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL
            const unsigned idx = ir + reuse * im;  // strided cover of [0, n_mult)
            if (idx >= CONFIG_T::n_mult)
                continue;
            // Decode flat product index -> (n, ki, c, o).
            const unsigned o  = idx % CO;
            const unsigned c  = (idx / CO) % CONFIG_T::n_in_chan;
            const unsigned ki = (idx / CICO) % CONFIG_T::k;
            const unsigned n  = idx / KCICO;
            // Map slot ki to weight row: ring index when sharing, slot otherwise.
            const unsigned row = CONFIG_T::share_neighbors
                                 ? (unsigned)(int)ring_idx[ki]
                                 : ki;
            const data_T in_val = input[(n * CONFIG_T::k + ki) * CONFIG_T::n_in_chan + c];
            const unsigned w_idx = (row * CONFIG_T::n_in_chan + c) * CO + o;
            acc[n * CO + o] += typename CONFIG_T::accum_t(in_val * weights[w_idx]);
        }
    }

// Write output
WriteOut:
    for (unsigned i = 0; i < CONFIG_T::n_out * CONFIG_T::n_out_chan; i++) {
        #pragma HLS UNROLL
        output[i] = res_T(acc[i]);
    }
}

} // namespace nnet

#endif // NNET_HEX_RING_MAC_H_
