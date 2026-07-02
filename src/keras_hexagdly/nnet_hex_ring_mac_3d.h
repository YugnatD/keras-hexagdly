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

    // Resource reuse: pipeline II = reuse_factor and cap the parallel multiplier
    // count at multiplier_limit. n_mult counts every multiply across all frames.
    static const unsigned reuse_factor    = 1;
    static const unsigned n_mult          = 1 * 81 * 7 * 1 * 4;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_mult, reuse_factor);

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
    // Resource-reuse structure modelled on hls4ml's dense_resource (see the 2D
    // nnet_hex_ring_mac.h for the rationale). The flat product index here covers
    // depth as the leading dimension: n_mult = n_depth*n_out*k*n_in_chan*n_out_chan.
    // The same weight ROM is applied to every depth frame (w_idx ignores d).
    // At reuse_factor=1 this is a single fully-parallel cycle, matching the
    // previous behaviour bit-for-bit.
    const unsigned reuse        = CONFIG_T::reuse_factor;
    const unsigned block_factor = DIV_ROUNDUP(CONFIG_T::n_mult, reuse);

    #pragma HLS ARRAY_PARTITION variable=ring_idx complete
    #pragma HLS ARRAY_PARTITION variable=bias     complete
    #pragma HLS ARRAY_PARTITION variable=output   complete
    #pragma HLS ARRAY_RESHAPE variable=weights block factor=block_factor
    #pragma HLS FUNCTION_INSTANTIATE variable=weights,bias

    const unsigned CO    = CONFIG_T::n_out_chan;
    const unsigned CICO  = CONFIG_T::n_in_chan * CONFIG_T::n_out_chan;
    const unsigned KCICO = CONFIG_T::k * CICO;
    const unsigned NKCICO = CONFIG_T::n_out * KCICO;  // products per depth frame

    typename CONFIG_T::accum_t acc[CONFIG_T::n_depth * CONFIG_T::n_out * CONFIG_T::n_out_chan];
    #pragma HLS ARRAY_PARTITION variable=acc complete

// Seed every (depth, pixel) accumulator with the bias.  The Conv3d export
// passes a real bias only on the final depth tap (zeros elsewhere), so the
// total contribution across taps is 1x.
InitAccum:
    for (unsigned i = 0; i < CONFIG_T::n_depth * CONFIG_T::n_out * CONFIG_T::n_out_chan; i++) {
        #pragma HLS UNROLL
        acc[i] = typename CONFIG_T::accum_t(bias[i % CO]);
    }

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
            // Decode flat product index -> (d, n, ki, c, o).
            const unsigned o  = idx % CO;
            const unsigned c  = (idx / CO) % CONFIG_T::n_in_chan;
            const unsigned ki = (idx / CICO) % CONFIG_T::k;
            const unsigned n  = (idx / KCICO) % CONFIG_T::n_out;
            const unsigned d  = idx / NKCICO;
            const unsigned row = CONFIG_T::share_neighbors
                                 ? (unsigned)(int)ring_idx[ki]
                                 : ki;
            const data_T in_val =
                input[d * CONFIG_T::n_out * CONFIG_T::k * CONFIG_T::n_in_chan
                      + (n * CONFIG_T::k + ki) * CONFIG_T::n_in_chan + c];
            const unsigned w_idx = (row * CONFIG_T::n_in_chan + c) * CO + o;
            acc[(d * CONFIG_T::n_out + n) * CO + o] +=
                typename CONFIG_T::accum_t(in_val * weights[w_idx]);
        }
    }

WriteOut:
    for (unsigned i = 0; i < CONFIG_T::n_depth * CONFIG_T::n_out * CONFIG_T::n_out_chan; i++) {
        #pragma HLS UNROLL
        output[i] = res_T(acc[i]);
    }
}

} // namespace nnet

#endif // NNET_HEX_RING_MAC_3D_H_
