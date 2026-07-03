#ifndef NNET_HEX_RING_MAC_H_
#define NNET_HEX_RING_MAC_H_

#include "nnet_common.h"
#include "nnet_helpers.h"

// Hexagonal ring-sharing MAC for keras-hexagdly hls4ml export.
//
// Takes the gathered tensor (N_out * K * Cin) from hex_gather and applies
// the learned convolution weights to produce (N_out * Cout).
//
// Structure mirrors hls4ml's conv_2d_latency_cl:
//
//   PartitionLoop (sequential, PIPELINE II=reuse_factor)
//     PixelLoop   (UNROLL over n_pixels concurrent output pixels)
//       Product   (fully unrolled K x Cin x Cout products, static indices)
//       Accum     (balanced adder tree into acc[pixel][cout])
//
// Two independent knobs:
//   n_pixels     (= hls4ml ParallelizationFactor): output pixels computed
//                concurrently.  Hardware cost scales with n_pixels, latency
//                with n_partitions = n_out / n_pixels.
//   reuse_factor: pipeline II of the partition loop; the multiplier bank is
//                time-shared reuse_factor times (multiplier_limit caps it).
//
// Ring weight sharing (the HexRingMAC share_neighbors mode) is baked at
// export time into the compile-time constant CONFIG_T::row_map[k], which
// maps each neighbor slot to its weight row (identity when not sharing).
// No runtime ring_idx indirection exists in hardware.
//
// Every index into mult/acc/weights is a compile-time constant after
// unrolling, so HLS builds no crossbars and no runtime div/mod - the
// pathologies of the previous flat-index implementation.
//
// Corresponds to the HexRingMAC Keras layer (hex_gather.py).

namespace nnet {

struct hex_ring_mac_config {
    static const unsigned n_out          = 81;    // output pixels N_out
    static const unsigned k              = 7;     // neighbor slots K
    static const unsigned n_in_chan      = 1;     // input channels Cin
    static const unsigned n_out_chan     = 4;     // output channels Cout
    static const unsigned num_weight_rows = 7;    // K (full) or num_rings (shared)

    // Parallelization: n_pixels output pixels per partition-loop iteration.
    static const unsigned n_pixels     = 1;
    static const unsigned n_partitions = 81;      // n_out / n_pixels

    // Resource reuse: partition-loop II; caps the parallel multiplier count.
    static const unsigned reuse_factor    = 1;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(n_pixels * k * n_in_chan * n_out_chan, reuse_factor);

    // Slot -> weight-row map, generated (out-of-line) in parameters.h.
    static const unsigned row_map[k];

    typedef float weight_t;
    typedef float bias_t;
    typedef float accum_t;
};

template<class data_T, class w_T, class b_T, class res_T, typename CONFIG_T>
void hex_ring_mac(
    data_T  input   [CONFIG_T::n_out * CONFIG_T::k * CONFIG_T::n_in_chan],
    w_T     weights [CONFIG_T::num_weight_rows * CONFIG_T::n_in_chan * CONFIG_T::n_out_chan],
    b_T     bias    [CONFIG_T::n_out_chan],
    res_T   output  [CONFIG_T::n_out * CONFIG_T::n_out_chan]
) {
    static_assert(CONFIG_T::n_out == CONFIG_T::n_pixels * CONFIG_T::n_partitions,
                  "hex_ring_mac: n_pixels (ParallelizationFactor) must divide n_out");
    static_assert(CONFIG_T::reuse_factor >= 1, "hex_ring_mac: reuse_factor must be >= 1");

    constexpr unsigned mult_n_in  = CONFIG_T::k * CONFIG_T::n_in_chan;
    constexpr unsigned mult_n_out = CONFIG_T::n_out_chan;

    #pragma HLS ARRAY_PARTITION variable=weights complete
    #pragma HLS ARRAY_PARTITION variable=bias    complete
    #pragma HLS FUNCTION_INSTANTIATE variable=weights,bias

    // Per-concurrent-pixel product and accumulator banks. Indexed only by
    // compile-time constants after unrolling -> registers, no crossbars.
    typename CONFIG_T::accum_t mult[CONFIG_T::n_pixels][mult_n_in * mult_n_out];
    #pragma HLS ARRAY_PARTITION variable=mult complete dim=0
    typename CONFIG_T::accum_t acc[CONFIG_T::n_pixels][mult_n_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    // Cap parallel multipliers: satisfiable by construction (per-partition work).
    #pragma HLS ALLOCATION operation instances=mul limit=CONFIG_T::multiplier_limit

PartitionLoop:
    for (unsigned i_part = 0; i_part < CONFIG_T::n_partitions; i_part++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor rewind

    PixelLoop:
        for (unsigned i_pxl = 0; i_pxl < CONFIG_T::n_pixels; i_pxl++) {
            #pragma HLS UNROLL
            const unsigned n = i_part * CONFIG_T::n_pixels + i_pxl;

        // Products: row_map bakes ring sharing at compile time.
        Product1:
            for (unsigned ki = 0; ki < CONFIG_T::k; ki++) {
                #pragma HLS UNROLL
                const unsigned row = CONFIG_T::row_map[ki];
            Product2:
                for (unsigned c = 0; c < CONFIG_T::n_in_chan; c++) {
                    #pragma HLS UNROLL
                    const data_T cache =
                        input[(n * CONFIG_T::k + ki) * CONFIG_T::n_in_chan + c];
                Product3:
                    for (unsigned o = 0; o < mult_n_out; o++) {
                        #pragma HLS UNROLL
                        mult[i_pxl][(ki * CONFIG_T::n_in_chan + c) * mult_n_out + o] =
                            cache * weights[(row * CONFIG_T::n_in_chan + c) * mult_n_out + o];
                    }
                }
            }

        // Initialize accumulators with the bias.
        ResetAccum:
            for (unsigned o = 0; o < mult_n_out; o++) {
                #pragma HLS UNROLL
                acc[i_pxl][o] = (typename CONFIG_T::accum_t)bias[o];
            }

        // Accumulate products (balanced adder tree).
        Accum1:
            for (unsigned i_in = 0; i_in < mult_n_in; i_in++) {
                #pragma HLS UNROLL
            Accum2:
                for (unsigned o = 0; o < mult_n_out; o++) {
                    #pragma HLS UNROLL
                    acc[i_pxl][o] += mult[i_pxl][i_in * mult_n_out + o];
                }
            }

        // Cast to result type and write out.
        Result:
            for (unsigned o = 0; o < mult_n_out; o++) {
                #pragma HLS UNROLL
                output[n * mult_n_out + o] = res_T(acc[i_pxl][o]);
            }
        }
    }
}

} // namespace nnet

#endif // NNET_HEX_RING_MAC_H_
