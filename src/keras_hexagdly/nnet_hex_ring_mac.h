#ifndef NNET_HEX_RING_MAC_H_
#define NNET_HEX_RING_MAC_H_

#include "nnet_common.h"
#include "nnet_helpers.h"

// Hexagonal ring-sharing MAC for keras-hexagdly hls4ml export.
//
// Takes the gathered tensor (N_out * K * Cin) from hex_gather and applies
// the learned convolution weights to produce (N_out * Cout).
//
//   PartitionLoop (sequential, PIPELINE II=reuse_factor)
//     PixelLoop   (UNROLL over n_pixels concurrent output pixels)
//       RingSum   (adds only: fold the K slots down to num_weight_rows ring
//                  sums, grouping by the compile-time row_map)
//       Product   (one multiply per (ring, Cin, Cout): num_weight_rows x
//                  Cin x Cout products, static indices)
//       Accum     (balanced adder tree into acc[pixel][cout])
//
// Accumulate-then-scale exploits ring weight sharing in HARDWARE: neighbors
// that share a weight (same ring) are summed BEFORE the multiply, so the
// multiplier count is num_weight_rows*Cin*Cout instead of K*Cin*Cout. For
// share_neighbors the ring is smaller than K (k=1: 2 rows vs 7 slots, ~3.5x
// fewer DSPs; k=2: 3 vs 19). When NOT sharing, num_weight_rows == k and
// row_map is the identity, so every ring holds exactly one slot: the ring sum
// is a no-op pass-through and this reduces to the plain K-product MAC,
// bit-identical to the per-slot form. Legal because w*(x_a+x_b) == w*x_a+w*x_b
// exactly when the sum is formed in the (widened) accum_t before the multiply.
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
// Every index into ring_sum/acc/weights is a compile-time constant after
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
    // num_weight_rows-based: the kernel ring-sums shared slots before the multiply.
    static const unsigned reuse_factor    = 1;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(n_pixels * num_weight_rows * n_in_chan * n_out_chan, reuse_factor);

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

    constexpr unsigned n_rows     = CONFIG_T::num_weight_rows;  // rings (== k when not sharing)
    constexpr unsigned mult_n_out = CONFIG_T::n_out_chan;

    #pragma HLS ARRAY_PARTITION variable=weights complete
    #pragma HLS ARRAY_PARTITION variable=bias    complete
    #pragma HLS FUNCTION_INSTANTIATE variable=weights,bias

    // Per-concurrent-pixel ring-sum and accumulator banks. Indexed only by
    // compile-time constants after unrolling -> registers, no crossbars.
    // ring_sum folds the K gathered slots into num_weight_rows ring totals BEFORE
    // the multiply, so a shared ring costs one multiplier instead of |ring|.
    typename CONFIG_T::accum_t ring_sum[CONFIG_T::n_pixels][n_rows * CONFIG_T::n_in_chan];
    #pragma HLS ARRAY_PARTITION variable=ring_sum complete dim=0
    typename CONFIG_T::accum_t acc[CONFIG_T::n_pixels][mult_n_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    // Cap parallel multipliers: satisfiable by construction (per-partition work).
    // multiplier_limit is num_weight_rows-based, so shared rings actually free DSPs.
    #pragma HLS ALLOCATION operation instances=mul limit=CONFIG_T::multiplier_limit

PartitionLoop:
    for (unsigned i_part = 0; i_part < CONFIG_T::n_partitions; i_part++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor rewind

    PixelLoop:
        for (unsigned i_pxl = 0; i_pxl < CONFIG_T::n_pixels; i_pxl++) {
            #pragma HLS UNROLL
            const unsigned n = i_part * CONFIG_T::n_pixels + i_pxl;

        // Stage 1 - RingSum (adds only): fold the K slots into num_weight_rows
        // ring totals. row_map[ki] is a compile-time constant, so each += targets
        // a static ring_sum element and HLS builds an adder tree (no multiplies).
        ResetRing:
            for (unsigned rc = 0; rc < n_rows * CONFIG_T::n_in_chan; rc++) {
                #pragma HLS UNROLL
                ring_sum[i_pxl][rc] = (typename CONFIG_T::accum_t)0;
            }
        RingSum1:
            for (unsigned ki = 0; ki < CONFIG_T::k; ki++) {
                #pragma HLS UNROLL
                const unsigned row = CONFIG_T::row_map[ki];
            RingSum2:
                for (unsigned c = 0; c < CONFIG_T::n_in_chan; c++) {
                    #pragma HLS UNROLL
                    ring_sum[i_pxl][row * CONFIG_T::n_in_chan + c] +=
                        (typename CONFIG_T::accum_t)
                        input[(n * CONFIG_T::k + ki) * CONFIG_T::n_in_chan + c];
                }
            }

        // Initialize accumulators with the bias.
        ResetAccum:
            for (unsigned o = 0; o < mult_n_out; o++) {
                #pragma HLS UNROLL
                acc[i_pxl][o] = (typename CONFIG_T::accum_t)bias[o];
            }

        // Stage 2 - Product + Accum: one multiply per (ring, Cin, Cout), summed
        // into acc via a balanced adder tree. All indices compile-time constant.
        Product1:
            for (unsigned row = 0; row < n_rows; row++) {
                #pragma HLS UNROLL
            Product2:
                for (unsigned c = 0; c < CONFIG_T::n_in_chan; c++) {
                    #pragma HLS UNROLL
                    const typename CONFIG_T::accum_t s =
                        ring_sum[i_pxl][row * CONFIG_T::n_in_chan + c];
                Product3:
                    for (unsigned o = 0; o < mult_n_out; o++) {
                        #pragma HLS UNROLL
                        acc[i_pxl][o] +=
                            s * weights[(row * CONFIG_T::n_in_chan + c) * mult_n_out + o];
                    }
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
