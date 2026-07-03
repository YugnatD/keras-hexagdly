#ifndef NNET_HEX_RING_MAC_3D_H_
#define NNET_HEX_RING_MAC_3D_H_

#include "nnet_common.h"
#include "nnet_helpers.h"

// Depth-aware hexagonal ring-sharing MAC for keras-hexagdly Conv3d hls4ml export.
//
// Takes the gathered tensor (n_depth * n_out * K * Cin) from hex_gather_3d and
// applies a single depth tap's convolution weights to produce
// (n_depth * n_out * Cout).  The same weight ROM is applied to every one of the
// n_depth frames - the depth axis is a passthrough leading dimension; summation
// across depth taps happens in the Keras graph (a binary Add tree over the
// per-tap HexRingMAC3D outputs), not here.
//
// Structure is the 2D nnet_hex_ring_mac partition-pipelined dense wrapped in
// an outer (sequential) depth loop; HLS flattens the perfect Depth x Partition
// nest into a single pipeline of n_depth * n_partitions iterations:
//
//   DepthLoop (sequential)
//     PartitionLoop (PIPELINE II=reuse_factor)
//       PixelLoop   (UNROLL over n_pixels concurrent output pixels)
//         Product / Accum with compile-time-constant indices
//
// Ring weight sharing is baked into the compile-time constant
// CONFIG_T::row_map[k] (identity when not sharing) - no runtime indirection.
//
// Corresponds to the HexRingMAC3D Keras layer (hex_gather.py).

namespace nnet {

struct hex_ring_mac_3d_config {
    static const unsigned n_depth         = 1;     // depth frames D
    static const unsigned n_out           = 81;    // output pixels per frame N_out
    static const unsigned k               = 7;     // neighbor slots K
    static const unsigned n_in_chan       = 1;     // input channels Cin
    static const unsigned n_out_chan      = 4;     // output channels Cout
    static const unsigned num_weight_rows = 7;     // K (full) or num_rings (shared)

    // Parallelization: n_pixels output pixels per partition-loop iteration
    // (per depth frame).
    static const unsigned n_pixels     = 1;
    static const unsigned n_partitions = 81;       // n_out / n_pixels

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
void hex_ring_mac_3d(
    data_T  input   [CONFIG_T::n_depth * CONFIG_T::n_out * CONFIG_T::k * CONFIG_T::n_in_chan],
    w_T     weights [CONFIG_T::num_weight_rows * CONFIG_T::n_in_chan * CONFIG_T::n_out_chan],
    b_T     bias    [CONFIG_T::n_out_chan],
    res_T   output  [CONFIG_T::n_depth * CONFIG_T::n_out * CONFIG_T::n_out_chan]
) {
    static_assert(CONFIG_T::n_out == CONFIG_T::n_pixels * CONFIG_T::n_partitions,
                  "hex_ring_mac_3d: n_pixels (ParallelizationFactor) must divide n_out");
    static_assert(CONFIG_T::reuse_factor >= 1, "hex_ring_mac_3d: reuse_factor must be >= 1");

    constexpr unsigned mult_n_in  = CONFIG_T::k * CONFIG_T::n_in_chan;
    constexpr unsigned mult_n_out = CONFIG_T::n_out_chan;

    #pragma HLS ARRAY_PARTITION variable=weights complete
    #pragma HLS ARRAY_PARTITION variable=bias    complete
    #pragma HLS FUNCTION_INSTANTIATE variable=weights,bias

    // Per-concurrent-pixel product and accumulator banks, reused across frames.
    typename CONFIG_T::accum_t mult[CONFIG_T::n_pixels][mult_n_in * mult_n_out];
    #pragma HLS ARRAY_PARTITION variable=mult complete dim=0
    typename CONFIG_T::accum_t acc[CONFIG_T::n_pixels][mult_n_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    #pragma HLS ALLOCATION operation instances=mul limit=CONFIG_T::multiplier_limit

DepthLoop:
    for (unsigned d = 0; d < CONFIG_T::n_depth; d++) {
        const unsigned in_base  = d * CONFIG_T::n_out * CONFIG_T::k * CONFIG_T::n_in_chan;
        const unsigned out_base = d * CONFIG_T::n_out * mult_n_out;

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
                            input[in_base + (n * CONFIG_T::k + ki) * CONFIG_T::n_in_chan + c];
                    Product3:
                        for (unsigned o = 0; o < mult_n_out; o++) {
                            #pragma HLS UNROLL
                            mult[i_pxl][(ki * CONFIG_T::n_in_chan + c) * mult_n_out + o] =
                                cache * weights[(row * CONFIG_T::n_in_chan + c) * mult_n_out + o];
                        }
                    }
                }

            // Seed accumulators with the bias.  The Conv3d export passes a real
            // bias only on the final depth tap (zeros elsewhere), so the total
            // contribution across taps is 1x.
            ResetAccum:
                for (unsigned o = 0; o < mult_n_out; o++) {
                    #pragma HLS UNROLL
                    acc[i_pxl][o] = (typename CONFIG_T::accum_t)bias[o];
                }

            Accum1:
                for (unsigned i_in = 0; i_in < mult_n_in; i_in++) {
                    #pragma HLS UNROLL
                Accum2:
                    for (unsigned o = 0; o < mult_n_out; o++) {
                        #pragma HLS UNROLL
                        acc[i_pxl][o] += mult[i_pxl][i_in * mult_n_out + o];
                    }
                }

            Result:
                for (unsigned o = 0; o < mult_n_out; o++) {
                    #pragma HLS UNROLL
                    output[out_base + n * mult_n_out + o] = res_T(acc[i_pxl][o]);
                }
            }
        }
    }
}

} // namespace nnet

#endif // NNET_HEX_RING_MAC_3D_H_
