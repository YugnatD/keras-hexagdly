#ifndef NNET_HEX_POOL_LINEBUFFER_3D_H_
#define NNET_HEX_POOL_LINEBUFFER_3D_H_

#include "nnet_common.h"
#include "nnet_helpers.h"
#include "nnet_hex_pool_linebuffer.h"

// Depth-aware fused hexagonal max-pool for the "linebuffer" strategy —
// io_parallel variant + the shared per-pixel body reused by the io_stream kernel
// (nnet_hex_pool_linebuffer_3d_stream.h).
//
// For each output frame t reduces the max over the depth pool window
// (depth_size taps at depth_stride) and the K spatial neighbor taps; border
// spatial taps contribute 0 (hexagdly zero-pad).  Hex stride 1 (output spatial
// grid == input grid).  n_depth_out = (n_depth_in - depth_size)/depth_stride + 1.
//
// Corresponds to the HexPoolLineBuffer3D Keras layer (hex_gather.py).

namespace nnet {

struct hex_pool_linebuffer_3d_config : hex_pool_linebuffer_config {
    static const unsigned n_depth_in   = 8;
    static const unsigned n_depth_out  = 8;
    static const unsigned depth_size   = 1;
    static const unsigned depth_stride = 1;
};

// Max over the depth window x K taps for one output pixel (border taps are 0).
template <typename CONFIG_T, class dval, class rval>
void hex_lb_pixel_max_3d(
    const dval taps[CONFIG_T::depth_size * CONFIG_T::k * CONFIG_T::n_chan],
    rval       res_out[CONFIG_T::n_chan]
) {
    #pragma HLS INLINE
    constexpr unsigned C = CONFIG_T::n_chan;
    constexpr unsigned M = CONFIG_T::depth_size * CONFIG_T::k;
    for (unsigned c = 0; c < C; c++) {
        #pragma HLS UNROLL
        dval acc = taps[c];  // (dd=0, ki=0)
        for (unsigned m = 1; m < M; m++) {
            #pragma HLS UNROLL
            const dval v = taps[m * C + c];
            acc = (v > acc) ? v : acc;
        }
        res_out[c] = (rval)acc;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void hex_pool_linebuffer_3d(
    data_T input  [CONFIG_T::n_depth_in * CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  output [CONFIG_T::n_depth_out * CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan]
) {
    constexpr unsigned N  = CONFIG_T::in_height * CONFIG_T::in_width;
    constexpr unsigned C  = CONFIG_T::n_chan;
    constexpr unsigned DK = CONFIG_T::depth_size;

DepthOut:
    for (unsigned t = 0; t < CONFIG_T::n_depth_out; t++) {
    SpatialPixel:
        for (unsigned n = 0; n < N; n++) {
            #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
            data_T taps[DK * CONFIG_T::k * C];
            #pragma HLS ARRAY_PARTITION variable=taps complete
            for (unsigned dd = 0; dd < DK; dd++) {
                #pragma HLS UNROLL
                const unsigned t_in = t * CONFIG_T::depth_stride + dd;
                for (unsigned ki = 0; ki < CONFIG_T::k; ki++) {
                    #pragma HLS UNROLL
                    const int idx = CONFIG_T::indices[n * CONFIG_T::k + ki];
                    for (unsigned c = 0; c < C; c++) {
                        #pragma HLS UNROLL
                        taps[(dd * CONFIG_T::k + ki) * C + c] =
                            (idx >= 0) ? input[(t_in * N + (unsigned)idx) * C + c] : (data_T)0;
                    }
                }
            }
            res_T res_out[C];
            #pragma HLS ARRAY_PARTITION variable=res_out complete
            hex_lb_pixel_max_3d<CONFIG_T, data_T, res_T>(taps, res_out);
            for (unsigned c = 0; c < C; c++) {
                #pragma HLS UNROLL
                output[(t * N + n) * C + c] = res_out[c];
            }
        }
    }
}

} // namespace nnet

#endif // NNET_HEX_POOL_LINEBUFFER_3D_H_
