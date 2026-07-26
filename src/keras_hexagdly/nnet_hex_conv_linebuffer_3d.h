#ifndef NNET_HEX_CONV_LINEBUFFER_3D_H_
#define NNET_HEX_CONV_LINEBUFFER_3D_H_

#include "nnet_common.h"
#include "nnet_helpers.h"
#include "nnet_mult.h"
#include "nnet_hex_conv_linebuffer.h"

// Depth-aware fused hexagonal conv for the "linebuffer" export strategy —
// io_parallel variant + the shared per-pixel depth-MAC body reused by the
// io_stream kernel (nnet_hex_conv_linebuffer_3d_stream.h).
//
// The spatial neighbor pattern is identical to the 2D case; the depth axis is a
// dense 1D conv of depth_size taps (depth_stride 1), summed, bias added once.
// depth_padding is baked into pad_top / pad_bot (frames outside [0, n_depth_in)
// contribute zero).  Output frames: n_depth_out.
//
// Weights layout (flat): [depth_size][num_weight_rows][n_chan][n_filt], i.e. the
// per-tap ring-MAC weight, matching the (depth_size, rows, Cin, Cout) Keras array.
//
// Corresponds to the HexConvLineBuffer3D Keras layer (hex_gather.py).

namespace nnet {

struct hex_conv_linebuffer_3d_config : hex_conv_linebuffer_config {
    static const unsigned n_depth_in  = 8;
    static const unsigned n_depth_out = 8;
    static const unsigned depth_size  = 1;
    static const int      pad_top     = 0;   // depth pad (frames) before
    static const int      pad_bot     = 0;   // depth pad (frames) after
};

// Per-output-pixel MAC accumulating over depth_size taps (bias added once).
template <typename CONFIG_T, class dval, class w_T, class b_T, class rval>
void hex_lb_pixel_mac_3d(
    const dval taps[CONFIG_T::depth_size * CONFIG_T::k * CONFIG_T::n_chan],
    const w_T  weights[CONFIG_T::depth_size * CONFIG_T::num_weight_rows *
                       CONFIG_T::n_chan * CONFIG_T::n_filt],
    const b_T  bias[CONFIG_T::n_filt],
    rval       res_out[CONFIG_T::n_filt]
) {
    #pragma HLS INLINE
    typedef typename CONFIG_T::accum_t acc_T;
    constexpr unsigned n_rows = CONFIG_T::num_weight_rows;
    constexpr unsigned Cin    = CONFIG_T::n_chan;
    constexpr unsigned Cout   = CONFIG_T::n_filt;
    constexpr unsigned DK     = CONFIG_T::depth_size;

    acc_T acc[Cout];
    #pragma HLS ARRAY_PARTITION variable=acc complete
    for (unsigned o = 0; o < Cout; o++) {
        #pragma HLS UNROLL
        acc[o] = (acc_T)bias[o];
    }

DepthTap:
    for (unsigned dd = 0; dd < DK; dd++) {
        #pragma HLS UNROLL
        acc_T ring_sum[n_rows * Cin];
        #pragma HLS ARRAY_PARTITION variable=ring_sum complete
        for (unsigned rc = 0; rc < n_rows * Cin; rc++) {
            #pragma HLS UNROLL
            ring_sum[rc] = (acc_T)0;
        }
        for (unsigned ki = 0; ki < CONFIG_T::k; ki++) {
            #pragma HLS UNROLL
            const unsigned row = CONFIG_T::row_map[ki];
            for (unsigned c = 0; c < Cin; c++) {
                #pragma HLS UNROLL
                ring_sum[row * Cin + c] += (acc_T)taps[(dd * CONFIG_T::k + ki) * Cin + c];
            }
        }
        for (unsigned row = 0; row < n_rows; row++) {
            #pragma HLS UNROLL
            for (unsigned c = 0; c < Cin; c++) {
                #pragma HLS UNROLL
                const acc_T s = ring_sum[row * Cin + c];
                for (unsigned o = 0; o < Cout; o++) {
                    #pragma HLS UNROLL
                    acc[o] += (acc_T)CONFIG_T::template product<acc_T, w_T>::product(
                        s, weights[((dd * n_rows + row) * Cin + c) * Cout + o]);
                }
            }
        }
    }
    for (unsigned o = 0; o < Cout; o++) {
        #pragma HLS UNROLL
        res_out[o] = (rval)acc[o];
    }
}

// -----------------------------------------------------------------------------
// Depth-aware counterpart of hex_lb_pixel_mac_reuse: same result as
// hex_lb_pixel_mac_3d above, but the products are folded over reuse_factor
// cycles instead of all existing at once.
//
// The (depth, ring, channel) axes flatten into a single n_in of
// depth_size*num_weight_rows*n_chan, because the weight index
// ((dd*num_weight_rows + row)*n_chan + c)*n_filt + o is already
// in_idx*n_filt + o. So the fold is character-for-character the 2D one.
//
// This is what lifts the synthesis-time ceiling documented in
// nnet_hex_conv_linebuffer_3d_stream.h: 3D reached it sooner than 2D purely
// because depth_size multiplies the product count.
// -----------------------------------------------------------------------------
template <typename CONFIG_T, class dval, class w_T, class b_T, class rval>
void hex_lb_pixel_mac_3d_reuse(
    const dval taps[CONFIG_T::depth_size * CONFIG_T::k * CONFIG_T::n_chan],
    const w_T  weights[CONFIG_T::depth_size * CONFIG_T::num_weight_rows *
                       CONFIG_T::n_chan * CONFIG_T::n_filt],
    const b_T  bias[CONFIG_T::n_filt],
    rval       res_out[CONFIG_T::n_filt]
) {
    #pragma HLS INLINE
    typedef typename CONFIG_T::accum_t acc_T;
    constexpr unsigned n_rows = CONFIG_T::num_weight_rows;
    constexpr unsigned Cin    = CONFIG_T::n_chan;
    constexpr unsigned Cout   = CONFIG_T::n_filt;
    constexpr unsigned DK     = CONFIG_T::depth_size;
    constexpr unsigned N_IN   = DK * n_rows * Cin;
    constexpr unsigned RF     = CONFIG_T::reuse_factor;
    constexpr unsigned IN_PER = DIV_ROUNDUP(N_IN, RF);

    acc_T ring_sum[N_IN];
    #pragma HLS ARRAY_PARTITION variable=ring_sum complete
RingReset3D:
    for (unsigned i = 0; i < N_IN; i++) {
        #pragma HLS UNROLL
        ring_sum[i] = (acc_T)0;
    }
RingSum3D:
    for (unsigned dd = 0; dd < DK; dd++) {
        #pragma HLS UNROLL
        for (unsigned ki = 0; ki < CONFIG_T::k; ki++) {
            #pragma HLS UNROLL
            const unsigned row = CONFIG_T::row_map[ki];
            for (unsigned c = 0; c < Cin; c++) {
                #pragma HLS UNROLL
                ring_sum[(dd * n_rows + row) * Cin + c] +=
                    (acc_T)taps[(dd * CONFIG_T::k + ki) * Cin + c];
            }
        }
    }

    acc_T acc[Cout];
    #pragma HLS ARRAY_PARTITION variable=acc complete
InitAccum3D:
    for (unsigned o = 0; o < Cout; o++) {
        #pragma HLS UNROLL
        acc[o] = (acc_T)bias[o];
    }

ReuseLoop3D:
    for (unsigned ir = 0; ir < RF; ir++) {
        // See the 2D sibling for why rewind is used and what it measured.
        #pragma HLS PIPELINE II=1 rewind
    MultLoop3D:
        for (unsigned j = 0; j < IN_PER; j++) {
            #pragma HLS UNROLL
            const unsigned in_idx = ir * IN_PER + j;
            if (in_idx < N_IN) {  // only bites when RF does not divide N_IN
                const acc_T s = ring_sum[in_idx];
                for (unsigned o = 0; o < Cout; o++) {
                    #pragma HLS UNROLL
                    acc[o] += (acc_T)CONFIG_T::template product<acc_T, w_T>::product(
                        s, weights[in_idx * Cout + o]);
                }
            }
        }
    }

    for (unsigned o = 0; o < Cout; o++) {
        #pragma HLS UNROLL
        res_out[o] = (rval)acc[o];
    }
}

// io_parallel fused gather + depth-MAC (fallback; same cost as strategy="gather").
template <class data_T, class w_T, class b_T, class res_T, typename CONFIG_T>
void hex_conv_linebuffer_3d(
    data_T input  [CONFIG_T::n_depth_in * CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    w_T    weights[CONFIG_T::depth_size * CONFIG_T::num_weight_rows * CONFIG_T::n_chan * CONFIG_T::n_filt],
    b_T    bias   [CONFIG_T::n_filt],
    res_T  output [CONFIG_T::n_depth_out * CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_filt]
) {
    constexpr unsigned N     = CONFIG_T::in_height * CONFIG_T::in_width;  // spatial pixels
    constexpr unsigned Cin   = CONFIG_T::n_chan;
    constexpr unsigned Cout  = CONFIG_T::n_filt;
    constexpr unsigned DK    = CONFIG_T::depth_size;

    #pragma HLS FUNCTION_INSTANTIATE variable=weights,bias
    #pragma HLS ARRAY_PARTITION variable=bias complete
    #pragma HLS ARRAY_PARTITION variable=weights complete

DepthOut:
    for (int t_out = 0; t_out < (int)CONFIG_T::n_depth_out; t_out++) {
    SpatialPixel:
        for (unsigned n = 0; n < N; n++) {
            #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
            data_T taps[DK * CONFIG_T::k * Cin];
            #pragma HLS ARRAY_PARTITION variable=taps complete
        DepthGather:
            for (unsigned dd = 0; dd < DK; dd++) {
                #pragma HLS UNROLL
                const int t_in = t_out + (int)dd - CONFIG_T::pad_top;  // real frame index
                const bool frame_ok = (t_in >= 0) && (t_in < (int)CONFIG_T::n_depth_in);
                for (unsigned ki = 0; ki < CONFIG_T::k; ki++) {
                    #pragma HLS UNROLL
                    const int idx = CONFIG_T::indices[n * CONFIG_T::k + ki];
                    for (unsigned c = 0; c < Cin; c++) {
                        #pragma HLS UNROLL
                        const bool ok = frame_ok && (idx >= 0);
                        taps[(dd * CONFIG_T::k + ki) * Cin + c] = ok
                            ? input[((unsigned)t_in * N + (unsigned)idx) * Cin + c]
                            : (data_T)0;
                    }
                }
            }
            res_T res_out[Cout];
            #pragma HLS ARRAY_PARTITION variable=res_out complete
            hex_lb_pixel_mac_3d<CONFIG_T, data_T, w_T, b_T, res_T>(
                taps, weights, bias, res_out);
            for (unsigned o = 0; o < Cout; o++) {
                #pragma HLS UNROLL
                output[((unsigned)t_out * N + n) * Cout + o] = res_out[o];
            }
        }
    }
}

} // namespace nnet

#endif // NNET_HEX_CONV_LINEBUFFER_3D_H_
