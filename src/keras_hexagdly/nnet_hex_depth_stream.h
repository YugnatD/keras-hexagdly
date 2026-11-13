#ifndef NNET_HEX_DEPTH_STREAM_H_
#define NNET_HEX_DEPTH_STREAM_H_

#include "ap_int.h"
#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_hex_conv_linebuffer.h" // hex_lb_bits
#include "nnet_types.h"

// Depth stage of the 3D hexagonal layers (io_stream).
//
// A 3D hex layer is exported as TWO kernels in a row:
//
//   Conv3d    = hex_depth_stack_stream  ->  hex_conv_linebuffer_stream (n_frames = D_out)
//   MaxPool3d = hex_depth_pool_stream   ->  hex_pool_linebuffer_stream (n_frames = D_out)
//
// The depth stage only looks along the depth axis, pixel by pixel:
//
//   stack: out[t][p] = { x[t*ds - pad + d][p] : d = 0 .. depth_size-1 }   (channels side by side)
//   pool : out[t][p] = max_d x[t*ds + d][p]
//
// and the 2D kernel then does the hexagonal part on each output frame, over
// depth_size*n_chan channels for a convolution: a 3D convolution over (depth, hex
// neighbourhood) IS a 2D hex convolution over the depth_size frames stacked as
// channels, with the 3D weights laid out as (k, depth_size*n_chan, n_filt). For
// the pooling the max over depth and the max over the hex neighbourhood commute.
//
// Why split it this way. The frames a 3D layer combines are H*W tokens apart in
// the stream, so (depth_size-1) whole frames must be stored whatever the design:
// that part is a plain delay line, written and read once per token, and it is
// all this stage is -- (depth_size-1) memories of one frame each (the channels of
// a pixel packed in one word), addressed by the pixel counter. Everything that is hard about the hexagonal layers (parity
// offsets, borders, strides, reuse factor, dataflow, weight sharing, accumulator
// types) stays in the 2D kernels, which are the ones validated by RTL
// co-simulation, and is inherited unchanged: hex stride, depth stride and depth
// padding all come for free.
//
// Stream contract: one pixel per token, frames in raster order (depth, row, col).
//   input : n_depth_in  * n_pix tokens of n_chan values
//   output: n_depth_out * n_pix tokens of depth_size*n_chan values (stack) or
//           n_chan values (pool)
// The loop runs over F_TOTAL >= n_depth_in frames: the real ones (one read per
// pixel) and, with depth padding at the end, virtual frames that read nothing
// and count as zeros.

namespace nnet {

struct hex_depth_config {
    static const unsigned n_depth_in   = 4;
    static const unsigned n_depth_out  = 3;
    static const unsigned depth_size   = 2;
    static const unsigned depth_stride = 1;
    static const unsigned pad_front    = 0;  // zero frames before the first one (depth_padding='same')
    static const unsigned n_pix        = 64; // pixels per frame (in_height * in_width)
    static const unsigned n_chan       = 1;
};

// Frame index (in the F_TOTAL traversal) at which output frame t is complete, and the
// traversal length. Output frame t uses input frames t*ds - pad .. t*ds - pad + depth_size-1.
template <typename CONFIG_T> struct hex_depth_geometry {
    static const int DK      = (int)CONFIG_T::depth_size;
    static const int NEWEST0 = DK - 1 - (int)CONFIG_T::pad_front;  // newest frame of output 0
    static const int LAST    = ((int)CONFIG_T::n_depth_out - 1) * (int)CONFIG_T::depth_stride + NEWEST0;
    static const int F_TOTAL = (LAST + 1 > (int)CONFIG_T::n_depth_in) ? LAST + 1 : (int)CONFIG_T::n_depth_in;
};

// True when frame `fi` of the traversal completes an output frame. Unrolled
// comparators against the n_depth_out constants (no division by the stride).
template <typename CONFIG_T> inline bool hex_depth_emits(int fi) {
    #pragma HLS INLINE
    bool hit = false;
    for (int t = 0; t < (int)CONFIG_T::n_depth_out; t++) {
        #pragma HLS UNROLL
        if (fi == t * (int)CONFIG_T::depth_stride + hex_depth_geometry<CONFIG_T>::NEWEST0) hit = true;
    }
    return hit;
}

// One traversal step shared by the two kernels: read (or zero) the pixel, push it
// down the frame delay chain, and return in s[i] the same pixel of frame fi - i
// (zero for a frame outside the volume).
template <class data_T, typename CONFIG_T>
void hex_depth_step(
    int fi, int p, hls::stream<data_T> &input,
    typename data_T::value_type
        buf[(CONFIG_T::depth_size > 1 ? CONFIG_T::depth_size - 1 : 1)][CONFIG_T::n_chan][CONFIG_T::n_pix],
    typename data_T::value_type s[CONFIG_T::depth_size][CONFIG_T::n_chan]
) {
    #pragma HLS INLINE
    constexpr int DK = (int)CONFIG_T::depth_size;
    constexpr unsigned C = CONFIG_T::n_chan;
    typedef typename data_T::value_type dval;

    const bool real = fi < (int)CONFIG_T::n_depth_in;
    data_T cur;
    if (real) {
        cur = input.read();
    }
    for (unsigned c = 0; c < C; c++) {
        #pragma HLS UNROLL
        dval carry = real ? (dval)cur[c] : (dval)0;
        s[0][c] = carry;
        for (int i = 1; i < DK; i++) {
            #pragma HLS UNROLL
            // Read-then-write of the same address: a one-frame delay.
            const dval old = buf[i - 1][c][p];
            buf[i - 1][c][p] = carry;
            // Frame fi - i does not exist before the first one. The memory then holds
            // whatever the previous volume left there, so it is masked, not trusted.
            s[i][c] = (fi >= i) ? old : (dval)0;
            carry = old;
        }
    }
}

#define HEX_DEPTH_TRAVERSAL_STATE                                                        \
    constexpr int      DK      = (int)CONFIG_T::depth_size;                              \
    constexpr unsigned C       = CONFIG_T::n_chan;                                       \
    constexpr int      N       = (int)CONFIG_T::n_pix;                                   \
    constexpr int      F_TOTAL = hex_depth_geometry<CONFIG_T>::F_TOTAL;                  \
    typedef typename data_T::value_type dval;                                            \
    static_assert(data_T::size == CONFIG_T::n_chan,                                      \
                  "hex_depth_*_stream: expects one pixel per input token");              \
    /* plain values, not nnet::array tokens (Vitis cannot partition an array of     */  \
    /* those); the pragmas after this macro make it one memory per delay stage      */  \
    dval buf[(DK > 1 ? DK - 1 : 1)][CONFIG_T::n_chan][CONFIG_T::n_pix];                  \
    constexpr int F_BITS = hex_lb_bits((unsigned)F_TOTAL);                               \
    constexpr int P_BITS = hex_lb_bits((unsigned)N);                                     \
    ap_uint<F_BITS> f_count = 0;                                                         \
    ap_uint<P_BITS> p_count = 0;

#define HEX_DEPTH_TRAVERSAL_NEXT                                                         \
    if (p_count == N - 1) {                                                              \
        p_count = 0;                                                                     \
        f_count = f_count + 1;                                                           \
    } else {                                                                             \
        p_count = p_count + 1;                                                           \
    }

// Conv3d depth stage: the depth_size frames of each output frame, stacked as channels.
// Output channel d*n_chan + c is channel c of input frame t*ds - pad + d.
template <class data_T, class res_T, typename CONFIG_T>
void hex_depth_stack_stream(hls::stream<data_T> &input, hls::stream<res_T> &output) {
    HEX_DEPTH_TRAVERSAL_STATE
    static_assert(res_T::size == CONFIG_T::depth_size * CONFIG_T::n_chan,
                  "hex_depth_stack_stream: expects depth_size * n_chan values per output token");
    #pragma HLS ARRAY_PARTITION variable=buf complete dim=1
    #pragma HLS ARRAY_RESHAPE variable=buf complete dim=2

Traverse:
    for (int it = 0; it < F_TOTAL * N; it++) {
        #pragma HLS PIPELINE II=1
        const int fi = (int)f_count;
        const int p  = (int)p_count;
        dval s[CONFIG_T::depth_size][CONFIG_T::n_chan];
        #pragma HLS ARRAY_PARTITION variable=s complete dim=0
        hex_depth_step<data_T, CONFIG_T>(fi, p, input, buf, s);
        if (hex_depth_emits<CONFIG_T>(fi)) {
            res_T tok;
            PRAGMA_DATA_PACK(tok)
            for (int d = 0; d < DK; d++) {
                #pragma HLS UNROLL
                for (unsigned c = 0; c < C; c++) {
                    #pragma HLS UNROLL
                    // depth tap d is frame fi - (DK-1-d): the oldest frame comes first
                    tok[d * C + c] = s[DK - 1 - d][c];
                }
            }
            output.write(tok);
        }
        HEX_DEPTH_TRAVERSAL_NEXT
    }
}

// MaxPool3d depth stage: the max over the depth window, per pixel and channel.
// Depth pooling has no padding, so every frame of a window exists.
template <class data_T, class res_T, typename CONFIG_T>
void hex_depth_pool_stream(hls::stream<data_T> &input, hls::stream<res_T> &output) {
    HEX_DEPTH_TRAVERSAL_STATE
    static_assert(res_T::size == CONFIG_T::n_chan,
                  "hex_depth_pool_stream: expects n_chan values per output token");
    static_assert(CONFIG_T::pad_front == 0, "hex_depth_pool_stream: depth pooling is not padded");
    #pragma HLS ARRAY_PARTITION variable=buf complete dim=1
    #pragma HLS ARRAY_RESHAPE variable=buf complete dim=2

Traverse:
    for (int it = 0; it < F_TOTAL * N; it++) {
        #pragma HLS PIPELINE II=1
        const int fi = (int)f_count;
        const int p  = (int)p_count;
        dval s[CONFIG_T::depth_size][CONFIG_T::n_chan];
        #pragma HLS ARRAY_PARTITION variable=s complete dim=0
        hex_depth_step<data_T, CONFIG_T>(fi, p, input, buf, s);
        if (hex_depth_emits<CONFIG_T>(fi)) {
            res_T tok;
            PRAGMA_DATA_PACK(tok)
            for (unsigned c = 0; c < C; c++) {
                #pragma HLS UNROLL
                dval m = s[0][c];
                for (int d = 1; d < DK; d++) {
                    #pragma HLS UNROLL
                    if (s[d][c] > m) m = s[d][c];
                }
                tok[c] = m;
            }
            output.write(tok);
        }
        HEX_DEPTH_TRAVERSAL_NEXT
    }
}

#undef HEX_DEPTH_TRAVERSAL_STATE
#undef HEX_DEPTH_TRAVERSAL_NEXT

} // namespace nnet

#endif // NNET_HEX_DEPTH_STREAM_H_
