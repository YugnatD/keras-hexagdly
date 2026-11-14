#ifndef NNET_HEX_CONV_LINEBUFFER_H_
#define NNET_HEX_CONV_LINEBUFFER_H_

#include "nnet_common.h"
#include "nnet_helpers.h"
#include "nnet_mult.h"

// Which output pixel, if any, the traversal emits at the iteration whose read head is
// rr = r - 1 - dr_max rows and ce = c - dc_max columns past it: (rr / S, ce / S) when both
// are multiples of the stride and inside the output grid. Shared by the conv and pool
// stream kernels.
//
// With a stride that is a power of two the division and the modulo are a shift and a mask.
// With any other stride (3...) Vitis built real dividers in the traversal loop: measured at
// stride 3 on a 9x9x2 window, a pipeline 39 stages deep with 4 DSP and 4972 FF; decoding
// rr == o*S against every output row o (compile-time constants) instead gives 5 stages,
// 0 DSP and 1020 FF, for the same values.
#ifndef NNET_HEX_LB_EMIT_INDEX_
#define NNET_HEX_LB_EMIT_INDEX_
namespace nnet {
template <typename CONFIG_T> bool hex_lb_emit_index(int rr, int ce, int &oh, int &ow) {
    #pragma HLS INLINE
    constexpr int S  = (int)CONFIG_T::stride;
    constexpr int OH = (int)CONFIG_T::out_height;
    constexpr int OW = (int)CONFIG_T::out_width;
    if ((S & (S - 1)) == 0) {
        oh = rr / S;
        ow = ce / S;
        return rr >= 0 && (rr % S) == 0 && oh < OH && ce >= 0 && (ce % S) == 0 && ow < OW;
    }
    bool row_hit = false, col_hit = false;
    oh = 0;
    ow = 0;
    for (int o = 0; o < OH; o++) {
        #pragma HLS UNROLL
        if (rr == o * S) {
            row_hit = true;
            oh = o;
        }
    }
    for (int o = 0; o < OW; o++) {
        #pragma HLS UNROLL
        if (ce == o * S) {
            col_hit = true;
            ow = o;
        }
    }
    return row_hit && col_hit;
}

// Bits needed to count from 0 to n - 1 (at least 1).
constexpr int hex_lb_bits(unsigned n) { return n <= 2 ? 1 : 1 + hex_lb_bits((n + 1) / 2); }
} // namespace nnet
#endif

// Fused hexagonal gather + ring-MAC for the keras-hexagdly "linebuffer" export
// strategy — io_parallel variant (and the shared per-pixel MAC body reused by
// the io_stream line-buffer kernel in nnet_hex_conv_linebuffer_stream.h).
//
// In io_parallel there is no memory win over the two-kernel gather+MAC path
// (hls4ml conv im2col does the same frame-buffered gather), so this variant is
// just gather+MAC fused into one function: for each output pixel it gathers its
// K neighbor taps from the full input array via the compile-time indices[] table
// (-1 = border -> zero) and applies the ring-shared MAC. Resource cost matches
// strategy="gather". The export handler prints a warning and selects this
// variant when the model is io_parallel; the real line-buffer benefit only
// exists in io_stream.
//
// Corresponds to the HexConvLineBuffer Keras layer (hex_gather.py).

namespace nnet {

struct hex_conv_linebuffer_config {
    static const unsigned in_height  = 8;  // H_in
    static const unsigned in_width   = 8;  // W_in
    static const unsigned out_height = 8;  // H_out (N_out = out_height*out_width)
    static const unsigned out_width  = 8;  // W_out
    static const unsigned stride     = 1;  // hex stride: input base = stride*output pos
    // io_stream only: number of (H, W) frames streamed back to back through the
    // kernel. 1 for a 2D layer. A 3D layer is a depth stage (nnet_hex_depth_stream.h)
    // followed by this kernel run over its n_frames output frames: the line buffer
    // and the window are simply traversed n_frames times, each frame with its own
    // zero border (virtual rows and columns flush the previous frame out).
    static const unsigned n_frames   = 1;
    static const int      dr_max     = 1;  // max tap row offset (emit-timing threshold)
    static const unsigned n_chan    = 1;   // Cin
    static const unsigned n_filt    = 4;   // Cout
    static const unsigned k         = 7;   // neighbor slots K
    static const unsigned num_weight_rows = 7;  // K (full) or num_rings (shared)
    static const unsigned rows_resident   = 2;  // dr_max - dr_min (== 2*kernel_size)
    // Column extent of the tap footprint, max(dc)-min(dc)+1 over both parities.
    // Used by the io_stream kernel to bank the row buffer's column axis so the
    // K taps of one output pixel never collide on the same memory port.
    static const unsigned col_span        = 3;
    // Max tap column offset. With col_span this pins the tap footprint's column
    // extent [dc_min, dc_max]; the io_stream kernel needs dc_max on its own to
    // place the emit column lag (see that file).
    static const int      dc_max          = 1;

    // Parallelization (parallel variant only): n_pixels output pixels per
    // partition-loop iteration; partition loop pipelines at II=reuse_factor.
    static const unsigned n_pixels     = 1;
    static const unsigned n_partitions = 64;
    static const unsigned reuse_factor = 1;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(n_pixels * num_weight_rows * n_chan * n_filt, reuse_factor);

    // Compute strategy (nnet::latency | nnet::resource), mirroring hls4ml's own
    // conv_2d_buffer_cl. In io_stream it decides whether the per-pixel loop is
    // pipelined (latency: one pixel per reuse_factor cycles) or left unpipelined
    // with a single time-shared emit instance whose MAC is reuse-folded
    // (resource). Ignored by the io_parallel kernel below.
    static const unsigned strategy = nnet::latency;

    // io_stream only: run the line buffer and the MAC as two concurrent dataflow
    // processes (hex_conv_linebuffer_stream_dataflow) instead of one fused loop.
    // Set per layer by the handler from the hls4ml config key "Dataflow".
    static const bool dataflow = false;

    // Slot -> weight row map (ring index when sharing, identity otherwise).
    static const unsigned row_map[k];
    // Flat neighbor index table (N_out*K), -1 = border -> zero (parallel gather).
    static const int indices[out_height * out_width * k];
    // Per-parity tap offsets (dr, dc) relative to the output pixel (stream
    // line-buffer variant); off_even for even output columns, off_odd for odd.
    static const int off_even[k][2];
    static const int off_odd[k][2];

    typedef float weight_t;
    typedef float bias_t;
    typedef float accum_t;
    // Type of a tap and of a ring sum of taps, as they enter the product: the
    // input type widened by ceil(log2(largest ring)) integer bits (the handler
    // emits it). It is NOT accum_t: the accumulator is sized for the sum of the
    // products w*x, which says nothing about x. With small weights it can be
    // narrower than the data, and a tap cast into it wrapped before it was
    // multiplied -- a wrong result, where a wrap of the accumulation itself
    // cancels out (modular arithmetic) as long as the final sum fits.
    typedef float ringsum_t;

    // Product kernel, selected by the handler from the data/weight precisions
    // exactly as hls4ml does for its own layers (nnet::product::mult for plain
    // fixed-point, but weight_binary / weight_ternary / weight_exponential when
    // the weights are quantized -- for exponential weights a plain `a * w` would
    // be numerically WRONG, since that product is a shift, not a multiply).
    // Ring sharing stays valid for all of them: every supported product is
    // distributive, so w*(x_a + x_b) == w*x_a + w*x_b still holds.
    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};

// -----------------------------------------------------------------------------
// Shared per-output-pixel ring-MAC body (adds-then-scale ring sharing).
//
// Arithmetic: fold the K taps into
// num_weight_rows ring sums (row_map is compile-time), one multiply per
// (ring, Cin, Cout), balanced accum into acc seeded with bias. w*(x_a+x_b) ==
// w*x_a + w*x_b holds exactly because the ring sum forms without loss in
// ringsum_t (the input type plus the bits of the sum) and the product is taken
// at full precision; only the product is cast into accum_t.
// -----------------------------------------------------------------------------
template <typename CONFIG_T, class dval, class w_T, class b_T, class rval>
void hex_lb_pixel_mac(
    const dval taps[CONFIG_T::k * CONFIG_T::n_chan],
    const w_T  weights[CONFIG_T::num_weight_rows * CONFIG_T::n_chan * CONFIG_T::n_filt],
    const b_T  bias[CONFIG_T::n_filt],
    rval       res_out[CONFIG_T::n_filt]
) {
    #pragma HLS INLINE
    typedef typename CONFIG_T::accum_t acc_T;
    typedef typename CONFIG_T::ringsum_t ring_T;  // taps and ring sums: see ringsum_t
    constexpr unsigned n_rows = CONFIG_T::num_weight_rows;
    constexpr unsigned Cin    = CONFIG_T::n_chan;
    constexpr unsigned Cout   = CONFIG_T::n_filt;

    ring_T ring_sum[n_rows * Cin];
    #pragma HLS ARRAY_PARTITION variable=ring_sum complete
RingReset:
    for (unsigned rc = 0; rc < n_rows * Cin; rc++) {
        #pragma HLS UNROLL
        ring_sum[rc] = (ring_T)0;
    }
RingSumK:
    for (unsigned ki = 0; ki < CONFIG_T::k; ki++) {
        #pragma HLS UNROLL
        const unsigned row = CONFIG_T::row_map[ki];
        for (unsigned c = 0; c < Cin; c++) {
            #pragma HLS UNROLL
            ring_sum[row * Cin + c] += (ring_T)taps[ki * Cin + c];
        }
    }

    acc_T acc[Cout];
    #pragma HLS ARRAY_PARTITION variable=acc complete
    for (unsigned o = 0; o < Cout; o++) {
        #pragma HLS UNROLL
        acc[o] = (acc_T)bias[o];
    }
Product:
    for (unsigned row = 0; row < n_rows; row++) {
        #pragma HLS UNROLL
        for (unsigned c = 0; c < Cin; c++) {
            #pragma HLS UNROLL
            const ring_T s = ring_sum[row * Cin + c];
            for (unsigned o = 0; o < Cout; o++) {
                #pragma HLS UNROLL
                acc[o] += (acc_T)CONFIG_T::template product<ring_T, w_T>::product(
                    s, weights[(row * Cin + c) * Cout + o]);
            }
        }
    }
    for (unsigned o = 0; o < Cout; o++) {
        #pragma HLS UNROLL
        res_out[o] = (rval)acc[o];
    }
}

// -----------------------------------------------------------------------------
// Same per-output-pixel result as hex_lb_pixel_mac above, but the product stage
// is folded over reuse_factor cycles instead of being unrolled all at once.
//
// Why this exists (io_stream only): hex_lb_pixel_mac unrolls ALL
// num_weight_rows*n_chan*n_filt products unconditionally, so reuse_factor only
// changed the caller's pipeline II and never reduced the amount of logic the
// scheduler had to place. Past a few thousand unrolled products Vitis's
// scheduler stalls for hours (measured: a 19*16*32 = 9728-product layer never
// finished scheduling). Here the products are split across reuse_factor
// iterations, so only ceil(n_in/RF)*n_filt of them exist in hardware.
//
// Note we deliberately do NOT call nnet::dense_resource here even though the
// shape is a plain matmul: hls4ml's two dense kernels disagree on weight
// layout -- dense_latency indexes weights[in*n_out + out] while
// dense_resource indexes weights[out*n_in + in] (verified directly: same
// weights, same input, different results). Our weights are emitted in the
// former layout, so delegating would have silently transposed the kernel in
// resource mode. Owning this loop keeps the layout assumption in one place.
//
// Bit-exactness vs hex_lb_pixel_mac: identical products, identical bias
// seeding, identical output cast. Only the accumulation ORDER differs, which is
// immaterial in fixed point given the headroom accum_t is already sized for.
// -----------------------------------------------------------------------------
template <typename CONFIG_T, class dval, class w_T, class b_T, class rval>
void hex_lb_pixel_mac_reuse(
    const dval taps[CONFIG_T::k * CONFIG_T::n_chan],
    const w_T  weights[CONFIG_T::num_weight_rows * CONFIG_T::n_chan * CONFIG_T::n_filt],
    const b_T  bias[CONFIG_T::n_filt],
    rval       res_out[CONFIG_T::n_filt]
) {
    typedef typename CONFIG_T::accum_t acc_T;
    typedef typename CONFIG_T::ringsum_t ring_T;  // taps and ring sums: see ringsum_t
    constexpr unsigned n_rows = CONFIG_T::num_weight_rows;
    constexpr unsigned Cin    = CONFIG_T::n_chan;
    constexpr unsigned Cout   = CONFIG_T::n_filt;
    constexpr unsigned N_IN   = n_rows * Cin;              // ring sums feeding the matmul
    constexpr unsigned RF     = CONFIG_T::reuse_factor;
    constexpr unsigned IN_PER = DIV_ROUNDUP(N_IN, RF);     // ring sums handled per cycle

    // Ring fold: K taps -> num_weight_rows ring sums (identity when weights are
    // not shared, in which case num_weight_rows == k and this is just a rename).
    ring_T ring_sum[N_IN];
    #pragma HLS ARRAY_PARTITION variable=ring_sum complete
RingReset:
    for (unsigned rc = 0; rc < N_IN; rc++) {
        #pragma HLS UNROLL
        ring_sum[rc] = (ring_T)0;
    }
RingSumK:
    for (unsigned ki = 0; ki < CONFIG_T::k; ki++) {
        #pragma HLS UNROLL
        const unsigned row = CONFIG_T::row_map[ki];
        for (unsigned c = 0; c < Cin; c++) {
            #pragma HLS UNROLL
            ring_sum[row * Cin + c] += (ring_T)taps[ki * Cin + c];
        }
    }

    acc_T acc[Cout];
    #pragma HLS ARRAY_PARTITION variable=acc complete
InitAccum:
    for (unsigned o = 0; o < Cout; o++) {
        #pragma HLS UNROLL
        acc[o] = (acc_T)bias[o];
    }

    // Iteration ir consumes ring sums [ir*IN_PER, (ir+1)*IN_PER) against the
    // CONTIGUOUS slice [ir*MAC_BLOCK, (ir+1)*MAC_BLOCK) of the weight array
    // (MAC_BLOCK = IN_PER*Cout), i.e. MAC_BLOCK parallel scalar reads. The
    // caller leaves `weights` unconstrained and lets Vitis split it into small
    // distributed ROMs, which costs 0 BRAM as long as MAC_BLOCK stays modest --
    // see the measured sweep in nnet_hex_conv_linebuffer_stream.h before adding
    // any ARRAY_RESHAPE here. If this loop is ever reordered to a strided index
    // (nnet_dense_resource's ir + j*RF), that analysis has to be redone: the
    // access pattern is what decides which reshape mode, if any, applies.
ReuseLoop:
    for (unsigned ir = 0; ir < RF; ir++) {
        // rewind: the loop restarts on the next output pixel without draining,
        // which Vitis confirms it applies here ("Rewinding loop 'ReuseLoop'").
        // Measured on a 19x16x32 layer at RF=64: Fmax 75 -> 130 MHz and wall
        // clock 254 -> 145 us, for +6% FF and +5% LUT of the device. Note this
        // is the same mechanism behind the HLS 200-656 "auto-rewind pipeline"
        // dataflow warning documented in the stream kernel; it is verified by
        // cosim there, but treat it as suspect if a deadlock ever reappears.
        #pragma HLS PIPELINE II=1 rewind
        // The products of this iteration are summed on their own, then added to
        // acc once: the loop-carried path is a single adder. Adding each product
        // straight into acc left, when RF does not divide N_IN, a chain of adders
        // and selects (the in_idx guard) on that path, which missed timing.
        acc_T part[Cout];
        #pragma HLS ARRAY_PARTITION variable=part complete
        for (unsigned o = 0; o < Cout; o++) {
            #pragma HLS UNROLL
            part[o] = (acc_T)0;
        }
    MultLoop:
        for (unsigned j = 0; j < IN_PER; j++) {
            #pragma HLS UNROLL
            const unsigned in_idx = ir * IN_PER + j;
            if (in_idx < N_IN) {  // only bites when RF does not divide N_IN
                const ring_T s = ring_sum[in_idx];
                for (unsigned o = 0; o < Cout; o++) {
                    #pragma HLS UNROLL
                    part[o] += (acc_T)CONFIG_T::template product<ring_T, w_T>::product(
                        s, weights[in_idx * Cout + o]);
                }
            }
        }
        for (unsigned o = 0; o < Cout; o++) {
            #pragma HLS UNROLL
            acc[o] += part[o];
        }
    }

    for (unsigned o = 0; o < Cout; o++) {
        #pragma HLS UNROLL
        res_out[o] = (rval)acc[o];
    }
}

// -----------------------------------------------------------------------------
// io_parallel fused gather + MAC (fallback; same cost as strategy="gather").
// -----------------------------------------------------------------------------
template <class data_T, class w_T, class b_T, class res_T, typename CONFIG_T>
void hex_conv_linebuffer(
    data_T input  [CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    w_T    weights[CONFIG_T::num_weight_rows * CONFIG_T::n_chan * CONFIG_T::n_filt],
    b_T    bias   [CONFIG_T::n_filt],
    res_T  output [CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt]
) {
    constexpr unsigned N_OUT = CONFIG_T::out_height * CONFIG_T::out_width;
    constexpr unsigned Cin   = CONFIG_T::n_chan;
    constexpr unsigned Cout  = CONFIG_T::n_filt;
    static_assert(N_OUT == CONFIG_T::n_pixels * CONFIG_T::n_partitions,
                  "hex_conv_linebuffer: n_pixels must divide N_out");

    #pragma HLS FUNCTION_INSTANTIATE variable=weights,bias
    #pragma HLS ARRAY_PARTITION variable=bias complete
    if (CONFIG_T::reuse_factor == 1) {
        #pragma HLS ARRAY_PARTITION variable=weights complete
    } else {
        #pragma HLS ARRAY_RESHAPE variable=weights block factor=CONFIG_T::reuse_factor
        #pragma HLS RESOURCE      variable=weights core=ROM_nP_BRAM
    }
    #pragma HLS ALLOCATION operation instances=mul limit=CONFIG_T::multiplier_limit

PartitionLoop:
    for (unsigned i_part = 0; i_part < CONFIG_T::n_partitions; i_part++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor rewind
    PixelLoop:
        for (unsigned i_pxl = 0; i_pxl < CONFIG_T::n_pixels; i_pxl++) {
            #pragma HLS UNROLL
            const unsigned n = i_part * CONFIG_T::n_pixels + i_pxl;

            data_T taps[CONFIG_T::k * Cin];
            #pragma HLS ARRAY_PARTITION variable=taps complete
        Gather:
            for (unsigned ki = 0; ki < CONFIG_T::k; ki++) {
                #pragma HLS UNROLL
                const int idx = CONFIG_T::indices[n * CONFIG_T::k + ki];
                for (unsigned c = 0; c < Cin; c++) {
                    #pragma HLS UNROLL
                    taps[ki * Cin + c] = (idx >= 0)
                        ? input[(unsigned)idx * Cin + c]
                        : (data_T)0;
                }
            }

            res_T res_out[Cout];
            #pragma HLS ARRAY_PARTITION variable=res_out complete
            hex_lb_pixel_mac<CONFIG_T, data_T, w_T, b_T, res_T>(
                taps, weights, bias, res_out);
        Write:
            for (unsigned o = 0; o < Cout; o++) {
                #pragma HLS UNROLL
                output[n * Cout + o] = res_out[o];
            }
        }
    }
}

} // namespace nnet

#endif // NNET_HEX_CONV_LINEBUFFER_H_
