"""hls4ml handler for the line-buffer hex layers.

Call ``register_hex_gather_layers()`` once before converting a model that uses
strategy="linebuffer".  This registers:
  - KerasV3 layer handlers  (convert the HexConvLineBuffer / HexPoolLineBuffer
    Keras layers into hls4ml IR nodes)
  - hls4ml IR layer classes (HHexConvLineBuffer and friends)
  - HLS config + function templates for the Vivado backend
  - The nnet_hex_conv_linebuffer*.h / nnet_hex_pool_linebuffer*.h sources

Phase 4 (the HLS C++ kernels) must be present before synthesis.  The handler
wires them in via backend.register_source(); they are loaded from the same
directory as this file.

Usage::

    from keras_hexagdly.hls4ml_handler import register_hex_gather_layers
    register_hex_gather_layers()

    cfg = hls4ml.utils.config_from_keras_model(patched, ...)
    hls_model = hls4ml.converters.convert_from_keras_model(patched, ...)
"""

from pathlib import Path

import hls4ml
import numpy as np
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.converters.keras_v3._base import KerasV3LayerHandler
from hls4ml.model.attributes import Attribute, TypeAttribute, WeightAttribute
from hls4ml.model.layers import Layer


def _choose_parallelization(node, n_out):
    """Resolve ParallelizationFactor (n_pixels) for a hex MAC node.

    Mirrors hls4ml's Conv1D/2D handling: reads the per-layer (or model)
    'ParallelizationFactor' config value (default 1 = one output pixel per
    partition-loop iteration) and clamps it to the closest divisor of n_out
    so that n_partitions = n_out / n_pixels is exact.

    Pipeline-style interaction (same physics as hls4ml conv): when the model
    resolves to 'pipeline' style, myproject() carries a top-level
    `#pragma HLS PIPELINE`, which force-unrolls every loop in every layer —
    including the MAC partition loop.  A rolled partition (PF < n_out) cannot
    exist in that regime, and a per-partition multiplier cap would strangle
    the scheduler (II explosion).  So under 'pipeline' style we emit
    n_pixels = n_out (one partition, fully unrolled — the multiplier cap
    then spans the whole layer, matching hls4ml's Latency-strategy conv).
    A rolled partition loop requires dataflow style: set
    cfg['Model']['Strategy'] = 'Resource' or cfg['Model']['PipelineStyle'] =
    'dataflow' — exactly as with stock hls4ml conv layers.
    """
    style = getattr(node.model.config, "pipeline_style", None)
    pf = node.model.config.get_layer_config_value(node, "ParallelizationFactor", 1)
    try:
        pf = int(pf)
    except (TypeError, ValueError):
        pf = 1

    if style == "pipeline":
        if pf not in (1, int(n_out)):
            print(
                f'WARNING: ParallelizationFactor={pf} in layer "{node.name}" is ignored: '
                "the model resolved to pipeline style (top-level PIPELINE unrolls the "
                "partition loop). Set Strategy=Resource or PipelineStyle=dataflow to "
                "use a rolled partition loop."
            )
        return int(n_out)

    pf = max(1, min(pf, int(n_out)))
    if n_out % pf != 0:
        valid = [d for d in range(1, n_out + 1) if n_out % d == 0]
        closest = min(valid, key=lambda d: (abs(d - pf), d))
        print(
            f'WARNING: Invalid ParallelizationFactor={pf} in layer "{node.name}". '
            f"Using ParallelizationFactor={closest} instead (must divide n_out={n_out})."
        )
        pf = closest
    return pf


def _int_csv(values):
    """Render an int array as a C initializer list body."""
    return ", ".join(str(int(v)) for v in np.asarray(values).reshape(-1))


def _int_pairs_csv(values):
    """Render an (N, 2) int array as a C initializer body of {a, b} pairs."""
    arr = np.asarray(values).reshape(-1, 2)
    return ", ".join(f"{{{int(a)}, {int(b)}}}" for a, b in arr)


def _col_span(off_even, off_odd):
    """Column extent of the tap footprint: max(dc) - min(dc) + 1 over both parities.

    The io_stream line-buffer kernel banks the row buffer's column axis by this
    value so that the K taps of one output pixel always land in distinct banks
    (two taps sharing a row differ in dc by less than the span, hence differ mod
    the span as well). Derived from the measured offset tables rather than from
    kernel_size, so it stays correct for any geometry those tables encode.
    """
    dcs = [int(dc) for _, dc in np.asarray(off_even).reshape(-1, 2)]
    dcs += [int(dc) for _, dc in np.asarray(off_odd).reshape(-1, 2)]
    return max(dcs) - min(dcs) + 1


def _warn_reuse_factor(node, n_in, n_out):
    """Warn when the ReuseFactor cannot be honoured exactly.

    The io_stream MAC folds n_in ring sums over reuse_factor cycles, handling
    ceil(n_in / RF) of them per cycle. Unlike hls4ml's dense kernels -- which
    assert on RF values they cannot express -- we accept any RF and simply round
    up, so nothing breaks; but a factor that does not divide n_in leaves idle
    multiplier slots and silently costs area for no throughput gain. hls4ml
    surfaces the same class of problem through get_valid_reuse_factors(); we
    only warn, since the rounding is safe.
    """
    if n_in <= 0:
        return
    rf = int(node.model.config.get_reuse_factor(node) or 1)
    if rf <= 1 or n_in % rf == 0:
        return
    in_per = -(-n_in // rf)  # ceil
    used, slots = n_in, in_per * rf
    print(
        f'WARNING: layer "{node.name}": ReuseFactor={rf} does not divide the '
        f"{n_in} ring sums; the MAC will process {in_per} per cycle, leaving "
        f"{slots - used} of {slots} multiplier slots idle "
        f"({n_out * (slots - used)} wasted products). Pick a divisor of {n_in} "
        f"for an exact fold."
    )


def _product_type(node, weight_name="mac_weights"):
    """Pick the nnet::product:: kernel for this layer's data/weight precisions.

    Defers to the backend's own product_type() so hex layers follow exactly the
    same rule as hls4ml's built-in layers: plain 'mult' for fixed-point, but
    weight_binary / weight_ternary / weight_exponential when the weights are
    quantized. That last one matters for correctness, not just speed -- an
    exponential ("power of 2") weight product is a shift, so computing it as a
    plain a*w would be numerically wrong.

    Ring sharing remains valid for every one of these: they are all distributive
    in the data operand, so folding taps into ring sums before the product does
    not change the result.
    """
    try:
        from hls4ml.backends import get_backend

        data_prec = node.get_input_variable().type.precision
        weight_prec = node.get_weights(weight_name).type.precision
        return get_backend("Vivado").product_type(data_prec, weight_prec)
    except Exception:
        # Never fail codegen over this: 'mult' is what every non-quantized model
        # resolves to anyway, and it is what the kernels did unconditionally
        # before this was wired up.
        return "mult"


def _dc_max(off_even, off_odd):
    """Largest tap column offset over both parities.

    The io_stream kernel emits an output pixel dc_max input columns after the
    read head passes it, so that every tap column it needs has already been
    shifted into the register window (the hex analogue of hls4ml's lShiftX).
    """
    dcs = [int(dc) for _, dc in np.asarray(off_even).reshape(-1, 2)]
    dcs += [int(dc) for _, dc in np.asarray(off_odd).reshape(-1, 2)]
    return max(dcs)


def _strategy_enum(node):
    """Map hls4ml's Strategy attribute onto the nnet:: enum our kernels branch on.

    Only latency and resource are emitted: those are the two code paths the hex
    kernels implement. resource_unrolled and distributed_arithmetic (which needs
    da4ml) have no hex equivalent, so they fall back to the resource path, which
    is always functionally correct -- just not their intended optimisation.

    Read through model.config.get_strategy(), NOT node.get_attr('strategy'):
    the latter is only populated by the backend's per-layer-type init_* hooks
    (init_dense, init_conv2d, ...), which never run for custom layers, so it is
    silently None here and every layer would look like 'latency' no matter what
    the user configured. get_strategy() resolves LayerName -> LayerType -> Model
    exactly as hls4ml does for its own layers.
    """
    strategy = None
    model = getattr(node, "model", None)
    if model is not None:
        strategy = model.config.get_strategy(node)
    if strategy is None:
        strategy = node.get_attr("strategy")
    strategy = (strategy or "latency").lower()
    return "latency" if strategy == "latency" else "resource"


def _accum_type_str(node, n_terms):
    """Return an ap_fixed<> string for the ring-MAC accumulator.

    The accumulator sums ``n_terms`` products; to hold that sum without overflow
    it needs ceil(log2(n_terms)) extra integer bits over the weight type. We grow
    both the total width and the integer part by the same amount, matching the
    bit-growth idiom hls4ml uses for its dense/conv accumulators.

    Falls back to the plain weight-type name if the precision can't be read
    (e.g. a non-FixedPrecision type), which preserves the old behavior.
    """
    import math

    prec = node.get_weights("mac_weights").type.precision
    width = getattr(prec, "width", None)
    integer = getattr(prec, "integer", None)
    signed = getattr(prec, "signed", True)
    if width is None or integer is None:
        return node.get_weights("mac_weights").type.name
    scale = int(math.ceil(math.log2(max(1, n_terms))))
    u = "" if signed else "u"
    return f"ap_{u}fixed<{width + scale}, {integer + scale}>"


# ---------------------------------------------------------------------------
# hls4ml IR layers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# KerasV3 handlers
# ---------------------------------------------------------------------------


def keras_ops_to_numpy(var):
    """Convert a Keras variable / tensor to numpy safely."""
    import keras

    try:
        return keras.ops.convert_to_numpy(var)
    except Exception:
        return np.asarray(var)


def _is_stream(node):
    """Return True when the model is configured for io_stream."""
    return node.model.config.get_config_value("IOType") == "io_stream"


# ---------------------------------------------------------------------------
# Line-buffer strategy — fused HexConvLineBuffer (Conv2d, stride 1)
#
# One IR node fuses gather + ring-MAC and keeps the (H, W) grid.  Two HLS
# variants selected by IOType:
#   io_stream   -> nnet_hex_conv_linebuffer_stream.h  (real line buffer: only
#                  rows_resident+1 image rows resident; the memory win).
#   io_parallel -> nnet_hex_conv_linebuffer.h         (frame-buffered gather+MAC;
#                  a warning is printed because line buffering only reduces
#                  memory in io_stream).
# ---------------------------------------------------------------------------


class HHexConvLineBuffer(Layer):
    """hls4ml IR node for HexConvLineBuffer (fused hex conv, stride 1)."""

    _expected_attributes = [
        Attribute("in_height"),
        Attribute("in_width"),
        Attribute("out_height"),
        Attribute("out_width"),
        Attribute("stride"),
        Attribute("dr_max"),
        Attribute("n_chan"),
        Attribute("n_filt"),
        Attribute("k"),
        Attribute("num_weight_rows"),
        Attribute("rows_resident"),
        Attribute("share_neighbors", value_type=bool),
        WeightAttribute("mac_weights"),
        TypeAttribute("mac_weights"),
        WeightAttribute("mac_bias"),
        TypeAttribute("mac_bias"),
    ]

    def initialize(self):
        n_out = self.attributes["out_height"] * self.attributes["out_width"]
        n_filt = self.attributes["n_filt"]
        # 2D shape [n_out, n_filt] so io_stream packs n_filt per token (depth n_out).
        self.add_output_variable(shape=[n_out, n_filt])
        self.add_weights_variable(name="mac_weights", var_name="w{index}")
        self.add_weights_variable(name="mac_bias", var_name="b{index}")


class HexConvLineBufferHandler(KerasV3LayerHandler):
    handles = ("keras_hexagdly.hex_gather.HexConvLineBuffer",)

    def handle(self, layer, in_tensors, out_tensors):
        from keras_hexagdly.indexed import build_parity_offset_tables

        in_shape = list(in_tensors[0].shape[1:])  # (H_in, W_in, Cin)
        out_shape = list(out_tensors[0].shape[1:])  # (H_out, W_out, Cout)
        H = int(in_shape[0])
        W = int(in_shape[1])
        n_chan = int(in_shape[2])
        H_out = int(out_shape[0])
        W_out = int(out_shape[1])
        n_filt = int(out_shape[2])
        stride = int(layer.stride)

        nbr = np.asarray(keras_ops_to_numpy(layer.neighbor_idx), dtype=np.int32)
        k = int(nbr.shape[1])
        off_even, off_odd, rows_resident = build_parity_offset_tables(nbr, W, H_out, W_out, stride)
        dr_max = int(max(off_even[:, 0].max(), off_odd[:, 0].max()))

        w = keras_ops_to_numpy(layer.mac_weights).astype(np.float32).reshape(-1)
        num_weight_rows = int(layer.mac_weights.shape[0])
        if layer.share_neighbors:
            ring_idx = keras_ops_to_numpy(layer.ring_idx).astype(np.int32).reshape(-1)
        else:
            ring_idx = np.arange(k, dtype=np.int32)  # identity slot -> row
        bias = keras_ops_to_numpy(layer.mac_bias).astype(np.float32).reshape(-1)

        return {
            "class_name": "HHexConvLineBuffer",
            "in_height": H,
            "in_width": W,
            "out_height": H_out,
            "out_width": W_out,
            "stride": stride,
            "dr_max": dr_max,
            "n_chan": n_chan,
            "n_filt": n_filt,
            "k": k,
            "num_weight_rows": num_weight_rows,
            "rows_resident": int(rows_resident),
            "share_neighbors": bool(layer.share_neighbors),
            "indices_data": nbr.reshape(-1),
            "off_even_data": np.asarray(off_even, dtype=np.int32).reshape(-1),
            "off_odd_data": np.asarray(off_odd, dtype=np.int32).reshape(-1),
            "row_map_data": ring_idx,
            "mac_weights_data": w,
            "mac_bias_data": bias,
        }


hex_conv_linebuffer_config_template = """\
struct config{index} : nnet::hex_conv_linebuffer_config {{
    static const unsigned in_height  = {in_height};
    static const unsigned in_width   = {in_width};
    static const unsigned out_height = {out_height};
    static const unsigned out_width  = {out_width};
    static const unsigned stride     = {stride};
    static const int      dr_max     = {dr_max};
    static const unsigned n_chan    = {n_chan};
    static const unsigned n_filt    = {n_filt};
    static const unsigned k         = {k};
    static const unsigned num_weight_rows = {num_weight_rows};
    static const unsigned rows_resident   = {rows_resident};
    static const unsigned col_span     = {col_span};
    static const int      dc_max       = {dc_max};
    static const unsigned strategy     = nnet::{strategy};
    static const unsigned n_pixels     = {n_pixels};
    static const unsigned n_partitions = {n_partitions};
    static const unsigned reuse_factor = {reuse};
    // io_parallel multiplier budget: n_pixels output pixels are computed per
    // partition-loop iteration, hence the n_pixels factor. The io_stream kernel
    // emits exactly one pixel per iteration and uses mult_config below instead.
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(n_pixels * num_weight_rows * n_chan * n_filt, reuse_factor);
    static const unsigned row_map[k];
    static const int indices[out_height * out_width * k];
    static const int off_even[k][2];
    static const int off_odd[k][2];
    typedef {mac_weights_t.name} weight_t;
    typedef {mac_bias_t.name}    bias_t;
    typedef {accum_t_str} accum_t;
    // Product kernel chosen from the data/weight precisions, exactly as
    // hls4ml does for its own layers (see _product_type).
    template<class x_T, class y_T> using product = nnet::product::{product_type}<x_T, y_T>;
}};
const unsigned config{index}::row_map[] = {{{row_map_vals}}};
const int config{index}::indices[] = {{{indices_vals}}};
const int config{index}::off_even[][2] = {{{off_even_vals}}};
const int config{index}::off_odd[][2] = {{{off_odd_vals}}};\n"""

hex_conv_linebuffer_function_template = (
    "nnet::hex_conv_linebuffer<{input_t}, {mac_weights_t}, {mac_bias_t}, "
    "{output_t}, {config}>({input}, w{index}, b{index}, {output});"
)
hex_conv_linebuffer_stream_function_template = (
    "nnet::hex_conv_linebuffer_stream<{input_t}, {mac_weights_t}, {mac_bias_t}, "
    "{output_t}, {config}>({input}, w{index}, b{index}, {output});"
)

hex_conv_linebuffer_include_list = ["nnet_utils/nnet_hex_conv_linebuffer.h"]
hex_conv_linebuffer_stream_include_list = ["nnet_utils/nnet_hex_conv_linebuffer_stream.h"]


class HexConvLineBufferConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(HHexConvLineBuffer)
        self.template = hex_conv_linebuffer_config_template

    def format(self, node):
        params = self._default_config_params(node)  # provides params['reuse']
        for a in (
            "in_height",
            "in_width",
            "out_height",
            "out_width",
            "stride",
            "dr_max",
            "n_chan",
            "n_filt",
            "k",
            "num_weight_rows",
            "rows_resident",
        ):
            params[a] = node.attributes[a]
        n_out = node.attributes["out_height"] * node.attributes["out_width"]
        pf = _choose_parallelization(node, n_out)
        params["n_pixels"] = pf
        params["n_partitions"] = n_out // pf
        params["row_map_vals"] = _int_csv(node.get_attr("row_map_data"))
        params["indices_vals"] = _int_csv(node.get_attr("indices_data"))
        off_even = node.get_attr("off_even_data")
        off_odd = node.get_attr("off_odd_data")
        params["off_even_vals"] = _int_pairs_csv(off_even)
        params["off_odd_vals"] = _int_pairs_csv(off_odd)
        params["col_span"] = _col_span(off_even, off_odd)
        params["dc_max"] = _dc_max(off_even, off_odd)
        params["strategy"] = _strategy_enum(node)
        if _is_stream(node):
            _warn_reuse_factor(
                node,
                node.attributes["num_weight_rows"] * node.attributes["n_chan"],
                node.attributes["n_filt"],
            )
        n_terms = node.attributes["k"] * node.attributes["n_chan"]
        params["accum_t_str"] = _accum_type_str(node, n_terms)
        params["product_type"] = _product_type(node)
        return self.template.format(**params)


class HexConvLineBufferFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(
            HHexConvLineBuffer,
            include_header=hex_conv_linebuffer_include_list
            + hex_conv_linebuffer_stream_include_list,
        )
        self.template = hex_conv_linebuffer_function_template
        self.template_stream = hex_conv_linebuffer_stream_function_template

    def format(self, node):
        params = {}
        params["config"] = f"config{node.index}"
        params["index"] = node.index
        params["input_t"] = node.get_input_variable(node.inputs[0]).type.name
        params["mac_weights_t"] = node.get_weights("mac_weights").type.name
        params["mac_bias_t"] = node.get_weights("mac_bias").type.name
        params["output_t"] = node.get_output_variable().type.name
        params["input"] = node.get_input_variable(node.inputs[0]).name
        params["output"] = node.get_output_variable().name
        if _is_stream(node):
            return self.template_stream.format(**params)
        print(
            f'WARNING: layer "{node.name}" uses the linebuffer strategy under '
            "io_parallel, where a line buffer gives no memory benefit (the gather "
            "im2col is already frame-buffered). Falling back to the frame-buffered "
            "gather+MAC kernel. Use io_type='io_stream' to get the line-buffer win."
        )
        return self.template.format(**params)


class HHexPoolLineBuffer(Layer):
    """hls4ml IR node for HexPoolLineBuffer (fused hex max-pool)."""

    _expected_attributes = [
        Attribute("in_height"),
        Attribute("in_width"),
        Attribute("out_height"),
        Attribute("out_width"),
        Attribute("stride"),
        Attribute("dr_max"),
        Attribute("n_chan"),
        Attribute("k"),
        Attribute("rows_resident"),
    ]

    def initialize(self):
        n_out = self.attributes["out_height"] * self.attributes["out_width"]
        self.add_output_variable(shape=[n_out, self.attributes["n_chan"]])


class HexPoolLineBufferHandler(KerasV3LayerHandler):
    handles = ("keras_hexagdly.hex_gather.HexPoolLineBuffer",)

    def handle(self, layer, in_tensors, out_tensors):
        from keras_hexagdly.indexed import build_parity_offset_tables

        in_shape = list(in_tensors[0].shape[1:])  # (H_in, W_in, C)
        out_shape = list(out_tensors[0].shape[1:])  # (H_out, W_out, C)
        H = int(in_shape[0])
        W = int(in_shape[1])
        n_chan = int(in_shape[2])
        H_out = int(out_shape[0])
        W_out = int(out_shape[1])
        stride = int(layer.stride)

        nbr = np.asarray(keras_ops_to_numpy(layer.neighbor_idx), dtype=np.int32)
        k = int(nbr.shape[1])
        off_even, off_odd, rows_resident = build_parity_offset_tables(nbr, W, H_out, W_out, stride)
        dr_max = int(max(off_even[:, 0].max(), off_odd[:, 0].max()))

        return {
            "class_name": "HHexPoolLineBuffer",
            "in_height": H,
            "in_width": W,
            "out_height": H_out,
            "out_width": W_out,
            "stride": stride,
            "dr_max": dr_max,
            "n_chan": n_chan,
            "k": k,
            "rows_resident": int(rows_resident),
            "indices_data": nbr.reshape(-1),
            "off_even_data": np.asarray(off_even, dtype=np.int32).reshape(-1),
            "off_odd_data": np.asarray(off_odd, dtype=np.int32).reshape(-1),
        }


hex_pool_linebuffer_config_template = """\
struct config{index} : nnet::hex_pool_linebuffer_config {{
    static const unsigned in_height  = {in_height};
    static const unsigned in_width   = {in_width};
    static const unsigned out_height = {out_height};
    static const unsigned out_width  = {out_width};
    static const unsigned stride     = {stride};
    static const int      dr_max     = {dr_max};
    static const unsigned n_chan     = {n_chan};
    static const unsigned k          = {k};
    static const unsigned rows_resident = {rows_resident};
    static const unsigned col_span      = {col_span};
    static const int      dc_max        = {dc_max};
    static const unsigned reuse_factor  = {reuse};
    static const int indices[out_height * out_width * k];
    static const int off_even[k][2];
    static const int off_odd[k][2];
}};
const int config{index}::indices[] = {{{indices_vals}}};
const int config{index}::off_even[][2] = {{{off_even_vals}}};
const int config{index}::off_odd[][2] = {{{off_odd_vals}}};\n"""

hex_pool_linebuffer_function_template = (
    "nnet::hex_pool_linebuffer<{input_t}, {output_t}, {config}>({input}, {output});"
)
hex_pool_linebuffer_stream_function_template = (
    "nnet::hex_pool_linebuffer_stream<{input_t}, {output_t}, {config}>({input}, {output});"
)
hex_pool_linebuffer_include_list = ["nnet_utils/nnet_hex_pool_linebuffer.h"]
hex_pool_linebuffer_stream_include_list = ["nnet_utils/nnet_hex_pool_linebuffer_stream.h"]


class HexPoolLineBufferConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(HHexPoolLineBuffer)
        self.template = hex_pool_linebuffer_config_template

    def format(self, node):
        params = self._default_config_params(node)
        for a in (
            "in_height",
            "in_width",
            "out_height",
            "out_width",
            "stride",
            "dr_max",
            "n_chan",
            "k",
            "rows_resident",
        ):
            params[a] = node.attributes[a]
        params["indices_vals"] = _int_csv(node.get_attr("indices_data"))
        off_even = node.get_attr("off_even_data")
        off_odd = node.get_attr("off_odd_data")
        params["off_even_vals"] = _int_pairs_csv(off_even)
        params["off_odd_vals"] = _int_pairs_csv(off_odd)
        params["col_span"] = _col_span(off_even, off_odd)
        params["dc_max"] = _dc_max(off_even, off_odd)
        return self.template.format(**params)


class HexPoolLineBufferFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(
            HHexPoolLineBuffer,
            include_header=hex_pool_linebuffer_include_list
            + hex_pool_linebuffer_stream_include_list,
        )
        self.template = hex_pool_linebuffer_function_template
        self.template_stream = hex_pool_linebuffer_stream_function_template

    def format(self, node):
        params = {}
        params["config"] = f"config{node.index}"
        params["input_t"] = node.get_input_variable(node.inputs[0]).type.name
        params["output_t"] = node.get_output_variable().type.name
        params["input"] = node.get_input_variable(node.inputs[0]).name
        params["output"] = node.get_output_variable().name
        if _is_stream(node):
            return self.template_stream.format(**params)
        print(
            f'WARNING: layer "{node.name}" uses the linebuffer strategy under '
            "io_parallel (no line-buffer benefit); using the frame-buffered pool "
            "kernel. Use io_type='io_stream' for the line-buffer win."
        )
        return self.template.format(**params)


class HHexPoolLineBuffer3D(Layer):
    """hls4ml IR node for HexPoolLineBuffer3D (fused depth-aware hex max-pool)."""

    _expected_attributes = [
        Attribute("n_depth_in"),
        Attribute("n_depth_out"),
        Attribute("depth_size"),
        Attribute("depth_stride"),
        Attribute("in_height"),
        Attribute("in_width"),
        Attribute("dr_max"),
        Attribute("n_chan"),
        Attribute("k"),
        Attribute("rows_resident"),
    ]

    def initialize(self):
        n_out = (
            self.attributes["n_depth_out"]
            * self.attributes["in_height"]
            * self.attributes["in_width"]
        )
        self.add_output_variable(shape=[n_out, self.attributes["n_chan"]])


class HexPoolLineBuffer3DHandler(KerasV3LayerHandler):
    handles = ("keras_hexagdly.hex_gather.HexPoolLineBuffer3D",)

    def handle(self, layer, in_tensors, out_tensors):
        from keras_hexagdly.indexed import build_parity_offset_tables

        in_shape = list(in_tensors[0].shape[1:])  # (D_in, H, W, C)
        out_shape = list(out_tensors[0].shape[1:])  # (D_out, H, W, C)
        D_in = int(in_shape[0])
        H = int(in_shape[1])
        W = int(in_shape[2])
        n_chan = int(in_shape[3])
        D_out = int(out_shape[0])

        nbr = np.asarray(keras_ops_to_numpy(layer.neighbor_idx), dtype=np.int32)
        k = int(nbr.shape[1])
        off_even, off_odd, rows_resident = build_parity_offset_tables(nbr, W, H, W, 1)
        dr_max = int(max(off_even[:, 0].max(), off_odd[:, 0].max()))

        return {
            "class_name": "HHexPoolLineBuffer3D",
            "n_depth_in": D_in,
            "n_depth_out": D_out,
            "depth_size": int(layer.depth_size),
            "depth_stride": int(layer.depth_stride),
            "in_height": H,
            "in_width": W,
            "dr_max": dr_max,
            "n_chan": n_chan,
            "k": k,
            "rows_resident": int(rows_resident),
            "indices_data": nbr.reshape(-1),
            "off_even_data": np.asarray(off_even, dtype=np.int32).reshape(-1),
            "off_odd_data": np.asarray(off_odd, dtype=np.int32).reshape(-1),
        }


hex_pool_linebuffer_3d_config_template = """\
struct config{index} : nnet::hex_pool_linebuffer_3d_config {{
    static const unsigned n_depth_in   = {n_depth_in};
    static const unsigned n_depth_out  = {n_depth_out};
    static const unsigned depth_size   = {depth_size};
    static const unsigned depth_stride = {depth_stride};
    static const unsigned in_height  = {in_height};
    static const unsigned in_width   = {in_width};
    static const unsigned out_height = {in_height};
    static const unsigned out_width  = {in_width};
    static const unsigned stride     = 1;
    static const int      dr_max     = {dr_max};
    static const unsigned n_chan     = {n_chan};
    static const unsigned k          = {k};
    static const unsigned rows_resident = {rows_resident};
    static const unsigned reuse_factor  = {reuse};
    static const int indices[in_height * in_width * k];
    static const int off_even[k][2];
    static const int off_odd[k][2];
}};
const int config{index}::indices[] = {{{indices_vals}}};
const int config{index}::off_even[][2] = {{{off_even_vals}}};
const int config{index}::off_odd[][2] = {{{off_odd_vals}}};\n"""

hex_pool_linebuffer_3d_function_template = (
    "nnet::hex_pool_linebuffer_3d<{input_t}, {output_t}, {config}>({input}, {output});"
)
hex_pool_linebuffer_3d_stream_function_template = (
    "nnet::hex_pool_linebuffer_3d_stream<{input_t}, {output_t}, {config}>({input}, {output});"
)
hex_pool_linebuffer_3d_include_list = ["nnet_utils/nnet_hex_pool_linebuffer_3d.h"]
hex_pool_linebuffer_3d_stream_include_list = ["nnet_utils/nnet_hex_pool_linebuffer_3d_stream.h"]


class HexPoolLineBuffer3DConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(HHexPoolLineBuffer3D)
        self.template = hex_pool_linebuffer_3d_config_template

    def format(self, node):
        params = self._default_config_params(node)
        for a in (
            "n_depth_in",
            "n_depth_out",
            "depth_size",
            "depth_stride",
            "in_height",
            "in_width",
            "dr_max",
            "n_chan",
            "k",
            "rows_resident",
        ):
            params[a] = node.attributes[a]
        params["indices_vals"] = _int_csv(node.get_attr("indices_data"))
        params["off_even_vals"] = _int_pairs_csv(node.get_attr("off_even_data"))
        params["off_odd_vals"] = _int_pairs_csv(node.get_attr("off_odd_data"))
        return self.template.format(**params)


class HexPoolLineBuffer3DFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(
            HHexPoolLineBuffer3D,
            include_header=hex_pool_linebuffer_3d_include_list
            + hex_pool_linebuffer_3d_stream_include_list,
        )
        self.template = hex_pool_linebuffer_3d_function_template
        self.template_stream = hex_pool_linebuffer_3d_stream_function_template

    def format(self, node):
        params = {}
        params["config"] = f"config{node.index}"
        params["input_t"] = node.get_input_variable(node.inputs[0]).type.name
        params["output_t"] = node.get_output_variable().type.name
        params["input"] = node.get_input_variable(node.inputs[0]).name
        params["output"] = node.get_output_variable().name
        if _is_stream(node):
            return self.template_stream.format(**params)
        print(
            f'WARNING: layer "{node.name}" uses the linebuffer strategy under '
            "io_parallel (no line-buffer benefit); using the frame-buffered pool "
            "kernel. Use io_type='io_stream' for the line-buffer win."
        )
        return self.template.format(**params)


class HHexConvLineBuffer3D(Layer):
    """hls4ml IR node for HexConvLineBuffer3D (fused depth-aware hex conv)."""

    _expected_attributes = [
        Attribute("n_depth_in"),
        Attribute("n_depth_out"),
        Attribute("depth_size"),
        Attribute("pad_top"),
        Attribute("pad_bot"),
        Attribute("in_height"),
        Attribute("in_width"),
        Attribute("n_chan"),
        Attribute("n_filt"),
        Attribute("k"),
        Attribute("num_weight_rows"),
        Attribute("rows_resident"),
        Attribute("share_neighbors", value_type=bool),
        WeightAttribute("mac_weights"),
        TypeAttribute("mac_weights"),
        WeightAttribute("mac_bias"),
        TypeAttribute("mac_bias"),
    ]

    def initialize(self):
        n_out = (
            self.attributes["n_depth_out"]
            * self.attributes["in_height"]
            * self.attributes["in_width"]
        )
        self.add_output_variable(shape=[n_out, self.attributes["n_filt"]])
        self.add_weights_variable(name="mac_weights", var_name="w{index}")
        self.add_weights_variable(name="mac_bias", var_name="b{index}")


class HexConvLineBuffer3DHandler(KerasV3LayerHandler):
    handles = ("keras_hexagdly.hex_gather.HexConvLineBuffer3D",)

    def handle(self, layer, in_tensors, out_tensors):
        from keras_hexagdly.indexed import build_parity_offset_tables

        in_shape = list(in_tensors[0].shape[1:])  # (D_in, H, W, Cin)
        out_shape = list(out_tensors[0].shape[1:])  # (D_out, H, W, Cout)
        D_in = int(in_shape[0])
        H = int(in_shape[1])
        W = int(in_shape[2])
        n_chan = int(in_shape[3])
        D_out = int(out_shape[0])
        n_filt = int(out_shape[3])

        depth_size = int(layer.depth_size)
        if layer.depth_padding == "same":
            pad_top = (depth_size - 1) // 2
            pad_bot = depth_size - 1 - pad_top
        else:
            pad_top = 0
            pad_bot = 0

        nbr = np.asarray(keras_ops_to_numpy(layer.neighbor_idx), dtype=np.int32)
        k = int(nbr.shape[1])
        # 3D linebuffer is hex-stride 1 (spatial output grid == input grid).
        off_even, off_odd, rows_resident = build_parity_offset_tables(nbr, W, H, W, 1)

        w = keras_ops_to_numpy(layer.mac_weights).astype(np.float32).reshape(-1)
        num_weight_rows = int(layer.mac_weights.shape[1])  # (depth, rows, Cin, Cout)
        if layer.share_neighbors:
            ring_idx = keras_ops_to_numpy(layer.ring_idx).astype(np.int32).reshape(-1)
        else:
            ring_idx = np.arange(k, dtype=np.int32)
        bias = keras_ops_to_numpy(layer.mac_bias).astype(np.float32).reshape(-1)

        return {
            "class_name": "HHexConvLineBuffer3D",
            "n_depth_in": D_in,
            "n_depth_out": D_out,
            "depth_size": depth_size,
            "pad_top": pad_top,
            "pad_bot": pad_bot,
            "in_height": H,
            "in_width": W,
            "n_chan": n_chan,
            "n_filt": n_filt,
            "k": k,
            "num_weight_rows": num_weight_rows,
            "rows_resident": int(rows_resident),
            "share_neighbors": bool(layer.share_neighbors),
            "indices_data": nbr.reshape(-1),
            "off_even_data": np.asarray(off_even, dtype=np.int32).reshape(-1),
            "off_odd_data": np.asarray(off_odd, dtype=np.int32).reshape(-1),
            "row_map_data": ring_idx,
            "mac_weights_data": w,
            "mac_bias_data": bias,
        }


hex_conv_linebuffer_3d_config_template = """\
struct config{index} : nnet::hex_conv_linebuffer_3d_config {{
    static const unsigned n_depth_in  = {n_depth_in};
    static const unsigned n_depth_out = {n_depth_out};
    static const unsigned depth_size  = {depth_size};
    static const int      pad_top     = {pad_top};
    static const int      pad_bot     = {pad_bot};
    static const unsigned in_height = {in_height};
    static const unsigned in_width  = {in_width};
    static const unsigned n_chan    = {n_chan};
    static const unsigned n_filt    = {n_filt};
    static const unsigned k         = {k};
    static const unsigned num_weight_rows = {num_weight_rows};
    static const unsigned rows_resident   = {rows_resident};
    static const unsigned n_pixels     = {n_pixels};
    static const unsigned n_partitions = {n_partitions};
    static const unsigned reuse_factor = {reuse};
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(n_pixels * depth_size * num_weight_rows * n_chan * n_filt, reuse_factor);
    static const unsigned row_map[k];
    static const int indices[in_height * in_width * k];
    static const int off_even[k][2];
    static const int off_odd[k][2];
    typedef {mac_weights_t.name} weight_t;
    typedef {mac_bias_t.name}    bias_t;
    typedef {accum_t_str} accum_t;
    // Product kernel chosen from the data/weight precisions, exactly as
    // hls4ml does for its own layers (see _product_type).
    template<class x_T, class y_T> using product = nnet::product::{product_type}<x_T, y_T>;
}};
const unsigned config{index}::row_map[] = {{{row_map_vals}}};
const int config{index}::indices[] = {{{indices_vals}}};
const int config{index}::off_even[][2] = {{{off_even_vals}}};
const int config{index}::off_odd[][2] = {{{off_odd_vals}}};\n"""

hex_conv_linebuffer_3d_function_template = (
    "nnet::hex_conv_linebuffer_3d<{input_t}, {mac_weights_t}, {mac_bias_t}, "
    "{output_t}, {config}>({input}, w{index}, b{index}, {output});"
)
hex_conv_linebuffer_3d_stream_function_template = (
    "nnet::hex_conv_linebuffer_3d_stream<{input_t}, {mac_weights_t}, {mac_bias_t}, "
    "{output_t}, {config}>({input}, w{index}, b{index}, {output});"
)

hex_conv_linebuffer_3d_include_list = ["nnet_utils/nnet_hex_conv_linebuffer_3d.h"]
hex_conv_linebuffer_3d_stream_include_list = ["nnet_utils/nnet_hex_conv_linebuffer_3d_stream.h"]


class HexConvLineBuffer3DConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(HHexConvLineBuffer3D)
        self.template = hex_conv_linebuffer_3d_config_template

    def format(self, node):
        params = self._default_config_params(node)
        for a in (
            "n_depth_in",
            "n_depth_out",
            "depth_size",
            "pad_top",
            "pad_bot",
            "in_height",
            "in_width",
            "n_chan",
            "n_filt",
            "k",
            "num_weight_rows",
            "rows_resident",
        ):
            params[a] = node.attributes[a]
        n_out = (
            node.attributes["n_depth_out"]
            * node.attributes["in_height"]
            * node.attributes["in_width"]
        )
        pf = _choose_parallelization(node, n_out)
        params["n_pixels"] = pf
        params["n_partitions"] = n_out // pf
        params["row_map_vals"] = _int_csv(node.get_attr("row_map_data"))
        params["indices_vals"] = _int_csv(node.get_attr("indices_data"))
        params["off_even_vals"] = _int_pairs_csv(node.get_attr("off_even_data"))
        params["off_odd_vals"] = _int_pairs_csv(node.get_attr("off_odd_data"))
        n_terms = node.attributes["depth_size"] * node.attributes["k"] * node.attributes["n_chan"]
        params["accum_t_str"] = _accum_type_str(node, n_terms)
        params["product_type"] = _product_type(node)
        return self.template.format(**params)


class HexConvLineBuffer3DFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(
            HHexConvLineBuffer3D,
            include_header=hex_conv_linebuffer_3d_include_list
            + hex_conv_linebuffer_3d_stream_include_list,
        )
        self.template = hex_conv_linebuffer_3d_function_template
        self.template_stream = hex_conv_linebuffer_3d_stream_function_template

    def format(self, node):
        params = {}
        params["config"] = f"config{node.index}"
        params["index"] = node.index
        params["input_t"] = node.get_input_variable(node.inputs[0]).type.name
        params["mac_weights_t"] = node.get_weights("mac_weights").type.name
        params["mac_bias_t"] = node.get_weights("mac_bias").type.name
        params["output_t"] = node.get_output_variable().type.name
        params["input"] = node.get_input_variable(node.inputs[0]).name
        params["output"] = node.get_output_variable().name
        if _is_stream(node):
            return self.template_stream.format(**params)
        print(
            f'WARNING: layer "{node.name}" uses the linebuffer strategy under '
            "io_parallel, where a line buffer gives no memory benefit. Falling back "
            "to the frame-buffered gather+MAC kernel. Use io_type='io_stream' for "
            "the line-buffer win."
        )
        return self.template.format(**params)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_registered = False


def register_hex_gather_layers(backend_name="Vivado"):
    """Register the line-buffer hex layers with hls4ml.

    Call once before ``hls4ml.converters.convert_from_keras_model`` when the
    model went through ``patch_model_for_hls(..., strategy='linebuffer')`` (the
    default).  Idempotent — safe to call multiple times.
    """
    global _registered
    if _registered:
        return

    # Register hls4ml IR layer types
    hls4ml.model.layers.register_layer("HHexConvLineBuffer", HHexConvLineBuffer)
    hls4ml.model.layers.register_layer("HHexConvLineBuffer3D", HHexConvLineBuffer3D)
    hls4ml.model.layers.register_layer("HHexPoolLineBuffer", HHexPoolLineBuffer)
    hls4ml.model.layers.register_layer("HHexPoolLineBuffer3D", HHexPoolLineBuffer3D)

    backend = hls4ml.backends.get_backend(backend_name)

    # Register config + function templates
    backend.register_template(HexConvLineBufferConfigTemplate)
    backend.register_template(HexConvLineBufferFunctionTemplate)
    backend.register_template(HexConvLineBuffer3DConfigTemplate)
    backend.register_template(HexConvLineBuffer3DFunctionTemplate)
    backend.register_template(HexPoolLineBufferConfigTemplate)
    backend.register_template(HexPoolLineBufferFunctionTemplate)
    backend.register_template(HexPoolLineBuffer3DConfigTemplate)
    backend.register_template(HexPoolLineBuffer3DFunctionTemplate)

    # Register HLS C++ source files (Phase 4 — must exist alongside this file)
    here = Path(__file__).parent
    for fname in (
        "nnet_hex_conv_linebuffer.h",
        "nnet_hex_conv_linebuffer_stream.h",
        "nnet_hex_conv_linebuffer_3d.h",
        "nnet_hex_conv_linebuffer_3d_stream.h",
        "nnet_hex_pool_linebuffer.h",
        "nnet_hex_pool_linebuffer_stream.h",
        "nnet_hex_pool_linebuffer_3d.h",
        "nnet_hex_pool_linebuffer_3d_stream.h",
    ):
        p = here / fname
        if p.exists():
            backend.register_source(p)

    _registered = True
