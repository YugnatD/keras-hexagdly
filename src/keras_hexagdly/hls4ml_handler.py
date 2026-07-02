"""hls4ml handler for HexGather and HexRingMAC (Phase 3 of gather strategy).

Call ``register_hex_gather_layers()`` once before converting a model that uses
strategy="gather".  This registers:
  - KerasV3 layer handlers  (convert HexGather / HexRingMAC → hls4ml IR nodes)
  - hls4ml IR layer classes (HHexGather, HHexRingMAC)
  - HLS config + function templates for the Vivado backend
  - The nnet_hex_gather.h / nnet_hex_ring_mac.h source files (Phase 4)

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

# ---------------------------------------------------------------------------
# hls4ml IR layers
# ---------------------------------------------------------------------------


class HHexGather(Layer):
    """hls4ml IR node for HexGather.

    Inputs:  flat pixel tensor (B, N_in*C_flat) — hls4ml flattens everything
    Weights: indices (N_out, K) int ROM  — synthesized as a constant BRAM/ROM
    Output:  gathered tensor (B, N_out*K*C)
    """

    _expected_attributes = [
        Attribute("n_in"),  # total input pixels N_in
        Attribute("n_out"),  # total output pixels N_out
        Attribute("k"),  # number of neighbor slots K
        Attribute("n_chan"),  # channels C
        WeightAttribute("indices"),
        TypeAttribute("indices"),
    ]

    def initialize(self):
        from hls4ml.model.types import IntegerPrecisionType

        n_out = self.attributes["n_out"]
        k = self.attributes["k"]
        n_chan = self.attributes["n_chan"]
        self.add_output_variable(shape=[n_out * k * n_chan])
        # Force integer precision for the index table — it holds pixel indices,
        # not fixed-point values.  Without this hls4ml assigns the global
        # ap_fixed precision which breaks the (idx >= 0) border check.
        self.add_weights_variable(
            name="indices",
            var_name="idx{index}",
            precision=IntegerPrecisionType(width=16, signed=True),
        )


class HHexRingMAC(Layer):
    """hls4ml IR node for HexRingMAC.

    Inputs:  gathered tensor (B, N_out*K*C_in)
    Weights: mac_weights  (num_rings or K, Cin, Cout)
             ring_idx     (K,) int — only present when share_neighbors=True
    Output:  (B, N_out*Cout)
    """

    _expected_attributes = [
        Attribute("n_out"),
        Attribute("k"),
        Attribute("n_in_chan"),
        Attribute("n_out_chan"),
        Attribute("num_weight_rows"),  # num_rings (shared) or K (full)
        Attribute("share_neighbors", value_type=bool),
        WeightAttribute("mac_weights"),
        TypeAttribute("mac_weights"),
        WeightAttribute("ring_idx"),
        TypeAttribute("ring_idx"),
    ]

    def initialize(self):
        n_out = self.attributes["n_out"]
        n_out_chan = self.attributes["n_out_chan"]
        from hls4ml.model.types import IntegerPrecisionType

        self.add_output_variable(shape=[n_out * n_out_chan])
        self.add_weights_variable(name="mac_weights", var_name="w{index}")
        # ring_idx holds ring indices (small integers) — force integer precision.
        self.add_weights_variable(
            name="ring_idx",
            var_name="ridx{index}",
            precision=IntegerPrecisionType(width=8, signed=False),
        )


class HHexMaxPool(Layer):
    """hls4ml IR node for HexMaxPool.

    Input:  gathered tensor (B, N_out*K*C)
    Output: (B, N_out*C)
    No weights — pure max reduction over K slots.
    Border slots contributed 0 from HexGather so max >= 0 for non-neg inputs.
    """

    _expected_attributes = [
        Attribute("n_out"),
        Attribute("k"),
        Attribute("n_chan"),
    ]

    def initialize(self):
        n_out = self.attributes["n_out"]
        n_chan = self.attributes["n_chan"]
        self.add_output_variable(shape=[n_out * n_chan])


# ---------------------------------------------------------------------------
# KerasV3 handlers
# ---------------------------------------------------------------------------


class HexGatherHandler(KerasV3LayerHandler):
    handles = ("keras_hexagdly.hex_gather.HexGather",)

    def handle(self, layer, in_tensors, out_tensors):
        in_shape = list(in_tensors[0].shape[1:])  # (N_in, C)
        out_shape = list(out_tensors[0].shape[1:])  # (N_out, K, C)

        n_in = int(in_shape[0])
        n_chan = int(in_shape[1])
        n_out = int(out_shape[0])
        k = int(out_shape[1])

        indices = np.asarray(keras_ops_to_numpy(layer.neighbor_idx), dtype=np.int32).reshape(
            -1
        )  # flatten (N_out, K) → (N_out*K,) for hls4ml weight storage

        return {
            "class_name": "HHexGather",
            "n_in": n_in,
            "n_out": n_out,
            "k": k,
            "n_chan": n_chan,
            "indices_data": indices,
        }


class HexRingMACHandler(KerasV3LayerHandler):
    handles = ("keras_hexagdly.hex_gather.HexRingMAC",)

    def handle(self, layer, in_tensors, out_tensors):
        in_shape = list(in_tensors[0].shape[1:])  # (N_out, K, Cin)
        out_shape = list(out_tensors[0].shape[1:])  # (N_out, Cout)

        n_out = int(in_shape[0])
        k = int(in_shape[1])
        n_in_chan = int(in_shape[2])
        n_out_chan = int(out_shape[1])
        share = layer.share_neighbors

        w = keras_ops_to_numpy(layer.mac_weights).astype(np.float32).reshape(-1)
        num_weight_rows = layer.mac_weights.shape[0]

        if share:
            ring_idx = keras_ops_to_numpy(layer.ring_idx).astype(np.int32).reshape(-1)
        else:
            # no ring sharing — dummy ring_idx that maps slot k → k (identity)
            ring_idx = np.arange(k, dtype=np.int32)

        return {
            "class_name": "HHexRingMAC",
            "n_out": n_out,
            "k": k,
            "n_in_chan": n_in_chan,
            "n_out_chan": n_out_chan,
            "num_weight_rows": num_weight_rows,
            "share_neighbors": share,
            "mac_weights_data": w,
            "ring_idx_data": ring_idx,
        }


def keras_ops_to_numpy(var):
    """Convert a Keras variable / tensor to numpy safely."""
    import keras

    try:
        return keras.ops.convert_to_numpy(var)
    except Exception:
        return np.asarray(var)


# ---------------------------------------------------------------------------
# HLS C++ config templates
# ---------------------------------------------------------------------------

hex_gather_config_template = """\
struct config{index} : nnet::hex_gather_config {{
    static const unsigned n_in   = {n_in};
    static const unsigned n_out  = {n_out};
    static const unsigned k      = {k};
    static const unsigned n_chan  = {n_chan};
    typedef {indices_t.name} indices_t;
}};\n"""

hex_gather_function_template = (
    "nnet::hex_gather<{input_t}, {indices_t}, {output_t}, {config}>({input}, idx{index}, {output});"
)

hex_ring_mac_config_template = """\
struct config{index} : nnet::hex_ring_mac_config {{
    static const unsigned n_out          = {n_out};
    static const unsigned k              = {k};
    static const unsigned n_in_chan      = {n_in_chan};
    static const unsigned n_out_chan     = {n_out_chan};
    static const unsigned num_weight_rows = {num_weight_rows};
    static const bool     share_neighbors = {share_neighbors_str};
    typedef {mac_weights_t.name} weight_t;
    typedef {ring_idx_t.name}    ring_idx_t;
    typedef {mac_weights_t.name} accum_t;
}};\n"""

hex_ring_mac_function_template = (
    "nnet::hex_ring_mac<{input_t}, {mac_weights_t}, {ring_idx_t}, {output_t}, {config}>"
    "({input}, w{index}, ridx{index}, {output});"
)

hex_gather_include_list = ["nnet_utils/nnet_hex_gather.h"]
hex_ring_mac_include_list = ["nnet_utils/nnet_hex_ring_mac.h"]


class HexGatherConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(HHexGather)
        self.template = hex_gather_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params["n_in"] = node.attributes["n_in"]
        params["n_out"] = node.attributes["n_out"]
        params["k"] = node.attributes["k"]
        params["n_chan"] = node.attributes["n_chan"]
        return self.template.format(**params)


class HexGatherFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(HHexGather, include_header=hex_gather_include_list)
        self.template = hex_gather_function_template

    def format(self, node):
        params = {}
        params["config"] = f"config{node.index}"
        params["index"] = node.index
        params["input_t"] = node.get_input_variable(node.inputs[0]).type.name
        params["indices_t"] = node.get_weights("indices").type.name
        params["output_t"] = node.get_output_variable().type.name
        params["input"] = node.get_input_variable(node.inputs[0]).name
        params["output"] = node.get_output_variable().name
        return self.template.format(**params)


class HexRingMACConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(HHexRingMAC)
        self.template = hex_ring_mac_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params["n_out"] = node.attributes["n_out"]
        params["k"] = node.attributes["k"]
        params["n_in_chan"] = node.attributes["n_in_chan"]
        params["n_out_chan"] = node.attributes["n_out_chan"]
        params["num_weight_rows"] = node.attributes["num_weight_rows"]
        params["share_neighbors_str"] = "true" if node.attributes["share_neighbors"] else "false"
        return self.template.format(**params)


class HexRingMACFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(HHexRingMAC, include_header=hex_ring_mac_include_list)
        self.template = hex_ring_mac_function_template

    def format(self, node):
        params = {}
        params["config"] = f"config{node.index}"
        params["index"] = node.index
        params["input_t"] = node.get_input_variable(node.inputs[0]).type.name
        params["mac_weights_t"] = node.get_weights("mac_weights").type.name
        params["ring_idx_t"] = node.get_weights("ring_idx").type.name
        params["output_t"] = node.get_output_variable().type.name
        params["input"] = node.get_input_variable(node.inputs[0]).name
        params["output"] = node.get_output_variable().name
        return self.template.format(**params)


class HexMaxPoolHandler(KerasV3LayerHandler):
    handles = ("keras_hexagdly.hex_gather.HexMaxPool",)

    def handle(self, layer, in_tensors, out_tensors):
        in_shape = list(in_tensors[0].shape[1:])  # (N_out, K, C)
        n_out = int(in_shape[0])
        k = int(in_shape[1])
        n_chan = int(in_shape[2])
        return {
            "class_name": "HHexMaxPool",
            "n_out": n_out,
            "k": k,
            "n_chan": n_chan,
        }


hex_max_pool_config_template = """\
struct config{index} : nnet::hex_max_pool_config {{
    static const unsigned n_out  = {n_out};
    static const unsigned k      = {k};
    static const unsigned n_chan  = {n_chan};
}};\n"""

hex_max_pool_function_template = (
    "nnet::hex_max_pool<{input_t}, {output_t}, {config}>({input}, {output});"
)

hex_max_pool_include_list = ["nnet_utils/nnet_hex_max_pool.h"]


class HexMaxPoolConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(HHexMaxPool)
        self.template = hex_max_pool_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params["n_out"] = node.attributes["n_out"]
        params["k"] = node.attributes["k"]
        params["n_chan"] = node.attributes["n_chan"]
        return self.template.format(**params)


class HexMaxPoolFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(HHexMaxPool, include_header=hex_max_pool_include_list)
        self.template = hex_max_pool_function_template

    def format(self, node):
        params = {}
        params["config"] = f"config{node.index}"
        params["input_t"] = node.get_input_variable(node.inputs[0]).type.name
        params["output_t"] = node.get_output_variable().type.name
        params["input"] = node.get_input_variable(node.inputs[0]).name
        params["output"] = node.get_output_variable().name
        return self.template.format(**params)


# ---------------------------------------------------------------------------
# 3D (depth-aware) variants — Conv3d gather export
#
# HHexGather3D / HHexRingMAC3D mirror the 2D nodes with an extra n_depth
# dimension.  hls4ml flattens the depth axis into the leading part of the flat
# array; the HLS kernels (nnet_hex_gather_3d.h / nnet_hex_ring_mac_3d.h) loop
# over n_depth frames applying the identical 2D spatial gather / MAC to each.
# ---------------------------------------------------------------------------


class HHexGather3D(Layer):
    """hls4ml IR node for HexGather3D.

    Inputs:  flat tensor (B, n_depth*n_in*n_chan)
    Weights: indices (n_out, k) int ROM (shared across all depth frames)
    Output:  gathered tensor (B, n_depth*n_out*k*n_chan)
    """

    _expected_attributes = [
        Attribute("n_depth"),  # number of depth frames D
        Attribute("n_in"),
        Attribute("n_out"),
        Attribute("k"),
        Attribute("n_chan"),
        WeightAttribute("indices"),
        TypeAttribute("indices"),
    ]

    def initialize(self):
        from hls4ml.model.types import IntegerPrecisionType

        n_depth = self.attributes["n_depth"]
        n_out = self.attributes["n_out"]
        k = self.attributes["k"]
        n_chan = self.attributes["n_chan"]
        self.add_output_variable(shape=[n_depth * n_out * k * n_chan])
        self.add_weights_variable(
            name="indices",
            var_name="idx{index}",
            precision=IntegerPrecisionType(width=16, signed=True),
        )


class HHexRingMAC3D(Layer):
    """hls4ml IR node for HexRingMAC3D.

    Inputs:  gathered tensor (B, n_depth*n_out*k*n_in_chan)
    Weights: mac_weights (num_rings or K, Cin, Cout); ring_idx (K,) int
    Output:  (B, n_depth*n_out*n_out_chan)
    """

    _expected_attributes = [
        Attribute("n_depth"),
        Attribute("n_out"),
        Attribute("k"),
        Attribute("n_in_chan"),
        Attribute("n_out_chan"),
        Attribute("num_weight_rows"),
        Attribute("share_neighbors", value_type=bool),
        WeightAttribute("mac_weights"),
        TypeAttribute("mac_weights"),
        WeightAttribute("ring_idx"),
        TypeAttribute("ring_idx"),
        WeightAttribute("mac_bias"),
        TypeAttribute("mac_bias"),
    ]

    def initialize(self):
        from hls4ml.model.types import IntegerPrecisionType

        n_depth = self.attributes["n_depth"]
        n_out = self.attributes["n_out"]
        n_out_chan = self.attributes["n_out_chan"]
        self.add_output_variable(shape=[n_depth * n_out * n_out_chan])
        self.add_weights_variable(name="mac_weights", var_name="w{index}")
        self.add_weights_variable(
            name="ring_idx",
            var_name="ridx{index}",
            precision=IntegerPrecisionType(width=8, signed=False),
        )
        self.add_weights_variable(name="mac_bias", var_name="b{index}")


class HexGather3DHandler(KerasV3LayerHandler):
    handles = ("keras_hexagdly.hex_gather.HexGather3D",)

    def handle(self, layer, in_tensors, out_tensors):
        in_shape = list(in_tensors[0].shape[1:])  # (D, N_in, C)
        out_shape = list(out_tensors[0].shape[1:])  # (D, N_out, K, C)

        n_depth = int(in_shape[0])
        n_in = int(in_shape[1])
        n_chan = int(in_shape[2])
        n_out = int(out_shape[1])
        k = int(out_shape[2])

        indices = np.asarray(keras_ops_to_numpy(layer.neighbor_idx), dtype=np.int32).reshape(
            -1
        )  # (N_out*K,) — shared across depth frames

        return {
            "class_name": "HHexGather3D",
            "n_depth": n_depth,
            "n_in": n_in,
            "n_out": n_out,
            "k": k,
            "n_chan": n_chan,
            "indices_data": indices,
        }


class HexRingMAC3DHandler(KerasV3LayerHandler):
    handles = ("keras_hexagdly.hex_gather.HexRingMAC3D",)

    def handle(self, layer, in_tensors, out_tensors):
        in_shape = list(in_tensors[0].shape[1:])  # (D, N_out, K, Cin)
        out_shape = list(out_tensors[0].shape[1:])  # (D, N_out, Cout)

        n_depth = int(in_shape[0])
        n_out = int(in_shape[1])
        k = int(in_shape[2])
        n_in_chan = int(in_shape[3])
        n_out_chan = int(out_shape[2])
        share = layer.share_neighbors

        w = keras_ops_to_numpy(layer.mac_weights).astype(np.float32).reshape(-1)
        num_weight_rows = layer.mac_weights.shape[0]

        if share:
            ring_idx = keras_ops_to_numpy(layer.ring_idx).astype(np.int32).reshape(-1)
        else:
            ring_idx = np.arange(k, dtype=np.int32)  # identity slot->row

        bias = keras_ops_to_numpy(layer.mac_bias).astype(np.float32).reshape(-1)

        return {
            "class_name": "HHexRingMAC3D",
            "n_depth": n_depth,
            "n_out": n_out,
            "k": k,
            "n_in_chan": n_in_chan,
            "n_out_chan": n_out_chan,
            "num_weight_rows": num_weight_rows,
            "share_neighbors": share,
            "mac_weights_data": w,
            "ring_idx_data": ring_idx,
            "mac_bias_data": bias,
        }


hex_gather_3d_config_template = """\
struct config{index} : nnet::hex_gather_3d_config {{
    static const unsigned n_depth = {n_depth};
    static const unsigned n_in    = {n_in};
    static const unsigned n_out   = {n_out};
    static const unsigned k       = {k};
    static const unsigned n_chan  = {n_chan};
    typedef {indices_t.name} indices_t;
}};\n"""

hex_gather_3d_function_template = (
    "nnet::hex_gather_3d<{input_t}, {indices_t}, {output_t}, {config}>"
    "({input}, idx{index}, {output});"
)

hex_ring_mac_3d_config_template = """\
struct config{index} : nnet::hex_ring_mac_3d_config {{
    static const unsigned n_depth         = {n_depth};
    static const unsigned n_out           = {n_out};
    static const unsigned k               = {k};
    static const unsigned n_in_chan       = {n_in_chan};
    static const unsigned n_out_chan      = {n_out_chan};
    static const unsigned num_weight_rows = {num_weight_rows};
    static const bool     share_neighbors = {share_neighbors_str};
    typedef {mac_weights_t.name} weight_t;
    typedef {ring_idx_t.name}    ring_idx_t;
    typedef {mac_bias_t.name}    bias_t;
    typedef {mac_weights_t.name} accum_t;
}};\n"""

hex_ring_mac_3d_function_template = (
    "nnet::hex_ring_mac_3d<{input_t}, {mac_weights_t}, {ring_idx_t}, "
    "{mac_bias_t}, {output_t}, {config}>"
    "({input}, w{index}, ridx{index}, b{index}, {output});"
)

hex_gather_3d_include_list = ["nnet_utils/nnet_hex_gather_3d.h"]
hex_ring_mac_3d_include_list = ["nnet_utils/nnet_hex_ring_mac_3d.h"]


class HexGather3DConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(HHexGather3D)
        self.template = hex_gather_3d_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params["n_depth"] = node.attributes["n_depth"]
        params["n_in"] = node.attributes["n_in"]
        params["n_out"] = node.attributes["n_out"]
        params["k"] = node.attributes["k"]
        params["n_chan"] = node.attributes["n_chan"]
        return self.template.format(**params)


class HexGather3DFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(HHexGather3D, include_header=hex_gather_3d_include_list)
        self.template = hex_gather_3d_function_template

    def format(self, node):
        params = {}
        params["config"] = f"config{node.index}"
        params["index"] = node.index
        params["input_t"] = node.get_input_variable(node.inputs[0]).type.name
        params["indices_t"] = node.get_weights("indices").type.name
        params["output_t"] = node.get_output_variable().type.name
        params["input"] = node.get_input_variable(node.inputs[0]).name
        params["output"] = node.get_output_variable().name
        return self.template.format(**params)


class HexRingMAC3DConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(HHexRingMAC3D)
        self.template = hex_ring_mac_3d_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params["n_depth"] = node.attributes["n_depth"]
        params["n_out"] = node.attributes["n_out"]
        params["k"] = node.attributes["k"]
        params["n_in_chan"] = node.attributes["n_in_chan"]
        params["n_out_chan"] = node.attributes["n_out_chan"]
        params["num_weight_rows"] = node.attributes["num_weight_rows"]
        params["share_neighbors_str"] = "true" if node.attributes["share_neighbors"] else "false"
        return self.template.format(**params)


class HexRingMAC3DFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(HHexRingMAC3D, include_header=hex_ring_mac_3d_include_list)
        self.template = hex_ring_mac_3d_function_template

    def format(self, node):
        params = {}
        params["config"] = f"config{node.index}"
        params["index"] = node.index
        params["input_t"] = node.get_input_variable(node.inputs[0]).type.name
        params["mac_weights_t"] = node.get_weights("mac_weights").type.name
        params["ring_idx_t"] = node.get_weights("ring_idx").type.name
        params["mac_bias_t"] = node.get_weights("mac_bias").type.name
        params["output_t"] = node.get_output_variable().type.name
        params["input"] = node.get_input_variable(node.inputs[0]).name
        params["output"] = node.get_output_variable().name
        return self.template.format(**params)


class HHexMaxPool3D(Layer):
    """hls4ml IR node for HexMaxPool3D.

    Input:  gathered tensor (B, n_depth_in*n_out*k*n_chan)
    Output: (B, n_depth_out*n_out*n_chan)
    No weights — reduces the max over the depth pool window (depth_size taps at
    stride depth_stride) and the K slots for each of the n_depth_out frames.
    """

    _expected_attributes = [
        Attribute("n_depth_in"),
        Attribute("n_depth_out"),
        Attribute("depth_size"),
        Attribute("depth_stride"),
        Attribute("n_out"),
        Attribute("k"),
        Attribute("n_chan"),
    ]

    def initialize(self):
        n_depth_out = self.attributes["n_depth_out"]
        n_out = self.attributes["n_out"]
        n_chan = self.attributes["n_chan"]
        self.add_output_variable(shape=[n_depth_out * n_out * n_chan])


class HexMaxPool3DHandler(KerasV3LayerHandler):
    handles = ("keras_hexagdly.hex_gather.HexMaxPool3D",)

    def handle(self, layer, in_tensors, out_tensors):
        in_shape = list(in_tensors[0].shape[1:])  # (D_in, N_out, K, C)
        out_shape = list(out_tensors[0].shape[1:])  # (D_out, N_out, C)

        return {
            "class_name": "HHexMaxPool3D",
            "n_depth_in": int(in_shape[0]),
            "n_depth_out": int(out_shape[0]),
            "depth_size": int(layer.depth_size),
            "depth_stride": int(layer.depth_stride),
            "n_out": int(in_shape[1]),
            "k": int(in_shape[2]),
            "n_chan": int(in_shape[3]),
        }


hex_max_pool_3d_config_template = """\
struct config{index} : nnet::hex_max_pool_3d_config {{
    static const unsigned n_depth_in   = {n_depth_in};
    static const unsigned n_depth_out  = {n_depth_out};
    static const unsigned depth_size   = {depth_size};
    static const unsigned depth_stride = {depth_stride};
    static const unsigned n_out        = {n_out};
    static const unsigned k            = {k};
    static const unsigned n_chan       = {n_chan};
}};\n"""

hex_max_pool_3d_function_template = (
    "nnet::hex_max_pool_3d<{input_t}, {output_t}, {config}>({input}, {output});"
)

hex_max_pool_3d_include_list = ["nnet_utils/nnet_hex_max_pool_3d.h"]


class HexMaxPool3DConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(HHexMaxPool3D)
        self.template = hex_max_pool_3d_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params["n_depth_in"] = node.attributes["n_depth_in"]
        params["n_depth_out"] = node.attributes["n_depth_out"]
        params["depth_size"] = node.attributes["depth_size"]
        params["depth_stride"] = node.attributes["depth_stride"]
        params["n_out"] = node.attributes["n_out"]
        params["k"] = node.attributes["k"]
        params["n_chan"] = node.attributes["n_chan"]
        return self.template.format(**params)


class HexMaxPool3DFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(HHexMaxPool3D, include_header=hex_max_pool_3d_include_list)
        self.template = hex_max_pool_3d_function_template

    def format(self, node):
        params = {}
        params["config"] = f"config{node.index}"
        params["input_t"] = node.get_input_variable(node.inputs[0]).type.name
        params["output_t"] = node.get_output_variable().type.name
        params["input"] = node.get_input_variable(node.inputs[0]).name
        params["output"] = node.get_output_variable().name
        return self.template.format(**params)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_registered = False


def register_hex_gather_layers(backend_name="Vivado"):
    """Register HexGather and HexRingMAC with hls4ml.

    Call once before ``hls4ml.converters.convert_from_keras_model`` when using
    ``strategy='gather'``.  Idempotent — safe to call multiple times.
    """
    global _registered
    if _registered:
        return

    # Register hls4ml IR layer types
    hls4ml.model.layers.register_layer("HHexGather", HHexGather)
    hls4ml.model.layers.register_layer("HHexRingMAC", HHexRingMAC)
    hls4ml.model.layers.register_layer("HHexMaxPool", HHexMaxPool)
    hls4ml.model.layers.register_layer("HHexGather3D", HHexGather3D)
    hls4ml.model.layers.register_layer("HHexRingMAC3D", HHexRingMAC3D)
    hls4ml.model.layers.register_layer("HHexMaxPool3D", HHexMaxPool3D)

    backend = hls4ml.backends.get_backend(backend_name)

    # Register config + function templates
    backend.register_template(HexGatherConfigTemplate)
    backend.register_template(HexGatherFunctionTemplate)
    backend.register_template(HexRingMACConfigTemplate)
    backend.register_template(HexRingMACFunctionTemplate)
    backend.register_template(HexMaxPoolConfigTemplate)
    backend.register_template(HexMaxPoolFunctionTemplate)
    backend.register_template(HexGather3DConfigTemplate)
    backend.register_template(HexGather3DFunctionTemplate)
    backend.register_template(HexRingMAC3DConfigTemplate)
    backend.register_template(HexRingMAC3DFunctionTemplate)
    backend.register_template(HexMaxPool3DConfigTemplate)
    backend.register_template(HexMaxPool3DFunctionTemplate)

    # Register HLS C++ source files (Phase 4 — must exist alongside this file)
    here = Path(__file__).parent
    for fname in (
        "nnet_hex_gather.h",
        "nnet_hex_ring_mac.h",
        "nnet_hex_max_pool.h",
        "nnet_hex_gather_3d.h",
        "nnet_hex_ring_mac_3d.h",
        "nnet_hex_max_pool_3d.h",
    ):
        p = here / fname
        if p.exists():
            backend.register_source(p)

    _registered = True
