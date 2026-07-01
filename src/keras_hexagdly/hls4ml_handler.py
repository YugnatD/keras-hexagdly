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

import numpy as np

import hls4ml
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.converters.keras_v3._base import KerasV3LayerHandler
from hls4ml.model.layers import Layer
from hls4ml.model.attributes import Attribute, WeightAttribute, TypeAttribute


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
        Attribute('n_in'),        # total input pixels N_in
        Attribute('n_out'),       # total output pixels N_out
        Attribute('k'),           # number of neighbor slots K
        Attribute('n_chan'),      # channels C
        WeightAttribute('indices'),
        TypeAttribute('indices'),
    ]

    def initialize(self):
        from hls4ml.model.types import IntegerPrecisionType
        n_out = self.attributes['n_out']
        k     = self.attributes['k']
        n_chan = self.attributes['n_chan']
        self.add_output_variable(shape=[n_out * k * n_chan])
        # Force integer precision for the index table — it holds pixel indices,
        # not fixed-point values.  Without this hls4ml assigns the global
        # ap_fixed precision which breaks the (idx >= 0) border check.
        self.add_weights_variable(
            name='indices', var_name='idx{index}',
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
        Attribute('n_out'),
        Attribute('k'),
        Attribute('n_in_chan'),
        Attribute('n_out_chan'),
        Attribute('num_weight_rows'),   # num_rings (shared) or K (full)
        Attribute('share_neighbors', value_type=bool),
        WeightAttribute('mac_weights'),
        TypeAttribute('mac_weights'),
        WeightAttribute('ring_idx'),
        TypeAttribute('ring_idx'),
    ]

    def initialize(self):
        n_out     = self.attributes['n_out']
        n_out_chan = self.attributes['n_out_chan']
        from hls4ml.model.types import IntegerPrecisionType
        self.add_output_variable(shape=[n_out * n_out_chan])
        self.add_weights_variable(name='mac_weights', var_name='w{index}')
        # ring_idx holds ring indices (small integers) — force integer precision.
        self.add_weights_variable(
            name='ring_idx', var_name='ridx{index}',
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
        Attribute('n_out'),
        Attribute('k'),
        Attribute('n_chan'),
    ]

    def initialize(self):
        n_out  = self.attributes['n_out']
        n_chan  = self.attributes['n_chan']
        self.add_output_variable(shape=[n_out * n_chan])


# ---------------------------------------------------------------------------
# KerasV3 handlers
# ---------------------------------------------------------------------------

class HexGatherHandler(KerasV3LayerHandler):
    handles = ('keras_hexagdly.hex_gather.HexGather',)

    def handle(self, layer, in_tensors, out_tensors):
        in_shape  = list(in_tensors[0].shape[1:])   # (N_in, C)
        out_shape = list(out_tensors[0].shape[1:])   # (N_out, K, C)

        n_in   = int(in_shape[0])
        n_chan  = int(in_shape[1])
        n_out  = int(out_shape[0])
        k      = int(out_shape[1])

        indices = np.asarray(
            keras_ops_to_numpy(layer.neighbor_idx), dtype=np.int32
        ).reshape(-1)   # flatten (N_out, K) → (N_out*K,) for hls4ml weight storage

        return {
            'class_name': 'HHexGather',
            'n_in':   n_in,
            'n_out':  n_out,
            'k':      k,
            'n_chan': n_chan,
            'indices_data': indices,
        }


class HexRingMACHandler(KerasV3LayerHandler):
    handles = ('keras_hexagdly.hex_gather.HexRingMAC',)

    def handle(self, layer, in_tensors, out_tensors):
        in_shape  = list(in_tensors[0].shape[1:])    # (N_out, K, Cin)
        out_shape = list(out_tensors[0].shape[1:])   # (N_out, Cout)

        n_out      = int(in_shape[0])
        k          = int(in_shape[1])
        n_in_chan  = int(in_shape[2])
        n_out_chan  = int(out_shape[1])
        share      = layer.share_neighbors

        w = keras_ops_to_numpy(layer.mac_weights).astype(np.float32).reshape(-1)
        num_weight_rows = layer.mac_weights.shape[0]

        if share:
            ring_idx = keras_ops_to_numpy(layer.ring_idx).astype(np.int32).reshape(-1)
        else:
            # no ring sharing — dummy ring_idx that maps slot k → k (identity)
            ring_idx = np.arange(k, dtype=np.int32)

        return {
            'class_name': 'HHexRingMAC',
            'n_out':           n_out,
            'k':               k,
            'n_in_chan':       n_in_chan,
            'n_out_chan':      n_out_chan,
            'num_weight_rows': num_weight_rows,
            'share_neighbors': share,
            'mac_weights_data': w,
            'ring_idx_data':    ring_idx,
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
    'nnet::hex_gather<{input_t}, {indices_t}, {output_t}, {config}>'
    '({input}, idx{index}, {output});'
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
    'nnet::hex_ring_mac<{input_t}, {mac_weights_t}, {ring_idx_t}, {output_t}, {config}>'
    '({input}, w{index}, ridx{index}, {output});'
)

hex_gather_include_list  = ['nnet_utils/nnet_hex_gather.h']
hex_ring_mac_include_list = ['nnet_utils/nnet_hex_ring_mac.h']


class HexGatherConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(HHexGather)
        self.template = hex_gather_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['n_in']   = node.attributes['n_in']
        params['n_out']  = node.attributes['n_out']
        params['k']      = node.attributes['k']
        params['n_chan']  = node.attributes['n_chan']
        return self.template.format(**params)


class HexGatherFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(HHexGather, include_header=hex_gather_include_list)
        self.template = hex_gather_function_template

    def format(self, node):
        params = {}
        params['config']    = f'config{node.index}'
        params['index']     = node.index
        params['input_t']   = node.get_input_variable(node.inputs[0]).type.name
        params['indices_t'] = node.get_weights('indices').type.name
        params['output_t']  = node.get_output_variable().type.name
        params['input']     = node.get_input_variable(node.inputs[0]).name
        params['output']    = node.get_output_variable().name
        return self.template.format(**params)


class HexRingMACConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(HHexRingMAC)
        self.template = hex_ring_mac_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['n_out']           = node.attributes['n_out']
        params['k']               = node.attributes['k']
        params['n_in_chan']       = node.attributes['n_in_chan']
        params['n_out_chan']      = node.attributes['n_out_chan']
        params['num_weight_rows'] = node.attributes['num_weight_rows']
        params['share_neighbors_str'] = (
            'true' if node.attributes['share_neighbors'] else 'false'
        )
        return self.template.format(**params)


class HexRingMACFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(HHexRingMAC, include_header=hex_ring_mac_include_list)
        self.template = hex_ring_mac_function_template

    def format(self, node):
        params = {}
        params['config']      = f'config{node.index}'
        params['index']       = node.index
        params['input_t']     = node.get_input_variable(node.inputs[0]).type.name
        params['mac_weights_t'] = node.get_weights('mac_weights').type.name
        params['ring_idx_t']  = node.get_weights('ring_idx').type.name
        params['output_t']    = node.get_output_variable().type.name
        params['input']       = node.get_input_variable(node.inputs[0]).name
        params['output']      = node.get_output_variable().name
        return self.template.format(**params)


class HexMaxPoolHandler(KerasV3LayerHandler):
    handles = ('keras_hexagdly.hex_gather.HexMaxPool',)

    def handle(self, layer, in_tensors, out_tensors):
        in_shape  = list(in_tensors[0].shape[1:])   # (N_out, K, C)
        n_out  = int(in_shape[0])
        k      = int(in_shape[1])
        n_chan  = int(in_shape[2])
        return {
            'class_name': 'HHexMaxPool',
            'n_out':  n_out,
            'k':      k,
            'n_chan': n_chan,
        }


hex_max_pool_config_template = """\
struct config{index} : nnet::hex_max_pool_config {{
    static const unsigned n_out  = {n_out};
    static const unsigned k      = {k};
    static const unsigned n_chan  = {n_chan};
}};\n"""

hex_max_pool_function_template = (
    'nnet::hex_max_pool<{input_t}, {output_t}, {config}>'
    '({input}, {output});'
)

hex_max_pool_include_list = ['nnet_utils/nnet_hex_max_pool.h']


class HexMaxPoolConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(HHexMaxPool)
        self.template = hex_max_pool_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['n_out']  = node.attributes['n_out']
        params['k']      = node.attributes['k']
        params['n_chan']  = node.attributes['n_chan']
        return self.template.format(**params)


class HexMaxPoolFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(HHexMaxPool, include_header=hex_max_pool_include_list)
        self.template = hex_max_pool_function_template

    def format(self, node):
        params = {}
        params['config']   = f'config{node.index}'
        params['input_t']  = node.get_input_variable(node.inputs[0]).type.name
        params['output_t'] = node.get_output_variable().type.name
        params['input']    = node.get_input_variable(node.inputs[0]).name
        params['output']   = node.get_output_variable().name
        return self.template.format(**params)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_registered = False


def register_hex_gather_layers(backend_name='Vivado'):
    """Register HexGather and HexRingMAC with hls4ml.

    Call once before ``hls4ml.converters.convert_from_keras_model`` when using
    ``strategy='gather'``.  Idempotent — safe to call multiple times.
    """
    global _registered
    if _registered:
        return

    # Register hls4ml IR layer types
    hls4ml.model.layers.register_layer('HHexGather',   HHexGather)
    hls4ml.model.layers.register_layer('HHexRingMAC',  HHexRingMAC)
    hls4ml.model.layers.register_layer('HHexMaxPool',  HHexMaxPool)

    backend = hls4ml.backends.get_backend(backend_name)

    # Register config + function templates
    backend.register_template(HexGatherConfigTemplate)
    backend.register_template(HexGatherFunctionTemplate)
    backend.register_template(HexRingMACConfigTemplate)
    backend.register_template(HexRingMACFunctionTemplate)
    backend.register_template(HexMaxPoolConfigTemplate)
    backend.register_template(HexMaxPoolFunctionTemplate)

    # Register HLS C++ source files (Phase 4 — must exist alongside this file)
    here = Path(__file__).parent
    for fname in ('nnet_hex_gather.h', 'nnet_hex_ring_mac.h', 'nnet_hex_max_pool.h'):
        p = here / fname
        if p.exists():
            backend.register_source(p)

    _registered = True
