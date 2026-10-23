"""Quantization-aware training (QAT) of the hexagonal layers, with HGQ2 quantizers.

Optional: needs ``hgq2`` (``pip install keras-hexagdly[qat]``). ``import keras_hexagdly`` never
imports this module; the names below are loaded from the package on first use.

Every quantized layer is the hexagonal layer it is named after, plus HGQ2 quantizers that you switch
on one by one with a ``QuantizerConfig``:

``kq_conf``  the kernel, one quantizer per sub-kernel (default: HGQ2's ``"weight"`` config)
``bq_conf``  the bias (default: off)
``iq_conf``  the input of the layer (default: off)
``oq_conf``  the output of the layer, after its activation (default: off)

====================== ========================= ============================================
quantized layer        hexagonal layer           notes
====================== ========================= ============================================
``QConv2d``            ``Conv2d``
``QConv3d``            ``Conv3d``                no hls4ml export in 3D yet
``QConv2d_CustomKernel`` ``Conv2d_CustomKernel`` kernels trained only with ``trainable=True``
``QConv3d_CustomKernel`` ``Conv3d_CustomKernel`` same
``QMaxPool2d``         ``MaxPool2d``             no weights: only ``iq_conf`` / ``oq_conf``
``QMaxPool3d``         ``MaxPool3d``             same
``QAvgPool2d``         ``AvgPool2d``             same; no hls4ml export
====================== ========================= ============================================

The config decides what is learned: fixed bit widths, or bit widths trained with a regulariser that
pushes them down (a weight that reaches 0 bit is removed)::

    from hgq.quantizer import QuantizerConfig
    from hgq.regularizers import MonoL1

    conf = QuantizerConfig("kbi", "weight", k0=1, b0=8, i0=2, br=MonoL1(1e-5))
    x = keras_hexagdly.QConv2d(16, kernel_size=1, kq_conf=conf)(x)

    model.fit(...)
    frozen, types = keras_hexagdly.freeze(model)   # plain layers + the ap_fixed types to use

For an export to hls4ml (``io_stream``) the activation quantizers must be per layer, not per element
(hls4ml only handles per-element masks in ``io_parallel``): give them ``homogeneous_axis`` over
every axis, e.g. ``QuantizerConfig("kif", "datalane", homogeneous_axis=(0, 1, 2, 3),
overflow_mode="SAT")``. A per-element quantizer fixes the spatial shape of the layer at build time.

Mixed precision: the quantized kernels are cast to the compute dtype, which is enough on the torch
and jax backends; on TensorFlow HGQ2's own quantizers do not support ``mixed_float16``.
"""

import math
import re

import keras
import numpy as np

from .layers import (
    AvgPool2d,
    Conv2d,
    Conv2d_CustomKernel,
    Conv3d,
    Conv3d_CustomKernel,
    MaxPool2d,
    MaxPool3d,
    _mapped_inputs,
)

try:
    from hgq.quantizer import Quantizer, QuantizerConfig
except ImportError as exc:  # pragma: no cover - exercised through the package __getattr__
    raise ImportError(
        "keras_hexagdly quantized layers need the optional dependency hgq2 (Python >= 3.10): "
        "pip install 'keras-hexagdly[qat]'"
    ) from exc

__all__ = [
    "QAvgPool2d",
    "QConv2d",
    "QConv2d_CustomKernel",
    "QConv3d",
    "QConv3d_CustomKernel",
    "QMaxPool2d",
    "QMaxPool3d",
    "freeze",
]

_QUANTIZER_ARGS = ("kq_conf", "bq_conf", "iq_conf", "oq_conf")
_PRUNED = "ap_fixed<2,1>"  # what is left to represent when every value was pruned away
# HGQ2 mode names that exist as AP_<name> in ap_types. The stochastic roundings (S_RND, S_RND_CONV)
# are for training only and have no hardware equivalent.
_ROUNDING = ("TRN", "TRN_ZERO", "RND", "RND_ZERO", "RND_INF", "RND_MIN_INF", "RND_CONV")
_OVERFLOW = ("WRAP", "SAT", "SAT_SYM")


def _to_numpy(x):
    """Tensor or variable of any backend (also on a GPU) -> numpy."""
    return keras.ops.convert_to_numpy(x)


def _parse_type(type_string):
    """``(width, integer bits, signed)`` of an ``ap_fixed<W,I...>`` / ``ap_ufixed<W,I...>`` string.

    The integer bits can be negative or larger than the width (``ap_fixed<4,-2>`` is a valid type),
    so the sign of the number is part of it."""
    width, integer = (int(v) for v in re.findall(r"-?\d+", type_string)[:2])
    return width, integer, not type_string.startswith("ap_ufixed")


def _exact_fixed_type(arrays):
    """The smallest signed ``ap_fixed<W,I>`` that holds every value of ``arrays`` exactly.

    Weights and biases are constants, so their type is read off their values (no assumption on how
    the quantizer produced them). Returns ``None`` when they are all zero."""
    values = np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrays])
    values = values[values != 0]
    if values.size == 0:
        return None
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "the weights contain NaN or infinite values: no fixed-point type holds them."
        )
    scaled = lambda f: values * 2.0**f  # noqa: E731 - exact: a power of two
    fractional = -128
    while not np.all(scaled(fractional) == np.round(scaled(fractional))):
        fractional += 1
    low, high = float(values.min()), float(values.max())
    integer = -128
    while not (-(2.0 ** (integer - 1)) <= low and high <= 2.0 ** (integer - 1) - 2.0**-fractional):
        integer += 1
    return f"ap_fixed<{integer + fractional},{integer}>"


def _activation_type(quantizer):
    """The ``ap_fixed`` / ``ap_ufixed`` type of what an activation quantizer can output.

    An element is alive when ``k + i + f > 0``, which is the mask HGQ2 itself applies at inference
    (a dead element outputs 0). The type carries the rounding and overflow modes
    (``ap_fixed<W,I,AP_RND,AP_SAT>``), and is an ``ap_ufixed`` when the quantizer is unsigned
    everywhere (the output of a ReLU). ``None`` when every element is dead.
    """
    inner = quantizer.quantizer
    if not hasattr(inner, "kif"):
        raise ValueError(
            f"A type can only be deduced from a fixed-point quantizer (kbi or kif), "
            f"not {type(inner).__name__}."
        )
    k, i, f = (np.round(_to_numpy(v)) for v in inner.kif)
    shape = np.broadcast_shapes(k.shape, i.shape, f.shape)
    k, i, f = (np.broadcast_to(v, shape) for v in (k, i, f))
    alive = (k + i + f) > 0
    if not alive.any():
        return None
    rounding, overflow = inner.round_mode, inner.overflow_mode
    if rounding not in _ROUNDING or overflow not in _OVERFLOW:
        raise ValueError(
            f"The rounding mode {rounding!r} / overflow mode {overflow!r} of this quantizer has no "
            f"ap_fixed equivalent (supported: {', '.join(_ROUNDING)} / {', '.join(_OVERFLOW)}); "
            "the stochastic roundings S_RND and S_RND_CONV are for training only."
        )
    signed = bool((k[alive] > 0).any())
    integer = int(i[alive].max()) + (1 if signed else 0)
    fractional = int(f[alive].max())
    kind = "ap_fixed" if signed else "ap_ufixed"
    return f"{kind}<{integer + fractional},{integer},AP_{rounding},AP_{overflow}>"


class _Quantized:
    """What the quantized layers share. Goes before the hexagonal layer in the bases.

    ``_plain_class`` is the layer ``freeze`` turns it back into; ``_has_weights`` is False for
    pooling.
    """

    #: read by ``patch_model_for_hls`` to refuse a model that was not frozen
    _hexagdly_qat = True
    _plain_class = None
    _has_weights = True

    def __init__(self, *args, kq_conf=None, bq_conf=None, iq_conf=None, oq_conf=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._has_weights and (kq_conf is not None or bq_conf is not None):
            raise ValueError(
                f"{type(self).__name__} has no weights: only iq_conf and oq_conf apply to it."
            )
        if bq_conf is not None and not getattr(self, "use_bias", False):
            raise ValueError(
                "bq_conf needs a bias: build the layer with use_bias=True (for the custom-kernel "
                "layers, give a bias array)."
            )
        default = QuantizerConfig("default", "weight")
        self.kq_conf = (default if kq_conf is None else kq_conf) if self._has_weights else None
        self.bq_conf, self.iq_conf, self.oq_conf = bq_conf, iq_conf, oq_conf
        # the quantizers themselves exist once the layer is built
        self._kqs, self._bq, self._iq, self._oq = [], None, None, None

    # -- the tensors that are quantized ---------------------------------------------------------
    def _kernel_sources(self):
        """The trainable kernel variables, in the order of their quantizers."""
        if not self._has_weights:
            return []
        return [self.ring_weights] if self.share_neighbors else list(self._base_kernels)

    def _ring_axis(self):
        return 0

    def _quantize(self, quantizer, variable):
        """The quantized variable, in the compute dtype of the layer (mixed precision)."""
        return keras.ops.cast(quantizer(variable), self.compute_dtype)

    def _materialize_kernels(self):
        """Same as the hexagonal layer's, but the kernels (and the bias) are the quantized ones."""
        if self.share_neighbors:
            ring = self._quantize(self._kqs[0], self.ring_weights)
            for i in range(self.hexbase_size + 1):
                kernel = keras.ops.take(ring, self._ring_idx[i], axis=self._ring_axis())
                setattr(self, "kernel" + str(i), kernel)
        else:
            for i in range(self.hexbase_size + 1):
                setattr(
                    self, "kernel" + str(i), self._quantize(self._kqs[i], self._base_kernels[i])
                )
        if self._bq is not None:
            self.kwargs = {"bias": self._quantize(self._bq, self.bias_tensor)}

    # -- build / call -----------------------------------------------------------------------------
    def build(self, input_shape):
        super().build(input_shape)
        sources = self._kernel_sources()
        self._kqs = [
            Quantizer(self.kq_conf, name=f"{self.name}_kq{i}") for i in range(len(sources))
        ]
        for quantizer, source in zip(self._kqs, sources):
            quantizer.build(keras.ops.shape(source))
        self._bq = None
        if self.bq_conf is not None:
            self._bq = Quantizer(self.bq_conf, name=f"{self.name}_bq")
            self._bq.build(keras.ops.shape(self.bias_tensor))
        self._iq = None
        if self.iq_conf is not None:
            self._iq = Quantizer(self.iq_conf, name=f"{self.name}_iq")
            self._iq.build(tuple(input_shape))
        self._oq = None
        if self.oq_conf is not None:
            self._oq = Quantizer(self.oq_conf, name=f"{self.name}_oq")
            self._oq.build(self._output_shape(input_shape))

    def _output_shape(self, input_shape):
        """The shape of the layer output (batch left open), found by running a one-sample probe."""
        self._materialize_kernels()
        probe = keras.ops.zeros((1,) + tuple(input_shape[1:]), dtype=self.compute_dtype)
        out = super().call(probe)
        return (None,) + tuple(out.shape[1:])

    def call(self, input):
        if self._iq is not None:
            input = self._iq(input)
        out = super().call(input)
        if self._oq is not None:
            out = self._oq(out)
        return out

    # -- the numbers a deployment needs ---------------------------------------------------
    def quantized_weights(self):
        """The quantized kernel values, as numpy arrays: one per sub-kernel (or the ring
        weights)."""
        return [_to_numpy(q(w)) for q, w in zip(self._kqs, self._kernel_sources())]

    def quantized_bias(self):
        """The quantized bias as a numpy array (the float bias when ``bq_conf`` is not set)."""
        if not self._has_weights or not self.use_bias:
            return None
        bias = self.bias_tensor
        return _to_numpy(self._bq(bias) if self._bq is not None else bias)

    def weight_type(self):
        """The ``ap_fixed<W,I>`` type that holds every quantized kernel weight exactly (for hls4ml).

        It is read off the values: the smallest signed type that represents all of them (``I``
        integer bits including the sign, which can be negative for small weights), so it follows
        what the quantizers produced, pruned weights included. ``ap_fixed<2,1>`` when every
        weight was pruned. ``None`` for a pooling layer.
        """
        if not self._has_weights:
            return None
        return _exact_fixed_type(self.quantized_weights()) or _PRUNED

    def bias_type(self):
        """The ``ap_fixed<W,I>`` type that holds the quantized bias exactly, or ``None`` without
        ``bq_conf``."""
        if self._bq is None:
            return None
        return _exact_fixed_type([self.quantized_bias()]) or _PRUNED

    def accum_type(self, input_type=None):
        """The accumulator type that holds the sum of the weight * input products plus the bias.

        For hls4ml: ``config["LayerName"][<patched layer>]["AccumPrecision"] = layer.accum_type()``
        (or ``config["Model"]["ExactAccumulator"] = True``, which builds the same type). The
        exported kernel otherwise keeps only the fractional bits of the finer of the weights and the
        input, which drops the low bits of each product: with narrow quantized inputs the C
        simulation then differs from the Keras model. Here the fractional bits are those of the
        exact product, ``f_weight + f_input`` (or the bias's, if finer), and the integer bits are
        ``max(I_weight + I_input + ceil(log2(terms)), I_bias) + 1``: the last bit is the carry of
        the bias addition.

        ``input_type`` is the ``ap_fixed`` type of the layer input; by default that of ``iq_conf``.
        Pass the result type of the previous layer when there is no input quantizer. The bias is
        part of the bound when ``bq_conf`` is set.
        """
        if not self._has_weights:
            return None
        if not self.built:
            raise ValueError("accum_type() needs a built layer: call it on a tensor first.")
        input_type = input_type or self.input_type()
        if input_type is None:
            raise ValueError(
                "accum_type() needs the type of the layer input: set iq_conf or pass input_type."
            )
        weight_width, weight_int, _ = _parse_type(self.weight_type())
        input_width, input_int, _ = _parse_type(input_type)
        radius = self.hexbase_size
        depth = getattr(self, "depth_size", None) or 1  # the 2D layers define it as None
        terms = (1 + 3 * radius * (radius + 1)) * self.in_channels * depth
        integer = weight_int + input_int + int(math.ceil(math.log2(max(1, terms))))
        fractional = (weight_width - weight_int) + (input_width - input_int)
        bias_type = self.bias_type()
        if bias_type is not None:
            bias_width, bias_int, _ = _parse_type(bias_type)
            integer = max(integer, bias_int)
            fractional = max(fractional, bias_width - bias_int)
        integer += 1  # the carry of the bias addition
        return f"ap_fixed<{integer + fractional},{integer}>"

    def input_type(self):
        """The ``ap_fixed<W,I,round,overflow>`` type of the input quantizer, or ``None``."""
        return self._activation_type_of(self._iq)

    def result_type(self):
        """The ``ap_fixed<W,I,round,overflow>`` type of the output quantizer, or ``None``."""
        return self._activation_type_of(self._oq)

    @staticmethod
    def _activation_type_of(quantizer):
        if quantizer is None:
            return None
        return _activation_type(quantizer) or _PRUNED

    # -- serialization --------------------------------------------------------------------
    def get_config(self):
        config = super().get_config()
        for name in _QUANTIZER_ARGS:
            conf = getattr(self, name)
            # serialized: JSON for h5, and from_config(get_config()) / clone_model work
            config[name] = None if conf is None else keras.saving.serialize_keras_object(conf)
        return config

    @classmethod
    def from_config(cls, config):
        # the quantizer configs are stored serialized: turn them back into HGQ2 QuantizerConfig
        # objects, as HGQ2's own layers do
        return super().from_config(keras.saving.deserialize_keras_object(config))


@keras.saving.register_keras_serializable(package="hexagdly_tf")
class QConv2d(_Quantized, Conv2d):
    """:class:`~keras_hexagdly.Conv2d` with HGQ2 quantizers (see the module docstring).

    Takes every argument of ``Conv2d`` plus ``kq_conf`` (default: ``QuantizerConfig("default",
    "weight")``), ``bq_conf``, ``iq_conf`` and ``oq_conf``. With ``share_neighbors`` the shared
    weights (``ring_weights``) are quantized, then spread over the cells of their group. The
    quantizer variables and the configs are saved with the model, so ``model.save()`` /
    ``keras.models.load_model()`` keep the quantization.
    """

    _plain_class = Conv2d


@keras.saving.register_keras_serializable(package="hexagdly_tf")
class QConv3d(_Quantized, Conv3d):
    """:class:`~keras_hexagdly.Conv3d` with HGQ2 quantizers (see :class:`QConv2d`).

    With ``share_neighbors`` the weights are shared over the hexagonal axes only: the quantized
    ``ring_weights`` keep an independent weight per depth (time) step.
    """

    _plain_class = Conv3d

    def _ring_axis(self):
        return 1


class _QuantizedCustomKernel(_Quantized):
    """The kernels of a ``*_CustomKernel`` layer are variables set once, named ``kernel0``,
    ``kernel1``... They are moved to ``_base_kernels``, which frees these names for the quantized
    tensors of each call (the base ``call`` does not derive its kernels, so it is done here)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.share_neighbors = False
        kernels = [getattr(self, f"kernel{i}") for i in range(self.hexbase_size + 1)]
        for i in range(self.hexbase_size + 1):
            delattr(self, f"kernel{i}")  # this also untracks the variable...
        self._base_kernels = kernels  # ...so it is tracked again here, once the names are free

    def call(self, input):
        self._materialize_kernels()
        return super().call(input)


@keras.saving.register_keras_serializable(package="hexagdly_tf")
class QConv2d_CustomKernel(_QuantizedCustomKernel, Conv2d_CustomKernel):
    """:class:`~keras_hexagdly.Conv2d_CustomKernel` with HGQ2 quantizers.

    The kernels you supply are quantized at every call. They are only trained with
    ``trainable=True`` (the default of the layer is ``False``). That flag is about the kernels and
    the bias: the bit widths of the quantizers are trained whenever their ``QuantizerConfig`` says
    so, even with fixed kernels (give it ``trainable=False`` for plain rounding). The weights of
    the layer are ordered ``bias, kernel0, kernel1, ...`` and then the quantizers, which is not the
    order of the plain layer: copy them by name, or with ``freeze``.
    """

    _plain_class = Conv2d_CustomKernel


@keras.saving.register_keras_serializable(package="hexagdly_tf")
class QConv3d_CustomKernel(_QuantizedCustomKernel, Conv3d_CustomKernel):
    """:class:`~keras_hexagdly.Conv3d_CustomKernel` with HGQ2 quantizers (see the 2D version)."""

    _plain_class = Conv3d_CustomKernel


class _QuantizedPool(_Quantized):
    """A pooling layer has no weights: only its input and output can be quantized."""

    _has_weights = False

    def _materialize_kernels(self):
        pass


@keras.saving.register_keras_serializable(package="hexagdly_tf")
class QMaxPool2d(_QuantizedPool, MaxPool2d):
    """:class:`~keras_hexagdly.MaxPool2d` with quantizers on its input and output (``iq_conf``,
    ``oq_conf``). A max of quantized values is already on their grid: ``oq_conf`` only changes
    something when the input is not quantized."""

    _plain_class = MaxPool2d


@keras.saving.register_keras_serializable(package="hexagdly_tf")
class QMaxPool3d(_QuantizedPool, MaxPool3d):
    """:class:`~keras_hexagdly.MaxPool3d` with quantizers on its input and output."""

    _plain_class = MaxPool3d


@keras.saving.register_keras_serializable(package="hexagdly_tf")
class QAvgPool2d(_QuantizedPool, AvgPool2d):
    """:class:`~keras_hexagdly.AvgPool2d` with quantizers on its input and output. A mean of
    quantized values needs more bits than they have, so ``oq_conf`` is what bounds it. Like the
    plain layer, it has no hls4ml export."""

    _plain_class = AvgPool2d


def _copy_layer(layer):
    """A new layer of the same class and config (through Keras serialization, which also handles
    HGQ2's own layers, whose ``get_config`` holds objects)."""
    return keras.saving.deserialize_keras_object(keras.saving.serialize_keras_object(layer))


def _frozen_quantizer(quantizer, name):
    """A standalone, non-trainable copy of an activation quantizer; its learned bit widths are
    copied once it is built."""
    serialized = keras.saving.serialize_keras_object(quantizer)
    serialized["config"]["name"] = name
    copy = keras.saving.deserialize_keras_object(serialized)
    copy.trainable = False
    return copy


def freeze(model, activation_quantizers="keep"):
    """Turn the quantized layers of ``model`` into plain hexagonal layers holding the quantized
    values.

    Returns ``(frozen_model, types)``. ``types`` is ``{layer name: {...}}`` with, for each quantized
    layer, the ``ap_fixed`` type to give hls4ml for what was quantized: ``"weight"``
    (``mac_weights``), ``"bias"`` (``mac_bias``), ``"input"`` and ``"result"``, and ``"accum"``
    (the exact accumulator, only when the layer has an input quantizer, which fixes the input
    type). Layers of a nested model are listed as ``"<nested name>/<layer name>"``. The frozen model
    goes through ``patch_model_for_hls`` and hls4ml like any other. ``model`` is not modified.

    The quantized layers of the frozen model (and the quantizers it keeps) are not trainable: it is
    for checking and exporting, not for training.

    ``activation_quantizers`` decides what becomes of the input / output quantizers:

    * ``"keep"`` (default): they stay in the frozen model as standalone HGQ2 ``Quantizer`` layers,
      so it computes exactly what ``model`` computes -- the way to check the quantization in Keras.
      hls4ml cannot convert them in ``io_stream``.
    * ``"types"``: they are dropped; their precision is in ``types`` (``"input"`` / ``"result"``),
      to be set in the hls4ml config. The frozen model is then the float-activation model, and
      equals ``model`` only where the activations are already on their grid.

    Works on built Functional and Sequential models (nested ones included) whose layers are each
    called once; the structure of the outputs (a dict, for instance) is kept.
    """
    if activation_quantizers not in ("keep", "types"):
        raise ValueError("activation_quantizers must be 'keep' or 'types'.")
    return _freeze(model, activation_quantizers == "keep", "_frozen")


def _freeze(model, keep, suffix):
    if getattr(model, "inputs", None) is None:
        raise ValueError(
            f"freeze needs a built Functional or Sequential model ({model.name!r} has no inputs "
            "yet: give it an Input or call it once)."
        )
    types, tensor_map = {}, {id(t): t for t in model.inputs}

    for layer in model.layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue
        x = _mapped_inputs(layer, tensor_map)
        if isinstance(layer, keras.Model):  # a nested model: freeze what it contains
            new, inner = _freeze(layer, keep, "")
            types.update({f"{layer.name}/{name}": entry for name, entry in inner.items()})
            y = new(x)
        elif not isinstance(layer, _Quantized):
            new = _copy_layer(layer)
            y = new(x)
            new.set_weights(layer.get_weights())
        else:
            if keep and layer._iq is not None:
                iq = _frozen_quantizer(layer._iq, f"{layer.name}_iq")
                x = iq(x)
                iq.set_weights(layer._iq.get_weights())
            config = layer.get_config()
            for name in _QUANTIZER_ARGS:
                config.pop(name)
            new = layer._plain_class.from_config(config)
            y = new(x)
            _copy_quantized_values(layer, new)
            new.trainable = False  # its weights are on the quantized grid: training would leave it
            if keep and layer._oq is not None:
                oq = _frozen_quantizer(layer._oq, f"{layer.name}_oq")
                y = oq(y)
                oq.set_weights(layer._oq.get_weights())
            entry = {
                "weight": layer.weight_type(),
                "bias": layer.bias_type(),
                "input": layer.input_type(),
                "result": layer.result_type(),
                "accum": layer.accum_type()
                if layer._has_weights and layer._iq is not None
                else None,
            }
            types[layer.name] = {k: v for k, v in entry.items() if v is not None}
        raw_out = layer.output
        if isinstance(raw_out, (list, tuple)):
            for orig, new_t in zip(raw_out, y if isinstance(y, (list, tuple)) else [y]):
                tensor_map[id(orig)] = new_t
        else:
            tensor_map[id(raw_out)] = y
    inputs, outputs = _graph_structure(model)
    outputs = keras.tree.map_structure(lambda t: tensor_map[id(t)], outputs)
    return keras.Model(inputs, outputs, name=model.name + suffix), types


def _graph_structure(model):
    """The inputs and outputs of the model's own graph, with the structure the user gave them.

    A Sequential that was called as a layer of another model reports, in ``.input`` / ``.output``,
    the tensors of that call, not those of the graph its layers live in: ``.inputs`` / ``.outputs``
    are the ones to read there."""
    if isinstance(model, keras.Sequential):
        inputs, outputs = model.inputs, model.outputs
        return inputs[0] if len(inputs) == 1 else inputs, outputs[0] if len(
            outputs
        ) == 1 else outputs
    return model.input, model.output


def _copy_quantized_values(quantized, plain):
    """The quantized kernels and bias of ``quantized`` become the weights of ``plain``."""
    for target, value in zip(_plain_kernels(plain), quantized.quantized_weights()):
        target.assign(value)
    if quantized._has_weights and quantized.use_bias:
        plain.bias_tensor.assign(quantized.quantized_bias())


def _plain_kernels(plain):
    """The kernel variables of a plain hexagonal layer, in the order of the quantizers."""
    if getattr(plain, "share_neighbors", False):
        return [plain.ring_weights]
    if hasattr(plain, "_base_kernels"):
        return list(plain._base_kernels)
    if not hasattr(plain, "hexbase_size") or isinstance(plain, (MaxPool2d, MaxPool3d, AvgPool2d)):
        return []
    return [getattr(plain, f"kernel{i}") for i in range(plain.hexbase_size + 1)]
