"""QConv2d / freeze: quantization-aware training of the hex convolution with HGQ2 (optional)."""

import json
import re
import sys

import keras
import numpy as np
import pytest

import keras_hexagdly as hgly
from keras_hexagdly.quantized import _parse_type

pytest.importorskip("hgq")
from hgq.quantizer import Quantizer, QuantizerConfig  # noqa: E402
from hgq.regularizers import MonoL1  # noqa: E402

H, W, C = 9, 8, 3
np_ = keras.ops.convert_to_numpy  # any backend, GPU tensors included
X = np.random.default_rng(0).standard_normal((4, H, W, C)).astype("float32")
needs_hls = pytest.mark.skipif(
    keras.backend.backend() == "jax",
    reason="the hls4ml export targets the tensorflow / torch backends",
)


def fixed(bits=8, ints=2):
    """Sign + `ints` integer bits + (bits - ints) fractional bits, not trained."""
    return QuantizerConfig("kbi", "weight", k0=1, b0=bits, i0=ints, trainable=False)


def fixed_k(k0, bits=8, ints=2):
    return QuantizerConfig("kbi", "weight", k0=k0, b0=bits, i0=ints, trainable=False)


def holds(type_string, values):
    """Every value is exactly representable in the ap_fixed / ap_ufixed type (modes ignored)."""
    width, integer, signed = _parse_type(type_string)
    fractional = width - integer
    values = np.asarray(values, dtype=np.float64)
    scaled = values * 2.0**fractional
    low = -(2.0 ** (integer - 1)) if signed else 0.0
    high = (2.0 ** (integer - 1) if signed else 2.0**integer) - 2.0**-fractional
    return bool(
        np.all(scaled == np.round(scaled)) and np.all(values >= low) and np.all(values <= high)
    )


def is_tight(type_string, values):
    """No smaller signed type holds the values: one bit less, fractional or integer, fails."""
    width, integer, _ = _parse_type(type_string)
    return not holds(f"ap_fixed<{width - 1},{integer}>", values) and not holds(
        f"ap_fixed<{width - 1},{integer - 1}>", values
    )


def copy_weights(src, dst):
    """Give the QAT layer the exact float weights of a plain layer."""
    for a, b in zip(
        [dst.ring_weights] if dst.share_neighbors else dst._base_kernels,
        [src.ring_weights] if src.share_neighbors else src._base_kernels,
    ):
        a.assign(b)
    if src.use_bias:
        dst.bias_tensor.assign(src.bias_tensor)


# every sharing mode, with the kernel sizes it is defined for (diag and sym: 1 and 2 only)
MODES = [(False, k) for k in (0, 1, 2)] + [("ring", k) for k in (0, 1, 2, 3)]
MODES += [(m, k) for m in ("diag", "sym") for k in (1, 2)]


@pytest.mark.parametrize("strides", [1, 2])
@pytest.mark.parametrize("share, kernel_size", MODES)
def test_many_bits_gives_back_the_float_layer(kernel_size, share, strides):
    kw = dict(kernel_size=kernel_size, share_neighbors=share, strides=strides, activation="relu")
    plain = hgly.Conv2d(4, **kw)
    expected = np_(plain(X))
    qat = hgly.QConv2d(4, kq_conf=fixed(bits=24, ints=4), **kw)
    qat(X)
    copy_weights(plain, qat)
    np.testing.assert_allclose(np_(qat(X)), expected, atol=1e-4)


def test_the_weights_are_on_the_fixed_point_grid():
    layer = hgly.QConv2d(4, kernel_size=1, kq_conf=fixed(bits=8, ints=2))
    layer(X)
    step = 2.0**-6  # 8 bits, 2 of them integer: 6 fractional bits
    for weights in layer.quantized_weights():
        np.testing.assert_allclose(weights / step, np.round(weights / step), atol=1e-6)
        assert np.abs(weights).max() <= 4.0


@pytest.mark.parametrize("bits, ints", [(8, 2), (6, 1), (4, 0), (5, -2), (3, -3)])
def test_weight_type_holds_the_quantized_weights_exactly_and_no_wider(bits, ints):
    """Includes negative integer bits (small weights, what training with HGQ2 produces): a type
    like ap_fixed<5,-2> must keep its sign."""
    layer = hgly.QConv2d(4, kernel_size=1, kq_conf=fixed(bits, ints))
    layer(X)
    values = np.concatenate([w.ravel() for w in layer.quantized_weights()])
    kind = layer.weight_type()
    assert holds(kind, values), (kind, values.min(), values.max())
    assert is_tight(kind, values), kind
    if ints < 0:
        assert _parse_type(kind)[1] < 1  # the integer width really went through the string


def test_weight_type_of_a_partly_pruned_layer_follows_the_weights_that_are_left():
    layer = hgly.QConv2d(4, kernel_size=1, kq_conf=fixed(8, 2))
    layer(X)
    for weight in layer._base_kernels:
        mask = np.zeros(weight.shape, "float32")
        mask.flat[:3] = 1  # three weights are left, the rest is pruned to 0
        weight.assign(np_(weight) * mask)
    values = np.concatenate([w.ravel() for w in layer.quantized_weights()])
    assert holds(layer.weight_type(), values) and is_tight(layer.weight_type(), values)


@pytest.mark.parametrize("k0", [1, 0])
def test_weight_type_holds_the_largest_weight_signed_or_not(k0):
    """ap_fixed is signed: an unsigned quantizer (k0=0, range [0, 2^i)) needs one more bit."""
    layer = hgly.QConv2d(4, kernel_size=1, kq_conf=fixed_k(k0, bits=6, ints=2))
    layer(X)
    for weight in layer._base_kernels:
        weight.assign(np.full(weight.shape, 3.9, "float32"))  # the top of the range
    width, integer = map(int, re.findall(r"\d+", layer.weight_type()))
    largest = max(float(w.max()) for w in layer.quantized_weights())
    assert largest == 3.875 and largest <= 2.0 ** (integer - 1) - 2.0 ** -(width - integer)


def test_the_package_exports_are_importable_by_name():
    from keras_hexagdly import QConv2d, freeze

    assert QConv2d is hgly.QConv2d and freeze is hgly.freeze


def test_a_zero_bit_weight_is_removed():
    layer = hgly.QConv2d(4, kernel_size=1, kq_conf=fixed(bits=0, ints=0))
    layer(X)
    assert all((w == 0).all() for w in layer.quantized_weights())
    assert layer.weight_type() == "ap_fixed<2,1>"  # nothing left to represent


def qat_model(conf):
    inp = keras.Input((H, W, C))
    x = keras.layers.ReLU()(hgly.QConv2d(4, kernel_size=1, kq_conf=conf, name="hex0")(inp))
    x = keras.layers.ReLU()(hgly.QConv2d(4, kernel_size=1, strides=2, kq_conf=conf, name="hex1")(x))
    return keras.Model(inp, keras.layers.Dense(1)(keras.layers.Flatten()(x)))


def test_training_moves_the_weights_and_the_bit_widths():
    conf = QuantizerConfig("kbi", "weight", k0=1, b0=8, i0=2, br=MonoL1(1e-3))
    model = qat_model(conf)
    assert len(model.losses) > 0  # the bit regulariser is part of the loss
    before = [np_(v) for v in model.trainable_variables]
    model.compile(keras.optimizers.Adam(0.05), "mse")
    model.fit(X, np.ones((len(X), 1), "float32"), epochs=30, verbose=0)
    after = [np_(v) for v in model.trainable_variables]
    assert sum(not np.allclose(a, b) for a, b in zip(before, after)) >= len(before) // 2
    layer = model.get_layer("hex0")
    bits = [np_(q.bits_(k.shape)).mean() for q, k in zip(layer._kqs, layer._base_kernels)]
    assert max(bits) < 7.5  # the regulariser pushed the bit widths down


@pytest.mark.parametrize("share, kernel_size", [(False, 1), ("ring", 2), ("diag", 2), ("sym", 1)])
def test_save_and_load_keep_the_quantization(tmp_path, share, kernel_size):
    inp = keras.Input((H, W, C))
    layer = hgly.QConv2d(
        4,
        kernel_size=kernel_size,
        share_neighbors=share,
        activation="relu",
        kq_conf=fixed(bits=6, ints=2),
        name="hex0",
    )
    model = keras.Model(inp, layer(inp))
    expected = np_(model(X))
    model.save(tmp_path / "m.keras")
    loaded = keras.models.load_model(tmp_path / "m.keras")
    assert isinstance(loaded.get_layer("hex0"), hgly.QConv2d)
    np.testing.assert_allclose(np_(loaded(X)), expected, atol=1e-6)
    assert loaded.get_layer("hex0").weight_type() == layer.weight_type()


def test_save_and_load_keep_the_quantizer_config(tmp_path):
    conf = QuantizerConfig("kbi", "weight", k0=1, b0=6, i0=1, br=MonoL1(1e-3))
    inp = keras.Input((H, W, C))
    model = keras.Model(inp, hgly.QConv2d(4, kernel_size=1, kq_conf=conf, name="hex0")(inp))
    model.save(tmp_path / "m.keras")
    loaded = keras.models.load_model(tmp_path / "m.keras")
    loaded_conf = loaded.get_layer("hex0").kq_conf
    assert (loaded_conf["b0"], loaded_conf["i0"], loaded_conf["k0"]) == (6, 1, 1)
    assert isinstance(loaded_conf["br"], MonoL1)
    assert len(loaded.losses) == len(model.losses) > 0  # the bit regulariser is still there


@pytest.mark.parametrize("share, kernel_size", [(False, 1), ("ring", 2), ("diag", 2), ("sym", 2)])
def test_freeze_computes_the_same_function(share, kernel_size):
    inp = keras.Input((H, W, C))
    x = hgly.QConv2d(
        4, kernel_size=kernel_size, share_neighbors=share, kq_conf=fixed(5, 2), name="hex0"
    )(inp)
    model = keras.Model(inp, keras.layers.Dense(2)(keras.layers.Flatten()(keras.layers.ReLU()(x))))
    model.compile("adam", "mse")
    model.fit(X, np.ones((len(X), 2), "float32"), epochs=2, verbose=0)

    frozen, types = hgly.freeze(model)
    np.testing.assert_allclose(np_(frozen(X)), np_(model(X)), atol=1e-6)
    assert type(frozen.get_layer("hex0")) is hgly.Conv2d  # a plain layer, not a QConv2d
    assert isinstance(model.get_layer("hex0"), hgly.QConv2d)  # the original is untouched
    assert list(types) == ["hex0"] and re.fullmatch(r"ap_fixed<\d+,\d+>", types["hex0"]["weight"])


@needs_hls
def test_the_frozen_model_goes_through_patch_model_for_hls():
    pytest.importorskip("hls4ml")
    from keras_hexagdly.hls4ml_ext import patch_model_for_hls

    model = qat_model(fixed(8, 2))
    frozen, _ = hgly.freeze(model)
    patched = patch_model_for_hls(frozen)
    np.testing.assert_allclose(np_(patched(X)), np_(model(X)), atol=1e-4)

    with pytest.raises(ValueError, match="freeze"):
        patch_model_for_hls(model)  # a QAT model left as it is would not be exported as quantized


def test_the_even_size_warning_points_at_the_line_that_built_the_layer():
    with pytest.warns(UserWarning, match="no even size") as record:
        hgly.QConv2d(4, kernel_size=2, kernel_size_unit="pixels")
    assert record[0].filename == __file__  # not keras_hexagdly/quantized.py


def test_the_package_imports_without_hgq(monkeypatch):
    monkeypatch.setitem(sys.modules, "hgq", None)
    # an already imported submodule would shadow the block above
    monkeypatch.setitem(sys.modules, "hgq.quantizer", None)
    monkeypatch.delitem(sys.modules, "keras_hexagdly.quantized", raising=False)
    monkeypatch.delattr(hgly, "quantized", raising=False)
    assert hgly.Conv2d is not None  # the rest of the package does not need hgq2
    with pytest.raises(ImportError, match=r"keras-hexagdly\[qat\]"):
        hgly.QConv2d


# ---- bias, input and output quantizers ----


def act(ints, fracs, k=1):
    """A per-layer (not per-element) activation quantizer: sign + `ints` integer + `fracs` bits."""
    return QuantizerConfig(
        "kif",
        "datalane",
        k0=k,
        i0=ints,
        f0=fracs,
        overflow_mode="SAT",
        round_mode="RND",
        homogeneous_axis=(0, 1, 2, 3),
        trainable=False,
    )


def standalone(conf, shape, name):
    quantizer = Quantizer(conf, name=name)
    quantizer.build(shape)
    return quantizer


def test_the_bias_quantizer_puts_the_bias_on_its_grid():
    layer = hgly.QConv2d(4, kernel_size=1, kq_conf=fixed(24, 4), bq_conf=fixed(6, 1))
    layer(X)
    layer.bias_tensor.assign(np.array([0.3, -0.7, 1.1, 0.04], "float32"))
    bias = layer.quantized_bias()
    np.testing.assert_allclose(bias * 2.0**5, np.round(bias * 2.0**5), atol=1e-6)
    assert holds(layer.bias_type(), bias) and is_tight(layer.bias_type(), bias)
    assert hgly.QConv2d(4, kernel_size=1).bias_type() is None  # off unless asked for


def test_the_bias_quantizer_needs_a_bias():
    with pytest.raises(ValueError, match="needs a bias"):
        hgly.QConv2d(4, kernel_size=1, use_bias=False, bq_conf=fixed(6, 1))
    with pytest.raises(ValueError, match="needs a bias"):
        hgly.QConv2d_CustomKernel(sub_kernels=SUB_2D, kq_conf=fixed(6, 1), bq_conf=fixed(5, 1))


def test_input_and_output_quantizers_wrap_the_layer_and_its_activation():
    """oq(layer(iq(x))): the output quantizer sees the output of the ReLU."""
    plain = hgly.Conv2d(4, kernel_size=1, activation="relu")
    plain(X)
    layer = hgly.QConv2d(
        4,
        kernel_size=1,
        activation="relu",
        kq_conf=fixed(24, 4),
        iq_conf=act(2, 3),
        oq_conf=act(2, 3, k=0),
    )
    layer(X)
    copy_weights(plain, layer)
    iq = standalone(act(2, 3), (None, H, W, C), "m_iq")
    oq = standalone(act(2, 3, k=0), (None, H, W, 4), "m_oq")
    expected = np_(oq(plain(iq(X))))
    np.testing.assert_allclose(np_(layer(X)), expected, atol=1e-4)
    assert np_(layer(X)).min() >= 0  # the unsigned output quantizer sits after the ReLU


def test_activation_types_carry_the_modes_and_the_sign():
    layer = hgly.QConv2d(
        4, kernel_size=1, activation="relu", iq_conf=act(2, 3), oq_conf=act(3, 2, k=0)
    )
    layer(X)
    assert layer.input_type() == "ap_fixed<6,3,AP_RND,AP_SAT>"
    assert layer.result_type() == "ap_ufixed<5,3,AP_RND,AP_SAT>"  # unsigned: no sign bit
    assert hgly.QConv2d(4, kernel_size=1).result_type() is None


def test_per_element_activation_quantizers_run_and_train_in_keras():
    """HGQ2's default datalane quantizer has one bit width per element (not exportable to
    io_stream, but valid for training)."""
    conf = QuantizerConfig("kif", "datalane")
    inp = keras.Input((H, W, C))
    layer = hgly.QConv2d(4, kernel_size=1, oq_conf=conf, name="hex0")
    model = keras.Model(inp, keras.layers.Dense(1)(keras.layers.Flatten()(layer(inp))))
    model.compile("adam", "mse")
    model.fit(X, np.ones((len(X), 1), "float32"), epochs=2, verbose=0)
    assert layer.result_type().startswith("ap_fixed<")


def accumulator_worst_case(layer, input_type):
    """The largest magnitude the accumulator can reach, from the extreme values of the types."""
    _, weight_int, _ = _parse_type(layer.weight_type())
    _, input_int, input_signed = _parse_type(input_type)
    radius = layer.hexbase_size
    terms = (
        (1 + 3 * radius * (radius + 1))
        * layer.in_channels
        * (getattr(layer, "depth_size", None) or 1)
    )
    largest_weight = 2.0 ** (weight_int - 1)
    largest_input = 2.0 ** (input_int - 1) if input_signed else 2.0**input_int
    bias = layer.bias_type()
    largest_bias = 2.0 ** (_parse_type(bias)[1] - 1) if bias else 0.0
    return terms * largest_weight * largest_input + largest_bias


ACCUM_CASES = [
    # (kernel conf, bias conf or None, input conf, kernel_size)
    (fixed(6, 1), fixed(5, 1), act(2, 3), 1),
    (fixed(6, 1), fixed(5, 1), act(3, 2, k=0), 1),  # unsigned input (what a ReLU feeds)
    (fixed(8, 2), None, act(2, 3), 2),
    (fixed(3, 1), fixed(10, 2), act(2, 1), 1),  # a bias finer than the product
    (fixed(5, -2), fixed(8, 3), act(3, 2, k=0), 2),  # small weights, negative integer bits
]


@pytest.mark.parametrize("kq, bq, iq, kernel_size", ACCUM_CASES)
def test_accum_type_holds_the_worst_case_sum_and_every_fractional_bit(kq, bq, iq, kernel_size):
    layer = hgly.QConv2d(4, kernel_size=kernel_size, kq_conf=kq, bq_conf=bq, iq_conf=iq)
    layer(X)
    layer.bias_tensor.assign(np.full((4,), 3.5, "float32"))  # a bias as large as the products
    accum = layer.accum_type()
    width, integer, _ = _parse_type(accum)
    # the largest possible sum, products and bias, fits (the bias addition carry included)
    assert accumulator_worst_case(layer, layer.input_type()) <= 2.0 ** (integer - 1)
    fractional = width - integer
    weight_width, weight_int, _ = _parse_type(layer.weight_type())
    input_width, input_int, _ = _parse_type(layer.input_type())
    product_fraction = (weight_width - weight_int) + (input_width - input_int)
    bias = layer.bias_type()
    bias_fraction = _parse_type(bias)[0] - _parse_type(bias)[1] if bias else 0
    assert fractional == max(product_fraction, bias_fraction)  # nothing is dropped


def test_accum_type_needs_a_built_layer_and_an_input_type():
    unquantized = hgly.QConv2d(4, kernel_size=1)
    with pytest.raises(ValueError, match="built layer"):
        unquantized.accum_type()
    unquantized(X)
    with pytest.raises(ValueError, match="iq_conf"):
        unquantized.accum_type()
    assert unquantized.accum_type("ap_fixed<8,4>").startswith("ap_fixed<")  # an explicit type wins


# ---- the activation types: HGQ2's own rule for what is alive, and the modes ----


def output_values(conf, shape=(1, 5, 4, 3), scale=40.0):
    quantizer = standalone(conf, (None,) + shape[1:], "probe")
    data = np.random.default_rng(0).standard_normal(shape).astype("float32") * scale
    return np_(quantizer(data)), quantizer


def test_a_dead_activation_quantizer_gives_a_pruned_type_and_really_outputs_zero():
    """k + i + f <= 0: HGQ2 masks the output to 0 at inference (the type used to come out as
    ap_fixed<0,3,...>). Training can push f down to this; a config can not start there (HGQ2
    refuses i + f < 0), so the quantizer here is unsigned with i + f = 0."""
    conf = QuantizerConfig(
        "kif",
        "datalane",
        k0=0,
        i0=2,
        f0=-2,
        overflow_mode="SAT",
        homogeneous_axis=(0, 1, 2, 3),
        trainable=False,
    )
    values, _ = output_values(conf)
    assert np.all(values == 0)
    layer = hgly.QConv2d(4, kernel_size=1, oq_conf=conf)
    layer(X)
    assert layer.result_type() == "ap_fixed<2,1>"


@pytest.mark.parametrize("overflow", ["WRAP", "SAT", "SAT_SYM"])
@pytest.mark.parametrize(
    "ints, fracs, k", [(2, -2, 1), (3, 0, 1), (-1, 4, 1), (3, 1, 0), (-2, 5, 0)]
)
def test_activation_types_are_valid_and_hold_everything_the_quantizer_outputs(
    ints, fracs, k, overflow
):
    conf = QuantizerConfig(
        "kif",
        "datalane",
        k0=k,
        i0=ints,
        f0=fracs,
        overflow_mode=overflow,
        homogeneous_axis=(0, 1, 2, 3),
        trainable=False,
    )
    values, quantizer = output_values(conf)
    layer = hgly.QConv2d(4, kernel_size=1, oq_conf=conf)
    layer(X)
    kind = layer.result_type()
    width, integer, signed = _parse_type(kind)
    assert width >= 1 and signed == bool(k)
    assert holds(re.sub(r",AP_\w+", "", kind), values), (kind, values.min(), values.max())
    assert f"AP_{overflow}" in kind


def test_a_wrapping_signed_quantizer_with_no_bits_still_outputs_its_sign_value():
    """HGQ2 keeps k + i + f = 1 > 0 alive: it outputs -2^i or 0, and the type must hold both."""
    conf = QuantizerConfig(
        "kbi",
        "datalane",
        k0=1,
        b0=0,
        i0=3,
        overflow_mode="WRAP",
        homogeneous_axis=(0, 1, 2, 3),
        trainable=False,
    )
    values, _ = output_values(conf)
    layer = hgly.QConv2d(4, kernel_size=1, oq_conf=conf)
    layer(X)
    assert holds(re.sub(r",AP_\w+", "", layer.result_type()), values)


@pytest.mark.parametrize("rounding", ["TRN", "RND", "RND_CONV"])
def test_the_rounding_mode_is_written_as_ap_does(rounding):
    conf = QuantizerConfig(
        "kif",
        "datalane",
        k0=1,
        i0=2,
        f0=2,
        round_mode=rounding,
        homogeneous_axis=(0, 1, 2, 3),
        trainable=False,
    )
    layer = hgly.QConv2d(4, kernel_size=1, oq_conf=conf)
    layer(X)
    assert f"AP_{rounding}," in layer.result_type()


@pytest.mark.parametrize("rounding", ["S_RND", "S_RND_CONV"])
def test_a_stochastic_rounding_has_no_hardware_type(rounding):
    """AP_S_RND does not exist: it used to be written without a word."""
    conf = QuantizerConfig(
        "kif",
        "datalane",
        k0=1,
        i0=2,
        f0=2,
        round_mode=rounding,
        homogeneous_axis=(0, 1, 2, 3),
        trainable=False,
    )
    layer = hgly.QConv2d(4, kernel_size=1, oq_conf=conf)
    layer(X)
    with pytest.raises(ValueError, match="stochastic"):
        layer.result_type()


# ---- the other layers ----

X3 = np.random.default_rng(1).standard_normal((2, 4, H, W, C)).astype("float32")
SUB_2D = [
    np.random.default_rng(2).standard_normal((2, C, 3, 1)).astype("float32"),
    np.random.default_rng(3).standard_normal((2, C, 2, 2)).astype("float32"),
]
SUB_3D = [
    np.random.default_rng(4).standard_normal((2, C, 2, 3, 1)).astype("float32"),
    np.random.default_rng(5).standard_normal((2, C, 2, 2, 2)).astype("float32"),
]


@pytest.mark.parametrize("depth_padding", ["valid", "same"])
@pytest.mark.parametrize("share", [False, "ring", "diag"])
def test_qconv3d_gives_back_conv3d_with_many_bits(share, depth_padding):
    kw = dict(kernel_size=(2, 1 if share != "diag" else 2), share_neighbors=share)
    kw.update(depth_padding=depth_padding, activation="relu")
    plain = hgly.Conv3d(4, **kw)
    expected = np_(plain(X3))
    qat = hgly.QConv3d(4, kq_conf=fixed(24, 4), **kw)
    qat(X3)
    copy_weights(plain, qat)
    np.testing.assert_allclose(np_(qat(X3)), expected, atol=1e-4)


@pytest.mark.parametrize("trainable", [True, False])
def test_custom_kernel_layers_train_only_when_asked(trainable):
    layer = hgly.QConv2d_CustomKernel(
        sub_kernels=SUB_2D, bias=[0.1, 0.2], trainable=trainable, kq_conf=fixed(24, 4), name="c"
    )
    plain = hgly.Conv2d_CustomKernel(sub_kernels=SUB_2D, bias=[0.1, 0.2], trainable=trainable)
    np.testing.assert_allclose(np_(layer(X)), np_(plain(X)), atol=1e-4)
    names = sorted(w.name for w in layer.trainable_weights)
    assert names == (["bias", "kernel0", "kernel1"] if trainable else [])
    assert {"kernel0", "kernel1", "bias"} <= {w.name for w in layer.weights}  # saved either way


def test_custom_kernels_are_trained_through_their_quantizers():
    layer = hgly.QConv2d_CustomKernel(
        sub_kernels=SUB_2D, trainable=True, kq_conf=fixed(8, 2), name="c"
    )
    inp = keras.Input((H, W, C))
    model = keras.Model(inp, keras.layers.Dense(1)(keras.layers.Flatten()(layer(inp))))
    before = [np_(w).copy() for w in layer._base_kernels]
    model.compile("adam", "mse")
    model.fit(X, np.ones((len(X), 1), "float32"), epochs=15, verbose=0)
    assert all(not np.allclose(a, np_(b)) for a, b in zip(before, layer._base_kernels))


def test_qconv3d_custom_kernel_gives_back_the_plain_layer():
    layer = hgly.QConv3d_CustomKernel(sub_kernels=SUB_3D, kq_conf=fixed(24, 4))
    plain = hgly.Conv3d_CustomKernel(sub_kernels=SUB_3D)
    np.testing.assert_allclose(np_(layer(X3)), np_(plain(X3)), atol=1e-4)


POOLS = [
    (hgly.MaxPool2d, hgly.QMaxPool2d, X),
    (hgly.AvgPool2d, hgly.QAvgPool2d, X),
    (hgly.MaxPool3d, hgly.QMaxPool3d, X3),
]


@pytest.mark.parametrize("plain_cls, q_cls, data", POOLS)
def test_pooling_layers_quantize_only_their_input_and_output(plain_cls, q_cls, data):
    shape = (None,) + data.shape[1:]
    layer = q_cls(1, strides=2, iq_conf=act(2, 3), oq_conf=act(3, 2))
    layer(data)
    iq = standalone(act(2, 3), shape, "m_iq")
    pooled = np_(plain_cls(1, strides=2)(iq(data)))
    oq = standalone(act(3, 2), (None,) + pooled.shape[1:], "m_oq")
    np.testing.assert_allclose(np_(layer(data)), np_(oq(pooled)), atol=1e-6)
    assert layer.weight_type() is None and layer.bias_type() is None
    with pytest.raises(ValueError, match="no weights"):
        q_cls(1, kq_conf=fixed())


@pytest.mark.parametrize("plain_cls, q_cls, data", POOLS)
def test_a_pooling_layer_without_quantizers_is_the_plain_layer(plain_cls, q_cls, data):
    np.testing.assert_allclose(np_(q_cls(1, strides=2)(data)), np_(plain_cls(1, strides=2)(data)))


# ---- freeze with activation quantizers; save / load of every layer ----


def with_quantizers_model():
    inp = keras.Input((H, W, C))
    x = hgly.QConv2d(
        4,
        kernel_size=1,
        activation="relu",
        kq_conf=fixed(6, 1),
        bq_conf=fixed(5, 1),
        iq_conf=act(2, 3),
        oq_conf=act(3, 2, k=0),
        name="hex0",
    )(inp)
    x = hgly.QMaxPool2d(1, strides=2, oq_conf=act(3, 2, k=0), name="pool")(x)
    return keras.Model(inp, keras.layers.Dense(1)(keras.layers.Flatten()(x)))


def test_freeze_keeps_the_activation_quantizers_so_the_outputs_are_the_same():
    model = with_quantizers_model()
    frozen, types = hgly.freeze(model)  # activation_quantizers="keep"
    np.testing.assert_allclose(np_(frozen(X)), np_(model(X)), atol=1e-6)
    names = [layer.name for layer in frozen.layers]
    assert {"hex0_iq", "hex0_oq", "pool_oq"} <= set(names)
    assert type(frozen.get_layer("hex0")) is hgly.Conv2d
    assert type(frozen.get_layer("pool")) is hgly.MaxPool2d
    assert set(types["hex0"]) == {"weight", "bias", "input", "result", "accum"}
    assert set(types["pool"]) == {"result"}  # no weights, no input quantizer


def test_freeze_can_hand_the_activation_types_to_hls4ml_instead():
    frozen, types = hgly.freeze(with_quantizers_model(), activation_quantizers="types")
    assert not any(layer.name.endswith(("_iq", "_oq")) for layer in frozen.layers)
    assert types["hex0"]["result"].startswith("ap_ufixed<5,3,AP_RND,AP_SAT>")
    with pytest.raises(ValueError, match="keep"):
        hgly.freeze(with_quantizers_model(), activation_quantizers="drop")


def test_freeze_names_the_layer_that_is_not_wired_to_the_model():
    layer = hgly.QConv2d(4, kernel_size=1, kq_conf=fixed())
    first = keras.Input((H, W, C))
    keras.Model(first, layer(first))
    second = keras.Input((H, W, C))
    with pytest.raises(ValueError, match="not wired to this model's graph"):
        hgly.freeze(keras.Model(second, layer(second)))


ALL_LAYERS = {
    "QConv2d": (
        lambda: hgly.QConv2d(
            4,
            kernel_size=1,
            kq_conf=fixed(6, 1),
            bq_conf=fixed(5, 1),
            iq_conf=act(2, 3),
            oq_conf=act(3, 2),
        ),
        X,
    ),
    "QConv3d": (
        lambda: hgly.QConv3d(
            4, kernel_size=(2, 1), kq_conf=fixed(6, 1), iq_conf=act(2, 3), oq_conf=act(3, 2)
        ),
        X3,
    ),
    "QConv2d_CustomKernel": (
        lambda: hgly.QConv2d_CustomKernel(
            sub_kernels=SUB_2D,
            bias=[0.1, 0.2],
            trainable=True,
            kq_conf=fixed(6, 1),
            bq_conf=fixed(5, 1),
            oq_conf=act(3, 2),
        ),
        X,
    ),
    "QConv3d_CustomKernel": (
        lambda: hgly.QConv3d_CustomKernel(
            sub_kernels=SUB_3D, kq_conf=fixed(6, 1), iq_conf=act(2, 3)
        ),
        X3,
    ),
    "QMaxPool2d": (lambda: hgly.QMaxPool2d(1, strides=2, iq_conf=act(2, 3), oq_conf=act(3, 2)), X),
    "QMaxPool3d": (lambda: hgly.QMaxPool3d(1, strides=2, oq_conf=act(3, 2)), X3),
    "QAvgPool2d": (lambda: hgly.QAvgPool2d(1, strides=2, iq_conf=act(2, 3), oq_conf=act(3, 2)), X),
}


@pytest.mark.parametrize("name", ALL_LAYERS)
def test_every_quantized_layer_survives_save_and_load(tmp_path, name):
    build, data = ALL_LAYERS[name]
    layer = build()
    inp = keras.Input(data.shape[1:])
    model = keras.Model(inp, layer(inp))
    expected = np_(model(data))
    model.save(tmp_path / "m.keras")
    loaded = keras.models.load_model(tmp_path / "m.keras")
    assert type(loaded.layers[-1]).__name__ == name
    np.testing.assert_allclose(np_(loaded(data)), expected, atol=1e-6)
    config = loaded.layers[-1].get_config()
    assert (config["iq_conf"] is None) == (layer.iq_conf is None)
    assert (config["oq_conf"] is None) == (layer.oq_conf is None)


@pytest.mark.parametrize("name", ALL_LAYERS)
def test_every_quantized_layer_freezes_to_its_plain_layer(name):
    build, data = ALL_LAYERS[name]
    layer = build()
    inp = keras.Input(data.shape[1:])
    model = keras.Model(inp, layer(inp))
    frozen, types = hgly.freeze(model)
    np.testing.assert_allclose(np_(frozen(data)), np_(model(data)), atol=1e-6)
    assert type(frozen.layers[-1]) is not type(layer)  # plain layer (after any output quantizer)
    plain = [x for x in frozen.layers if type(x) is layer._plain_class]
    assert len(plain) == 1 and list(types) == [layer.name]


# ---- serialization beyond .keras: json, from_config, clone_model, h5 ----


def everything_layer():
    return hgly.QConv2d(
        4,
        kernel_size=1,
        activation="relu",
        kq_conf=fixed(6, 1),
        bq_conf=fixed(5, 1),
        iq_conf=act(2, 3),
        oq_conf=act(3, 2, k=0),
        name="hex0",
    )


def everything_model():
    inp = keras.Input((H, W, C))
    return keras.Model(inp, everything_layer()(inp))


def test_the_layer_config_is_json_and_round_trips():
    layer = everything_layer()
    layer(X)
    config = layer.get_config()
    json.dumps(config)  # what h5 needs: no raw HGQ2 objects in there
    clone = type(layer).from_config(config)
    clone(X)
    assert json.dumps(clone.get_config(), sort_keys=True) == json.dumps(config, sort_keys=True)


def test_clone_model_keeps_the_quantizers():
    model = everything_model()
    clone = keras.models.clone_model(model)
    assert isinstance(clone.get_layer("hex0"), hgly.QConv2d)
    assert clone.get_layer("hex0").oq_conf is not None


def test_h5_save_and_load(tmp_path):
    model = everything_model()
    expected = np_(model(X))
    model.save(tmp_path / "m.h5")
    loaded = keras.models.load_model(tmp_path / "m.h5")
    np.testing.assert_allclose(np_(loaded(X)), expected, atol=1e-6)


# ---- freeze on harder models ----


def test_freezing_a_frozen_model_changes_nothing():
    """The kept HGQ2 Quantizer layers used to make a second freeze (and clone_model) crash."""
    model = with_quantizers_model()
    frozen, _ = hgly.freeze(model)
    again, types = hgly.freeze(frozen)
    np.testing.assert_allclose(np_(again(X)), np_(model(X)), atol=1e-6)
    assert types == {}  # nothing left to freeze


def test_freeze_keeps_the_structure_of_the_outputs():
    inp = keras.Input((H, W, C))
    hidden = hgly.QConv2d(4, kernel_size=1, kq_conf=fixed(6, 1), name="c")(inp)
    model = keras.Model(
        inp,
        {
            "flat": keras.layers.Flatten()(hidden),
            "pooled": keras.layers.GlobalAveragePooling2D()(hidden),
        },
    )
    frozen, _ = hgly.freeze(model)
    out, expected = frozen(X), model(X)
    assert isinstance(out, dict) and set(out) == {"flat", "pooled"}
    for key in out:
        np.testing.assert_allclose(np_(out[key]), np_(expected[key]), atol=1e-6)


def nested_model():
    inner = keras.Sequential(
        [keras.Input((H, W, C)), hgly.QConv2d(4, kernel_size=1, kq_conf=fixed(6, 1), name="deep")],
        name="inner",
    )
    outer_in = keras.Input((H, W, C))
    return keras.Model(outer_in, keras.layers.ReLU()(inner(outer_in)))


def test_freeze_goes_into_nested_models_and_names_their_layers():
    model = nested_model()
    frozen, types = hgly.freeze(model)
    assert list(types) == ["inner/deep"]
    inner = frozen.get_layer("inner")
    assert not any(isinstance(layer, hgly.QConv2d) for layer in inner.layers)  # was a silent no-op
    np.testing.assert_allclose(np_(frozen(X)), np_(model(X)), atol=1e-6)


def test_the_export_guard_looks_inside_nested_models():
    pytest.importorskip("hls4ml")
    from keras_hexagdly.hls4ml_ext import patch_model_for_hls

    with pytest.raises(ValueError, match="freeze"):
        patch_model_for_hls(nested_model())


def test_the_export_guard_comes_before_the_unsupported_layer_errors():
    pytest.importorskip("hls4ml")
    from keras_hexagdly.hls4ml_ext import patch_model_for_hls

    inp = keras.Input((H, W, C))
    model = keras.Model(inp, hgly.QAvgPool2d(1, oq_conf=act(3, 2), name="avg")(inp))
    with pytest.raises(ValueError, match="freeze"):  # not the AvgPool2d NotImplementedError
        patch_model_for_hls(model)


def test_the_frozen_model_cannot_be_trained_by_accident():
    """Its kernels are on the quantized grid and the kept quantizers hold the learned bit widths:
    a fit would leave the grid."""
    frozen, _ = hgly.freeze(with_quantizers_model())
    names = {w.name for w in frozen.trainable_weights}
    assert not {n for n in names if n.startswith(("base_kernel", "f", "i", "b"))} - {"bias"}
    before = [np_(v).copy() for v in frozen.weights]
    frozen.compile("adam", "mse")
    frozen.fit(X, np.ones((len(X), 1), "float32"), epochs=3, verbose=0)
    for old, var in zip(before, frozen.weights):
        if var.name.startswith(("base_kernel", "f", "i", "b")) and var.name != "bias":
            np.testing.assert_array_equal(old, np_(var))


def test_freeze_explains_that_it_needs_a_built_model():
    unbuilt = keras.Sequential([hgly.QConv2d(2, kernel_size=1)])
    with pytest.raises(ValueError, match="built"):
        hgly.freeze(unbuilt)


def test_a_shared_layer_error_says_what_to_do():
    shared = keras.layers.Dense(C)
    inp = keras.Input((H, W, C))
    model = keras.Model(inp, keras.layers.Add()([shared(inp), shared(inp)]))
    with pytest.raises(ValueError, match="called more than once"):
        hgly.freeze(model)


# ---- the rest of the audit ----


def test_the_custom_kernel_bit_widths_train_even_when_the_kernels_do_not():
    trained = QuantizerConfig("kbi", "weight", k0=1, b0=6, i0=1)  # trainable bit widths
    layer = hgly.QConv2d_CustomKernel(sub_kernels=SUB_2D, trainable=False, kq_conf=trained)
    layer(X)
    assert {w.name for w in layer.trainable_weights} == {"b", "i"}  # no kernel, no bias
    fixed_bits = hgly.QConv2d_CustomKernel(sub_kernels=SUB_2D, trainable=False, kq_conf=fixed(6, 1))
    fixed_bits(X)
    assert not fixed_bits.trainable_weights  # plain rounding


def test_the_quantized_names_are_listed_by_dir():
    assert {"QConv2d", "QAvgPool2d", "freeze"} <= set(dir(hgly))


@pytest.mark.skipif(
    keras.backend.backend() == "tensorflow",
    reason="HGQ2's own quantizers do not support mixed_float16 on TensorFlow",
)
def test_mixed_precision_runs_on_the_other_backends():
    keras.config.set_dtype_policy("mixed_float16")
    try:
        layer = hgly.QConv2d(4, kernel_size=1, kq_conf=fixed(6, 1), bq_conf=fixed(5, 1))
        out = layer(X)
        assert np.all(np.isfinite(np_(out).astype("float32")))
    finally:
        keras.config.set_dtype_policy("float32")
