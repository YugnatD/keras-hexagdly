"""QConv2d / freeze: quantization-aware training of the hex convolution with HGQ2 (optional)."""

import json
import os
import re
import subprocess
import sys

import keras
import numpy as np
import pytest

import keras_hexagdly as hgly

pytest.importorskip("hgq")
from hgq.quantizer import Quantizer, QuantizerConfig  # noqa: E402
from hgq.regularizers import MonoL1  # noqa: E402

from keras_hexagdly.quantized import _parse_type  # noqa: E402

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


def run_python(code, *, hide_hgq=False):
    """Run ``code`` in a fresh interpreter (same backend); returns the completed process."""
    prefix = "import sys; sys.modules['hgq'] = None\n" if hide_hgq else ""
    # the package under test, not whichever version is installed
    src = os.path.dirname(os.path.dirname(hgly.__file__))
    env = {
        **os.environ,
        "KERAS_BACKEND": keras.backend.backend(),
        "PYTHONWARNINGS": "ignore",
        "PYTHONPATH": os.pathsep.join([src, os.environ.get("PYTHONPATH", "")]),
    }
    return subprocess.run(
        [sys.executable, "-c", prefix + code], capture_output=True, text=True, env=env, timeout=300
    )


def test_the_package_imports_without_hgq_and_says_what_is_missing():
    code = """
import keras_hexagdly as h
assert h.Conv2d is not None  # the rest of the package does not need hgq2
assert not hasattr(h, 'QConv2d') and getattr(h, 'freeze', None) is None
assert 'QConv2d' not in dir(h)
try:
    h.QConv2d
except AttributeError as e:
    assert 'keras-hexagdly[qat]' in str(e), e
else:
    raise SystemExit('no error')
try:
    import keras_hexagdly.quantized
except ImportError as e:
    assert 'keras-hexagdly[qat]' in str(e), e
else:
    raise SystemExit('no error from quantized')
import inspect, pydoc
inspect.getmembers(h); pydoc.render_doc(h)  # introspection must not raise
print('ok')
"""
    done = run_python(code, hide_hgq=True)
    assert done.returncode == 0 and done.stdout.strip().endswith("ok"), done.stderr[-2000:]


def test_a_saved_qat_model_loads_in_a_fresh_process_that_only_imports_the_package(tmp_path):
    inp = keras.Input((H, W, C))
    out = hgly.QMaxPool2d(2, iq_conf=act(2, 3))(
        hgly.QConv2d(4, kernel_size=1, kq_conf=fixed(6, 1), iq_conf=act(2, 3))(inp)
    )
    model = keras.Model(inp, out)
    expected = np_(model(X))
    for name in ("model.keras", "model.h5"):
        model.save(tmp_path / name)
    np.save(tmp_path / "expected.npy", expected)
    np.save(tmp_path / "x.npy", X)
    code = f"""
import keras, numpy as np
import keras_hexagdly  # nothing else imported
for name in ('model.keras', 'model.h5'):
    m = keras.saving.load_model({str(tmp_path)!r} + '/' + name)
    y = keras.ops.convert_to_numpy(m(np.load({str(tmp_path / "x.npy")!r})))
    assert np.array_equal(y, np.load({str(tmp_path / "expected.npy")!r})), name
print('ok')
"""
    done = run_python(code)
    assert done.returncode == 0 and done.stdout.strip().endswith("ok"), done.stderr[-2000:]


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


def test_freeze_follows_the_graph_of_the_model_when_the_layer_is_reused_by_another_one():
    layer = hgly.QConv2d(4, kernel_size=1, kq_conf=fixed())
    first = keras.Input((H, W, C))
    keras.Model(first, layer(first))
    second = keras.Input((H, W, C))
    model = keras.Model(second, layer(second))  # the layer's first graph node is not this one
    frozen, _ = hgly.freeze(model)
    assert np.allclose(np_(frozen(X)), np_(model(X)), atol=1e-5)


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


def test_a_plain_layer_called_twice_is_shared_in_the_frozen_model():
    shared = keras.layers.Dense(C)
    inp = keras.Input((H, W, C))
    model = keras.Model(inp, keras.layers.Add()([shared(inp), shared(inp * 2.0)]))
    frozen, _ = hgly.freeze(model)
    assert np.allclose(np_(frozen(X)), np_(model(X)), atol=1e-5)
    assert len([layer for layer in frozen.layers if isinstance(layer, keras.layers.Dense)]) == 1


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


# ---- audit round 2: freeze on any graph, scaler / affine, dtype, unbuilt layers ----


def test_freeze_keeps_the_operations_between_layers():
    inp = keras.Input((H, W, C))
    y = hgly.QConv2d(C, kernel_size=1, kq_conf=fixed(6, 1), oq_conf=act(3, 3))(inp)
    y = keras.ops.relu(y) + inp  # a residual connection made of raw operations
    model = keras.Model(inp, y)
    frozen, types = hgly.freeze(model)
    assert np.allclose(np_(frozen(X)), np_(model(X)), atol=1e-5)
    assert set(types) == {model.layers[1].name}


def test_freeze_handles_a_layer_called_twice_and_a_nested_functional_model():
    shared = hgly.QConv2d(C, kernel_size=1, kq_conf=fixed(6, 1), iq_conf=act(2, 3))
    inp = keras.Input((H, W, C))
    twice = shared(shared(inp))  # the second call is the model output
    model = keras.Model(inp, twice)
    frozen, types = hgly.freeze(model)
    assert np.allclose(np_(frozen(X)), np_(model(X)), atol=1e-5)
    assert list(types) == [shared.name]

    inner_in = keras.Input((H, W, C))
    inner = keras.Model(
        inner_in, hgly.QConv2d(C, kernel_size=1, kq_conf=fixed(6, 1), name="deep")(inner_in)
    )
    outer_in = keras.Input((H, W, C))
    outer = keras.Model(outer_in, keras.layers.Flatten()(inner(outer_in)))  # built, then called
    frozen, types = hgly.freeze(outer)
    assert np.allclose(np_(frozen(X)), np_(outer(X)), atol=1e-5)
    assert f"{inner.name}/deep" in types


def test_freeze_keeps_the_names_of_the_inputs_and_dict_inputs():
    a, b = keras.Input((H, W, C), name="a"), keras.Input((H, W, C), name="b")
    conv = hgly.QConv2d(C, kernel_size=1, kq_conf=fixed(6, 1))
    model = keras.Model({"a": a, "b": b}, conv(a) + keras.layers.Conv2D(C, 1)(b))
    frozen, _ = hgly.freeze(model)
    assert [t.name for t in frozen.inputs] == [t.name for t in model.inputs]
    data = {"a": X, "b": X}
    assert np.allclose(np_(frozen(data)), np_(model(data)), atol=1e-5)


def scaled(**extra):
    return QuantizerConfig(
        "kif", "datalane", k0=1, i0=2, f0=2, homogeneous_axis=(0, 1, 2, 3), trainable=False, **extra
    )


@pytest.mark.parametrize("extra", [{"scaler": 4.0}, {"affine": (0.25, 0.5)}])
def test_a_scaler_or_affine_on_an_activation_quantizer_has_no_ap_fixed_type(extra):
    layer = hgly.QConv2d(C, kernel_size=1, kq_conf=fixed(6, 1), iq_conf=scaled(**extra))
    layer(X)
    with pytest.raises(ValueError, match="scaler or an affine"):
        layer.input_type()
    model = keras.Model(inp := keras.Input((H, W, C)), layer(inp))
    frozen, types = hgly.freeze(model)  # "keep": the quantizer stays, only the type is missing
    assert "input" not in types[layer.name] and "weight" in types[layer.name]
    assert np.allclose(np_(frozen(X)), np_(model(X)), atol=1e-5)
    with pytest.raises(ValueError, match="scaler or an affine"):
        hgly.freeze(model, activation_quantizers="types")


@pytest.mark.skipif(
    keras.backend.backend() == "tensorflow",
    reason="HGQ2's own quantizers do not support half precision on TensorFlow",
)
def test_quantizer_outputs_are_cast_back_to_the_compute_dtype():
    layer = hgly.QConv2d(
        2, kernel_size=1, dtype="float16", iq_conf=act(3, 3), oq_conf=act(3, 3), kq_conf=fixed(6, 1)
    )
    out = layer(X)
    assert keras.backend.standardize_dtype(out.dtype) == "float16"
    pool = hgly.QMaxPool2d(2, dtype="float16", iq_conf=act(3, 3))
    assert keras.backend.standardize_dtype(pool(X).dtype) == "float16"


def test_the_accessors_of_an_unbuilt_layer_say_so():
    layer = hgly.QConv2d(
        2,
        kernel_size=1,
        kq_conf=fixed(6, 1),
        bq_conf=fixed(6, 1),
        iq_conf=act(2, 2),
        oq_conf=act(2, 2),
    )
    for accessor in (
        layer.quantized_weights,
        layer.quantized_bias,
        layer.weight_type,
        layer.bias_type,
        layer.input_type,
        layer.result_type,
        layer.accum_type,
    ):
        with pytest.raises(ValueError, match="needs a built layer"):
            accessor()


def test_a_pool_with_a_quantizer_needs_static_spatial_dimensions():
    with pytest.raises(ValueError, match="statically-known"):
        hgly.QMaxPool2d(2, oq_conf=act(2, 2)).build((None, None, None, C))


# ---- EBOPs: the cost measure HGQ2's callbacks read ----

hgq_sugar = pytest.importorskip("hgq.utils.sugar")


def valid_taps(height, width, **kwargs):
    """How many (output cell, input cell) pairs a hexagonal window connects in an H x W image,
    counted with the plain layer and all-ones kernels (the zero padding of the border is out)."""
    plain = hgly.Conv2d(1, use_bias=False, activation=None, **kwargs)
    ones = np.ones((1, height, width, 1), "float32")
    plain(ones)
    for variable in plain.trainable_weights:
        variable.assign(np.ones(variable.shape, "float32"))
    return float(np_(plain(ones)).sum())


def train_one_step(model, x, filters=4):
    model.compile("sgd", "mse")
    model.fit(x, np.zeros((len(x), H, W, filters), "float32"), epochs=1, verbose=0)


def ebops_of(layer, x=X, filters=4):
    inp = keras.Input(x.shape[1:])
    model = keras.Model(inp, layer(inp))
    train_one_step(model, x, filters)
    return float(np_(layer.ebops))


def test_ebops_are_the_bits_of_the_weights_times_those_of_the_input_over_the_hexagon():
    layer = hgly.QConv2d(
        4, kernel_size=1, kq_conf=fixed(6, 1), iq_conf=act(2, 3), use_bias=False, name="e"
    )
    got = ebops_of(layer)
    input_bits = np_(layer._iq.bits_((1, H, W, C))).max()
    kernel_bits = np_(layer._kqs[0].bits_(keras.ops.shape(layer._kernel_sources()[0]))).max()
    assert input_bits > 0 and kernel_bits > 0
    assert got == valid_taps(H, W, kernel_size=1) * C * 4 * input_bits * kernel_bits


def test_the_bias_adds_its_bits_for_every_output():
    without = hgly.QConv2d(4, kernel_size=1, kq_conf=fixed(6, 1), iq_conf=act(2, 3), use_bias=True)
    with_bias = hgly.QConv2d(
        4, kernel_size=1, kq_conf=fixed(6, 1), bq_conf=fixed(5, 1), iq_conf=act(2, 3)
    )
    base, plus = ebops_of(without), ebops_of(with_bias)
    bias_bits = np_(with_bias._bq.bits_((4,))).mean()
    assert plus - base == bias_bits * H * W * 4


def test_ebops_are_tracked_in_training_and_enter_the_loss_with_beta():
    layer = hgly.QConv2d(4, kernel_size=1, kq_conf=fixed(6, 1), iq_conf=act(2, 3), beta0=0.5)
    layer(X)
    assert int(np_(layer.ebops)) == 0  # inference does not count
    assert not layer.losses
    layer(X, training=True)
    ebops = float(np_(layer.ebops))
    assert ebops > 0
    np.testing.assert_allclose(
        float(sum(np_(loss) for loss in layer.losses)), 0.5 * ebops, rtol=1e-5
    )


def test_the_hgq2_callbacks_see_the_layers():
    from hgq.layers import QLayerBase

    first = hgly.QConv2d(C, kernel_size=1, kq_conf=fixed(6, 1), iq_conf=act(2, 3), name="first")
    second = hgly.QConv2d(2, kernel_size=1, kq_conf=fixed(6, 1), iq_conf=act(2, 3), name="second")
    inp = keras.Input((H, W, C))
    model = keras.Model(inp, second(first(inp)))
    assert isinstance(first, QLayerBase) and first.enable_ebops
    model.compile("sgd", "mse")
    free = hgq_sugar.FreeEBOPs()
    history = model.fit(X, np.zeros((len(X), H, W, 2), "float32"), epochs=1, callbacks=[free])
    total = int(np_(first.ebops)) + int(np_(second.ebops))
    assert history.history["ebops"] == [total] and total > 0
    scheduler = hgq_sugar.BetaScheduler(lambda epoch: 3e-3)
    scheduler.set_model(model)
    scheduler.on_epoch_begin(0)
    assert float(np_(first.beta)) == pytest.approx(3e-3) == float(np_(second.beta))


def test_a_beta_term_shrinks_the_bit_widths_and_so_the_ebops():
    trained = QuantizerConfig("kbi", "weight", k0=1, b0=6, i0=1)  # learned bit widths
    layer = hgly.QConv2d(
        4, kernel_size=1, kq_conf=trained, iq_conf=act(2, 3), beta0=1e-2, use_bias=False
    )
    inp = keras.Input((H, W, C))
    model = keras.Model(inp, layer(inp))
    model.compile(keras.optimizers.Adam(5e-2), "mse")
    target = np.zeros((len(X), H, W, 4), "float32")
    free = hgq_sugar.FreeEBOPs()
    history = model.fit(X, target, epochs=30, batch_size=2, verbose=0, callbacks=[free])
    ebops = history.history["ebops"]
    assert ebops[-1] < ebops[0] / 1.5


def test_trace_minmax_finds_the_integer_bits_of_a_wrapping_quantizer():
    from hgq.utils import trace_minmax

    wrapping = QuantizerConfig(
        "kif", "datalane", k0=1, i0=0, f0=3, overflow_mode="WRAP", homogeneous_axis=(0, 1, 2, 3)
    )
    layer = hgly.QConv2d(4, kernel_size=1, kq_conf=fixed(6, 1), oq_conf=wrapping)
    inp = keras.Input((H, W, C))
    model = keras.Model(inp, layer(inp))
    before = _parse_type(layer.result_type())[1]
    trace_minmax(model, X * 3)
    assert _parse_type(layer.result_type())[1] > before


def test_ebops_need_weights_and_an_input_quantizer():
    layer = hgly.QConv2d(4, kernel_size=1, kq_conf=fixed(6, 1))
    assert not layer.enable_ebops  # no input quantizer, no bit widths for the input
    layer(X)
    assert not [w for w in layer.non_trainable_weights if w.name in ("beta", "ebops")]
    assert hgly.QConv2d(4, kernel_size=1, iq_conf=act(2, 3)).enable_ebops
    assert not hgly.QConv2d(4, kernel_size=1, iq_conf=act(2, 3), enable_ebops=False).enable_ebops
    with pytest.raises(ValueError, match="enable_ebops"):
        hgly.QConv2d(4, kernel_size=1, enable_ebops=True)
    with pytest.raises(ValueError, match="enable_ebops"):
        hgly.QMaxPool2d(2, iq_conf=act(2, 3), enable_ebops=True)
    assert not hgly.QMaxPool2d(2, iq_conf=act(2, 3)).enable_ebops


@pytest.mark.parametrize(
    "name", ["QConv2d", "QConv3d", "QConv3d_CustomKernel", "QConv2d_CustomKernel"]
)
def test_every_weighted_layer_with_an_input_quantizer_counts_ebops(name):
    build, x = ALL_LAYERS[name]
    layer = build()
    if layer.iq_conf is None:
        pytest.skip("this variant has no input quantizer")
    inp = keras.Input(x.shape[1:])
    model = keras.Model(inp, layer(inp))
    model.compile("sgd", "mse")
    model.fit(
        x, np.zeros((len(x),) + tuple(model.output_shape[1:]), "float32"), epochs=1, verbose=0
    )
    ebops = float(np_(layer.ebops))
    assert layer.enable_ebops and ebops > 0
    # every kernel and input element has 6 bits here: a multiple of 36, which the stale kernels of
    # a previous call (random floats) would not give
    assert ebops % 36 == 0


def test_ebops_settings_survive_save_load_and_clone(tmp_path):
    layer = hgly.QConv2d(
        4, kernel_size=1, kq_conf=fixed(6, 1), iq_conf=act(2, 3), beta0=0.25, ebops_factor=2.0
    )
    inp = keras.Input((H, W, C))
    model = keras.Model(inp, layer(inp))
    model.save(tmp_path / "m.keras")
    for other in (
        keras.saving.load_model(tmp_path / "m.keras"),
        keras.models.clone_model(model),
    ):
        copy = other.layers[1]
        assert copy.enable_ebops and copy._beta0 == 0.25 and copy.ebops_factor == 2.0
        assert float(np_(copy._beta)) == 0.25


# ---- audit round 3 ----


def test_a_model_saved_before_the_ebops_existed_loads_with_them_off():
    layer = hgly.QConv2d(4, kernel_size=1, kq_conf=fixed(6, 1), iq_conf=act(2, 3))
    layer(X)
    old = layer.get_config()
    for key in ("enable_ebops", "beta0", "ebops_factor"):
        del old[key]  # what a 0.7.0 development version wrote
    loaded = hgly.QConv2d.from_config(old)
    loaded(X)
    assert not loaded.enable_ebops
    assert not [w for w in loaded.weights if w.name in ("beta", "ebops")]
    assert len(loaded.weights) == len(layer.weights) - 2  # the weights file of the old model fits


def test_freeze_reuses_a_weightless_layer_it_cannot_copy():
    inp = keras.Input((H, W, C))
    y = hgly.QConv2d(C, kernel_size=1, kq_conf=fixed(6, 1))(inp)
    model = keras.Model(
        inp, keras.layers.Lambda(lambda t: t * 2.0)(y)
    )  # not rebuildable from config
    frozen, _ = hgly.freeze(model)
    assert np.allclose(np_(frozen(X)), np_(model(X)), atol=1e-5)


@pytest.mark.skipif(
    keras.backend.backend() == "tensorflow",
    reason="HGQ2's own quantizers do not support half precision on TensorFlow",
)
def test_the_frozen_standalone_quantizers_keep_the_compute_dtype_and_ebops_stay_finite():
    keras.config.set_dtype_policy("mixed_float16")
    try:
        layer = hgly.QConv2d(
            16, kernel_size=1, kq_conf=fixed(8, 2), iq_conf=act(3, 3), oq_conf=act(3, 3), beta0=1e-3
        )
        inp = keras.Input((H, W, 8))
        model = keras.Model(inp, layer(inp))
        x = np.random.default_rng(1).standard_normal((2, H, W, 8)).astype("float32")
        out = model(x, training=True)
        assert keras.backend.standardize_dtype(out.dtype) == "float16"
        ebops = float(np_(layer.ebops))
        assert np.isfinite(ebops) and ebops > 65504  # above what half precision can hold
        assert all(np.isfinite(float(np_(loss))) for loss in layer.losses)
        frozen, _ = hgly.freeze(model)
        assert keras.backend.standardize_dtype(frozen(x).dtype) == "float16"
    finally:
        keras.config.set_dtype_policy("float32")


@pytest.mark.parametrize(
    "failure", ["raise RuntimeError('boom')", "raise ImportError('numpy ABI mismatch')"]
)
def test_a_broken_hgq_leaves_the_plain_layers_usable(tmp_path, failure):
    fake = tmp_path / "hgq"
    fake.mkdir()
    (fake / "__init__.py").write_text(failure + "\n")
    code = f"""
import sys, warnings
sys.path.insert(0, {str(tmp_path)!r})
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    import keras_hexagdly as h
assert h.Conv2d is not None
assert not hasattr(h, 'QConv2d')
assert any('quantized layers are unavailable' in str(w.message) for w in caught), caught
try:
    h.QConv2d
except AttributeError as e:
    assert 'quantized' in str(e) or 'boom' in str(e) or 'ABI' in str(e), e
print('ok')
"""
    done = run_python(code)
    assert done.returncode == 0 and done.stdout.strip().endswith("ok"), done.stderr[-2000:]


# ---- the hls4ml config from freeze's types ----


@needs_hls
def test_apply_quantized_types_fills_the_config_of_a_chain_of_layers():
    pytest.importorskip("hls4ml")
    import hls4ml

    from keras_hexagdly.hls4ml_ext import apply_quantized_types, patch_model_for_hls
    from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

    register_hex_gather_layers("Vitis")
    inp = keras.Input((H, W, C), name="x")
    first = hgly.QConv2d(
        4,
        kernel_size=1,
        activation="relu",
        name="c1",
        kq_conf=fixed(6, 1),
        bq_conf=fixed(5, 1),
        iq_conf=act(2, 3),
        oq_conf=act(3, 2, k=0),
    )
    second = hgly.QConv2d(
        2, kernel_size=1, name="c2", kq_conf=fixed(6, 1), iq_conf=act(3, 2, k=0), oq_conf=act(3, 3)
    )
    model = keras.Model(inp, second(first(inp)))
    frozen, types = hgly.freeze(model, activation_quantizers="types")
    patched = patch_model_for_hls(frozen)
    config = hls4ml.utils.config_from_keras_model(patched, granularity="name", backend="Vitis")
    assert apply_quantized_types(config, patched, types) is config
    layer = config["LayerName"]
    assert layer["x"]["Precision"]["result"] == types["c1"]["input"]
    precision = dict(layer["c1_linebuffer"]["Precision"])
    precision.pop("accum", None)  # the node declares an accumulator type, hls4ml adds its default
    assert precision == {
        "result": types["c1"]["accum"],  # the MAC hands its sum on to the ReLU without wrapping it
        "mac_weights": types["c1"]["weight"],
        "mac_bias": types["c1"]["bias"],
    }
    assert layer["c1_activation"]["Precision"]["result"] == types["c1"]["result"]
    assert layer["c1_linebuffer"]["AccumPrecision"] == types["c1"]["accum"]
    # c2 has no activation: its linebuffer holds its result, and its producer holds its input type
    assert layer["c2_linebuffer"]["Precision"]["result"] == types["c2"]["result"]
    assert (
        "AccumPrecision"
        not in apply_quantized_types({}, patched, types, accumulator="default")["LayerName"][
            "c1_linebuffer"
        ]
    )


@needs_hls
def test_apply_quantized_types_rejects_types_of_another_model():
    pytest.importorskip("hls4ml")
    from keras_hexagdly.hls4ml_ext import apply_quantized_types, patch_model_for_hls
    from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

    register_hex_gather_layers("Vitis")
    inp = keras.Input((H, W, C), name="x")
    model = keras.Model(inp, hgly.QConv2d(2, kernel_size=1, name="c", kq_conf=fixed(6, 1))(inp))
    frozen, types = hgly.freeze(model, activation_quantizers="types")
    patched = patch_model_for_hls(frozen)
    with pytest.raises(ValueError, match="not in the patched model"):
        apply_quantized_types({}, patched, {"other": types["c"]})
    with pytest.raises(ValueError, match="accumulator must be"):
        apply_quantized_types({}, patched, types, accumulator="x")


# ---- audit round 3: tests for the mutations that survived (accum_type, types, EBOPs, freeze) ----

from keras_hexagdly.quantized import _exact_fixed_type  # noqa: E402


def _qb(cls, ks, c_in, *, kq=None, bq=None, iq=None, oq=None, shape=None, **kw):
    layer = cls(
        4, kernel_size=ks, kq_conf=kq or fixed(6, 1), bq_conf=bq, iq_conf=iq, oq_conf=oq, **kw
    )
    x = np.random.default_rng(1).standard_normal(shape or (2, H, W, c_in)).astype("float32")
    layer(x)
    return layer, x


def _set_all(layer, value):
    for v in layer._base_kernels:
        v.assign(np.full(v.shape, value, "float32"))


# ---- accum_type: exact, hand-computed strings -------------------------------------------------
# terms = (1 + 3 r (r+1)) * channels * depth ; I = Iw + Ii + ceil(log2 terms) (max with Ib) + 1
# F = max(Fw + Fi, Fb).  weights 0.5 -> ap_fixed<2,1> ; input act(2,3) -> ap_fixed<6,3>
def test_accum_type_2d_hand_computed():
    layer, _ = _qb(hgly.QConv2d, 1, 3, iq=act(2, 3))
    _set_all(layer, 0.5)
    assert layer.weight_type() == "ap_fixed<2,1>"
    assert layer.accum_type() == "ap_fixed<14,10>"  # 21 terms -> 5 ; 1+3+5+1 ; F=1+3


def test_accum_type_terms_at_a_power_of_two_boundary():
    layer, _ = _qb(hgly.QConv2d, 1, 10, iq=act(2, 3))
    _set_all(layer, 0.5)
    assert layer.accum_type() == "ap_fixed<16,12>"  # 70 terms -> 7 (60 would give 6)


def test_accum_type_3d_counts_the_depth():
    layer, _ = _qb(hgly.QConv3d, (3, 1), 3, iq=act(2, 3), shape=(2, 3, H, W, 3))
    _set_all(layer, 0.5)
    assert layer.accum_type() == "ap_fixed<15,11>"  # 63 terms -> 6 ; 1+3+6+1


def test_accum_type_bias_wider_than_products_sets_the_integer_part():
    layer, _ = _qb(hgly.QConv2d, 1, 3, iq=act(2, 3), bq=fixed(14, 12))
    _set_all(layer, 0.5)
    layer.bias_tensor.assign(np.full((4,), 1000.25, "float32"))
    assert layer.bias_type() == "ap_fixed<13,11>"
    assert layer.accum_type() == "ap_fixed<16,12>"  # max(9, 11) + 1 ; F=max(4, 2)


def test_accum_type_finer_bias_sets_the_fractional_part():
    layer, _ = _qb(hgly.QConv2d, 1, 3, iq=act(2, 3), bq=fixed(10, 2))
    _set_all(layer, 0.5)
    layer.bias_tensor.assign(np.full((4,), 1.0078125, "float32"))  # 8 fractional bits
    assert layer.accum_type() == "ap_fixed<17,10>"  # F = max(4, 7)


# ---- _exact_fixed_type boundaries ------------------------------------------------------------
@pytest.mark.parametrize(
    "values, expected",
    [
        ([-1.0], "ap_fixed<1,1>"),  # -2^(I-1) exactly, the lowest value of the type
        ([0.5], "ap_fixed<2,1>"),  # the highest value of ap_fixed<2,1> is 0.5
        ([1.0], "ap_fixed<2,2>"),  # 2^(I-1) does not fit in I
        ([-0.5, 0.25], "ap_fixed<2,0>"),
    ],
)
def test_exact_fixed_type_boundaries(values, expected):
    assert _exact_fixed_type([np.array(values)]) == expected


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_exact_fixed_type_refuses_non_finite(bad):
    with pytest.raises(ValueError, match="NaN or infinite"):
        _exact_fixed_type([np.array([0.5, bad])])


# ---- activation types: heterogeneous / non-integer bit widths ---------------------------------
PER_ELEMENT_KIF = QuantizerConfig("kif", "datalane", k0=1, i0=2, f0=3, trainable=True)


def test_activation_type_takes_the_max_over_the_alive_elements_and_rounds_the_bits():
    layer, _ = _qb(hgly.QConv2d, 1, C, iq=PER_ELEMENT_KIF)
    q = {v.name: v for v in layer._iq.variables}
    i = np.full(q["i"].shape, 1.0, "float32")
    f = np.full(q["f"].shape, 2.0, "float32")
    i.flat[0], f.flat[1] = 4.0, 6.0  # two different elements are the widest
    q["i"].assign(i)
    q["f"].assign(f)
    assert layer.input_type().startswith("ap_fixed<11,5,")  # I = 4 + sign, F = 6
    i[:] = 1.6
    f[:] = 2.4  # trained bit widths are not integers
    q["i"].assign(i)
    q["f"].assign(f)
    assert layer.input_type().startswith(
        "ap_fixed<5,3,"
    )  # round(1.6)=2, round(2.4)=2 -> <2+2+1, 3>


# ---- freeze copies the *learned* activation quantizers ----------------------------------------
def test_freeze_keep_copies_the_learned_bit_widths_of_the_activation_quantizers():
    inp = keras.Input((H, W, C))
    layer = hgly.QConv2d(
        4, kernel_size=1, kq_conf=fixed(6, 1), iq_conf=PER_ELEMENT_KIF, oq_conf=PER_ELEMENT_KIF
    )
    model = keras.Model(inp, layer(inp))
    for q in (layer._iq, layer._oq):  # as if training had changed them
        for v in q.variables:
            if v.name in ("f", "i"):
                v.assign(np.random.default_rng(3).uniform(0.0, 3.0, v.shape).astype("float32"))
    frozen, _ = hgly.freeze(model)
    np.testing.assert_allclose(np_(frozen(X)), np_(model(X)), atol=1e-6)
    assert not any(l.trainable for l in frozen.layers if "_iq" in l.name or "_oq" in l.name)


def test_freeze_keeps_the_dict_structure_of_the_inputs():
    a, b = keras.Input((H, W, C), name="a"), keras.Input((H, W, C), name="b")
    model = keras.Model(
        {"a": a, "b": b}, hgly.QConv2d(C, kernel_size=1, kq_conf=fixed(6, 1))(a) + b
    )
    frozen, _ = hgly.freeze(model)
    assert isinstance(frozen.input, dict) and set(frozen.input) == {"a", "b"}


# ---- EBOPs bookkeeping ------------------------------------------------------------------------
def _fit_one_step(layer, x, **fit):
    inp = keras.Input(x.shape[1:])
    model = keras.Model(inp, layer(inp))
    model.compile("sgd", "mse")
    model.fit(
        x, np.zeros((len(x),) + tuple(model.output_shape[1:]), "float32"), epochs=1, verbose=0
    )
    return model


def test_inference_after_a_training_step_is_unchanged_and_the_layer_state_is_restored():
    kw = dict(
        kernel_size=1,
        kq_conf=fixed(6, 1),
        bq_conf=fixed(5, 1),
        iq_conf=act(2, 3),
        oq_conf=act(3, 3),
        activation="relu",
    )
    layer = hgly.QConv2d(4, **kw)
    inp = keras.Input(X.shape[1:])
    model = keras.Model(inp, layer(inp))
    before = np_(model.predict(X, verbose=0))
    model.compile(keras.optimizers.SGD(0.0), "mse")  # a training step that changes nothing
    model.fit(X, np.zeros_like(before), epochs=1, verbose=0)
    assert float(np_(layer.ebops)) > 0  # the EBOPs pass did run
    assert layer._ebops_pass is False
    assert layer.kwargs.get("bias") is not None  # the quantized bias, not the EBOPs pass's None
    np.testing.assert_allclose(np_(model.predict(X, verbose=0)), before, atol=1e-6)
    np.testing.assert_allclose(np_(layer(X)), before, atol=1e-6)
    assert before.min() >= 0 and (before > 0).any()  # the ReLU is still there


def test_ebops_factor_scales_the_reported_ebops_and_the_loss():
    base = dict(kernel_size=1, kq_conf=fixed(6, 1), iq_conf=act(2, 3))
    a, b = hgly.QConv2d(4, **base), hgly.QConv2d(4, ebops_factor=3.0, **base)
    _fit_one_step(a, X)
    _fit_one_step(b, X)
    assert float(np_(b.ebops)) == 3.0 * float(np_(a.ebops))


def test_enable_ebops_false_survives_get_config_and_clone():
    layer = hgly.QConv2d(
        4, kernel_size=1, kq_conf=fixed(6, 1), iq_conf=act(2, 3), enable_ebops=False
    )
    assert hgly.QConv2d.from_config(layer.get_config()).enable_ebops is False
    inp = keras.Input((H, W, C))
    m = keras.Model(inp, layer(inp))
    assert keras.models.clone_model(m).layers[1].enable_ebops is False


def test_global_enable_ebops_false_is_respected():
    from hgq.config.layer import global_config

    was = global_config["enable_ebops"]
    try:
        global_config["enable_ebops"] = False
        assert hgly.QConv2d(4, kernel_size=1, iq_conf=act(2, 3)).enable_ebops is False
    finally:
        global_config["enable_ebops"] = was


def test_default_kernel_quantizer_is_hgq2s_weight_config():
    ref = hgly.QConv2d(4, kernel_size=1, kq_conf=QuantizerConfig("default", "weight"))
    ref(X)
    default = hgly.QConv2d(4, kernel_size=1)
    default(X)
    q, r = default._kqs[0], ref._kqs[0]
    shape = keras.ops.shape(default._kernel_sources()[0])
    assert (type(q.quantizer), q.quantizer.round_mode, q.quantizer.overflow_mode) == (
        type(r.quantizer),
        r.quantizer.round_mode,
        r.quantizer.overflow_mode,
    )
    np.testing.assert_array_equal(np_(q.bits_(shape)), np_(r.bits_(shape)))  # the same bit widths


def test_quantized_bias_is_none_without_a_bias():
    layer = hgly.QConv2d(4, kernel_size=1, use_bias=False, kq_conf=fixed(6, 1))
    layer(X)
    assert layer.quantized_bias() is None and layer.bias_type() is None


def test_activation_type_is_signed_when_any_alive_element_is_signed():
    layer, _ = _qb(hgly.QConv2d, 1, C, iq=PER_ELEMENT_KIF)
    q = {v.name: v for v in layer._iq.variables}
    k = np.zeros(q["k"].shape, "float32")
    k.flat[0] = 1.0  # one signed element among unsigned ones
    q["k"].assign(k)
    assert layer.input_type().startswith("ap_fixed<")  # not ap_ufixed


@pytest.mark.parametrize("activation", ["sigmoid", "tanh", "softplus", "exponential"])
def test_ebops_do_not_depend_on_the_activation_of_the_layer(activation):
    kw = dict(kernel_size=1, kq_conf=fixed(6, 1), iq_conf=act(2, 3))
    a, b = hgly.QConv2d(4, **kw), hgly.QConv2d(4, activation=activation, **kw)
    _fit_one_step(a, X)
    _fit_one_step(b, X)
    assert float(np_(a.ebops)) == float(np_(b.ebops))


@pytest.mark.skipif(
    keras.backend.backend() == "tensorflow", reason="HGQ2 has no half precision on TF"
)
def test_ebops_in_half_precision_train():
    layer = hgly.QConv2d(
        4,
        kernel_size=1,
        dtype="float16",
        kq_conf=fixed(6, 1),
        bq_conf=fixed(5, 1),
        iq_conf=act(2, 3),
    )
    _fit_one_step(layer, X)
    assert float(np_(layer.ebops)) > 0


def test_ebops_saturate_in_the_uint32_variable_instead_of_wrapping():
    layer = hgly.QConv2d(4, kernel_size=1, kq_conf=fixed(6, 1), iq_conf=act(2, 3), ebops_factor=1e9)
    inp = keras.Input((H, W, C))
    out = layer(inp, training=True)
    keras.Model(inp, out)
    layer(X, training=True)
    assert int(np_(layer.ebops)) == 4294967040
    assert all(np.isfinite(float(np_(l))) for l in layer.losses)


@pytest.mark.skipif(
    keras.backend.backend() != "tensorflow", reason="sparse inputs are a TF feature"
)
def test_freeze_keeps_the_sparse_flag_of_the_inputs():
    import tensorflow as tf

    inp = keras.Input((H, W, C), sparse=True)
    dense = keras.layers.Lambda(lambda t: tf.sparse.to_dense(t), output_shape=(H, W, C))(inp)
    model = keras.Model(inp, hgly.QConv2d(2, kernel_size=1, kq_conf=fixed(6, 1))(dense))
    frozen, _ = hgly.freeze(model)
    assert frozen.inputs[0].sparse


@needs_hls
def test_apply_quantized_types_says_when_a_producer_keeps_its_own_result_type():
    pytest.importorskip("hls4ml")
    import hls4ml

    from keras_hexagdly.hls4ml_ext import apply_quantized_types, patch_model_for_hls
    from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

    register_hex_gather_layers("Vitis")
    inp = keras.Input((H, W, C), name="x")
    first = hgly.QConv2d(
        4, kernel_size=1, name="c1", kq_conf=fixed(6, 1), iq_conf=act(2, 3), oq_conf=act(3, 2, k=0)
    )
    second = hgly.QConv2d(  # reads c1's output with other widths than c1's output quantizer
        2, kernel_size=1, name="c2", kq_conf=fixed(6, 1), iq_conf=act(3, 3, k=0)
    )
    frozen, types = hgly.freeze(keras.Model(inp, second(first(inp))), activation_quantizers="types")
    patched = patch_model_for_hls(frozen)
    config = hls4ml.utils.config_from_keras_model(patched, granularity="name", backend="Vitis")
    with pytest.warns(UserWarning, match="kept over"):
        apply_quantized_types(config, patched, types)
    assert config["LayerName"]["c1_linebuffer"]["Precision"]["result"] == types["c1"]["result"]


@needs_hls
def test_apply_quantized_types_refuses_consumers_that_want_different_input_types():
    pytest.importorskip("hls4ml")
    import hls4ml

    from keras_hexagdly.hls4ml_ext import apply_quantized_types, patch_model_for_hls
    from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

    register_hex_gather_layers("Vitis")
    inp = keras.Input((H, W, C), name="x")
    left = hgly.QConv2d(2, kernel_size=1, name="left", kq_conf=fixed(6, 1), iq_conf=act(2, 3))
    right = hgly.QConv2d(2, kernel_size=1, name="right", kq_conf=fixed(6, 1), iq_conf=act(3, 4))
    model = keras.Model(inp, [left(inp), right(inp)])
    frozen, types = hgly.freeze(model, activation_quantizers="types")
    patched = patch_model_for_hls(frozen)
    config = hls4ml.utils.config_from_keras_model(patched, granularity="name", backend="Vitis")
    with pytest.raises(ValueError, match="different input types"):
        apply_quantized_types(config, patched, types)


# ---- audit final: clear errors at the door ----


def test_freeze_refuses_what_is_not_a_model():
    with pytest.raises(TypeError, match="freeze expects a Keras model"):
        hgly.freeze("model")


@pytest.mark.parametrize(
    "kwargs", [{"iq_conf": "foo"}, {"kq_conf": {"a": 1}}, {"bq_conf": 3}, {"oq_conf": object()}]
)
def test_a_quantizer_config_of_the_wrong_type_is_refused_at_construction(kwargs):
    with pytest.raises(TypeError, match="QuantizerConfig"):
        hgly.QConv2d(4, kernel_size=1, **kwargs)


def test_a_negative_beta_is_refused():
    with pytest.raises(ValueError, match="beta0 must be >= 0"):
        hgly.QConv2d(4, kernel_size=1, iq_conf=act(2, 3), beta0=-1.0)


@pytest.mark.skipif(keras.backend.backend() != "tensorflow", reason="a TensorFlow limitation")
def test_half_precision_on_tensorflow_is_explained():
    keras.config.set_dtype_policy("mixed_float16")
    try:
        with pytest.raises(ValueError, match="do not support half precision"):
            hgly.QConv2d(4, kernel_size=1, kq_conf=fixed(6, 1))
    finally:
        keras.config.set_dtype_policy("float32")


@needs_hls
def test_patch_explains_the_standalone_quantizers_when_hls4ml_lacks_the_bit_exact_flow(monkeypatch):
    from keras_hexagdly import hls4ml_ext
    from keras_hexagdly.hls4ml_ext import patch_model_for_hls

    inp = keras.Input((H, W, C))
    model = keras.Model(
        inp, hgly.QConv2d(2, kernel_size=1, kq_conf=fixed(6, 1), iq_conf=act(2, 3))(inp)
    )
    kept, _ = hgly.freeze(model)  # the default: the activation quantizers stay as layers
    patch_model_for_hls(kept)  # fine with an hls4ml that has the bit-exact flow
    monkeypatch.setattr(hls4ml_ext, "_hls4ml_has_bit_exact_flow", lambda: False)
    with pytest.raises(ValueError, match="bit-exact precision flow"):
        patch_model_for_hls(kept)
    patch_model_for_hls(hgly.freeze(model, activation_quantizers="types")[0])
