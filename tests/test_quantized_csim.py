"""C simulation of a quantized hex convolution: the hls4ml export must compute what Keras computes.

Each case compiles a one-layer hls4ml project (about a minute). The input, the weights, the bias and
the output are quantized in the Keras model; ``freeze`` gives the types to put in the hls4ml config.
"""

import re
import shutil

import numpy as np
import pytest

hls4ml = pytest.importorskip("hls4ml")
keras = pytest.importorskip("keras")
pytest.importorskip("hgq")
from hgq.quantizer import QuantizerConfig  # noqa: E402

import keras_hexagdly as hgly  # noqa: E402
from keras_hexagdly.hls4ml_ext import hex_reuse_config, patch_model_for_hls  # noqa: E402
from keras_hexagdly.hls4ml_handler import register_hex_gather_layers  # noqa: E402
from keras_hexagdly.quantized import _parse_type  # noqa: E402

pytestmark = [
    pytest.mark.skipif(shutil.which("g++") is None, reason="C simulation needs g++"),
    pytest.mark.skipif(
        keras.backend.backend() == "jax",
        reason="the hls4ml export targets the tensorflow / torch backends",
    ),
]

H, W, C = 13, 11, 2
WIDE = "ap_fixed<32,12>"


def weight_conf(bits, ints):
    return QuantizerConfig("kbi", "weight", k0=1, b0=bits, i0=ints, trainable=False)


def act_conf(ints, fracs, k=1):
    """One bit width for the whole layer: the only activation quantizer io_stream can export."""
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


def simulate(
    tmp_path, *, iq, oq, accum, kq=None, bq=None, channels=C, kernel_size=1, x=None, prepare=None
):
    """(C simulation, Keras QAT model) outputs, and the accumulator type that was generated.

    ``prepare(layer)`` may set the weights, ``x`` the input data."""
    register_hex_gather_layers("Vitis")
    if x is None:
        x = np.random.default_rng(0).normal(0, 1.5, (6, H, W, channels)).astype("float32")
    grid = keras.Input((H, W, channels), name="x")
    layer = hgly.QConv2d(
        4,
        kernel_size=kernel_size,
        activation="relu",
        kq_conf=kq or weight_conf(6, 1),
        bq_conf=bq or weight_conf(5, 1),
        iq_conf=iq,
        oq_conf=oq,
        name="hexq",
    )
    model = keras.Model(grid, layer(grid))
    if prepare is not None:
        prepare(layer)
    reference = keras.ops.convert_to_numpy(model(x))

    frozen, types = hgly.freeze(model, activation_quantizers="types")
    types = types["hexq"]
    patched = patch_model_for_hls(frozen)
    config = hls4ml.utils.config_from_keras_model(patched, granularity="name", backend="Vitis")
    hex_reuse_config(config, patched)
    config["Model"]["Precision"] = WIDE
    if iq is not None:
        config["LayerName"]["x"]["Precision"] = {"result": types["input"]}
    config["LayerName"]["hexq_linebuffer"]["Precision"] = {
        "result": WIDE,
        "mac_weights": types["weight"],
        "mac_bias": types["bias"],
    }
    if accum == "layer":
        config["LayerName"]["hexq_linebuffer"]["AccumPrecision"] = types["accum"]
    elif accum == "model":
        config["Model"]["ExactAccumulator"] = True  # the global switch, nothing per layer
    if oq is not None:
        config["LayerName"]["hexq_activation"]["Precision"] = {"result": types["result"]}
    hls_model = hls4ml.converters.convert_from_keras_model(
        patched,
        hls_config=config,
        io_type="io_stream",
        backend="Vitis",
        output_dir=str(tmp_path / "prj"),
        part="xc7vx485tffg1927-2",
        clock_period=5,
        project_name="m",
    )
    hls_model.compile()
    parameters = (tmp_path / "prj" / "firmware" / "parameters.h").read_text()
    generated = re.findall(r"typedef (ap_u?fixed<\d+, *\d+>) accum_t", parameters)[0]
    simulated = hls_model.predict(np.ascontiguousarray(x)).reshape(reference.shape)
    return simulated, reference, generated, types


def test_the_c_simulation_equals_the_keras_model_with_every_quantizer(tmp_path):
    simulated, reference, generated, types = simulate(
        tmp_path, iq=act_conf(2, 3), oq=act_conf(3, 2, k=0), accum="layer"
    )
    assert generated.replace(" ", "") == types["accum"]
    np.testing.assert_array_equal(simulated, reference)


def test_without_accum_precision_narrow_inputs_lose_the_low_bits_of_each_product(tmp_path):
    """The default accumulator keeps the fractional bits of the finer operand only: documented, and
    the reason ``AccumPrecision`` exists."""
    simulated, reference, generated, types = simulate(
        tmp_path, iq=act_conf(2, 3), oq=act_conf(3, 2, k=0), accum=False
    )
    assert generated.replace(" ", "") != types["accum"]
    assert not np.array_equal(simulated, reference)


def test_a_quantized_output_and_wide_input_need_no_accumulator_setting(tmp_path):
    simulated, reference, _, _ = simulate(tmp_path, iq=None, oq=act_conf(3, 2, k=0), accum=False)
    np.testing.assert_array_equal(simulated, reference)


def test_the_exact_accumulator_switch_gives_the_same_accumulator_and_the_same_bits(tmp_path):
    """``config["Model"]["ExactAccumulator"] = True``: nothing to set per layer, and the generated
    accumulator is the one ``freeze`` computes."""
    simulated, reference, generated, types = simulate(
        tmp_path, iq=act_conf(2, 3), oq=act_conf(3, 2, k=0), accum="model"
    )
    assert generated.replace(" ", "") == types["accum"]
    np.testing.assert_array_equal(simulated, reference)


def small_weights(layer):
    """Weights well inside +-2^-3, on a 2^-7 grid, as training with HGQ2 leaves them."""
    rng = np.random.default_rng(3)
    for weight in layer._base_kernels:
        weight.assign((rng.uniform(-0.12, 0.12, weight.shape)).astype("float32"))
    layer.bias_tensor.assign(rng.uniform(-0.1, 0.1, (4,)).astype("float32"))


@pytest.mark.parametrize("accum", ["layer", "model"])
def test_weights_with_negative_integer_bits_are_exported_exactly(tmp_path, accum):
    """ap_fixed<5,-2>: the type used to lose its sign on the way, and the C simulation then
    rounded the weights to a coarser grid than Keras (a difference of 0.25 on this layer)."""
    kq, bq = weight_conf(5, -3), weight_conf(5, -3)
    simulated, reference, _, types = simulate(
        tmp_path,
        iq=act_conf(2, 3),
        oq=act_conf(3, 2, k=0),
        accum=accum,
        kq=kq,
        bq=bq,
        prepare=small_weights,
    )
    assert _parse_type(types["weight"])[1] < 0
    np.testing.assert_array_equal(simulated, reference)


@pytest.mark.parametrize("accum", ["layer", "model"])
def test_a_bias_as_large_as_the_products_does_not_overflow_the_accumulator(tmp_path, accum):
    """One product 0.875 * 1.75 = 1.53 plus a bias of 3.5 is 5.03: it needs the carry bit of the
    bias addition (the accumulator used to be ap_fixed<9,3>, which wraps at 4)."""

    def prepare(layer):
        for weight in layer._base_kernels:
            weight.assign(np.full(weight.shape, 0.875, "float32"))
        layer.bias_tensor.assign(np.full((4,), 3.5, "float32"))

    x = np.full((6, H, W, 1), 1.75, "float32")
    simulated, reference, _, _ = simulate(
        tmp_path,
        iq=act_conf(2, 2),
        oq=act_conf(4, 2, k=0),
        accum=accum,
        kq=weight_conf(4, 0),
        bq=weight_conf(5, 3),
        channels=1,
        kernel_size=0,
        x=x,
        prepare=prepare,
    )
    assert reference.max() > 4  # beyond what the old accumulator could hold (it wrapped at 4)
    np.testing.assert_array_equal(simulated, reference)
