"""C simulation of an exported hex convolution: its accumulator must not lose or wrap anything.

Each case compiles a one-layer hls4ml project (about a minute), so they are kept to the three ways
the accumulator used to be wrong: activations larger than the weight type implies, an input finer
than the weights, and a bias wider than the sum of the products.
"""

import re
import shutil

import numpy as np
import pytest

hls4ml = pytest.importorskip("hls4ml")
keras = pytest.importorskip("keras")

import keras_hexagdly as hgly  # noqa: E402
from keras_hexagdly.hls4ml_ext import patch_model_for_hls  # noqa: E402
from keras_hexagdly.hls4ml_handler import register_hex_gather_layers  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("g++") is None, reason="C simulation needs g++")

CENTRE_SHAPE, RING_SHAPE = (3, 1, 1, 2), (2, 2, 1, 2)


def simulate(tmp_path, weights, bias, x, *, input_p, weight_p, bias_p):
    """(hls4ml C simulation, Keras) outputs of a 4x4-grid hex convolution, and its accumulator."""
    register_hex_gather_layers("Vitis")
    grid = keras.Input((4, 4, 1), name="x")
    conv = hgly.Conv2d(2, kernel_size=1, use_bias=True, share_neighbors=False, name="c")
    model = keras.Model(grid, conv(grid))
    conv.set_weights([*weights, bias])
    patched = patch_model_for_hls(model)
    config = hls4ml.utils.config_from_keras_model(
        patched, granularity="name", backend="Vitis", default_precision="ap_fixed<24,12>"
    )
    config["Model"].update(Strategy="Latency", ReuseFactor=1)
    config["LayerName"]["x"] = {"Precision": {"result": input_p}}
    config["LayerName"]["c_linebuffer"] = {
        "Precision": {"result": "ap_fixed<24,12>", "mac_weights": weight_p, "mac_bias": bias_p}
    }
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
    reference = model.predict(x, verbose=0)
    parameters = (tmp_path / "prj" / "firmware" / "parameters.h").read_text()
    accum = re.findall(r"typedef (ap_u?fixed<\d+, *\d+>) accum_t", parameters)[0]
    return hls_model.predict(np.ascontiguousarray(x)).reshape(reference.shape), reference, accum


def multiples_of(rng, step, shape):
    return (rng.integers(-8, 9, shape) * step).astype("float32")


def test_large_activations_do_not_wrap(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.integers(0, 500, (20, 4, 4, 1)).astype("float32") / 10  # up to 50
    weights = [multiples_of(rng, 0.25, CENTRE_SHAPE), multiples_of(rng, 0.25, RING_SHAPE)]
    out, ref, accum = simulate(
        tmp_path,
        weights,
        np.zeros(2, "float32"),
        x,
        input_p="ap_fixed<18,7>",
        weight_p="ap_fixed<16,3>",
        bias_p="ap_fixed<20,8>",
    )
    assert np.abs(out - ref).max() < 0.05 * np.abs(ref).max(), accum


def test_input_finer_than_the_weights_is_not_truncated(tmp_path):
    rng = np.random.default_rng(1)
    x = (rng.integers(0, 2**11, (20, 4, 4, 1)) / 2**11).astype("float32")  # 11 fractional bits
    weights = [multiples_of(rng, 1 / 8, CENTRE_SHAPE), multiples_of(rng, 1 / 8, RING_SHAPE)]
    out, ref, accum = simulate(
        tmp_path,
        weights,
        np.zeros(2, "float32"),
        x,
        input_p="ap_fixed<18,2>",
        weight_p="ap_fixed<8,3>",
        bias_p="ap_fixed<16,4>",
    )
    assert np.abs(out - ref).max() < 0.01, accum


def test_bias_wider_than_the_products_does_not_wrap(tmp_path):
    rng = np.random.default_rng(2)
    x = rng.integers(0, 2, (20, 4, 4, 1)).astype("float32")
    weights = [np.full(shape, 0.25, "float32") for shape in (CENTRE_SHAPE, RING_SHAPE)]
    out, ref, accum = simulate(
        tmp_path,
        weights,
        np.array([1000.0, -1000.0], "float32"),
        x,
        input_p="ap_fixed<4,2>",
        weight_p="ap_fixed<16,3>",
        bias_p="ap_fixed<24,12>",
    )
    assert np.abs(out - ref).max() < 0.1, accum
