"""The io_stream line-buffer kernels (hex convolution and pooling): C simulation equals Keras.

The kernels emit each output pixel from the loop indices alone: output row (r - 1 - dr_max) / stride
and column (c - dc_max) / stride. Carrying a counter from one iteration to the next instead held the
loop at II=2 in Vitis (twice the latency); these tests pin the indexing for every stride, which is
where a mistake in that arithmetic would show. The II itself needs Vitis HLS: see the CHANGELOG.
"""

import shutil

import numpy as np
import pytest

hls4ml = pytest.importorskip("hls4ml")
keras = pytest.importorskip("keras")

import keras_hexagdly as hgly  # noqa: E402
from keras_hexagdly.hls4ml_ext import patch_model_for_hls  # noqa: E402
from keras_hexagdly.hls4ml_handler import register_hex_gather_layers  # noqa: E402

pytestmark = [
    pytest.mark.skipif(shutil.which("g++") is None, reason="C simulation needs g++"),
    pytest.mark.skipif(
        keras.backend.backend() == "jax", reason="the hls4ml export targets tensorflow / torch"
    ),
]

# (height, width): odd and even sizes in both directions, big enough for stride 3
GRIDS = [(9, 8), (8, 11)]
CASES = [(kernel, stride) for kernel in (0, 1, 2) for stride in (1, 2, 3)]
# Grids with no fully-interior output pixel of one column parity (or of either): the late layers
# of a CNN. The tap offsets are then measured on a larger virtual grid (they do not depend on it).
SMALL = [(3, 3, 1, 1), (5, 5, 2, 1), (4, 5, 1, 2), (2, 2, 1, 1), (3, 1, 1, 1)]


def simulate(tmp_path, layer, height, width, strategy="Latency", reuse=1):
    register_hex_gather_layers("Vitis")
    grid = keras.Input((height, width, 2), name="x")
    model = keras.Model(grid, layer(grid))
    x = np.random.default_rng(0).uniform(-1, 1, (3, height, width, 2)).astype("float32")
    reference = keras.ops.convert_to_numpy(model(x))
    patched = patch_model_for_hls(model)
    config = hls4ml.utils.config_from_keras_model(
        patched, granularity="name", backend="Vitis", default_precision="ap_fixed<32,10>"
    )
    config["Model"].update(Strategy=strategy, ReuseFactor=reuse)
    for name in config["LayerName"]:
        config["LayerName"][name].update(Strategy=strategy, ReuseFactor=reuse)
    for hex_layer in patched.layers:  # the entry of a hex layer has no Strategy / ReuseFactor
        config["LayerName"].setdefault(hex_layer.name, {}).update(
            Strategy=strategy, ReuseFactor=reuse
        )
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
    simulated = hls_model.predict(np.ascontiguousarray(x))
    return simulated.reshape(reference.shape), reference


@pytest.mark.parametrize("height, width", GRIDS)
@pytest.mark.parametrize("kernel, stride", CASES)
def test_the_convolution_equals_keras_for_every_stride(tmp_path, kernel, stride, height, width):
    layer = hgly.Conv2d(3, kernel_size=kernel, strides=stride, name="c")
    simulated, reference = simulate(tmp_path, layer, height, width)
    np.testing.assert_allclose(simulated, reference, atol=1e-4)


@pytest.mark.parametrize("height, width", GRIDS)
@pytest.mark.parametrize("kernel, stride", CASES)
def test_the_max_pooling_equals_keras_for_every_stride(tmp_path, kernel, stride, height, width):
    layer = hgly.MaxPool2d(kernel_size=kernel, strides=stride, name="p")
    simulated, reference = simulate(tmp_path, layer, height, width)
    np.testing.assert_allclose(simulated, reference, atol=1e-4)


@pytest.mark.parametrize("height, width, kernel, stride", SMALL)
def test_small_grids_are_exported(tmp_path, height, width, kernel, stride):
    """They used to stop the conversion ("Grid too small"): no interior pixel to measure from."""
    conv = hgly.Conv2d(3, kernel_size=kernel, strides=stride, name="c")
    simulated, reference = simulate(tmp_path / "conv", conv, height, width)
    np.testing.assert_allclose(simulated, reference, atol=1e-4)
    pool = hgly.MaxPool2d(kernel_size=kernel, strides=stride, name="p")
    simulated, reference = simulate(tmp_path / "pool", pool, height, width)
    np.testing.assert_allclose(simulated, reference, atol=1e-4)


@pytest.mark.parametrize("strategy", ["Latency", "Resource"])
@pytest.mark.parametrize("reuse", [1, 3, 4])
def test_both_strategies_and_any_reuse_factor_equal_keras(tmp_path, strategy, reuse):
    """The two versions of the traversal loop (pipelined or not) and the folded MAC, with a reuse
    factor that divides the 14 ring sums or not, compute the same thing."""
    layer = hgly.Conv2d(3, kernel_size=1, strides=2, name="c")
    simulated, reference = simulate(tmp_path, layer, 9, 8, strategy=strategy, reuse=reuse)
    np.testing.assert_allclose(simulated, reference, atol=1e-4)


def test_the_max_pooling_takes_the_type_of_its_input(tmp_path):
    """Its result type, left to hls4ml, was the model default: a wider input wrapped around."""
    register_hex_gather_layers("Vitis")
    grid = keras.Input((8, 7, 2), name="x")
    conv = hgly.Conv2d(3, kernel_size=1, name="c")
    model = keras.Model(grid, hgly.MaxPool2d(kernel_size=1, strides=2, name="p")(conv(grid)))
    conv.set_weights([np.full(w.shape, 3.0, "float32") for w in conv.get_weights()])
    x = np.random.default_rng(0).uniform(0, 4, (3, 8, 7, 2)).astype("float32")
    reference = keras.ops.convert_to_numpy(model(x))
    assert reference.max() > 64  # beyond the 6 integer bits of the model default
    patched = patch_model_for_hls(model)
    config = hls4ml.utils.config_from_keras_model(
        patched, granularity="name", backend="Vitis", default_precision="ap_fixed<16,6>"
    )
    config["LayerName"]["x"]["Precision"]["result"] = "ap_fixed<24,12>"
    config["LayerName"].setdefault("c_linebuffer", {})["Precision"] = {
        "result": "ap_fixed<24,12>", "mac_weights": "ap_fixed<16,6>", "mac_bias": "ap_fixed<16,6>",
    }  # fmt: skip
    hls_model = hls4ml.converters.convert_from_keras_model(
        patched, hls_config=config, io_type="io_stream", backend="Vitis",
        output_dir=str(tmp_path / "prj"), part="xc7vx485tffg1927-2", clock_period=5,
        project_name="m",
    )  # fmt: skip
    hls_model.compile()
    simulated = hls_model.predict(np.ascontiguousarray(x)).reshape(reference.shape)
    np.testing.assert_allclose(simulated, reference, atol=0.05)
