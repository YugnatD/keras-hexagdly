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


def simulate(tmp_path, layer, height, width):
    register_hex_gather_layers("Vitis")
    grid = keras.Input((height, width, 2), name="x")
    model = keras.Model(grid, layer(grid))
    x = np.random.default_rng(0).uniform(-1, 1, (3, height, width, 2)).astype("float32")
    reference = keras.ops.convert_to_numpy(model(x))
    patched = patch_model_for_hls(model)
    config = hls4ml.utils.config_from_keras_model(
        patched, granularity="name", backend="Vitis", default_precision="ap_fixed<32,10>"
    )
    config["Model"].update(Strategy="Latency", ReuseFactor=1)
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
    try:
        simulated, reference = simulate(tmp_path, layer, height, width)
    except ValueError as error:  # a grid too small for the window: nothing to export
        pytest.skip(str(error)[:60])
    np.testing.assert_allclose(simulated, reference, atol=1e-4)


@pytest.mark.parametrize("height, width", GRIDS)
@pytest.mark.parametrize("kernel, stride", CASES)
def test_the_max_pooling_equals_keras_for_every_stride(tmp_path, kernel, stride, height, width):
    layer = hgly.MaxPool2d(kernel_size=kernel, strides=stride, name="p")
    try:
        simulated, reference = simulate(tmp_path, layer, height, width)
    except ValueError as error:
        pytest.skip(str(error)[:60])
    np.testing.assert_allclose(simulated, reference, atol=1e-4)
