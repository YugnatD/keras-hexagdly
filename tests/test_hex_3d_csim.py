"""The exported 3D layers in C simulation: the generated C++ equals Keras.

A Conv3d / MaxPool3d is exported as a depth stage (``nnet_hex_depth_stream.h``: a delay line of
``depth_size - 1`` frames) followed by the 2D line-buffer kernel run over the frames one after the
other (its ``n_frames``). These tests pin the two together: frame order, depth padding, depth
stride, hex stride, weight sharing, and every strategy of the 2D kernel. The RTL was co-simulated
with Vitis HLS on a small network (see the CHANGELOG); that needs Vitis and is not run here.
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

SHAPE = (5, 8, 7, 2)  # (depth, height, width, channels)


def simulate(
    tmp_path, build, shape=SHAPE, strategy="Resource", reuse=1, dataflow=False, io="io_stream"
):
    register_hex_gather_layers("Vitis")
    rng = np.random.default_rng(0)
    volume = keras.Input(shape, name="x")
    model = keras.Model(volume, build(volume))
    for w in model.weights:  # values on a 1/64 grid: exact in the fixed-point type below
        w.assign(np.round(rng.uniform(-1, 1, w.shape) * 64) / 64)
    x = (np.round(rng.uniform(-1, 1, (3, *shape)) * 64) / 64).astype("float32")
    reference = keras.ops.convert_to_numpy(model(x))
    patched = patch_model_for_hls(model)
    config = hls4ml.utils.config_from_keras_model(
        patched, granularity="name", backend="Vitis", default_precision="ap_fixed<40,16>"
    )
    config["Model"].update(Strategy=strategy, ReuseFactor=reuse)
    for layer in patched.layers:
        config["LayerName"].setdefault(layer.name, {}).update(
            Strategy=strategy, ReuseFactor=reuse, Dataflow=dataflow
        )
    hls_model = hls4ml.converters.convert_from_keras_model(
        patched,
        hls_config=config,
        io_type=io,
        backend="Vitis",
        output_dir=str(tmp_path / "prj"),
        part="xc7vx485tffg1927-2",
        clock_period=5,
        project_name="m",
    )
    hls_model.compile()
    simulated = hls_model.predict(np.ascontiguousarray(x))
    return simulated.reshape(reference.shape), reference


@pytest.mark.parametrize("share", [False, "ring"])
@pytest.mark.parametrize(
    "kernel, strides, padding",
    [
        ((2, 1), (1, 1), "valid"),
        ((3, 2), (1, 1), "same"),
        ((2, 1), (2, 3), "same"),
        ((3, 1), (2, 2), "valid"),
        ((1, 1), (1, 2), "valid"),
        ((4, 0), (1, 1), "same"),
    ],
)
def test_the_conv3d_equals_keras(tmp_path, kernel, strides, padding, share):
    simulated, reference = simulate(
        tmp_path,
        lambda x: hgly.Conv3d(
            3, kernel_size=kernel, strides=strides, depth_padding=padding, share_neighbors=share
        )(x),
    )
    np.testing.assert_allclose(simulated, reference, atol=1e-6)


@pytest.mark.parametrize(
    "kernel, strides", [((2, 1), (1, 1)), ((2, 1), (2, 2)), ((3, 2), (1, 3)), ((2, 0), (3, 1))]
)
def test_the_maxpool3d_equals_keras(tmp_path, kernel, strides):
    simulated, reference = simulate(
        tmp_path, lambda x: hgly.MaxPool3d(kernel_size=kernel, strides=strides)(x), (6, 8, 7, 2)
    )
    np.testing.assert_array_equal(simulated, reference)


@pytest.mark.parametrize(
    "strategy, reuse, dataflow",
    [
        ("Latency", 1, False),
        ("Latency", 4, False),
        ("Resource", 4, False),
        ("Resource", 1, True),
        ("Resource", 4, True),
    ],
)
def test_every_strategy_of_the_2d_kernel_runs_the_frames(tmp_path, strategy, reuse, dataflow):
    simulated, reference = simulate(
        tmp_path,
        lambda x: hgly.Conv3d(3, kernel_size=(2, 1), strides=(1, 2), share_neighbors="ring")(x),
        strategy=strategy,
        reuse=reuse,
        dataflow=dataflow,
    )
    np.testing.assert_allclose(simulated, reference, atol=1e-6)


@pytest.mark.parametrize(
    "build, shape",
    [
        (lambda x: hgly.Conv3d(2, kernel_size=(5, 1))(x), (5, 6, 5, 2)),  # one output frame
        (lambda x: hgly.Conv3d(2, kernel_size=(2, 1), strides=(3, 1))(x), (7, 6, 5, 2)),
        (lambda x: hgly.Conv3d(2, kernel_size=(4, 1), strides=(5, 1), depth_padding="same")(x),
         (7, 6, 5, 2)),
        (lambda x: hgly.Conv3d(2, kernel_size=(3, 1), depth_padding="same")(x), (1, 6, 5, 2)),
        (lambda x: hgly.Conv3d(2, kernel_size=(2, 1))(x), (3, 6, 1, 2)),  # one column
        (lambda x: hgly.Conv3d(2, kernel_size=(2, 1))(x), (3, 1, 6, 2)),  # one row
        (lambda x: hgly.Conv3d(2, kernel_size=(2, 3), kernel_size_unit="pixels")(x), (3, 6, 5, 2)),
        (lambda x: hgly.MaxPool3d(kernel_size=(2, 1), strides=(5, 1))(x), (7, 6, 5, 2)),
        (lambda x: hgly.MaxPool3d(kernel_size=(6, 1))(x), (6, 3, 3, 2)),  # one output frame
    ],
    ids=["one-frame-out", "stride>size", "stride>size-same", "one-frame-in", "one-column",
         "one-row", "pixels", "pool-stride>size", "pool-one-frame-out"],
)  # fmt: skip
def test_edge_sizes_of_the_depth_axis_and_of_the_grid(tmp_path, build, shape):
    simulated, reference = simulate(tmp_path, build, shape)
    np.testing.assert_allclose(simulated, reference, atol=1e-6)


def test_a_small_3d_network_equals_keras(tmp_path):
    def network(x):
        y = hgly.Conv3d(4, kernel_size=(2, 1), activation="relu")(x)
        y = hgly.MaxPool3d(kernel_size=(2, 1), strides=(2, 2))(y)
        y = hgly.Conv3d(3, kernel_size=(2, 1), depth_padding="same", share_neighbors="ring")(y)
        return keras.layers.Dense(3)(keras.layers.Flatten()(y))

    simulated, reference = simulate(tmp_path, network, (6, 8, 6, 2), reuse=4, dataflow=True)
    np.testing.assert_allclose(simulated, reference, atol=1e-4)  # float32 sums on the Keras side


def test_the_3d_layers_are_refused_under_io_parallel(tmp_path):
    with pytest.raises(NotImplementedError, match="io_stream only"):
        simulate(tmp_path, lambda x: hgly.Conv3d(2, kernel_size=(2, 1))(x), io="io_parallel")
