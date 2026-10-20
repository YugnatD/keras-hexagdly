"""Dataflow kernel of the hex convolution (io_stream): same results as the fused kernel, selectable.

The dataflow kernel duplicates the traversal of the fused one, so these tests compile one-layer
hls4ml projects both ways and compare the C simulations bit for bit (and with Keras). Each case
compiles two projects (about a minute).
"""

import shutil
import subprocess

import numpy as np
import pytest

hls4ml = pytest.importorskip("hls4ml")
keras = pytest.importorskip("keras")

import keras_hexagdly as hgly  # noqa: E402
from keras_hexagdly.hls4ml_ext import patch_model_for_hls  # noqa: E402
from keras_hexagdly.hls4ml_handler import register_hex_gather_layers  # noqa: E402

needs_gxx = pytest.mark.skipif(shutil.which("g++") is None, reason="C simulation needs g++")

CIN, COUT = 2, 3


def hex_model(kernel_size, strides, share_neighbors, height=7, width=6):
    grid = keras.Input((height, width, CIN), name="x")
    conv = hgly.Conv2d(
        COUT,
        kernel_size=kernel_size,
        strides=strides,
        use_bias=True,
        share_neighbors=share_neighbors,
        name="c",
    )
    model = keras.Model(grid, conv(grid))
    rng = np.random.default_rng(7)
    for weight in conv.trainable_variables:
        weight.assign((rng.integers(-8, 9, weight.shape) / 8).astype("float32"))
    return model


def convert(tmp_path, model, *, dataflow, io_type="io_stream", reuse=1, tag="prj"):
    register_hex_gather_layers("Vitis")
    patched = patch_model_for_hls(model)
    config = hls4ml.utils.config_from_keras_model(
        patched, granularity="name", backend="Vitis", default_precision="ap_fixed<24,12>"
    )
    config["Model"].update(Strategy="Latency", ReuseFactor=1)
    config["LayerName"]["c_linebuffer"] = {"ReuseFactor": reuse, "Strategy": "Latency"}
    if dataflow is not None:
        config["LayerName"]["c_linebuffer"]["Dataflow"] = dataflow
    return hls4ml.converters.convert_from_keras_model(
        patched,
        hls_config=config,
        io_type=io_type,
        backend="Vitis",
        output_dir=str(tmp_path / tag),
        part="xc7vx485tffg1927-2",
        clock_period=5,
        project_name="m",
    )


def parameters(tmp_path, tag):
    return (tmp_path / tag / "firmware" / "parameters.h").read_text()


@pytest.mark.parametrize("flag,expected", [(None, "false"), (False, "false"), (True, "true")])
def test_the_config_key_reaches_the_kernel_config(tmp_path, flag, expected):
    model = hex_model(1, 1, False)
    convert(tmp_path, model, dataflow=flag).write()
    assert f"static const bool     dataflow     = {expected};" in parameters(tmp_path, "prj")


def test_dataflow_is_ignored_with_a_warning_under_io_parallel(tmp_path):
    model = hex_model(1, 1, False)
    with pytest.warns(UserWarning, match="only exists for io_stream"):
        convert(tmp_path, model, dataflow=True, io_type="io_parallel").write()
    assert "dataflow     = false;" in parameters(tmp_path, "prj")


@needs_gxx
@pytest.mark.parametrize(
    "kernel_size,strides,share_neighbors,reuse,height,width",
    [
        (1, 1, False, 1, 7, 6),
        (1, 2, False, 2, 7, 6),
        (2, 1, False, 1, 7, 6),
        (1, 2, "ring", 1, 7, 6),
        (2, 2, False, 3, 13, 12),  # the strided 19-cell kernel needs a larger grid
    ],
)
def test_dataflow_and_fused_kernels_agree(
    tmp_path, kernel_size, strides, share_neighbors, reuse, height, width
):
    model = hex_model(kernel_size, strides, share_neighbors, height, width)
    x = (np.random.default_rng(1).integers(0, 64, (5, height, width, CIN)) / 4).astype("float32")
    reference = model.predict(x, verbose=0)
    outputs = {}
    for name, flag in (("fused", False), ("dataflow", True)):
        hls_model = convert(tmp_path, model, dataflow=flag, reuse=reuse, tag=name)
        hls_model.compile()
        outputs[name] = hls_model.predict(np.ascontiguousarray(x)).reshape(reference.shape)
        assert f"dataflow     = {'true' if flag else 'false'};" in parameters(tmp_path, name)
        if shutil.which("nm"):  # the flag really selects another code path
            library = next((tmp_path / name / "firmware").glob("*.so"))
            symbols = subprocess.run(
                ["nm", "-C", str(library)], capture_output=True, text=True
            ).stdout
            assert ("hex_lb_mac_stream" in symbols) == flag
            assert ("hex_conv_linebuffer_stream_fused" in symbols) == (not flag)
    np.testing.assert_array_equal(outputs["dataflow"], outputs["fused"])
    assert np.abs(outputs["fused"] - reference).max() < 0.05 * np.abs(reference).max()
