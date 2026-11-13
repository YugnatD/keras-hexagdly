"""The hls4ml export of the 3D layers: a depth stage followed by the 2D line-buffer layer.

``patch_model_for_hls`` turns a Conv3d into HexDepthStack + HexConvLineBuffer and a MaxPool3d into
HexDepthPool + HexPoolLineBuffer (see ``hex_gather.py``). These tests pin the Keras side of that:
the patched model computes what the original does for every hex stride, depth stride, depth
padding and weight sharing. The generated C++ is checked in ``test_hex_3d_csim.py``.
"""

import numpy as np
import pytest

keras = pytest.importorskip("keras")

import keras_hexagdly as hgly  # noqa: E402
from keras_hexagdly.hex_gather import (  # noqa: E402
    HexConvLineBuffer,
    HexDepthPool,
    HexDepthStack,
    HexPoolLineBuffer,
)
from keras_hexagdly.hls4ml_ext import patch_model_for_hls  # noqa: E402

D, H, W, C = 6, 7, 8, 3


def _randomize(model, seed=0):
    rng = np.random.default_rng(seed)
    for w in model.weights:
        w.assign(rng.normal(size=w.shape).astype("float32"))


def _compare(layer, depth=D):
    inp = keras.Input((depth, H, W, C))
    model = keras.Model(inp, layer(inp))
    _randomize(model)
    patched = patch_model_for_hls(model)
    x = np.random.default_rng(1).normal(size=(2, depth, H, W, C)).astype("float32")
    reference = keras.ops.convert_to_numpy(model(x))
    exported = keras.ops.convert_to_numpy(patched(x))
    assert exported.shape == reference.shape
    np.testing.assert_allclose(exported, reference, rtol=1e-5, atol=1e-5)
    return patched


@pytest.mark.parametrize("share", [False, "ring"])
@pytest.mark.parametrize("padding", ["valid", "same"])
@pytest.mark.parametrize("depth_stride, hex_stride", [(1, 1), (2, 1), (1, 2), (2, 3)])
@pytest.mark.parametrize("depth_size, hex_size", [(1, 1), (2, 1), (3, 2), (2, 0)])
def test_an_exported_conv3d_equals_the_layer(
    depth_size, hex_size, depth_stride, hex_stride, padding, share
):
    layer = hgly.Conv3d(
        4,
        kernel_size=(depth_size, hex_size),
        strides=(depth_stride, hex_stride),
        depth_padding=padding,
        share_neighbors=share,
    )
    patched = _compare(layer)
    assert [type(x) for x in patched.layers[1:]] == [HexDepthStack, HexConvLineBuffer]


@pytest.mark.parametrize("depth_stride, hex_stride", [(1, 1), (2, 1), (1, 2), (3, 3)])
@pytest.mark.parametrize("depth_size, hex_size", [(1, 1), (2, 1), (3, 2), (2, 0)])
def test_an_exported_maxpool3d_equals_the_layer(depth_size, hex_size, depth_stride, hex_stride):
    layer = hgly.MaxPool3d(kernel_size=(depth_size, hex_size), strides=(depth_stride, hex_stride))
    patched = _compare(layer)
    assert [type(x) for x in patched.layers[1:]] == [HexDepthPool, HexPoolLineBuffer]


def test_an_even_depth_kernel_with_same_padding_pads_as_the_layer_does():
    # 'same' with an even depth size pads one frame more at the end than at the start
    _compare(hgly.Conv3d(2, kernel_size=(4, 1), depth_padding="same"), depth=5)


def test_the_activation_and_the_bias_of_a_conv3d_are_kept():
    _compare(hgly.Conv3d(4, kernel_size=(2, 1), activation="relu", use_bias=True))


def test_the_3d_export_needs_no_opt_in():
    inp = keras.Input((D, H, W, C))
    model = keras.Model(inp, hgly.MaxPool3d((2, 1))(hgly.Conv3d(2, kernel_size=(2, 1))(inp)))
    patch_model_for_hls(model)  # used to raise NotImplementedError without allow_unvalidated=True


def test_an_exported_3d_model_survives_a_save_and_a_load(tmp_path):
    inp = keras.Input((D, H, W, C))
    y = hgly.Conv3d(4, kernel_size=(2, 1), strides=(1, 2), depth_padding="same")(inp)
    y = hgly.MaxPool3d(kernel_size=(2, 1), strides=(2, 1))(y)
    model = keras.Model(inp, y)
    _randomize(model)
    patched = patch_model_for_hls(model)
    path = tmp_path / "patched.keras"
    patched.save(path)
    reloaded = keras.saving.load_model(path)
    x = np.random.default_rng(2).normal(size=(2, D, H, W, C)).astype("float32")
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(reloaded(x)), keras.ops.convert_to_numpy(model(x)), atol=1e-5
    )


def test_the_depth_stack_puts_the_oldest_frame_first():
    x = np.arange(4 * 1 * 1 * 2, dtype="float32").reshape(1, 4, 1, 1, 2)  # frame t holds (2t, 2t+1)
    out = keras.ops.convert_to_numpy(HexDepthStack(depth_size=2)(x))
    assert out.shape == (1, 3, 1, 1, 4)
    np.testing.assert_array_equal(out[0, :, 0, 0], [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7]])
    padded = keras.ops.convert_to_numpy(HexDepthStack(depth_size=3, pad_front=1, pad_back=1)(x))
    np.testing.assert_array_equal(padded[0, 0, 0, 0], [0, 0, 0, 1, 2, 3])  # a zero frame in front
    np.testing.assert_array_equal(padded[0, 3, 0, 0], [4, 5, 6, 7, 0, 0])  # and one at the end


@pytest.mark.parametrize("which", ["batch_normalization", "concatenate"])
def test_layers_hls4ml_mishandles_after_a_3d_layer_are_refused(which):
    """hls4ml 1.3.0 reads the scale and bias of a BatchNormalization out of bounds on a
    (D, H, W, C) tensor (wrong values, no error) and has no 4D concatenation."""
    inp = keras.Input((D, H, W, C))
    y = hgly.Conv3d(2, kernel_size=(2, 1))(inp)
    if which == "batch_normalization":
        y = keras.layers.BatchNormalization()(y)
    else:
        y = keras.layers.Concatenate()([y, hgly.Conv3d(2, kernel_size=(2, 1))(inp)])
    with pytest.raises(NotImplementedError, match="3D layer"):
        patch_model_for_hls(keras.Model(inp, y))
    # after a Flatten (or on a 2D layer) the same layers are left alone
    flat = keras.layers.Flatten()(hgly.Conv3d(2, kernel_size=(2, 1))(inp))
    patch_model_for_hls(keras.Model(inp, keras.layers.BatchNormalization()(flat)))


def test_check_hls_config_refuses_the_3d_layers_under_io_parallel():
    from keras_hexagdly.hls4ml_ext import check_hls_config

    inp = keras.Input((D, H, W, C))
    patched = patch_model_for_hls(keras.Model(inp, hgly.MaxPool3d((2, 1))(inp)))
    with pytest.raises(NotImplementedError, match="io_stream"):
        check_hls_config({"Model": {}}, patched, io_type="io_parallel", allow_unvalidated=True)
    assert check_hls_config({"Model": {}}, patched) == []


def test_quantized_types_of_a_3d_model_go_where_the_data_is_narrow():
    """The input type of a 3D layer belongs on what feeds its depth stage (put on the depth stage
    itself, the delay line stored the unquantized frames), and a max pooling without an output
    quantizer keeps the type of its input instead of a wide one."""
    pytest.importorskip("hgq")
    from hgq.quantizer import QuantizerConfig

    from keras_hexagdly.hls4ml_ext import apply_quantized_types

    def act(ints, fracs):
        return QuantizerConfig(
            "kif", "datalane", k0=1, i0=ints, f0=fracs, overflow_mode="SAT", round_mode="RND",
            homogeneous_axis=(0, 1, 2, 3, 4), trainable=False,
        )  # fmt: skip

    inp = keras.Input((D, H, W, C), name="x")
    y = hgly.QConv3d(
        2, kernel_size=(2, 1), name="c1", iq_conf=act(2, 3), oq_conf=act(3, 2),
        kq_conf=QuantizerConfig("kbi", "weight", k0=1, b0=6, i0=1, trainable=False),
    )(inp)  # fmt: skip
    y = hgly.QMaxPool3d(kernel_size=(2, 1), name="p1", iq_conf=act(3, 2))(y)
    frozen, types = hgly.freeze(keras.Model(inp, y), activation_quantizers="types")
    patched = patch_model_for_hls(frozen)
    config = apply_quantized_types({"LayerName": {}}, patched, types)["LayerName"]
    assert config["x"]["Precision"]["result"] == types["c1"]["input"]
    assert "c1_depth" not in config and "p1_depth" not in config
    assert config["c1_linebuffer"]["Precision"]["result"] == types["c1"]["result"]
    assert "Precision" not in config.get("p1_linebuffer", {})
