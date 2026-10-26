"""Export through hls4ml's bit-exact flow: a model with HGQ2 quantizers or layers, no types to give.

When a model contains HGQ2 quantizers (what ``freeze(model)`` keeps) or HGQ2 layers (``QDense``...),
hls4ml derives every type itself, by interval arithmetic from the input quantizer and the values of
the weights. The exported hex layers take part in it (``_register_bit_exact_kif``); these tests
compile the C simulation and require it to equal the Keras model exactly, with nothing set by hand.
"""

import shutil

import numpy as np
import pytest

hls4ml = pytest.importorskip("hls4ml")
keras = pytest.importorskip("keras")
pytest.importorskip("hgq")
from hgq.layers import QDense, QGlobalAveragePooling2D  # noqa: E402
from hgq.quantizer import QuantizerConfig  # noqa: E402

import keras_hexagdly as hgly  # noqa: E402
from keras_hexagdly.hls4ml_ext import patch_model_for_hls  # noqa: E402
from keras_hexagdly.hls4ml_handler import register_hex_gather_layers  # noqa: E402

pytestmark = [
    pytest.mark.skipif(shutil.which("g++") is None, reason="C simulation needs g++"),
    pytest.mark.skipif(
        keras.backend.backend() == "jax", reason="the hls4ml export targets tensorflow / torch"
    ),
]

H, W = 12, 11


def act(ints, fracs, signed=1, axes=(0, 1, 2, 3)):
    """One bit width for the whole layer (what io_stream needs), fixed."""
    return QuantizerConfig(
        "kif", "datalane", k0=signed, i0=ints, f0=fracs, overflow_mode="SAT", round_mode="RND",
        homogeneous_axis=axes, trainable=False,
    )  # fmt: skip


def weights(bits=6, ints=1):
    return QuantizerConfig("kbi", "weight", k0=1, b0=bits, i0=ints, trainable=True)


def head(**kwargs):
    return QDense(
        3, name="head", kq_conf=weights(), iq_conf=act(4, 2, 0, (0, 1)),
        oq_conf=act(5, 3, 1, (0, 1)), **kwargs,
    )  # fmt: skip


def converted_equals_keras(tmp_path, model, tweak=None, signed_input=False):
    """Freeze (default: the quantizers stay), patch, convert with nothing set by hand, simulate."""
    low = -1 if signed_input else 0
    x = (
        np.random.default_rng(0)
        .uniform(low, 1, (8,) + tuple(model.input_shape[1:]))
        .astype("float32")
    )
    reference = keras.ops.convert_to_numpy(model(x))
    frozen, _ = hgly.freeze(model)
    register_hex_gather_layers("Vitis")
    patched = patch_model_for_hls(frozen)
    config = hls4ml.utils.config_from_keras_model(patched, granularity="name", backend="Vitis")
    if tweak is not None:
        tweak(config, patched)
    hls_model = hls4ml.converters.convert_from_keras_model(
        patched, hls_config=config, io_type="io_stream", backend="Vitis",
        output_dir=str(tmp_path / "prj"), part="xc7vx485tffg1927-2", clock_period=5,
    )  # fmt: skip
    hls_model.compile()
    simulated = hls_model.predict(np.ascontiguousarray(x)).reshape(reference.shape)
    return simulated, reference


@pytest.mark.parametrize("signed_output", [0, 1])
def test_a_convolution_with_its_quantizers_is_exact(tmp_path, signed_output):
    inp = keras.Input((H, W, 2), name="x")
    layer = hgly.QConv2d(
        4, kernel_size=1, activation="relu" if signed_output == 0 else None, name="c1",
        kq_conf=weights(), bq_conf=weights(8, 3), iq_conf=act(2, 3, 0),
        oq_conf=act(3, 3, signed_output),
    )  # fmt: skip
    model = keras.Model(inp, keras.layers.Flatten(name="flat")(layer(inp)))
    simulated, reference = converted_equals_keras(tmp_path, model)
    np.testing.assert_array_equal(simulated, reference)


def test_a_max_pooling_is_exact(tmp_path):
    inp = keras.Input((H, W, 2), name="x")
    layer = hgly.QMaxPool2d(
        kernel_size=1, strides=1, name="p1", iq_conf=act(2, 3, 0), oq_conf=act(2, 3, 0)
    )
    model = keras.Model(inp, keras.layers.Flatten(name="flat")(layer(inp)))
    simulated, reference = converted_equals_keras(tmp_path, model)
    np.testing.assert_array_equal(simulated, reference)


def chain_model():
    inp = keras.Input((H, W, 2), name="x")
    x = hgly.QConv2d(
        6, kernel_size=2, share_neighbors=True, activation="relu", name="c1",
        kq_conf=weights(), bq_conf=weights(7, 2), iq_conf=act(2, 3, 0), oq_conf=act(3, 3, 0),
    )(inp)  # fmt: skip
    x = hgly.QMaxPool2d(
        kernel_size=1, strides=2, name="p1", iq_conf=act(3, 3, 0), oq_conf=act(3, 3, 0)
    )(x)
    x = hgly.QConv2d(
        5, kernel_size=1, strides=2, activation="relu", name="c2",
        kq_conf=weights(), iq_conf=act(3, 3, 0), oq_conf=act(4, 2, 0),
    )(x)  # fmt: skip
    return keras.Model(inp, head()(keras.layers.Flatten(name="flat")(x)))


def with_hex_reuse(config, patched):
    from keras_hexagdly.hls4ml_ext import hex_reuse_config

    hex_reuse_config(config, patched, verbose=False)  # the recipe of the README


def with_reuse_factor_4(config, patched):
    for name in ("c1_linebuffer", "c2_linebuffer"):
        config["LayerName"][name].update(ReuseFactor=4, Strategy="Resource")


def with_dataflow(config, patched):
    with_hex_reuse(config, patched)
    for name in ("c1_linebuffer", "c2_linebuffer"):
        config["LayerName"][name]["Dataflow"] = True


@pytest.mark.parametrize(
    "tweak", [None, with_hex_reuse, with_reuse_factor_4, with_dataflow],
    ids=["defaults", "hex_reuse_config", "reuse_factor_4", "dataflow"],
)  # fmt: skip
def test_hex_layers_and_an_hgq2_dense_head_export_together(tmp_path, tweak):
    """Convolution (rings shared, learned bit widths) -> max pooling (stride 2) -> convolution
    (stride 2) -> Flatten -> HGQ2's own QDense: what used to stop with ``No implementation of
    produce_kif for HHexConvLineBuffer``. Exact with the settings a user is likely to use."""
    simulated, reference = converted_equals_keras(tmp_path, chain_model(), tweak)
    np.testing.assert_array_equal(simulated, reference)


def test_a_convolution_without_an_output_quantizer_in_front_of_an_hgq2_layer(tmp_path):
    inp = keras.Input((H, W, 2), name="x")
    x = hgly.QConv2d(
        4, kernel_size=1, activation="relu", name="c1", kq_conf=weights(), iq_conf=act(2, 3, 0)
    )(inp)
    model = keras.Model(inp, head()(keras.layers.Flatten(name="flat")(x)))
    simulated, reference = converted_equals_keras(tmp_path, model)
    np.testing.assert_array_equal(simulated, reference)


def test_a_first_layer_without_an_input_quantizer_is_explained():
    inp = keras.Input((H, W, 2), name="x")
    x = hgly.QConv2d(4, kernel_size=1, name="c1", kq_conf=weights(), oq_conf=act(3, 3, 0))(inp)
    model = keras.Model(inp, head()(keras.layers.Flatten(name="flat")(x)))
    frozen, _ = hgly.freeze(model)
    register_hex_gather_layers("Vitis")
    with pytest.raises(ValueError, match="must be quantized"):
        patch_model_for_hls(frozen)


def single_quantizer_model():
    """One quantizer per tensor (the iq_conf of each consumer), a different grid for each."""
    inp = keras.Input((H, W, 2), name="x")
    x = hgly.QConv2d(
        6, kernel_size=2, share_neighbors=True, activation="relu", name="c1",
        kq_conf=weights(), bq_conf=weights(7, 2), iq_conf=act(2, 3, 0),
    )(inp)  # fmt: skip
    x = hgly.QMaxPool2d(kernel_size=1, strides=2, name="p1")(x)
    x = hgly.QConv2d(
        5, kernel_size=1, strides=2, activation="relu", name="c2",
        kq_conf=weights(), iq_conf=act(3, 1, 0),
    )(x)  # fmt: skip
    x = keras.layers.Flatten(name="flat")(x)
    x = QDense(8, activation="relu", name="fc1", kq_conf=weights(), iq_conf=act(2, 2, 0, (0, 1)))(x)
    return keras.Model(
        inp,
        QDense(
            3, name="logits", kq_conf=weights(), iq_conf=act(3, 3, 0, (0, 1)),
            oq_conf=act(5, 4, 1, (0, 1)),
        )(x),
    )  # fmt: skip


def test_one_quantizer_per_tensor_with_different_grids_is_exact(tmp_path):
    """The rule the warning below asks for: the grids may differ, nothing is rounded twice."""
    simulated, reference = converted_equals_keras(tmp_path, single_quantizer_model())
    np.testing.assert_array_equal(simulated, reference)


def test_a_tensor_quantized_twice_in_a_row_is_reported():
    inp = keras.Input((H, W, 2), name="x")
    x = hgly.QConv2d(
        4, kernel_size=1, activation="relu", name="c1",
        kq_conf=weights(), iq_conf=act(2, 3, 0), oq_conf=act(3, 3, 0),
    )(inp)  # fmt: skip
    twice = keras.Model(inp, head()(keras.layers.Flatten(name="flat")(x)))
    register_hex_gather_layers("Vitis")
    with pytest.warns(UserWarning, match="quantized twice in a row"):
        patch_model_for_hls(hgly.freeze(twice)[0])

    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        patch_model_for_hls(hgly.freeze(single_quantizer_model())[0])
    assert not [w for w in caught if "twice" in str(w.message)]


def test_a_relu_layer_without_a_quantizer_after_it_gives_zero_for_negative_values(tmp_path):
    """hls4ml 1.3 converts keras.layers.ReLU to a thresholded ReLU whose threshold gets an
    ap_ufixed<2,32> in the bit-exact flow; the unsigned wrapping result of a last ReLU then turned
    a negative value into a large positive one. The patched model uses Activation("relu")."""
    inp = keras.Input((H, W, 2), name="x")
    x = hgly.QConv2d(
        4, kernel_size=1, name="c1", kq_conf=weights(), bq_conf=weights(8, 3), iq_conf=act(2, 3, 1)
    )(inp)
    model = keras.Model(inp, keras.layers.ReLU(name="relu_out")(x))  # nothing quantizes its output
    reference = keras.ops.convert_to_numpy(
        model(np.random.default_rng(0).uniform(-1, 1, (8, H, W, 2)).astype("float32"))
    )
    assert (reference == 0).any()  # the convolution does give negative values here
    simulated, reference = converted_equals_keras(tmp_path, model, signed_input=True)
    np.testing.assert_array_equal(simulated, reference)


def global_pooling_model():
    inp = keras.Input((H, W, 2), name="x")
    x = hgly.QConv2d(
        6, kernel_size=1, strides=2, activation="relu", name="c1",
        kq_conf=weights(), bq_conf=weights(7, 2), iq_conf=act(2, 3, 0),
    )(inp)  # fmt: skip
    x = hgly.QMaxPool2d(kernel_size=1, strides=1, name="p1", iq_conf=act(3, 3, 0))(x)
    x = keras.layers.ReLU(name="r1")(x)
    x = QGlobalAveragePooling2D(name="mean", iq_conf=act(3, 3, 0))(x)
    return keras.Model(
        inp,
        QDense(
            3, name="logits", kq_conf=weights(), iq_conf=act(3, 3, 0, (0, 1)),
            oq_conf=act(5, 4, 1, (0, 1)),
        )(x),
    )  # fmt: skip


def test_a_global_average_pooling_after_hex_layers_is_exact(tmp_path):
    """hls4ml 1.3.0 hands the layers after a global 2D pooling a (1, C) array where a (C,) vector is
    expected (its quantizer then asserts equal ranks), and the hex nodes used to have a flattened
    (pixels, channels) output shape that no pooling can read."""
    simulated, reference = converted_equals_keras(tmp_path, global_pooling_model())
    np.testing.assert_array_equal(simulated, reference)


def test_the_exported_hex_nodes_have_an_image_shaped_output(tmp_path):
    register_hex_gather_layers("Vitis")
    patched = patch_model_for_hls(hgly.freeze(chain_model())[0])
    config = hls4ml.utils.config_from_keras_model(patched, granularity="name", backend="Vitis")
    hls_model = hls4ml.converters.convert_from_keras_model(
        patched, hls_config=config, io_type="io_stream", backend="Vitis",
        output_dir=str(tmp_path / "prj"), part="xc7vx485tffg1927-2", clock_period=5,
    )  # fmt: skip
    shapes = {
        name: tuple(node.get_output_variable().shape)
        for name, node in hls_model.graph.items()
        if name.endswith("_linebuffer")
    }
    # 12 x 11, a stride-2 pooling (6 x 6), a stride-2 convolution (3 x 3)
    assert shapes == {
        "c1_linebuffer": (12, 11, 6),
        "p1_linebuffer": (6, 6, 6),
        "c2_linebuffer": (3, 3, 5),
    }
