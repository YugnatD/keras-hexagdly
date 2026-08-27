"""Tests for the Keras-native constructor API: filters/kernel_size/strides/
use_bias, positional exactly like ``keras.layers.Conv2D``, with the original
PyTorch-hexagdly-style names (in_channels/out_channels/stride/bias) removed
entirely. See CHANGELOG.md [0.4.0] -- this is a breaking change from 0.3.1,
the last release where both forms worked side by side.
"""

import keras
import numpy as np
import pytest

import keras_hexagdly as hgly


@pytest.fixture
def x():
    return np.random.default_rng(0).uniform(size=(2, 9, 9, 3)).astype("float32")


def test_positional_call_matches_real_keras_conv2d_signature(x):
    """The whole point of removing the old names: `Conv2d(filters, kernel_size,
    strides=...)` now means exactly what it would for `keras.layers.Conv2D` --
    no silent misinterpretation as (in_channels, out_channels)."""
    x_t = keras.ops.convert_to_tensor(x)

    keras.utils.set_random_seed(0)
    positional = hgly.Conv2d(4, 1, strides=1, use_bias=True)
    positional.build(x.shape)
    assert positional.filters == 4
    assert positional.kernel_size == 1
    assert positional.strides == 1

    keras.utils.set_random_seed(0)
    keyword = hgly.Conv2d(filters=4, kernel_size=1, strides=1, use_bias=True)
    keyword.build(x.shape)

    y_pos = keras.ops.convert_to_numpy(positional(x_t))
    y_kw = keras.ops.convert_to_numpy(keyword(x_t))
    assert np.allclose(y_pos, y_kw)


def test_old_names_are_rejected_not_misinterpreted():
    """Old PyTorch-style kwargs no longer exist at all -- passing them raises,
    they are never silently swallowed or misread as something else."""
    with pytest.raises((TypeError, ValueError)):
        hgly.Conv2d(in_channels=3, out_channels=4, kernel_size=1)
    with pytest.raises((TypeError, ValueError)):
        hgly.Conv2d(filters=4, kernel_size=1, stride=2)
    with pytest.raises((TypeError, ValueError)):
        hgly.Conv2d(filters=4, kernel_size=1, bias=False)


def test_in_channels_is_not_a_constructor_argument(x):
    """Unlike the 0.3.1 transitional release, there is no way to pass
    in_channels at all -- it's always inferred in build(), like every
    standard Keras layer."""
    layer = hgly.Conv2d(4, kernel_size=1)
    assert layer.in_channels is None  # not yet built
    layer.build(x.shape)
    assert layer.in_channels == 3  # inferred from x's channel axis


def test_get_config_has_no_legacy_keys():
    layer = hgly.Conv2d(4, kernel_size=1, strides=2, use_bias=False)
    config = layer.get_config()
    assert config["filters"] == 4
    assert config["strides"] == 2
    assert config["use_bias"] is False
    for legacy_key in ("in_channels", "out_channels", "stride", "bias"):
        assert legacy_key not in config


def test_get_config_from_config_roundtrip(x):
    x_t = keras.ops.convert_to_tensor(x)
    keras.utils.set_random_seed(0)
    layer = hgly.Conv2d(4, kernel_size=1, strides=1, use_bias=True)
    layer.build(x.shape)
    y1 = keras.ops.convert_to_numpy(layer(x_t))

    rebuilt = hgly.Conv2d.from_config(layer.get_config())
    rebuilt.build(x.shape)
    rebuilt.set_weights(layer.get_weights())
    y2 = keras.ops.convert_to_numpy(rebuilt(x_t))
    assert np.allclose(y1, y2)


def test_real_save_load_roundtrip(x, tmp_path):
    model = keras.Sequential([keras.layers.Input((9, 9, 3)), hgly.Conv2d(4, kernel_size=1)])
    model.build((None, 9, 9, 3))
    path = tmp_path / "model.keras"
    model.save(path)
    reloaded = keras.saving.load_model(path)
    assert np.allclose(model.predict(x, verbose=0), reloaded.predict(x, verbose=0))


def test_full_model_built_the_way_a_keras_user_actually_would(x):
    """No hex-specific naming anywhere in this function -- filters/kernel_size/
    strides/use_bias throughout, a real Sequential model, real .fit(), real
    .predict(). This is what "full Keras" is meant to feel like end to end."""
    model = keras.Sequential(
        [
            keras.layers.Input((9, 9, 3)),
            hgly.Conv2d(8, kernel_size=1, strides=1, use_bias=True),
            keras.layers.ReLU(),
            hgly.MaxPool2d(kernel_size=1, strides=2),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")

    y = np.random.default_rng(1).uniform(size=(2, 1)).astype("float32")
    history = model.fit(x, y, epochs=1, verbose=0)
    assert np.isfinite(history.history["loss"][0])

    preds = model.predict(x, verbose=0)
    assert preds.shape == (2, 1)
    assert np.all(np.isfinite(preds))


@pytest.mark.parametrize(
    "layer_cls, kwargs",
    [
        (hgly.layers.MaxPool2d, {"kernel_size": 1, "strides": 2}),
        (hgly.layers.MaxPool3d, {"kernel_size": 1, "strides": 1}),
    ],
)
def test_maxpool_keras_native_kwargs(layer_cls, kwargs):
    layer = layer_cls(**kwargs)
    assert layer.strides == kwargs["strides"]
    with pytest.raises((TypeError, ValueError)):
        layer_cls(kernel_size=1, stride=2)  # old name gone


def test_conv3d_positional_and_keyword():
    layer = hgly.Conv3d(2, kernel_size=(1, 1), strides=(1, 1))
    layer.build((2, 4, 9, 9, 3))
    assert layer.filters == 2
    assert layer.strides == (1, 1)


def test_customkernel_strides_rename_bias_semantics_unchanged():
    """Conv2d_CustomKernel/Conv3d_CustomKernel's 'bias' is the bias *array*
    (data), not a boolean flag like Conv2d/Conv3d's -- it keeps its name
    unchanged; only 'stride' -> 'strides' applies here."""
    sub_kernels = [np.ones((2, 3, 3, 1)), np.ones((2, 3, 2, 2))]
    layer = hgly.layers.Conv2d_CustomKernel(sub_kernels=sub_kernels, strides=2)
    assert layer.strides == 2
    assert layer.hexbase_stride == 2
    with pytest.raises((TypeError, ValueError)):
        hgly.layers.Conv2d_CustomKernel(sub_kernels=sub_kernels, stride=2)
