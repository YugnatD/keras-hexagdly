"""Tests for the Keras-native constructor kwarg rename (filters/strides/use_bias)
and its backward compatibility with the original PyTorch-hexagdly-style names
(in_channels/out_channels/stride/bias) -- both as still-working direct calls and,
critically, as configs saved by previous versions of this package (whose
get_config() emitted the old key names). See CHANGELOG.md [Unreleased].
"""

import warnings

import keras
import numpy as np
import pytest

import keras_hexagdly as hgly


@pytest.fixture
def x():
    return np.random.default_rng(0).uniform(size=(2, 9, 9, 3)).astype("float32")


def test_new_and_old_style_construction_are_numerically_identical(x):
    keras.utils.set_random_seed(0)
    new = hgly.Conv2d(filters=4, kernel_size=1, strides=1, use_bias=True)
    new.build(x.shape)

    keras.utils.set_random_seed(0)
    old = hgly.Conv2d(3, 4, kernel_size=1, stride=1, bias=True)
    old.build(x.shape)

    assert np.allclose(new(x), old(x))


def test_old_style_kwargs_emit_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="stride"):
        hgly.Conv2d(filters=4, kernel_size=1, stride=2)
    with pytest.warns(DeprecationWarning, match="bias"):
        hgly.Conv2d(filters=4, kernel_size=1, bias=False)


def test_in_channels_out_channels_do_not_warn():
    """Unlike stride/bias, in_channels/out_channels are not deprecated --
    filters is just the additional Keras-idiomatic alias for out_channels."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here fails the test
        hgly.Conv2d(3, 4, kernel_size=1)


def test_get_config_emits_new_names_only():
    layer = hgly.Conv2d(filters=4, kernel_size=1, strides=2, use_bias=False)
    config = layer.get_config()
    assert config["filters"] == 4
    assert config["strides"] == 2
    assert config["use_bias"] is False
    assert "out_channels" not in config
    assert "stride" not in config
    assert "bias" not in config


def test_legacy_saved_config_still_loads():
    """The exact shape get_config() used to emit before this change -- this is
    what a .keras file saved by an older version of this package has baked
    into it. Must keep loading correctly, indefinitely."""
    legacy_config = {
        "name": "legacy_conv",
        "trainable": True,
        "dtype": "float32",
        "in_channels": 3,
        "out_channels": 4,
        "kernel_size": 1,
        "stride": 2,
        "bias": False,
        "debug": False,
        "share_neighbors": False,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        layer = hgly.Conv2d.from_config(legacy_config)
        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]

    assert layer.filters == 4
    assert layer.strides == 2
    assert layer.use_bias is False
    assert len(dep_warnings) == 2  # stride, bias


def test_real_save_load_roundtrip(x, tmp_path):
    model = keras.Sequential([keras.layers.Input((9, 9, 3)), hgly.Conv2d(filters=4, kernel_size=1)])
    model.build((None, 9, 9, 3))
    path = tmp_path / "model.keras"
    model.save(path)
    reloaded = keras.saving.load_model(path)
    assert np.allclose(model.predict(x, verbose=0), reloaded.predict(x, verbose=0))


def test_conflicting_channel_kwargs_raise():
    with pytest.raises(ValueError, match="filters.*out_channels"):
        hgly.Conv2d(filters=4, out_channels=5, kernel_size=1)


def test_conflicting_stride_kwargs_raise():
    with pytest.raises(ValueError, match="strides.*stride"):
        hgly.Conv2d(filters=4, kernel_size=1, strides=2, stride=3)


@pytest.mark.parametrize(
    "layer_cls, kwargs",
    [
        (hgly.layers.MaxPool2d, {"kernel_size": 1}),
        (hgly.layers.MaxPool3d, {"kernel_size": 1}),
    ],
)
def test_maxpool_strides_alias(layer_cls, kwargs):
    layer = layer_cls(strides=2, **kwargs)
    assert layer.strides == 2
    assert layer.hexbase_stride == 2


def test_conv3d_filters_and_strides_alias():
    layer = hgly.Conv3d(filters=2, kernel_size=(1, 1), strides=(1, 1))
    layer.build((2, 4, 9, 9, 3))
    assert layer.filters == 2
    assert layer.strides == (1, 1)


def test_customkernel_strides_alias_does_not_touch_bias_semantics():
    """Conv2d_CustomKernel/Conv3d_CustomKernel's 'bias' is the bias *array*
    (data), not a boolean flag like Conv2d/Conv3d's -- it must NOT be renamed
    to use_bias, and must not warn."""
    sub_kernels = [np.ones((2, 3, 3, 1)), np.ones((2, 3, 2, 2))]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        layer = hgly.layers.Conv2d_CustomKernel(sub_kernels=sub_kernels, strides=2)
    assert layer.strides == 2
    assert layer.hexbase_stride == 2
    assert not hasattr(layer, "use_bias") or isinstance(layer.use_bias, bool)
