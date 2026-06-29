"""Serialization round-trips: a common failure mode for custom Keras layers
is that get_config/from_config silently drops a constructor argument, or a
saved-then-reloaded model produces different output than the original. Every
layer here is `@keras.saving.register_keras_serializable`, so these checks
matter for anyone shipping a trained model.
"""

import keras
import numpy as np
import pytest

import keras_hexagdly as hgly


def _roundtrip_config(layer):
    config = layer.get_config()
    cls = type(layer)
    return cls.from_config(config)


@pytest.mark.parametrize("share_neighbors", [False, True])
def test_conv2d_config_roundtrip(share_neighbors):
    layer = hgly.Conv2d(
        in_channels=2,
        out_channels=3,
        kernel_size=2,
        stride=2,
        bias=True,
        share_neighbors=share_neighbors,
    )
    clone = _roundtrip_config(layer)
    assert clone.in_channels == layer.in_channels
    assert clone.out_channels == layer.out_channels
    assert clone.kernel_size == layer.kernel_size
    assert clone.stride == layer.stride
    assert clone.share_neighbors == layer.share_neighbors


def test_conv3d_config_roundtrip_includes_depth_padding():
    """depth_padding is new in this port -- make sure it survives get_config,
    since it's easy to forget when adding a new constructor argument."""
    layer = hgly.Conv3d(
        in_channels=2, out_channels=3, kernel_size=(3, 1), stride=1, depth_padding="same"
    )
    clone = _roundtrip_config(layer)
    assert clone.depth_padding == "same"


def test_custom_kernel_2d_config_roundtrip():
    subk = [np.ones((1, 1, 3, 1), np.float32), np.ones((1, 1, 2, 2), np.float32)]
    layer = hgly.Conv2d_CustomKernel(sub_kernels=subk, stride=2, bias=np.array([0.5]))
    clone = _roundtrip_config(layer)
    assert clone.hexbase_stride == 2
    np.testing.assert_allclose(clone.kernel0.numpy(), layer.kernel0.numpy())


def test_full_model_save_load_roundtrip(tmp_path):
    """Save a tiny model using a keras_hexagdly layer to a .keras file and
    reload it: outputs must match exactly (weights + config both round-trip)."""
    x_in = keras.Input(shape=(9, 8, 2))
    y = hgly.Conv2d(out_channels=3, kernel_size=2, stride=1, share_neighbors=True)(x_in)
    y = hgly.MaxPool2d(kernel_size=1, stride=2)(y)
    model = keras.Model(x_in, y)

    x = np.random.randn(2, 9, 8, 2).astype(np.float32)
    out_before = keras.ops.convert_to_numpy(model(x))

    path = tmp_path / "model.keras"
    model.save(path)
    reloaded = keras.models.load_model(path)
    out_after = keras.ops.convert_to_numpy(reloaded(x))

    np.testing.assert_allclose(out_before, out_after, rtol=1e-6, atol=1e-6)


def test_full_model_save_load_roundtrip_depth_padding(tmp_path):
    x_in = keras.Input(shape=(5, 9, 8, 2))
    y = hgly.Conv3d(out_channels=3, kernel_size=(3, 1), stride=1, depth_padding="same")(x_in)
    model = keras.Model(x_in, y)

    x = np.random.randn(2, 5, 9, 8, 2).astype(np.float32)
    out_before = keras.ops.convert_to_numpy(model(x))

    path = tmp_path / "model3d.keras"
    model.save(path)
    reloaded = keras.models.load_model(path)
    out_after = keras.ops.convert_to_numpy(reloaded(x))

    assert out_after.shape[1] == 5  # depth preserved after reload too
    np.testing.assert_allclose(out_before, out_after, rtol=1e-6, atol=1e-6)
