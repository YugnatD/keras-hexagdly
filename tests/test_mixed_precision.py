"""Mixed precision (keras.mixed_precision / per-layer dtype policies).

Standard Keras mixed-precision pattern: trainable weights stay float32 (the
"variable dtype"), inputs/activations are cast to float16/bfloat16 (the
"compute dtype") for the actual op. These tests check keras_hexagdly's layers
follow that pattern correctly under both the per-layer policy (preferred --
doesn't leak state) and the deprecated-but-common global policy.
"""

import keras
import numpy as np
import pytest

import keras_hexagdly as hgly


@pytest.fixture
def global_mixed_float16():
    """Global policies are process-wide state; always restore it, even if
    the test fails, so this doesn't bleed into unrelated tests."""
    previous = keras.mixed_precision.global_policy()
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        yield
    finally:
        keras.mixed_precision.set_global_policy(previous)


@pytest.mark.parametrize("share_neighbors", [False, True])
def test_conv2d_per_layer_float16_policy(share_neighbors):
    """The per-layer `dtype=` argument is the recommended way to opt into
    mixed precision -- it shouldn't require touching global state."""
    x = np.random.randn(1, 9, 8, 2).astype(np.float32)
    layer = hgly.Conv2d(
        out_channels=3,
        kernel_size=2,
        stride=1,
        share_neighbors=share_neighbors,
        dtype="mixed_float16",
    )
    out = layer(keras.ops.convert_to_tensor(x))

    assert keras.backend.standardize_dtype(out.dtype) == "float16"
    assert keras.mixed_precision.global_policy().name == "float32"  # untouched
    weights = layer.ring_weights if share_neighbors else layer._base_kernels[0]
    assert keras.backend.standardize_dtype(weights.dtype) == "float32"  # variables stay float32
    assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))


def test_conv3d_depth_padding_same_per_layer_float16_policy():
    x = np.random.randn(1, 9, 9, 8, 2).astype(np.float32)
    layer = hgly.Conv3d(
        out_channels=2, kernel_size=(3, 1), depth_padding="same", dtype="mixed_float16"
    )
    out = layer(keras.ops.convert_to_tensor(x))

    assert keras.backend.standardize_dtype(out.dtype) == "float16"
    assert out.shape[1] == 9  # depth still preserved under fp16 compute
    assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))


def test_maxpool2d_per_layer_float16_policy():
    x = np.random.randn(1, 10, 9, 3).astype(np.float32)
    layer = hgly.MaxPool2d(kernel_size=2, stride=1, dtype="mixed_float16")
    out = layer(keras.ops.convert_to_tensor(x))
    assert keras.backend.standardize_dtype(out.dtype) == "float16"


def test_conv2d_global_mixed_float16_policy(global_mixed_float16):
    """The deprecated-but-still-common global policy must also work, and
    must not be required to leave variables in float16 (it shouldn't)."""
    x = np.random.randn(1, 9, 8, 2).astype(np.float32)
    layer = hgly.Conv2d(out_channels=3, kernel_size=2, stride=1)
    out = layer(keras.ops.convert_to_tensor(x))

    assert keras.backend.standardize_dtype(out.dtype) == "float16"
    assert keras.backend.standardize_dtype(layer._base_kernels[0].dtype) == "float32"
    assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))


def test_float16_output_reasonably_close_to_float32(global_mixed_float16):
    """fp16 compute shouldn't silently produce garbage: compare against the
    same weights run at float32, with a tolerance appropriate for fp16."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal((2, 9, 8, 2)).astype(np.float32)

    keras.mixed_precision.set_global_policy("float32")
    layer_f32 = hgly.Conv2d(out_channels=3, kernel_size=2, stride=1, bias=True)
    out_f32 = keras.ops.convert_to_numpy(layer_f32(keras.ops.convert_to_tensor(x)))

    keras.mixed_precision.set_global_policy("mixed_float16")
    layer_f16 = hgly.Conv2d(out_channels=3, kernel_size=2, stride=1, bias=True)
    _ = layer_f16(keras.ops.zeros((1, 9, 8, 2)))
    for i in range(layer_f16.hexbase_size + 1):
        layer_f16._base_kernels[i].assign(layer_f32._base_kernels[i].numpy())
    layer_f16.bias_tensor.assign(layer_f32.bias_tensor.numpy())
    out_f16 = keras.ops.convert_to_numpy(layer_f16(keras.ops.convert_to_tensor(x)))

    np.testing.assert_allclose(out_f32, out_f16, rtol=1e-2, atol=1e-2)
