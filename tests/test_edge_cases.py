"""Edge cases and failure modes worth guarding explicitly: input validation,
minimum viable sizes, dtype handling, and the in_channels-inference path
(new in this port). No PyTorch needed.
"""

import keras
import numpy as np
import pytest

import keras_hexagdly as hgly


def test_in_channels_mismatch_raises():
    """An explicit in_channels that disagrees with the actual input must fail
    loudly, not silently use the wrong value."""
    layer = hgly.Conv2d(in_channels=3, out_channels=2, kernel_size=1)
    with pytest.raises(ValueError):
        layer(keras.ops.zeros((1, 9, 8, 5)))  # 5 channels, declared 3


def test_dynamic_spatial_dims_raise():
    """The hex addressing arithmetic needs static H/W; a None spatial dim
    must fail with a clear error, not crash deep inside the conv."""
    layer = hgly.Conv2d(out_channels=2, kernel_size=1)
    with pytest.raises(ValueError):
        layer.build((None, None, None, 1))


def test_dynamic_batch_dim_is_fine():
    """Only the spatial/channel dims need to be static; batch can be None,
    as in a normal Keras functional Input."""
    x = keras.Input(shape=(9, 8, 2))
    layer = hgly.Conv2d(out_channels=3, kernel_size=1)
    y = layer(x)
    assert y.shape == (None, 9, 8, 3)


def test_out_channels_only_form_3d():
    """The new Conv3d(out_channels, ...) call form infers in_channels too."""
    x = np.zeros((1, 5, 9, 8, 4), dtype=np.float32)
    layer = hgly.Conv3d(out_channels=2, kernel_size=1)
    out = layer(keras.ops.convert_to_tensor(x))
    assert out.shape == (1, 5, 9, 8, 2)


@pytest.mark.parametrize("kernel_size", [1, 2, 3])
def test_minimum_viable_input_size_stride1(kernel_size):
    """stride=1 only needs H big enough to contain the kernel once."""
    H = 2 * kernel_size + 1
    W = 4
    x = np.random.randn(1, H, W, 1).astype(np.float32)
    layer = hgly.Conv2d(out_channels=1, kernel_size=kernel_size, stride=1)
    out = layer(keras.ops.convert_to_tensor(x))
    assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))


def test_too_few_rows_for_stride_raises():
    """operation_with_arbitrary_stride asserts on inputs too small for the
    requested stride -- verify it fails loudly rather than returning garbage."""
    layer = hgly.Conv2d(out_channels=1, kernel_size=1, stride=4)
    x = np.random.randn(1, 1, 4, 1).astype(np.float32)  # H=1, way too small
    with pytest.raises(AssertionError):
        layer(keras.ops.convert_to_tensor(x))


def test_float64_input_does_not_crash():
    x = np.random.randn(1, 9, 8, 2).astype(np.float64)
    layer = hgly.Conv2d(out_channels=3, kernel_size=2, stride=1)
    out = layer(keras.ops.convert_to_tensor(x))
    assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))


@pytest.mark.parametrize("kernel_size", [1, 2, 3])
@pytest.mark.parametrize("stride", [1, 3, 5])
def test_no_nan_or_inf_random_inputs(kernel_size, stride):
    rng = np.random.default_rng(99)
    x = (rng.standard_normal((2, 17, 14, 3)) * 1000).astype(np.float32)
    layer = hgly.Conv2d(out_channels=2, kernel_size=kernel_size, stride=stride, bias=True)
    out = keras.ops.convert_to_numpy(layer(keras.ops.convert_to_tensor(x)))
    assert np.all(np.isfinite(out))


def test_conv2d_custom_kernel_bad_subkernel_shape_raises():
    """The first sub-kernel must have exactly 1 column; a malformed one must
    be rejected at construction, not produce a silently wrong convolution."""
    bad = [np.ones((1, 1, 3, 2), np.float32)]  # should have 1 column, not 2
    with pytest.raises(AssertionError):
        hgly.Conv2d_CustomKernel(sub_kernels=bad, stride=1)


def test_debug_weights_are_exactly_one():
    layer = hgly.Conv2d(out_channels=1, kernel_size=2, stride=1, bias=False, debug=True)
    _ = layer(keras.ops.zeros((1, 9, 8, 1)))
    for i in range(layer.hexbase_size + 1):
        np.testing.assert_array_equal(layer._base_kernels[i].numpy(), 1.0)


# ----------------------------------------------------------------------------
# kernel_size/stride validation: upstream PyTorch HexagDLy accepts these and
# fails with an obscure, backend/location-dependent error deep inside the
# first call (ZeroDivisionError, a stray AttributeError, a backend-specific
# "stride must be > 0"...). This port validates at construction time instead.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("bad_kernel_size", [0, -1, 1.5, "1", None])
def test_conv2d_bad_kernel_size_raises_at_construction(bad_kernel_size):
    with pytest.raises(ValueError):
        hgly.Conv2d(out_channels=1, kernel_size=bad_kernel_size)


@pytest.mark.parametrize("bad_stride", [0, -1, 1.5, "1", None])
def test_conv2d_bad_stride_raises_at_construction(bad_stride):
    with pytest.raises(ValueError):
        hgly.Conv2d(out_channels=1, kernel_size=1, stride=bad_stride)


@pytest.mark.parametrize("bad_kernel_size", [0, -1, 1.5])
def test_maxpool2d_bad_kernel_size_raises_at_construction(bad_kernel_size):
    with pytest.raises(ValueError):
        hgly.MaxPool2d(kernel_size=bad_kernel_size)


@pytest.mark.parametrize("bad_stride", [0, -1, 1.5])
def test_conv2d_custom_kernel_bad_stride_raises_at_construction(bad_stride):
    with pytest.raises(ValueError):
        hgly.Conv2d_CustomKernel(stride=bad_stride)


@pytest.mark.parametrize("bad_kernel_size", [(1, 0), (0, 1), (1, 2, 3), "x", -1])
def test_conv3d_bad_kernel_size_raises_at_construction(bad_kernel_size):
    with pytest.raises(ValueError):
        hgly.Conv3d(out_channels=1, kernel_size=bad_kernel_size)


@pytest.mark.parametrize("bad_stride", [(1, 0), (0, 1), (1, 2, 3), "x", -1])
def test_conv3d_bad_stride_raises_at_construction(bad_stride):
    with pytest.raises(ValueError):
        hgly.Conv3d(out_channels=1, kernel_size=1, stride=bad_stride)


def test_maxpool3d_bad_kernel_size_raises_at_construction():
    with pytest.raises(ValueError):
        hgly.MaxPool3d(kernel_size=(1, 0))


def test_conv2d_valid_kernel_size_and_stride_still_work():
    """The validation itself must not reject legitimate values."""
    layer = hgly.Conv2d(out_channels=2, kernel_size=3, stride=2)
    out = layer(keras.ops.zeros((1, 15, 14, 1)))
    assert out is not None


def test_conv3d_valid_tuple_kernel_size_and_stride_still_work():
    layer = hgly.Conv3d(out_channels=2, kernel_size=(3, 1), stride=(2, 1))
    out = layer(keras.ops.zeros((1, 5, 9, 8, 1)))
    assert out is not None
