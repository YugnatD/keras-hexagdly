"""Standalone Conv2d tests with hand-verified expected output (no PyTorch
needed); ground truth ported from HexagDLy's tests/test_Conv2d.py.
"""

import keras
import numpy as np
import pytest
from _hex_reference import conv2d_expected, conv2d_input_nhwc

import keras_hexagdly as hgly


@pytest.mark.parametrize("in_channels", [1, 5])
@pytest.mark.parametrize("kernel_size", [1, 2])
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("bias", [False, True])
def test_conv2d_hand_verified(in_channels, kernel_size, stride, bias):
    bias_value = 1.0 if bias else 0.0
    x = conv2d_input_nhwc(in_channels)
    expected = conv2d_expected(in_channels, kernel_size, stride, bias_value)[None, ..., None]

    layer = hgly.Conv2d(in_channels, 1, kernel_size, stride, bias, debug=True)
    out = keras.ops.convert_to_numpy(layer(keras.ops.convert_to_tensor(x)))

    np.testing.assert_allclose(out, expected, rtol=5e-4, atol=1e-2)


def test_conv2d_in_channels_inferred():
    """The new out_channels-only call form infers in_channels from the input."""
    x = conv2d_input_nhwc(3)
    layer = hgly.Conv2d(out_channels=2, kernel_size=1, stride=1, debug=True)
    out = keras.ops.convert_to_numpy(layer(keras.ops.convert_to_tensor(x)))
    assert out.shape == (1, 5, 8, 2)
