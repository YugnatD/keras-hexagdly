"""Standalone Conv2d_CustomKernel tests with hand-verified expected output (no
PyTorch needed); ground truth ported from HexagDLy's
tests/test_Conv2d_CustomKernel.py. Sub-kernels are filled with ones, so the
expected output is the same "sum of hex neighbours" ground truth as
test_Conv2d.py -- this exercises the custom sub-kernel API path instead of
the debug-weights path.
"""

import keras
import numpy as np
import pytest
from _hex_reference import conv2d_expected, conv2d_input_nhwc

import keras_hexagdly as hgly


def ones_sub_kernels(in_channels, kernel_size):
    # PyTorch hexagdly layout: (out, in, rows, cols).
    if kernel_size == 1:
        return [
            np.ones((1, in_channels, 3, 1), np.float32),
            np.ones((1, in_channels, 2, 2), np.float32),
        ]
    return [
        np.ones((1, in_channels, 5, 1), np.float32),
        np.ones((1, in_channels, 4, 2), np.float32),
        np.ones((1, in_channels, 3, 2), np.float32),
    ]


@pytest.mark.parametrize("in_channels", [1, 5])
@pytest.mark.parametrize("kernel_size", [1, 2])
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("bias", [False, True])
def test_conv2d_custom_kernel_hand_verified(in_channels, kernel_size, stride, bias):
    bias_value = 1.0 if bias else 0.0
    bias_arg = np.array([1.0]) if bias else None
    x = conv2d_input_nhwc(in_channels)
    expected = conv2d_expected(in_channels, kernel_size, stride, bias_value)[None, ..., None]

    layer = hgly.Conv2d_CustomKernel(
        ones_sub_kernels(in_channels, kernel_size), stride=stride, bias=bias_arg
    )
    out = keras.ops.convert_to_numpy(layer(keras.ops.convert_to_tensor(x)))

    np.testing.assert_allclose(out, expected, rtol=5e-4, atol=1e-2)
