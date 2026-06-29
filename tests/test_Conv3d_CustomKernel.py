"""Standalone Conv3d_CustomKernel tests with hand-verified expected output (no
PyTorch needed); ground truth ported from HexagDLy's
tests/test_Conv3d_CustomKernel.py. Sub-kernels are filled with ones, exercising
the custom sub-kernel API path with the same ground truth as test_Conv3d.py.
"""

import keras
import numpy as np
import pytest
from _hex_reference import conv3d_expected, conv3d_input_ndhwc

import keras_hexagdly as hgly


def ones_sub_kernels(in_channels, kernel_size_depth, kernel_size_hex):
    # PyTorch hexagdly layout: (out, in, depth, rows, cols).
    if kernel_size_hex == 1:
        return [
            np.ones((1, in_channels, kernel_size_depth, 3, 1), np.float32),
            np.ones((1, in_channels, kernel_size_depth, 2, 2), np.float32),
        ]
    return [
        np.ones((1, in_channels, kernel_size_depth, 5, 1), np.float32),
        np.ones((1, in_channels, kernel_size_depth, 4, 2), np.float32),
        np.ones((1, in_channels, kernel_size_depth, 3, 2), np.float32),
    ]


# (in_channels, depth, kernel_size_depth, kernel_size_hex, stride_depth, stride_hex, bias)
CASES = [
    (1, 1, 1, 1, 1, 1, False),
    (1, 1, 1, 1, 1, 2, False),
    (1, 9, 2, 1, 2, 1, False),
    (1, 9, 7, 2, 1, 1, False),
    (5, 9, 7, 2, 1, 1, True),
]


@pytest.mark.parametrize("in_channels,depth,kd,kh,sd,sh,bias", CASES)
def test_conv3d_custom_kernel_hand_verified(in_channels, depth, kd, kh, sd, sh, bias):
    bias_value = 1.0 if bias else 0.0
    bias_arg = np.array([1.0]) if bias else None
    x = conv3d_input_ndhwc(in_channels, depth)
    expected = conv3d_expected(in_channels, depth, kd, kh, sd, sh, bias_value)[None, ..., None]

    layer = hgly.Conv3d_CustomKernel(
        ones_sub_kernels(in_channels, kd, kh), stride=(sd, sh), bias=bias_arg
    )
    out = keras.ops.convert_to_numpy(layer(keras.ops.convert_to_tensor(x)))

    np.testing.assert_allclose(out, expected, rtol=5e-4, atol=1e-2)
