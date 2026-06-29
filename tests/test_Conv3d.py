"""Standalone Conv3d tests with hand-verified expected output (no PyTorch
needed); ground truth ported from HexagDLy's tests/test_Conv3d.py.
"""

import keras
import numpy as np
import pytest
from _hex_reference import conv3d_expected, conv3d_input_ndhwc

import keras_hexagdly as hgly

# (in_channels, depth, kernel_size_depth, kernel_size_hex, stride_depth, stride_hex, bias)
CASES = [
    (1, 1, 1, 1, 1, 1, False),
    (1, 1, 1, 1, 1, 2, False),
    (1, 1, 1, 1, 1, 3, False),
    (1, 1, 1, 2, 1, 1, False),
    (1, 1, 1, 2, 1, 2, False),
    (1, 1, 1, 2, 1, 3, False),
    (1, 9, 1, 1, 1, 1, False),
    (1, 9, 1, 1, 2, 1, False),
    (1, 9, 1, 1, 3, 1, False),
    (1, 9, 2, 1, 1, 1, False),
    (1, 9, 2, 1, 2, 1, False),
    (1, 9, 2, 1, 2, 2, False),
    (1, 9, 7, 2, 1, 1, False),
    (5, 9, 7, 2, 1, 1, False),
    (5, 9, 7, 2, 1, 1, True),
]


@pytest.mark.parametrize("in_channels,depth,kd,kh,sd,sh,bias", CASES)
def test_conv3d_hand_verified(in_channels, depth, kd, kh, sd, sh, bias):
    bias_value = 1.0 if bias else 0.0
    x = conv3d_input_ndhwc(in_channels, depth)
    expected = conv3d_expected(in_channels, depth, kd, kh, sd, sh, bias_value)[None, ..., None]

    layer = hgly.Conv3d(in_channels, 1, (kd, kh), (sd, sh), bias, debug=True)
    out = keras.ops.convert_to_numpy(layer(keras.ops.convert_to_tensor(x)))

    np.testing.assert_allclose(out, expected, rtol=5e-4, atol=1e-2)
