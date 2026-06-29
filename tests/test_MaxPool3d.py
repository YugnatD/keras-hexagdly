"""Standalone MaxPool3d tests with hand-verified expected output (no PyTorch
needed); ground truth ported from HexagDLy's tests/test_MaxPool3d.py.
"""

import keras
import numpy as np
import pytest
from _hex_reference import conv3d_input_ndhwc, maxpool3d_expected

import keras_hexagdly as hgly

# (in_channels, depth, kernel_size_depth, kernel_size_hex, stride_depth, stride_hex)
CASES = [
    (1, 1, 1, 1, 1, 1),
    (1, 1, 1, 1, 1, 2),
    (1, 1, 1, 1, 1, 3),
    (1, 1, 1, 2, 1, 1),
    (1, 9, 1, 1, 1, 1),
    (1, 9, 1, 1, 2, 1),
    (1, 9, 2, 1, 1, 1),
    (1, 9, 2, 1, 2, 1),
    (5, 9, 2, 2, 2, 2),
]


@pytest.mark.parametrize("in_channels,depth,kd,kh,sd,sh", CASES)
def test_maxpool3d_hand_verified(in_channels, depth, kd, kh, sd, sh):
    x = conv3d_input_ndhwc(in_channels, depth)
    expected = maxpool3d_expected(in_channels, depth, kd, kh, sd, sh)[None, ...]

    layer = hgly.MaxPool3d((kd, kh), (sd, sh))
    out = keras.ops.convert_to_numpy(layer(keras.ops.convert_to_tensor(x)))

    np.testing.assert_allclose(out, expected, rtol=5e-4, atol=1e-2)
