"""Standalone MaxPool2d tests with hand-verified expected output (no PyTorch
needed); ground truth ported from HexagDLy's tests/test_MaxPool2d.py.
"""

import keras
import numpy as np
import pytest
from _hex_reference import maxpool2d_expected, maxpool2d_input_nhwc

import keras_hexagdly as hgly


@pytest.mark.parametrize("in_channels", [1, 5])
@pytest.mark.parametrize("kernel_size", [1, 2])
@pytest.mark.parametrize("stride", [1, 2, 3])
def test_maxpool2d_hand_verified(in_channels, kernel_size, stride):
    x = maxpool2d_input_nhwc(in_channels)
    expected = maxpool2d_expected(in_channels, kernel_size, stride)[None, ...]

    layer = hgly.MaxPool2d(kernel_size, stride)
    out = keras.ops.convert_to_numpy(layer(keras.ops.convert_to_tensor(x)))

    np.testing.assert_allclose(out, expected, rtol=5e-4, atol=1e-2)
