"""Tests for `depth_padding="same"` on Conv3d, a functionality NEW in this
port with no equivalent in upstream PyTorch HexagDLy (which is "valid"-only
on the depth axis) -- so there is no oracle to check against here. Instead
these tests verify the defining property directly: output depth == input
depth, and the result equals manually zero-padding the depth axis then
running the (default) "valid" Conv3d.
"""

import keras
import numpy as np
import pytest

import keras_hexagdly as hgly


@pytest.mark.parametrize("share", [False, True])
@pytest.mark.parametrize("D,kd,n", [(9, 5, 1), (9, 3, 2), (7, 5, 1)])
def test_depth_padding_same_preserves_depth(share, D, kd, n):
    rng = np.random.default_rng(40)
    Cin, Cout, H, W = 2, 3, 9, 8
    x = rng.standard_normal((2, D, H, W, Cin)).astype(np.float32)

    layer = hgly.Conv3d(
        Cin,
        Cout,
        kernel_size=(kd, n),
        stride=1,
        bias=True,
        share_neighbors=share,
        depth_padding="same",
    )
    out = layer(keras.ops.convert_to_tensor(x))
    assert out.shape[1] == D


@pytest.mark.parametrize("D,kd,n", [(9, 5, 1), (7, 3, 2)])
def test_depth_padding_same_equals_manual_pad_then_valid(D, kd, n):
    """depth_padding="same" must equal: zero-pad depth by (kd-1)//2 each side
    (centred kernel), then run the default "valid" Conv3d."""
    rng = np.random.default_rng(41)
    Cin, Cout, H, W = 2, 3, 9, 8
    x = rng.standard_normal((1, D, H, W, Cin)).astype(np.float32)

    same_layer = hgly.Conv3d(
        Cin, Cout, kernel_size=(kd, n), stride=1, bias=False, depth_padding="same"
    )
    out_same = keras.ops.convert_to_numpy(same_layer(keras.ops.convert_to_tensor(x)))

    valid_layer = hgly.Conv3d(Cin, Cout, kernel_size=(kd, n), stride=1, bias=False)
    _ = valid_layer(keras.ops.zeros((1, D + kd - 1, H, W, Cin)))
    for i in range(valid_layer.hexbase_size + 1):
        valid_layer._base_kernels[i].assign(same_layer._base_kernels[i].numpy())

    pad = (kd - 1) // 2
    top, bot = pad, kd - 1 - pad
    x_padded = np.pad(x, [(0, 0), (top, bot), (0, 0), (0, 0), (0, 0)])
    out_valid = keras.ops.convert_to_numpy(valid_layer(keras.ops.convert_to_tensor(x_padded)))

    np.testing.assert_allclose(out_same, out_valid, atol=1e-5)


def test_depth_padding_invalid_value_raises():
    with pytest.raises(ValueError):
        hgly.Conv3d(1, 1, kernel_size=(3, 1), depth_padding="bogus")
