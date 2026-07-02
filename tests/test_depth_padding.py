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


# ---------------------------------------------------------------------------
# Graph-mode (model.predict / tf.function) regression tests.
#
# A pad op feeding the 3D hex column-split convolution used to trigger a
# graph-mode shape-inference mismatch: keras.ops.conv traced a column width one
# larger than it produced at run time, so the reorder gather indexed out of
# bounds under model.predict (e.g. "indices[9] = 10 is not in [0, 10)"), while
# eager execution worked.  The layer now launders the input (_stabilize_3d_shape)
# so graph and eager agree.  These tests run in graph mode on purpose -- the
# older tests above call the layer eagerly and so never exercised the bug.
# ---------------------------------------------------------------------------

D, H, W, CIN, COUT = 8, 13, 11, 2, 3


def _rand_init(layer):
    rng = np.random.default_rng(7)
    for w in layer.trainable_variables:
        w.assign(rng.standard_normal(w.shape).astype(np.float32))


@pytest.mark.parametrize("share", [False, True])
@pytest.mark.parametrize("kernel_size", [(2, 1), (3, 1), (2, 2), (3, 3)])
def test_depth_padding_same_graph_matches_eager(share, kernel_size):
    """depth_padding='same' must run under model.predict (graph mode) and match
    the eager result.  Regression for the pad-then-column-split reorder crash.

    Uses bias=False: a trailing bias-add happens to perturb the graph enough to
    hide the shape mismatch, so bias=False is the configuration that actually
    exercises the bug (and the one the original failing tests used)."""
    rng = np.random.default_rng(3)
    layer = hgly.Conv3d(
        CIN,
        COUT,
        kernel_size=kernel_size,
        bias=False,
        share_neighbors=share,
        depth_padding="same",
    )
    inp = keras.Input((D, H, W, CIN))
    model = keras.Model(inp, layer(inp))
    _rand_init(layer)

    x = rng.standard_normal((1, D, H, W, CIN)).astype(np.float32)
    y_eager = keras.ops.convert_to_numpy(layer(keras.ops.convert_to_tensor(x)))
    y_graph = model.predict(x, verbose=0)

    assert y_graph.shape == y_eager.shape
    np.testing.assert_allclose(y_graph, y_eager, atol=1e-5)


@pytest.mark.parametrize("kernel_size", [(2, 1), (3, 1), (2, 2), (1, 1)])
def test_external_zeropad_before_conv3d_graph(kernel_size):
    """A user ZeroPadding3D feeding a hex Conv3d must also run under graph mode
    (same underlying laundering).  Compares against eager pad-then-conv.
    bias=False, the configuration that actually triggers the bug."""
    rng = np.random.default_rng(5)
    layer = hgly.Conv3d(CIN, COUT, kernel_size=kernel_size, bias=False)
    inp = keras.Input((D, H, W, CIN))
    padded = keras.layers.ZeroPadding3D(((0, 1), (0, 0), (0, 0)))(inp)
    model = keras.Model(inp, layer(padded))
    _rand_init(layer)

    x = rng.standard_normal((1, D, H, W, CIN)).astype(np.float32)
    x_pad = np.pad(x, [(0, 0), (0, 1), (0, 0), (0, 0), (0, 0)])
    y_eager = keras.ops.convert_to_numpy(layer(keras.ops.convert_to_tensor(x_pad)))
    y_graph = model.predict(x, verbose=0)

    assert y_graph.shape == y_eager.shape
    np.testing.assert_allclose(y_graph, y_eager, atol=1e-5)


def test_external_zeropad_before_maxpool3d_graph():
    """ZeroPadding3D feeding a hex MaxPool3d must run under graph mode too."""
    rng = np.random.default_rng(6)
    layer = hgly.MaxPool3d(kernel_size=(2, 2))
    inp = keras.Input((D, H, W, CIN))
    padded = keras.layers.ZeroPadding3D(((0, 1), (0, 0), (0, 0)))(inp)
    model = keras.Model(inp, layer(padded))

    x = rng.standard_normal((1, D, H, W, CIN)).astype(np.float32)
    x_pad = np.pad(x, [(0, 0), (0, 1), (0, 0), (0, 0), (0, 0)])
    y_eager = keras.ops.convert_to_numpy(layer(keras.ops.convert_to_tensor(x_pad)))
    y_graph = model.predict(x, verbose=0)

    assert y_graph.shape == y_eager.shape
    np.testing.assert_allclose(y_graph, y_eager, atol=1e-5)
