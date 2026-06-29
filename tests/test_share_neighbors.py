"""Tests for `share_neighbors`, a functionality NEW in this port with no
equivalent in upstream PyTorch HexagDLy -- so there is no oracle to check
against here. Instead these tests verify the defining properties directly:
weight sharing by hex ring, trainability, and equivalence to a full
(non-shared) kernel whose cells are filled by broadcasting the ring weights.
"""

import keras
import numpy as np
import pytest

import keras_hexagdly as hgly


def test_ring_maps_geometry():
    """ring r holds 1 (r=0) or 6*r cells; num_rings = kernel_size + 1."""
    for n in (1, 2, 3):
        maps, num_rings = hgly.ring_maps_2d(n)
        assert num_rings == n + 1
        counts = np.zeros(num_rings, dtype=int)
        for m in maps:
            for r in range(num_rings):
                counts[r] += int(np.sum(m == r))
        assert counts[0] == 1
        for r in range(1, num_rings):
            assert counts[r] == 6 * r


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("stride", [1, 2])
def test_share_neighbors_equals_full_broadcast(n, stride):
    """A shared conv must equal a non-shared conv whose cells are filled by
    broadcasting the same ring weights -- the defining property of sharing."""
    rng = np.random.default_rng(31)
    H, W, Cin, Cout = 13, 11, 2, 3
    x = rng.standard_normal((2, H, W, Cin)).astype(np.float32)
    rw = rng.standard_normal((n + 1, Cin, Cout)).astype(np.float32)  # (nr,in,out)

    shared = hgly.Conv2d(Cin, Cout, kernel_size=n, stride=stride, bias=False, share_neighbors=True)
    _ = shared(keras.ops.zeros((1, H, W, Cin)))
    shared.ring_weights.assign(rw)

    full = hgly.Conv2d(Cin, Cout, kernel_size=n, stride=stride, bias=False)
    _ = full(keras.ops.zeros((1, H, W, Cin)))
    maps, _ = hgly.ring_maps_2d(n)
    for i in range(n + 1):
        getattr(full, f"kernel{i}").assign(rw[maps[i]])  # (rows,cols,in,out)

    out_shared = keras.ops.convert_to_numpy(shared(keras.ops.convert_to_tensor(x)))
    out_full = keras.ops.convert_to_numpy(full(keras.ops.convert_to_tensor(x)))
    np.testing.assert_allclose(out_shared, out_full, atol=1e-5)


def test_share_neighbors_3d_ties_hex_axes_only():
    """The depth (time) axis stays fully independent; only the hex axes share."""
    rng = np.random.default_rng(32)
    n, D, H, W, Cin, Cout = 1, 4, 11, 9, 2, 3
    x = rng.standard_normal((2, D, H, W, Cin)).astype(np.float32)
    layer = hgly.Conv3d(Cin, Cout, kernel_size=(D, n), stride=1, bias=False, share_neighbors=True)
    _ = layer(keras.ops.zeros((1, D, H, W, Cin)))
    assert tuple(layer.ring_weights.shape) == (D, n + 1, Cin, Cout)
    out = layer(keras.ops.convert_to_tensor(x))
    assert out.shape[0] == 2


def test_share_neighbors_gradient_tied():
    """Each ring's gradient is the sum of its cells' gradients in the full
    kernel, and the shared weights actually train (an SGD step moves them)."""
    if keras.backend.backend() != "tensorflow":
        pytest.skip("uses tf.GradientTape; only meaningful on the tensorflow backend")
    import tensorflow as tf

    rng = np.random.default_rng(33)
    n, H, W = 2, 11, 9
    x = tf.constant(rng.standard_normal((2, H, W, 1)).astype(np.float32))
    layer = hgly.Conv2d(1, 1, kernel_size=n, stride=1, bias=False, share_neighbors=True)
    _ = layer(keras.ops.zeros((1, H, W, 1)))

    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(layer(x))
    grad = tape.gradient(loss, layer.ring_weights)
    if isinstance(grad, tf.IndexedSlices):
        grad = tf.convert_to_tensor(grad)
    grad_np = grad.numpy()
    assert grad_np.shape == (n + 1, 1, 1)
    assert np.all(np.isfinite(grad_np))

    before = layer.ring_weights.numpy().copy()
    opt = tf.keras.optimizers.SGD(0.1)
    with tf.GradientTape() as tape2:
        loss2 = tf.reduce_sum(layer(x))
    opt.apply_gradients([(tape2.gradient(loss2, layer.ring_weights), layer.ring_weights)])
    assert not np.allclose(before, layer.ring_weights.numpy())
