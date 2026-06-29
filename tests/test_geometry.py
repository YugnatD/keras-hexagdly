"""Independent geometry check (no PyTorch oracle, no hand-built arrays).

A debug kernel (all weights = 1, no bias) computes, at each output pixel, the
SUM of its hex neighbourhood. Feed a single impulse: the output then equals 1
exactly at the impulse and at each of its hex neighbours, and the COUNT of
ones must be the hex neighbourhood size 1 + 3n(n+1). This validates that the
conv really sums the right hexagonal support, independent of any reference
implementation.
"""

import keras
import numpy as np
import pytest

import keras_hexagdly as hgly


@pytest.mark.parametrize("n", [1, 2, 3])
def test_impulse_response_hex_neighbourhood_size(n):
    H, W = 15, 15
    x = np.zeros((1, H, W, 1), dtype=np.float32)
    x[0, H // 2, W // 2, 0] = 1.0

    layer = hgly.Conv2d(1, 1, kernel_size=n, stride=1, bias=False, debug=True)
    _ = layer(keras.ops.zeros((1, H, W, 1)))
    out = keras.ops.convert_to_numpy(layer(keras.ops.convert_to_tensor(x)))[0, :, :, 0]

    n_hits = int(np.sum(np.isclose(out, 1.0)))
    expected = 1 + 3 * n * (n + 1)  # hex cells within radius n
    clean = np.all(np.isclose(out, 0.0) | np.isclose(out, 1.0))

    assert clean
    assert n_hits == expected


def test_determinism():
    """Same input twice -> identical output (no hidden state / nondeterminism)."""
    rng = np.random.default_rng(23)
    x = keras.ops.convert_to_tensor(rng.standard_normal((2, 12, 9, 2)).astype(np.float32))
    layer = hgly.Conv2d(2, 3, kernel_size=3, stride=2, bias=True)
    _ = layer(keras.ops.zeros((1, 12, 9, 2)))
    a, b = keras.ops.convert_to_numpy(layer(x)), keras.ops.convert_to_numpy(layer(x))
    np.testing.assert_array_equal(a, b)
