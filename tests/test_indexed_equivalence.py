"""Equivalence tests: indexed forward == layer.call() on the zig-zag grid.

These tests are the load-bearing correctness proof for the hls4ml export plan.
If they pass, the neighbor-table builder is exact and the export path will
produce numerically identical results to PC inference.

The flat-index convention throughout is m = h * W + w (raster scan):
  - Input  x_flat has shape (H*W,  Cin).
  - Output y_flat has shape (H_out*W_out, Cout).
  - neighbor_idx[n, k] is the flat input index that feeds output pixel n via
    kernel slot k, or -1 for border/zero-padded slots.

Coverage
--------
  Conv2d    x  {share=False, share=True} x {kernel=1,2,3} x {stride=1,2}
              x  border pixels (every output pixel, not just center)

  MaxPool2d x  {kernel=1,2,3} x {stride=1,2}  x  border pixels (all-neg input)

  Conv3d    x  {share=False, share=True} x {kernel=(1,1),(2,2),(1,2)} x {stride=1,2}
              x  border spatial pixels

  MaxPool3d x  {kernel=(1,1),(2,2),(1,2)} x {stride=1,2}  x  all-neg input
"""

import numpy as np
import pytest
import keras

import keras_hexagdly as hgly
from keras_hexagdly.indexed import (
    build_neighbor_table,
    get_cell_weights,
    indexed_conv2d_forward,
    indexed_maxpool2d_forward,
    build_neighbor_table_3d,
    get_cell_weights_3d,
    indexed_conv3d_forward,
    indexed_maxpool3d_forward,
)

# ---- helpers ----------------------------------------------------------------

H, W = 13, 11   # moderately sized grid; odd W exercises parity path

RNG = np.random.default_rng(42)


def _make_conv2d(kernel_size, stride, share_neighbors, Cin=2, Cout=3):
    layer = hgly.Conv2d(
        Cin, Cout,
        kernel_size=kernel_size,
        stride=stride,
        bias=False,
        share_neighbors=share_neighbors,
    )
    layer(keras.ops.zeros((1, H, W, Cin)))  # build
    # Set weights to random values so we aren't testing on a trivial kernel.
    for w in layer.trainable_variables:
        w.assign(RNG.standard_normal(w.shape).astype(np.float32))
    return layer


def _make_maxpool2d(kernel_size, stride):
    layer = hgly.MaxPool2d(kernel_size=kernel_size, stride=stride)
    layer(keras.ops.zeros((1, H, W, 1)))  # build
    return layer


def _grid_to_flat(grid_out):
    """(1, H_out, W_out, C) -> (H_out*W_out, C)."""
    return grid_out[0].reshape(-1, grid_out.shape[-1])


# ---- Conv2d equivalence tests -----------------------------------------------

@pytest.mark.parametrize("kernel_size", [1, 2, 3])
@pytest.mark.parametrize("stride",      [1, 2])
@pytest.mark.parametrize("share",       [False, True])
def test_conv2d_indexed_equals_call(kernel_size, stride, share):
    """indexed_conv2d_forward == layer.call() for every output pixel."""
    layer = _make_conv2d(kernel_size, stride, share)
    Cin = layer.in_channels

    x_flat = RNG.standard_normal((H * W, Cin)).astype(np.float32)
    x_grid = x_flat.reshape(1, H, W, Cin)

    # reference: run through the existing zig-zag call()
    y_grid = keras.ops.convert_to_numpy(layer(keras.ops.convert_to_tensor(x_grid)))
    y_ref  = _grid_to_flat(y_grid)                        # (N_out, Cout)

    # indexed path
    nbr, cells, (H_out, W_out) = build_neighbor_table(layer, H, W)
    W_k = get_cell_weights(layer, cells)                   # (K, Cin, Cout)
    y_idx = indexed_conv2d_forward(x_flat, nbr, W_k)       # (N_out, Cout)

    assert y_idx.shape == y_ref.shape, (
        f"shape mismatch: indexed={y_idx.shape}, call={y_ref.shape}"
    )
    max_err = float(np.max(np.abs(y_idx - y_ref)))
    assert max_err < 1e-4, (
        f"Conv2d(kernel={kernel_size}, stride={stride}, share={share}): "
        f"max abs err={max_err:.2e} (expected < 1e-4)"
    )


@pytest.mark.parametrize("kernel_size", [1, 2, 3])
@pytest.mark.parametrize("stride",      [1, 2])
@pytest.mark.parametrize("share",       [False, True])
def test_conv2d_border_pixels(kernel_size, stride, share):
    """Border/edge output pixels must match exactly — these are where the
    zero-padding interacts with the indexed -1 sentinel."""
    layer = _make_conv2d(kernel_size, stride, share)
    Cin = layer.in_channels

    # Use an input where each pixel's value is a unique large integer so any
    # wrong neighbor pick shows up as a large numerical discrepancy.
    x_flat = (RNG.integers(1, 100, size=(H * W, Cin))).astype(np.float32)
    x_grid = x_flat.reshape(1, H, W, Cin)

    y_grid = keras.ops.convert_to_numpy(layer(keras.ops.convert_to_tensor(x_grid)))
    y_ref  = _grid_to_flat(y_grid)

    nbr, cells, _ = build_neighbor_table(layer, H, W)
    W_k = get_cell_weights(layer, cells)
    y_idx = indexed_conv2d_forward(x_flat, nbr, W_k)

    max_err = float(np.max(np.abs(y_idx - y_ref)))
    assert max_err < 1e-3, (
        f"Border test Conv2d(kernel={kernel_size}, stride={stride}, share={share}): "
        f"max abs err={max_err:.2e}"
    )


# ---- MaxPool2d equivalence tests --------------------------------------------

@pytest.mark.parametrize("kernel_size", [1, 2, 3])
@pytest.mark.parametrize("stride",      [1, 2])
def test_maxpool2d_indexed_equals_call(kernel_size, stride):
    """indexed_maxpool2d_forward == MaxPool2d.call() for every output pixel."""
    layer = _make_maxpool2d(kernel_size, stride)
    C = 3

    x_flat = RNG.standard_normal((H * W, C)).astype(np.float32)
    x_grid = x_flat.reshape(1, H, W, C)

    y_grid = keras.ops.convert_to_numpy(layer(keras.ops.convert_to_tensor(x_grid)))
    y_ref  = _grid_to_flat(y_grid)

    nbr, _, _ = build_neighbor_table(layer, H, W)
    y_idx = indexed_maxpool2d_forward(x_flat, nbr)

    assert y_idx.shape == y_ref.shape
    max_err = float(np.max(np.abs(y_idx - y_ref)))
    assert max_err < 1e-5, (
        f"MaxPool2d(kernel={kernel_size}, stride={stride}): "
        f"max abs err={max_err:.2e}"
    )


@pytest.mark.parametrize("kernel_size", [1, 2, 3])
@pytest.mark.parametrize("stride",      [1, 2])
def test_maxpool2d_border_pixels(kernel_size, stride):
    """MaxPool border behavior: invalid slots contribute 0 (same as hexagdly
    zero-padding), including the case where real neighbors are all negative."""
    layer = _make_maxpool2d(kernel_size, stride)
    C = 2

    # All-negative input — forces border behavior to matter.
    x_flat = -np.abs(RNG.standard_normal((H * W, C))).astype(np.float32)
    x_grid = x_flat.reshape(1, H, W, C)

    y_grid = keras.ops.convert_to_numpy(layer(keras.ops.convert_to_tensor(x_grid)))
    y_ref  = _grid_to_flat(y_grid)

    nbr, _, _ = build_neighbor_table(layer, H, W)
    y_idx = indexed_maxpool2d_forward(x_flat, nbr)

    max_err = float(np.max(np.abs(y_idx - y_ref)))
    assert max_err < 1e-5, (
        f"MaxPool2d border (all-neg) kernel={kernel_size}, stride={stride}: "
        f"max abs err={max_err:.2e}"
    )


# ---- Sanity: neighbor table shape + sentinel coverage -----------------------

@pytest.mark.parametrize("kernel_size", [1, 2, 3])
def test_neighbor_table_shape_and_sentinels(kernel_size):
    """neighbor_idx has the right shape and at least some -1 sentinels
    (border pixels), and all valid indices are in [0, H*W)."""
    layer = _make_conv2d(kernel_size, stride=1, share_neighbors=False)
    nbr, cells, (H_out, W_out) = build_neighbor_table(layer, H, W)

    N_out = H_out * W_out
    K = len(cells)
    assert nbr.shape == (N_out, K), f"Expected ({N_out},{K}), got {nbr.shape}"
    assert (nbr >= -1).all(), "Unexpected value < -1"
    assert (nbr < H * W).all(), "Index out of bounds"
    # Border pixels must have at least one -1 slot for kernel_size >= 1.
    if kernel_size >= 1:
        assert (nbr == -1).any(), "Expected some -1 border slots"


# ============================================================================
# Phase 2: Conv3d and MaxPool3d
# ============================================================================

D = 8   # depth dimension; small enough to keep tests fast


def _make_conv3d(kernel_size, stride, share_neighbors, Cin=2, Cout=3):
    layer = hgly.Conv3d(
        Cin, Cout,
        kernel_size=kernel_size,
        stride=stride,
        bias=False,
        share_neighbors=share_neighbors,
    )
    layer(keras.ops.zeros((1, D, H, W, Cin)))
    for w in layer.trainable_variables:
        w.assign(RNG.standard_normal(w.shape).astype(np.float32))
    return layer


def _make_maxpool3d(kernel_size, stride):
    layer = hgly.MaxPool3d(kernel_size=kernel_size, stride=stride)
    layer(keras.ops.zeros((1, D, H, W, 1)))
    return layer


def _grid3d_to_flat(grid_out):
    """(1, D_out, H_out, W_out, C) -> (D_out, H_out*W_out, C)."""
    _, D_out, H_out, W_out, C = grid_out.shape
    return grid_out[0].reshape(D_out, H_out * W_out, C)


# ---- Conv3d equivalence tests -----------------------------------------------

@pytest.mark.parametrize("kernel_size", [(1, 1), (2, 2), (1, 2)])
@pytest.mark.parametrize("stride",      [(1, 1), (1, 2)])
@pytest.mark.parametrize("share",       [False, True])
def test_conv3d_indexed_equals_call(kernel_size, stride, share):
    """indexed_conv3d_forward == Conv3d.call() for every output pixel."""
    layer = _make_conv3d(kernel_size, stride, share)
    Cin = layer.in_channels
    depth_stride = layer.depth_stride

    x_flat = RNG.standard_normal((D, H * W, Cin)).astype(np.float32)
    x_grid = x_flat.reshape(1, D, H, W, Cin)

    y_grid = keras.ops.convert_to_numpy(layer(keras.ops.convert_to_tensor(x_grid)))
    y_ref  = _grid3d_to_flat(y_grid)              # (D_out, N_out, Cout)

    nbr, cells, (D_out, H_out, W_out) = build_neighbor_table_3d(layer, D, H, W)
    W_k = get_cell_weights_3d(layer, cells)        # (D_kernel, K, Cin, Cout)
    y_idx = indexed_conv3d_forward(x_flat, nbr, W_k, depth_stride=depth_stride)

    assert y_idx.shape == y_ref.shape, (
        f"shape mismatch: indexed={y_idx.shape}, call={y_ref.shape}"
    )
    max_err = float(np.max(np.abs(y_idx - y_ref)))
    assert max_err < 1e-3, (
        f"Conv3d(kernel={kernel_size}, stride={stride}, share={share}): "
        f"max abs err={max_err:.2e}"
    )


@pytest.mark.parametrize("kernel_size", [(1, 1), (2, 2), (1, 2)])
@pytest.mark.parametrize("stride",      [(1, 1), (1, 2)])
@pytest.mark.parametrize("share",       [False, True])
def test_conv3d_border_pixels(kernel_size, stride, share):
    """Border spatial pixels must match — depth border is implicit via valid stride."""
    layer = _make_conv3d(kernel_size, stride, share)
    Cin = layer.in_channels
    depth_stride = layer.depth_stride

    x_flat = (RNG.integers(1, 50, size=(D, H * W, Cin))).astype(np.float32)
    x_grid = x_flat.reshape(1, D, H, W, Cin)

    y_grid = keras.ops.convert_to_numpy(layer(keras.ops.convert_to_tensor(x_grid)))
    y_ref  = _grid3d_to_flat(y_grid)

    nbr, cells, _ = build_neighbor_table_3d(layer, D, H, W)
    W_k = get_cell_weights_3d(layer, cells)
    y_idx = indexed_conv3d_forward(x_flat, nbr, W_k, depth_stride=depth_stride)

    max_err = float(np.max(np.abs(y_idx - y_ref)))
    assert max_err < 1e-2, (
        f"Conv3d border kernel={kernel_size}, stride={stride}, share={share}: "
        f"max abs err={max_err:.2e}"
    )


# ---- MaxPool3d equivalence tests --------------------------------------------

@pytest.mark.parametrize("kernel_size", [(1, 1), (2, 2), (1, 2)])
@pytest.mark.parametrize("stride",      [(1, 1), (1, 2)])
def test_maxpool3d_indexed_equals_call(kernel_size, stride):
    """indexed_maxpool3d_forward == MaxPool3d.call() for every output pixel."""
    layer = _make_maxpool3d(kernel_size, stride)
    C = 3
    depth_size   = layer.depth_size
    depth_stride = layer.depth_stride

    x_flat = RNG.standard_normal((D, H * W, C)).astype(np.float32)
    x_grid = x_flat.reshape(1, D, H, W, C)

    y_grid = keras.ops.convert_to_numpy(layer(keras.ops.convert_to_tensor(x_grid)))
    y_ref  = _grid3d_to_flat(y_grid)

    nbr, _, _ = build_neighbor_table_3d(layer, D, H, W)
    y_idx = indexed_maxpool3d_forward(x_flat, nbr, depth_size, depth_stride)

    assert y_idx.shape == y_ref.shape
    max_err = float(np.max(np.abs(y_idx - y_ref)))
    assert max_err < 1e-5, (
        f"MaxPool3d(kernel={kernel_size}, stride={stride}): "
        f"max abs err={max_err:.2e}"
    )


@pytest.mark.parametrize("kernel_size", [(1, 1), (2, 2), (1, 2)])
@pytest.mark.parametrize("stride",      [(1, 1), (1, 2)])
def test_maxpool3d_border_pixels(kernel_size, stride):
    """All-negative input forces border slots (0-pad) to dominate — must match."""
    layer = _make_maxpool3d(kernel_size, stride)
    C = 2
    depth_size   = layer.depth_size
    depth_stride = layer.depth_stride

    x_flat = -np.abs(RNG.standard_normal((D, H * W, C))).astype(np.float32)
    x_grid = x_flat.reshape(1, D, H, W, C)

    y_grid = keras.ops.convert_to_numpy(layer(keras.ops.convert_to_tensor(x_grid)))
    y_ref  = _grid3d_to_flat(y_grid)

    nbr, _, _ = build_neighbor_table_3d(layer, D, H, W)
    y_idx = indexed_maxpool3d_forward(x_flat, nbr, depth_size, depth_stride)

    max_err = float(np.max(np.abs(y_idx - y_ref)))
    assert max_err < 1e-5, (
        f"MaxPool3d border (all-neg) kernel={kernel_size}, stride={stride}: "
        f"max abs err={max_err:.2e}"
    )
