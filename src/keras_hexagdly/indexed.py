"""Neighbor-table builder and indexed forward references.

Covers: Conv2d, MaxPool2d, Conv3d, MaxPool3d.

No hls4ml dependency here — this module is used by:
  - tests/test_indexed_equivalence.py   (bit-exact equivalence proof)
  - hls4ml_ext.py                        (EinsumDense / MaxPooling1D export)

Core idea
---------
A hex Conv2d on a zig-zag (H, W) grid and a flat-pixel gather+MAC over a
(N, K) neighbor table are mathematically the same operation.  We derive the
table *empirically* by firing single-cell impulses through the existing call()
path — exactly the same impulse-measurement technique the package already uses
in ring_maps_2d / _tap_offset.  No coordinate arithmetic, no assumptions about
the zig-zag layout: the table is exact by construction.

For 3D layers (Conv3d, MaxPool3d) the spatial neighbor table is identical to
the 2D case — the depth axis is an ordinary dense axis handled separately:
  - Conv3d:   out[t_out, n, o] = sum_{d, k, c} x[t_in(t_out,d), nbr[n,k], c]
                                              * W[d, k, c, o]
  - MaxPool3d: out[t_out, n, c] = max over depth-window and hex-neighbors of x

The layers themselves are never modified; this module only *reads* their weights
and geometry.

Flat-index convention
---------------------
For a grid of shape (H, W), flat index m = h * W + w (standard raster scan).
``build_neighbor_table`` returns indices in this convention.  For the hls4ml
export the caller is responsible for mapping CameraGeometry pixel IDs to (h, w)
positions (via GridTransform) and re-indexing if needed.
"""

import numpy as np
import keras

from keras_hexagdly.layers import Conv2d_CustomKernel, ring_maps_2d


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cell_list(kernel_size):
    """Ordered list of (sub_kernel_idx, row, col) for all kernel cells.

    Iterates sub-kernels 0..n, rows, then cols — the same order the HexBase
    forward pass processes them.  This order is the canonical slot ordering k
    used throughout this module.
    """
    cells = []
    n = kernel_size
    for i in range(n + 1):
        kh = 1 + 2 * n - i
        kw = 1 if i == 0 else 2
        for r in range(kh):
            for c in range(kw):
                cells.append((i, r, c))
    return cells


def _probe_layer(kernel_size, stride, i_active, r_active, c_active):
    """Return a Conv2d_CustomKernel with weight=1.0 ONLY at cell (i,r,c)."""
    n = kernel_size
    sub_kernels = []
    for ii in range(n + 1):
        kh = 1 + 2 * n - ii
        kw = 1 if ii == 0 else 2
        sk = np.zeros((1, 1, kh, kw), np.float32)  # (out, in, rows, cols)
        if ii == i_active:
            sk[0, 0, r_active, c_active] = 1.0
        sub_kernels.append(sk)
    return Conv2d_CustomKernel(sub_kernels=sub_kernels, stride=stride, bias=None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_neighbor_table(layer, H, W):
    """Measure the neighbor table for a Conv2d or MaxPool2d layer.

    For each kernel cell k, fire a Conv2d_CustomKernel probe with weight=1
    only at cell k, and input x[m] = m+1 (distinct per pixel).  The output at
    position n_out equals the flat input index of the pixel that feeds n_out
    via cell k, or 0 if that slot is zero-padded (border).

    Spatial routing is identical for Conv2d and MaxPool2d at the same
    kernel_size / stride: both use the same HexBase pad/slice/stride arithmetic,
    differing only in what they DO with the gathered values (MAC vs max).

    Args:
        layer:   A keras_hexagdly Conv2d or MaxPool2d (built or unbuilt).
        H, W:    Spatial input dimensions.

    Returns:
        neighbor_idx:  int64 ndarray (N_out, K), -1 = border/invalid slot.
        cells:         list of (i, r, c) tuples, length K.  cells[k] is the
                       sub-kernel cell for slot k.
        out_shape:     (H_out, W_out) tuple.
    """
    kernel_size = layer.kernel_size
    stride = layer.hexbase_stride
    cells = _cell_list(kernel_size)
    K = len(cells)

    # Determine output shape with a single forward pass using the layer's actual
    # channel count (Conv2d may have Cin > 1; MaxPool2d is channel-agnostic but
    # also accepts any C, so we use 1 there).
    cin = getattr(layer, "in_channels", None) or 1
    dummy = keras.ops.zeros((1, H, W, cin))
    if not layer.built:
        layer(dummy)
    out_dummy = keras.ops.convert_to_numpy(layer(dummy))
    _, H_out, W_out, _ = out_dummy.shape
    N_out = H_out * W_out

    # Probe input: pixel m carries value m+1 (≥1) so any nonzero output
    # unambiguously identifies the contributing input pixel.  Probes are always
    # Cin=1 / Cout=1 so this is always valid regardless of layer.in_channels.
    x_probe = np.arange(1, H * W + 1, dtype=np.float32).reshape(1, H, W, 1)
    x_t = keras.ops.convert_to_tensor(x_probe)

    neighbor_idx = np.full((N_out, K), -1, dtype=np.int64)

    for k_idx, (i, r, c) in enumerate(cells):
        probe = _probe_layer(kernel_size, stride, i, r, c)
        out_np = keras.ops.convert_to_numpy(probe(x_t))[0, :, :, 0]  # (H_out, W_out)
        out_flat = out_np.reshape(N_out)
        # Nonzero (≥0.5) output: the value is the contributing flat input index +1.
        # Zero: this slot is zero-padded (border), leave as -1.
        for n_out_idx in range(N_out):
            val = out_flat[n_out_idx]
            if val > 0.5:
                neighbor_idx[n_out_idx, k_idx] = int(round(val)) - 1

    return neighbor_idx, cells, (H_out, W_out)


def get_cell_weights(layer, cells):
    """Extract a (K, Cin, Cout) weight array from a built Conv2d.

    For share_neighbors=True:  each cell's weight = ring_weights[ring_of_cell].
    For share_neighbors=False: each cell's weight = _base_kernels[i][r, c, :, :].

    Args:
        layer:  A built Conv2d instance.
        cells:  The cell list returned by build_neighbor_table (or _cell_list).

    Returns:
        W:  float32 ndarray (K, Cin, Cout).
    """
    n = layer.kernel_size

    if layer.share_neighbors:
        ring_maps, _ = ring_maps_2d(n)
        W_rings = layer.ring_weights.numpy()          # (num_rings, Cin, Cout)
        W = np.stack(
            [W_rings[int(ring_maps[i][r, c])] for i, r, c in cells]
        )
    else:
        base = [k.numpy() for k in layer._base_kernels]  # [(kh,kw,Cin,Cout)]
        W = np.stack([base[i][r, c] for i, r, c in cells])

    return W.astype(np.float32)  # (K, Cin, Cout)


def indexed_conv2d_forward(x_flat, neighbor_idx, W):
    """Vectorized indexed gather + MAC (NumPy reference for Conv2d).

    Invalid neighbor slots (neighbor_idx == -1) contribute 0, matching
    hexagdly's zero-padding at the grid border.

    Args:
        x_flat:       float32 ndarray (N_in, Cin).
        neighbor_idx: int64  ndarray (N_out, K), -1 = invalid.
        W:            float32 ndarray (K, Cin, Cout).

    Returns:
        float32 ndarray (N_out, Cout).
    """
    valid = neighbor_idx >= 0                                # (N_out, K)
    safe_idx = np.where(valid, neighbor_idx, 0)              # clamp for indexing
    gathered = x_flat[safe_idx]                             # (N_out, K, Cin)
    gathered = np.where(valid[:, :, np.newaxis], gathered, 0.0)
    return np.einsum("nkc,kco->no", gathered, W)


def indexed_maxpool2d_forward(x_flat, neighbor_idx):
    """Vectorized indexed gather + max (NumPy reference for MaxPool2d).

    Invalid neighbor slots contribute 0.0, matching hexagdly's zero-padding
    at the grid border.  NOTE: if all valid neighbors of a pixel are negative,
    the 0-padded slots will dominate — this is the same behavior as hexagdly.

    Args:
        x_flat:       float32 ndarray (N_in, C).
        neighbor_idx: int64  ndarray (N_out, K), -1 = invalid.

    Returns:
        float32 ndarray (N_out, C).
    """
    valid = neighbor_idx >= 0                                # (N_out, K)
    safe_idx = np.where(valid, neighbor_idx, 0)
    gathered = x_flat[safe_idx]                             # (N_out, K, C)
    gathered = np.where(valid[:, :, np.newaxis], gathered, 0.0)
    return gathered.max(axis=1)                             # (N_out, C)


# ---------------------------------------------------------------------------
# 3D extensions (Conv3d, MaxPool3d)
# ---------------------------------------------------------------------------
# The spatial neighbor table is shared with the 2D case — we reuse
# build_neighbor_table() on a Conv2d/MaxPool2d with the same hex geometry.
# The depth axis is handled separately as a regular dense dimension.


def build_neighbor_table_3d(layer, D, H, W):
    """Measure the spatial neighbor table for a Conv3d or MaxPool3d layer.

    Constructs a temporary Conv2d/MaxPool2d proxy with the same hex geometry
    (hexbase_size, hexbase_stride) and runs the 2D probe to get the spatial
    neighbor table.  The depth axis is not probed here — it is a regular dense
    axis; see ``get_cell_weights_3d`` for how depth taps are handled.

    Args:
        layer:    A keras_hexagdly Conv3d or MaxPool3d (built or unbuilt).
        D, H, W:  Input spatial dimensions (depth, height, width).

    Returns:
        neighbor_idx:  int64 ndarray (N_out, K), -1 = border/invalid slot.
        cells:         list of (i, r, c) tuples, length K.
        out_shape:     (D_out, H_out, W_out) tuple.
    """
    import keras_hexagdly as hgly

    hex_size   = layer.hexbase_size
    hex_stride = layer.hexbase_stride
    depth_size   = layer.depth_size
    depth_stride = layer.depth_stride

    # Build a 2D proxy to derive the spatial table.
    proxy = hgly.MaxPool2d(kernel_size=hex_size, stride=hex_stride)
    nbr, cells, (H_out, W_out) = build_neighbor_table(proxy, H, W)

    # Compute output depth (same arithmetic as a 1D valid conv).
    D_out = (D - depth_size) // depth_stride + 1

    return nbr, cells, (D_out, H_out, W_out)


def get_cell_weights_3d(layer, cells):
    """Extract a (D, K, Cin, Cout) weight array from a built Conv3d.

    Depth taps are kept fully independent (one weight per depth slice), as
    in the original Conv3d design.  The K spatial slots are ordered to match
    ``cells`` from build_neighbor_table_3d / _cell_list.

    For share_neighbors=True:  ring_weights is (D, num_rings, Cin, Cout);
                                W[d, k] = ring_weights[d, ring_of_cell_k].
    For share_neighbors=False: _base_kernels[i] is (D, kh, kw, Cin, Cout);
                                W[d, k=(i,r,c)] = _base_kernels[i][d, r, c].

    Returns:
        W:  float32 ndarray (D, K, Cin, Cout).
    """
    n = layer.hexbase_size

    if layer.share_neighbors:
        ring_maps, _ = ring_maps_2d(n)
        W_rings = layer.ring_weights.numpy()     # (D, num_rings, Cin, Cout)
        D = W_rings.shape[0]
        ring_indices = [int(ring_maps[i][r, c]) for i, r, c in cells]
        W = W_rings[:, ring_indices, :, :]       # (D, K, Cin, Cout)
    else:
        base = [k.numpy() for k in layer._base_kernels]  # [(D,kh,kw,Cin,Cout)]
        D = base[0].shape[0]
        K = len(cells)
        Cin  = base[0].shape[-2]
        Cout = base[0].shape[-1]
        W = np.zeros((D, K, Cin, Cout), np.float32)
        for k_idx, (i, r, c) in enumerate(cells):
            W[:, k_idx, :, :] = base[i][:, r, c, :, :]

    return W.astype(np.float32)  # (D, K, Cin, Cout)


def indexed_conv3d_forward(x_flat, neighbor_idx, W, depth_stride=1):
    """Vectorized indexed gather + MAC for Conv3d (NumPy reference).

    Args:
        x_flat:        float32 ndarray (D_in, N_in, Cin).
        neighbor_idx:  int64  ndarray (N_out, K), -1 = invalid.
        W:             float32 ndarray (D_kernel, K, Cin, Cout).
        depth_stride:  int, stride along the depth axis.

    Returns:
        float32 ndarray (D_out, N_out, Cout).
    """
    D_in, N_in, Cin = x_flat.shape
    D_kernel, K, _, Cout = W.shape
    D_out = (D_in - D_kernel) // depth_stride + 1
    N_out = neighbor_idx.shape[0]

    valid   = neighbor_idx >= 0                    # (N_out, K)
    safe_idx = np.where(valid, neighbor_idx, 0)    # (N_out, K)

    out = np.zeros((D_out, N_out, Cout), np.float32)
    for t_out in range(D_out):
        for d in range(D_kernel):
            t_in = t_out * depth_stride + d
            x_t = x_flat[t_in]                                # (N_in, Cin)
            gathered = x_t[safe_idx]                          # (N_out, K, Cin)
            gathered = np.where(valid[:, :, np.newaxis], gathered, 0.0)
            out[t_out] += np.einsum("nkc,kco->no", gathered, W[d])

    return out


def indexed_maxpool3d_forward(x_flat, neighbor_idx, depth_size, depth_stride=1):
    """Vectorized indexed gather + max for MaxPool3d (NumPy reference).

    Invalid spatial slots contribute 0.0 (same as hexagdly zero-padding).

    Args:
        x_flat:        float32 ndarray (D_in, N_in, C).
        neighbor_idx:  int64  ndarray (N_out, K), -1 = invalid.
        depth_size:    int, depth kernel size.
        depth_stride:  int, stride along the depth axis.

    Returns:
        float32 ndarray (D_out, N_out, C).
    """
    D_in, N_in, C = x_flat.shape
    N_out = neighbor_idx.shape[0]
    D_out = (D_in - depth_size) // depth_stride + 1

    valid    = neighbor_idx >= 0
    safe_idx = np.where(valid, neighbor_idx, 0)

    out = np.full((D_out, N_out, C), -np.inf, np.float32)
    for t_out in range(D_out):
        for d in range(depth_size):
            t_in = t_out * depth_stride + d
            x_t = x_flat[t_in]                               # (N_in, C)
            gathered = x_t[safe_idx]                         # (N_out, K, C)
            gathered = np.where(valid[:, :, np.newaxis], gathered, 0.0)
            out[t_out] = np.maximum(out[t_out], gathered.max(axis=1))

    return out
