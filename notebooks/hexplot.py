"""Minimal hex-grid plotting helper for the example notebooks.

A small, channels-last re-implementation of HexagDLy's
notebooks/hexagdly_tools.py:plot_hextensor -- same hexagon placement
(addressing scheme), just reading (N, H, W, C) tensors instead of (N, C, H, W).
"""

import numpy as np
import keras
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from matplotlib import gridspec


def plot_hextensor(tensor, image_range=(0, None), channel_range=(0, None),
                    cmap="Greys", figname="figure"):
    """Plot a channels-last (N, H, W, C) tensor in its hexagonal layout.

    Pass either one image with several channels, or one channel across
    several images (exactly one of the two ranges must select a single
    element). `tensor` may be a plain numpy array or a Keras layer's output
    tensor on any backend/device -- converted via keras.ops.convert_to_numpy
    so a GPU-resident torch/jax tensor doesn't crash a bare np.asarray.
    """
    tensor = keras.ops.convert_to_numpy(tensor)
    sub = tensor[image_range[0]:image_range[1], :, :, channel_range[0]:channel_range[1]]
    n_images, H, W, n_channels = sub.shape
    if n_images != 1 and n_channels != 1:
        raise ValueError("Select one image and N channels, or one channel and N images.")
    n_panels = max(n_images, n_channels)

    fig = plt.figure(figname, (5, 5))
    fig.clear()
    nrows = int(np.ceil(np.sqrt(n_panels)))
    gs = gridspec.GridSpec(nrows, nrows)
    gs.update(wspace=0, hspace=0)

    for i in range(n_panels):
        img = sub[i, :, :, 0] if n_images >= n_channels else sub[0, :, :, i]
        hexagons, intensities = [], []
        for x in range(W):
            for y in range(H):
                if np.isnan(img[y, x]):
                    continue
                hexagons.append(RegularPolygon(
                    (x * np.sqrt(3) / 2, -(y + np.mod(x, 2) * 0.5)),
                    6, radius=0.577349, orientation=np.pi / 6))
                intensities.append(img[y, x])
        ax = fig.add_subplot(gs[i])
        ax.set_xlim([-1, H])
        ax.set_ylim([-1.15 * W - 1, 1])
        ax.set_axis_off()
        p = PatchCollection(np.array(hexagons), cmap=cmap, alpha=0.9,
                             edgecolors="k", linewidth=1)
        p.set_array(np.array(intensities))
        ax.add_collection(p)
        ax.set_aspect("equal")
    plt.tight_layout()
    return fig


def _hex_shell_offsets(grid_size, center, ring):
    """(dr, dc) offsets of the cells at hex distance exactly ``ring``.

    Built with ``toy_data.put_shape`` -- the same generator the example images
    use -- so the offsets land in exactly the layout ``plot_hextensor`` draws.
    The hex ball of radius r is ``put_shape([(r**2 + 0.5, 0)])``; the shell is
    ball(r) minus ball(r-1). (``put_shape`` leaves small non-zero values around
    the centre, so ring 0 is special-cased to just the centre cell.)
    """
    from toy_data import put_shape

    def ball(r):
        if r == 0:
            return {(0, 0)}
        d = put_shape(grid_size, grid_size, center, center, [(r * r + 0.5, 0)])
        ys, xs = np.where(d > 0)
        return set((int(y - center), int(x - center)) for y, x in zip(ys, xs))

    return ball(ring) - ball(ring - 1) if ring else ball(0)


def _crop_to_footprint(grid):
    """Crop a (g, g, C) grid to its non-NaN cells, keeping an even left column.

    plot_hextensor reads each cell's vertical shift from its column index
    *within the array it is handed*, so the crop must start on an even column
    or every cell's parity (and shift) flips. Returns a (1, h, w, C) tensor.
    """
    rows = np.where(~np.all(np.isnan(grid), axis=(1, 2)))[0]
    cols = np.where(~np.all(np.isnan(grid), axis=(0, 2)))[0]
    col_lo = cols.min() - (cols.min() % 2)  # keep an even left edge
    return grid[np.newaxis, rows.min() : rows.max() + 1, col_lo : cols.max() + 1, :]


def _shared_kernel_hextensor(layer, in_channel):
    """Ring-shared kernels: paint each ring's weight onto its true hex shell.

    With share_neighbors the layer has one weight per hexagonal ring
    (``layer.ring_weights``, shape ``(num_rings, in, out)``). Each weight is
    placed on every cell at that hex distance, using put_shape geometry so the
    rings line up with how plot_hextensor draws the grid.

    The geometry is taken from put_shape (not the layer's own kernel offsets)
    because the conv's internal sub-kernel ring indexing is vertically mirrored
    relative to plot_hextensor's drawing convention -- colouring by the conv's
    ring index does NOT give concentric rings on screen. Hex *distance* is
    frame-independent, so placing ring weight w_r on the true distance-r shell
    is both what the weight means and consistent with the example images.
    """
    n = layer.hexbase_size
    ring_weights = keras.ops.convert_to_numpy(layer.ring_weights)  # (rings, in, out)
    out_channels = ring_weights.shape[-1]

    g = 4 * n + 3
    cen = (g // 2) - (g // 2) % 2  # an even centre column
    grid = np.full((g, g, out_channels), np.nan, dtype=np.float32)
    for ring in range(n + 1):
        for dr, dc in _hex_shell_offsets(g, cen, ring):
            grid[cen + dr, cen + dc, :] = ring_weights[ring, in_channel, :]
    return _crop_to_footprint(grid)


def _impulse_response_hextensor(layer, in_channel):
    """General kernels: visualise the filter via its impulse response.

    A single 1.0 (in input channel ``in_channel``) is fed through the trained
    layer and the bias subtracted back out; the output is the filter's response
    to a point source -- the standard way to picture a conv kernel, and it works
    for any layer (ring-shared or not). The impulse is placed on an odd column,
    where the hexagonal response footprint coincides exactly with a put_shape
    hexagon, so it crops to a clean hex tile. Cells outside the footprint are
    left NaN so plot_hextensor skips them rather than drawing them as zeros.
    """
    n = layer.hexbase_size
    in_channels = layer.in_channels
    g = 6 * n + 11
    cen = g // 2
    cen = cen if cen % 2 == 1 else cen + 1  # impulse on an odd column
    impulse = np.zeros((1, g, g, in_channels), dtype=np.float32)
    impulse[0, cen, cen, in_channel] = 1.0

    response = keras.ops.convert_to_numpy(layer(keras.ops.convert_to_tensor(impulse)))[0]
    if layer.use_bias:
        response = response - keras.ops.convert_to_numpy(layer.bias_tensor)

    grid = np.where(
        np.any(np.abs(response) > 1e-6, axis=-1, keepdims=True), response, np.nan
    ).astype(np.float32)
    return _crop_to_footprint(grid)


def kernel_to_hextensor(layer, in_channel=0):
    """Turn a trained hexagdly Conv2d layer's kernel into a hex-image tensor.

    Returns a ``(1, h, w, out_channels)`` tensor -- one panel per output channel
    -- ready for ``plot_hextensor``. Works for any 2D hexagdly conv:

      * share_neighbors=True  -> the three (or n+1) ring weights are drawn as
        concentric hexagonal rings.
      * share_neighbors=False -> the filter is shown via its impulse response
        (its reaction to a single point input), a textured hexagonal tile.

    ``in_channel`` selects which input channel's slice of the kernel to show
    (relevant when the layer has more than one input channel).
    """
    if getattr(layer, "share_neighbors", False):
        return _shared_kernel_hextensor(layer, in_channel)
    return _impulse_response_hextensor(layer, in_channel)
