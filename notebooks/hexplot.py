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
