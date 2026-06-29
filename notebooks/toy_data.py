"""Toy hexagonal shapes for the example notebooks, ported from HexagDLy's
notebooks/example_utils.py (`put_shape`), kept as plain numpy (no torch
dependency) and returning channels-last arrays.
"""

import numpy as np

SHAPES = {
    "small_hexagon": [(1, 0)],
    "medium_hexagon": [(4, 0)],
    "snowflake_1": [(3, 0)],
    "snowflake_2": [(1, 0), (4.1, 3.9)],
    "snowflake_3": [(7, 3)],
    "snowflake_4": [(7, 0)],
    "double_hex": [(10, 5)],
}


def put_shape(nx, ny, cx, cy, params):
    d = np.zeros((nx, ny))
    i = np.indices((nx, ny)).astype(float)
    i[0] -= cx
    i[1] -= cy
    i[0] *= 1.73205 / 2
    if np.mod(cx, 2) == 0:
        i[1][np.mod(cx + 1, 2)::2] += 0.5
    else:
        i[1][np.mod(cx + 1, 2)::2] -= 0.5
    di = i[0] ** 2 + i[1] ** 2
    for t1, t2 in params:
        di = np.where(np.logical_and(di >= t2, di <= t1), 1, di)
    di = np.where(di > 1.1, 0, di)
    return di.transpose()


def toy_hex_image(shape, H, W, channels=1, px=None, py=None):
    """A single (1, H, W, channels) channels-last array with `shape` stamped
    at (px, py), or at a random position if either is None."""
    out = np.zeros((1, H, W, channels), dtype=np.float32)
    for c in range(channels):
        cx = px if px is not None else int(W * np.random.random())
        cy = py if py is not None else int(H * np.random.random())
        out[0, :, :, c] += put_shape(W, H, cx, cy, SHAPES[shape])
    return out


class ToyDataset:
    """A labelled, batched set of toy hexagonal images, ported from
    HexagDLy's notebooks/example_utils.py:toy_dataset (there backed by a
    torch DataLoader; here just numpy arrays, since `model.fit` needs no
    custom dataloader)."""

    def __init__(self, shape_list, n_per_shape, H=16, W=16, channels=1, noisy=None):
        self.shape_list = list(shape_list)
        self.n_per_shape = n_per_shape
        self.H, self.W, self.channels = H, W, channels
        self.noisy = noisy
        self.images = None
        self.labels = None

    def create(self, seed=None):
        rng = np.random.default_rng(seed)
        images, labels = [], []
        for label, shape in enumerate(self.shape_list):
            for _ in range(self.n_per_shape):
                img = np.zeros((self.H, self.W, self.channels), dtype=np.float32)
                for c in range(self.channels):
                    cx = int(rng.integers(0, self.W))
                    cy = int(rng.integers(0, self.H))
                    img[:, :, c] += put_shape(self.W, self.H, cx, cy, SHAPES[shape])
                if self.noisy:
                    img += rng.normal(0, self.noisy, img.shape).astype(np.float32)
                images.append(img)
                labels.append(label)
        order = rng.permutation(len(images))
        self.images = np.stack(images)[order]
        self.labels = np.asarray(labels)[order]
        return self

    def to_arrays(self):
        return self.images, self.labels
