"""Hand-verified reference arrays, ported from HexagDLy's own test suite
(https://github.com/ai4iacts/hexagdly/tree/master/tests), translated from
PyTorch's channels-first layout to this package's channels-last layout.

These are *independent* ground truth (worked out by hand on a small 5x8
input), not derived from running either implementation -- the only thing
shared with upstream is the input array and the verified outputs.
"""

import numpy as np


def base_array():
    return np.array([[j * 5 + 1 + i for j in range(8)] for i in range(5)], dtype=np.float32)


CONV2D_SIZE1_STRIDE1 = np.array(
    [
        [9, 39, 45, 99, 85, 159, 125, 136],
        [19, 51, 82, 121, 152, 191, 222, 176],
        [24, 58, 89, 128, 159, 198, 229, 181],
        [29, 65, 96, 135, 166, 205, 236, 186],
        [28, 39, 87, 79, 147, 119, 207, 114],
    ],
    dtype=np.float32,
)

CONV2D_SIZE2_STRIDE1 = np.array(
    [
        [42, 96, 128, 219, 238, 349, 265, 260],
        [67, 141, 194, 312, 354, 492, 388, 361],
        [84, 162, 243, 346, 433, 536, 494, 408],
        [90, 145, 246, 302, 426, 462, 474, 343],
        [68, 104, 184, 213, 314, 323, 355, 245],
    ],
    dtype=np.float32,
)

N_NEIGHBORS_SIZE1 = np.array(
    [
        [3, 6, 4, 6, 4, 6, 4, 4],
        [5, 7, 7, 7, 7, 7, 7, 5],
        [5, 7, 7, 7, 7, 7, 7, 5],
        [5, 7, 7, 7, 7, 7, 7, 5],
        [4, 4, 6, 4, 6, 4, 6, 3],
    ],
    dtype=np.float32,
)

N_NEIGHBORS_SIZE2 = np.array(
    [
        [7, 11, 11, 13, 11, 13, 9, 8],
        [10, 15, 16, 18, 16, 18, 13, 11],
        [12, 16, 19, 19, 19, 19, 16, 12],
        [11, 13, 18, 16, 18, 16, 15, 10],
        [8, 9, 13, 11, 13, 11, 11, 7],
    ],
    dtype=np.float32,
)

MAXPOOL2D_SIZE1_STRIDE1 = np.array(
    [
        [6, 12, 16, 22, 26, 32, 36, 37],
        [7, 13, 17, 23, 27, 33, 37, 38],
        [8, 14, 18, 24, 28, 34, 38, 39],
        [9, 15, 19, 25, 29, 35, 39, 40],
        [10, 15, 20, 25, 30, 35, 40, 40],
    ],
    dtype=np.float32,
)

MAXPOOL2D_SIZE2_STRIDE1 = np.array(
    [
        [12, 17, 22, 27, 32, 37, 37, 38],
        [13, 18, 23, 28, 33, 38, 38, 39],
        [14, 19, 24, 29, 34, 39, 39, 40],
        [15, 20, 25, 30, 35, 40, 40, 40],
        [15, 20, 25, 30, 35, 40, 40, 40],
    ],
    dtype=np.float32,
)


def slice_stride2(a):
    out = np.zeros((2, 4), dtype=np.float32)
    pos = [
        (0, 0, 0, 0),
        (0, 1, 1, 2),
        (0, 2, 0, 4),
        (0, 3, 1, 6),
        (1, 0, 2, 0),
        (1, 1, 3, 2),
        (1, 2, 2, 4),
        (1, 3, 3, 6),
    ]
    for p in pos:
        out[p[0], p[1]] = a[p[2], p[3]]
    return out


def slice_stride3(a):
    out = np.zeros((2, 3), dtype=np.float32)
    pos = [(0, 0, 0, 0), (0, 1, 1, 3), (0, 2, 0, 6), (1, 0, 3, 0), (1, 1, 4, 3), (1, 2, 3, 6)]
    for p in pos:
        out[p[0], p[1]] = a[p[2], p[3]]
    return out


def apply_stride(a, stride):
    if stride == 2:
        return slice_stride2(a)
    if stride == 3:
        return slice_stride3(a)
    return a


def conv2d_expected(in_channels, kernel_size, stride, bias_value):
    """Sum-of-neighbours expected output for an all-ones kernel, out_channels=1."""
    conv, nn = (
        (CONV2D_SIZE1_STRIDE1, N_NEIGHBORS_SIZE1)
        if kernel_size == 1
        else (CONV2D_SIZE2_STRIDE1, N_NEIGHBORS_SIZE2)
    )
    channel_dist = 1000
    out = np.sum(
        np.stack([channel * channel_dist * nn + conv for channel in range(in_channels)]),
        axis=0,
    )
    out = apply_stride(out, stride)
    return out + bias_value


def conv2d_input_nhwc(in_channels):
    channel_dist = 1000
    array = base_array()
    x = np.stack([j * channel_dist + array for j in range(in_channels)], axis=-1)
    return x[None, ...].astype(np.float32)  # (1, H, W, in_channels)


def maxpool2d_expected(in_channels, kernel_size, stride):
    pooled = MAXPOOL2D_SIZE1_STRIDE1 if kernel_size == 1 else MAXPOOL2D_SIZE2_STRIDE1
    pooled = apply_stride(pooled, stride)
    channel_dist = 1000
    return np.stack([channel * channel_dist + pooled for channel in range(in_channels)], axis=-1)


def maxpool2d_input_nhwc(in_channels):
    return conv2d_input_nhwc(in_channels)


def conv3d_expected(
    in_channels, depth, kernel_size_depth, kernel_size_hex, stride_depth, stride_hex, bias_value
):
    """Sum-of-neighbours expected output (depth axis) for an all-ones kernel,
    out_channels=1, channels-last (D, H, W) with the channel axis already
    summed away (out_channels=1)."""
    depth_dist = 40
    channel_dist = 1000
    conv, nn = (
        (CONV2D_SIZE1_STRIDE1, N_NEIGHBORS_SIZE1)
        if kernel_size_hex == 1
        else (CONV2D_SIZE2_STRIDE1, N_NEIGHBORS_SIZE2)
    )
    depth_steps = int(np.ceil((depth - kernel_size_depth + 1) / stride_depth))
    layers = []
    for dstep in range(depth_steps):
        layer = np.sum(
            np.stack(
                [
                    (channel * channel_dist + ((dstep * stride_depth) + dsize) * depth_dist) * nn
                    + conv
                    for dsize in range(kernel_size_depth)
                    for channel in range(in_channels)
                ]
            ),
            axis=0,
        )
        layers.append(apply_stride(layer, stride_hex))
    return np.stack(layers, axis=0) + bias_value  # (depth_out, H_out, W_out)


def conv3d_input_ndhwc(in_channels, depth):
    depth_dist = 40
    channel_dist = 1000
    array = base_array()
    x = np.stack(
        [
            channel * channel_dist + np.stack([d * depth_dist + array for d in range(depth)])
            for channel in range(in_channels)
        ],
        axis=-1,
    )  # (D, H, W, in_channels)
    return x[None, ...].astype(np.float32)


def maxpool3d_expected(
    in_channels, depth, kernel_size_depth, kernel_size_hex, stride_depth, stride_hex
):
    """channels-last (D_out, H_out, W_out, in_channels)."""
    depth_dist = 40
    channel_dist = 1000
    pool2d = MAXPOOL2D_SIZE1_STRIDE1 if kernel_size_hex == 1 else MAXPOOL2D_SIZE2_STRIDE1
    pool2d = apply_stride(pool2d, stride_hex)
    depth_steps = int(np.ceil((depth - kernel_size_depth + 1) / stride_depth))
    depth_layers = np.stack(
        [
            (dstep * stride_depth + kernel_size_depth - 1) * depth_dist + pool2d
            for dstep in range(depth_steps)
        ]
    )  # (D_out, H_out, W_out)
    return np.stack(
        [channel * channel_dist + depth_layers for channel in range(in_channels)], axis=-1
    )
