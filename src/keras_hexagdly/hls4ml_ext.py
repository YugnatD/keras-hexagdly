"""hls4ml export bridge for keras-hexagdly layers.

Usage
-----
    from keras_hexagdly.hls4ml_ext import patch_model_for_hls
    import hls4ml

    hls_model_input = patch_model_for_hls(trained_model, strategy="slotwise")
    cfg = hls4ml.utils.config_from_keras_model(hls_model_input, backend="Vivado")
    hls_model = hls4ml.converters.convert_from_keras_model(
        hls_model_input, hls_config=cfg, backend="Vivado", ...
    )

What it does
------------
keras-hexagdly layers operate on zig-zag (H, W) grid tensors and use the
HexBase sub-kernel decomposition internally.  hls4ml cannot synthesize that
decomposition.  This module builds an equivalent Keras model using only
hls4ml-native layers (EinsumDense, MaxPooling1D, Reshape, Cropping1D,
ZeroPadding1D) that produce **bit-identical float32 results** to the original.

Export strategies
-----------------
Two strategies are available via the ``strategy`` argument to
``patch_model_for_hls``:

"slotwise" (default, recommended for synthesis):
    Conv2d/Conv3d: K separate (gather + MAC) pairs, one per neighbor slot.
    Each gather is an EinsumDense with a (N_out, N_in) selection matrix;
    each MAC is an EinsumDense with a (Cin, Cout) weight matrix.
    Results are summed via Add.  The largest single static array is
    (N_out, N_in) per slot — ~160k entries at 20×20, well within Vitis HLS
    elaboration limits.

"folded":
    Conv2d/Conv3d: one EinsumDense with a (N_in, Cin, N_out, Cout) kernel
    where the gather is folded into the weight.  Simpler graph but the matrix
    is N_in*Cin*N_out*Cout entries — 1.28M at 20×20/8-filter, which causes
    Vitis HLS clang to segfault during static elaboration.  Useful for
    C-simulation at any size but not for RTL synthesis beyond small grids.

"gather":
    A fused hex conv layer (HexConvLineBuffer), backed by custom
    hls4ml KerasV3LayerHandlers + HLS C++ kernels.  Scales to full camera size
    (DigiCam N=1296) without a large static array.  Works in io_parallel and
    io_stream; in io_stream it materialises the frame and the K× gathered tensor
    in registers (see the linebuffer strategy for the memory-lean alternative).

"linebuffer" (Conv2d/Conv3d and MaxPool2d/MaxPool3d):
    A single fused layer per op (HexConvLineBuffer / HexConvLineBuffer3D /
    HexPoolLineBuffer / HexPoolLineBuffer3D) that keeps the (H, W) grid.  In
    io_stream the HLS kernel slides a line buffer over the streamed image — only
    rows_resident+1 (= 2*kernel_size + 1) input rows resident — instead of
    buffering the whole frame + gathered tensor, which sharply cuts FF/LUT.  In
    io_parallel a line buffer gives no benefit (the gather im2col is already
    frame-buffered), so the handler prints a warning and falls back to a
    frame-buffered fused kernel (same cost as "gather").

    2D conv/pool support any hex stride: output pixel (oh, ow) maps to input base
    (stride*oh, stride*ow) with constant per-parity tap offsets.  The 3D variants
    are hex-stride 1 (depth handled as a dense/pooled axis); the depth axis is
    depth_size frames resident — the optimal streaming-3D memory floor (3D cannot
    reach O(rows) like 2D: depth taps are N=H*W tokens apart in the stream).

MaxPool2d uses a gather EinsumDense + MaxPooling1D regardless of strategy
(max is not linear so neither folded nor slotwise applies; the gather matrix
is (N_out*K, N_in) which is smaller and has not caused elaboration issues at
toy scale).

The original model is NOT modified; a new Keras model is returned.
hls4ml is NOT imported here; the patched model is plain Keras.

Flat-index convention
---------------------
Spatial flat index m = h * W + w (raster scan).

Border behavior
---------------
Invalid neighbor slots (-1 in the table) contribute 0.0, matching hexagdly's
zero-padding at the grid border.
"""

import keras
import numpy as np

import keras_hexagdly as hgly
from keras_hexagdly.indexed import (
    build_neighbor_table,
    build_neighbor_table_3d,
    get_cell_weights,
    get_cell_weights_3d,
)

# ---------------------------------------------------------------------------
# Per-layer replacement builders
# ---------------------------------------------------------------------------


def _conv2d_replacement(layer, x):
    """Conv2d -> Reshape + EinsumDense + Reshape.

    The (N_in, Cin, N_out, Cout) kernel A encodes the neighbor gather folded
    into the MAC: A[m, c, n, o] += W_k[k, c, o] for each slot k where
    nbr[n, k] == m.  Invalid slots (-1) are skipped so they contribute 0.
    """
    H, W = int(x.shape[1]), int(x.shape[2])
    Cin = int(x.shape[3])
    N_in = H * W
    Cout = layer.out_channels

    nbr, cells, (H_out, W_out) = build_neighbor_table(layer, H, W)
    W_k = get_cell_weights(layer, cells)  # (K, Cin, Cout)
    N_out = H_out * W_out
    K = len(cells)

    A = np.zeros((N_in, Cin, N_out, Cout), np.float32)
    for n in range(N_out):
        for k in range(K):
            m = int(nbr[n, k])
            if m >= 0:
                A[m, :, n, :] += W_k[k]

    x_flat = keras.layers.Reshape((N_in, Cin), name=f"{layer.name}_reshape_in")(x)

    use_bias = layer.use_bias
    einsum = keras.layers.EinsumDense(
        "amc,mcno->ano",
        output_shape=(N_out, Cout),
        bias_axes="o" if use_bias else None,
        name=f"{layer.name}_einsum",
    )
    y_flat = einsum(x_flat)
    if use_bias:
        einsum.set_weights([A, layer.bias_tensor.numpy()])
    else:
        einsum.set_weights([A])

    return keras.layers.Reshape((H_out, W_out, Cout), name=f"{layer.name}_reshape_out")(y_flat)


def _maxpool2d_replacement(layer, x):
    """MaxPool2d -> Reshape + EinsumDense (gather) + MaxPooling1D(K,K) + Reshape.

    The (N_out*K, N_in) selection matrix S gathers the K neighbors of every
    output pixel into a contiguous N_out*K-length sequence.  MaxPooling1D with
    pool_size=K and strides=K then reduces each K-block to a single max value.
    """
    H, W = int(x.shape[1]), int(x.shape[2])
    C = int(x.shape[3])
    N_in = H * W

    nbr, cells, (H_out, W_out) = build_neighbor_table(layer, H, W)
    N_out = H_out * W_out
    K = len(cells)

    S = np.zeros((N_out * K, N_in), np.float32)
    for n in range(N_out):
        for k in range(K):
            m = int(nbr[n, k])
            if m >= 0:
                S[n * K + k, m] = 1.0

    x_flat = keras.layers.Reshape((N_in, C), name=f"{layer.name}_reshape_in")(x)

    gather = keras.layers.EinsumDense(
        "amc,pm->apc",
        output_shape=(N_out * K, C),
        bias_axes=None,
        name=f"{layer.name}_gather",
    )
    gathered = gather(x_flat)
    gather.set_weights([S])

    pooled = keras.layers.MaxPooling1D(pool_size=K, strides=K, name=f"{layer.name}_pool1d")(
        gathered
    )

    return keras.layers.Reshape((H_out, W_out, C), name=f"{layer.name}_reshape_out")(pooled)


def _conv3d_replacement(layer, x):
    """Conv3d -> per-depth-tap (Crop1D + EinsumDense) + Add + Reshape.

    Only depth_stride=1 is supported; raises NotImplementedError otherwise.
    Both depth_padding='valid' and 'same' are handled.

    For each depth tap d the input slice x[:, d:D_out+d, :, :, :] is extracted
    via Cropping1D (after optional ZeroPadding1D for 'same'), then a spatial
    EinsumDense with equation 'abmc,mcno->abno' (a=batch, b=D_out, m=N_in,
    c=Cin, n=N_out, o=Cout) applies the gather+MAC.  The D_kernel taps are
    finally summed via Add.
    """
    D_in = int(x.shape[1])
    H, W = int(x.shape[2]), int(x.shape[3])
    Cin = int(x.shape[4])
    N_in = H * W
    D_kernel = layer.depth_size
    depth_stride = layer.depth_stride
    Cout = layer.out_channels

    if depth_stride != 1:
        raise NotImplementedError(
            f"Conv3d hls4ml export only supports depth_stride=1, got {depth_stride}. "
            "Layers with depth_stride > 1 require a custom temporal-unfold handler."
        )

    # Handle depth_padding: pad the depth axis before cropping per tap.
    if layer.depth_padding == "same":
        pad_top = (D_kernel - 1) // 2
        pad_bot = D_kernel - 1 - pad_top
        D_eff = D_in  # output depth == input depth for 'same'
    else:
        pad_top = 0
        pad_bot = 0
        D_eff = D_in - D_kernel + 1  # valid conv output depth

    nbr, cells, _ = build_neighbor_table_3d(layer, D_in, H, W)
    W_k = get_cell_weights_3d(layer, cells)  # (D_kernel, K, Cin, Cout)
    K = len(cells)

    # Get H_out, W_out from a proxy
    proxy = hgly.MaxPool2d(kernel_size=layer.hexbase_size, stride=layer.hexbase_stride)
    _, _, (H_out, W_out) = build_neighbor_table(proxy, H, W)
    N_out = H_out * W_out

    # Reshape to (B, D_padded, N_in*Cin) for Cropping1D / ZeroPadding1D.
    D_padded = D_in + pad_top + pad_bot
    x_seq = keras.layers.Reshape((D_in, N_in * Cin), name=f"{layer.name}_reshape_seq")(x)
    if pad_top > 0 or pad_bot > 0:
        x_seq = keras.layers.ZeroPadding1D(padding=(pad_top, pad_bot), name=f"{layer.name}_zpad")(
            x_seq
        )

    tap_outputs = []
    for d in range(D_kernel):
        # Crop to extract the D_eff frames for this tap.
        crop_start = d
        crop_end = D_padded - D_eff - d  # = D_kernel - 1 - d
        if crop_start == 0 and crop_end == 0:
            x_d_seq = x_seq
        else:
            x_d_seq = keras.layers.Cropping1D(
                cropping=(crop_start, crop_end),
                name=f"{layer.name}_crop_d{d}",
            )(x_seq)  # (B, D_eff, N_in*Cin)

        x_d = keras.layers.Reshape((D_eff, N_in, Cin), name=f"{layer.name}_reshape_d{d}")(x_d_seq)

        # Build folded spatial kernel for this depth tap: (N_in, Cin, N_out, Cout)
        A_d = np.zeros((N_in, Cin, N_out, Cout), np.float32)
        for n in range(N_out):
            for k in range(K):
                m = int(nbr[n, k])
                if m >= 0:
                    A_d[m, :, n, :] += W_k[d, k]

        # Bias only on the last depth tap to avoid adding it D_kernel times.
        use_bias = layer.use_bias and (d == D_kernel - 1)
        einsum = keras.layers.EinsumDense(
            "abmc,mcno->abno",
            output_shape=(D_eff, N_out, Cout),
            bias_axes="o" if use_bias else None,
            name=f"{layer.name}_einsum_d{d}",
        )
        y_d = einsum(x_d)
        if use_bias:
            einsum.set_weights([A_d, layer.bias_tensor.numpy()])
        else:
            einsum.set_weights([A_d])

        tap_outputs.append(y_d)

    # Sum over depth taps.
    if D_kernel == 1:
        y_flat = tap_outputs[0]
    else:
        y_flat = keras.layers.Add(name=f"{layer.name}_add")(tap_outputs)

    return keras.layers.Reshape((D_eff, H_out, W_out, Cout), name=f"{layer.name}_reshape_out")(
        y_flat
    )


def _maxpool3d_replacement(layer, x):
    raise NotImplementedError(
        "MaxPool3d export is only implemented for strategy='gather'. "
        "Max is not linear, so the folded/slotwise EinsumDense decompositions do "
        "not apply. Use patch_model_for_hls(model, strategy='gather'), which maps "
        "MaxPool3d to HexPoolLineBuffer3D."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _binary_add(tensors, name_prefix):
    """Reduce a list of tensors to one value using a binary tree of Add layers.

    hls4ml's Add handler asserts exactly 2 inputs, so a flat multi-input Add
    is not allowed.  A left-fold binary tree produces only 2-input Add nodes
    and is equivalent for commutative addition.
    """
    assert len(tensors) > 0
    acc = tensors[0]
    for i, t in enumerate(tensors[1:], start=1):
        acc = keras.layers.Add(name=f"{name_prefix}_add{i}")([acc, t])
    return acc


# ---------------------------------------------------------------------------
# Slotwise builders (strategy="slotwise")
# ---------------------------------------------------------------------------


def _get_ring_idx(layer, cells):
    """Return (K,) int32 array mapping each cell slot to its weight-group index.

    Only meaningful when share_neighbors is truthy. Uses weight_maps_2d (the
    same map already used by get_cell_weights, keyed by layer.share_neighbors'
    mode) so the group assignment is guaranteed consistent with the weight
    layout regardless of which sharing mode ("ring"/"diag"/"sym") is active.

    Keyed off hexbase_size (the spatial kernel size), not kernel_size: for
    Conv2d the two are identical, but Conv3d's kernel_size may be a (depth,
    spatial) tuple, so hexbase_size is the correct spatial ring geometry.
    """
    from keras_hexagdly.layers import weight_maps_2d

    ring_maps, _ = weight_maps_2d(layer.hexbase_size, layer.share_neighbors)
    return np.array([int(ring_maps[i][r, c]) for i, r, c in cells], dtype=np.int32)


def _conv2d_replacement(layer, x):
    """Conv2d -> Reshape + EinsumDense + Reshape.

    The (N_in, Cin, N_out, Cout) kernel A encodes the neighbor gather folded
    into the MAC: A[m, c, n, o] += W_k[k, c, o] for each slot k where
    nbr[n, k] == m.  Invalid slots (-1) are skipped so they contribute 0.
    """
    H, W = int(x.shape[1]), int(x.shape[2])
    Cin = int(x.shape[3])
    N_in = H * W
    Cout = layer.out_channels

    nbr, cells, (H_out, W_out) = build_neighbor_table(layer, H, W)
    W_k = get_cell_weights(layer, cells)  # (K, Cin, Cout)
    N_out = H_out * W_out
    K = len(cells)

    A = np.zeros((N_in, Cin, N_out, Cout), np.float32)
    for n in range(N_out):
        for k in range(K):
            m = int(nbr[n, k])
            if m >= 0:
                A[m, :, n, :] += W_k[k]

    x_flat = keras.layers.Reshape((N_in, Cin), name=f"{layer.name}_reshape_in")(x)

    use_bias = layer.use_bias
    einsum = keras.layers.EinsumDense(
        "amc,mcno->ano",
        output_shape=(N_out, Cout),
        bias_axes="o" if use_bias else None,
        name=f"{layer.name}_einsum",
    )
    y_flat = einsum(x_flat)
    if use_bias:
        einsum.set_weights([A, layer.bias_tensor.numpy()])
    else:
        einsum.set_weights([A])

    return keras.layers.Reshape((H_out, W_out, Cout), name=f"{layer.name}_reshape_out")(y_flat)


def _maxpool2d_replacement(layer, x):
    """MaxPool2d -> Reshape + EinsumDense (gather) + MaxPooling1D(K,K) + Reshape.

    The (N_out*K, N_in) selection matrix S gathers the K neighbors of every
    output pixel into a contiguous N_out*K-length sequence.  MaxPooling1D with
    pool_size=K and strides=K then reduces each K-block to a single max value.
    """
    H, W = int(x.shape[1]), int(x.shape[2])
    C = int(x.shape[3])
    N_in = H * W

    nbr, cells, (H_out, W_out) = build_neighbor_table(layer, H, W)
    N_out = H_out * W_out
    K = len(cells)

    S = np.zeros((N_out * K, N_in), np.float32)
    for n in range(N_out):
        for k in range(K):
            m = int(nbr[n, k])
            if m >= 0:
                S[n * K + k, m] = 1.0

    x_flat = keras.layers.Reshape((N_in, C), name=f"{layer.name}_reshape_in")(x)

    gather = keras.layers.EinsumDense(
        "amc,pm->apc",
        output_shape=(N_out * K, C),
        bias_axes=None,
        name=f"{layer.name}_gather",
    )
    gathered = gather(x_flat)
    gather.set_weights([S])

    pooled = keras.layers.MaxPooling1D(pool_size=K, strides=K, name=f"{layer.name}_pool1d")(
        gathered
    )

    return keras.layers.Reshape((H_out, W_out, C), name=f"{layer.name}_reshape_out")(pooled)


def _conv3d_replacement(layer, x):
    """Conv3d -> per-depth-tap (Crop1D + EinsumDense) + Add + Reshape.

    Only depth_stride=1 is supported; raises NotImplementedError otherwise.
    Both depth_padding='valid' and 'same' are handled.

    For each depth tap d the input slice x[:, d:D_out+d, :, :, :] is extracted
    via Cropping1D (after optional ZeroPadding1D for 'same'), then a spatial
    EinsumDense with equation 'abmc,mcno->abno' (a=batch, b=D_out, m=N_in,
    c=Cin, n=N_out, o=Cout) applies the gather+MAC.  The D_kernel taps are
    finally summed via Add.
    """
    D_in = int(x.shape[1])
    H, W = int(x.shape[2]), int(x.shape[3])
    Cin = int(x.shape[4])
    N_in = H * W
    D_kernel = layer.depth_size
    depth_stride = layer.depth_stride
    Cout = layer.out_channels

    if depth_stride != 1:
        raise NotImplementedError(
            f"Conv3d hls4ml export only supports depth_stride=1, got {depth_stride}. "
            "Layers with depth_stride > 1 require a custom temporal-unfold handler."
        )

    # Handle depth_padding: pad the depth axis before cropping per tap.
    if layer.depth_padding == "same":
        pad_top = (D_kernel - 1) // 2
        pad_bot = D_kernel - 1 - pad_top
        D_eff = D_in  # output depth == input depth for 'same'
    else:
        pad_top = 0
        pad_bot = 0
        D_eff = D_in - D_kernel + 1  # valid conv output depth

    nbr, cells, _ = build_neighbor_table_3d(layer, D_in, H, W)
    W_k = get_cell_weights_3d(layer, cells)  # (D_kernel, K, Cin, Cout)
    K = len(cells)

    # Get H_out, W_out from a proxy
    proxy = hgly.MaxPool2d(kernel_size=layer.hexbase_size, stride=layer.hexbase_stride)
    _, _, (H_out, W_out) = build_neighbor_table(proxy, H, W)
    N_out = H_out * W_out

    # Reshape to (B, D_padded, N_in*Cin) for Cropping1D / ZeroPadding1D.
    D_padded = D_in + pad_top + pad_bot
    x_seq = keras.layers.Reshape((D_in, N_in * Cin), name=f"{layer.name}_reshape_seq")(x)
    if pad_top > 0 or pad_bot > 0:
        x_seq = keras.layers.ZeroPadding1D(padding=(pad_top, pad_bot), name=f"{layer.name}_zpad")(
            x_seq
        )

    tap_outputs = []
    for d in range(D_kernel):
        # Crop to extract the D_eff frames for this tap.
        crop_start = d
        crop_end = D_padded - D_eff - d  # = D_kernel - 1 - d
        if crop_start == 0 and crop_end == 0:
            x_d_seq = x_seq
        else:
            x_d_seq = keras.layers.Cropping1D(
                cropping=(crop_start, crop_end),
                name=f"{layer.name}_crop_d{d}",
            )(x_seq)  # (B, D_eff, N_in*Cin)

        x_d = keras.layers.Reshape((D_eff, N_in, Cin), name=f"{layer.name}_reshape_d{d}")(x_d_seq)

        # Build folded spatial kernel for this depth tap: (N_in, Cin, N_out, Cout)
        A_d = np.zeros((N_in, Cin, N_out, Cout), np.float32)
        for n in range(N_out):
            for k in range(K):
                m = int(nbr[n, k])
                if m >= 0:
                    A_d[m, :, n, :] += W_k[d, k]

        # Bias only on the last depth tap to avoid adding it D_kernel times.
        use_bias = layer.use_bias and (d == D_kernel - 1)
        einsum = keras.layers.EinsumDense(
            "abmc,mcno->abno",
            output_shape=(D_eff, N_out, Cout),
            bias_axes="o" if use_bias else None,
            name=f"{layer.name}_einsum_d{d}",
        )
        y_d = einsum(x_d)
        if use_bias:
            einsum.set_weights([A_d, layer.bias_tensor.numpy()])
        else:
            einsum.set_weights([A_d])

        tap_outputs.append(y_d)

    # Sum over depth taps.
    if D_kernel == 1:
        y_flat = tap_outputs[0]
    else:
        y_flat = keras.layers.Add(name=f"{layer.name}_add")(tap_outputs)

    return keras.layers.Reshape((D_eff, H_out, W_out, Cout), name=f"{layer.name}_reshape_out")(
        y_flat
    )


def _maxpool3d_replacement(layer, x):
    raise NotImplementedError(
        "MaxPool3d export is only implemented for strategy='gather'. "
        "Max is not linear, so the folded/slotwise EinsumDense decompositions do "
        "not apply. Use patch_model_for_hls(model, strategy='gather'), which maps "
        "MaxPool3d to HexPoolLineBuffer3D."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _conv2d_linebuffer(layer, x):
    """Conv2d -> single fused HexConvLineBuffer (linebuffer strategy).

    Same neighbor table + weights as _conv2d_gather, but gather and MAC are
    fused into ONE layer that keeps the (H, W) grid.  No Reshape / flatten and
    no intermediate gathered FIFO: in io_stream this lets the HLS kernel slide a
    line buffer over the image instead of materialising the whole frame + K×
    gathered tensor.  Any hex stride is supported: output pixel (oh, ow) maps to
    input base (stride*oh, stride*ow) and the per-parity tap offsets are constant
    (build_parity_offset_tables).
    """
    from keras_hexagdly.hex_gather import HexConvLineBuffer

    H, W = int(x.shape[1]), int(x.shape[2])
    nbr, cells, (H_out, W_out) = build_neighbor_table(layer, H, W)
    bias = layer.bias_tensor.numpy() if layer.use_bias else None

    if layer.share_neighbors:
        weights = layer.ring_weights.numpy()  # (num_rings, Cin, Cout)
        ring_idx = _get_ring_idx(layer, cells)  # (K,)
    else:
        weights = get_cell_weights(layer, cells)  # (K, Cin, Cout)
        ring_idx = None

    return HexConvLineBuffer(
        neighbor_idx=nbr,
        H=H,
        W=W,
        stride=layer.hexbase_stride,
        H_out=H_out,
        W_out=W_out,
        weights_array=weights,
        ring_idx=ring_idx,
        bias=bias,
        name=f"{layer.name}_linebuffer",
    )(x)


def _maxpool2d_linebuffer(layer, x):
    """MaxPool2d -> single fused HexPoolLineBuffer (linebuffer strategy).

    Line-buffer pooling: keeps the (H, W) grid and reduces
    the max over the K neighbor taps as the image streams, instead of
    materialising the gathered tensor.  Any hex stride is supported.
    """
    from keras_hexagdly.hex_gather import HexPoolLineBuffer

    H, W = int(x.shape[1]), int(x.shape[2])
    nbr, cells, (H_out, W_out) = build_neighbor_table(layer, H, W)

    return HexPoolLineBuffer(
        neighbor_idx=nbr,
        H=H,
        W=W,
        stride=layer.hexbase_stride,
        H_out=H_out,
        W_out=W_out,
        name=f"{layer.name}_linebuffer",
    )(x)


def _conv3d_linebuffer(layer, x):
    """Conv3d -> single fused HexConvLineBuffer3D (linebuffer strategy).

    Fuses the per-depth-tap gather+MAC+Add scaffolding into one
    layer that keeps the (D, H, W) grid.  Spatial neighbor table and per-tap
    weights are identical; the depth axis rides through as a dense conv handled
    inside the layer.  Only depth_stride=1 is supported.
    """
    from keras_hexagdly.hex_gather import HexConvLineBuffer3D

    if layer.hexbase_stride != 1:
        raise NotImplementedError(
            "linebuffer strategy supports hex stride=1 only "
            f"(got stride={layer.hexbase_stride}). Use strategy='gather'."
        )
    if layer.depth_stride != 1:
        raise NotImplementedError(
            f"Conv3d linebuffer export only supports depth_stride=1, got {layer.depth_stride}."
        )

    H, W = int(x.shape[2]), int(x.shape[3])
    nbr, cells, _ = build_neighbor_table_3d(layer, int(x.shape[1]), H, W)

    proxy = hgly.MaxPool2d(kernel_size=layer.hexbase_size, stride=layer.hexbase_stride)
    _, _, (H_out, W_out) = build_neighbor_table(proxy, H, W)

    bias = layer.bias_tensor.numpy() if layer.use_bias else None

    if layer.share_neighbors:
        weights = layer.ring_weights.numpy()  # (D_kernel, num_rings, Cin, Cout)
        ring_idx = _get_ring_idx(layer, cells)  # (K,)
    else:
        weights = get_cell_weights_3d(layer, cells)  # (D_kernel, K, Cin, Cout)
        ring_idx = None

    return HexConvLineBuffer3D(
        neighbor_idx=nbr,
        H=H_out,
        W=W_out,
        weights_array=weights,
        depth_size=layer.depth_size,
        depth_stride=layer.depth_stride,
        depth_padding=layer.depth_padding,
        ring_idx=ring_idx,
        bias=bias,
        name=f"{layer.name}_linebuffer",
    )(x)


def _maxpool3d_linebuffer(layer, x):
    """MaxPool3d -> single fused HexPoolLineBuffer3D (linebuffer strategy).

    Line-buffer analogue of _maxpool3d_gather: keeps the (D, H, W) grid and maxes
    over the depth pool window + K spatial taps as the volume streams.  Hex stride
    1 only (spatial output grid == input grid); depth_size/depth_stride arbitrary.
    """
    from keras_hexagdly.hex_gather import HexPoolLineBuffer3D

    if layer.hexbase_stride != 1:
        raise NotImplementedError(
            "MaxPool3d linebuffer supports hex stride=1 only "
            f"(got {layer.hexbase_stride}). Use strategy='gather'."
        )
    H, W = int(x.shape[2]), int(x.shape[3])
    nbr, cells, _ = build_neighbor_table_3d(layer, int(x.shape[1]), H, W)

    return HexPoolLineBuffer3D(
        neighbor_idx=nbr,
        H=H,
        W=W,
        depth_size=layer.depth_size,
        depth_stride=layer.depth_stride,
        name=f"{layer.name}_linebuffer",
    )(x)


_STRATEGIES = ("linebuffer", "folded")

# linebuffer: fused single-layer conv (line-buffer HLS kernel in io_stream),
# with matching line-buffer pooling. This is the production path.
_REPLACEMENTS_LINEBUFFER = {
    hgly.Conv2d: _conv2d_linebuffer,
    hgly.MaxPool2d: _maxpool2d_linebuffer,
    hgly.Conv3d: _conv3d_linebuffer,
    hgly.MaxPool3d: _maxpool3d_linebuffer,
}

# folded: the most direct transcription of the maths, kept as an INDEPENDENT
# cross-check for the test suite. Not a deployment path -- Vitis HLS clang
# segfaults at synthesis on grids larger than ~9x9.
_REPLACEMENTS_FOLDED = {
    hgly.Conv2d: _conv2d_replacement,
    hgly.MaxPool2d: _maxpool2d_replacement,
    hgly.Conv3d: _conv3d_replacement,
    hgly.MaxPool3d: _maxpool3d_replacement,
}


def patch_model_for_hls(model, strategy="linebuffer"):
    """Return a new Keras functional model with hex layers replaced by hls4ml-native ops.

    Walks the model's layer graph in topological order.  Every hex layer is
    replaced by its native equivalent (see module docstring); all other layers
    are re-applied unchanged so weights, activations, and topology are preserved.

    Args:
        model:    A built keras.Model whose hex layers have been trained.
        strategy: Export strategy for conv layers. One of:
                  - "linebuffer" (default): the production path. One fused node
                    per hex layer, keeping the (H, W, C) grid; in io_stream it
                    slides a register window over shift-register line buffers,
                    so only 2*kernel_size+1 image rows are ever resident. In
                    io_parallel it falls back to a frame-buffered gather+MAC of
                    equivalent cost. Covers Conv2d, Conv3d, MaxPool2d and
                    MaxPool3d.
                  - "folded": one EinsumDense with the full
                    (N_in,Cin,N_out,Cout) kernel. This is the most direct
                    transcription of the maths, kept as an INDEPENDENT
                    CROSS-CHECK for the tests -- it is not a deployment path:
                    Vitis HLS clang segfaults at synthesis on grids larger than
                    ~9x9.

                  The "slotwise" and "gather" strategies were removed: in
                  io_stream, gather materialised the whole frame AND the
                  N_out*K*Cin gathered tensor in registers (measured 34378 FF
                  against 3077 for a native conv of comparable neighbourhood),
                  which is exactly what "linebuffer" was written to avoid, and
                  in io_parallel the two collapse to the same kernel.

    Returns:
        A new keras.Model — plain Keras, no hls4ml dependency.  Pass this to
        ``hls4ml.converters.convert_from_keras_model()``.

    Raises:
        TypeError:           if ``model`` is not a keras.Model.
        ValueError:          if ``strategy`` is not a recognised value.
        NotImplementedError: for MaxPool3d layers, or Conv3d with depth_stride > 1.
    """
    if not isinstance(model, keras.Model):
        raise TypeError(f"Expected a keras.Model, got {type(model).__name__}.")
    if strategy not in _STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}. Choose from: {_STRATEGIES}.")

    replacements = (_REPLACEMENTS_LINEBUFFER if strategy == "linebuffer"
                    else _REPLACEMENTS_FOLDED)

    # Map from original tensor id -> replacement tensor.
    tensor_map = {id(t): t for t in model.inputs}

    for layer in model.layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue

        # Resolve this layer's input tensor(s) through the map.
        raw_in = layer.input
        if isinstance(raw_in, (list, tuple)):
            x = [tensor_map[id(t)] for t in raw_in]
        else:
            x = tensor_map[id(raw_in)]

        # Apply replacement or pass-through.
        fn = replacements.get(type(layer))
        if fn is not None:
            y = fn(layer, x)
        else:
            y = layer(x) if not isinstance(x, list) else layer(x)

        # Store replacement output tensor(s).
        raw_out = layer.output
        if isinstance(raw_out, (list, tuple)):
            if not isinstance(y, (list, tuple)):
                y = [y]
            for orig, new_t in zip(raw_out, y):
                tensor_map[id(orig)] = new_t
        else:
            tensor_map[id(raw_out)] = y

    new_outputs = [tensor_map[id(t)] for t in model.outputs]
    return keras.Model(model.inputs, new_outputs, name=model.name + "_hls")


# -----------------------------------------------------------------------------
# hls4ml configuration helpers
# -----------------------------------------------------------------------------

# Largest MAC_BLOCK (parallel weight reads per cycle) the stream kernels stay
# well-behaved at. Past it Vitis stops splitting the weight array into small
# distributed ROMs and starts duplicating it instead: measured on a 19x16x32
# layer, MAC_BLOCK 512 costs 0 BRAM while 1216 costs 6080 (295% of a Virtex-7).
# See the sweep table in nnet_hex_conv_linebuffer_stream.h.
_MAX_MAC_BLOCK = 512

def _hex_mac_shape(layer):
    """(num_weight_rows * n_chan, n_filt) for a patched hex layer, else None."""
    w = getattr(layer, "mac_weights", None)
    if w is None:
        return None
    shape = tuple(w.shape)
    if len(shape) == 3:              # (rows, Cin, Cout)
        return shape[0] * shape[1], shape[2]
    if len(shape) == 4:              # 3D: (depth, rows, Cin, Cout)
        return shape[0] * shape[1] * shape[2], shape[3]
    return None


def recommended_reuse_factor(layer, max_mac_block=_MAX_MAC_BLOCK):
    """Smallest ReuseFactor keeping this layer's MAC_BLOCK within budget.

    MAC_BLOCK = ceil(n_in / rf) * n_filt is how many multipliers the kernel
    instantiates, and also how many weights it reads per cycle. Returns 1 for
    layers that are already small enough.
    """
    shape = _hex_mac_shape(layer)
    if shape is None:
        return 1
    n_in, n_filt = shape
    if n_filt >= max_mac_block:      # a single output slice already exceeds it
        return n_in
    per_cycle = max(1, max_mac_block // n_filt)
    return max(1, -(-n_in // per_cycle))


def hex_reuse_config(config, model, max_mac_block=_MAX_MAC_BLOCK, verbose=True):
    """Give every hex layer in `model` an explicit per-layer ReuseFactor.

    Two problems this solves, both measured:

    1. `config_from_keras_model(granularity='name')` writes ReuseFactor=1 into
       the LayerName entry of every layer it RECOGNISES, and that per-layer
       value overrides Model.ReuseFactor. Hex layers are not recognised, so they
       get no entry and silently fall through to the model-level value instead --
       a hex-vs-standard comparison then runs the two at different reuse factors
       without saying so.

    2. The kernels have a narrow viable window. Too low a reuse factor and the
       multiplier count explodes; in the middle, the weight ROM is duplicated
       and BRAM explodes (measured 906% of a Virtex-7 at MAC_BLOCK=2432); too
       high and latency explodes. This picks the low end of the safe range.

    Mutates and returns `config`.
    """
    import warnings

    layer_cfg = config.setdefault("LayerName", {})
    for layer in model.layers:
        if _hex_mac_shape(layer) is None:
            continue
        rf = recommended_reuse_factor(layer, max_mac_block)
        entry = layer_cfg.setdefault(layer.name, {})
        entry["ReuseFactor"] = rf
        entry.setdefault("Strategy", "Resource")
        if verbose:
            n_in, n_filt = _hex_mac_shape(layer)
            warnings.warn(
                f"hex_reuse_config: {layer.name} -> ReuseFactor={rf} "
                f"(MAC_BLOCK={-(-n_in // rf) * n_filt}, n_in={n_in}, n_filt={n_filt})",
                stacklevel=2,
            )
    return config


def check_hls_config(config, model):
    """Warn about hls4ml settings known to mis-synthesise with these kernels.

    Returns the list of warning strings (also issued via `warnings.warn`), so a
    caller can assert on it in tests.
    """
    import warnings

    issues = []
    model_cfg = config.get("Model", {})
    layer_cfg = config.get("LayerName", {})

    # Known RTL-correctness failure, see nnet_hex_conv_linebuffer_stream.h:
    # Strategy='Latency' + ReuseFactor=1 + a repack_stream-triggering layer
    # downstream (Flatten/Reshape) gives a constant, bias-only output in real
    # cosim while C-simulation stays correct.
    reshapes = any(isinstance(l, (keras.layers.Flatten, keras.layers.Reshape))
                   for l in model.layers)
    for layer in model.layers:
        if _hex_mac_shape(layer) is None:
            continue
        entry = layer_cfg.get(layer.name, {})
        strat = str(entry.get("Strategy", model_cfg.get("Strategy", "Latency"))).lower()
        rf = int(entry.get("ReuseFactor", model_cfg.get("ReuseFactor", 1)))
        if reshapes and strat == "latency" and rf == 1:
            issues.append(
                f"{layer.name}: Strategy='Latency' with ReuseFactor=1 and a "
                "Flatten/Reshape downstream is known to produce WRONG RTL "
                "(bias-only output) while C-simulation passes. Use "
                "Strategy='Resource' with ReuseFactor>=2."
            )
        shape = _hex_mac_shape(layer)
        mac_block = -(-shape[0] // max(rf, 1)) * shape[1]
        if mac_block > _MAX_MAC_BLOCK:
            issues.append(
                f"{layer.name}: ReuseFactor={rf} gives MAC_BLOCK={mac_block} "
                f"(> {_MAX_MAC_BLOCK}); the weight ROM is likely to be duplicated "
                f"into BRAM. Use ReuseFactor>={recommended_reuse_factor(layer)}."
            )
    for msg in issues:
        warnings.warn("check_hls_config: " + msg, stacklevel=2)
    return issues
