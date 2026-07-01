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

"gather" (not yet implemented):
    A proper sparse gather layer backed by a custom hls4ml KerasV3LayerHandler
    + HLS C++ template.  Scales to full camera size (DigiCam N=1296) without
    a large static array.  Raises NotImplementedError until implemented.

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

import numpy as np
import keras

import keras_hexagdly as hgly
from keras_hexagdly.indexed import (
    build_neighbor_table,
    get_cell_weights,
    build_neighbor_table_3d,
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
    Cin  = int(x.shape[3])
    N_in = H * W
    Cout = layer.out_channels

    nbr, cells, (H_out, W_out) = build_neighbor_table(layer, H, W)
    W_k = get_cell_weights(layer, cells)          # (K, Cin, Cout)
    N_out = H_out * W_out
    K = len(cells)

    A = np.zeros((N_in, Cin, N_out, Cout), np.float32)
    for n in range(N_out):
        for k in range(K):
            m = int(nbr[n, k])
            if m >= 0:
                A[m, :, n, :] += W_k[k]

    x_flat = keras.layers.Reshape(
        (N_in, Cin), name=f"{layer.name}_reshape_in"
    )(x)

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

    return keras.layers.Reshape(
        (H_out, W_out, Cout), name=f"{layer.name}_reshape_out"
    )(y_flat)


def _maxpool2d_replacement(layer, x):
    """MaxPool2d -> Reshape + EinsumDense (gather) + MaxPooling1D(K,K) + Reshape.

    The (N_out*K, N_in) selection matrix S gathers the K neighbors of every
    output pixel into a contiguous N_out*K-length sequence.  MaxPooling1D with
    pool_size=K and strides=K then reduces each K-block to a single max value.
    """
    H, W = int(x.shape[1]), int(x.shape[2])
    C    = int(x.shape[3])
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

    x_flat = keras.layers.Reshape(
        (N_in, C), name=f"{layer.name}_reshape_in"
    )(x)

    gather = keras.layers.EinsumDense(
        "amc,pm->apc",
        output_shape=(N_out * K, C),
        bias_axes=None,
        name=f"{layer.name}_gather",
    )
    gathered = gather(x_flat)
    gather.set_weights([S])

    pooled = keras.layers.MaxPooling1D(
        pool_size=K, strides=K, name=f"{layer.name}_pool1d"
    )(gathered)

    return keras.layers.Reshape(
        (H_out, W_out, C), name=f"{layer.name}_reshape_out"
    )(pooled)


def _maxpool2d_gather(layer, x):
    """MaxPool2d with strategy='gather' -> Reshape + HexGather + HexMaxPool + Reshape.

    HexGather replaces the large EinsumDense selection matrix with a tiny
    (N_out, K) integer index table.  HexMaxPool takes the max over K slots.
    Border slots were set to 0 by HexGather — matches hexagdly zero-padding.
    """
    from keras_hexagdly.hex_gather import HexGather, HexMaxPool

    H, W = int(x.shape[1]), int(x.shape[2])
    C    = int(x.shape[3])
    N_in = H * W

    nbr, cells, (H_out, W_out) = build_neighbor_table(layer, H, W)
    N_out = H_out * W_out

    x_flat = keras.layers.Reshape(
        (N_in, C), name=f"{layer.name}_reshape_in"
    )(x)

    gathered = HexGather(
        neighbor_idx=nbr, name=f"{layer.name}_gather"
    )(x_flat)                                           # (B, N_out, K, C)

    pooled = HexMaxPool(name=f"{layer.name}_maxpool")(gathered)  # (B, N_out, C)

    return keras.layers.Reshape(
        (H_out, W_out, C), name=f"{layer.name}_reshape_out"
    )(pooled)


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
    Cin  = int(x.shape[4])
    N_in = H * W
    D_kernel    = layer.depth_size
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
    W_k = get_cell_weights_3d(layer, cells)          # (D_kernel, K, Cin, Cout)
    N_out_spatial = int(nbr.shape[0])                # H_out * W_out (spatial)
    K = len(cells)

    # Get H_out, W_out from a proxy
    proxy = hgly.MaxPool2d(kernel_size=layer.hexbase_size, stride=layer.hexbase_stride)
    _, _, (H_out, W_out) = build_neighbor_table(proxy, H, W)
    N_out = H_out * W_out

    # Reshape to (B, D_padded, N_in*Cin) for Cropping1D / ZeroPadding1D.
    D_padded = D_in + pad_top + pad_bot
    x_seq = keras.layers.Reshape(
        (D_in, N_in * Cin), name=f"{layer.name}_reshape_seq"
    )(x)
    if pad_top > 0 or pad_bot > 0:
        x_seq = keras.layers.ZeroPadding1D(
            padding=(pad_top, pad_bot), name=f"{layer.name}_zpad"
        )(x_seq)

    tap_outputs = []
    for d in range(D_kernel):
        # Crop to extract the D_eff frames for this tap.
        crop_start = d
        crop_end = D_padded - D_eff - d   # = D_kernel - 1 - d
        if crop_start == 0 and crop_end == 0:
            x_d_seq = x_seq
        else:
            x_d_seq = keras.layers.Cropping1D(
                cropping=(crop_start, crop_end),
                name=f"{layer.name}_crop_d{d}",
            )(x_seq)                                   # (B, D_eff, N_in*Cin)

        x_d = keras.layers.Reshape(
            (D_eff, N_in, Cin), name=f"{layer.name}_reshape_d{d}"
        )(x_d_seq)

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

    return keras.layers.Reshape(
        (D_eff, H_out, W_out, Cout), name=f"{layer.name}_reshape_out"
    )(y_flat)


def _maxpool3d_replacement(layer, x):
    raise NotImplementedError(
        "MaxPool3d hls4ml export is not yet implemented. "
        "The temporal+spatial max requires an (N_out, D_kernel, K) -> (N_out, D_kernel*K) "
        "reordering that needs a Transpose not available in stock hls4ml ops. "
        "Workaround: replace MaxPool3d with MaxPool2d on each depth slice manually, "
        "or use a custom KerasV3LayerHandler + HLS template."
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


def _conv2d_slotwise(layer, x):
    """Conv2d -> K × (gather EinsumDense + MAC EinsumDense) + Add + Reshape.

    For each neighbor slot k:
      1. A (N_out, N_in) binary selection matrix S_k gathers input pixel
         nbr[n,k] into output position n (0 for invalid slots).
         EinsumDense equation: 'amc,nm->anc'  (batch a, N_in m, N_out n, Cin c)
      2. A (Cin, Cout) weight matrix W_k applies the learned kernel.
         EinsumDense equation: 'anc,co->ano'
    Outputs are summed over all K slots via Add.

    Largest single static array: (N_out, N_in) per slot — ~160k at 20×20,
    well within Vitis HLS elaboration limits.
    """
    H, W = int(x.shape[1]), int(x.shape[2])
    Cin  = int(x.shape[3])
    N_in = H * W
    Cout = layer.out_channels

    nbr, cells, (H_out, W_out) = build_neighbor_table(layer, H, W)
    W_k = get_cell_weights(layer, cells)          # (K, Cin, Cout)
    N_out = H_out * W_out
    K = len(cells)

    x_flat = keras.layers.Reshape(
        (N_in, Cin), name=f"{layer.name}_reshape_in"
    )(x)

    slot_outputs = []
    for k in range(K):
        # Binary selection matrix: S_k[n, m] = 1 iff nbr[n, k] == m.
        S_k = np.zeros((N_out, N_in), np.float32)
        for n in range(N_out):
            m = int(nbr[n, k])
            if m >= 0:
                S_k[n, m] = 1.0

        # Skip zero-weight slots (e.g. all-zero kernel cell).
        if not np.any(S_k) and not np.any(W_k[k]):
            continue

        # Gather: (B, N_in, Cin) -> (B, N_out, Cin)
        gather = keras.layers.EinsumDense(
            "amc,nm->anc",
            output_shape=(N_out, Cin),
            bias_axes=None,
            name=f"{layer.name}_gather_k{k}",
        )
        gathered = gather(x_flat)
        gather.set_weights([S_k])

        # MAC: (B, N_out, Cin) -> (B, N_out, Cout)
        mac = keras.layers.EinsumDense(
            "anc,co->ano",
            output_shape=(N_out, Cout),
            bias_axes=None,
            name=f"{layer.name}_mac_k{k}",
        )
        y_k = mac(gathered)
        mac.set_weights([W_k[k]])

        slot_outputs.append(y_k)

    if len(slot_outputs) == 0:
        # Degenerate: all-zero kernel — return zeros.
        y_flat = keras.layers.Lambda(
            lambda t: t * 0.0, name=f"{layer.name}_zero"
        )(keras.layers.EinsumDense(
            "amc,co->ao", output_shape=(Cout,), bias_axes=None,
            name=f"{layer.name}_zero_proj",
        )(x_flat))
    else:
        y_flat = _binary_add(slot_outputs, name_prefix=layer.name)

    if layer.use_bias:
        y_flat = keras.layers.Add(name=f"{layer.name}_bias")([
            y_flat,
            keras.layers.Lambda(
                lambda t, b=layer.bias_tensor.numpy(): t * 0.0 + b,
                name=f"{layer.name}_bias_const",
            )(y_flat),
        ])

    return keras.layers.Reshape(
        (H_out, W_out, Cout), name=f"{layer.name}_reshape_out"
    )(y_flat)


def _conv3d_slotwise(layer, x):
    """Conv3d -> per-depth-tap slotwise replacement + Add + Reshape.

    Combines the per-depth-tap structure of _conv3d_replacement with the
    per-slot decomposition of _conv2d_slotwise.  Only depth_stride=1 supported.
    """
    D_in = int(x.shape[1])
    H, W = int(x.shape[2]), int(x.shape[3])
    Cin  = int(x.shape[4])
    N_in = H * W
    D_kernel    = layer.depth_size
    depth_stride = layer.depth_stride
    Cout = layer.out_channels

    if depth_stride != 1:
        raise NotImplementedError(
            f"Conv3d hls4ml export only supports depth_stride=1, got {depth_stride}."
        )

    if layer.depth_padding == "same":
        pad_top = (D_kernel - 1) // 2
        pad_bot = D_kernel - 1 - pad_top
        D_eff = D_in
    else:
        pad_top = 0
        pad_bot = 0
        D_eff = D_in - D_kernel + 1

    nbr, cells, _ = build_neighbor_table_3d(layer, D_in, H, W)
    W_k = get_cell_weights_3d(layer, cells)          # (D_kernel, K, Cin, Cout)
    K = len(cells)

    proxy = hgly.MaxPool2d(kernel_size=layer.hexbase_size, stride=layer.hexbase_stride)
    _, _, (H_out, W_out) = build_neighbor_table(proxy, H, W)
    N_out = H_out * W_out

    D_padded = D_in + pad_top + pad_bot
    x_seq = keras.layers.Reshape(
        (D_in, N_in * Cin), name=f"{layer.name}_reshape_seq"
    )(x)
    if pad_top > 0 or pad_bot > 0:
        x_seq = keras.layers.ZeroPadding1D(
            padding=(pad_top, pad_bot), name=f"{layer.name}_zpad"
        )(x_seq)

    tap_slot_outputs = []
    for d in range(D_kernel):
        crop_start = d
        crop_end   = D_padded - D_eff - d
        if crop_start == 0 and crop_end == 0:
            x_d_seq = x_seq
        else:
            x_d_seq = keras.layers.Cropping1D(
                cropping=(crop_start, crop_end),
                name=f"{layer.name}_crop_d{d}",
            )(x_seq)

        x_d = keras.layers.Reshape(
            (D_eff, N_in, Cin), name=f"{layer.name}_reshape_d{d}"
        )(x_d_seq)

        for k in range(K):
            S_k = np.zeros((N_out, N_in), np.float32)
            for n in range(N_out):
                m = int(nbr[n, k])
                if m >= 0:
                    S_k[n, m] = 1.0

            if not np.any(S_k) and not np.any(W_k[d, k]):
                continue

            gather = keras.layers.EinsumDense(
                "abmc,nm->abnc",
                output_shape=(D_eff, N_out, Cin),
                bias_axes=None,
                name=f"{layer.name}_gather_d{d}_k{k}",
            )
            gathered = gather(x_d)
            gather.set_weights([S_k])

            mac = keras.layers.EinsumDense(
                "abnc,co->abno",
                output_shape=(D_eff, N_out, Cout),
                bias_axes=None,
                name=f"{layer.name}_mac_d{d}_k{k}",
            )
            y_dk = mac(gathered)
            mac.set_weights([W_k[d, k]])
            tap_slot_outputs.append(y_dk)

    y_flat = _binary_add(tap_slot_outputs, name_prefix=layer.name)

    if layer.use_bias:
        y_flat = keras.layers.Add(name=f"{layer.name}_bias")([
            y_flat,
            keras.layers.Lambda(
                lambda t, b=layer.bias_tensor.numpy(): t * 0.0 + b,
                name=f"{layer.name}_bias_const",
            )(y_flat),
        ])

    return keras.layers.Reshape(
        (D_eff, H_out, W_out, Cout), name=f"{layer.name}_reshape_out"
    )(y_flat)


# ---------------------------------------------------------------------------
# Gather builders (strategy="gather")
# ---------------------------------------------------------------------------


def _get_ring_idx(layer, cells):
    """Return (K,) int32 array mapping each cell slot to its hex ring index.

    Only meaningful when share_neighbors=True.  Uses ring_maps_2d (the same
    empirical ring map already used by get_cell_weights) so the ring assignment
    is guaranteed consistent with the weight layout.
    """
    from keras_hexagdly.layers import ring_maps_2d
    ring_maps, _ = ring_maps_2d(layer.kernel_size)
    return np.array(
        [int(ring_maps[i][r, c]) for i, r, c in cells], dtype=np.int32
    )


def _conv2d_gather(layer, x):
    """Conv2d -> HexGather + HexRingMAC + Reshape.

    One HexGather replaces all K EinsumDense selection matrices.
    One HexRingMAC replaces all K MAC EinsumDense layers, exploiting
    share_neighbors to store only (num_rings, Cin, Cout) instead of (K, Cin, Cout).

    Graph:
        Reshape(N_in, Cin)
        → HexGather(N_out, K)              # (B, N_out, K, Cin)
        → HexRingMAC(weights, ring_idx)    # (B, N_out, Cout)
        → Reshape(H_out, W_out, Cout)
    """
    from keras_hexagdly.hex_gather import HexGather, HexRingMAC

    H, W = int(x.shape[1]), int(x.shape[2])
    Cin  = int(x.shape[3])
    N_in = H * W
    Cout = layer.out_channels

    nbr, cells, (H_out, W_out) = build_neighbor_table(layer, H, W)
    N_out = H_out * W_out

    x_flat = keras.layers.Reshape(
        (N_in, Cin), name=f"{layer.name}_reshape_in"
    )(x)

    # HexGather: index table (N_out, K) — tiny integer ROM
    gathered = HexGather(
        neighbor_idx=nbr, name=f"{layer.name}_gather"
    )(x_flat)                                           # (B, N_out, K, Cin)

    # HexRingMAC: weights + optional ring_idx
    if layer.share_neighbors:
        W_rings = layer.ring_weights.numpy()            # (num_rings, Cin, Cout)
        ring_idx = _get_ring_idx(layer, cells)          # (K,)
        y_flat = HexRingMAC(
            weights_array=W_rings,
            ring_idx=ring_idx,
            name=f"{layer.name}_mac",
        )(gathered)
    else:
        from keras_hexagdly.indexed import get_cell_weights
        W_k = get_cell_weights(layer, cells)            # (K, Cin, Cout)
        y_flat = HexRingMAC(
            weights_array=W_k,
            ring_idx=None,
            name=f"{layer.name}_mac",
        )(gathered)

    if layer.use_bias:
        y_flat = keras.layers.Add(name=f"{layer.name}_bias")([
            y_flat,
            keras.layers.Lambda(
                lambda t, b=layer.bias_tensor.numpy(): t * 0.0 + b,
                name=f"{layer.name}_bias_const",
            )(y_flat),
        ])

    return keras.layers.Reshape(
        (H_out, W_out, Cout), name=f"{layer.name}_reshape_out"
    )(y_flat)


def _conv3d_gather(layer, x):
    """Conv3d with strategy='gather' falls back to slotwise.

    HexGather operates on (B, N_in, C) — adding a depth axis requires
    reshaping the batch and depth dims together, which is awkward in a
    functional Keras graph and adds no benefit over slotwise for the temporal
    axis (depth taps are already dense, not sparse).  The gather savings apply
    to the spatial axis only, and the slotwise path already handles the spatial
    gather correctly.  A dedicated 3D gather is a future extension.
    """
    return _conv3d_slotwise(layer, x)


# ---------------------------------------------------------------------------
# Model surgery: walk the functional DAG and swap hex layers
# ---------------------------------------------------------------------------

_STRATEGIES = ("folded", "slotwise", "gather")

_REPLACEMENTS_FOLDED = {
    hgly.Conv2d:    _conv2d_replacement,
    hgly.MaxPool2d: _maxpool2d_replacement,
    hgly.Conv3d:    _conv3d_replacement,
    hgly.MaxPool3d: _maxpool3d_replacement,
}

_REPLACEMENTS_SLOTWISE = {
    hgly.Conv2d:    _conv2d_slotwise,
    hgly.MaxPool2d: _maxpool2d_replacement,   # pool is strategy-agnostic
    hgly.Conv3d:    _conv3d_slotwise,
    hgly.MaxPool3d: _maxpool3d_replacement,
}

_REPLACEMENTS_GATHER = {
    hgly.Conv2d:    _conv2d_gather,
    hgly.MaxPool2d: _maxpool2d_gather,        # HexGather + HexMaxPool (no large matrix)
    hgly.Conv3d:    _conv3d_gather,           # falls back to slotwise (see docstring)
    hgly.MaxPool3d: _maxpool3d_replacement,
}


def patch_model_for_hls(model, strategy="slotwise"):
    """Return a new Keras functional model with hex layers replaced by hls4ml-native ops.

    Walks the model's layer graph in topological order.  Every hex layer is
    replaced by its native equivalent (see module docstring); all other layers
    are re-applied unchanged so weights, activations, and topology are preserved.

    Args:
        model:    A built keras.Model whose hex layers have been trained.
        strategy: Export strategy for conv layers. One of:
                  - "slotwise" (default): K separate gather+MAC pairs per layer.
                    Largest static array is (N_out, N_in) per slot — safe for
                    Vitis HLS synthesis at typical camera sizes.
                  - "folded": one EinsumDense with the full (N_in,Cin,N_out,Cout)
                    kernel. Simpler graph; fine for C-simulation but causes Vitis
                    HLS clang to segfault at synthesis on grids larger than ~9×9.
                  - "gather": HexGather + HexRingMAC layers.  Conv2d uses a
                    sparse (N_out, K) integer index table instead of dense
                    selection matrices — the key improvement for synthesis at
                    full camera scale.  Requires Phases 3-4 (custom hls4ml
                    handler + HLS C++ kernel) to synthesize; C-simulation
                    with stock hls4ml is not yet supported.  Conv3d falls back
                    to slotwise (spatial gather savings don't apply to the
                    temporal axis).

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
        raise ValueError(
            f"Unknown strategy {strategy!r}. Choose from: {_STRATEGIES}."
        )

    if strategy == "gather":
        replacements = _REPLACEMENTS_GATHER
    elif strategy == "slotwise":
        replacements = _REPLACEMENTS_SLOTWISE
    else:
        replacements = _REPLACEMENTS_FOLDED

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
