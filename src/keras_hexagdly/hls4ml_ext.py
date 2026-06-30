"""hls4ml export bridge for keras-hexagdly layers.

Usage
-----
    from keras_hexagdly.hls4ml_ext import patch_model_for_hls
    import hls4ml

    hls_model_input = patch_model_for_hls(trained_model)
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

Replacement mapping
-------------------
  Conv2d    ->  Reshape + EinsumDense('amc,mcno->ano') + Reshape
  MaxPool2d ->  Reshape + EinsumDense('amc,pm->apc') + MaxPooling1D(K,K) + Reshape
  Conv3d    ->  per-depth-tap (ZeroPad1D? + Crop1D + Reshape + EinsumDense('abmc,mcno->abno'))
                + Add over taps + Reshape  [depth_stride=1 only]
  MaxPool3d ->  NotImplementedError (temporal+spatial max requires Transpose not in hls4ml)

The original model is NOT modified; a new Keras model is returned.
hls4ml is NOT imported here; the patched model is plain Keras.

Flat-index convention
---------------------
Spatial flat index m = h * W + w (raster scan).  The (N_in, Cin, N_out, Cout)
EinsumDense kernel encodes the neighbor gather: A[m,:,n,:] accumulates the
weights of all kernel slots k where nbr[n,k] == m.

Border behavior
---------------
Invalid neighbor slots (-1 in the table) contribute 0.0, matching hexagdly's
zero-padding at the grid border.  For max pooling, a 0-pad slot can dominate
when all real neighbors are negative — this is identical to hexagdly's behavior.
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
# Model surgery: walk the functional DAG and swap hex layers
# ---------------------------------------------------------------------------

_HEX_LAYER_TYPES = (hgly.Conv2d, hgly.MaxPool2d, hgly.Conv3d, hgly.MaxPool3d)

_REPLACEMENTS = {
    hgly.Conv2d:    _conv2d_replacement,
    hgly.MaxPool2d: _maxpool2d_replacement,
    hgly.Conv3d:    _conv3d_replacement,
    hgly.MaxPool3d: _maxpool3d_replacement,
}


def patch_model_for_hls(model):
    """Return a new Keras functional model with hex layers replaced by hls4ml-native ops.

    Walks the model's layer graph in topological order.  Every hex layer is
    replaced by its native equivalent (see module docstring); all other layers
    are re-applied unchanged so weights, activations, and topology are preserved.

    Args:
        model:  A built keras.Model whose hex layers have been trained.

    Returns:
        A new keras.Model — plain Keras, no hls4ml dependency.  Pass this to
        ``hls4ml.converters.convert_from_keras_model()``.

    Raises:
        TypeError:          if ``model`` is not a keras.Model.
        NotImplementedError: if the model contains a MaxPool3d layer.
        ValueError:          if depth_stride > 1 is used in a Conv3d layer.
    """
    if not isinstance(model, keras.Model):
        raise TypeError(f"Expected a keras.Model, got {type(model).__name__}.")

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
        fn = _REPLACEMENTS.get(type(layer))
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
