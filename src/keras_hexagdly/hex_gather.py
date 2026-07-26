"""HexGather and HexRingMAC — sparse gather layers for hls4ml export.

These layers implement the indexed gather+MAC path for the "gather" export
strategy in hls4ml_ext.py.  They are NOT used during normal training or PC
inference — the zig-zag hexagdly layers are used instead.  They exist solely
to give hls4ml a synthesizable representation.

HexGather
---------
Takes a flat pixel tensor (B, N_in, C) and gathers each pixel's K neighbors
into (B, N_out, K, C) using a precomputed integer index table.  Border slots
(neighbor_idx == -1) produce zeros, matching hexagdly's zero-padding behavior.

HexRingMAC
----------
Takes the gathered tensor (B, N_out, K, C_in) and applies the learned MAC.
Two modes depending on share_neighbors:

  share_neighbors=False: weights are (K, Cin, Cout) — one weight set per slot.
  share_neighbors=True:  weights are (num_rings, Cin, Cout) — one weight set
                         per hexagonal ring; a (K,) ring_idx table maps each
                         slot to its ring.  On the FPGA this means far fewer
                         ROM entries (2 rings instead of 7 slots for kernel=1).

Output: (B, N_out, Cout).
"""

import keras
import numpy as np


@keras.saving.register_keras_serializable(package="hexagdly_tf")
class HexGather(keras.layers.Layer):
    """Sparse hex neighbor gather.

    Input:  (B, N_in, C)
    Output: (B, N_out, K, C)

    neighbor_idx: int32 ndarray (N_out, K), -1 = border/invalid → zero output.
    Stored as a non-trainable weight so it serializes with the model.
    """

    def __init__(self, neighbor_idx, **kwargs):
        super().__init__(**kwargs)
        self._neighbor_idx_init = np.asarray(neighbor_idx, dtype=np.int32)
        self.N_out, self.K = self._neighbor_idx_init.shape

    def build(self, input_shape):
        self.neighbor_idx = self.add_weight(
            name="neighbor_idx",
            shape=self._neighbor_idx_init.shape,
            dtype="int32",
            initializer=keras.initializers.Constant(self._neighbor_idx_init),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x):
        # x: (B, N_in, C)
        # neighbor_idx: (N_out, K)
        valid = self.neighbor_idx >= 0  # (N_out, K)
        safe_idx = keras.ops.where(valid, self.neighbor_idx, 0)  # clamp -1 → 0

        # gather: x[(B), safe_idx[n,k], (C)] → (B, N_out, K, C)
        gathered = keras.ops.take(x, safe_idx, axis=1)

        # zero out border slots — expand (N_out, K) → (1, N_out, K, 1)
        mask = keras.ops.cast(keras.ops.reshape(valid, (1, self.N_out, self.K, 1)), gathered.dtype)
        return gathered * mask

    def get_config(self):
        config = super().get_config()
        config["neighbor_idx"] = self._neighbor_idx_init.tolist()
        return config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config["neighbor_idx"] = np.asarray(config["neighbor_idx"], dtype=np.int32)
        return cls(**config)


@keras.saving.register_keras_serializable(package="hexagdly_tf")
class HexRingMAC(keras.layers.Layer):
    """Hex neighbor MAC with optional ring weight sharing.

    Input:  (B, N_out, K, Cin)   — output of HexGather
    Output: (B, N_out, Cout)

    share_neighbors=False: weights (K, Cin, Cout), ring_idx=None.
    share_neighbors=True:  weights (num_rings, Cin, Cout), ring_idx (K,) int.
      On the FPGA this stores only num_rings unique weight vectors instead of K,
      mapping each slot to its ring via a tiny ROM (ring_idx).

    ``bias`` (Cout,) is added to the output; it lives inside this layer (seeded
    into the accumulator on the FPGA) so the export graph needs no extra Add /
    Lambda layer — hls4ml has no handler for those. Pass None for no bias.
    """

    def __init__(self, weights_array, ring_idx=None, bias=None, **kwargs):
        super().__init__(**kwargs)
        self._weights_init = np.asarray(weights_array, dtype=np.float32)
        self._ring_idx_init = np.asarray(ring_idx, dtype=np.int32) if ring_idx is not None else None
        self.share_neighbors = ring_idx is not None
        Cout = self._weights_init.shape[-1]
        if bias is None:
            self._bias_init = np.zeros((Cout,), dtype=np.float32)
        else:
            self._bias_init = np.asarray(bias, dtype=np.float32).reshape(Cout)

    def build(self, input_shape):
        self.mac_weights = self.add_weight(
            name="mac_weights",
            shape=self._weights_init.shape,
            dtype="float32",
            initializer=keras.initializers.Constant(self._weights_init),
            trainable=False,
        )
        if self.share_neighbors:
            self.ring_idx = self.add_weight(
                name="ring_idx",
                shape=self._ring_idx_init.shape,
                dtype="int32",
                initializer=keras.initializers.Constant(self._ring_idx_init),
                trainable=False,
            )
        self.mac_bias = self.add_weight(
            name="mac_bias",
            shape=self._bias_init.shape,
            dtype="float32",
            initializer=keras.initializers.Constant(self._bias_init),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x):
        # x: (B, N_out, K, Cin)
        if self.share_neighbors:
            # expand ring weights to (K, Cin, Cout) by gathering along ring axis
            W = keras.ops.take(self.mac_weights, self.ring_idx, axis=0)  # (K, Cin, Cout)
        else:
            W = self.mac_weights  # (K, Cin, Cout)

        # sum over K slots and Cin channels
        # x: (B, N_out, K, Cin), W: (K, Cin, Cout) → (B, N_out, Cout)
        y = keras.ops.einsum("bnkc,kco->bno", x, W)
        return y + keras.ops.reshape(self.mac_bias, (1, 1, -1))

    def get_config(self):
        config = super().get_config()
        config["weights_array"] = self._weights_init.tolist()
        config["ring_idx"] = (
            self._ring_idx_init.tolist() if self._ring_idx_init is not None else None
        )
        config["bias"] = self._bias_init.tolist()
        return config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config["weights_array"] = np.asarray(config["weights_array"], dtype=np.float32)
        if config["ring_idx"] is not None:
            config["ring_idx"] = np.asarray(config["ring_idx"], dtype=np.int32)
        if config.get("bias") is not None:
            config["bias"] = np.asarray(config["bias"], dtype=np.float32)
        return cls(**config)


@keras.saving.register_keras_serializable(package="hexagdly_tf")
class HexMaxPool(keras.layers.Layer):
    """Hex neighbor max pooling.

    Input:  (B, N_out, K, C)  — output of HexGather
    Output: (B, N_out, C)

    Takes the element-wise max over the K neighbor slots.  Border slots were
    set to 0.0 by HexGather, so the max is always >= 0 for non-negative
    inputs — this matches hexagdly's zero-padding border behavior for MaxPool.
    """

    def call(self, x):
        # x: (B, N_out, K, C) → (B, N_out, C)
        return keras.ops.max(x, axis=2)

    def get_config(self):
        return super().get_config()

    def compute_output_shape(self, input_shape):
        # (B, N_out, K, C) → (B, N_out, C)
        return (input_shape[0], input_shape[1], input_shape[3])


# ---------------------------------------------------------------------------
# 3D (depth-aware) variants — used by the Conv3d/MaxPool3d gather export.
#
# The depth axis is carried as a passthrough leading dimension: the spatial
# gather / MAC / max is applied independently to every one of the D frames.
# This keeps the graph free of any batch<->depth reshape (impossible in Keras)
# and free of high-rank Transpose (fragile in hls4ml); the matching HLS kernels
# (nnet_hex_gather_3d.h / nnet_hex_ring_mac_3d.h) are just an outer depth loop
# around the 2D body.
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="hexagdly_tf")
class HexGather3D(keras.layers.Layer):
    """Depth-aware sparse hex neighbor gather.

    Input:  (B, D, N_in, C)
    Output: (B, D, N_out, K, C)

    Identical spatial gather to HexGather, applied to every depth frame.
    neighbor_idx: int32 ndarray (N_out, K), -1 = border/invalid → zero output.
    """

    def __init__(self, neighbor_idx, **kwargs):
        super().__init__(**kwargs)
        self._neighbor_idx_init = np.asarray(neighbor_idx, dtype=np.int32)
        self.N_out, self.K = self._neighbor_idx_init.shape

    def build(self, input_shape):
        self.neighbor_idx = self.add_weight(
            name="neighbor_idx",
            shape=self._neighbor_idx_init.shape,
            dtype="int32",
            initializer=keras.initializers.Constant(self._neighbor_idx_init),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x):
        # x: (B, D, N_in, C)
        valid = self.neighbor_idx >= 0  # (N_out, K)
        safe_idx = keras.ops.where(valid, self.neighbor_idx, 0)  # clamp -1 → 0

        # gather over the spatial axis (axis=2) → (B, D, N_out, K, C)
        gathered = keras.ops.take(x, safe_idx, axis=2)

        # zero out border slots — expand (N_out, K) → (1, 1, N_out, K, 1)
        mask = keras.ops.cast(
            keras.ops.reshape(valid, (1, 1, self.N_out, self.K, 1)), gathered.dtype
        )
        return gathered * mask

    def compute_output_shape(self, input_shape):
        # (B, D, N_in, C) → (B, D, N_out, K, C)
        return (input_shape[0], input_shape[1], self.N_out, self.K, input_shape[3])

    def get_config(self):
        config = super().get_config()
        config["neighbor_idx"] = self._neighbor_idx_init.tolist()
        return config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config["neighbor_idx"] = np.asarray(config["neighbor_idx"], dtype=np.int32)
        return cls(**config)


@keras.saving.register_keras_serializable(package="hexagdly_tf")
class HexRingMAC3D(keras.layers.Layer):
    """Depth-aware hex neighbor MAC with optional ring weight sharing.

    Input:  (B, D, N_out, K, Cin)   — output of HexGather3D
    Output: (B, D, N_out, Cout)

    Holds a single depth tap's weight and applies it to every frame (the depth
    axis is a passthrough leading dimension).  Weight modes match HexRingMAC:
    share_neighbors=False → weights (K, Cin, Cout), ring_idx=None;
    share_neighbors=True  → weights (num_rings, Cin, Cout), ring_idx (K,) int.

    Bias is baked in as an always-present (Cout,) vector (defaults to zeros when
    ``bias`` is None).  Unlike the 2D HexRingMAC — where the Conv2d export adds
    bias via a Lambda — the bias lives inside this layer so the graph stays free
    of the Lambda op that stock hls4ml cannot convert.  The Conv3d export passes
    the real bias only to the final depth tap (zeros elsewhere) so it is added
    exactly once.
    """

    def __init__(self, weights_array, ring_idx=None, bias=None, **kwargs):
        super().__init__(**kwargs)
        self._weights_init = np.asarray(weights_array, dtype=np.float32)
        self._ring_idx_init = np.asarray(ring_idx, dtype=np.int32) if ring_idx is not None else None
        self.share_neighbors = ring_idx is not None
        Cout = int(self._weights_init.shape[-1])
        if bias is None:
            self._bias_init = np.zeros((Cout,), dtype=np.float32)
        else:
            self._bias_init = np.asarray(bias, dtype=np.float32).reshape(Cout)

    def build(self, input_shape):
        self.mac_weights = self.add_weight(
            name="mac_weights",
            shape=self._weights_init.shape,
            dtype="float32",
            initializer=keras.initializers.Constant(self._weights_init),
            trainable=False,
        )
        if self.share_neighbors:
            self.ring_idx = self.add_weight(
                name="ring_idx",
                shape=self._ring_idx_init.shape,
                dtype="int32",
                initializer=keras.initializers.Constant(self._ring_idx_init),
                trainable=False,
            )
        self.mac_bias = self.add_weight(
            name="mac_bias",
            shape=self._bias_init.shape,
            dtype="float32",
            initializer=keras.initializers.Constant(self._bias_init),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x):
        # x: (B, D, N_out, K, Cin)
        if self.share_neighbors:
            # expand ring weights to (K, Cin, Cout) by gathering along ring axis
            W = keras.ops.take(self.mac_weights, self.ring_idx, axis=0)  # (K, Cin, Cout)
        else:
            W = self.mac_weights  # (K, Cin, Cout)

        # sum over K slots and Cin channels; depth axis d passes through
        # x: (B, D, N_out, K, Cin), W: (K, Cin, Cout) → (B, D, N_out, Cout)
        y = keras.ops.einsum("bdnkc,kco->bdno", x, W)
        return y + keras.ops.reshape(self.mac_bias, (1, 1, 1, -1))

    def compute_output_shape(self, input_shape):
        # (B, D, N_out, K, Cin) → (B, D, N_out, Cout)
        return (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            int(self._weights_init.shape[-1]),
        )

    def get_config(self):
        config = super().get_config()
        config["weights_array"] = self._weights_init.tolist()
        config["ring_idx"] = (
            self._ring_idx_init.tolist() if self._ring_idx_init is not None else None
        )
        config["bias"] = self._bias_init.tolist()
        return config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config["weights_array"] = np.asarray(config["weights_array"], dtype=np.float32)
        if config["ring_idx"] is not None:
            config["ring_idx"] = np.asarray(config["ring_idx"], dtype=np.int32)
        if config.get("bias") is not None:
            config["bias"] = np.asarray(config["bias"], dtype=np.float32)
        return cls(**config)


@keras.saving.register_keras_serializable(package="hexagdly_tf")
class HexMaxPool3D(keras.layers.Layer):
    """Depth-aware hex neighbor max pooling.

    Input:  (B, D_in, N_out, K, C)   — output of HexGather3D over every frame
    Output: (B, D_out, N_out, C)

    For each output frame t (0 <= t < D_out) reduces the max over both the
    depth pool window (depth_size taps at stride depth_stride) and the K
    spatial neighbor slots:

        out[b, t, n, c] = max over d in [0, depth_size), ki in [0, K) of
                          x[b, t*depth_stride + d, n, ki, c]

    D_out = (D_in - depth_size) // depth_stride + 1 (valid depth pooling; hex
    MaxPool has no depth padding).  Border slots were zeroed by HexGather3D, so
    the max matches hexagdly's zero-padding at the grid border.
    """

    def __init__(self, depth_size, depth_stride, **kwargs):
        super().__init__(**kwargs)
        self.depth_size = int(depth_size)
        self.depth_stride = int(depth_stride)

    def _d_out(self, d_in):
        return (int(d_in) - self.depth_size) // self.depth_stride + 1

    def call(self, x):
        # x: (B, D_in, N_out, K, C)
        d_in = x.shape[1]
        ds, dk = self.depth_stride, self.depth_size
        frames = []
        for t in range(self._d_out(d_in)):
            window = x[:, t * ds : t * ds + dk]  # (B, dk, N_out, K, C)
            # max over the K slots (axis 3) then the depth taps (axis 1)
            m = keras.ops.max(keras.ops.max(window, axis=3), axis=1)  # (B, N_out, C)
            frames.append(m)
        return keras.ops.stack(frames, axis=1)  # (B, D_out, N_out, C)

    def compute_output_shape(self, input_shape):
        # (B, D_in, N_out, K, C) → (B, D_out, N_out, C)
        d_out = self._d_out(input_shape[1]) if input_shape[1] is not None else None
        return (input_shape[0], d_out, input_shape[2], input_shape[4])

    def get_config(self):
        config = super().get_config()
        config["depth_size"] = self.depth_size
        config["depth_stride"] = self.depth_stride
        return config


# ---------------------------------------------------------------------------
# Line-buffer (fused gather + MAC) layers — used by the "linebuffer" export
# strategy.  Numerically identical to HexGather -> HexRingMAC, but kept as ONE
# layer that preserves the (H, W) grid so the hls4ml line-buffer kernel can
# slide a window over the streamed image instead of materialising the whole
# gathered tensor (see nnet_hex_conv_linebuffer_stream.h).  Stride-1 only.
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="hexagdly_tf")
class HexConvLineBuffer(keras.layers.Layer):
    """Fused hex neighbor gather + ring-MAC on the 2D grid.

    Input:  (B, H, W, Cin)
    Output: (B, H, W, Cout)   — stride-1 hex conv, output grid == input grid.

    Carries the same data as HexGather + HexRingMAC combined:
      neighbor_idx (N_out, K), -1 = border → zero (N_out == H*W).
      weights_array: (K, Cin, Cout) when ring_idx is None (share_neighbors=False),
                     or (num_rings, Cin, Cout) with ring_idx (K,) mapping each
                     slot to its ring (share_neighbors=True).
      bias (Cout,): baked in, added to the output (None → zeros).

    ``call`` reshapes to flat pixels, gathers, applies the (ring-expanded) MAC and
    reshapes back to the grid — bit-identical to the gather path.  The 2D shape is
    retained purely so the export handler can emit a line-buffer HLS kernel.
    """

    def __init__(
        self,
        neighbor_idx,
        H,
        W,
        weights_array,
        ring_idx=None,
        bias=None,
        stride=1,
        H_out=None,
        W_out=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._neighbor_idx_init = np.asarray(neighbor_idx, dtype=np.int32)
        self.N_out, self.K = self._neighbor_idx_init.shape
        self.H = int(H)  # input grid height
        self.W = int(W)  # input grid width
        self.stride = int(stride)
        self.H_out = int(H_out) if H_out is not None else int(H)
        self.W_out = int(W_out) if W_out is not None else int(W)
        self._weights_init = np.asarray(weights_array, dtype=np.float32)
        self._ring_idx_init = np.asarray(ring_idx, dtype=np.int32) if ring_idx is not None else None
        self.share_neighbors = ring_idx is not None
        self.Cout = int(self._weights_init.shape[-1])
        if bias is None:
            self._bias_init = np.zeros((self.Cout,), dtype=np.float32)
        else:
            self._bias_init = np.asarray(bias, dtype=np.float32).reshape(self.Cout)

    def build(self, input_shape):
        self.neighbor_idx = self.add_weight(
            name="neighbor_idx",
            shape=self._neighbor_idx_init.shape,
            dtype="int32",
            initializer=keras.initializers.Constant(self._neighbor_idx_init),
            trainable=False,
        )
        self.mac_weights = self.add_weight(
            name="mac_weights",
            shape=self._weights_init.shape,
            dtype="float32",
            initializer=keras.initializers.Constant(self._weights_init),
            trainable=False,
        )
        if self.share_neighbors:
            self.ring_idx = self.add_weight(
                name="ring_idx",
                shape=self._ring_idx_init.shape,
                dtype="int32",
                initializer=keras.initializers.Constant(self._ring_idx_init),
                trainable=False,
            )
        self.mac_bias = self.add_weight(
            name="mac_bias",
            shape=self._bias_init.shape,
            dtype="float32",
            initializer=keras.initializers.Constant(self._bias_init),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x):
        # x: (B, H, W, Cin)
        Cin = x.shape[-1]
        x_flat = keras.ops.reshape(x, (-1, self.H * self.W, Cin))  # (B, N_in, Cin)

        valid = self.neighbor_idx >= 0  # (N_out, K)
        safe_idx = keras.ops.where(valid, self.neighbor_idx, 0)
        gathered = keras.ops.take(x_flat, safe_idx, axis=1)  # (B, N_out, K, Cin)
        mask = keras.ops.cast(keras.ops.reshape(valid, (1, self.N_out, self.K, 1)), gathered.dtype)
        gathered = gathered * mask

        if self.share_neighbors:
            W = keras.ops.take(self.mac_weights, self.ring_idx, axis=0)  # (K, Cin, Cout)
        else:
            W = self.mac_weights  # (K, Cin, Cout)

        y = keras.ops.einsum("bnkc,kco->bno", gathered, W)  # (B, N_out, Cout)
        y = y + keras.ops.reshape(self.mac_bias, (1, 1, -1))
        return keras.ops.reshape(y, (-1, self.H_out, self.W_out, self.Cout))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.H_out, self.W_out, self.Cout)

    def get_config(self):
        config = super().get_config()
        config["neighbor_idx"] = self._neighbor_idx_init.tolist()
        config["H"] = self.H
        config["W"] = self.W
        config["stride"] = self.stride
        config["H_out"] = self.H_out
        config["W_out"] = self.W_out
        config["weights_array"] = self._weights_init.tolist()
        config["ring_idx"] = (
            self._ring_idx_init.tolist() if self._ring_idx_init is not None else None
        )
        config["bias"] = self._bias_init.tolist()
        return config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config["neighbor_idx"] = np.asarray(config["neighbor_idx"], dtype=np.int32)
        config["weights_array"] = np.asarray(config["weights_array"], dtype=np.float32)
        if config["ring_idx"] is not None:
            config["ring_idx"] = np.asarray(config["ring_idx"], dtype=np.int32)
        if config.get("bias") is not None:
            config["bias"] = np.asarray(config["bias"], dtype=np.float32)
        return cls(**config)


@keras.saving.register_keras_serializable(package="hexagdly_tf")
class HexConvLineBuffer3D(keras.layers.Layer):
    """Depth-aware fused hex gather + ring-MAC on the 2D grid, per depth frame.

    Input:  (B, D_in, H, W, Cin)
    Output: (B, D_out, H, W, Cout)

    Equivalent to the Conv3d gather export (_conv3d_gather): the spatial neighbor
    gather is identical to HexConvLineBuffer; the depth axis is a dense 1D conv
    of ``depth_size`` taps at ``depth_stride`` (=1 only), summed, with bias added
    once.  depth_padding 'valid' or 'same' controls D_out (== D_in for 'same',
    D_in - depth_size + 1 for 'valid').

    weights_array is per depth tap:
      (depth_size, K, Cin, Cout)        when ring_idx is None, or
      (depth_size, num_rings, Cin, Cout) with ring_idx (K,) when sharing rings.
    """

    def __init__(
        self,
        neighbor_idx,
        H,
        W,
        weights_array,
        depth_size,
        depth_stride=1,
        depth_padding="valid",
        ring_idx=None,
        bias=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._neighbor_idx_init = np.asarray(neighbor_idx, dtype=np.int32)
        self.N_out, self.K = self._neighbor_idx_init.shape
        self.H = int(H)
        self.W = int(W)
        self.depth_size = int(depth_size)
        self.depth_stride = int(depth_stride)
        self.depth_padding = str(depth_padding)
        self._weights_init = np.asarray(weights_array, dtype=np.float32)
        self._ring_idx_init = np.asarray(ring_idx, dtype=np.int32) if ring_idx is not None else None
        self.share_neighbors = ring_idx is not None
        self.Cout = int(self._weights_init.shape[-1])
        if bias is None:
            self._bias_init = np.zeros((self.Cout,), dtype=np.float32)
        else:
            self._bias_init = np.asarray(bias, dtype=np.float32).reshape(self.Cout)

    def build(self, input_shape):
        self.neighbor_idx = self.add_weight(
            name="neighbor_idx",
            shape=self._neighbor_idx_init.shape,
            dtype="int32",
            initializer=keras.initializers.Constant(self._neighbor_idx_init),
            trainable=False,
        )
        self.mac_weights = self.add_weight(
            name="mac_weights",
            shape=self._weights_init.shape,
            dtype="float32",
            initializer=keras.initializers.Constant(self._weights_init),
            trainable=False,
        )
        if self.share_neighbors:
            self.ring_idx = self.add_weight(
                name="ring_idx",
                shape=self._ring_idx_init.shape,
                dtype="int32",
                initializer=keras.initializers.Constant(self._ring_idx_init),
                trainable=False,
            )
        self.mac_bias = self.add_weight(
            name="mac_bias",
            shape=self._bias_init.shape,
            dtype="float32",
            initializer=keras.initializers.Constant(self._bias_init),
            trainable=False,
        )
        super().build(input_shape)

    def _pads(self):
        if self.depth_padding == "same":
            pad_top = (self.depth_size - 1) // 2
            return pad_top, self.depth_size - 1 - pad_top
        return 0, 0

    def call(self, x):
        # x: (B, D_in, H, W, Cin)
        Cin = x.shape[-1]
        D_in = x.shape[1]
        x_flat = keras.ops.reshape(x, (-1, D_in, self.H * self.W, Cin))

        pad_top, pad_bot = self._pads()
        if pad_top or pad_bot:
            x_flat = keras.ops.pad(x_flat, [[0, 0], [pad_top, pad_bot], [0, 0], [0, 0]])
        D_padded = D_in + pad_top + pad_bot
        D_eff = D_padded - self.depth_size + 1

        valid = self.neighbor_idx >= 0
        safe_idx = keras.ops.where(valid, self.neighbor_idx, 0)
        G = keras.ops.take(x_flat, safe_idx, axis=2)  # (B, D_padded, N_out, K, Cin)
        mask = keras.ops.cast(keras.ops.reshape(valid, (1, 1, self.N_out, self.K, 1)), G.dtype)
        G = G * mask

        if self.share_neighbors:
            # (depth_size, K, Cin, Cout) by gathering rings along axis 1
            W = keras.ops.take(self.mac_weights, self.ring_idx, axis=1)
        else:
            W = self.mac_weights  # (depth_size, K, Cin, Cout)

        acc = None
        for d in range(self.depth_size):
            frames = G[:, d : d + D_eff]  # (B, D_eff, N_out, K, Cin)
            y_d = keras.ops.einsum("btnkc,kco->btno", frames, W[d])
            acc = y_d if acc is None else acc + y_d
        acc = acc + keras.ops.reshape(self.mac_bias, (1, 1, 1, -1))
        return keras.ops.reshape(acc, (-1, D_eff, self.H, self.W, self.Cout))

    def compute_output_shape(self, input_shape):
        D_in = input_shape[1]
        if D_in is None:
            D_out = None
        elif self.depth_padding == "same":
            D_out = D_in
        else:
            D_out = D_in - self.depth_size + 1
        return (input_shape[0], D_out, self.H, self.W, self.Cout)

    def get_config(self):
        config = super().get_config()
        config["neighbor_idx"] = self._neighbor_idx_init.tolist()
        config["H"] = self.H
        config["W"] = self.W
        config["weights_array"] = self._weights_init.tolist()
        config["depth_size"] = self.depth_size
        config["depth_stride"] = self.depth_stride
        config["depth_padding"] = self.depth_padding
        config["ring_idx"] = (
            self._ring_idx_init.tolist() if self._ring_idx_init is not None else None
        )
        config["bias"] = self._bias_init.tolist()
        return config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config["neighbor_idx"] = np.asarray(config["neighbor_idx"], dtype=np.int32)
        config["weights_array"] = np.asarray(config["weights_array"], dtype=np.float32)
        if config["ring_idx"] is not None:
            config["ring_idx"] = np.asarray(config["ring_idx"], dtype=np.int32)
        if config.get("bias") is not None:
            config["bias"] = np.asarray(config["bias"], dtype=np.float32)
        return cls(**config)


@keras.saving.register_keras_serializable(package="hexagdly_tf")
class HexPoolLineBuffer(keras.layers.Layer):
    """Fused hex neighbor max-pool on the 2D grid (line-buffer export).

    Input:  (B, H, W, C)
    Output: (B, H_out, W_out, C)

    Numerically identical to HexGather -> HexMaxPool, but kept as ONE layer that
    preserves the (H, W) grid so the hls4ml line-buffer kernel can slide a window
    over the streamed image.  Border slots (neighbor_idx == -1) contribute 0.0,
    matching hexagdly's zero-padding at the grid border.  Supports any hex stride
    (output pixel (oh, ow) maps to input base (stride*oh, stride*ow)).
    """

    def __init__(self, neighbor_idx, H, W, stride=1, H_out=None, W_out=None, **kwargs):
        super().__init__(**kwargs)
        self._neighbor_idx_init = np.asarray(neighbor_idx, dtype=np.int32)
        self.N_out, self.K = self._neighbor_idx_init.shape
        self.H = int(H)
        self.W = int(W)
        self.stride = int(stride)
        self.H_out = int(H_out) if H_out is not None else int(H)
        self.W_out = int(W_out) if W_out is not None else int(W)

    def build(self, input_shape):
        self.neighbor_idx = self.add_weight(
            name="neighbor_idx",
            shape=self._neighbor_idx_init.shape,
            dtype="int32",
            initializer=keras.initializers.Constant(self._neighbor_idx_init),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x):
        # x: (B, H, W, C)
        C = x.shape[-1]
        x_flat = keras.ops.reshape(x, (-1, self.H * self.W, C))
        valid = self.neighbor_idx >= 0
        safe_idx = keras.ops.where(valid, self.neighbor_idx, 0)
        gathered = keras.ops.take(x_flat, safe_idx, axis=1)  # (B, N_out, K, C)
        mask = keras.ops.cast(keras.ops.reshape(valid, (1, self.N_out, self.K, 1)), gathered.dtype)
        gathered = gathered * mask
        pooled = keras.ops.max(gathered, axis=2)  # (B, N_out, C)
        return keras.ops.reshape(pooled, (-1, self.H_out, self.W_out, C))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.H_out, self.W_out, input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config["neighbor_idx"] = self._neighbor_idx_init.tolist()
        config["H"] = self.H
        config["W"] = self.W
        config["stride"] = self.stride
        config["H_out"] = self.H_out
        config["W_out"] = self.W_out
        return config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config["neighbor_idx"] = np.asarray(config["neighbor_idx"], dtype=np.int32)
        return cls(**config)


@keras.saving.register_keras_serializable(package="hexagdly_tf")
class HexPoolLineBuffer3D(keras.layers.Layer):
    """Depth-aware fused hex neighbor max-pool on the 2D grid (line-buffer export).

    Input:  (B, D_in, H, W, C)
    Output: (B, D_out, H, W, C)

    Equivalent to HexGather3D -> HexMaxPool3D: for each output frame t reduces the
    max over the depth pool window (depth_size taps at depth_stride) and the K
    spatial neighbor taps.  Border spatial taps contribute 0 (hexagdly zero-pad).
    D_out = (D_in - depth_size) // depth_stride + 1 (valid depth pooling).
    Hex stride is 1 (output grid == input grid).
    """

    def __init__(self, neighbor_idx, H, W, depth_size, depth_stride=1, **kwargs):
        super().__init__(**kwargs)
        self._neighbor_idx_init = np.asarray(neighbor_idx, dtype=np.int32)
        self.N_out, self.K = self._neighbor_idx_init.shape
        self.H = int(H)
        self.W = int(W)
        self.depth_size = int(depth_size)
        self.depth_stride = int(depth_stride)

    def build(self, input_shape):
        self.neighbor_idx = self.add_weight(
            name="neighbor_idx",
            shape=self._neighbor_idx_init.shape,
            dtype="int32",
            initializer=keras.initializers.Constant(self._neighbor_idx_init),
            trainable=False,
        )
        super().build(input_shape)

    def _d_out(self, d_in):
        return (int(d_in) - self.depth_size) // self.depth_stride + 1

    def call(self, x):
        # x: (B, D_in, H, W, C)
        C = x.shape[-1]
        d_in = x.shape[1]
        x_flat = keras.ops.reshape(x, (-1, d_in, self.H * self.W, C))
        valid = self.neighbor_idx >= 0
        safe_idx = keras.ops.where(valid, self.neighbor_idx, 0)
        G = keras.ops.take(x_flat, safe_idx, axis=2)  # (B, D_in, N_out, K, C)
        mask = keras.ops.cast(keras.ops.reshape(valid, (1, 1, self.N_out, self.K, 1)), G.dtype)
        G = G * mask
        ds, dk = self.depth_stride, self.depth_size
        frames = []
        for t in range(self._d_out(d_in)):
            window = G[:, t * ds : t * ds + dk]  # (B, dk, N_out, K, C)
            m = keras.ops.max(keras.ops.max(window, axis=3), axis=1)  # (B, N_out, C)
            frames.append(m)
        pooled = keras.ops.stack(frames, axis=1)  # (B, D_out, N_out, C)
        return keras.ops.reshape(pooled, (-1, self._d_out(d_in), self.H, self.W, C))

    def compute_output_shape(self, input_shape):
        d_out = self._d_out(input_shape[1]) if input_shape[1] is not None else None
        return (input_shape[0], d_out, self.H, self.W, input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config["neighbor_idx"] = self._neighbor_idx_init.tolist()
        config["H"] = self.H
        config["W"] = self.W
        config["depth_size"] = self.depth_size
        config["depth_stride"] = self.depth_stride
        return config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config["neighbor_idx"] = np.asarray(config["neighbor_idx"], dtype=np.int32)
        return cls(**config)
