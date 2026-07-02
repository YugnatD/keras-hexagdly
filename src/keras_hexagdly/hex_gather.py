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
    """

    def __init__(self, weights_array, ring_idx=None, **kwargs):
        super().__init__(**kwargs)
        self._weights_init = np.asarray(weights_array, dtype=np.float32)
        self._ring_idx_init = np.asarray(ring_idx, dtype=np.int32) if ring_idx is not None else None
        self.share_neighbors = ring_idx is not None

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
        return keras.ops.einsum("bnkc,kco->bno", x, W)

    def get_config(self):
        config = super().get_config()
        config["weights_array"] = self._weights_init.tolist()
        config["ring_idx"] = (
            self._ring_idx_init.tolist() if self._ring_idx_init is not None else None
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config["weights_array"] = np.asarray(config["weights_array"], dtype=np.float32)
        if config["ring_idx"] is not None:
            config["ring_idx"] = np.asarray(config["ring_idx"], dtype=np.int32)
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
