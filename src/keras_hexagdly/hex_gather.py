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

import numpy as np
import keras


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
        valid = self.neighbor_idx >= 0                          # (N_out, K)
        safe_idx = keras.ops.where(valid, self.neighbor_idx, 0) # clamp -1 → 0

        # gather: x[(B), safe_idx[n,k], (C)] → (B, N_out, K, C)
        gathered = keras.ops.take(x, safe_idx, axis=1)

        # zero out border slots — expand (N_out, K) → (1, N_out, K, 1)
        mask = keras.ops.cast(
            keras.ops.reshape(valid, (1, self.N_out, self.K, 1)), gathered.dtype
        )
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
        self._ring_idx_init = (
            np.asarray(ring_idx, dtype=np.int32) if ring_idx is not None else None
        )
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
