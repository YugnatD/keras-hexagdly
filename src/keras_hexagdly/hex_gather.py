"""Line-buffer layers for hls4ml export.

These layers implement the "linebuffer" export strategy in hls4ml_ext.py. They
are NOT used during normal training or PC inference -- the zig-zag hexagdly
layers are used for that. They exist solely to give hls4ml a synthesizable
representation, and their call() reproduces the original layer's arithmetic
exactly so the patched model can be validated against it on the PC.

Each one fuses the neighbour gather and the MAC (or the max) into a SINGLE
layer that keeps the (H, W, C) grid, which is what lets the HLS kernel slide a
register window over shift-register line buffers instead of materialising the
whole frame. An earlier design split gather and MAC into separate layers; in
io_stream that materialised both the frame and the N_out*K*Cin gathered tensor
in registers (34378 FF against 3077 for a comparable native conv), and it was
removed on 2026-07-28.

  HexConvLineBuffer / HexConvLineBuffer3D  -- Conv2d / Conv3d
  HexPoolLineBuffer / HexPoolLineBuffer3D  -- MaxPool2d / MaxPool3d
"""

import keras
import numpy as np


def _opt_out_of_automatic_xla(layer):
    """Keep Keras from compiling a model that contains ``layer`` with XLA by itself.

    The gather table of these layers is an int32 weight. TensorFlow keeps int32 variables on the
    CPU, and the XLA compilation Keras turns on automatically on a GPU cannot read them from there
    ("resource located in device CPU from device GPU"): ``predict()`` / ``fit()`` of an exported
    model failed on a GPU. ``supports_jit = False`` is the opt-out Keras provides for this. It
    switches the automatic compilation of the whole model off, so it is only done on TensorFlow:
    JAX keeps its jit (turning it off made ``predict`` about 4x slower there)."""
    if keras.backend.backend() == "tensorflow":
        layer.supports_jit = False


@keras.saving.register_keras_serializable(package="hexagdly_tf")
class HexConvLineBuffer(keras.layers.Layer):
    """Fused hex neighbor gather + ring-MAC on the 2D grid.

    Input:  (B, H, W, Cin)
    Output: (B, H, W, Cout)   — stride-1 hex conv, output grid == input grid.

    Carries the gather table and the MAC weights together:
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
        _opt_out_of_automatic_xla(self)
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

        valid = keras.ops.greater_equal(self.neighbor_idx, 0)  # (N_out, K)
        safe_idx = keras.ops.maximum(self.neighbor_idx, 0)
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

    The spatial neighbor
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
        _opt_out_of_automatic_xla(self)
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

        # maximum(), not where(): the border sentinel is -1, so clamping at 0
        # picks the same placeholder slot the mask below zeroes out anyway. The
        # where() form is what it replaced, and on the jax backend that one
        # raises during predict() -- ops.where over a Variable index forces a
        # read outside the trainer's stateless scope, where the variable has
        # been purged (TypeError: 'NoneType' object is not callable). Direct
        # __call__ and the TensorFlow backend never hit it.
        valid = keras.ops.greater_equal(self.neighbor_idx, 0)
        safe_idx = keras.ops.maximum(self.neighbor_idx, 0)
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

    Numerically identical to a separate gather then max, but kept as ONE layer that
    preserves the (H, W) grid so the hls4ml line-buffer kernel can slide a window
    over the streamed image.  Border slots (neighbor_idx == -1) contribute 0.0,
    matching hexagdly's zero-padding at the grid border.  Supports any hex stride
    (output pixel (oh, ow) maps to input base (stride*oh, stride*ow)).
    """

    def __init__(self, neighbor_idx, H, W, stride=1, H_out=None, W_out=None, **kwargs):
        super().__init__(**kwargs)
        _opt_out_of_automatic_xla(self)
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
        # maximum(), not where(): the border sentinel is -1, so clamping at 0
        # picks the same placeholder slot the mask below zeroes out anyway. The
        # where() form is what it replaced, and on the jax backend that one
        # raises during predict() -- ops.where over a Variable index forces a
        # read outside the trainer's stateless scope, where the variable has
        # been purged (TypeError: 'NoneType' object is not callable). Direct
        # __call__ and the TensorFlow backend never hit it.
        valid = keras.ops.greater_equal(self.neighbor_idx, 0)
        safe_idx = keras.ops.maximum(self.neighbor_idx, 0)
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

    For each output frame t, reduces the
    max over the depth pool window (depth_size taps at depth_stride) and the K
    spatial neighbor taps.  Border spatial taps contribute 0 (hexagdly zero-pad).
    D_out = (D_in - depth_size) // depth_stride + 1 (valid depth pooling).
    Hex stride is 1 (output grid == input grid).
    """

    def __init__(self, neighbor_idx, H, W, depth_size, depth_stride=1, **kwargs):
        super().__init__(**kwargs)
        _opt_out_of_automatic_xla(self)
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
        # maximum(), not where(): the border sentinel is -1, so clamping at 0
        # picks the same placeholder slot the mask below zeroes out anyway. The
        # where() form is what it replaced, and on the jax backend that one
        # raises during predict() -- ops.where over a Variable index forces a
        # read outside the trainer's stateless scope, where the variable has
        # been purged (TypeError: 'NoneType' object is not callable). Direct
        # __call__ and the TensorFlow backend never hit it.
        valid = keras.ops.greater_equal(self.neighbor_idx, 0)
        safe_idx = keras.ops.maximum(self.neighbor_idx, 0)
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
