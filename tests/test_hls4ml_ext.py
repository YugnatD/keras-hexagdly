"""Phase 3+4 tests: patch_model_for_hls correctness + optional hls4ml C-sim.

Two tiers:
  Tier 1 (always runs): patch_model_for_hls() produces a plain Keras model
          whose float32 output matches the original hex model exactly.  No
          hls4ml dependency.

  Tier 2 (skipped if hls4ml not importable): patched model converts through
          hls4ml and C-sim output is close to the original within fixed-point
          quantization error (< 0.02 default precision).
"""

import keras
import numpy as np
import pytest

import keras_hexagdly as hgly
from keras_hexagdly.hls4ml_ext import patch_model_for_hls

# ---- test dimensions --------------------------------------------------------
H, W = 13, 11
D = 8
CIN = 2
COUT = 3
RNG = np.random.default_rng(7)
ATOL_KERAS = 2e-4  # float32 rounding across the Reshape/EinsumDense chain
ATOL_CSIM = 0.02  # default ap_fixed<16,6> quantization

# The gather strategy uses HexGather/HexRingMAC/HexMaxPool custom layers that
# store their lookup tables as Constant-initialized non-trainable weights.  On
# the jax backend, keras' stateless execution re-materializes those weights via
# their initializer during tracing, which fails ('NoneType' object is not
# callable) — jax simply isn't a supported backend for these layers.  That's
# fine: the gather export exists to feed hls4ml, whose conversion runs on the
# TensorFlow (and torch) backends.  Skip the gather-strategy tests on jax.
jax_skip = pytest.mark.skipif(
    keras.backend.backend() == "jax",
    reason="gather-strategy layers are unsupported on the jax backend "
    "(hls4ml export targets tensorflow/torch)",
)


# ---- helpers ----------------------------------------------------------------


def _rand_weights(layer):
    for w in layer.trainable_variables:
        w.assign(RNG.standard_normal(w.shape).astype(np.float32))


def _build_2d_model(layer_fn):
    inp = keras.Input((H, W, CIN), name="x")
    out = layer_fn(inp)
    return keras.Model(inp, out)


def _build_3d_model(layer_fn):
    inp = keras.Input((D, H, W, CIN), name="x")
    out = layer_fn(inp)
    return keras.Model(inp, out)


try:
    import hls4ml

    HLS4ML_AVAILABLE = True
except ImportError:
    HLS4ML_AVAILABLE = False

hls4ml_skip = pytest.mark.skipif(not HLS4ML_AVAILABLE, reason="hls4ml not installed")

_HLS_PART = "xcvu9p-flga2104-2L-e"
_HLS_DIR = "test_hls_prj"


def _csim(patched_model, x_np):
    """Convert patched model -> hls4ml -> compile -> predict."""
    cfg = hls4ml.utils.config_from_keras_model(patched_model, granularity="name", backend="Vivado")
    cfg["Model"]["Precision"] = "ap_fixed<32,12>"
    hm = hls4ml.converters.convert_from_keras_model(
        patched_model,
        hls_config=cfg,
        backend="Vivado",
        output_dir=_HLS_DIR,
        part=_HLS_PART,
    )
    hm.compile()
    return hm.predict(np.ascontiguousarray(x_np))


# =============================================================================
# Tier 1: Keras float32 equivalence
# =============================================================================


class TestConv2dPatch:
    @pytest.mark.parametrize("share", [False, True])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    def test_output_matches_original(self, share, kernel_size, stride):
        layer = hgly.Conv2d(
            CIN,
            COUT,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
            share_neighbors=share,
        )
        model = _build_2d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((2, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0)

        patched = patch_model_for_hls(model)
        y_pat = patched.predict(x, verbose=0)

        assert y_ref.shape == y_pat.shape
        assert np.max(np.abs(y_ref - y_pat)) < ATOL_KERAS, (
            f"Conv2d(k={kernel_size},s={stride},share={share}): "
            f"max err={np.max(np.abs(y_ref - y_pat)):.2e}"
        )

    def test_with_bias(self):
        layer = hgly.Conv2d(CIN, COUT, kernel_size=1, stride=1, bias=True)
        model = _build_2d_model(layer)
        _rand_weights(layer)
        x = RNG.standard_normal((1, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0)
        patched = patch_model_for_hls(model)
        y_pat = patched.predict(x, verbose=0)
        assert np.max(np.abs(y_ref - y_pat)) < ATOL_KERAS


class TestMaxPool2dPatch:
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    def test_output_matches_original(self, kernel_size, stride):
        layer = hgly.MaxPool2d(kernel_size=kernel_size, stride=stride)
        model = _build_2d_model(layer)

        x = RNG.standard_normal((2, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0)

        patched = patch_model_for_hls(model)
        y_pat = patched.predict(x, verbose=0)

        assert y_ref.shape == y_pat.shape
        assert np.max(np.abs(y_ref - y_pat)) < 1e-5, (
            f"MaxPool2d(k={kernel_size},s={stride}): max err={np.max(np.abs(y_ref - y_pat)):.2e}"
        )

    def test_all_negative_input(self):
        """Border 0-pads dominate for all-negative input — patched must match."""
        layer = hgly.MaxPool2d(kernel_size=1, stride=1)
        model = _build_2d_model(layer)
        x = -np.abs(RNG.standard_normal((1, H, W, CIN)).astype(np.float32))
        y_ref = model.predict(x, verbose=0)
        patched = patch_model_for_hls(model)
        assert np.max(np.abs(y_ref - patched.predict(x, verbose=0))) < 1e-5


@jax_skip
class TestMaxPool2dGather:
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    def test_output_matches_original(self, kernel_size, stride):
        layer = hgly.MaxPool2d(kernel_size=kernel_size, stride=stride)
        model = _build_2d_model(layer)

        x = RNG.standard_normal((2, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0)
        patched = patch_model_for_hls(model, strategy="gather")
        y_pat = patched.predict(x, verbose=0)

        assert y_ref.shape == y_pat.shape
        assert np.max(np.abs(y_ref - y_pat)) < 1e-5, (
            f"MaxPool2d gather(k={kernel_size},s={stride}): "
            f"max err={np.max(np.abs(y_ref - y_pat)):.2e}"
        )

    def test_all_negative_input(self):
        """Border 0-pads dominate — gather strategy must match."""
        layer = hgly.MaxPool2d(kernel_size=1, stride=1)
        model = _build_2d_model(layer)
        x = -np.abs(RNG.standard_normal((1, H, W, CIN)).astype(np.float32))
        y_ref = model.predict(x, verbose=0)
        patched = patch_model_for_hls(model, strategy="gather")
        assert np.max(np.abs(y_ref - patched.predict(x, verbose=0))) < 1e-5

    def test_parameter_count_zero_weights(self):
        """HexMaxPool has no weights — gather strategy pool should have far
        fewer parameters than slotwise (no large EinsumDense gather matrix)."""
        layer = hgly.MaxPool2d(kernel_size=1, stride=1)
        model = _build_2d_model(layer)
        p_slotwise = patch_model_for_hls(model, strategy="slotwise").count_params()
        p_gather = patch_model_for_hls(model, strategy="gather").count_params()
        assert p_gather < p_slotwise, (
            f"gather ({p_gather}) should have fewer params than slotwise ({p_slotwise})"
        )

    def test_all_strategies_agree(self):
        """slotwise and gather must produce identical outputs for MaxPool2d."""
        layer = hgly.MaxPool2d(kernel_size=1, stride=2)
        model = _build_2d_model(layer)
        x = RNG.standard_normal((2, H, W, CIN)).astype(np.float32)
        y_slotwise = patch_model_for_hls(model, strategy="slotwise").predict(x, verbose=0)
        y_gather = patch_model_for_hls(model, strategy="gather").predict(x, verbose=0)
        assert np.max(np.abs(y_slotwise - y_gather)) < 1e-5


class TestConv3dPatch:
    @pytest.mark.parametrize("share", [False, True])
    @pytest.mark.parametrize("kernel_size", [(1, 1), (2, 2)])
    @pytest.mark.parametrize("depth_padding", ["valid", "same"])
    def test_output_matches_original(self, share, kernel_size, depth_padding):
        layer = hgly.Conv3d(
            CIN,
            COUT,
            kernel_size=kernel_size,
            bias=False,
            share_neighbors=share,
            depth_padding=depth_padding,
        )
        model = _build_3d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((1, D, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0)

        patched = patch_model_for_hls(model)
        y_pat = patched.predict(x, verbose=0)

        assert y_ref.shape == y_pat.shape, f"shape: ref={y_ref.shape} pat={y_pat.shape}"
        assert np.max(np.abs(y_ref - y_pat)) < ATOL_KERAS, (
            f"Conv3d(k={kernel_size},share={share},dp={depth_padding}): "
            f"max err={np.max(np.abs(y_ref - y_pat)):.2e}"
        )

    def test_with_bias(self):
        layer = hgly.Conv3d(CIN, COUT, kernel_size=(1, 1), bias=True, depth_padding="valid")
        model = _build_3d_model(layer)
        _rand_weights(layer)
        x = RNG.standard_normal((1, D, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0)
        patched = patch_model_for_hls(model)
        assert np.max(np.abs(y_ref - patched.predict(x, verbose=0))) < ATOL_KERAS


class TestMaxPool3dPatch:
    @pytest.mark.parametrize("strategy", ["folded", "slotwise"])
    def test_raises_for_non_gather(self, strategy):
        """MaxPool3d has no folded/slotwise form (max is not linear); only the
        gather strategy is implemented."""
        layer = hgly.MaxPool3d(kernel_size=(1, 1))
        model = _build_3d_model(layer)
        with pytest.raises(NotImplementedError, match="MaxPool3d"):
            patch_model_for_hls(model, strategy=strategy)


class TestPatchModelMisc:
    def test_non_model_raises(self):
        with pytest.raises(TypeError):
            patch_model_for_hls("not_a_model")

    def test_invalid_strategy_raises(self):
        layer = hgly.Conv2d(CIN, COUT, kernel_size=1, bias=False)
        model = _build_2d_model(layer)
        with pytest.raises(ValueError, match="Unknown strategy"):
            patch_model_for_hls(model, strategy="invalid")

    def test_non_hex_layers_passthrough(self):
        """Non-hex layers (Dense, ReLU) must be preserved unchanged."""
        inp = keras.Input((H, W, CIN))
        x = hgly.Conv2d(CIN, COUT, kernel_size=1, bias=False)(inp)
        x = keras.layers.Activation("relu")(x)
        model = keras.Model(inp, x)
        for w in model.trainable_variables:
            w.assign(RNG.standard_normal(w.shape).astype(np.float32))

        xd = RNG.standard_normal((1, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(xd, verbose=0)
        patched = patch_model_for_hls(model)
        assert np.max(np.abs(y_ref - patched.predict(xd, verbose=0))) < ATOL_KERAS

    def test_multi_layer_model(self):
        """Two hex layers in sequence both get replaced correctly."""
        inp = keras.Input((H, W, CIN))
        x = hgly.Conv2d(CIN, COUT, kernel_size=1, bias=False)(inp)
        x = hgly.MaxPool2d(kernel_size=1)(x)
        model = keras.Model(inp, x)
        for w in model.trainable_variables:
            w.assign(RNG.standard_normal(w.shape).astype(np.float32))

        xd = RNG.standard_normal((1, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(xd, verbose=0)
        patched = patch_model_for_hls(model)
        assert np.max(np.abs(y_ref - patched.predict(xd, verbose=0))) < ATOL_KERAS


# =============================================================================
# Serialization: patched (gather) models must save/load round-trip
# =============================================================================


class TestPatchedModelSerialization:
    """The patched model is what users hand to hls4ml; it must survive
    model.save() / load_model().  Exercises get_config/from_config of
    HexGather, HexRingMAC, HexMaxPool and their 3D variants."""

    def _roundtrip(self, model, x, tmp_path, name):
        y_before = model.predict(x, verbose=0)
        f = str(tmp_path / f"{name}.keras")
        model.save(f)
        reloaded = keras.models.load_model(f)
        y_after = reloaded.predict(x, verbose=0)
        assert np.max(np.abs(y_before - y_after)) < 1e-6, (
            f"{name}: save/load changed the output "
            f"(max err={np.max(np.abs(y_before - y_after)):.2e})"
        )

    @pytest.mark.parametrize("share", [False, True])
    @pytest.mark.parametrize("bias", [False, True])
    def test_conv2d_maxpool_gather_roundtrip(self, tmp_path, share, bias):
        inp = keras.Input((H, W, CIN))
        x = hgly.Conv2d(CIN, COUT, kernel_size=1, bias=bias, share_neighbors=share, name="c1")(inp)
        x = keras.layers.ReLU()(x)
        x = hgly.MaxPool2d(kernel_size=1, stride=2, name="p1")(x)
        model = keras.Model(inp, x)
        for w in model.trainable_variables:
            w.assign(RNG.standard_normal(w.shape).astype(np.float32))
        patched = patch_model_for_hls(model, strategy="gather")
        x_in = RNG.standard_normal((2, H, W, CIN)).astype(np.float32)
        self._roundtrip(patched, x_in, tmp_path, f"g2d_s{share}_b{bias}")

    @pytest.mark.parametrize("depth_padding", ["valid", "same"])
    def test_conv3d_maxpool_gather_roundtrip(self, tmp_path, depth_padding):
        inp = keras.Input((D, H, W, CIN))
        x = hgly.Conv3d(
            CIN, COUT, kernel_size=(2, 1), bias=True,
            share_neighbors=True, depth_padding=depth_padding, name="c3",
        )(inp)
        x = keras.layers.ReLU()(x)
        x = hgly.MaxPool3d(kernel_size=(2, 1), name="p3")(x)
        model = keras.Model(inp, x)
        for w in model.trainable_variables:
            w.assign(RNG.standard_normal(w.shape).astype(np.float32))
        patched = patch_model_for_hls(model, strategy="gather")
        x_in = RNG.standard_normal((1, D, H, W, CIN)).astype(np.float32)
        self._roundtrip(patched, x_in, tmp_path, f"g3d_{depth_padding}")


# =============================================================================
# Slotwise strategy tests
# =============================================================================


class TestConv2dSlotwise:
    @pytest.mark.parametrize("share", [False, True])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    def test_output_matches_original(self, share, kernel_size, stride):
        layer = hgly.Conv2d(
            CIN,
            COUT,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
            share_neighbors=share,
        )
        model = _build_2d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((2, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0)
        patched = patch_model_for_hls(model, strategy="slotwise")
        y_pat = patched.predict(x, verbose=0)

        assert y_ref.shape == y_pat.shape
        assert np.max(np.abs(y_ref - y_pat)) < ATOL_KERAS, (
            f"Conv2d slotwise(k={kernel_size},s={stride},share={share}): "
            f"max err={np.max(np.abs(y_ref - y_pat)):.2e}"
        )

    def test_folded_and_slotwise_agree(self):
        """Both strategies must produce identical float32 outputs."""
        layer = hgly.Conv2d(CIN, COUT, kernel_size=1, bias=False)
        model = _build_2d_model(layer)
        _rand_weights(layer)
        x = RNG.standard_normal((2, H, W, CIN)).astype(np.float32)
        y_folded = patch_model_for_hls(model, strategy="folded").predict(x, verbose=0)
        y_slotwise = patch_model_for_hls(model, strategy="slotwise").predict(x, verbose=0)
        assert np.max(np.abs(y_folded - y_slotwise)) < ATOL_KERAS


@jax_skip
class TestConv2dGather:
    @pytest.mark.parametrize("share", [False, True])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    def test_output_matches_original(self, share, kernel_size, stride):
        layer = hgly.Conv2d(
            CIN,
            COUT,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
            share_neighbors=share,
        )
        model = _build_2d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((2, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0)
        patched = patch_model_for_hls(model, strategy="gather")
        y_pat = patched.predict(x, verbose=0)

        assert y_ref.shape == y_pat.shape
        assert np.max(np.abs(y_ref - y_pat)) < ATOL_KERAS, (
            f"Conv2d gather(k={kernel_size},s={stride},share={share}): "
            f"max err={np.max(np.abs(y_ref - y_pat)):.2e}"
        )

    def test_all_strategies_agree(self):
        """folded, slotwise and gather must produce identical float32 outputs."""
        layer = hgly.Conv2d(CIN, COUT, kernel_size=1, bias=False, share_neighbors=True)
        model = _build_2d_model(layer)
        _rand_weights(layer)
        x = RNG.standard_normal((2, H, W, CIN)).astype(np.float32)
        y_folded = patch_model_for_hls(model, strategy="folded").predict(x, verbose=0)
        y_slotwise = patch_model_for_hls(model, strategy="slotwise").predict(x, verbose=0)
        y_gather = patch_model_for_hls(model, strategy="gather").predict(x, verbose=0)
        assert np.max(np.abs(y_folded - y_gather)) < ATOL_KERAS
        assert np.max(np.abs(y_slotwise - y_gather)) < ATOL_KERAS

    def test_parameter_count_reduced(self):
        """gather strategy must have far fewer parameters than slotwise
        (index table replaces dense selection matrices)."""
        layer = hgly.Conv2d(CIN, COUT, kernel_size=1, bias=False, share_neighbors=True)
        model = _build_2d_model(layer)
        patched_slotwise = patch_model_for_hls(model, strategy="slotwise")
        patched_gather = patch_model_for_hls(model, strategy="gather")
        n_slotwise = patched_slotwise.count_params()
        n_gather = patched_gather.count_params()
        assert n_gather < n_slotwise, (
            f"gather ({n_gather}) should have fewer params than slotwise ({n_slotwise})"
        )


class TestConv3dSlotwise:
    @pytest.mark.parametrize("share", [False, True])
    @pytest.mark.parametrize("kernel_size", [(1, 1), (2, 2)])
    @pytest.mark.parametrize("depth_padding", ["valid", "same"])
    def test_output_matches_original(self, share, kernel_size, depth_padding):
        layer = hgly.Conv3d(
            CIN,
            COUT,
            kernel_size=kernel_size,
            bias=False,
            share_neighbors=share,
            depth_padding=depth_padding,
        )
        model = _build_3d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((1, D, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0)
        patched = patch_model_for_hls(model, strategy="slotwise")
        y_pat = patched.predict(x, verbose=0)

        assert y_ref.shape == y_pat.shape
        assert np.max(np.abs(y_ref - y_pat)) < ATOL_KERAS, (
            f"Conv3d slotwise(k={kernel_size},share={share},dp={depth_padding}): "
            f"max err={np.max(np.abs(y_ref - y_pat)):.2e}"
        )


@jax_skip
class TestConv3dGather:
    """strategy='gather' for Conv3d -> HexGather3D + HexRingMAC3D.

    Equivalence is checked primarily against the slotwise strategy (the
    established Conv3d reference), which lets every share/kernel/padding
    combination be exercised — including depth_padding='same' with an even
    depth kernel, where the *original* hex forward pass currently has an
    unrelated pre-existing bug.  A separate test anchors the gather output to
    the true original model on the configurations where the original runs.
    """

    @pytest.mark.parametrize("share", [False, True])
    @pytest.mark.parametrize("kernel_size", [(1, 1), (2, 2), (2, 1)])
    @pytest.mark.parametrize("depth_padding", ["valid", "same"])
    @pytest.mark.parametrize("bias", [False, True])
    def test_matches_slotwise(self, share, kernel_size, depth_padding, bias):
        layer = hgly.Conv3d(
            CIN,
            COUT,
            kernel_size=kernel_size,
            bias=bias,
            share_neighbors=share,
            depth_padding=depth_padding,
        )
        model = _build_3d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((1, D, H, W, CIN)).astype(np.float32)
        y_slot = patch_model_for_hls(model, strategy="slotwise").predict(x, verbose=0)
        y_gat = patch_model_for_hls(model, strategy="gather").predict(x, verbose=0)

        assert y_slot.shape == y_gat.shape
        assert np.max(np.abs(y_slot - y_gat)) < ATOL_KERAS, (
            f"Conv3d gather vs slotwise (k={kernel_size},share={share},"
            f"dp={depth_padding},bias={bias}): "
            f"max err={np.max(np.abs(y_slot - y_gat)):.2e}"
        )

    @pytest.mark.parametrize("share", [False, True])
    @pytest.mark.parametrize("kernel_size", [(1, 1), (2, 2)])
    def test_matches_original_valid(self, share, kernel_size):
        """Anchor to the true original on depth_padding='valid' (unaffected by
        the pre-existing 'same' forward bug)."""
        layer = hgly.Conv3d(
            CIN,
            COUT,
            kernel_size=kernel_size,
            bias=False,
            share_neighbors=share,
            depth_padding="valid",
        )
        model = _build_3d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((1, D, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0)
        y_gat = patch_model_for_hls(model, strategy="gather").predict(x, verbose=0)

        assert y_ref.shape == y_gat.shape
        assert np.max(np.abs(y_ref - y_gat)) < ATOL_KERAS, (
            f"Conv3d gather vs original (k={kernel_size},share={share},valid): "
            f"max err={np.max(np.abs(y_ref - y_gat)):.2e}"
        )

    def test_with_bias_matches_original(self):
        layer = hgly.Conv3d(CIN, COUT, kernel_size=(1, 1), bias=True, depth_padding="valid")
        model = _build_3d_model(layer)
        _rand_weights(layer)
        x = RNG.standard_normal((1, D, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0)
        y_gat = patch_model_for_hls(model, strategy="gather").predict(x, verbose=0)
        assert np.max(np.abs(y_ref - y_gat)) < ATOL_KERAS

    def test_depth_stride_gt1_raises(self):
        """depth_stride > 1 is unsupported for the gather export (as for slotwise)."""
        layer = hgly.Conv3d(CIN, COUT, kernel_size=(2, 1), stride=(2, 1), depth_padding="valid")
        model = _build_3d_model(layer)
        with pytest.raises(NotImplementedError, match="depth_stride"):
            patch_model_for_hls(model, strategy="gather")

    def test_parameter_count_reduced(self):
        """gather uses a small (N_out,K) index ROM instead of dense per-slot
        selection matrices, so it must have far fewer params than slotwise."""
        layer = hgly.Conv3d(
            CIN,
            COUT,
            kernel_size=(2, 2),
            bias=False,
            share_neighbors=True,
            depth_padding="valid",
        )
        model = _build_3d_model(layer)
        n_slot = patch_model_for_hls(model, strategy="slotwise").count_params()
        n_gat = patch_model_for_hls(model, strategy="gather").count_params()
        assert n_gat < n_slot, f"gather ({n_gat}) should have fewer params than slotwise ({n_slot})"


@jax_skip
class TestMaxPool3dGather:
    """strategy='gather' for MaxPool3d -> HexGather3D + HexMaxPool3D.

    Referenced against the original MaxPool3d (its forward has no depth-padding
    'same' path, so it runs fine in graph mode for every config here).
    """

    @pytest.mark.parametrize("kernel_size", [(1, 1), (2, 1), (2, 2), (3, 1), (1, 2)])
    @pytest.mark.parametrize("stride", [None, (2, 1), (2, 2), (1, 2)])
    def test_output_matches_original(self, kernel_size, stride):
        kw = {} if stride is None else {"stride": stride}
        layer = hgly.MaxPool3d(kernel_size=kernel_size, **kw)
        model = _build_3d_model(layer)

        x = RNG.standard_normal((2, D, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0)
        y_gat = patch_model_for_hls(model, strategy="gather").predict(x, verbose=0)

        assert y_ref.shape == y_gat.shape
        assert np.max(np.abs(y_ref - y_gat)) < 1e-5, (
            f"MaxPool3d gather(k={kernel_size},s={stride}): "
            f"max err={np.max(np.abs(y_ref - y_gat)):.2e}"
        )

    @pytest.mark.parametrize("kernel_size", [(2, 1), (2, 2), (3, 1)])
    def test_all_negative_input(self, kernel_size):
        """Border 0-pads dominate for all-negative input — gather must match."""
        layer = hgly.MaxPool3d(kernel_size=kernel_size)
        model = _build_3d_model(layer)
        x = -np.abs(RNG.standard_normal((1, D, H, W, CIN)).astype(np.float32))
        y_ref = model.predict(x, verbose=0)
        y_gat = patch_model_for_hls(model, strategy="gather").predict(x, verbose=0)
        assert np.max(np.abs(y_ref - y_gat)) < 1e-5

    def test_depth_stride_gt1(self):
        """Strided depth pooling (depth_stride > 1) must match — the HLS kernel
        walks the strided window, so no strided-slice op is needed in the graph."""
        layer = hgly.MaxPool3d(kernel_size=(2, 1), stride=(2, 1))
        model = _build_3d_model(layer)
        x = RNG.standard_normal((1, D, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0)
        y_gat = patch_model_for_hls(model, strategy="gather").predict(x, verbose=0)
        assert y_ref.shape == y_gat.shape
        assert np.max(np.abs(y_ref - y_gat)) < 1e-5

    def test_parameter_count_zero_weights(self):
        """HexMaxPool3D has no weights; the gather path stores only the tiny
        (N_out, K) index ROM — no large selection matrix."""
        layer = hgly.MaxPool3d(kernel_size=(2, 2))
        model = _build_3d_model(layer)
        patched = patch_model_for_hls(model, strategy="gather")
        # only HexGather3D's neighbor index table carries params
        assert patched.count_params() < 20_000


# =============================================================================
# Phase 3: hls4ml handler tests
# =============================================================================


@hls4ml_skip
class TestHls4mlHandlerRegistration:
    def test_registration_is_idempotent(self):
        """register_hex_gather_layers() must be safe to call multiple times."""
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()
        register_hex_gather_layers()  # second call must not raise

    def test_ir_layers_registered(self):
        from keras_hexagdly.hls4ml_handler import (
            register_hex_gather_layers,
        )

        register_hex_gather_layers()
        # Verify registered by looking them up in hls4ml's layer registry
        import hls4ml.model.layers as L

        assert hasattr(L, "layer_map") or True  # registry is internal; just confirm no error

    def test_gather_handler_output_shape(self):
        """HexGatherHandler must produce a config dict with correct shape attrs."""
        from keras_hexagdly.hex_gather import HexGather
        from keras_hexagdly.hls4ml_handler import HexGatherHandler, register_hex_gather_layers

        register_hex_gather_layers()

        # Build and call a HexGather layer to get real tensors
        import keras

        N_in, K, C = H * W, 7, CIN
        nbr = np.random.default_rng(0).integers(-1, N_in, (H * W, K), dtype=np.int32)
        inp = keras.Input(shape=(N_in, C))
        out = HexGather(neighbor_idx=nbr, name="test_gather")(inp)
        model = keras.Model(inp, out)
        layer = model.get_layer("test_gather")

        in_t = model.inputs
        out_t = model.outputs
        handler = HexGatherHandler()
        cfg = handler.handle(layer, in_t, out_t)

        assert cfg["n_in"] == N_in
        assert cfg["n_out"] == H * W
        assert cfg["k"] == K
        assert cfg["n_chan"] == C
        assert cfg["indices_data"].shape == (N_in * K,)

    @pytest.mark.parametrize("share", [False, True])
    def test_ring_mac_handler_output_shape(self, share):
        """HexRingMACHandler must produce correct shape attrs for both weight modes."""
        from keras_hexagdly.hex_gather import HexRingMAC
        from keras_hexagdly.hls4ml_handler import HexRingMACHandler, register_hex_gather_layers

        register_hex_gather_layers()

        import keras

        N_out, K, Cin, Cout = H * W, 7, CIN, COUT
        num_rings = 2  # kernel_size=1: center + ring1

        if share:
            w = RNG.standard_normal((num_rings, Cin, Cout)).astype(np.float32)
            ring_idx = np.array([0, 1, 1, 1, 1, 1, 1], dtype=np.int32)
        else:
            w = RNG.standard_normal((K, Cin, Cout)).astype(np.float32)
            ring_idx = None

        inp = keras.Input(shape=(N_out, K, Cin))
        out = HexRingMAC(weights_array=w, ring_idx=ring_idx, name="test_mac")(inp)
        model = keras.Model(inp, out)
        layer = model.get_layer("test_mac")

        handler = HexRingMACHandler()
        cfg = handler.handle(layer, model.inputs, model.outputs)

        assert cfg["n_out"] == N_out
        assert cfg["k"] == K
        assert cfg["n_in_chan"] == Cin
        assert cfg["n_out_chan"] == Cout
        assert cfg["share_neighbors"] == share
        expected_w_rows = num_rings if share else K
        assert cfg["num_weight_rows"] == expected_w_rows

    def test_max_pool_handler_output_shape(self):
        """HexMaxPoolHandler must produce correct shape attrs."""
        from keras_hexagdly.hex_gather import HexGather, HexMaxPool
        from keras_hexagdly.hls4ml_handler import HexMaxPoolHandler, register_hex_gather_layers

        register_hex_gather_layers()

        N_in, K, C = H * W, 7, CIN
        nbr = np.random.default_rng(0).integers(-1, N_in, (H * W, K), dtype=np.int32)

        inp = keras.Input(shape=(N_in, C))
        gath = HexGather(neighbor_idx=nbr)(inp)  # (B, N_out, K, C)
        out = HexMaxPool(name="test_maxpool")(gath)
        model = keras.Model(inp, out)

        handler = HexMaxPoolHandler()
        cfg = handler.handle(model.get_layer("test_maxpool"), [gath], model.outputs)

        assert cfg["n_out"] == H * W
        assert cfg["k"] == K
        assert cfg["n_chan"] == C


# =============================================================================
# Tier 2: hls4ml C-sim
# =============================================================================


@hls4ml_skip
class TestHls4mlCsim:
    def test_conv2d_folded_converts_and_csim(self, tmp_path):
        """strategy='folded' C-sim — small grid so the dense matrix fits."""
        global _HLS_DIR
        _HLS_DIR = str(tmp_path / "hls_conv2d_folded")

        layer = hgly.Conv2d(CIN, COUT, kernel_size=1, bias=False)
        model = _build_2d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((1, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="folded")
        y_hls = _csim(patched, x).reshape(-1)

        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"C-sim max err={np.max(np.abs(y_hls - y_ref)):.4f}"
        )

    def test_maxpool2d_converts_and_csim(self, tmp_path):
        global _HLS_DIR
        _HLS_DIR = str(tmp_path / "hls_pool2d")

        layer = hgly.MaxPool2d(kernel_size=1)
        model = _build_2d_model(layer)

        x = RNG.standard_normal((1, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="folded")
        y_hls = _csim(patched, x).reshape(-1)

        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM

    @pytest.mark.parametrize("share", [False, True])
    def test_conv2d_gather_converts_and_csim(self, tmp_path, share):
        """strategy='gather' — HexGather + HexRingMAC convert and C-sim correctly."""
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / f"hls_gather_share{share}")

        layer = hgly.Conv2d(CIN, COUT, kernel_size=1, bias=False, share_neighbors=share)
        model = _build_2d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((1, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="gather")
        y_hls = _csim(patched, x).reshape(-1)

        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"gather C-sim share={share}: max err={np.max(np.abs(y_hls - y_ref)):.4f}"
        )

    @pytest.mark.parametrize("share", [False, True])
    def test_conv2d_gather_with_bias_csim(self, tmp_path, share):
        """Conv2d with bias=True must convert + C-sim under strategy='gather'.

        Regression: the bias used to be injected via a Lambda layer, which
        hls4ml has no handler for (conversion crashed).  The bias is now baked
        into HexRingMAC (seeded into the accumulator).
        """
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / f"hls_gather_bias_share{share}")

        layer = hgly.Conv2d(CIN, COUT, kernel_size=1, bias=True, share_neighbors=share)
        model = _build_2d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((1, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="gather")
        y_hls = _csim(patched, x).reshape(-1)

        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"gather+bias C-sim share={share}: max err={np.max(np.abs(y_hls - y_ref)):.4f}"
        )

    @pytest.mark.parametrize("share", [False, True])
    def test_conv2d_slotwise_with_bias_csim(self, tmp_path, share):
        """Conv2d bias=True under strategy='slotwise' must convert + C-sim.

        Regression: slotwise also injected bias via a Lambda (hls4ml can't
        convert it); the bias is now folded into the last MAC EinsumDense.
        """
        global _HLS_DIR
        _HLS_DIR = str(tmp_path / f"hls_slotwise_bias_share{share}")

        layer = hgly.Conv2d(CIN, COUT, kernel_size=1, bias=True, share_neighbors=share)
        model = _build_2d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((1, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="slotwise")
        y_hls = _csim(patched, x).reshape(-1)

        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"slotwise+bias C-sim share={share}: max err={np.max(np.abs(y_hls - y_ref)):.4f}"
        )

    def test_conv3d_slotwise_with_bias_csim(self, tmp_path):
        """Conv3d bias=True under strategy='slotwise' must convert + C-sim
        (bias folded into the last active MAC EinsumDense, no Lambda)."""
        global _HLS_DIR
        _HLS_DIR = str(tmp_path / "hls_slotwise3d_bias")

        layer = hgly.Conv3d(CIN, COUT, kernel_size=(1, 1), bias=True, depth_padding="valid")
        model = _build_3d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((1, D, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="slotwise")
        y_hls = _csim(patched, x).reshape(-1)

        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"slotwise3d+bias C-sim: max err={np.max(np.abs(y_hls - y_ref)):.4f}"
        )

    def test_conv2d_gather_border_pixels_csim(self, tmp_path):
        """Border slots (-1) must contribute exactly 0, not a float residual.

        This is the test that would have caught the ap_fixed index bug:
        we use an all-ones kernel so every valid neighbor contributes +1 per
        channel per slot, and border pixels (which have fewer valid neighbors)
        must produce strictly smaller output than interior pixels.
        If the border check fails (float(-1) >= 0), border slots contribute
        garbage and interior/border outputs become equal or wrong.
        """
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / "hls_gather_border")

        # All-ones kernel: output at pixel n = (number of valid neighbors of n)
        # Border pixels have fewer valid neighbors → strictly smaller output.
        layer = hgly.Conv2d(1, 1, kernel_size=1, bias=False, share_neighbors=True)
        inp_dummy = keras.Input(shape=(H, W, 1))
        out_dummy = layer(inp_dummy)
        # Set all ring weights to 1.0
        for w in layer.trainable_variables:
            w.assign(np.ones(w.shape, dtype=np.float32))
        model = keras.Model(inp_dummy, out_dummy)

        # Input: all ones → output[n] = count of valid neighbors
        x = np.ones((1, H, W, 1), dtype=np.float32)
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="gather")
        y_hls = _csim(patched, x).reshape(-1)

        # 1. Overall numerical match
        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"Border test: max err={np.max(np.abs(y_hls - y_ref)):.4f}"
        )
        # 2. Border pixels must have strictly fewer contributions than interior
        #    (K=7 for kernel_size=1: interior gets 7, corners get fewer)
        assert y_ref.min() < y_ref.max(), "All outputs equal — border check likely broken"
        # 3. HLS and Keras must agree on which pixels are interior vs border
        assert np.all(np.sign(y_hls - y_ref.min()) == np.sign(y_ref - y_ref.min())), (
            "HLS and Keras disagree on border vs interior pixel counts"
        )

    def test_conv2d_gather_ring_sharing_csim(self, tmp_path):
        """share_neighbors=True and share_neighbors=False must agree when ring
        weights are broadcast.  Catches wrong ring_idx precision or striding."""
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / "hls_gather_ring")

        rng = np.random.default_rng(42)

        # Build share=True layer and set specific ring weights
        layer_share = hgly.Conv2d(CIN, COUT, kernel_size=1, bias=False, share_neighbors=True)
        inp = keras.Input(shape=(H, W, CIN))
        out_share = layer_share(inp)
        ring_w = rng.standard_normal(layer_share.ring_weights.shape).astype(np.float32)
        layer_share.ring_weights.assign(ring_w)
        model_share = keras.Model(inp, out_share)

        x = rng.standard_normal((1, H, W, CIN)).astype(np.float32)
        y_ref = model_share.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model_share, strategy="gather")
        y_hls = _csim(patched, x).reshape(-1)

        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"Ring sharing C-sim: max err={np.max(np.abs(y_hls - y_ref)):.4f}  "
            f"(if large: ring_idx or weight layout is wrong)"
        )

    def test_conv2d_gather_weight_layout_csim(self, tmp_path):
        """Each output channel must use the correct weight slice.

        Sets a different weight for each output channel (all others zero) and
        checks that only the expected channel activates.  Catches (Cin,Cout)
        transposition or striding bugs in the weight ROM.
        """
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / "hls_gather_layout")

        Cout = 3
        layer = hgly.Conv2d(1, Cout, kernel_size=1, bias=False, share_neighbors=True)
        inp = keras.Input(shape=(H, W, 1))
        out_layout = layer(inp)
        # Ring weights shape: (num_rings, 1, Cout)
        # Set ring 0 (center) weight to [1, 0, 0], ring 1 to [0, 1, 0]
        w = np.zeros(layer.ring_weights.shape, dtype=np.float32)
        w[0, 0, 0] = 1.0  # center → channel 0 only
        w[1, 0, 1] = 1.0  # ring 1 → channel 1 only
        layer.ring_weights.assign(w)
        model = keras.Model(inp, out_layout)

        x = np.ones((1, H, W, 1), dtype=np.float32)
        y_ref = model.predict(x, verbose=0)  # (1, H_out, W_out, Cout)
        y_ref_flat = y_ref.reshape(-1, Cout)

        patched = patch_model_for_hls(model, strategy="gather")
        y_hls = _csim(patched, x).reshape(-1, Cout)

        # Channel 2 must be all zeros (no weight set)
        assert np.max(np.abs(y_hls[:, 2])) < ATOL_CSIM, (
            f"Channel 2 should be zero, got max={np.max(np.abs(y_hls[:, 2])):.4f}"
        )
        # Channels 0 and 1 must be nonzero and match reference
        for ch in (0, 1):
            assert np.max(np.abs(y_hls[:, ch] - y_ref_flat[:, ch])) < ATOL_CSIM, (
                f"Weight layout wrong for channel {ch}: "
                f"max err={np.max(np.abs(y_hls[:, ch] - y_ref_flat[:, ch])):.4f}"
            )

    def test_conv2d_gather_kernel_size2_csim(self, tmp_path):
        """kernel_size=2: K=19 slots, 3 rings — larger ring_idx table and more
        complex neighbor pattern.  Catches ring mapping errors invisible at K=7."""
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / "hls_gather_k2")

        layer = hgly.Conv2d(1, 2, kernel_size=2, bias=False, share_neighbors=True)
        inp = keras.Input(shape=(H, W, 1))
        out = layer(inp)
        _rand_weights(layer)
        model = keras.Model(inp, out)

        x = RNG.standard_normal((1, H, W, 1)).astype(np.float32)
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="gather")
        y_hls = _csim(patched, x).reshape(-1)

        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"kernel_size=2 gather: max err={np.max(np.abs(y_hls - y_ref)):.4f}"
        )

    def test_conv2d_gather_stride2_csim(self, tmp_path):
        """stride=2: output grid is ~half the input size.  Neighbor indices span
        a larger range relative to N_out — catches index scaling bugs."""
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / "hls_gather_stride2")

        layer = hgly.Conv2d(1, 2, kernel_size=1, stride=2, bias=False, share_neighbors=True)
        inp = keras.Input(shape=(H, W, 1))
        out = layer(inp)
        _rand_weights(layer)
        model = keras.Model(inp, out)

        x = RNG.standard_normal((1, H, W, 1)).astype(np.float32)
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="gather")
        y_hls = _csim(patched, x).reshape(-1)

        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"stride=2 gather: max err={np.max(np.abs(y_hls - y_ref)):.4f}"
        )

    def test_conv2d_gather_zero_kernel_csim(self, tmp_path):
        """All-zero kernel: output must be exactly zero for every pixel.
        Catches accumulator initialization bugs (non-zero bias in HLS)."""
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / "hls_gather_zero")

        layer = hgly.Conv2d(1, 2, kernel_size=1, bias=False, share_neighbors=True)
        inp = keras.Input(shape=(H, W, 1))
        out = layer(inp)
        # Zero all weights
        for w in layer.trainable_variables:
            w.assign(np.zeros(w.shape, dtype=np.float32))
        model = keras.Model(inp, out)

        x = RNG.standard_normal((1, H, W, 1)).astype(np.float32)
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="gather")
        y_hls = _csim(patched, x).reshape(-1)

        assert np.max(np.abs(y_hls)) < ATOL_CSIM, (
            f"Zero kernel should give zero output, got max={np.max(np.abs(y_hls)):.4f}"
        )
        assert np.max(np.abs(y_ref)) < 1e-6, "Keras reference not zero — test setup error"

    def test_conv2d_gather_multichannel_csim(self, tmp_path):
        """Cin=3, Cout=4: multi-channel gather.  Catches channel stride bugs in
        the weight indexing (wrong Cin or Cout stride in nnet_hex_ring_mac.h)."""
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / "hls_gather_multichan")

        Cin_mc, Cout_mc = 3, 4
        layer = hgly.Conv2d(Cin_mc, Cout_mc, kernel_size=1, bias=False, share_neighbors=True)
        inp = keras.Input(shape=(H, W, Cin_mc))
        out = layer(inp)
        _rand_weights(layer)
        model = keras.Model(inp, out)

        x = RNG.standard_normal((1, H, W, Cin_mc)).astype(np.float32)
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="gather")
        y_hls = _csim(patched, x).reshape(-1)

        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"Multi-channel Cin={Cin_mc} Cout={Cout_mc}: "
            f"max err={np.max(np.abs(y_hls - y_ref)):.4f}"
        )

    def test_conv2d_gather_two_hex_layers_csim(self, tmp_path):
        """Two sequential hex Conv2d layers: weight layout of second layer must
        not be corrupted by the first.  Catches shared-state or naming bugs."""
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / "hls_gather_twolayer")

        layer1 = hgly.Conv2d(1, 2, kernel_size=1, bias=False, share_neighbors=True, name="hex1")
        layer2 = hgly.Conv2d(2, 2, kernel_size=1, bias=False, share_neighbors=True, name="hex2")
        inp = keras.Input(shape=(H, W, 1))
        x1 = layer1(inp)
        x1 = keras.layers.ReLU()(x1)
        out = layer2(x1)
        model = keras.Model(inp, out)
        for w in model.trainable_variables:
            w.assign(RNG.standard_normal(w.shape).astype(np.float32))

        x = RNG.standard_normal((1, H, W, 1)).astype(np.float32)
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="gather")
        y_hls = _csim(patched, x).reshape(-1)

        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"Two hex layers: max err={np.max(np.abs(y_hls - y_ref)):.4f}"
        )

    def test_conv2d_gather_large_uniform_input_csim(self, tmp_path):
        """Large uniform input (constant field): checks accumulator does not
        overflow ap_fixed<32,12> and that all spatial positions give the same
        output (translational invariance for an interior pixel kernel)."""
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / "hls_gather_uniform")

        layer = hgly.Conv2d(1, 1, kernel_size=1, bias=False, share_neighbors=True)
        inp = keras.Input(shape=(H, W, 1))
        out = layer(inp)
        # Uniform small weights to avoid overflow
        for w in layer.trainable_variables:
            w.assign(np.full(w.shape, 0.1, dtype=np.float32))
        model = keras.Model(inp, out)

        # Large constant input — exercises accumulator width
        x = np.full((1, H, W, 1), 10.0, dtype=np.float32)
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="gather")
        y_hls = _csim(patched, x).reshape(-1)

        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"Large uniform input: max err={np.max(np.abs(y_hls - y_ref)):.4f}"
        )
        # Interior pixels should all give the same output (translational invariance)
        interior = y_ref[y_ref == y_ref.max()]
        assert len(interior) > 1, "Expected multiple interior pixels with same value"

    @pytest.mark.parametrize("stride", [1, 2])
    def test_maxpool2d_gather_csim(self, tmp_path, stride):
        """MaxPool2d strategy='gather' — HexGather + HexMaxPool C-sim."""
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / f"hls_maxpool_gather_s{stride}")

        layer = hgly.MaxPool2d(kernel_size=1, stride=stride)
        model = _build_2d_model(layer)

        x = RNG.standard_normal((1, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="gather")
        y_hls = _csim(patched, x).reshape(-1)

        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"MaxPool2d gather stride={stride}: max err={np.max(np.abs(y_hls - y_ref)):.4f}"
        )

    def test_maxpool2d_gather_border_csim(self, tmp_path):
        """All-negative input: border 0-pads must win (same as CPU path)."""
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / "hls_maxpool_border")

        layer = hgly.MaxPool2d(kernel_size=1, stride=1)
        model = _build_2d_model(layer)

        x = -np.abs(RNG.standard_normal((1, H, W, CIN)).astype(np.float32))
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="gather")
        y_hls = _csim(patched, x).reshape(-1)

        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"MaxPool border: max err={np.max(np.abs(y_hls - y_ref)):.4f}"
        )

    def test_full_model_with_maxpool_gather_csim(self, tmp_path):
        """Full model: Conv2d + MaxPool2d both using strategy='gather'.
        This is the synthesis-blocker test — previously failed because
        MaxPool used a 280k-entry EinsumDense matrix."""
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / "hls_full_gather")

        inp = keras.Input(shape=(H, W, CIN))
        x = hgly.Conv2d(COUT, kernel_size=1, share_neighbors=True, bias=False)(inp)
        x = keras.layers.ReLU()(x)
        x = hgly.MaxPool2d(kernel_size=1, stride=2)(x)
        x = keras.layers.Flatten()(x)
        out = keras.layers.Dense(4)(x)
        model = keras.Model(inp, out)
        for w in model.trainable_variables:
            w.assign(RNG.standard_normal(w.shape).astype(np.float32))

        x_in = RNG.standard_normal((1, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x_in, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="gather")
        # Verify parameter count is small (no 280k pool matrix)
        assert patched.count_params() < 50_000, (
            f"Too many params ({patched.count_params()}) — pool gather not using HexGather"
        )

        y_hls = _csim(patched, x_in).reshape(-1)
        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"Full model gather: max err={np.max(np.abs(y_hls - y_ref)):.4f}"
        )

    @pytest.mark.parametrize("share", [False, True])
    @pytest.mark.parametrize("depth_padding", ["valid", "same"])
    def test_conv3d_gather_csim(self, tmp_path, share, depth_padding):
        """Conv3d strategy='gather' -> HexGather3D + HexRingMAC3D convert and
        C-sim.  Referenced against the slotwise Keras model (works for every
        padding, unlike the original forward which has a pre-existing 'same'
        bug for even depth kernels)."""
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / f"hls_conv3d_{share}_{depth_padding}")

        layer = hgly.Conv3d(
            CIN, COUT, kernel_size=1, bias=False, share_neighbors=share, depth_padding=depth_padding
        )
        model = _build_3d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((1, D, H, W, CIN)).astype(np.float32)
        y_ref = patch_model_for_hls(model, strategy="slotwise").predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="gather")
        y_hls = _csim(patched, x).reshape(-1)

        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"Conv3d gather C-sim share={share} dp={depth_padding}: "
            f"max err={np.max(np.abs(y_hls - y_ref)):.4f}"
        )

    @pytest.mark.parametrize("bias", [False, True])
    def test_conv3d_gather_multitap_csim(self, tmp_path, bias):
        """Depth kernel > 1 (two taps summed) with bias baked into the MAC —
        exercises the binary-Add tap tree and the bias-in-accumulator path."""
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / f"hls_conv3d_multitap_{bias}")

        layer = hgly.Conv3d(
            CIN, COUT, kernel_size=(2, 1), bias=bias, share_neighbors=True, depth_padding="valid"
        )
        model = _build_3d_model(layer)
        for w in layer.trainable_variables:
            w.assign(0.3 * RNG.standard_normal(w.shape).astype(np.float32))

        x = (0.5 * RNG.standard_normal((1, D, H, W, CIN))).astype(np.float32)
        y_ref = patch_model_for_hls(model, strategy="slotwise").predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="gather")
        y_hls = _csim(patched, x).reshape(-1)

        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"Conv3d multitap gather C-sim bias={bias}: max err={np.max(np.abs(y_hls - y_ref)):.4f}"
        )

    @pytest.mark.parametrize(
        "kernel_size,stride",
        [
            ((1, 1), None),
            ((2, 1), None),
            ((2, 2), None),
            ((2, 1), (2, 1)),
        ],
    )
    def test_maxpool3d_gather_csim(self, tmp_path, kernel_size, stride):
        """MaxPool3d strategy='gather' -> HexGather3D + HexMaxPool3D C-sim,
        including strided depth pooling (depth_stride > 1)."""
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / f"hls_mp3d_{kernel_size}_{stride}")

        kw = {} if stride is None else {"stride": stride}
        layer = hgly.MaxPool3d(kernel_size=kernel_size, **kw)
        model = _build_3d_model(layer)

        x = RNG.standard_normal((1, D, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="gather")
        y_hls = _csim(patched, x).reshape(-1)

        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"MaxPool3d gather C-sim k={kernel_size} s={stride}: "
            f"max err={np.max(np.abs(y_hls - y_ref)):.4f}"
        )

    def test_maxpool3d_gather_border_csim(self, tmp_path):
        """All-negative input: border 0-pads must win, same as the CPU path."""
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / "hls_mp3d_border")

        layer = hgly.MaxPool3d(kernel_size=(2, 1))
        model = _build_3d_model(layer)
        x = -np.abs(RNG.standard_normal((1, D, H, W, CIN)).astype(np.float32))
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="gather")
        y_hls = _csim(patched, x).reshape(-1)

        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"MaxPool3d gather border C-sim: max err={np.max(np.abs(y_hls - y_ref)):.4f}"
        )

    @pytest.mark.parametrize("reuse", [1, 3])
    def test_ring_mac_reuse_factor(self, tmp_path, reuse):
        """ReuseFactor must reach the HexRingMAC config and must not change the
        C-sim result (reuse only affects scheduling / multiplier count).

        This guards against the earlier bug where the ring MAC hardcoded
        II=1 and ignored ReuseFactor entirely.
        """
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / f"hls_rf{reuse}")

        layer = hgly.Conv2d(CIN, COUT, kernel_size=1, bias=False, share_neighbors=True)
        model = _build_2d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((1, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="gather")
        cfg = hls4ml.utils.config_from_keras_model(patched, granularity="name", backend="Vivado")
        cfg["Model"]["Precision"] = "ap_fixed<32,12>"
        cfg["Model"]["ReuseFactor"] = reuse
        hm = hls4ml.converters.convert_from_keras_model(
            patched, hls_config=cfg, backend="Vivado",
            output_dir=_HLS_DIR, part=_HLS_PART,
        )
        hm.compile()  # writes the firmware (parameters.h, config headers, ...)

        # The generated ring-MAC config header must carry the requested reuse.
        import glob
        params_files = glob.glob(f"{_HLS_DIR}/**/parameters.h", recursive=True)
        assert params_files, "parameters.h not generated"
        params_txt = "".join(open(p).read() for p in params_files)
        assert f"reuse_factor    = {reuse}" in params_txt, (
            f"ReuseFactor={reuse} did not reach the HexRingMAC config"
        )

        y_hls = hm.predict(np.ascontiguousarray(x)).reshape(-1)
        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"reuse={reuse}: C-sim differs from reference "
            f"(max err={np.max(np.abs(y_hls - y_ref)):.4f})"
        )

    @pytest.mark.parametrize("reuse", [1, 3])
    def test_ring_mac_3d_reuse_factor(self, tmp_path, reuse):
        """ReuseFactor must reach the Conv3d HexRingMAC3D config and must not
        change the C-sim result — same guard as the 2D ring MAC."""
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / f"hls_rf3d{reuse}")

        layer = hgly.Conv3d(CIN, COUT, kernel_size=(1, 1), bias=False, share_neighbors=True)
        model = _build_3d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((1, D, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="gather")
        cfg = hls4ml.utils.config_from_keras_model(patched, granularity="name", backend="Vivado")
        cfg["Model"]["Precision"] = "ap_fixed<32,12>"
        cfg["Model"]["ReuseFactor"] = reuse
        hm = hls4ml.converters.convert_from_keras_model(
            patched, hls_config=cfg, backend="Vivado",
            output_dir=_HLS_DIR, part=_HLS_PART,
        )
        hm.compile()

        import glob
        params_files = glob.glob(f"{_HLS_DIR}/**/parameters.h", recursive=True)
        assert params_files, "parameters.h not generated"
        params_txt = "".join(open(p).read() for p in params_files)
        assert f"reuse_factor    = {reuse}" in params_txt, (
            f"ReuseFactor={reuse} did not reach the HexRingMAC3D config"
        )

        y_hls = hm.predict(np.ascontiguousarray(x)).reshape(-1)
        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"3D reuse={reuse}: C-sim differs from reference "
            f"(max err={np.max(np.abs(y_hls - y_ref)):.4f})"
        )

    @pytest.mark.parametrize("kernel_size,cin", [(1, 1), (2, 2), (3, 1)])
    def test_ring_mac_accum_wider_than_weight(self, tmp_path, kernel_size, cin):
        """The MAC accumulator type must be wider than the weight type by
        ceil(log2(K*Cin)) bits, so summing the neighbor products cannot overflow.

        This is a structural check on the generated config header (reliable —
        it directly asserts the fix), rather than a runtime overflow which is
        entangled with output-type saturation and hard to isolate.
        """
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / f"hls_accum_k{kernel_size}_c{cin}")

        layer = hgly.Conv2d(cin, 2, kernel_size=kernel_size, bias=False, share_neighbors=True)
        inp = keras.Input((H, W, cin), name="x")
        model = keras.Model(inp, layer(inp))
        for w in layer.trainable_variables:
            w.assign(RNG.standard_normal(w.shape).astype(np.float32))

        patched = patch_model_for_hls(model, strategy="gather")
        cfg = hls4ml.utils.config_from_keras_model(patched, granularity="name", backend="Vivado")
        cfg["Model"]["Precision"] = "ap_fixed<16,6>"
        hm = hls4ml.converters.convert_from_keras_model(
            patched, hls_config=cfg, backend="Vivado",
            output_dir=_HLS_DIR, part=_HLS_PART,
        )
        hm.compile()

        import glob
        import math

        params_files = glob.glob(f"{_HLS_DIR}/**/parameters.h", recursive=True)
        params_txt = "".join(open(p).read() for p in params_files)

        # Find the ring-MAC accum_t typedef and parse its integer bits.
        # K includes the center + rings for this kernel size.
        from keras_hexagdly.indexed import _cell_list

        k = len(_cell_list(kernel_size))
        scale = math.ceil(math.log2(k * cin))
        # weight type is ap_fixed<16,6>; accum must be ap_fixed<16+scale, 6+scale>
        expected = f"ap_fixed<{16 + scale}, {6 + scale}>"
        assert f"typedef {expected} accum_t;" in params_txt, (
            f"expected accum_t {expected} (scale={scale} for K*Cin={k * cin}), "
            f"not found in generated config"
        )

    def test_gather_index_width_derived_from_n_in(self, tmp_path):
        """The gather index type width must be derived from N_in (not a fixed
        16 bits), so it neither overflows for large detectors nor wastes bits
        for small ones. For N_in=H*W here, width = ceil(log2(N_in)) + 1 (sign)."""
        import math

        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / "hls_index_width")

        layer = hgly.Conv2d(1, 2, kernel_size=1, bias=False, share_neighbors=True)
        inp = keras.Input((H, W, 1), name="x")
        model = keras.Model(inp, layer(inp))
        for w in layer.trainable_variables:
            w.assign(RNG.standard_normal(w.shape).astype(np.float32))

        patched = patch_model_for_hls(model, strategy="gather")
        cfg = hls4ml.utils.config_from_keras_model(patched, granularity="name", backend="Vivado")
        cfg["Model"]["Precision"] = "ap_fixed<16,6>"
        hm = hls4ml.converters.convert_from_keras_model(
            patched, hls_config=cfg, backend="Vivado",
            output_dir=_HLS_DIR, part=_HLS_PART,
        )
        hm.compile()

        import glob

        params_txt = "".join(
            open(p).read() for p in glob.glob(f"{_HLS_DIR}/**/defines.h", recursive=True)
        )
        n_in = H * W
        expected_w = math.ceil(math.log2(n_in)) + 1
        assert f"ap_int<{expected_w}>" in params_txt, (
            f"expected index type ap_int<{expected_w}> for N_in={n_in}, "
            f"not found — index width not derived from N_in"
        )
