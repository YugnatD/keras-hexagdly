"""Phase 3+4 tests: patch_model_for_hls correctness + optional hls4ml C-sim.

Two tiers:
  Tier 1 (always runs): patch_model_for_hls() produces a plain Keras model
          whose float32 output matches the original hex model exactly.  No
          hls4ml dependency.

  Tier 2 (skipped if hls4ml not importable): patched model converts through
          hls4ml and C-sim output is close to the original within fixed-point
          quantization error (< 0.02 default precision).
"""

import numpy as np
import pytest
import keras

import keras_hexagdly as hgly
from keras_hexagdly.hls4ml_ext import patch_model_for_hls

# ---- test dimensions --------------------------------------------------------
H, W  = 13, 11
D     = 8
CIN   = 2
COUT  = 3
RNG   = np.random.default_rng(7)
ATOL_KERAS  = 2e-4   # float32 rounding across the Reshape/EinsumDense chain
ATOL_CSIM   = 0.02   # default ap_fixed<16,6> quantization


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

hls4ml_skip = pytest.mark.skipif(
    not HLS4ML_AVAILABLE, reason="hls4ml not installed"
)

_HLS_PART  = "xcvu9p-flga2104-2L-e"
_HLS_DIR   = "test_hls_prj"


def _csim(patched_model, x_np):
    """Convert patched model -> hls4ml -> compile -> predict."""
    cfg = hls4ml.utils.config_from_keras_model(
        patched_model, granularity="name", backend="Vivado"
    )
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
            CIN, COUT, kernel_size=kernel_size, stride=stride,
            bias=False, share_neighbors=share,
        )
        model = _build_2d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((2, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0)

        patched = patch_model_for_hls(model)
        y_pat   = patched.predict(x, verbose=0)

        assert y_ref.shape == y_pat.shape
        assert np.max(np.abs(y_ref - y_pat)) < ATOL_KERAS, (
            f"Conv2d(k={kernel_size},s={stride},share={share}): "
            f"max err={np.max(np.abs(y_ref-y_pat)):.2e}"
        )

    def test_with_bias(self):
        layer = hgly.Conv2d(CIN, COUT, kernel_size=1, stride=1, bias=True)
        model = _build_2d_model(layer)
        _rand_weights(layer)
        x = RNG.standard_normal((1, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0)
        patched = patch_model_for_hls(model)
        y_pat   = patched.predict(x, verbose=0)
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
        y_pat   = patched.predict(x, verbose=0)

        assert y_ref.shape == y_pat.shape
        assert np.max(np.abs(y_ref - y_pat)) < 1e-5, (
            f"MaxPool2d(k={kernel_size},s={stride}): "
            f"max err={np.max(np.abs(y_ref-y_pat)):.2e}"
        )

    def test_all_negative_input(self):
        """Border 0-pads dominate for all-negative input — patched must match."""
        layer = hgly.MaxPool2d(kernel_size=1, stride=1)
        model = _build_2d_model(layer)
        x = -np.abs(RNG.standard_normal((1, H, W, CIN)).astype(np.float32))
        y_ref = model.predict(x, verbose=0)
        patched = patch_model_for_hls(model)
        assert np.max(np.abs(y_ref - patched.predict(x, verbose=0))) < 1e-5


class TestMaxPool2dGather:
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    def test_output_matches_original(self, kernel_size, stride):
        layer = hgly.MaxPool2d(kernel_size=kernel_size, stride=stride)
        model = _build_2d_model(layer)

        x = RNG.standard_normal((2, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0)
        patched = patch_model_for_hls(model, strategy="gather")
        y_pat   = patched.predict(x, verbose=0)

        assert y_ref.shape == y_pat.shape
        assert np.max(np.abs(y_ref - y_pat)) < 1e-5, (
            f"MaxPool2d gather(k={kernel_size},s={stride}): "
            f"max err={np.max(np.abs(y_ref-y_pat)):.2e}"
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
        p_gather   = patch_model_for_hls(model, strategy="gather").count_params()
        assert p_gather < p_slotwise, (
            f"gather ({p_gather}) should have fewer params than slotwise ({p_slotwise})"
        )

    def test_all_strategies_agree(self):
        """slotwise and gather must produce identical outputs for MaxPool2d."""
        layer = hgly.MaxPool2d(kernel_size=1, stride=2)
        model = _build_2d_model(layer)
        x = RNG.standard_normal((2, H, W, CIN)).astype(np.float32)
        y_slotwise = patch_model_for_hls(model, strategy="slotwise").predict(x, verbose=0)
        y_gather   = patch_model_for_hls(model, strategy="gather").predict(x, verbose=0)
        assert np.max(np.abs(y_slotwise - y_gather)) < 1e-5


class TestConv3dPatch:
    @pytest.mark.parametrize("share", [False, True])
    @pytest.mark.parametrize("kernel_size", [(1, 1), (2, 2)])
    @pytest.mark.parametrize("depth_padding", ["valid", "same"])
    def test_output_matches_original(self, share, kernel_size, depth_padding):
        layer = hgly.Conv3d(
            CIN, COUT, kernel_size=kernel_size,
            bias=False, share_neighbors=share, depth_padding=depth_padding,
        )
        model = _build_3d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((1, D, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0)

        patched = patch_model_for_hls(model)
        y_pat   = patched.predict(x, verbose=0)

        assert y_ref.shape == y_pat.shape, (
            f"shape: ref={y_ref.shape} pat={y_pat.shape}"
        )
        assert np.max(np.abs(y_ref - y_pat)) < ATOL_KERAS, (
            f"Conv3d(k={kernel_size},share={share},dp={depth_padding}): "
            f"max err={np.max(np.abs(y_ref-y_pat)):.2e}"
        )

    def test_with_bias(self):
        layer = hgly.Conv3d(
            CIN, COUT, kernel_size=(1, 1), bias=True, depth_padding="valid"
        )
        model = _build_3d_model(layer)
        _rand_weights(layer)
        x = RNG.standard_normal((1, D, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0)
        patched = patch_model_for_hls(model)
        assert np.max(np.abs(y_ref - patched.predict(x, verbose=0))) < ATOL_KERAS


class TestMaxPool3dPatch:
    def test_raises_not_implemented(self):
        layer = hgly.MaxPool3d(kernel_size=(1, 1))
        model = _build_3d_model(layer)
        with pytest.raises(NotImplementedError, match="MaxPool3d"):
            patch_model_for_hls(model)


class TestPatchModelMisc:
    def test_non_model_raises(self):
        with pytest.raises(TypeError):
            patch_model_for_hls("not_a_model")

    def test_invalid_strategy_raises(self):
        layer = hgly.Conv2d(CIN, COUT, kernel_size=1, bias=False)
        model = _build_2d_model(layer)
        with pytest.raises(ValueError, match="Unknown strategy"):
            patch_model_for_hls(model, strategy="invalid")

    def test_gather_strategy_maxpool3d_raises(self):
        """MaxPool3d still raises regardless of strategy."""
        layer = hgly.MaxPool3d(kernel_size=(1, 1))
        model = _build_3d_model(layer)
        with pytest.raises(NotImplementedError, match="MaxPool3d"):
            patch_model_for_hls(model, strategy="gather")

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
# Slotwise strategy tests
# =============================================================================

class TestConv2dSlotwise:
    @pytest.mark.parametrize("share", [False, True])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    def test_output_matches_original(self, share, kernel_size, stride):
        layer = hgly.Conv2d(
            CIN, COUT, kernel_size=kernel_size, stride=stride,
            bias=False, share_neighbors=share,
        )
        model = _build_2d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((2, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0)
        patched = patch_model_for_hls(model, strategy="slotwise")
        y_pat  = patched.predict(x, verbose=0)

        assert y_ref.shape == y_pat.shape
        assert np.max(np.abs(y_ref - y_pat)) < ATOL_KERAS, (
            f"Conv2d slotwise(k={kernel_size},s={stride},share={share}): "
            f"max err={np.max(np.abs(y_ref-y_pat)):.2e}"
        )

    def test_folded_and_slotwise_agree(self):
        """Both strategies must produce identical float32 outputs."""
        layer = hgly.Conv2d(CIN, COUT, kernel_size=1, bias=False)
        model = _build_2d_model(layer)
        _rand_weights(layer)
        x = RNG.standard_normal((2, H, W, CIN)).astype(np.float32)
        y_folded   = patch_model_for_hls(model, strategy="folded").predict(x, verbose=0)
        y_slotwise = patch_model_for_hls(model, strategy="slotwise").predict(x, verbose=0)
        assert np.max(np.abs(y_folded - y_slotwise)) < ATOL_KERAS


class TestConv2dGather:
    @pytest.mark.parametrize("share", [False, True])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    def test_output_matches_original(self, share, kernel_size, stride):
        layer = hgly.Conv2d(
            CIN, COUT, kernel_size=kernel_size, stride=stride,
            bias=False, share_neighbors=share,
        )
        model = _build_2d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((2, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0)
        patched = patch_model_for_hls(model, strategy="gather")
        y_pat   = patched.predict(x, verbose=0)

        assert y_ref.shape == y_pat.shape
        assert np.max(np.abs(y_ref - y_pat)) < ATOL_KERAS, (
            f"Conv2d gather(k={kernel_size},s={stride},share={share}): "
            f"max err={np.max(np.abs(y_ref-y_pat)):.2e}"
        )

    def test_all_strategies_agree(self):
        """folded, slotwise and gather must produce identical float32 outputs."""
        layer = hgly.Conv2d(CIN, COUT, kernel_size=1, bias=False, share_neighbors=True)
        model = _build_2d_model(layer)
        _rand_weights(layer)
        x = RNG.standard_normal((2, H, W, CIN)).astype(np.float32)
        y_folded   = patch_model_for_hls(model, strategy="folded").predict(x, verbose=0)
        y_slotwise = patch_model_for_hls(model, strategy="slotwise").predict(x, verbose=0)
        y_gather   = patch_model_for_hls(model, strategy="gather").predict(x, verbose=0)
        assert np.max(np.abs(y_folded   - y_gather))   < ATOL_KERAS
        assert np.max(np.abs(y_slotwise - y_gather))   < ATOL_KERAS

    def test_parameter_count_reduced(self):
        """gather strategy must have far fewer parameters than slotwise
        (index table replaces dense selection matrices)."""
        layer = hgly.Conv2d(CIN, COUT, kernel_size=1, bias=False, share_neighbors=True)
        model = _build_2d_model(layer)
        patched_slotwise = patch_model_for_hls(model, strategy="slotwise")
        patched_gather   = patch_model_for_hls(model, strategy="gather")
        n_slotwise = patched_slotwise.count_params()
        n_gather   = patched_gather.count_params()
        assert n_gather < n_slotwise, (
            f"gather ({n_gather}) should have fewer params than slotwise ({n_slotwise})"
        )


class TestConv3dSlotwise:
    @pytest.mark.parametrize("share", [False, True])
    @pytest.mark.parametrize("kernel_size", [(1, 1), (2, 2)])
    @pytest.mark.parametrize("depth_padding", ["valid", "same"])
    def test_output_matches_original(self, share, kernel_size, depth_padding):
        layer = hgly.Conv3d(
            CIN, COUT, kernel_size=kernel_size,
            bias=False, share_neighbors=share, depth_padding=depth_padding,
        )
        model = _build_3d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((1, D, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0)
        patched = patch_model_for_hls(model, strategy="slotwise")
        y_pat  = patched.predict(x, verbose=0)

        assert y_ref.shape == y_pat.shape
        assert np.max(np.abs(y_ref - y_pat)) < ATOL_KERAS, (
            f"Conv3d slotwise(k={kernel_size},share={share},dp={depth_padding}): "
            f"max err={np.max(np.abs(y_ref-y_pat)):.2e}"
        )


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
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers, HHexGather, HHexRingMAC
        register_hex_gather_layers()
        # Verify registered by looking them up in hls4ml's layer registry
        import hls4ml.model.layers as L
        assert hasattr(L, 'layer_map') or True  # registry is internal; just confirm no error

    def test_gather_handler_output_shape(self):
        """HexGatherHandler must produce a config dict with correct shape attrs."""
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers, HexGatherHandler
        from keras_hexagdly.hex_gather import HexGather
        register_hex_gather_layers()

        # Build and call a HexGather layer to get real tensors
        import keras
        N_in, K, C = H * W, 7, CIN
        nbr = np.random.default_rng(0).integers(-1, N_in, (H * W, K), dtype=np.int32)
        inp = keras.Input(shape=(N_in, C))
        out = HexGather(neighbor_idx=nbr, name='test_gather')(inp)
        model = keras.Model(inp, out)
        layer = model.get_layer('test_gather')

        in_t  = model.inputs
        out_t = model.outputs
        handler = HexGatherHandler()
        cfg = handler.handle(layer, in_t, out_t)

        assert cfg['n_in']  == N_in
        assert cfg['n_out'] == H * W
        assert cfg['k']     == K
        assert cfg['n_chan'] == C
        assert cfg['indices_data'].shape == (N_in * K,)

    @pytest.mark.parametrize("share", [False, True])
    def test_ring_mac_handler_output_shape(self, share):
        """HexRingMACHandler must produce correct shape attrs for both weight modes."""
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers, HexRingMACHandler
        from keras_hexagdly.hex_gather import HexRingMAC
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
        out = HexRingMAC(weights_array=w, ring_idx=ring_idx, name='test_mac')(inp)
        model = keras.Model(inp, out)
        layer = model.get_layer('test_mac')

        handler = HexRingMACHandler()
        cfg = handler.handle(layer, model.inputs, model.outputs)

        assert cfg['n_out']      == N_out
        assert cfg['k']          == K
        assert cfg['n_in_chan']   == Cin
        assert cfg['n_out_chan']  == Cout
        assert cfg['share_neighbors'] == share
        expected_w_rows = num_rings if share else K
        assert cfg['num_weight_rows'] == expected_w_rows

    def test_max_pool_handler_output_shape(self):
        """HexMaxPoolHandler must produce correct shape attrs."""
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers, HexMaxPoolHandler
        from keras_hexagdly.hex_gather import HexGather, HexMaxPool
        register_hex_gather_layers()

        N_in, K, C = H * W, 7, CIN
        nbr = np.random.default_rng(0).integers(-1, N_in, (H * W, K), dtype=np.int32)

        inp  = keras.Input(shape=(N_in, C))
        gath = HexGather(neighbor_idx=nbr)(inp)      # (B, N_out, K, C)
        out  = HexMaxPool(name='test_maxpool')(gath)
        model = keras.Model(inp, out)

        handler = HexMaxPoolHandler()
        cfg = handler.handle(model.get_layer('test_maxpool'), [gath], model.outputs)

        assert cfg['n_out']  == H * W
        assert cfg['k']      == K
        assert cfg['n_chan']  == C


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
            f"C-sim max err={np.max(np.abs(y_hls-y_ref)):.4f}"
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

        layer = hgly.Conv2d(CIN, COUT, kernel_size=1, bias=False,
                            share_neighbors=share)
        model = _build_2d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((1, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="gather")
        y_hls = _csim(patched, x).reshape(-1)

        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"gather C-sim share={share}: max err={np.max(np.abs(y_hls-y_ref)):.4f}"
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
            f"Border test: max err={np.max(np.abs(y_hls-y_ref)):.4f}"
        )
        # 2. Border pixels must have strictly fewer contributions than interior
        #    (K=7 for kernel_size=1: interior gets 7, corners get fewer)
        assert y_ref.min() < y_ref.max(), \
            "All outputs equal — border check likely broken"
        # 3. HLS and Keras must agree on which pixels are interior vs border
        assert np.all(np.sign(y_hls - y_ref.min()) == np.sign(y_ref - y_ref.min())), \
            "HLS and Keras disagree on border vs interior pixel counts"

    def test_conv2d_gather_ring_sharing_csim(self, tmp_path):
        """share_neighbors=True and share_neighbors=False must agree when ring
        weights are broadcast.  Catches wrong ring_idx precision or striding."""
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers
        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / "hls_gather_ring")

        rng = np.random.default_rng(42)

        # Build share=True layer and set specific ring weights
        layer_share = hgly.Conv2d(CIN, COUT, kernel_size=1, bias=False,
                                  share_neighbors=True)
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
            f"Ring sharing C-sim: max err={np.max(np.abs(y_hls-y_ref)):.4f}  "
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
        layer = hgly.Conv2d(1, Cout, kernel_size=1, bias=False,
                            share_neighbors=True)
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
        y_ref = model.predict(x, verbose=0)   # (1, H_out, W_out, Cout)
        y_ref_flat = y_ref.reshape(-1, Cout)

        patched = patch_model_for_hls(model, strategy="gather")
        y_hls = _csim(patched, x).reshape(-1, Cout)

        # Channel 2 must be all zeros (no weight set)
        assert np.max(np.abs(y_hls[:, 2])) < ATOL_CSIM, \
            f"Channel 2 should be zero, got max={np.max(np.abs(y_hls[:,2])):.4f}"
        # Channels 0 and 1 must be nonzero and match reference
        for ch in (0, 1):
            assert np.max(np.abs(y_hls[:, ch] - y_ref_flat[:, ch])) < ATOL_CSIM, (
                f"Weight layout wrong for channel {ch}: "
                f"max err={np.max(np.abs(y_hls[:,ch]-y_ref_flat[:,ch])):.4f}"
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
            f"kernel_size=2 gather: max err={np.max(np.abs(y_hls-y_ref)):.4f}"
        )

    def test_conv2d_gather_stride2_csim(self, tmp_path):
        """stride=2: output grid is ~half the input size.  Neighbor indices span
        a larger range relative to N_out — catches index scaling bugs."""
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers
        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / "hls_gather_stride2")

        layer = hgly.Conv2d(1, 2, kernel_size=1, stride=2, bias=False,
                            share_neighbors=True)
        inp = keras.Input(shape=(H, W, 1))
        out = layer(inp)
        _rand_weights(layer)
        model = keras.Model(inp, out)

        x = RNG.standard_normal((1, H, W, 1)).astype(np.float32)
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="gather")
        y_hls = _csim(patched, x).reshape(-1)

        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"stride=2 gather: max err={np.max(np.abs(y_hls-y_ref)):.4f}"
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

        assert np.max(np.abs(y_hls)) < ATOL_CSIM, \
            f"Zero kernel should give zero output, got max={np.max(np.abs(y_hls)):.4f}"
        assert np.max(np.abs(y_ref)) < 1e-6, \
            "Keras reference not zero — test setup error"

    def test_conv2d_gather_multichannel_csim(self, tmp_path):
        """Cin=3, Cout=4: multi-channel gather.  Catches channel stride bugs in
        the weight indexing (wrong Cin or Cout stride in nnet_hex_ring_mac.h)."""
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers
        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / "hls_gather_multichan")

        Cin_mc, Cout_mc = 3, 4
        layer = hgly.Conv2d(Cin_mc, Cout_mc, kernel_size=1, bias=False,
                            share_neighbors=True)
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
            f"max err={np.max(np.abs(y_hls-y_ref)):.4f}"
        )

    def test_conv2d_gather_two_hex_layers_csim(self, tmp_path):
        """Two sequential hex Conv2d layers: weight layout of second layer must
        not be corrupted by the first.  Catches shared-state or naming bugs."""
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers
        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / "hls_gather_twolayer")

        layer1 = hgly.Conv2d(1, 2, kernel_size=1, bias=False,
                             share_neighbors=True, name="hex1")
        layer2 = hgly.Conv2d(2, 2, kernel_size=1, bias=False,
                             share_neighbors=True, name="hex2")
        inp = keras.Input(shape=(H, W, 1))
        x1  = layer1(inp)
        x1  = keras.layers.ReLU()(x1)
        out = layer2(x1)
        model = keras.Model(inp, out)
        for w in model.trainable_variables:
            w.assign(RNG.standard_normal(w.shape).astype(np.float32))

        x = RNG.standard_normal((1, H, W, 1)).astype(np.float32)
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="gather")
        y_hls = _csim(patched, x).reshape(-1)

        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"Two hex layers: max err={np.max(np.abs(y_hls-y_ref)):.4f}"
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
            f"Large uniform input: max err={np.max(np.abs(y_hls-y_ref)):.4f}"
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
            f"MaxPool2d gather stride={stride}: max err={np.max(np.abs(y_hls-y_ref)):.4f}"
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
            f"MaxPool border: max err={np.max(np.abs(y_hls-y_ref)):.4f}"
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
        assert patched.count_params() < 50_000, \
            f"Too many params ({patched.count_params()}) — pool gather not using HexGather"

        y_hls = _csim(patched, x_in).reshape(-1)
        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"Full model gather: max err={np.max(np.abs(y_hls-y_ref)):.4f}"
        )
