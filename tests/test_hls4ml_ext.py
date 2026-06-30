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
# Tier 2: hls4ml C-sim
# =============================================================================

@hls4ml_skip
class TestHls4mlCsim:
    def test_conv2d_converts_and_csim(self, tmp_path):
        global _HLS_DIR
        _HLS_DIR = str(tmp_path / "hls_conv2d")

        layer = hgly.Conv2d(CIN, COUT, kernel_size=1, bias=False)
        model = _build_2d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((1, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model)
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

        patched = patch_model_for_hls(model)
        y_hls = _csim(patched, x).reshape(-1)

        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM
