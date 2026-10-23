"""patch_model_for_hls: clear errors for models it cannot walk."""

import keras
import pytest

import keras_hexagdly as hgly
from keras_hexagdly.hls4ml_ext import patch_model_for_hls

pytest.importorskip("hls4ml")


def test_a_layer_already_used_in_another_model_gives_a_clear_error():
    """layer.input is the input of the layer's first graph node: reusing a layer object in a second
    model used to fail with a bare KeyError."""
    layer = hgly.Conv2d(3, kernel_size=1)
    first = keras.Input((9, 8, 2))
    keras.Model(first, layer(first))

    second = keras.Input((9, 8, 2))
    reused = keras.Model(second, layer(second))
    with pytest.raises(ValueError, match="not wired to this model's graph"):
        patch_model_for_hls(reused)


def test_a_model_built_once_from_fresh_layers_still_patches():
    inp = keras.Input((9, 8, 2))
    model = keras.Model(inp, hgly.Conv2d(3, kernel_size=1)(inp))
    patch_model_for_hls(model)


def test_the_export_layers_opt_out_of_automatic_xla_compilation_on_tensorflow_only():
    """Their int32 gather table lives on the CPU, which XLA cannot read from a GPU (Keras compiles
    predict()/fit() with XLA by itself there). The opt-out switches the compilation of the whole
    model off, so the other backends keep their jit (JAX was 4x slower without it)."""
    inp = keras.Input((9, 8, 2))
    model = keras.Model(inp, hgly.MaxPool2d(1)(hgly.Conv2d(3, kernel_size=1)(inp)))
    patched = patch_model_for_hls(model)
    exported = [layer for layer in patched.layers if hasattr(layer, "neighbor_idx")]
    assert len(exported) == 2
    expected = keras.backend.backend() != "tensorflow"
    assert all(layer.supports_jit is expected for layer in exported)
