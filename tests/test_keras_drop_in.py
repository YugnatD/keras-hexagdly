"""Tests for using the hex layers as drop-in replacements for the Keras ones.

Conv2d accepts ``padding`` and ``activation`` like ``keras.layers.Conv2D``,
MaxPool2d/AvgPool2d accept ``pool_size`` like ``keras.layers.MaxPool2D`` /
``AveragePooling2D``, so a model can pick either family of layers and build
them from one set of keyword arguments. See CHANGELOG.md [0.5.0].
"""

import warnings

import keras
import numpy as np
import pytest

import keras_hexagdly as hgly
from keras_hexagdly.hls4ml_ext import patch_model_for_hls

H, W, C = 9, 8, 3
RNG = np.random.default_rng(0)


@pytest.fixture
def x():
    return RNG.standard_normal((2, H, W, C)).astype("float32")


def _numpy(t):
    return keras.ops.convert_to_numpy(t)


# ---- activation ---------------------------------------------------------------


@pytest.mark.parametrize("activation", ["relu", "tanh"])
def test_conv2d_activation_is_applied_to_the_conv_output(x, activation):
    keras.utils.set_random_seed(0)
    plain = hgly.Conv2d(4, kernel_size=1)
    plain.build(x.shape)
    fused = hgly.Conv2d(4, kernel_size=1, activation=activation)
    fused.build(x.shape)
    fused.set_weights(plain.get_weights())

    expected = _numpy(keras.activations.get(activation)(plain(x)))
    np.testing.assert_allclose(_numpy(fused(x)), expected, atol=1e-6)


def test_conv2d_without_activation_is_unchanged(x):
    layer = hgly.Conv2d(4, kernel_size=1)
    assert layer.activation is keras.activations.linear
    assert layer.get_config()["activation"] == "linear"


def test_conv3d_activation_is_applied():
    x3 = RNG.standard_normal((2, 4, H, W, C)).astype("float32")
    keras.utils.set_random_seed(0)
    plain = hgly.Conv3d(4, kernel_size=1)
    plain.build(x3.shape)
    fused = hgly.Conv3d(4, kernel_size=1, activation="relu")
    fused.build(x3.shape)
    fused.set_weights(plain.get_weights())
    np.testing.assert_allclose(_numpy(fused(x3)), np.maximum(_numpy(plain(x3)), 0), atol=1e-6)


def test_activation_survives_save_load(x, tmp_path):
    """A model reloaded from disk must still apply the activation -- losing it
    would change every prediction without raising anything."""
    model = keras.Sequential(
        [keras.layers.Input((H, W, C)), hgly.Conv2d(4, kernel_size=1, activation="relu")]
    )
    path = tmp_path / "model.keras"
    model.save(path)
    reloaded = keras.saving.load_model(path)
    assert reloaded.layers[0].get_config()["activation"] == "relu"
    np.testing.assert_allclose(
        model.predict(x, verbose=0), reloaded.predict(x, verbose=0), atol=1e-6
    )


# ---- padding ------------------------------------------------------------------


@pytest.mark.parametrize("layer_cls", [hgly.Conv2d, hgly.MaxPool2d, hgly.AvgPool2d])
def test_padding_same_is_accepted(layer_cls):
    args = (4,) if layer_cls is hgly.Conv2d else ()
    for padding in ("same", "SAME"):
        assert layer_cls(*args, padding=padding).get_config()["padding"] == "same"


@pytest.mark.parametrize("layer_cls", [hgly.Conv2d, hgly.MaxPool2d, hgly.AvgPool2d])
def test_padding_other_than_same_is_rejected(layer_cls):
    args = (4,) if layer_cls is hgly.Conv2d else ()
    with pytest.raises(ValueError, match="same"):
        layer_cls(*args, padding="valid")


# ---- pool_size ----------------------------------------------------------------


@pytest.mark.parametrize("layer_cls", [hgly.MaxPool2d, hgly.AvgPool2d])
@pytest.mark.parametrize("strides", [1, 2])
def test_pool_size_and_kernel_size_are_the_same_argument(x, layer_cls, strides):
    by_pool_size = _numpy(layer_cls(pool_size=2, strides=strides)(x))
    by_kernel_size = _numpy(layer_cls(kernel_size=2, strides=strides)(x))
    by_position = _numpy(layer_cls(2, strides=strides)(x))
    np.testing.assert_array_equal(by_pool_size, by_kernel_size)
    np.testing.assert_array_equal(by_pool_size, by_position)


@pytest.mark.parametrize("layer_cls", [hgly.MaxPool2d, hgly.AvgPool2d, hgly.MaxPool3d])
def test_pool_size_and_kernel_size_together_is_an_error(layer_cls):
    with pytest.raises(ValueError, match="either"):
        layer_cls(pool_size=1, kernel_size=1)


def test_maxpool_config_uses_pool_size_but_old_configs_still_load(x):
    layer = hgly.MaxPool2d(pool_size=2, strides=2)
    config = layer.get_config()
    assert config["pool_size"] == 2
    assert "kernel_size" not in config

    old_style = {k: v for k, v in config.items() if k != "pool_size"}
    old_style["kernel_size"] = 2  # how 0.4.x serialized it
    reloaded = hgly.MaxPool2d.from_config(old_style)
    np.testing.assert_array_equal(_numpy(reloaded(x)), _numpy(layer(x)))


def test_pool_positional_arguments_follow_the_keras_order(x):
    keyword = _numpy(hgly.MaxPool2d(pool_size=2, strides=2, padding="same")(x))
    positional = _numpy(hgly.MaxPool2d(2, 2, "same")(x))  # as keras.layers.MaxPool2D
    np.testing.assert_array_equal(positional, keyword)


def test_old_positional_name_on_a_pool_fails_loudly():
    """0.4.x took ``name`` third; that slot is now ``padding``, so an old call
    raises instead of silently misreading the name."""
    with pytest.raises(ValueError, match="padding"):
        hgly.MaxPool2d(2, 2, "pool1")


@pytest.mark.parametrize("layer_cls", [hgly.Conv2d, hgly.Conv3d])
def test_keras_style_positional_padding_on_a_conv_fails_loudly(layer_cls):
    """keras.layers.Conv2D takes ``padding`` fourth; here the fourth argument
    is ``use_bias``, so a Keras-style positional call must not read "same" as
    a truthy use_bias."""
    with pytest.raises(TypeError, match="use_bias"):
        layer_cls(8, 1, 1, "same")


def test_numpy_bool_flags_are_accepted_and_stored_as_plain_bools():
    config = hgly.Conv2d(4, use_bias=np.bool_(False), debug=np.bool_(False)).get_config()
    assert config["use_bias"] is False
    assert config["debug"] is False


def test_maxpool3d_accepts_pool_size():
    layer = hgly.MaxPool3d(pool_size=(1, 1), strides=(1, 1))
    assert layer.get_config()["pool_size"] == (1, 1)


# ---- drop-in ------------------------------------------------------------------


@pytest.mark.parametrize(
    "conv_cls, max_cls, avg_cls",
    [
        (keras.layers.Conv2D, keras.layers.MaxPool2D, keras.layers.AveragePooling2D),
        (hgly.Conv2d, hgly.MaxPool2d, hgly.AvgPool2d),
    ],
)
def test_same_keyword_arguments_build_either_family(x, conv_cls, max_cls, avg_cls):
    """The pattern a model uses to switch backend: pick the classes, build
    them with one set of keyword arguments, no per-backend branching."""
    conv_kwargs = dict(filters=4, kernel_size=1, strides=1, padding="same", activation="relu")
    pool_kwargs = dict(pool_size=1, strides=2)

    inp = keras.Input((H, W, C))
    y = conv_cls(name="conv", **conv_kwargs)(inp)
    y = max_cls(name="max", **pool_kwargs)(y)
    y = avg_cls(name="avg", **pool_kwargs)(y)
    model = keras.Model(inp, y)

    out = model.predict(x, verbose=0)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0)  # relu, and pooling preserves the sign


# ---- grids no wider than the stride ---------------------------------------------


@pytest.mark.parametrize(
    "make_layer",
    [
        lambda k, s: hgly.Conv2d(4, kernel_size=k, strides=s),
        lambda k, s: hgly.MaxPool2d(k, strides=s),
    ],
    ids=["conv2d", "maxpool2d"],
)
@pytest.mark.parametrize("width, strides", [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3)])
@pytest.mark.parametrize("kernel_size", [1, 2])
def test_grid_no_wider_than_the_stride(make_layer, width, strides, kernel_size):
    """With width <= stride the even-column group of the output is empty.
    Computing it anyway crashed on the TensorFlow and torch backends, as it
    does in upstream HexagDLy. The single output column must match the first
    output column of the same grid widened with zero columns, which takes the
    ordinary path."""
    narrow = RNG.standard_normal((2, 7, width, 3)).astype("float32")
    wide = np.concatenate([narrow, np.zeros((2, 7, 2 * strides + 1, 3), "float32")], axis=2)
    layer = make_layer(kernel_size, strides)
    narrow_out = _numpy(layer(narrow))
    wide_out = _numpy(layer(wide))
    assert narrow_out.shape[2] == 1
    np.testing.assert_allclose(narrow_out[:, :, 0], wide_out[:, :, 0], atol=1e-5)


@pytest.mark.parametrize("width, strides", [(1, 1), (2, 2)])
def test_conv3d_grid_no_wider_than_the_stride(width, strides):
    x3 = RNG.standard_normal((2, 3, 7, width, 3)).astype("float32")
    out = _numpy(hgly.Conv3d(4, kernel_size=(1, 2), strides=(1, strides))(x3))
    assert out.shape[3] == 1
    assert np.all(np.isfinite(out))


# ---- AvgPool2d ----------------------------------------------------------------


@pytest.mark.parametrize("pool_size", [1, 2])
@pytest.mark.parametrize("strides", [1, 2])
def test_avgpool_of_a_constant_is_that_constant_everywhere(pool_size, strides):
    """Holds at the grid border only if the divisor counts exactly the in-grid
    cells of each window."""
    const = np.full((1, H, W, 2), 3.5, "float32")
    out = _numpy(hgly.AvgPool2d(pool_size, strides=strides)(const))
    np.testing.assert_allclose(out, 3.5, rtol=1e-6)


@pytest.mark.parametrize("pool_size", [1, 2, 3])
@pytest.mark.parametrize("width, strides", [(3, 3), (2, 2), (1, 2), (4, 3)])
def test_avgpool_on_a_narrow_grid_with_a_large_stride(pool_size, width, strides):
    """Narrow grids with a large stride leave one column group empty or
    nearly so. Summing with keras.ops.add failed there on the TensorFlow
    backend (rewritten as bias_add); MaxPool2d never had the problem."""
    const = np.full((1, 7, width, 1), 2.0, "float32")
    out = _numpy(hgly.AvgPool2d(pool_size, strides=strides)(const))
    assert out.shape == _numpy(hgly.MaxPool2d(pool_size, strides=strides)(const)).shape
    np.testing.assert_allclose(out, 2.0, rtol=1e-6)


def _ones_custom_kernel(pool_size, strides):
    """Conv2d_CustomKernel summing a single channel over the hex window -- an
    independent path to the same window sums (conv rather than slicing)."""
    sub_kernels = [
        np.ones((1, 1, 1 + 2 * pool_size - i, 1 if i == 0 else 2), "float32")
        for i in range(pool_size + 1)
    ]
    return hgly.Conv2d_CustomKernel(sub_kernels=sub_kernels, strides=strides)


@pytest.mark.parametrize("pool_size", [1, 2])
@pytest.mark.parametrize("strides", [1, 2])
def test_avgpool_matches_window_sum_over_window_count(x, pool_size, strides):
    window_sum = _ones_custom_kernel(pool_size, strides)
    count = _numpy(window_sum(np.ones((1, H, W, 1), "float32")))

    expected = np.concatenate(
        [_numpy(window_sum(x[..., c : c + 1])) / count for c in range(C)], axis=-1
    )
    out = _numpy(hgly.AvgPool2d(pool_size, strides=strides)(x))
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_avgpool_interior_cell_averages_its_seven_cells():
    """Hand check: a unit impulse is shared by the 7 windows it belongs to."""
    impulse = np.zeros((1, H, W, 1), "float32")
    impulse[0, 4, 4, 0] = 1.0
    out = _numpy(hgly.AvgPool2d(1)(impulse))[0, :, :, 0]
    assert np.isclose(out[4, 4], 1 / 7)
    assert np.count_nonzero(out) == 7
    np.testing.assert_allclose(out[out != 0], 1 / 7, rtol=1e-6)


def test_avgpool_trains_and_roundtrips(x, tmp_path):
    model = keras.Sequential(
        [
            keras.layers.Input((H, W, C)),
            hgly.Conv2d(4, kernel_size=1, activation="relu"),
            hgly.AvgPool2d(pool_size=1, strides=2),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    history = model.fit(x, np.ones((2, 1), "float32"), epochs=1, verbose=0)
    assert np.isfinite(history.history["loss"][0])

    path = tmp_path / "model.keras"
    model.save(path)
    reloaded = keras.saving.load_model(path)
    np.testing.assert_allclose(model.predict(x, verbose=0), reloaded.predict(x, verbose=0))


# ---- hls4ml export --------------------------------------------------------------

jax_skip = pytest.mark.skipif(
    keras.backend.backend() == "jax",
    reason="the linebuffer export layers are unsupported on the jax backend",
)


@jax_skip
@pytest.mark.parametrize("strategy", ["linebuffer", "folded"])
@pytest.mark.parametrize("strides", [1, 2])
def test_hls_export_keeps_the_conv_activation(strategy, strides):
    """The replacements are rebuilt from weights only; without re-applying
    the activation the exported model would silently differ."""
    layer = hgly.Conv2d(3, kernel_size=1, strides=strides, activation="relu")
    inp = keras.Input((13, 11, 2))
    model = keras.Model(inp, layer(inp))
    for w in layer.trainable_variables:
        w.assign(RNG.standard_normal(w.shape).astype("float32"))

    xs = RNG.standard_normal((2, 13, 11, 2)).astype("float32")
    y_ref = model.predict(xs, verbose=0)
    assert np.any(y_ref == 0)  # the relu really does clip something here

    patched = patch_model_for_hls(model, strategy=strategy)
    assert any(isinstance(l, keras.layers.Activation) for l in patched.layers)
    np.testing.assert_allclose(patched.predict(xs, verbose=0), y_ref, atol=2e-4)


@jax_skip
def test_hls_export_adds_nothing_without_activation():
    inp = keras.Input((13, 11, 2))
    model = keras.Model(inp, hgly.Conv2d(3, kernel_size=1)(inp))
    patched = patch_model_for_hls(model)
    assert not any(isinstance(l, keras.layers.Activation) for l in patched.layers)


def test_hls_export_refuses_avgpool():
    inp = keras.Input((13, 11, 2))
    model = keras.Model(inp, hgly.AvgPool2d(1)(inp))
    with pytest.raises(NotImplementedError, match="AvgPool2d"):
        patch_model_for_hls(model)


# ---- pointwise (kernel_size=0) -------------------------------------------------
#
# 0 rings is a plain per-cell channel projection: no neighbours at all. See
# CHANGELOG.md [0.6.0] and https://github.com/ctlearn-project/ctlearn/issues/302.


def test_conv2d_pointwise_matches_a_plain_dense_projection(x):
    """No neighbour mixing at all, independent of debug mode: every output
    cell is exactly that same cell's input vector times the (1x1) kernel,
    plus bias -- a manual per-pixel matmul, not run through the hex
    machinery at all."""
    layer = hgly.Conv2d(5, kernel_size=0)
    out = _numpy(layer(x))
    weight = _numpy(layer.kernel0)[0, 0]  # (in, out), squeezed from (1, 1, in, out)
    bias = _numpy(layer.bias_tensor)
    expected = x @ weight + bias
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_conv2d_pointwise_has_no_border_effect():
    """Unlike kernel_size>=1, a pointwise conv pads nothing, so debug weights
    (all 1s) give the exact channel sum at every cell, borders included --
    kernel_size>=1 would undercount at the border, where some neighbours
    fall outside the grid."""
    image = RNG.standard_normal((1, H, W, C)).astype("float32")
    out = _numpy(hgly.Conv2d(1, kernel_size=0, debug=True)(image))[..., 0]
    np.testing.assert_allclose(out, image.sum(axis=-1) + 1.0, atol=1e-5)


@pytest.mark.parametrize("strides", [1, 2, 3])
def test_conv2d_pointwise_strided_output_is_a_subset_of_the_input(strides):
    """A strided pointwise conv only ever subsamples: debug weights (all 1s,
    no bias) make every output value the channel sum of some single input
    cell, never a combination of several -- which a non-pointwise kernel, or
    a genuine spatial mixing bug, would produce instead."""
    image = RNG.standard_normal((1, H, W, C)).astype("float32")
    out = _numpy(hgly.Conv2d(1, kernel_size=0, strides=strides, use_bias=False, debug=True)(image))
    possible_sums = image.sum(axis=-1)
    for value in out.ravel():
        assert np.any(np.isclose(possible_sums, value, atol=1e-5))


def test_maxpool2d_pool_size_zero_is_the_identity_at_stride_one(x):
    """A single-cell window: the max of one element is that element."""
    np.testing.assert_allclose(_numpy(hgly.MaxPool2d(0)(x)), x)


@pytest.mark.parametrize("strides", [1, 2, 3])
def test_avgpool2d_pool_size_zero_is_the_identity(x, strides):
    """A single-cell window: the mean of one element is that element,
    whatever the stride (narrow-grid cases included, see
    test_avgpool_on_a_narrow_grid_with_a_large_stride)."""
    out = _numpy(hgly.AvgPool2d(0, strides=strides)(x))
    sampled = _numpy(hgly.MaxPool2d(0, strides=strides)(x))  # same window walk, exact subsample
    np.testing.assert_allclose(out, sampled, atol=1e-5)


def test_conv3d_pointwise_hex_element_only_affects_the_hex_plane():
    """kernel_size=(depth, 0): the hex plane is pointwise (no neighbour
    mixing), but the depth/time axis is untouched -- it still convolves
    across its 5 frames exactly as it would with a non-zero hex radius."""
    x3 = RNG.standard_normal((1, 5, H, W, C)).astype("float32")
    layer = hgly.Conv3d(4, kernel_size=(5, 0))
    out = _numpy(layer(x3))
    assert out.shape == (1, 1, H, W, 4)  # depth_padding="valid": 5 - 5 + 1 = 1
    assert np.all(np.isfinite(out))


# ---- kernel_size_unit="pixels" --------------------------------------------------
#
# "pixels" reads kernel_size as a square-kernel width (radius = size // 2),
# so the same architecture means a comparable receptive field on both grids.
# "rings" (the default) is unchanged: kernel_size already is the ring count.


def test_kernel_size_unit_defaults_to_rings():
    assert hgly.Conv2d(4, kernel_size=1).kernel_size_unit == "rings"
    assert hgly.MaxPool2d(1).kernel_size_unit == "rings"


def test_kernel_size_unit_rejects_an_unknown_value():
    with pytest.raises(ValueError, match="kernel_size_unit"):
        hgly.Conv2d(4, kernel_size=1, kernel_size_unit="bogus")


@pytest.mark.parametrize(
    "layer_cls, build",
    [
        (hgly.Conv2d, lambda cls, k, unit: cls(4, kernel_size=k, kernel_size_unit=unit)),
        (hgly.MaxPool2d, lambda cls, k, unit: cls(k, kernel_size_unit=unit)),
        (hgly.AvgPool2d, lambda cls, k, unit: cls(k, kernel_size_unit=unit)),
    ],
    ids=["conv2d", "maxpool2d", "avgpool2d"],
)
@pytest.mark.parametrize("pixels, rings", [(1, 0), (3, 1), (5, 2), (7, 3)])
def test_pixels_unit_resolves_to_the_matching_ring_count(layer_cls, build, pixels, rings):
    px = build(layer_cls, pixels, "pixels")
    assert px.hexbase_size == rings


@pytest.mark.parametrize("pixels", [2, 3])
def test_pixels_unit_even_width_is_the_window_of_the_next_odd_width(x, pixels):
    """pool_size=2 is CTLearn's default; a hexagonal window has no even size, so
    2 is rounded up to the 7-cell window of 3 (1 ring), not down to the single
    cell of 1."""
    a = hgly.Conv2d(4, kernel_size=1)  # 1 ring directly
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the warning is tested below
        b = hgly.Conv2d(4, kernel_size=pixels, kernel_size_unit="pixels")  # 1 ring too
    # Weights are drawn lazily, on the first call (build()) -- the seed has
    # to be reset right before each call, not before construction.
    keras.utils.set_random_seed(0)
    out_a = _numpy(a(x))
    keras.utils.set_random_seed(0)
    out_b = _numpy(b(x))
    np.testing.assert_allclose(out_a, out_b, atol=1e-6)


def test_pixels_unit_gives_bit_identical_output_to_the_matching_rings_call(x):
    """The whole point of the option: build the SAME layer either way, with
    identical weight initialization (same seed) and identical output -- not
    just a matching hexbase_size."""
    for size in (1, 2, 3):
        rings = hgly.Conv2d(4, kernel_size=size)
        pixels = hgly.Conv2d(4, kernel_size=2 * size + 1, kernel_size_unit="pixels")
        # Weights are drawn lazily, on the first call -- reset the seed right
        # before each call, not before construction.
        keras.utils.set_random_seed(1)
        out_rings = _numpy(rings(x))
        keras.utils.set_random_seed(1)
        out_pixels = _numpy(pixels(x))
        np.testing.assert_allclose(out_rings, out_pixels, atol=1e-6)


# ---- even sizes warn -----------------------------------------------------------
#
# A hexagonal window is centred on a cell, so it only exists for odd sizes. In
# "pixels" mode an even size is rounded up to the next odd one (2 -> 3, 4 -> 5)
# and a UserWarning says so; "rings" counts rings, which are never even or odd.

EVEN_LAYERS = [
    pytest.param(lambda k: hgly.Conv2d(4, kernel_size=k, kernel_size_unit="pixels"), id="Conv2d"),
    pytest.param(lambda k: hgly.MaxPool2d(k, kernel_size_unit="pixels"), id="MaxPool2d"),
    pytest.param(lambda k: hgly.AvgPool2d(k, kernel_size_unit="pixels"), id="AvgPool2d"),
    pytest.param(
        lambda k: hgly.Conv3d(4, kernel_size=(3, k), kernel_size_unit="pixels"), id="Conv3d"
    ),
    pytest.param(lambda k: hgly.MaxPool3d((3, k), kernel_size_unit="pixels"), id="MaxPool3d"),
]


@pytest.mark.parametrize("build", EVEN_LAYERS)
@pytest.mark.parametrize("size, rounded, cells", [(2, 3, 7), (4, 5, 19), (6, 7, 37)])
def test_even_size_in_pixels_mode_warns_with_the_rounded_size(build, size, rounded, cells):
    with pytest.warns(UserWarning, match="no even size") as record:
        build(size)
    (message,) = [str(w.message) for w in record]
    assert f"the size {size} is rounded up to {rounded}" in message
    assert f"{cells} cells" in message


@pytest.mark.parametrize("build", EVEN_LAYERS)
@pytest.mark.parametrize("size", [1, 3, 5, 7])
def test_odd_size_in_pixels_mode_does_not_warn(build, size):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build(size)


def test_pixels_zero_is_the_pointwise_case_not_a_rounding():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hgly.Conv2d(4, kernel_size=0, kernel_size_unit="pixels")
        hgly.MaxPool2d(0, kernel_size_unit="pixels")


@pytest.mark.parametrize("size", [0, 1, 2, 3, 4])
def test_rings_mode_never_warns(size):
    """In "rings" mode the size counts rings, so 2 is the 19-cell window: nothing
    is rounded."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hgly.Conv2d(4, kernel_size=size)
        hgly.MaxPool2d(size)
        hgly.AvgPool2d(size)


def test_the_warning_points_at_the_line_that_built_the_layer():
    with pytest.warns(UserWarning) as record:
        hgly.Conv2d(4, kernel_size=2, kernel_size_unit="pixels")
    assert record[0].filename == __file__


def test_conv3d_only_warns_about_the_hex_element():
    """The depth element is a number of frames, never a hexagonal window."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hgly.Conv3d(4, kernel_size=(2, 3), kernel_size_unit="pixels")  # even depth: fine
    with pytest.warns(UserWarning, match="rounded up to 3"):
        hgly.Conv3d(4, kernel_size=(3, 2), kernel_size_unit="pixels")


def test_conv3d_kernel_size_unit_leaves_the_depth_element_alone():
    """Only the hex element follows kernel_size_unit; depth is never a hex
    radius, so it always stays the literal value."""
    layer = hgly.Conv3d(4, kernel_size=(5, 3), kernel_size_unit="pixels")
    assert layer.depth_size == 5  # unaffected by the unit
    assert layer.hexbase_size == 1  # 3 pixels -> 1 ring


def test_conv3d_pixels_unit_gives_bit_identical_output_to_the_matching_rings_call():
    """Same idea as test_pixels_unit_gives_bit_identical_output_to_the_matching_
    rings_call, but for Conv3d, where the hex element shares the tuple with a
    non-trivial depth -- the equivalence has to hold with depth left alone."""
    x3 = RNG.standard_normal((2, 5, H, W, C)).astype("float32")
    keras.utils.set_random_seed(8)
    rings = hgly.Conv3d(4, kernel_size=(3, 1))
    out_rings = _numpy(rings(x3))
    keras.utils.set_random_seed(8)
    pixels = hgly.Conv3d(4, kernel_size=(3, 3), kernel_size_unit="pixels")  # 3 pixels -> 1 ring
    out_pixels = _numpy(pixels(x3))
    np.testing.assert_allclose(out_rings, out_pixels, atol=1e-6)


def test_maxpool3d_pixels_unit_matches_the_matching_rings_call():
    """MaxPool3d has no weights, so this is an output-only equivalence (no
    seeding needed), but it is the only check that the pixels-unit hex
    radius is actually threaded through MaxPool3d's own window geometry."""
    x3 = RNG.standard_normal((2, 5, H, W, C)).astype("float32")
    rings = hgly.MaxPool3d(pool_size=(3, 1))
    pixels = hgly.MaxPool3d(pool_size=(3, 3), kernel_size_unit="pixels")
    np.testing.assert_allclose(_numpy(rings(x3)), _numpy(pixels(x3)), atol=1e-6)


def test_kernel_size_unit_survives_save_load(x, tmp_path):
    keras.utils.set_random_seed(2)
    model = keras.Sequential([hgly.Conv2d(4, kernel_size=3, kernel_size_unit="pixels")])
    model.build((None, H, W, C))
    ref = _numpy(model(x))

    path = tmp_path / "model.keras"
    model.save(path)
    restored = keras.saving.load_model(path)

    assert restored.layers[0].kernel_size_unit == "pixels"
    assert restored.layers[0].hexbase_size == 1
    np.testing.assert_allclose(_numpy(restored(x)), ref, atol=1e-6)
