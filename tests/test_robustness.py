"""Edge cases found by an audit of the layers: each test pins one of them."""

import warnings

import numpy as np
import pytest

keras = pytest.importorskip("keras")

import keras_hexagdly as hgly  # noqa: E402
from keras_hexagdly import layers as hex_layers  # noqa: E402
from keras_hexagdly.indexed import build_neighbor_table, build_parity_offset_tables  # noqa: E402
from keras_hexagdly.torch_interop import load_torch_state_dict  # noqa: E402


def test_numpy_integers_are_accepted_and_stored_as_plain_ints():
    conv = hgly.Conv2d(np.int64(4), np.int64(1), np.int32(2))
    pool = hgly.MaxPool2d(np.int64(1), np.int32(2))
    conv3d = hgly.Conv3d(np.int64(2), np.array([2, 1]))
    for layer in (conv, pool, hgly.AvgPool2d(np.int16(1)), conv3d):
        for value in layer.get_config().values():
            for item in value if isinstance(value, (tuple, list)) else (value,):
                assert not isinstance(item, np.integer)
    assert type(conv.kernel_size) is int and type(pool.strides) is int
    with pytest.raises(ValueError):
        hgly.Conv2d(4, True)
    with pytest.raises(ValueError):
        hgly.Conv2d(4, 1.0)


@pytest.mark.parametrize("pool", [hgly.MaxPool2d, hgly.AvgPool2d])
def test_a_pooling_on_a_dynamic_grid_says_so(pool):
    with pytest.raises(ValueError, match="statically-known spatial dimensions"):
        pool(1, 1)(keras.Input((None, 8, 2)))


def test_too_few_rows_for_the_stride_is_a_readable_error():
    with pytest.raises(ValueError, match="needs more than 1 row"):
        hgly.Conv2d(1, 1, 2)(np.zeros((1, 1, 5, 1), "float32"))
    # no even-column group on a grid no wider than the stride: one row is enough
    assert hgly.MaxPool2d(1, 2)(np.zeros((1, 1, 2, 1), "float32")).shape == (1, 1, 1, 1)


def test_the_neighbor_table_does_not_depend_on_the_dtype_policy():
    """Pixel indices travel as values through a probe layer: float16 rounds them above 2048."""
    reference = build_neighbor_table(hgly.MaxPool2d(1, 1), 60, 60)[0]
    hex_layers._TAP_OFFSET_CACHE.clear()  # the probes below must run, not be read from the cache
    expected_rings, _ = hex_layers._ring_maps_2d_uncached(2)
    policy = keras.mixed_precision.global_policy()
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        table = build_neighbor_table(hgly.MaxPool2d(1, 1), 60, 60)[0]
        hex_layers._TAP_OFFSET_CACHE.clear()
        rings, _ = hex_layers._ring_maps_2d_uncached(2)
    finally:
        keras.mixed_precision.set_global_policy(policy)
    np.testing.assert_array_equal(table, reference)
    for got, expected in zip(rings, expected_rings):
        np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("kernel", [0, 1, 2, 3])
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("height, width", [(3, 1), (4, 2), (5, 5), (7, 6), (9, 8)])
def test_max_pooling_equals_the_gather_over_its_neighbor_table(kernel, stride, height, width):
    """The reference definition: a maximum over the neighbours (0 outside the grid)."""
    if height <= stride // 2:
        pytest.skip("too few rows for this stride")
    layer = hgly.MaxPool2d(kernel, stride)
    x = np.random.default_rng(0).normal(size=(2, height, width, 3)).astype("float32")
    got = keras.ops.convert_to_numpy(layer(x))
    table = build_neighbor_table(hgly.MaxPool2d(kernel, stride), height, width)[0]
    flat = x.reshape(2, height * width, 3)
    gathered = np.where((table >= 0)[None, :, :, None], flat[:, np.maximum(table, 0)], 0.0)
    np.testing.assert_array_equal(got.reshape(2, -1, 3), gathered.max(axis=2))


def test_average_pooling_reused_on_two_grids():
    layer = hgly.AvgPool2d(1, 2)
    for height, width in ((7, 9), (6, 5), (7, 9)):
        x = np.random.default_rng(0).normal(size=(2, height, width, 3)).astype("float32")
        table = build_neighbor_table(hgly.MaxPool2d(1, 2), height, width)[0]
        flat = x.reshape(2, height * width, 3)
        valid = (table >= 0)[None, :, :, None]
        mean = np.where(valid, flat[:, np.maximum(table, 0)], 0.0).sum(axis=2) / valid.sum(axis=2)
        got = keras.ops.convert_to_numpy(layer(x)).reshape(2, -1, 3)
        np.testing.assert_allclose(got, mean, atol=1e-5)


@pytest.mark.parametrize(
    "height, width, kernel, stride",
    [(3, 3, 1, 1), (5, 5, 2, 1), (4, 5, 1, 2), (9, 8, 2, 3), (1, 1, 1, 1), (3, 1, 2, 2)],
)
def test_offset_tables_of_a_grid_too_small_to_measure_them(height, width, kernel, stride):
    table, _, (h_out, w_out) = build_neighbor_table(hgly.MaxPool2d(kernel, stride), height, width)
    even, odd, _ = build_parity_offset_tables(table, width, h_out, w_out, stride, H_in=height)
    big, _, (bh, bw) = build_neighbor_table(hgly.MaxPool2d(kernel, stride), 30, 30)
    big_even, big_odd, _ = build_parity_offset_tables(big, 30, bh, bw, stride)
    np.testing.assert_array_equal(even, big_even)
    np.testing.assert_array_equal(odd, big_odd)


def test_ring_maps_from_one_probe_per_kernel_size():
    for size, populations in ((1, [1, 6]), (2, [1, 6, 12]), (3, [1, 6, 12, 18])):
        maps, count = hex_layers._ring_maps_2d_uncached(size)
        assert count == size + 1
        rings = np.concatenate([m.reshape(-1) for m in maps])
        assert np.bincount(rings).tolist() == populations


def test_a_dense_after_a_flatten_is_loaded_in_the_keras_feature_order():
    """PyTorch flattens (C, H, W), Keras (H, W, C): a transpose of the weight is not enough."""
    inp = keras.Input((4, 3, 2))
    model = keras.Model(inp, keras.layers.Dense(5, name="fc")(keras.layers.Flatten()(inp)))
    rng = np.random.default_rng(0)
    weight = rng.normal(size=(5, 2 * 4 * 3)).astype("float32")  # torch: (out, C*H*W)
    bias = rng.normal(size=(5,)).astype("float32")
    load_torch_state_dict(model, {"fc.weight": weight, "fc.bias": bias}, {"fc": "fc"})
    x_nchw = rng.normal(size=(6, 2, 4, 3)).astype("float32")
    expected = x_nchw.reshape(6, -1) @ weight.T + bias
    got = model.predict(np.transpose(x_nchw, (0, 2, 3, 1)), verbose=0)
    np.testing.assert_allclose(got, expected, atol=1e-5)


def _dense_after(build):
    """A model whose Dense reads what `build(inp)` returns, and what PyTorch would compute."""
    shape = (4, 3, 2)
    inp = keras.Input(shape)
    model = keras.Model(inp, keras.layers.Dense(5, name="fc")(build(inp)))
    rng = np.random.default_rng(0)
    weight = rng.normal(size=(5, int(np.prod(shape)))).astype("float32")  # torch: (out, C*H*W)
    bias = rng.normal(size=(5,)).astype("float32")
    x = rng.normal(size=(6,) + shape).astype("float32")  # channels-last
    torch_order = np.moveaxis(x, -1, 1).reshape(6, -1) @ weight.T + bias
    keras_order = x.reshape(6, -1) @ weight.T + bias
    return model, {"fc.weight": weight, "fc.bias": bias}, x, torch_order, keras_order


def _nested_flatten(inp):
    inner_in = keras.Input((4, 3, 2))
    inner = keras.Model(inner_in, keras.layers.Flatten()(inner_in))
    return inner(inp)


def _flatten_then_nested_dropout(inp):
    inner_in = keras.Input((24,))
    inner = keras.Model(inner_in, keras.layers.Dropout(0.5)(inner_in))
    return inner(keras.layers.Flatten()(inp))


@pytest.mark.parametrize(
    "build",
    [
        lambda t: keras.layers.Dropout(0.5)(keras.layers.Flatten()(t)),
        lambda t: keras.layers.Activation("linear")(keras.layers.Flatten()(t)),
        lambda t: keras.layers.Reshape((-1,))(t),
        _nested_flatten,
        lambda t: keras.layers.Flatten()(keras.layers.Flatten()(t)),
        lambda t: keras.layers.Reshape((-1,))(keras.layers.Flatten()(t)),
        _flatten_then_nested_dropout,
    ],
    ids=[
        "dropout",
        "activation",
        "reshape",
        "nested_model",
        "flatten_twice",
        "flatten_reshape",
        "nested_after",
    ],
)
def test_the_flatten_is_found_behind_the_layers_between_it_and_the_dense(build):
    model, state, x, torch_order, keras_order = _dense_after(build)
    load_torch_state_dict(model, state, {"fc": "fc"})
    np.testing.assert_allclose(model.predict(x, verbose=0), torch_order, atol=1e-5)
    # a PyTorch model that permutes to channels-last itself before flattening
    load_torch_state_dict(model, state, {"fc": "fc"}, flatten_order="channels_last")
    np.testing.assert_allclose(model.predict(x, verbose=0), keras_order, atol=1e-5)


def test_flattened_features_concatenated_with_others_are_reordered_in_place():
    """CNN features next to auxiliary scalars: only the columns of the flattened part move."""
    inp, aux = keras.Input((4, 3, 2)), keras.Input((3,))
    merged = keras.layers.Concatenate()([aux, keras.layers.Flatten()(inp)])
    model = keras.Model([inp, aux], keras.layers.Dense(5, name="fc")(merged))
    rng = np.random.default_rng(0)
    weight = rng.normal(size=(5, 3 + 24)).astype("float32")
    bias = rng.normal(size=(5,)).astype("float32")
    load_torch_state_dict(model, {"fc.weight": weight, "fc.bias": bias}, {"fc": "fc"})
    x, a = (
        rng.normal(size=(6, 4, 3, 2)).astype("float32"),
        rng.normal(size=(6, 3)).astype("float32"),
    )
    torch_features = np.concatenate([a, np.moveaxis(x, -1, 1).reshape(6, -1)], axis=1)
    np.testing.assert_allclose(
        model.predict([x, a], verbose=0), torch_features @ weight.T + bias, atol=1e-5
    )


def test_a_dense_after_a_global_pooling_is_only_transposed():
    inp = keras.Input((4, 3, 2))
    pooled = keras.layers.GlobalAveragePooling2D()(inp)
    model = keras.Model(inp, keras.layers.Dense(5, name="fc")(pooled))
    weight = np.random.default_rng(0).normal(size=(5, 2)).astype("float32")
    state = {"fc.weight": weight, "fc.bias": np.zeros(5, "float32")}
    load_torch_state_dict(model, state, {"fc": "fc"})
    np.testing.assert_array_equal(model.get_layer("fc").get_weights()[0], weight.T)


def test_a_nan_goes_through_the_max_pooling():
    """TensorFlow's own max pooling drops it; a diverged activation must stay visible."""
    x = np.arange(1, 13, dtype="float32").reshape(1, 3, 4, 1)
    x[0, 1, 1, 0] = np.nan
    pooled = keras.ops.convert_to_numpy(hgly.MaxPool2d(1, 1)(x))
    assert np.isnan(pooled).sum() == 7  # the cell and its six neighbours


def test_the_patched_layers_do_not_spread_a_non_finite_value_to_the_border():
    """Out-of-grid taps read pixel 0 as a placeholder: masking it by a product gave inf * 0."""
    pytest.importorskip("hls4ml")
    from keras_hexagdly.hls4ml_ext import patch_model_for_hls

    inp = keras.Input((5, 5, 1))
    model = keras.Model(inp, hgly.MaxPool2d(1, 1)(inp))
    x = np.ones((1, 5, 5, 1), "float32")
    x[0, 0, 0, 0] = np.inf
    reference = keras.ops.convert_to_numpy(model(x))
    patched = keras.ops.convert_to_numpy(patch_model_for_hls(model)(x))
    assert not np.isnan(patched).any()
    np.testing.assert_array_equal(patched.reshape(reference.shape), reference)


def test_registration_covers_both_backends_and_a_later_call_adds_one():
    pytest.importorskip("hls4ml")
    from keras_hexagdly import hls4ml_handler

    hls4ml_handler.register_hex_gather_layers("Vivado")
    hls4ml_handler.register_hex_gather_layers("Vitis")  # used to return early: already registered
    hls4ml_handler.register_hex_gather_layers()
    assert {"Vivado", "Vitis"} <= hls4ml_handler._registered_backends


def test_registering_one_backend_gives_the_other_its_kernel_sources():
    """An explicit "Vivado" followed by a Vitis conversion used to miss the headers, and the
    default registration put the templates in both flows, formatting every layer twice."""
    hls4ml = pytest.importorskip("hls4ml")
    from hls4ml.model.optimizer import get_backend_passes

    from keras_hexagdly import hls4ml_handler

    hls4ml_handler.register_hex_gather_layers("Vivado")
    for name in ("Vivado", "Vitis"):
        sources = hls4ml.backends.get_backend(name).get_custom_source()
        assert any(str(dst).endswith("nnet_hex_conv_linebuffer_stream.h") for dst in sources), name
    assert any("hexconvlinebuffer" in p.lower() for p in get_backend_passes("vivado"))
    assert not any("hexconvlinebuffer" in p.lower() for p in get_backend_passes("vitis"))


def _checked_config(strategy, reuse, flatten=True):
    pytest.importorskip("hls4ml")
    from keras_hexagdly.hls4ml_ext import check_hls_config, patch_model_for_hls

    inputs = keras.Input((6, 6, 4))
    outputs = hgly.Conv2d(4, kernel_size=1, name="conv")(inputs)
    if flatten:
        outputs = keras.layers.Flatten()(outputs)
    patched = patch_model_for_hls(keras.Model(inputs, outputs))
    config = {"Model": {"Strategy": strategy, "ReuseFactor": reuse}}
    with pytest.warns() if (strategy, reuse) != ("Resource", 1) else warnings.catch_warnings():
        warnings.simplefilter("always")
        return check_hls_config(config, patched)


def test_the_wrong_rtl_warning_is_for_latency_only():
    """Resource with ReuseFactor=1 co-simulates correctly and is what hex_reuse_config picks."""
    assert any("wrong RTL" in issue for issue in _checked_config("Latency", 1))
    assert _checked_config("Resource", 1) == []


def test_latency_with_a_reuse_factor_says_what_it_costs():
    """'Latency' is the hls4ml default and now really pipelines the 2D kernel."""
    issues = _checked_config("Latency", 2, flatten=False)
    assert any("pipelines" in issue and "Strategy='Resource'" in issue for issue in issues)
    assert not any("pipelines" in issue for issue in _checked_config("Latency", 1))


def test_check_hls_config_reads_layer_type_entries_and_knows_the_dataflow_kernel():
    pytest.importorskip("hls4ml")
    from keras_hexagdly.hls4ml_ext import check_hls_config, patch_model_for_hls

    inputs = keras.Input((6, 6, 4))
    patched = patch_model_for_hls(keras.Model(inputs, hgly.Conv2d(4, 1, name="conv")(inputs)))
    by_type = {"LayerType": {"HHexConvLineBuffer": {"Strategy": "Latency", "ReuseFactor": 4}}}
    with pytest.warns(UserWarning, match="pipelines"):
        assert check_hls_config(by_type, patched)
    # the Dataflow kernel has a single implementation, whatever the strategy
    dataflow = {"Model": {"Strategy": "Latency", "ReuseFactor": 2, "Dataflow": True}}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_hls_config(dataflow, patched) == []


# ---- third audit --------------------------------------------------------------------------


def _gradient_of_the_sum(layer, x):
    """d sum(layer(x)) / dx on the current backend (a layer without weights)."""
    backend = keras.backend.backend()
    if backend == "torch":
        import torch

        t = torch.tensor(x, requires_grad=True)
        layer(t).sum().backward()
        return t.grad.detach().cpu().numpy()
    if backend == "tensorflow":
        import tensorflow as tf

        t = tf.Variable(x)
        with tf.GradientTape() as tape:
            total = tf.reduce_sum(layer(t))
        return tape.gradient(total, t).numpy()
    import jax

    return np.asarray(jax.grad(lambda t: layer(t).sum())(jax.numpy.asarray(x)))


def test_the_max_pooling_shares_the_gradient_between_tied_cells():
    """The same gradient on every backend: a backend's own max pooling gives it to one cell."""
    grad = _gradient_of_the_sum(hgly.MaxPool2d(1, 1), np.zeros((1, 5, 5, 1), "float32"))
    np.testing.assert_allclose(grad[0, 0, :, 0], [0.3125, 0.75, 0.375, 0.75, 0.3125], atol=1e-6)
    # tf.maximum gives everything to its first argument: 3D and strides go through the same fix.
    # Part of the gradient goes to the zero padding, the rest is spread over the tied cells.
    pool = hgly.MaxPool3d((2, 1), (1, 2))
    x = np.zeros((1, 3, 5, 5, 1), "float32")
    outputs = int(np.prod(pool(x).shape))
    grad = _gradient_of_the_sum(pool, x)
    assert grad.min() >= 0 and 0 < grad.sum() < outputs
    assert np.count_nonzero(grad) > outputs  # more cells than outputs: shared, not one each


def test_a_conv_refuses_another_channel_count_or_rank():
    """It ran a grouped convolution on the mismatched kernel, without an error."""
    conv = hgly.Conv2d(2, 1)
    conv(np.zeros((1, 5, 5, 1), "float32"))
    with pytest.raises(ValueError):
        conv(np.zeros((1, 5, 5, 2), "float32"))
    for layer in (hgly.Conv2d(2, 1), hgly.MaxPool2d(1), hgly.AvgPool2d(1)):
        with pytest.raises(ValueError, match="rank 4"):
            layer(np.zeros((1, 5, 5), "float32"))


@pytest.mark.parametrize("filters", [0, -1, 2.5, True])
def test_filters_must_be_a_positive_integer(filters):
    for conv in (hgly.Conv2d, hgly.Conv3d):
        with pytest.raises(ValueError, match="filters"):
            conv(filters, 1)


def test_a_3d_layer_needs_enough_frames():
    for layer in (hgly.Conv3d(2, (3, 1)), hgly.MaxPool3d((3, 1))):
        with pytest.raises(ValueError, match="frames"):
            layer(np.zeros((1, 2, 5, 5, 1), "float32"))
    assert (
        hgly.Conv3d(2, (3, 1), depth_padding="same")(np.zeros((1, 2, 5, 5, 1), "float32")).shape[1]
        == 2
    )
    with pytest.raises(ValueError, match="statically-known"):
        hgly.MaxPool3d((1, 1)).build((None, 3, None, 5, 1))


def test_a_custom_kernel_layer_keeps_trainable_false_through_its_config():
    layer = hgly.Conv2d_CustomKernel(trainable=True)
    layer.trainable = False
    clone = hgly.Conv2d_CustomKernel.from_config(layer.get_config())
    clone(np.zeros((1, 5, 5, 1), "float32"))
    assert clone.trainable_weights == []


def test_offset_tables_are_checked_against_the_border_of_the_table():
    conv = hgly.Conv2d(1, 1)
    table = build_neighbor_table(conv, 8, 8)[0].copy()
    build_parity_offset_tables(table, 8, 8, 8, H_in=8)  # the real table is accepted
    border = np.argwhere(table[:8] < 0)[0]  # a tap of the first row that falls outside the grid
    table[border[0], border[1]] = 40
    with pytest.raises(ValueError, match="border"):
        build_parity_offset_tables(table, 8, 8, 8, H_in=8)


def test_a_table_the_virtual_offsets_do_not_reproduce_is_refused():
    conv = hgly.Conv2d(1, 1)
    table = build_neighbor_table(conv, 3, 3)[0].copy()
    slot = np.argwhere(table < 0)[0]
    table[slot[0], slot[1]] = 4
    with pytest.raises(ValueError):
        build_parity_offset_tables(table, 3, 3, 3, H_in=3)


def test_the_neighbour_table_refuses_a_grid_too_large_for_its_probe():
    with pytest.raises(ValueError, match="2\\*\\*24"):
        build_neighbor_table(hgly.MaxPool2d(1), 4096, 4096)


def test_avg_pool_first_called_in_a_traced_function():
    """The count cannot be a NumPy constant there: it is computed in the graph, then cached."""
    pool = hgly.AvgPool2d(1)
    x = np.random.default_rng(0).normal(size=(2, 6, 5, 3)).astype("float32")
    model = keras.Sequential([keras.Input((6, 5, 3)), pool])
    traced = model.predict(x, verbose=0)
    np.testing.assert_allclose(keras.ops.convert_to_numpy(pool(x)), traced, atol=1e-6)
    table = build_neighbor_table(hgly.MaxPool2d(1), 6, 5)[0]
    valid = (table >= 0)[None, :, :, None]
    flat = x.reshape(2, 30, 3)
    mean = np.where(valid, flat[:, np.maximum(table, 0)], 0.0).sum(axis=2) / valid.sum(axis=2)
    np.testing.assert_allclose(traced.reshape(2, -1, 3), mean, atol=1e-5)


def test_patch_model_for_hls_names_a_nested_model_and_patches_a_subclass():
    pytest.importorskip("hls4ml")
    from keras_hexagdly.hex_gather import HexConvLineBuffer
    from keras_hexagdly.hls4ml_ext import patch_model_for_hls

    inner_in = keras.Input((5, 5, 1))
    inner = keras.Model(inner_in, hgly.Conv2d(2, 1)(inner_in), name="inner")
    outer_in = keras.Input((5, 5, 1))
    with pytest.raises(NotImplementedError, match="nested"):
        patch_model_for_hls(keras.Model(outer_in, inner(outer_in)))

    class MyConv(hgly.Conv2d):
        pass

    inp = keras.Input((5, 5, 1))
    patched = patch_model_for_hls(keras.Model(inp, MyConv(2, 1)(inp)))
    assert any(isinstance(layer, HexConvLineBuffer) for layer in patched.layers)


def test_verify_against_warns_about_a_model_in_training_mode():
    torch = pytest.importorskip("torch")
    from keras_hexagdly.torch_interop import verify_against

    inp = keras.Input((3,))
    model = keras.Model(inp, keras.layers.Dense(2, name="fc")(inp))
    linear = torch.nn.Sequential(torch.nn.Linear(3, 2))
    load_torch_state_dict(model, linear.state_dict(), {"fc": "0"})
    linear.train()
    with pytest.warns(UserWarning, match="training mode"):
        verify_against(model, linear, np.zeros((1, 3), "float32"))


def test_kernel_plots_show_quantized_and_pruned_weights():
    pytest.importorskip("hgq")
    pytest.importorskip("matplotlib")
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "notebooks"))
    import hexplot

    # a pruned (exactly zero) weight is a cell of the kernel, drawn as 0
    conv = hgly.Conv2d(1, 1, use_bias=False)
    conv(np.zeros((1, 7, 7, 1), "float32"))
    for kernel in conv._base_kernels:
        kernel.assign(np.zeros(kernel.shape, "float32"))
    tensor = hexplot.kernel_to_hextensor(conv)
    assert np.sum(~np.isnan(tensor)) == 7 and np.nansum(np.abs(tensor)) == 0


# ---- fourth audit -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "layers, shape",
    [
        (lambda: [hgly.Conv2d(3, 1, share_neighbors="ring"), hgly.MaxPool2d(1, 2)], (6, 6, 2)),
        (lambda: [hgly.Conv2d(2, 2, 2)], (6, 6, 2)),
        (lambda: [hgly.Conv3d(2, (2, 1), share_neighbors="ring"), hgly.MaxPool3d((2, 1))],
         (4, 5, 5, 2)),
    ],
    ids=["conv2d-ring", "conv2d-stride", "conv3d-ring"],
)  # fmt: skip
def test_a_model_without_an_input_is_built_by_its_first_fit(layers, shape):
    """Keras then builds the layers inside the traced training step. The initial values were
    drawn in build() and converted to NumPy there: an error on TensorFlow and JAX."""
    model = keras.Sequential([*layers(), keras.layers.Flatten(), keras.layers.Dense(1)])
    model.compile("sgd", "mse")
    x = np.random.default_rng(0).normal(size=(4, *shape)).astype("float32")
    model.fit(x, np.zeros((4, 1), "float32"), batch_size=2, epochs=1, verbose=0)
    assert np.all(np.isfinite(model.predict(x, verbose=0)))


def test_a_ring_shared_layer_can_be_used_by_two_models():
    conv = hgly.Conv2d(2, 1, share_neighbors="ring")
    x = np.ones((1, 6, 6, 1), "float32")
    outputs = []
    for _ in range(2):
        inp = keras.Input((6, 6, 1))
        outputs.append(keras.Model(inp, conv(inp)).predict(x, verbose=0))
    np.testing.assert_array_equal(outputs[0], outputs[1])
    assert all(isinstance(m, np.ndarray) for m in conv._ring_idx)


def test_the_initial_weights_follow_the_seed_and_the_combined_fan_in():
    def weights():
        keras.utils.set_random_seed(7)
        conv = hgly.Conv2d(64, 2)
        conv.build((None, 9, 9, 32))
        return [keras.ops.convert_to_numpy(w) for w in conv._base_kernels]

    first, second = weights(), weights()
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(first[1][0], first[2][0])  # each piece has its own draw
    values = np.concatenate([w.ravel() for w in first])
    assert np.isclose(values.std(), np.sqrt(2 / (19 * 32)), rtol=0.03)  # 19 taps, 32 channels


@pytest.mark.parametrize("dims", [2, 3])
def test_a_custom_kernel_layer_refuses_another_channel_count_or_rank(dims):
    """It ran a grouped convolution on the mismatched kernel, without an error."""
    cls = hgly.Conv2d_CustomKernel if dims == 2 else hgly.Conv3d_CustomKernel
    grid = (5, 5) if dims == 2 else (2, 5, 5)
    layer = cls()  # the default kernels: one input channel
    layer(np.zeros((1, *grid, 1), "float32"))
    with pytest.raises(ValueError):
        cls()(np.zeros((1, *grid, 2), "float32"))
    with pytest.raises(ValueError, match="rank"):
        cls()(np.zeros((1, *grid), "float32"))
    depth = () if dims == 2 else (1,)
    with pytest.raises(ValueError, match="must agree"):
        cls([np.ones((2, 1, *depth, 3, 1)), np.ones((3, 1, *depth, 2, 2))])


def test_a_float64_custom_kernel_layer_computes_in_float64():
    if keras.backend.backend() == "jax":
        pytest.skip("JAX computes in float32 unless x64 is enabled")
    kernels = [np.full((1, 1, 3, 1), 1 / 3), np.full((1, 1, 2, 2), 1 / 3)]
    layer = hgly.Conv2d_CustomKernel(kernels, dtype="float64")
    out = keras.ops.convert_to_numpy(layer(np.ones((1, 5, 5, 1), "float64")))
    assert out.dtype == np.float64
    assert abs(out[0, 2, 2, 0] - 7 / 3) < 1e-12  # float32 kernels gave 2.3333334922790527


def test_a_float64_max_pooling_returns_float64():
    """keras.ops.maximum of two float64 tensors gives float32 on torch (Keras 3.12)."""
    if keras.backend.backend() == "jax":
        pytest.skip("JAX computes in float32 unless x64 is enabled")
    x = np.full((1, 5, 5, 1), 1 / 3)
    for layer in (hgly.MaxPool2d(1, dtype="float64"), hgly.MaxPool2d(1, 2, dtype="float64")):
        out = keras.ops.convert_to_numpy(layer(x))
        assert out.dtype == np.float64 and out.max() == 1 / 3


@pytest.mark.parametrize(
    "layer, shape",
    [
        (lambda: hgly.Conv2d(4, 2, 3), (7, 9, 2)),
        (lambda: hgly.Conv2d(4, 1, 2), (8, 2, 2)),  # no more columns than the stride
        (lambda: hgly.MaxPool2d(1, 2), (5, 8, 3)),
        (lambda: hgly.AvgPool2d(2, 4), (9, 7, 3)),
        (lambda: hgly.Conv2d_CustomKernel(strides=2), (6, 7, 1)),
        (lambda: hgly.Conv3d(4, (3, 1), (2, 2), depth_padding="same"), (5, 7, 6, 2)),
        (lambda: hgly.Conv3d(4, (2, 1), (2, 3)), (5, 7, 6, 2)),
        (lambda: hgly.MaxPool3d((2, 1), (3, 2)), (7, 6, 5, 2)),
    ],
)
def test_compute_output_shape_is_the_shape_the_layer_returns(layer, shape):
    out = layer()(np.zeros((2, *shape), "float32"))
    assert layer().compute_output_shape((2, *shape)) == tuple(out.shape)
    assert layer().compute_output_shape((None, *shape)) == (None, *out.shape[1:])


def test_a_hex_layer_runs_inside_time_distributed():
    """Keras asks compute_output_shape there, which the layers did not have."""
    wrapped = keras.layers.TimeDistributed(hgly.Conv2d(4, 1, 2))
    assert tuple(wrapped(np.zeros((2, 3, 8, 8, 2), "float32")).shape) == (2, 3, 4, 4, 4)


def test_the_weight_maps_returned_are_copies():
    maps, _ = hex_layers.weight_maps_2d(2, "ring")
    maps[0][:] = 99
    assert hex_layers.ring_maps_2d(2)[0][0].max() < 99


def test_a_dense_outside_a_functional_graph_is_loaded_with_a_warning():
    """After a Flatten its weight needs reordering, which the loader cannot see there: it was
    loaded transposed only, silently, and the model was wrong."""

    class Net(keras.Model):
        def __init__(self):
            super().__init__()
            self.flat = keras.layers.Flatten()
            self.fc = keras.layers.Dense(2, name="fc")

        def call(self, x):
            return self.fc(self.flat(x))

    model = Net()
    model(np.zeros((1, 2, 2, 3), "float32"))
    state = {"fc.weight": np.zeros((2, 12), "float32"), "fc.bias": np.zeros(2, "float32")}
    with pytest.warns(UserWarning, match="functional graph"):
        load_torch_state_dict(model, state, {"fc": "fc"})
    with warnings.catch_warnings():  # the caller says the order is already the Keras one
        warnings.simplefilter("error")
        load_torch_state_dict(model, state, {"fc": "fc"}, flatten_order="channels_last")


def test_a_bias_the_keras_layer_does_not_have_is_refused():
    inp = keras.Input((4,))
    model = keras.Model(inp, keras.layers.Dense(3, use_bias=False, name="fc")(inp))
    state = {"fc.weight": np.zeros((3, 4), "float32"), "fc.bias": np.zeros(3, "float32")}
    with pytest.raises(ValueError, match="use_bias=False"):
        load_torch_state_dict(model, state, {"fc": "fc"})
    del state["fc.bias"]
    load_torch_state_dict(model, state, {"fc": "fc"})


def test_the_kernel_picture_shows_each_weight_on_the_neighbour_it_applies_to():
    """It was the kernel turned by 180 degrees: invisible with ring sharing, wrong without."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")  # a figure is made below: no display needed
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "notebooks"))
    import hexplot

    conv = hgly.Conv2d(1, 1, use_bias=False)
    conv(np.zeros((1, 9, 9, 1), "float32"))
    conv._base_kernels[0].assign(np.array([1, 2, 3], "float32").reshape(3, 1, 1, 1))
    conv._base_kernels[1].assign(np.array([[4, 5], [6, 7]], "float32").reshape(2, 2, 1, 1))
    picture = hexplot.kernel_to_hextensor(conv)[0, :, :, 0]
    centre_row, centre_col = np.argwhere(picture == 2)[0]
    assert centre_col % 2 == 0  # so the picture is the neighbourhood of an even-column cell
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            x = np.zeros((1, 9, 9, 1), "float32")
            x[0, 4 + dr, 4 + dc, 0] = 1
            applied = float(keras.ops.convert_to_numpy(conv(x))[0, 4, 4, 0])
            row, col = centre_row + dr, centre_col + dc
            inside = 0 <= row < picture.shape[0] and 0 <= col < picture.shape[1]
            drawn = picture[row, col] if inside else np.nan
            assert drawn == applied or (np.isnan(drawn) and applied == 0), (dr, dc)

    # the default kernels of a custom-kernel layer: used to fail (no `filters`)
    assert hexplot.kernel_to_hextensor(hgly.Conv2d_CustomKernel()).shape == (1, 3, 4, 1)

    # the axis limits follow the grid: x over the columns, y over the rows (they were swapped)
    axes = hexplot.plot_hextensor(np.ones((1, 4, 20, 1), "float32")).axes[0]
    assert axes.get_xlim()[1] > 19 * np.sqrt(3) / 2 and axes.get_ylim()[0] < -3.5
