"""Load PyTorch hexagonal-conv weights into the Keras layers.

Works with BOTH PyTorch implementations, which name their parameters
identically for the unshared case (`kernel0`, `kernel1`, ..., `bias_tensor`):

  * upstream `hexagdly` -- unshared only, it has no weight-sharing concept;
  * `pytorch-hexagdly` (this project's fork) -- same names unshared, plus
    `shared_weights` when share_neighbors is "ring"/"diag"/"sym".

**This module does not import torch.** It takes an already-loaded mapping of
name -> array-like and duck-types the values, so it works equally on a live
`net.state_dict()` and on arrays restored from an `.npz`:

    # once, on the training machine (torch available)
    sd = torch.load("model.pth", map_location="cpu")["model_state_dict"]
    np.savez("model.npz", **{k: v.numpy() for k, v in sd.items()})

    # afterwards, anywhere (keras + numpy only)
    load_torch_state_dict(keras_model, dict(np.load("model.npz")), mapping={...})

What this module contributes is the LAYOUT knowledge -- PyTorch keeps
channels first, Keras channels last -- which is otherwise re-derived by hand in
every export script. What it deliberately does not do is guess which PyTorch
parameter belongs to which Keras layer: the two graphs are built independently,
so the caller states the correspondence. An implicit by-order pairing would fail
silently on any mismatch, and a silently mis-transposed model still runs and
still produces plausible numbers.

For that same reason `verify_against` exists and the docstrings point at it:
transposition errors are invisible without a numerical check.
"""

import numpy as np

__all__ = [
    "to_numpy",
    "load_hex_conv2d_weights",
    "load_torch_state_dict",
    "verify_against",
]


def _key(prefix, name):
    """Join a parameter prefix and a name, tolerating an empty prefix.

    `prefix=""` is the legitimate case of a bare layer whose parameters sit at
    the state dict's root ("kernel0" rather than "block.0.kernel0").
    """
    return f"{prefix}.{name}" if prefix else name


def to_numpy(value):
    """array-like -> numpy, accepting torch tensors without importing torch."""
    if hasattr(value, "detach"):  # torch.Tensor, duck-typed
        value = value.detach().cpu()
    return np.asarray(value)


def load_hex_conv2d_weights(layer, state_dict, prefix):
    """Copy a pytorch-hexagdly Conv2d's weights into a keras-hexagdly Conv2d.

    `prefix` is the PyTorch parameter prefix, e.g. "spatial.0" for parameters
    named "spatial.0.kernel0", "spatial.0.kernel1", ...

    Layout: PyTorch stores each sub-kernel as (out, in, kh, kw), Keras as
    (kh, kw, in, out) -- a (2, 3, 1, 0) transpose.

    Weight sharing is supported for all of pytorch-hexagdly's modes
    ("ring"/"diag"/"sym"). Upstream `hexagdly` never takes this branch -- it has
    no sharing -- so an unshared checkpoint from either implementation loads the
    same way. In the shared case: PyTorch keeps `shared_weights` as
    (out, in, num_shared) and Keras keeps `ring_weights` as
    (num_shared, in, out) -- a (2, 1, 0) transpose. That correspondence is the
    one hexagdly-oracle's test_cross_framework.py exercises numerically against
    the PyTorch implementation, mode by mode.
    """
    if getattr(layer, "share_neighbors", False):
        key = _key(prefix, "shared_weights")
        if key not in state_dict:
            raise KeyError(
                f"{layer.name}: share_neighbors={layer.share_neighbors!r} expects "
                f"{key} in the state dict. Is the PyTorch layer sharing weights too?"
            )
        w = to_numpy(state_dict[key])  # (out, in, num_shared)
        got = np.transpose(w, (2, 1, 0)).shape  # (num_shared, in, out)
        expected = tuple(layer.ring_weights.shape)
        if got != expected:
            raise ValueError(
                f"{layer.name}: PyTorch shared_weights {w.shape} transposes to {got}, "
                f"but the Keras layer expects {expected}. The sharing modes most "
                "likely differ -- check share_neighbors on both sides."
            )
        layer.ring_weights.assign(np.transpose(w, (2, 1, 0)))
        _load_hex_bias(layer, state_dict, prefix)
        return

    n_sub = layer.hexbase_size + 1
    missing = [
        _key(prefix, f"kernel{i}")
        for i in range(n_sub)
        if _key(prefix, f"kernel{i}") not in state_dict
    ]
    if missing:
        raise KeyError(
            f"{layer.name}: missing {missing} in the state dict. Available keys "
            f"available: {sorted(state_dict)[:12]}"
        )

    for i in range(n_sub):
        w = to_numpy(state_dict[_key(prefix, f"kernel{i}")])  # (out, in, kh, kw)
        expected = tuple(layer._base_kernels[i].shape)
        got = np.transpose(w, (2, 3, 1, 0)).shape  # (kh, kw, in, out)
        if got != expected:
            raise ValueError(
                f"{layer.name}.kernel{i}: PyTorch weight {w.shape} transposes to "
                f"{got}, but the Keras layer expects {expected}. The two models "
                "are not the same architecture."
            )
        layer._base_kernels[i].assign(np.transpose(w, (2, 3, 1, 0)))

    _load_hex_bias(layer, state_dict, prefix)


def _load_hex_bias(layer, state_dict, prefix):
    bias_key = _key(prefix, "bias_tensor")
    if getattr(layer, "use_bias", False):
        if bias_key not in state_dict:
            raise KeyError(f"{layer.name}: expects a bias but {bias_key} is absent")
        layer.bias_tensor.assign(to_numpy(state_dict[bias_key]))
    elif bias_key in state_dict:
        raise ValueError(
            f"{layer.name}: built with use_bias=False but the state dict carries "
            f"{bias_key}; the weights would be silently dropped"
        )


def _load_dense(layer, state_dict, prefix):
    """PyTorch Linear (out, in) -> Keras Dense (in, out)."""
    w = to_numpy(state_dict[_key(prefix, "weight")]).T
    weights = [w]
    if layer.use_bias:
        weights.append(to_numpy(state_dict[_key(prefix, "bias")]))
    layer.set_weights(weights)


def _load_conv2d(layer, state_dict, prefix):
    """PyTorch Conv2d (out, in, kh, kw) -> Keras Conv2D (kh, kw, in, out)."""
    raw = to_numpy(state_dict[_key(prefix, "weight")])
    if raw.ndim != 4:
        raise ValueError(
            f"{layer.name}: expected a 4-D PyTorch Conv2d weight (out, in, kh, kw) "
            f"but {prefix}.weight has shape {raw.shape}. A Conv3d reformulated as "
            "Conv2D is a model-specific rewrite -- set that layer's weights yourself."
        )
    w = np.transpose(raw, (2, 3, 1, 0))
    weights = [w]
    if layer.use_bias:
        weights.append(to_numpy(state_dict[_key(prefix, "bias")]))
    layer.set_weights(weights)


def load_torch_state_dict(model, state_dict, mapping):
    """Copy PyTorch weights into `model`, layer by layer.

    Args:
        model:      a built keras.Model.
        state_dict: mapping of parameter name -> array-like (torch tensors or
                    numpy arrays; nothing torch-specific is imported).
        mapping:    {keras layer name: pytorch parameter prefix}. Explicit on
                    purpose -- see the module docstring.

    Handles keras_hexagdly.Conv2d, keras.layers.Conv2D and keras.layers.Dense.
    Any other layer type raises rather than being skipped, so a layer that
    silently keeps its random initialisation is impossible.

    Always follow this with `verify_against` -- a transposed weight loads
    without complaint and only shows up as degraded accuracy.
    """
    import keras

    import keras_hexagdly as hgly

    for keras_name, prefix in mapping.items():
        layer = model.get_layer(keras_name)  # raises if absent
        if isinstance(layer, hgly.Conv2d):
            load_hex_conv2d_weights(layer, state_dict, prefix)
        elif isinstance(layer, keras.layers.Dense):
            _load_dense(layer, state_dict, prefix)
        elif isinstance(layer, keras.layers.Conv2D):
            _load_conv2d(layer, state_dict, prefix)
        else:
            raise NotImplementedError(
                f"{keras_name}: no loader for {type(layer).__name__}. Set its "
                "weights directly rather than leaving it at its initialisation."
            )

    unmapped = [l.name for l in model.layers if l.weights and l.name not in mapping]
    if unmapped:
        raise ValueError(
            f"these layers carry weights but are not in `mapping`, so they would "
            f"keep their random initialisation: {unmapped}"
        )
    return model


def verify_against(keras_model, torch_model, x_nchw, atol=1e-4):
    """Assert the two models agree on `x_nchw` (PyTorch's channels-first layout).

    Not optional in practice: a wrong transpose produces a model that runs and
    returns plausible numbers. `torch_model` is called as-is, so this is the one
    place torch enters -- through the caller's object, never through an import.
    """
    to_last = lambda a: (0,) + tuple(range(2, a.ndim)) + (1,)  # noqa: E731

    y_torch = to_numpy(torch_model(_as_tensor_like(x_nchw, torch_model)))
    y_keras = np.asarray(keras_model.predict(np.transpose(x_nchw, to_last(x_nchw)), verbose=0))

    # The OUTPUT needs converting too, not just the input: torch returns
    # (N, C, ...) and Keras (N, ..., C). Flattening both and subtracting -- as an
    # earlier revision did -- compares different memory layouts and reports a
    # large error on a perfectly good model. Outputs of rank <= 2 (after a
    # Flatten or Dense) have no channel axis to move.
    if y_torch.ndim > 2:
        y_torch = np.transpose(y_torch, to_last(y_torch))
    err = float(np.max(np.abs(y_torch.reshape(-1) - y_keras.reshape(-1))))
    if err > atol:
        raise AssertionError(
            f"PyTorch and Keras disagree by {err:.3e} (> {atol:.0e}). The most "
            "likely cause is a weight left in PyTorch's layout."
        )
    return err


def _as_tensor_like(x, torch_model):
    """Turn `x` into a tensor of the model's own kind, without importing torch.

    Borrows it from the model's first parameter via `new_tensor`, which carries
    the right dtype and device. Inspecting the model's module name instead does
    not work -- a layer defined in `pytorch_hexagdly` reports that module, not
    `torch`, so the conversion was silently skipped and the raw ndarray reached
    the forward pass.
    """
    try:
        param = next(torch_model.parameters())
    except (AttributeError, StopIteration):
        return x  # not a torch module; let the caller's object cope
    return param.new_tensor(np.asarray(x))
