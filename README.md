# keras-hexagdly

Keras 3 port of [HexagDLy](https://github.com/ai4iacts/hexagdly): convolution
and pooling methods for hexagonally sampled data, originally written for
PyTorch by Tim Lukas Holch and Constantin Steppa (ai4iacts).

This port reproduces HexagDLy's hexagonal addressing scheme and sub-kernel
decomposition exactly (bit-for-bit equivalent outputs, see
[tests/test_vs_pytorch_hexagdly.py](tests/test_vs_pytorch_hexagdly.py)), but
is built on [Keras 3](https://keras.io) so it runs on any backend
(TensorFlow, JAX, PyTorch) and uses a channels-last (`NHWC`/`NDHWC`) tensor
layout instead of PyTorch's channels-first.

It also adds two functionalities that do not exist in upstream HexagDLy:

- **`share_neighbors`** (`Conv2d`, `Conv3d`): ties the weights of a hexagonal
  kernel by ring (ring 0 = center cell, ring *r* = the `6*r` cells at hex
  distance *r*), instead of every cell having its own independent weight.
  Reduces parameter count and enforces exact 6-fold rotational symmetry of
  the learned kernel.
- **`depth_padding="same"`** (`Conv3d` only): zero-pads the depth/time axis
  so the temporal kernel is centred on each time step and the output depth
  equals the input depth, instead of HexagDLy's `"valid"`-only behaviour
  (output depth shrinks by `kernel - 1`).

See [NOTICE.md](NOTICE.md) for attribution details and citation information.

## Installation

```
pip install keras-hexagdly
```

For development (running the test suite, which also checks parity against
upstream PyTorch HexagDLy, and the example notebooks):

```
pip install keras-hexagdly[dev]
```

## Usage

```python
import keras
import keras_hexagdly as hgly

kernel_size, stride = 1, 4
in_channels, out_channels = 1, 3

hexconv = hgly.Conv2d(in_channels, out_channels, kernel_size, stride)
x = keras.random.uniform((1, 21, 21, 1))  # channels-last: (N, H, W, C)
y = hexconv(x)
```

`in_channels` can be omitted; it is then inferred from the input on first
call, like a standard Keras layer: `hgly.Conv2d(out_channels, kernel_size=kernel_size, stride=stride)`.

### New: weight sharing by hexagonal ring

```python
hexconv = hgly.Conv2d(in_channels, out_channels, kernel_size=3, share_neighbors=True)
```

### New: same-padded temporal convolution (Conv3d)

```python
conv3d = hgly.Conv3d(in_channels, out_channels, kernel_size=(depth_k, hex_k),
                      depth_padding="same")  # output depth == input depth
```

Before applying these layers, your data must already be arranged on the
square-tensor layout HexagDLy expects (zig-zag columns); see
[notebooks/keras_hexagdly_addressing_scheme.ipynb](notebooks/keras_hexagdly_addressing_scheme.ipynb)
for how to get there from raw detector coordinates, and
[notebooks/keras_hexagdly_2d_example.ipynb](notebooks/keras_hexagdly_2d_example.ipynb)
for a worked convolution/pooling example, including the new features above.

## Notebooks

Ported from [HexagDLy's own notebooks](https://github.com/ai4iacts/hexagdly/tree/master/notebooks), one-to-one where the content is framework-specific, lightly adapted where it depends on a torch-specific dataloader/training loop:

- [`keras_hexagdly_2d_example.ipynb`](notebooks/keras_hexagdly_2d_example.ipynb) -- basic `Conv2d`/`MaxPool2d` usage, hex-vs-square symmetry, and the new `share_neighbors`/`depth_padding` features.
- [`keras_hexagdly_addressing_scheme.ipynb`](notebooks/keras_hexagdly_addressing_scheme.ipynb) -- how to map raw hexagonal detector coordinates onto the square-tensor layout the layers expect (backend-independent; near-identical to upstream).
- [`keras_hexagdly_custom_kernels_example.ipynb`](notebooks/keras_hexagdly_custom_kernels_example.ipynb) -- building a custom Gaussian smoothing kernel with `Conv2d_CustomKernel`.
- [`keras_hexagdly_cnn_example.ipynb`](notebooks/keras_hexagdly_cnn_example.ipynb) -- a small CNN classifying toy hexagonal shapes, trained with `model.fit`.
- [`keras_hexagdly_hex_vs_square.ipynb`](notebooks/keras_hexagdly_hex_vs_square.ipynb) -- parameter-count and timing benchmark of hex vs. square kernels, plus a hex-CNN-vs-square-CNN classification comparison.

## Testing

```
pip install -e .[dev] --no-build-isolation   # see note below
pytest tests/
```

(`--no-build-isolation`: only needed if your `pip` is old -- pip 22.0.2's
isolated build environment was observed to pick up a `setuptools` version
that mis-names the built wheel `UNKNOWN`. Verified clean with a modern pip
(>=23) in a fresh venv: plain `pip install .` works with no workaround.
Either way, `pytest tests/` works without installing anything -- `conftest.py`
puts `src/` and `tests/` on `sys.path`.)

Verified to pass on all three Keras 3 backends (set `KERAS_BACKEND=tensorflow|torch|jax`
before importing keras; tensorflow is the default if unset):

```
KERAS_BACKEND=tensorflow pytest tests/   # 272 passed, 7 skipped
KERAS_BACKEND=torch       pytest tests/   # 269 passed, 10 skipped
KERAS_BACKEND=jax         pytest tests/   # 269 passed, 10 skipped (slower: per-shape JIT compile)
```

A GitHub Actions workflow ([.github/workflows/test.yml](.github/workflows/test.yml))
runs this matrix (3 backends x 3 Python versions) plus `ruff check`/`ruff format --check`
on every push and PR.

The test suite has six parts:

- `test_Conv2d.py`, `test_Conv3d.py`, `test_*_CustomKernel.py`,
  `test_MaxPool2d.py`, `test_MaxPool3d.py`: standalone tests with
  hand-computed expected outputs, ported from
  [HexagDLy's own test suite](https://github.com/ai4iacts/hexagdly/tree/master/tests).
- `test_vs_pytorch_hexagdly.py`: cross-checks every layer against the
  upstream PyTorch `hexagdly` PyPI package (random inputs, weights copied
  across frameworks, gradients, batch independence, odd/even column parity,
  large strides, asymmetric 3D kernels) -- the oracle that proves this port
  faithful. Also includes two *forward-compatibility* tests that stay
  skipped today (upstream hexagdly 2.0.2 has neither feature) but will
  automatically start cross-checking `share_neighbors`/`depth_padding`
  against upstream the day a future hexagdly release adds them.
- `test_mixed_precision.py`: `keras.mixed_precision` / per-layer `dtype=`
  policies -- variables stay float32, compute happens in float16, on every
  backend.
- `test_share_neighbors.py`, `test_depth_padding.py`: standalone tests for
  the two new functionalities, which have no PyTorch equivalent to check
  against.
- `test_edge_cases.py`, `test_serialization.py`, `test_geometry.py`: input
  validation, minimum viable sizes, dtype handling, NaN/Inf checks, and
  `get_config`/`from_config` + full `model.save`/`load_model` round-trips.

## Disclaimer

Like upstream HexagDLy, this is a prototyping tool: it favors flexibility
over performance. Once a model's architecture (kernel size, stride, input
shape) is fixed, hard-coding those parameters would yield a faster
implementation.

## Performance

See [benchmarks/](benchmarks/) for a speed comparison against upstream
PyTorch HexagDLy. Short version: run eagerly on CPU, this port is 1-7x
slower than upstream for the same reason upstream itself is slow (the hex
sub-kernel decomposition costs several op-dispatches per call -- a design
choice, not a regression). Wrapped in a compiled call (`jax.jit`/
`tf.function`, which `model.fit`/`model.predict` do automatically) it is
typically *faster* than upstream's eager PyTorch, sometimes by an order of
magnitude. `torch.compile` support is currently unreliable for this layer
(see the benchmarks README for why); eager execution on a GPU is the
recommended way to get speed on the torch backend.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

MIT, see [LICENSE](LICENSE). This is a derivative work of HexagDLy
(Copyright (c) 2018 ai4iacts); see [NOTICE.md](NOTICE.md).
