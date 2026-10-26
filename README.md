# keras-hexagdly

Keras 3 port of [HexagDLy](https://github.com/ai4iacts/hexagdly): convolution
and pooling methods for hexagonally sampled data, originally written for
PyTorch by Tim Lukas Holch and Constantin Steppa (ai4iacts).

This port reproduces HexagDLy's hexagonal addressing scheme and sub-kernel
decomposition exactly (bit-for-bit equivalent outputs, cross-checked against the
upstream PyTorch package in
[hexagdly-oracle](https://github.com/YugnatD/hexagdly-oracle)), but
is built on [Keras 3](https://keras.io) so it runs on any backend
(TensorFlow, JAX, PyTorch) and uses a channels-last (`NHWC`/`NDHWC`) tensor
layout instead of PyTorch's channels-first.

It also adds functionality that does not exist in upstream HexagDLy:

- **`share_neighbors`** (`Conv2d`, `Conv3d`): ties the weights of several cells
  of a hexagonal kernel together instead of giving every cell its own
  independent weight, which cuts the parameter count and imposes a geometric
  symmetry on the learned kernel. Three modes -- `"ring"`, `"diag"`, `"sym"` --
  are [illustrated below](#new-share_neighbors----weight-sharing-across-kernel-cells).
- **`depth_padding="same"`** (`Conv3d` only): zero-pads the depth/time axis
  so the temporal kernel is centred on each time step and the output depth
  equals the input depth, instead of HexagDLy's `"valid"`-only behaviour
  (output depth shrinks by `kernel - 1`).
- **[hls4ml export](#fpga-export-via-hls4ml)** for FPGA synthesis, currently
  limited to `io_stream` and the 2D layers.
- **`AvgPool2d`**: hexagonal average pooling over the same window as
  `MaxPool2d` (upstream HexagDLy only provides max pooling).
- **`kernel_size_unit="pixels"`** and `kernel_size`/`pool_size=0`: an
  alternative kernel-size convention and the true 1x1 it needs, [described
  below](#new-kernel_size_unit----reading-kernel_size-as-a-square-kernel-width).
- **Keras-compatible arguments**: `Conv2d` accepts `padding` and `activation`,
  and the pooling layers accept `pool_size`, so the hex layers can be built from
  the same keyword arguments as `keras.layers.Conv2D`, `MaxPool2D` and
  `AveragePooling2D`:

  ```python
  conv_cls, pool_cls = (hgly.Conv2d, hgly.MaxPool2d) if hexagonal else (
      keras.layers.Conv2D, keras.layers.MaxPool2D)
  x = conv_cls(filters=32, kernel_size=1, padding="same", activation="relu")(x)
  x = pool_cls(pool_size=1, strides=2)(x)
  ```

  `padding` only accepts `"same"`: the hex grid is always padded, so the output
  size depends only on the stride. `kernel_size` is still accepted on the
  pooling layers as an alias of `pool_size`. The pooling layers also take
  their arguments in the Keras positional order; for `Conv2d`, pass anything
  after `strides` by keyword (its fourth positional argument is `use_bias`,
  and a Keras-style positional `padding` there raises an error).

Weights trained with [pytorch-hexagdly](https://github.com/YugnatD/pytorch-hexagdly)
can be loaded into these layers -- see
[`keras_hexagdly.torch_interop`](src/keras_hexagdly/torch_interop.py) and
[`notebooks/pytorch_to_keras_example.ipynb`](notebooks/pytorch_to_keras_example.ipynb).
No torch dependency is required: a plain `.npz` of the state dict works.

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

hexconv = hgly.Conv2d(filters=3, kernel_size=1, strides=4)
x = keras.random.uniform((1, 21, 21, 1))  # channels-last: (N, H, W, C)
y = hexconv(x)
```

`filters`/`kernel_size`/`strides`/`use_bias` follow standard Keras layer
conventions exactly (`filters`/`kernel_size` are positional-capable, just
like `keras.layers.Conv2D(filters, kernel_size, strides=...)`). `in_channels`
is never a constructor argument -- it's always inferred from the input on
first call, like every standard Keras layer.

**Breaking change in 0.4.0**: the original PyTorch-hexagdly-style
constructor names (`in_channels`, `out_channels`, `stride`, `bias`) have
been removed entirely -- passing them now raises, rather than being
silently accepted. This was a deliberate, one-time breaking change (see
[CHANGELOG.md](CHANGELOG.md)) made specifically because those old names,
combined with Python's positional argument binding, meant
`Conv2d(32, 3, strides=2)` -- written the way anyone familiar with
`keras.layers.Conv2D` naturally would -- silently built the wrong layer
(`in_channels=32, out_channels=3, kernel_size=1`) instead of raising. If you
have code or saved models depending on the old names, pin
`keras-hexagdly==0.3.1`, the one transitional release where both forms
worked side by side.

### New: `share_neighbors` -- weight sharing across kernel cells

Available on `Conv2d` and `Conv3d`. `share_neighbors` reduces the number of
learnable parameters by grouping kernel cells that share a single weight.
Three modes are available, illustrated below for `kernel_size=2` (19 cells):

| `share_neighbors="ring"` | `share_neighbors="diag"` | `share_neighbors="sym"` |
|:---:|:---:|:---:|
| ![ring](figures/share_ring_k2.png) | ![diag](figures/share_diag_k2.png) | ![sym](figures/share_sym_k2.png) |
| **3 weights** -- cells at the same hex distance from center share one weight (concentric rings). | **10 weights** -- visually opposite (antipodal) cells share one weight. | **10 weights** -- geometrically adjacent 60 degree pairs share one weight. |

- **`"ring"`**: the most aggressive reduction. All 6 direct neighbours share one
  weight, all 12 outer cells share another. Enforces exact 6-fold rotational
  symmetry of the learned kernel.
- **`"diag"`**: antipodal symmetry -- each cell and its mirror image through the
  center share a weight. Useful when the kernel should be point-symmetric.
- **`"sym"`**: 60 degree adjacent pairs -- consecutive neighbours along the kernel
  boundary share a weight. Useful when the kernel should reflect local
  rotational symmetry.

For `kernel_size=1` (7 cells): ring=2 weights, diag=4, sym=4.
For `kernel_size=2` (19 cells): ring=3 weights, diag=10, sym=10.
Compare to the default `share_neighbors=None`, which gives 7 and 19 independent
weights. `"ring"` works at any kernel size; `"diag"` and `"sym"` are defined for
`kernel_size` 1 and 2 only, matching
[pytorch-hexagdly](https://github.com/YugnatD/pytorch-hexagdly), which does not
define them beyond n=2 either.

```python
hexconv = hgly.Conv2d(filters=3, kernel_size=2, share_neighbors="ring")
```

`share_neighbors=True` is accepted as an alias for `"ring"`.

### New: `kernel_size_unit` -- reading `kernel_size` as a square-kernel width

`kernel_size` counts hexagonal rings by default (`kernel_size_unit="rings"`):
`kernel_size=1` is the 7-cell hexagon (a cell and its six neighbours),
`kernel_size=2` the 19-cell one. The same number therefore means a much
larger kernel than on a square grid, where `kernel_size=3` is 9 cells.

`kernel_size_unit="pixels"` reads `kernel_size` as a square-kernel width
instead -- radius = `kernel_size // 2` -- so the same architecture means a
comparable receptive field on both grids, as in
[indexedconv](https://github.com/IndexedConv/IndexedConv)'s `MaskedConv2d`:

```python
hgly.Conv2d(filters=32, kernel_size=3, kernel_size_unit="pixels")  # the 7-cell hexagon
hgly.Conv2d(filters=32, kernel_size=1, kernel_size_unit="pixels")  # a plain 1x1, no neighbours
```

`kernel_size=0` (0 rings, a plain per-cell channel projection with no
neighbours) is what `kernel_size_unit="pixels"` needs to express a true 1x1,
and is accepted directly in `"rings"` mode too. A hexagonal window is centred on a
cell, so it only exists for odd widths. An even width is rounded **up** to the next odd
one (`kernel_size=2` in `"pixels"` mode gives the 7-cell window of `3`, not the single
cell of `1`, and `4` gives that of `5`), and a `UserWarning` says so; use an odd width
to silence it. Applies to `MaxPool2d`/`AvgPool2d` (via `pool_size`) the same way;
on `Conv3d`/`MaxPool3d`, only the hex element of the `(depth, hex)` size
follows the unit, since depth is never a hex radius.

### New: quantization-aware training with HGQ2 (`QConv2d` and friends)

> A complete example, from data to a synthesised design (training with EBOPs, `freeze`, hls4ml, C simulation,
> Vitis HLS against a uniform-precision design): [notebooks/keras_hexagdly_qat_hgq2_to_hls4ml.ipynb](notebooks/keras_hexagdly_qat_hgq2_to_hls4ml.ipynb).

Optional: `pip install keras-hexagdly[qat]` (needs Python >= 3.10 and HGQ2 >= 0.1.6, tested up to
0.2.0). Each hexagonal layer has a quantized
version that puts [HGQ2](https://github.com/calad0i/HGQ2) quantizers on what you ask for, so the
network trains with the rounding it will have in fixed point:

| quantized layer | hexagonal layer | quantizers |
|---|---|---|
| `QConv2d`, `QConv3d` | `Conv2d`, `Conv3d` | kernel, bias, input, output |
| `QConv2d_CustomKernel`, `QConv3d_CustomKernel` | `Conv2d_CustomKernel`, `Conv3d_CustomKernel` | same; the kernels are only trained with `trainable=True` |
| `QMaxPool2d`, `QMaxPool3d`, `QAvgPool2d` | `MaxPool2d`, `MaxPool3d`, `AvgPool2d` | input, output (no weights) |

`kq_conf` (kernel; HGQ2's default weight config if omitted), `bq_conf` (bias), `iq_conf` (input of the
layer) and `oq_conf` (output, after the activation) take an HGQ2 `QuantizerConfig`; all but `kq_conf` are
off unless given. Every sub-kernel of the hexagonal decomposition has its own quantizer (with
`share_neighbors` the shared weights are quantized, in every mode, then spread over their cells).

```python
import keras
import keras_hexagdly as hgly
from hgq.quantizer import QuantizerConfig
from hgq.regularizers import MonoL1

# start at 8 bits per weight; the bit widths are learned, the L1 term pushes them down
weights = QuantizerConfig("kbi", "weight", k0=1, b0=8, i0=2, br=MonoL1(1e-5))

inp = keras.Input((24, 24, 8))
x = keras.layers.ReLU()(hgly.QConv2d(16, kernel_size=1, kq_conf=weights, name="hex0")(inp))
x = keras.layers.ReLU()(hgly.QConv2d(32, kernel_size=1, strides=2, kq_conf=weights, name="hex1")(x))
model = keras.Model(inp, keras.layers.Dense(1)(keras.layers.GlobalAveragePooling2D()(x)))

model.compile("adam", "binary_crossentropy")
model.fit(...)                                   # a normal fit; the bit regulariser is in the loss

frozen, types = hgly.freeze(model)               # plain layers holding the quantized values
print(types["hex0"])                             # the ap_fixed types the training found, per tensor
```

With `trainable=False` in a `QuantizerConfig` the bit widths stay fixed (plain fixed-point QAT); with a
regulariser they are learned, and a weight that reaches 0 bit is removed. `model.save()` and
`load_model()` keep the quantizers and their configs. `freeze` returns a new model in which every
quantized layer is back to its plain hexagonal layer, with the same outputs (its default
`activation_quantizers="keep"` leaves the activation quantizers in the model as standalone layers, which
hls4ml's bit-exact flow turns into the types of the export; see the export below, and `"types"` for setting
them by hand), and, for each quantized layer, the `ap_fixed` types to give hls4ml: `weight`, `bias`, `input`, `result` (for what was quantized) and
`accum`. `patch_model_for_hls` refuses a model that still has quantized layers.

**EBOPs.** A convolution with an input quantizer (`iq_conf`) counts its cost in EBOPs (effective bit
operations: the bits of each weight times the bits of the input it multiplies, as HGQ2 counts bits
(`i + f`: the sign bit of a signed quantizer is not counted), over the hexagon,
zero padding of the border excluded, a pruned weight costing nothing) exactly like HGQ2's own layers:
during training the loss gets `beta * EBOPs` (`beta0=1e-5` by default, like HGQ2; `enable_ebops=False`
turns it off, `ebops_factor` scales it), `layer.ebops` holds the last value, and HGQ2's callbacks
work on a model that mixes these layers with HGQ2's: `FreeEBOPs`, `BetaPID` (aim at a target
EBOPs), `BetaScheduler`, `PBar`, `ParetoFront`, `EarlyStoppingWithEbopsThres`, and `trace_minmax` to
calibrate the integer bits of `WRAP` quantizers. The EBOPs assume a fully parallel layer (no
`parallelization_factor`) and have not been validated against the synthesis of the hexagonal kernels:
treat them as a training signal, not as a prediction of LUT / DSP (the notebook compares just one network).
The extra pass that counts them costs a few percent of a training step with TensorFlow and 20-50 % with
torch on a small model (measured on a CPU); `enable_ebops=False` removes it. The pooling
layers have none. The two ways to shrink the bit widths,
`MonoL1` on the quantizer and `beta * EBOPs`, add up: use one of them.

```python
from hgq.utils.sugar import BetaPID, FreeEBOPs

model.fit(x, y, callbacks=[FreeEBOPs(), BetaPID(target_ebops=20_000, p=1e-5, i=1e-5)])
```

**Exporting to hls4ml (`io_stream`), the types derived by hls4ml.** With the default `freeze(model)` the
activation quantizers stay in the frozen model as HGQ2 layers. hls4ml then runs its bit-exact precision
flow, which computes the exact range of every value from the input quantizer and the values of the weights
(interval arithmetic), and sets every type itself: result, weights, bias and accumulator. Nothing to
give by hand; the exported hex layers take part in it, so the model can also contain HGQ2's own layers
(`QDense`...):

```python
import hls4ml
from keras_hexagdly.hls4ml_ext import hex_reuse_config, patch_model_for_hls
from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

frozen, _ = hgly.freeze(model)                       # the quantizers stay as layers
register_hex_gather_layers("Vitis")
patched = patch_model_for_hls(frozen)
config = hls4ml.utils.config_from_keras_model(patched, granularity="name", backend="Vitis")
hex_reuse_config(config, patched)                    # optional: an explicit reuse factor per hex layer
hls_model = hls4ml.converters.convert_from_keras_model(
    patched, hls_config=config, io_type="io_stream", backend="Vitis",
    output_dir="prj", part="xc7vx485tffg1927-2", clock_period=5)
hls_model.compile()                                  # C simulation: equal to the Keras model, bit for bit
```

One quantizer per tensor. An output quantizer (`oq_conf`) on a layer and the input quantizer (`iq_conf`) of the
next one quantize the same tensor twice, and hls4ml merges them into one type: with different grids Keras
rounds twice and hls4ml once, so the C simulation can differ by a step (`patch_model_for_hls` warns). Give
the `iq_conf` to the consuming layer and no `oq_conf` to the producer, except on the output of the model.
With that rule the C simulation of a trained model with learned bit widths equals Keras exactly.

What it needs: the model input must be quantized (give the first hex layer an `iq_conf`, or start with an
HGQ2 layer; `patch_model_for_hls` says so otherwise), and, for `io_stream`, one bit width per tensor for the
activation quantizers (`homogeneous_axis=(0, 1, 2, 3)`): hls4ml stops on a per-channel or per-element one with
`Heterogenous quantization for activations is only supported with IOType=io_parallel` (and with the exported
hex layers those also hit a shape assertion in `io_parallel`, a path that is not validated anyway). The
weights may have a bit width per element. The 3D layers
and `QAvgPool2d` have no export. It relies on a private function of hls4ml (`_produce_kif`, in
`optimizer/passes/bit_exact.py`), checked with hls4ml 1.3.0. `tests/test_hls_bit_exact_csim.py` compares
the C simulation with Keras for a convolution, a max pooling, and a chain with ring sharing, strides and a
`QDense` head: exactly equal.

**Exporting to hls4ml (`io_stream`), the types set by hand.** To choose the types yourself, or when the rest
of the model is plain Keras layers (a `Dense` head, say): hls4ml only handles per-element activation
quantizers in `io_parallel`, so give `iq_conf` / `oq_conf` one bit width per layer
(`homogeneous_axis=(0, 1, 2, 3)`), freeze with `activation_quantizers="types"` (the activation
quantizers are dropped and their precision is in `types`), and put the types in the hls4ml config:

```python
import hls4ml
from hgq.quantizer import QuantizerConfig
from keras_hexagdly.hls4ml_ext import apply_quantized_types, patch_model_for_hls
from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

register_hex_gather_layers("Vitis")

def per_layer(ints, fracs, k=1):   # one bit width for the whole layer
    return QuantizerConfig("kif", "datalane", k0=k, i0=ints, f0=fracs, overflow_mode="SAT",
                           homogeneous_axis=(0, 1, 2, 3), trainable=False)

inp = keras.Input((13, 11, 2), name="x")
model = keras.Model(inp, hgly.QConv2d(
    4, kernel_size=1, activation="relu", name="hex0",
    kq_conf=weights, bq_conf=weights, iq_conf=per_layer(2, 3), oq_conf=per_layer(3, 2, k=0))(inp))

frozen, types = hgly.freeze(model, activation_quantizers="types")
print(types["hex0"])   # weight, bias, input, result, accum
patched = patch_model_for_hls(frozen)
config = hls4ml.utils.config_from_keras_model(patched, granularity="name")
apply_quantized_types(config, patched, types)   # all the types below, in the right places
```

`apply_quantized_types` (from `keras_hexagdly.hls4ml_ext`, after `register_hex_gather_layers`) does
this, for each quantized layer:

* `weight` / `bias` become `mac_weights` / `mac_bias` of `hex0_linebuffer`, and `accum` its
  `AccumPrecision` (`accumulator="default"` leaves hls4ml's accumulator alone, see below);
* `input` becomes the result precision of the layer that feeds it (`x` here, the input layer);
* `result` is the output quantizer: `hex0_activation` exists because the layer has a fused
  `activation`, and the quantizer comes after it, so its type goes there (without an activation it
  goes on `hex0_linebuffer`).

With these types the C simulation gives, bit for bit, what the Keras model gives
(`tests/test_quantized_csim.py`). The one thing that is not a type of your model is the width of the
accumulator of an exported hex convolution. By default it keeps only the fractional bits of the finer of
the weights and the input, so with narrow quantized inputs each product loses its low bits and the
simulation differs from Keras by a few steps. Two settings (both off by default, so existing exports do
not change) make it exact:

* `config["Model"]["ExactAccumulator"] = True`: every accumulator keeps all the fractional bits of the
  product (`f_weight + f_input`, or those of the bias if finer) and one integer bit for the carry of the
  bias addition. It can also be set on one layer (`config["LayerName"][name]
  ["ExactAccumulator"]`), which then wins over the model. The accumulator is wider, so it costs logic:
  see the measurement in the FPGA section below.
* `config["LayerName"][name]["AccumPrecision"] = "ap_fixed<W,I>"`: the type itself. `types[...]
  ["accum"]` (also `layer.accum_type()`) is the exact one for a layer; it needs the type of the input,
  i.e. an `iq_conf`, or the `result` type of the previous layer passed to `accum_type(input_type)`.
  `AccumPrecision` wins over `ExactAccumulator`. `AccumPrecision` is read per layer only (it does not fall
  back to the `Model` section); `ExactAccumulator` does. The typedef hls4ml generates equals
  `types[...]["accum"]` when the layer has a bias type (`bq_conf`); without one hls4ml's default bias
  type enters the computation and the typedef can differ.

The weight and bias types in `types` are derived from the quantized values themselves (the smallest
signed `ap_fixed` that holds them exactly, negative integer bits included), the activation types from
the quantizer parameters. `freeze` works on Sequential and Functional models, with raw operations
(`+`, `keras.ops.*`), layers called more than once and nested models; `get_config` / `from_config`
round-trip the quantizers, and the quantized layers are saved/cloned (`clone_model`, `.h5`) with their quantizers. `QAvgPool2d` and the 3D layers
have no hls4ml export. Mixed precision / half precision is not supported by HGQ2 on the TensorFlow backend (it
works on torch and JAX: the quantizer outputs are cast back to the compute dtype). An activation
quantizer with a `scaler` or `affine` has no `ap_fixed` type: `freeze(model)` leaves its type out of
`types`, `freeze(model, activation_quantizers="types")` refuses it. `import keras_hexagdly` imports the
quantized layers when hgq2 is installed (a saved QAT model needs them registered to load) and without
hgq2 `hasattr(keras_hexagdly, "QConv2d")` is simply False.

Only what you give a quantizer is quantized: the other layers of a model, the activations without an
`oq_conf`, and the bias without a `bq_conf` stay in float. The 3D layers have no hls4ml export yet.
A model that also contains HGQ2's own layers (`QDense`, ...) trains fine, counts them in the EBOPs, and
exports through the bit-exact flow described above.

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

Beyond the ports:

- [`keras_hexagdly_hls4ml_export.ipynb`](notebooks/keras_hexagdly_hls4ml_export.ipynb) -- a trained hex CNN to an hls4ml project: patching, C simulation, Vitis HLS synthesis and RTL co-simulation (see the FPGA section below).
- [`keras_hexagdly_qat_hgq2_to_hls4ml.ipynb`](notebooks/keras_hexagdly_qat_hgq2_to_hls4ml.ipynb) -- quantization-aware training with HGQ2 and EBOPs, `freeze`, hls4ml and a resource comparison with a uniform-precision design.
- [`cta_lst_hexconv_gamma_vs_noise.ipynb`](notebooks/cta_lst_hexconv_gamma_vs_noise.ipynb) -- a 3-D hexagonal CNN telling a gamma-ray shower from night-sky-background noise in an LST camera, on ctapipe's public sample data (needs `ctapipe`).
- [`pytorch_to_keras_example.ipynb`](notebooks/pytorch_to_keras_example.ipynb) -- train with `pytorch-hexagdly`, save the weights, reload them in `keras-hexagdly` and check that both give the same outputs (in French).

## FPGA export via hls4ml

The hex layers can be synthesised to HLS C++ through
[hls4ml](https://github.com/fastmachinelearning/hls4ml). The layers are replaced
by a fused line-buffer kernel that keeps only the resident rows of the frame in
a shift register, rather than materialising the whole gathered tensor.

Needs **hls4ml >= 1.3.0** (`pip install keras-hexagdly[hls]`; 1.3.0 is the only version tested, and
`register_hex_gather_layers` stops with a message on an older one: hls4ml 1.2 does not register the Keras
handlers of the hex layers by itself and its C simulations of these kernels came out all zeros).

```python
import hls4ml
from keras_hexagdly.hls4ml_handler import register_hex_gather_layers
from keras_hexagdly.hls4ml_ext import patch_model_for_hls, hex_reuse_config, check_hls_config

register_hex_gather_layers("Vitis")
hls_ready = patch_model_for_hls(model)                 # strategy="linebuffer"

config = hls4ml.utils.config_from_keras_model(hls_ready, granularity="name")
hex_reuse_config(config, hls_ready)                    # per-layer ReuseFactor
check_hls_config(config, hls_ready, io_type="io_stream")

hls_model = hls4ml.converters.convert_from_keras_model(
    hls_ready, hls_config=config, io_type="io_stream", backend="Vitis",
)
```

### Dataflow mode: lower latency, more LUT

By default each hex convolution is one fused loop: a pixel waits for the previous
pixel's multiply, so the latency is dominated by the layer, whatever the reuse
factor. Set `Dataflow` on a hex layer to run its line buffer and its MAC as two
concurrent processes instead (`io_stream`, 2D only):

```python
for layer in ("conv1_linebuffer", "conv2_linebuffer"):
    config["LayerName"][layer]["Dataflow"] = True      # default False
```

On a 19x14 board grid with two hex layers this took a whole CNN from 6083 to 824
cycles (x7.4) for about +60 % LUT and +5 % FF (Vitis HLS 2024.2, VC707); it pays
when latency is the constraint and the LUT are available. The key is read per
layer (a layer with a `LayerName` entry does not fall back to the `Model`
section) and ignored with a warning under `io_parallel`. It is validated in C
simulation, bit for bit against the fused kernel (`tests/test_hls_dataflow_csim.py`), and in
RTL co-simulation on a small network, with and without `share_neighbors="ring"`, where it matches the
C result like the fused kernel does (`tests/smoke_cosim_dataflow.py`, needs Vitis HLS). A C simulation
only predicts the RTL while no type overflows: keep the types wide enough.

### Supported scope

**Only `io_stream` and the 2D layers (`Conv2d`, `MaxPool2d`) are supported for
now.** That is the combination that is covered by C-simulation and validated by
RTL co-simulation.

| | `io_stream` | `io_parallel` |
|---|---|---|
| `Conv2d`, `MaxPool2d` | **supported** | not supported |
| `Conv3d`, `MaxPool3d` | not supported | not supported |
| `AvgPool2d` | not supported | not supported |

The unsupported paths are not silently wrong -- they raise. `patch_model_for_hls`
raises `NotImplementedError` on a 3D layer, and `check_hls_config` raises on
`io_type="io_parallel"`. Both accept `allow_unvalidated=True` if you want to
experiment with them anyway, but nothing about their numerics or resource usage
is guaranteed. `AvgPool2d` has no export path at all yet, so
`patch_model_for_hls` refuses it unconditionally.

A `Conv2d` built with `activation=` keeps it in the exported model: the
activation is re-applied as a separate `keras.layers.Activation` after the
layer's replacement.

`hex_reuse_config` matters more than it looks: hls4ml's
`config_from_keras_model(granularity="name")` writes `ReuseFactor=1` into every
layer entry it recognises, which overrides the model-level value -- but custom
layers get no entry and inherit the model-level one instead. Without an explicit
per-layer setting, a single model ends up mixing two different reuse factors.

### Accumulator width: `ExactAccumulator`, and the cost against a square layer

The accumulator of an exported hex convolution keeps, by default, the fractional bits of the finer of the
weights and the input; `config["Model"]["ExactAccumulator"] = True` keeps all the bits of each product
instead (`f_weight + f_input`), so the C simulation of a quantized model equals its Keras model bit for
bit. It makes the accumulator wider, and that costs logic.

Vitis HLS 2024.2 estimates for one 12x12 convolution, 8 channels to 16, the same on-grid random weights in
every run, `io_stream`, `Strategy='Latency'`, reuse factor 1, `xc7vx485tffg1927-2` (XC7VX485T-2FFG1927C),
5 ns. The square row is hls4ml's own `Conv2D` 3x3 (`padding="same"`, 9 taps instead of the hexagon's 7),
given the same accumulator type as the hex layer's default one, so that only the geometry and the kernel
differ:

| types (input / weight) | layer | accumulator | LUT | FF | DSP | BRAM_18K | latency (cycles) |
|---|---|---|---|---|---|---|---|
| `<6,3>` / `<7,2>` | hex, default | `<16,11>` | 38 989 | 40 018 | 0 | 14 | 195 |
| | hex, `Dataflow` | `<16,11>` | 40 151 | 39 060 | 0 | 33 | 194 |
| | hex, exact | `<20,12>` | 38 152 | 47 338 | 130 | 14 | 195 |
| | square 3x3 | `<16,11>` | 43 589 | 40 291 | 0 | 17 | 207 |
| `<10,4>` / `<9,3>` | hex, default | `<19,13>` | 54 354 | 53 049 | 0 | 19 | 195 |
| | hex, `Dataflow` | `<19,13>` | 55 230 | 52 156 | 0 | 51 | 194 |
| | hex, exact | `<26,14>` | 46 417 | 74 660 | 272 | 19 | 195 |
| | square 3x3 | `<19,13>` | 66 600 | 56 085 | 0 | 24 | 207 |
| `<16,6>` / `<16,6>` | hex, default | `<28,18>` | 38 584 | 126 550 | 755 | 28 | 196 |
| | hex, `Dataflow` | `<28,18>` | 39 193 | 123 690 | 755 | 78 | 194 |
| | hex, exact | `<39,19>` | 116 642 | 238 235 | 925 | 28 | 199 |
| | square 3x3 | `<28,18>` | 46 107 | 144 452 | 971 | 36 | 207 |

Against the square layer the hex layer has the cost its 7 taps promise: **755 DSPs against 971 (0.78 =
7/9)** with the wide types, 11 to 18 % fewer LUTs, 1 to 12 % fewer flip-flops and 18 to 22 % fewer BRAMs,
and a latency of 195 cycles (14 x 13 pixel iterations: the 12x12 image plus the rows and columns that flush
the pipeline) against 207. The `Dataflow` kernel reaches the same latency
as the default one on this layer for 2 to 3 times the BRAM (the stream between its two processes), so the
default is the one to use here; `Dataflow` was designed for larger layers and was not measured with a reuse
factor above 1.

With narrow types the exact accumulator mostly moves work from LUTs to DSPs and flip-flops; with wide
types it triples the LUTs. The latency and the clock are about the same. These are synthesis estimates,
not results after place and route, for dense weights: pruned weights (a weight at 0 bit is a
multiplication that is not built) should lower the difference, which was not measured. It is therefore
off by default and meant for checking that the firmware does what the trained model does; the
`AccumPrecision` of a single layer is available when only some layers need it.

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

Most of the suite no longer lives here. It has moved to
**[hexagdly-oracle](https://github.com/YugnatD/hexagdly-oracle)**, the shared
test repo for this library and
[pytorch-hexagdly](https://github.com/YugnatD/pytorch-hexagdly): hand-verified
layer outputs, `share_neighbors` weight-sharing oracles, `depth_padding`,
mixed precision, serialization, edge cases, indexed-equivalence and the hls4ml
export tests (including C-simulation). `tests/` here keeps only what is
genuinely local.

To run the full suite, check the oracle out as a sibling directory:

```
git clone https://github.com/YugnatD/hexagdly-oracle
PYTHONPATH=hexagdly-oracle/src pytest tests/ hexagdly-oracle/tests/
```

Verified to pass on all three Keras 3 backends (set `KERAS_BACKEND=tensorflow|torch|jax`
before importing keras; tensorflow is the default if unset):

```
KERAS_BACKEND=tensorflow   # 1157 passed, 16 skipped
KERAS_BACKEND=torch        # 1149 passed, 24 skipped
KERAS_BACKEND=jax          # 984 passed, 189 skipped (slower: per-shape JIT compile)
```

A GitHub Actions workflow ([.github/workflows/test.yml](.github/workflows/test.yml))
runs this matrix (3 backends x 3 Python versions) plus `ruff check`/`ruff format --check`
on every push and PR, checking out the oracle repo as part of the job.

Note for GPU users on the torch backend: the Keras torch backend runs on CUDA
when a GPU is visible, and PyTorch defaults to `cudnn.allow_tf32 = True`, so
convolutions are computed in TF32 (~1e-3 relative precision). That is enough to
break equivalence assertions on hex kernels, which are wide by construction
because dilation is done by zero insertion. The oracle's `conftest.py` pins the
flag off for the test session; the library itself never touches global torch
settings.

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
