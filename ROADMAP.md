# Roadmap

Ideas and open questions for keras-hexagdly, most of them about putting a hexagonal CNN on an FPGA.
Nothing here is promised. Each item says why it was written down and what is known about it; "not
measured" means exactly that.

## Ideas, by expected usefulness

### 0. A pixel mask for grids with holes (camera outlines)

**Why.** A hexagonal camera mapped on a rectangular offset grid leaves positions that are not pixels (432
real pixels on a 24x24 = 576 grid for the camera this library is aimed at). The kernels only know the
rectangle: their `in_bounds` test makes the rectangle's border behave as zero padding. A hole inside the
grid is not handled, so after a convolution with a bias (and an activation) a position that does not exist
holds a non-zero value, and its real neighbours read it in the next layer. Training and hardware agree with
each other (the C simulation is bit-exact against Keras), but both compute something the detector does not.
How much this matters for the accuracy was not measured.

**What.**

1. *Correctness (software and hardware).* A constant boolean grid, given to the layers (or a `HexMask`
   layer), multiplies the output of every layer so that holes stay exactly 0 and are read as zero padding by
   their neighbours. It is propagated through strides and pooling (an output pixel exists if its centre pixel
   exists, to be defined per layer). In the kernels it is a constant table: the emitted value is replaced by
   0 where the pixel does not exist. The EBOPs should then count only the real pixels.
2. *Skipping the work (hardware).* Where the pixel does not exist, do not read a token, do not run the MAC,
   do not write a token: the streams between layers carry 432 values instead of 576.

**What to expect from 2, honestly.** The cycle count of the line-buffer kernels is set by the scan of the
whole grid (one iteration per grid position at II=1), and the multiplier bank is shared across pixels, so at
`ReuseFactor=1` skipping the MAC saves neither latency nor resources, only power. The gain should appear when
the MAC is the bottleneck: with `ReuseFactor > 1` (or the `Dataflow` MAC process) the trip count falls from
576 to 432, about 25 %. Not measured; a prototype on one layer with `ReuseFactor=4` would tell. Skipping the
holes in the scan itself would need an addressable line buffer instead of shift registers, and per-row
variable trip counts cost more pipeline fills than they save.

**Related.** Nothing of this kind exists in the library today: the only masks are the neighbour-validity
slots (`-1`) of the gather path and the `in_bounds` test of the kernels, which concern the border of the
rectangle only. `nnet_hex_conv_linebuffer_stream.h` cites an `hls/hexcnn_mask` folder of the
digicam-tdscan-triggering repository, but that folder is not in the repository (nor in its git history); its
`_interior_pixels` mask in `test_hexagdly_vs_tdscan.py` only selects pixels with all their neighbours, to
compare two implementations.

**Effort.** Medium for 1 (layers, propagation rules, kernel tables, tests), medium to large for 2.

### 1. A regression test for the initiation interval (II) and the latency

**Why.** A defect in the fused hex-convolution kernel held its main loop at II=2 (twice the latency) while
the C simulation stayed bit-exact, so no test caught it. It showed only in the Vitis HLS log
(`Unable to enforce a carried dependence constraint`), found while comparing against hls4ml's square
`Conv2D`.

**What.** A test that synthesises a small layer (12x12, a few channels) with Vitis HLS and checks that the
main loop of each kernel (`ReadRows_ReadCols`, and `MacPixels` in `Dataflow` mode) reports `Final II = 1`,
and that the latency matches `R_TOTAL x C_TOTAL x II + depth`. It needs Vitis HLS, so it would be a marked
test run by hand or on a dedicated runner, not on GitHub's hosted CI. The scripts used for the README
measurements (one layer, `Strategy='Latency'`, reuse factor 1, one process per synthesis) are the
starting point; they are not in the repository yet.

**Effort.** Small.

### 2. A fast estimator of latency and DSPs before synthesis

**Why.** A synthesis takes 10 to 15 minutes. Two things were checked against Vitis and agree:

* latency of the fused kernel = `(rows + flush rows) x (columns + flush columns) x II + pipeline depth`
  (14 x 13 x 1 + about 13 = 195 cycles for a 12x12 image);
* DSPs of a hex convolution = `ceil(PF x K x C_in x C_out / RF)`, with K = 7 for `kernel_size=1`
  (755 against 971 for a square 3x3 with 16-bit types: 0.78 = 7/9).

A function giving these in a second for a model and an hls4ml config would make architecture exploration
much faster, in the spirit of the EBOPs.

**Not known.** LUT and FF have no clean formula yet (they depend on the bit widths, on whether Vitis maps a
multiplication to a DSP or to LUTs, and on the target part).

**Effort.** Medium (latency and DSP), larger if LUT/FF are wanted.

### 3. Exporting a model that mixes these layers with HGQ2's own (`QDense`, ...)

**Why.** Only the hexagonal layers are quantized in the notebook; the dense head runs in one uniform
`ap_fixed` (`<24,12>`, because the logits reach 84). A real model nearly always has a head, and the learned
bit widths would save there too.

**What works today.** The *training* side: a model with `QConv2d` and HGQ2's `QDense` counts both in the
EBOPs (checked: convolution 168 192 + head 55 320 = the 223 512 that `FreeEBOPs` reports), and `freeze`
leaves the `QDense` untouched.

**What fails today.** The *export*: as soon as the model contains an HGQ2 layer, hls4ml follows HGQ2's own
path, where every layer must say which precision it produces (`produce_kif`), and the exported hex layers
cannot:

    NotImplementedError: No implementation of produce_kif for ... (HHexConvLineBuffer)

**What to do: two routes.**

* *A. Make the exported hex layers acceptable to that path*: implement `produce_kif` (and whatever else the
  path asks for) for `HHexConvLineBuffer` and `HHexPoolLineBuffer`, and decide how the types of the hex layers
  (today set through `apply_quantized_types`) and the quantizers of the HGQ2 layers (which carry their own)
  meet in one configuration. Needed only to export models built with HGQ2's own layers. Check first what
  hls4ml's HGQ2 path requires from a custom layer.
* *B. A `QDense` of this library* (a Keras `Dense` with HGQ2 quantizers, the same pattern as `QConv2d`).
  After `freeze` the model holds only ordinary layers, so hls4ml uses its official handlers and never enters
  the HGQ2 path; the learned types are given in the config as for the hex layers (the `Dense` precision keys
  `weight`, `bias`, `accum`, `result` should take them, not verified). Simpler and consistent with the
  rest of the library; the cost is one more class to maintain, and models that already use HGQ2's layers
  still cannot be exported.

Which one: A has the higher ceiling, B the lower risk. A covers every HGQ2 layer (dense, convolution, batch
normalisation, ...), and hls4ml's precision propagation (`model/optimizer/passes/bit_exact.py`) is a
per-layer-class registry (`_produce_kif.register(...)`) whose `Conv2D` entry is about thirty lines: put the
input precisions in columns of neighbours (`im2col`), multiply by the kernel with interval arithmetic, read
the output precision. The hex version would use this library's neighbour tables. If it works, hls4ml would
derive every type, accumulators included, and `apply_quantized_types` / `AccumPrecision` would no longer be
needed for such models: `freeze`, `patch_model_for_hls`, convert. To settle it, a small spike first:
`produce_kif` for the hex convolution alone (no weight sharing), a model with a `QDense` head, and the C
simulation against Keras. Unknown before that: the parity-dependent kernels and the sharing modes; whether
that flow overwrites the result and accumulator types the kernels' templates read (today they read what
`apply_quantized_types` sets); and whether it works in `io_stream` with the line-buffer kernels.

**Effort.** A: medium to large. B: medium.

### 4. Batch-normalisation folding

**What.** `fold_batchnorm(model)`: fold a `BatchNormalization` that follows a hex convolution into its
kernel and bias before quantizing, so that it costs nothing in hardware. Standard before an FPGA export;
hls4ml's own fusing does not apply to these custom layers.

**Effort.** Small to medium.

### 5. Post-training quantization with calibration

**What.** Start from a trained float model, measure activation ranges on data, and derive the `ap_fixed` types
(then `apply_quantized_types` as usual). Many people start there before doing QAT.

**Not known.** How much accuracy is lost without retraining on this kind of model.

### 6. One-call export check

**What.** `export_check(model, x)`: patch, convert, C-simulate, compare with Keras, and print a report. This
is what the quantization notebook does in six cells.

## Smaller or more speculative

* **Quantization per ring** (one quantizer per hexagonal ring, between "per weight" and "per layer"). No
  evidence yet that it is worth it.
* **Structured pruning that changes the hardware** (dropping whole taps or channels). A weight at 0 bit is
  already pruned in training, but it was never measured whether the generated design drops the matching
  multiplications, especially with a reuse factor above 1.
* **EBOPs against measured resources.** The EBOPs are documented as a training signal, not as a prediction of
  LUT/DSP. A calibration over several modes (`Latency`/`Resource`, `Dataflow`, reuse and parallelization
  factors) was judged premature while the kernels are still moving. Worth doing once they are stable,
  particularly if `BetaPID` is used to aim at a firmware size. Related: the EBOPs assume a fully parallel
  layer, while the line-buffer kernel computes one pixel per iteration (the sum over all pixels overstates
  the hardware of a layer that is time-multiplexed, and weighs layers of different resolution unequally).
* **`parallelization_factor` for the EBOPs**, as in HGQ2's own convolutions. Dropped for now for the
  reason above.

## Known limits and open questions

* **`Dataflow` mode.** Now at II=1 like the fused kernel (194 cycles on the 12x12 test layer, three sets of
  types). It keeps a structural cost: BRAM (33, 51 and 78 for the narrow, mid and wide types, against 14, 19
  and 28 for the fused kernel), because of the stream between its two processes. Its reuse-factor > 1 behaviour
  was not re-measured after the changes.
* **3D kernels** (`nnet_hex_conv_linebuffer_3d_stream.h`, `nnet_hex_pool_linebuffer_3d_stream.h`) carry a
  similar emit counter (`cur_emit_p`) but have a different structure (whole frames are buffered). They were
  not looked at, and their export is not a validated path.
* **Mixed precision** is not supported by HGQ2 on the TensorFlow backend.
* **`hgq2`** has no upper bound in `pyproject.toml`. Versions 0.1.6, 0.1.9 and 0.2.0 were tested; 0.2.0
  needs Keras >= 3.15, hence Python >= 3.11.
* Python 3.11 and the real GitHub CI were not run before the 0.7.0 work.
* A model frozen with a nested model cannot go through `patch_model_for_hls`, and the keys of `types` of a
  nested model carry a `<nested>/` prefix.
