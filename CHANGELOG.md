# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.7.1] - 2026-11-13

Fixes from four audits of the code (bugs and performance), with regression tests
(`tests/test_robustness.py`, and new cases in `tests/test_hex_linebuffer_csim.py`,
`tests/test_hls_bit_exact_csim.py` and `tests/test_hls_accum_type.py`). What only a Vitis HLS synthesis
shows (II, clock, area) has no test in the suite; the numbers quoted are from Vitis HLS 2024.2.

### Added

- **hls4ml export of the 3D layers (`Conv3d`, `MaxPool3d`), `io_stream`, experimental.** Exact in C
  simulation and validated by RTL co-simulation on small networks; listed as "supported (experimental)"
  in the README.
  `patch_model_for_hls` used to refuse them (`allow_unvalidated=True` gave a single 3D kernel, hex stride 1
  and depth stride 1 only, whose synthesis never finished). A 3D layer is now exported as a depth stage
  followed by the 2D layer: `HexDepthStack` + `HexConvLineBuffer` for a convolution (the frames of each
  depth window stacked as channels, the 3D kernel laid out as `(k, depth_size * in_channels, filters)`),
  `HexDepthPool` + `HexPoolLineBuffer` for a max pooling (the max over depth, then over the hex
  neighbourhood). The depth stage (`nnet_hex_depth_stream.h`) is a delay line of `depth_size - 1` frames,
  the least a streamed 3D layer can store, with no arithmetic; the 2D kernels gained a frame count
  (`n_frames`) and run over the output frames one after the other. Any hex stride, depth stride and
  `depth_padding`, `share_neighbors`, `Strategy`, `ReuseFactor`, `Dataflow`, `hex_reuse_config`, and the
  HGQ2 bit-exact flow (`QConv3d`, `QMaxPool3d`) are inherited from the 2D kernels. `io_parallel` has no 3D
  kernel: the conversion raises. `allow_unvalidated` of `patch_model_for_hls` is now ignored.
  Vitis HLS 2024.2, 4 ns: a network `Conv3d -> MaxPool3d -> Conv3d -> Flatten -> Dense` co-simulated in RTL
  in four configurations (`Resource` RF=1, `Resource` RF=4 with `Dataflow`, `Latency` RF=1 and RF=4), RTL
  equal to the C simulation each time, about two minutes for synthesis and co-simulation; a
  `Conv3d(4, (3, 1))` on 8 frames of 12x12x4 at RF=4 with `Dataflow`: 3480 cycles, 3.03 ns, 84 DSP,
  33 117 FF, 21 814 LUT, 22 BRAM_18K, of which 6 BRAM, 1 635 FF and 960 LUT for the depth stage. The
  hardware of a 2D model is unchanged (identical latency, clock and resources on three configurations).
  A third loop nested over the frames set the clock of the pooling (4.78 ns for 4 frames of 12x12) and
  held the fused convolution at II=2 with `Resource` at `ReuseFactor=1` (714 cycles where `Latency` took
  361): with several frames both kernels run as one flat loop with narrow counters (3.03 ns; 365 cycles).
  `patch_model_for_hls` refuses a `BatchNormalization` or a `Concatenate` applied to the `(D, H, W, C)`
  output of a 3D layer: hls4ml 1.3.0 reads the scale and bias of the first out of bounds there (wrong
  values, no error) and has no kernel for the second. `check_hls_config` refuses a 3D model under
  `io_parallel`. `apply_quantized_types` puts the input type of a 3D layer on what feeds its depth stage
  (so the delay line stores the quantized frames) and no longer gives a max pooling without an output
  quantizer the wide type (it keeps the type of its input; 2D too). Tests: `tests/test_hex_3d_export.py`
  (Keras), `tests/test_hex_3d_csim.py` and a 3D case in `tests/test_hls_bit_exact_csim.py` (C simulation).
  `HexConvLineBuffer3D` / `HexPoolLineBuffer3D` and their kernels are no longer produced; they stay in the
  package so that a model saved with them still loads.

### Fixed

- **hls4ml export: the hex convolution wrapped around with small weights and full-range inputs.** The MAC
  cast each tap, and each ring sum of taps, into the accumulator type before multiplying. That type is
  sized for the sum of the products `w * x`; with small weights (HGQ2's bit-exact flow, `ExactAccumulator`,
  `AccumPrecision`) it is narrower than the data, the tap wrapped and the wrapped value was multiplied: the
  C simulation differed from Keras (846 values of 1056 on a ring-shared `QConv2d` with weights 0.5 / -0.25
  and inputs up to 3.9, off by up to 4; the tests used inputs in [0, 1]). The taps and ring sums now have
  their own type (`ringsum_t`: the input type plus `ceil(log2(largest ring))` integer bits), the product is
  taken at full precision and only the product enters the accumulator. Results are unchanged wherever
  nothing wrapped, with one exception: an `AccumPrecision` with fewer fractional bits than the input used
  to truncate each tap before the product and now truncates the product, which can differ by one LSB of
  the accumulator. 2D and 3D kernels, `io_stream` and `io_parallel`.
- **hls4ml export: the accumulator took its sign from the weight type alone.** Unsigned weights on signed
  data (or with a negative bias) gave an `ap_ufixed` accumulator in which every negative sum wrapped (137
  outputs of 320 wrong on a layer with `ap_ufixed<8,1>` weights and a wider result type). It is now signed
  unless the weights, the input and the bias are all unsigned, and an unsigned bias gets its sign bit in a
  signed accumulator (a bias of 12 typed `ap_ufixed<8,4>` came out as -4).
- **hls4ml export: the `Dataflow` convolution missed its II with a `ReuseFactor` above 1.** The cap on
  the adders of its MAC process was the number of multipliers, which left no adder for the per-cycle
  accumulation nor for the ring sums of a `share_neighbors` layer: Vitis scheduled II=15 for a target of 4
  on a ring-shared kernel-2 layer (1221 cycles for 81 pixels) and 3 for 2 on a plain one, without an error.
  The cap now counts every addition of a pixel (`hex_lb_add_budget`): both reach their II (1223 -> 335
  cycles for the test project). The same budget replaces the cap of the fused `Latency` variant.
- **hls4ml export: `Dataflow`, `AccumPrecision` and `ExactAccumulator` set in `config["Model"]` or in a
  `LayerType` entry were ignored** whenever the layer had a `LayerName` entry, which
  `config_from_keras_model(granularity="name")` gives every layer. They are now read from the layer's
  entry, else its type's, else the model's. `check_hls_config` reads `LayerType` entries too.
- `patch_model_for_hls`: a model nested in the model raised an error that blamed a shared layer; it now
  says nested models are not supported. A subclass of a hex layer went through unpatched without a word;
  it is now replaced like its base class.
- A hex convolution called on another number of channels than it was built for ran a grouped convolution
  on the mismatched kernel, without an error; it is now refused, like `keras.layers.Conv2D` does, and so is
  an input of the wrong rank. `filters` must be a positive integer. A 3D layer with fewer frames than its
  depth size raises a `ValueError` (it returned an empty tensor on TensorFlow and JAX).
- `Conv2d_CustomKernel` / `Conv3d_CustomKernel`: `layer.trainable = False` was lost by
  `from_config(get_config())`.
- `build_parity_offset_tables` checks the offsets against the whole neighbour table, border included, on
  every grid (it did on small grids only).
- **The neighbour table was wrong under a `mixed_float16` policy** on grids of more than 2048 pixels
  (`build_neighbor_table` carries pixel indices as values through a probe layer, which followed the global
  policy), and so was the model `patch_model_for_hls` built from it, without any error. The probe is now
  float32 whatever the policy (as is the one behind `share_neighbors="ring"`).
- **hls4ml export: the result type of the hex max pooling**, when left to hls4ml (`auto`), was the model
  default precision instead of the type of its input: after a layer with a wider result type the pooled
  values wrapped silently. It now takes the input type, like hls4ml's own pooling layers.
- **hls4ml export: small grids** (no fully-interior output pixel of each column parity: 3x3 at kernel
  size 1, 5x5 at kernel size 2, a single output column...) stopped the conversion with "Grid too small".
  The tap offsets do not depend on the grid: they are then measured on a larger virtual grid and checked
  against the real neighbour table.
- **`register_hex_gather_layers()`** registered the templates and kernel sources with one backend only,
  the first one asked for (default `"Vivado"`): a later `register_hex_gather_layers("Vitis")` did nothing
  and the conversion failed on a missing header. Any of `"Vivado"`, `"Vitis"` or nothing now covers both:
  the kernel sources go to both backends and the templates to `"Vivado"` only (the Vitis flow applies
  them too; registered twice, every layer was formatted twice and its warnings printed twice).
- **`Conv2d`, `Conv3d` and their quantized versions could not be built inside a traced function** on
  TensorFlow and JAX: a `keras.Sequential` without an `Input`, built by its first `fit`, failed. `build`
  drew the initial values itself and converted them to NumPy, which a symbolic tensor does not allow, and
  kept the ring maps of a shared layer as backend tensors tied to that trace. The weights now have a real
  initializer (He-normal scaled by the fan-in of the whole hex kernel, as before) and the ring maps stay
  NumPy arrays. **The initial values for a given seed are not the ones of 0.7.0** (same distribution;
  each weight has its own draw instead of a slice of a common one). `keras.utils.set_random_seed` still
  makes them reproducible.
- **Max pooling on TensorFlow: the gradient at ties.** Where several cells of a window hold the maximum
  (zeros after a ReLU, the zero padding), `tf.maximum` gave the whole gradient to one of them; torch and
  JAX share it. The three backends now share it (`MaxPool2d`, `MaxPool3d`): same values, and the same
  gradients everywhere. Training a model with max pooling on TensorFlow can give slightly different
  weights than with 0.7.0.
- `Conv2d_CustomKernel` / `Conv3d_CustomKernel` check their input like the other layers: a wrong rank, an
  unknown grid size or another channel count than the kernels' is a `ValueError` (the backend ran a
  grouped convolution on the mismatched kernel, silently), and sub-kernels that disagree on their
  channels are refused. Their kernels take the dtype of the layer: with `dtype="float64"` they were
  float32 and the first call failed.
- A max pooling under a float64 policy returned float32 on torch (`keras.ops.maximum` of two float64
  tensors, Keras 3.12).
- The hex layers have a `compute_output_shape`: `keras.layers.TimeDistributed` (and whatever else asks
  for the shape without calling the layer) works on them.
- `weight_maps_2d` / `ring_maps_2d` return copies: writing into the returned arrays changed the sharing
  of every layer built afterwards.
- `load_torch_state_dict`: a `Dense` of a subclassed model (no functional graph to read) was loaded
  transposed only, without a word, also after a `Flatten` of a tensor with channels, where that is wrong.
  It now warns and says what to do. A `bias` in the state dict for a `Dense` or `Conv2D` built with
  `use_bias=False` is an error, as it already was for the hex layers.
- `notebooks/hexplot.py`: the picture of an unshared (or `"sym"`) kernel was turned by 180 degrees (the
  cell drawn above the centre held the weight applied to the cell below); it now shows each weight on the
  neighbour it applies to. `kernel_to_hextensor` works on a `Conv2d_CustomKernel`. `plot_hextensor` set
  the axis limits from the wrong dimensions (a grid much wider than tall was cut).
- The extras `keras-hexagdly[qat]` and `keras-hexagdly[hls]` require `quantizers<2`: with quantizers 2.0.0
  neither hgq2 0.1.x nor hls4ml 1.3.0 can be imported (`cannot import name 'get_fixed_quantizer_np'`).
- README: the `Conv3d` example of the same-padding section passed `in_channels` (a `TypeError`).
- `load_torch_state_dict`: a `Dense` that reads a flattened tensor was loaded with a plain transpose,
  although PyTorch flattens `(C, ...)` and Keras `(..., C)`: no error, a wrong model. The input features
  are now reordered, for a `Flatten` (or a `Reshape` to a vector) of a tensor of any rank, also behind
  layers that keep the shape (`Dropout`, an activation, a second `Flatten`...), inside or after a nested
  model, and when the flattened features are concatenated with other vectors (only their columns move). The new argument
  `flatten_order="channels_last"` turns it off, for a PyTorch model that permutes to channels-last itself
  before flattening. `verify_against` warns when the PyTorch model is in training mode.
- The line-buffer layers `patch_model_for_hls` builds masked the out-of-grid taps by a product with 0:
  a non-finite value at pixel 0 (the placeholder those taps read) became NaN on every border output
  (`inf * 0`). They are now selected out.
- NumPy integers (`np.int64(1)`...) are accepted for `kernel_size`, `pool_size` and `strides` (an integer
  array for the 3D pairs), and stored as plain ints, like `filters`.
- Too few rows for the stride (e.g. one row at `strides=2`) is a `ValueError` that says so, instead of a
  backend concatenation error (the `assert` guarding it was off by one).
- The pooling layers on a dynamic spatial dimension raise the same clear `ValueError` as the convolutions.
- `notebooks/hexplot.py`: `kernel_to_hextensor` showed `relu(w + b) - b` instead of the weights for a
  `strides=1` layer with a fused activation, and weights re-quantized at 8 bits for a `QConv2d` (it now
  shows the layer's quantized weights, ring-shared layers included). A weight that is exactly 0 (pruned)
  was left out of the picture, and a fully pruned kernel raised an error: the footprint is now the
  kernel's geometry.

### Changed

- **hls4ml export: `Strategy` now does what the documentation says for the fused `io_stream` convolution.**
  The `#pragma HLS PIPELINE` selected by the strategy sat inside an `if`, where Vitis HLS ignores it
  (warning HLS 207-5575): `Latency` and `Resource` gave the same hardware, never explicitly pipelined. The
  loop now exists in versions chosen at compile time. `Latency` pipelines the per-pixel loop at
  `II = ReuseFactor`; `Resource` is what both were before. At `ReuseFactor=1` nothing changes. **A model
  exported with `Strategy="Latency"` and `ReuseFactor > 1` gets different hardware than before**: faster
  (about three times on the two layers below), and larger. Measured at `ReuseFactor=4` against `Resource`, same DSP count: 735
  cycles instead of 2214 for 1.75x the LUT on a 12x12, 4->4 layer (64 products); 369 instead of 1105 for
  2.7x the LUT on an 8x8, 8->8 one (448 products); not measured beyond. Use `Resource` (what
  `hex_reuse_config` sets) to keep the previous hardware; `check_hls_config` now says so for every hex
  convolution left on `Latency` with a `ReuseFactor` above 1 (`Latency` is hls4ml's default). In that
  variant the additions are capped like the multiplications: without it Vitis was still scheduling the
  448-product layer after 7 minutes and 11.6 GB (39 s and 0.5 GB with it). The window process of the
  `Dataflow` kernel is always at II=1. The 3D stream kernel is never pipelined, whatever the strategy: the
  ignored pragma is removed and the header says so.
- hls4ml export: the max pooling kernels are pipelined at II=1 instead of `II = ReuseFactor`. They have no
  multiplier, and inheriting the model-level `ReuseFactor` only slowed them down (228 cycles instead of 62
  on a 6x6x4 pooling at `ReuseFactor=4`). The `io_stream` 2D kernel derives the output row and column from
  the loop indices instead of counting them: the counters were a loop-carried path that set the clock at
  II=1 (estimated 3.57 ns on an 8x8x2 pooling, against 4.57 ns with them).
- hls4ml export, `io_stream` kernels with a stride that is not a power of two: the traversal no longer
  divides by the stride (Vitis built dividers: at stride 3, a window process 39 stages deep with 4 DSP and
  5018 FF, now 6 stages, 0 DSP and 1317 FF, 140 -> 106 cycles). Same values.
- hls4ml export, `Dataflow` convolution: the stream between its two processes is held in shift registers
  instead of block RAM (Vitis HLS; it took one BRAM per 18 bits of width whatever the depth), and the
  window process counts rows and columns in counters of the width they need (estimated clock 3.46 ->
  3.05 ns at a 4 ns target).
  On a 19x14 model with two ring-shared `QConv2d` (8 -> 16 and 16 -> 32 channels): 331 -> 236 BRAM_18K
  for the same latency (353 cycles), and RTL co-simulation equal to the C simulation. The LUTs of the
  shift registers (about one per bit of stream width) are not in the HLS estimate.
- `check_hls_config`: the warning about `Strategy="Latency"` with `ReuseFactor=1` before a
  `Flatten`/`Reshape` no longer calls the RTL wrong: the failure was seen with earlier kernels and both
  cases tried now co-simulate correctly. It asks for a co-simulation instead. No warning about `Latency`
  for a `Dataflow` layer, which has a single implementation.
- hls4ml export: the folded MAC sums the products of a cycle on their own and adds them to the accumulator
  once, so the loop-carried path is a single adder (a `ReuseFactor` that does not divide the number of ring
  sums left a chain of adders and selects there, which missed timing). `recommended_reuse_factor` prefers a
  nearby divisor of the number of ring sums (at most twice the smallest factor that fits: fewer
  multipliers, up to twice the latency of that layer).
- `AvgPool2d` computes the number of in-grid cells of its windows once per grid instead of at every call
  (1.4x to 2x faster in eager mode). Under `jax.jit` the result can differ from before by one unit in
  the last place (the count is now a constant).
- `share_neighbors="ring"`: the ring maps come from one probe per kernel size instead of one per cell
  (5.4 s -> 0.13 s at kernel size 5, in the layer's constructor). `build_neighbor_table` is vectorised.
- `MaxPool2d` pads the border with zeros (inherited from HexagDLy): now documented.

## [0.7.0] - 2026-10-26

### Added

- **Quantization-aware training of the hexagonal layers with HGQ2** (optional: `pip install
  keras-hexagdly[qat]`, needs Python >= 3.10 and HGQ2 >= 0.1.6; tested with 0.1.6 to 0.2.0, the latter needs Python >= 3.11). `QConv2d`, `QConv3d`, `QConv2d_CustomKernel`,
  `QConv3d_CustomKernel`, `QMaxPool2d`, `QMaxPool3d` and `QAvgPool2d` are the hexagonal layers with HGQ2
  quantizers on the kernel (`kq_conf`, HGQ2's weight config by default), the bias (`bq_conf`), the input
  (`iq_conf`) and the output after the activation (`oq_conf`); all but the kernel are off unless
  given. The pooling layers have no weights, so only their input and output can be quantized; the
  custom-kernel layers train their kernels only with `trainable=True`. The bit widths are fixed or
  learned under a regulariser that pushes them down (a weight that reaches 0 bit is removed). Every
  sub-kernel of the hexagonal decomposition has its own quantizer; with `share_neighbors` (any mode)
  the shared weights are quantized. The quantizer variables and configs are saved with the model
  (`model.save()` / `load_model()`, also in a fresh process that only imports `keras_hexagdly`:
  the package imports the quantized layers when hgq2 is installed). `import keras_hexagdly` still
  works without hgq2, where the quantized names are simply absent (`hasattr` is False).
- **Export through hls4ml's bit-exact flow: no types to give, HGQ2's own layers allowed.** With the default
  `freeze(model)` the activation quantizers stay in the model; hls4ml then derives every type (result,
  weights, bias, accumulator) by interval arithmetic from the input quantizer and the values of the weights.
  The exported hex convolution and max pooling now take part in that flow (they register `_produce_kif`
  with hls4ml, from the layer's own neighbour table), and the convolution's accumulator is the one hls4ml
  computed. A model with `QConv2d`, `QMaxPool2d` and HGQ2's `QDense` converts and its C simulation equals
  the Keras model exactly (`tests/test_hls_bit_exact_csim.py`: convolution, max pooling, and chains with
  ring sharing, strides, learned bit widths, a global average pooling and a `QDense` head). Before, any HGQ2 layer in the model
  stopped the conversion with `No implementation of produce_kif for HHexConvLineBuffer`. One quantizer per tensor
  is the rule that keeps it exact: an `oq_conf` followed by the `iq_conf` of the next layer quantizes the same
  tensor twice, hls4ml merges the two into one type and the double rounding of Keras can differ from it by a
  step with different grids (`patch_model_for_hls` warns). The model input must be quantized (an `iq_conf`
  on the first hex layer): `patch_model_for_hls` says so otherwise. Needs an hls4ml
  with that flow (1.3.0 tested); it uses a private function of it, `_produce_kif`. The 3D layers are not
  covered. The manual route (`freeze(..., "types")` + `apply_quantized_types`) is unchanged.
- **The exported hex convolution and pooling nodes have an image-shaped output `(H, W, C)`**, like hls4ml's
  own convolutions (they were `(H * W, C)`). The stream is the same (C values per token, H * W tokens): the generated
  firmware is identical (compared file by file on a two-convolution, one-pooling model), but the layers
  after a hex node can now read the image shape: a global average pooling (`QGlobalAveragePooling2D`) or
  HGQ2's per-pixel quantizers.
  `tests/test_hls_bit_exact_csim.py` checks the shapes and a hex convolution -> max pooling -> global average
  pooling -> `QDense` chain bit for bit. hls4ml 1.3.0 hands the layers after a global 2D pooling a `(1, C)`
  array where a `(C,)` vector is expected (an HGQ2 quantizer there fails with an `AssertionError` on the
  ranks); the bit-exact registration reshapes it. The hardware cost of hls4ml's global pooling was not
  investigated (see ROADMAP.md).
- `patch_model_for_hls` replaces a plain `keras.layers.ReLU` by `Activation("relu")` (the same function). hls4ml
  1.3 turns a `ReLU` layer into a thresholded ReLU whose threshold gets the type `ap_ufixed<2,32>` in the
  bit-exact flow; when no saturating quantizer follows (a last ReLU, with an unsigned wrapping result type),
  negative inputs then wrap around instead of giving 0 in the C simulation (a regression test covers it);
  `Activation("relu")` converts correctly.
- **hls4ml >= 1.3.0 is required for the export**, declared as the extra `keras-hexagdly[hls]` and checked by
  `register_hex_gather_layers`. Checked: hls4ml 1.0 and 1.1 have no Keras 3 converter; 1.2 does not register the
  Keras handlers of the hex layers by itself (1.3 does, through a metaclass), and with them registered by hand the
  C simulations of the line-buffer kernels came out all zeros. Only 1.3.0 is tested; there is no upper bound.
- **A notebook** (`notebooks/keras_hexagdly_qat_hgq2_to_hls4ml.ipynb`) taking a hexagonal CNN from
  data to a Vitis HLS synthesis: float reference, QAT with EBOPs, `freeze`, hls4ml, C simulation, and a
  resource comparison with a uniform-precision export.
- `notebooks/keras_hexagdly_hls4ml_export.ipynb` re-run end to end on this version (fused kernels at II=1,
  C simulation, and the RTL co-simulation, which passes).
- **EBOPs for the hexagonal convolutions** (`QConv2d`, `QConv3d`, `QConv2d_CustomKernel`,
  `QConv3d_CustomKernel` with an `iq_conf`): the same cost measure and the same training term
  (`beta * EBOPs`) as HGQ2's layers, with `enable_ebops`, `beta0` (default `1e-5`) and
  `ebops_factor`. HGQ2's `FreeEBOPs`, `BetaPID`, `BetaScheduler`, `PBar`, `ParetoFront`,
  `EarlyStoppingWithEbopsThres` and `trace_minmax` work on them. The EBOPs were checked against a
  by-hand count (taps of the hexagon x channels x bits), and shrink as the bit widths do.
- **`keras_hexagdly.hls4ml_ext.apply_quantized_types(config, patched_model, types)`** puts the
  `ap_fixed` types `freeze` computed into the hls4ml config (weights, bias, accumulator, the input
  type on the producing layer, the output type after the fused activation): the recipe was six
  lines to copy by hand per layer. The C-simulation tests use it.
- **`keras_hexagdly.freeze(model, activation_quantizers="keep")`** returns `(frozen_model, types)`: the
  quantized layers become plain layers holding the quantized values, with the same outputs (checked),
  and `types` gives, per layer, the `ap_fixed` type of the `weight`, `bias`, `input`, `result` (the
  input / output quantizer, with its rounding and overflow modes) and `accum`. `"keep"` leaves the
  activation quantizers in the frozen model as standalone HGQ2 layers (to check the quantization in
  Keras, and for hls4ml's bit-exact flow, see below); `"types"` drops them, their precision being in
  `types`, to be set by hand in the hls4ml config (per-layer activation quantizers only for `io_stream`:
  hls4ml handles per-element ones in `io_parallel` only). Per-layer methods:
  `weight_type()`, `bias_type()`, `input_type()`, `result_type()`, `accum_type()`.
- **`ExactAccumulator` and `AccumPrecision`**, two hls4ml config keys for the accumulator of an exported
  hex convolution, both off by default (existing exports do not change). The default accumulator keeps
  the fractional bits of the finer of the weights and the input, which drops the low bits of each
  product (visible with narrow quantized inputs: the C simulation differed from Keras by a few steps).
  `config["Model"]["ExactAccumulator"] = True` (or on one layer, which wins) keeps all the bits of the
  product, `f_weight + f_input`; `config["LayerName"][name]["AccumPrecision"] = "ap_fixed<W,I>"` sets the
  type itself (`types[...]["accum"]` is the exact one) and wins over `ExactAccumulator`. With either, the
  C simulation of a layer with a quantized input, weights, bias and output equals the Keras model bit
  for bit (`tests/test_quantized_csim.py`). The wider accumulator costs logic: measured with Vitis HLS
  on one hex convolution, from +18 % flip-flops and 130 DSPs with narrow types to 3x the LUTs with
  16-bit types (README, "Accumulator width", measured again with the final accumulator), which is
  why it is opt-in. The exact accumulator also has one extra integer bit (the carry of the bias addition)
  and keeps the fractional bits of a finer bias. `AccumPrecision` is read per layer only.

### Fixed

- **Hex convolution kernels in `io_stream`: half the latency.** The fused line-buffer kernel and the window
  process of its `Dataflow` version carried two counters (`cur_emit_oh`, `cur_emit_col`) from one iteration
  to the next, tested and incremented in the same iteration. Vitis could not pipeline that loop below II=2
  (`Unable to enforce a carried dependence constraint`), so every layer took twice the cycles: 376 instead of
  195 for a 12x12 image with 8 to 16 channels. The row and column of the pixel to emit are functions of the
  loop indices, so the kernels now compute them and carry nothing: II=1, and the C-simulation outputs are
  unchanged (compared with the previous kernel on 62 convolution configurations: strides 1 to 3, kernel sizes
  0 to 2, even and odd grids, with and without ring sharing; new tests in `tests/test_hex_linebuffer_csim.py`).
  Synthesis, `xc7vx485tffg1927-2`, 5 ns, one layer: latency 376 -> 195, LUT -1.8 %, FF -0.4 %, DSP and BRAM
  unchanged. The `Dataflow` version had the same defect in its window process and a second cause in its MAC process
  (`MacPixels`): the cap `ALLOCATION operation instances=add limit=MAC_BLOCK` is below what the MAC needs at
  `reuse_factor=1` (constant multiplications become shift-and-add networks, plus the accumulation tree and the
  loop counter), so Vitis shared adders and the loop fell to II=2. The cap is now 4x `MAC_BLOCK` when the reuse
  factor is 1 and unchanged otherwise. Same layer: 375 -> 194 cycles, LUT -3.8 %, FF -6.4 % (narrow types); with
  the 9/10-bit and 16-bit types of the README table it also reaches 194 cycles at II=1. Removing the cap
  altogether is not an option: Vitis then used about 49 GB of memory while scheduling. The max pooling kernel was already at II=1 and is unchanged; the 3D
  kernels (a different structure, export not validated) were not looked at.
- `freeze` follows the graph of the model instead of walking `model.layers`: raw operations
  (`+`, `keras.ops.*`, residual connections), a layer called more than once, a layer reused by another
  model and nested Functional models used to give a bare `KeyError` or a "not wired" error.
- The quantizers of `iq_conf` / `oq_conf` return float32; the layers now cast them back to their
  compute dtype (`dtype="float16"` crashed on torch). A quantizer with a `scaler` or `affine` has no
  `ap_fixed` type (the type of its bit widths would be too narrow): refused instead of silently wrong.
- `ExactAccumulator` ignored that a layer without bias still has a zero bias array typed with the
  model default precision, and made the accumulator wider than `accum_type()`; a text value such as
  `"False"` for it switched it on.
- The accessors of an unbuilt quantized layer (`weight_type()`, `quantized_bias()`, `accum_type()`, and
  `bias_type()` / `input_type()` / `result_type()` when their quantizer is set) say that the layer must be built;
  a pool with a quantizer on a dynamic spatial shape gives the clear "statically-known" error.
- `tests/test_quantized.py` is skipped instead of failing at collection without hgq2.
- `apply_quantized_types`: with a fused activation, the MAC result was cast to a fixed `ap_fixed<32,12>`
  before the ReLU, so a sum above 2048 wrapped around in the C simulation whatever the accumulator
  type; it now gets the accumulator type. Two layers that read the same tensor with different input
  types are refused instead of letting the last one win.
- Clearer errors: `freeze` of something that is not a model, a quantizer config of the wrong type or a
  negative `beta0` (refused when the layer is built, not at the first `fit`), half precision on the
  TensorFlow backend (HGQ2 cannot do it), and `patch_model_for_hls` on a model whose input is not quantized while it holds HGQ2 layers or quantizers
  (hls4ml's bit-exact flow needs a quantized input), or with an hls4ml that lacks that flow.
- `patch_model_for_hls` gives a clear error for a model whose output comes from a layer called more
  than once (a bare `KeyError` before).
- The sdist lists its files explicitly (`pyproject.toml`): a local `python -m build` used to pick up every
  untracked directory.
- `import keras_hexagdly` no longer fails when hgq2 is installed but unusable (wrong version, broken
  install): the plain layers stay available, the quantized ones are missing and a warning says why.
- `freeze` keeps the `sparse` / `ragged` flags of the inputs, reuses a weightless layer it cannot copy
  (a `Lambda`), and keeps the compute dtype of the standalone quantizers (mixed precision).
- The EBOPs sum is done in float32 (it overflowed in half precision) and saturates in the uint32
  variable instead of wrapping around.
- A QAT model saved before the EBOPs existed (an `iq_conf` layer without the `enable_ebops` key)
  loads with them off, so its weights file still fits.
- Weight and bias `ap_fixed` types of quantized layers are derived from the quantized values (they used to
  be computed from the quantizer parameters, which was wrong for negative integer bits); `freeze` handles
  nested models and `get_config` / `from_config` round-trip the quantizers.
- `patch_model_for_hls` failed with a bare `KeyError` on a model that reuses a layer object that already
  belongs to another model (or is called twice): `layer.input` is the input of the layer's first graph
  node. It now raises a `ValueError` that names the layer and says to rebuild it (`freeze` gives the
  same error). It also refuses a model that still contains quantized layers (they would have been
  exported as unquantized float layers) and asks for `freeze(model)` first.
- The exported hex layers (`HexConvLineBuffer`, `HexPoolLineBuffer` and their 3D versions) now opt out
  of the XLA compilation Keras turns on by itself on a GPU: their int32 gather table lives on the CPU
  and XLA could not read it (`predict()` of a patched model failed on a TensorFlow GPU with "resource
  located in device CPU from device GPU").
- The even-size warning of 0.6.1 used a fixed stack level, so it would have pointed inside the
  library for a layer that subclasses another one (such as `QConv2d`); it now always points at the
  line that built the layer.

## [0.6.1] - 2026-10-22

### Added

- **A `UserWarning` for even sizes with `kernel_size_unit="pixels"`.** A hexagonal window is centred on
  a cell, so it only exists for odd widths. An even width has always been rounded up to the next odd
  one (2 -> 3, the 7-cell window; 4 -> 5, the 19-cell one); the layers now say so, with the warning
  pointing at the line that built the layer. `"rings"` mode and the depth element of the
  `Conv3d`/`MaxPool3d` size are not concerned. Behaviour is unchanged: only the warning is new.

### Fixed

- The 0.6.0 README and docstrings said an even width "floors to the odd width below it" (so that
  `kernel_size=2` would resolve like `1`). That was wrong: it is rounded **up** (2 -> 3). Wording
  corrected; the computation never changed.

## [0.6.0] - 2026-10-14

### Added

- **Dataflow kernel for the hls4ml hex convolution (io_stream, 2D), selectable per layer.** Set
  `config["LayerName"]["<layer>_linebuffer"]["Dataflow"] = True` (default `False`; a layer that has a
  `LayerName` entry does not fall back to the `Model` section, like every other per-layer key).
  The fused kernel stays the default and is unchanged. With the flag, the line buffer and the MAC run
  as two concurrent dataflow processes joined by a FIFO of tap packs, so a pixel no longer waits for
  the previous one's multiply. Measured on a 19x14 board grid (a whole CNN with two hex layers, Vitis
  HLS 2024.2, VC707): latency 6083 -> 824 cycles (x7.4) for about +60 % LUT and +5 % FF (the first hex
  layer's reuse factor was 1 for the fused kernel and 2 for the dataflow one). Layers that are not
  hex convolutions are untouched.
  The flag is ignored with a warning under io_parallel; the 3D kernel has no dataflow variant.
  Checked in C simulation (`tests/test_hls_dataflow_csim.py` compiles both kernels on several kernel
  sizes, strides, reuse factors and `ring` sharing and requires bit-identical results, and agreement
  with Keras) and in RTL co-simulation (`tests/smoke_cosim_dataflow.py`, needs Vitis HLS): a small
  network with the structure of a real one, fused and dataflow kernel, with and without
  `share_neighbors="ring"`, reuse factors (1, 1) and (1, 2): in all six cases the C and the RTL results
  are identical. The traversal is duplicated between the two kernels (the C test guards against drift).
  Co-simulation caveat: once a type overflows, C simulation and RTL do not wrap alike, so a mismatch
  can come from the types of the network under test rather than from a kernel (the smoke test uses
  wide types for that reason).

- **`kernel_size_unit` on `Conv2d`, `Conv3d`, `MaxPool2d`, `AvgPool2d` and
  `MaxPool3d`.** `"rings"` (default, unchanged) reads `kernel_size`/
  `pool_size` as the hexagonal ring count, as always. `"pixels"` reads it as
  a square-kernel width instead -- radius = `size // 2` -- so the same value
  means a comparable extent on a square and a hex grid: `1` is a plain
  per-cell projection (0 rings), `3` the 7-cell hexagon (1 ring), `5` the
  19-cell one (2 rings), as in indexedconv's `MaskedConv2d`. Even widths have
  no exact hex equivalent and floor instead of raising, so a default
  `pool_size=2` resolves rather than being rejected. On `Conv3d`/`MaxPool3d`,
  only the hex element of the `(depth, hex)` kernel_size follows the unit --
  depth is never a hex radius. Serialized in `get_config()`, so a saved
  model keeps the unit it was built with. Prompted by
  https://github.com/ctlearn-project/ctlearn/issues/302: a config written
  for the square backend now means the same thing for the hex one.
- **`kernel_size`/`pool_size=0`** (0 hexagonal rings: a plain per-cell
  channel projection, no neighbours at all) is now accepted everywhere it
  used to be rejected -- it was a missing capability, not a deliberate
  restriction, and `kernel_size_unit="pixels"` needs it to express a true
  1x1. On `Conv3d`/`MaxPool3d`, only the hex element can be 0; a bare int
  kernel_size still has to be positive, since it also sets the depth axis,
  and a 0-depth kernel is degenerate.

### Fixed

- `HexBase.operation_with_single_hexbase_stride` (the ``stride == 1`` path
  shared by every layer) crashed with `hexbase_size=0`: with no kernel1+,
  the odd/even-column lists it concatenates at the end stayed empty.
  `kernel0` alone, at full resolution, already *is* the complete pointwise
  output, so it is returned directly instead.
- `keras_hexagdly.indexed.build_neighbor_table`/`get_cell_weights` (used by
  the hls4ml export) read `layer.kernel_size` -- the raw, unit-dependent
  constructor argument -- instead of `layer.hexbase_size`, the resolved
  ring count the neighbour-table geometry is actually defined in terms of.
  The two were always equal before `kernel_size_unit` existed, so this
  never surfaced; with `kernel_size_unit="pixels"` they genuinely differ,
  and `patch_model_for_hls` crashed with an `IndexError` on any such layer.
  The 3D versions of these functions already used `hexbase_size` correctly.
- **The accumulator of an hls4ml-exported hex convolution was sized from the weight type alone**,
  which was wrong in three ways, each reproduced by a C-simulation test
  (`tests/test_hls_accum_csim.py`) that fails on the previous code:
  - it overflowed whenever the input was larger than about 1: a layer summing 112 products of
    activations up to ~50 with weights below 4 got a +-512 accumulator for sums reaching ~800 and
    wrapped around, silently. It now adds the integer bits of the layer's input type;
  - it kept only the fractional bits of the weights, so an input finer than the weights was
    truncated when cast into the accumulator (error ~10 % in the test). It now keeps the finer of
    the two;
  - it wrapped a bias wider than the sum of the products (a bias of 1000 into a +-128 accumulator
    came out off by 1024). It now has at least the integer bits of the bias.

  The accumulators of exported models are wider as a result, so resource counts measured with
  earlier versions are slightly off.

## [0.5.0] - 2026-10-04

### Added

- **Keras-compatible arguments, so the hex layers can be swapped for the
  Keras ones from a single set of keyword arguments.** Asked for in the
  CTLearn integration, which picks `hgly.Conv2d` or `keras.layers.Conv2D`
  with a config option and should not need a wrapper around either.
  - `Conv2d(..., activation=None)` and `Conv3d(..., activation=None)`: same
    values as `keras.layers.Conv2D`, applied to the layer output and kept in
    `get_config()`, so a saved model still applies it after reloading.
    `patch_model_for_hls` keeps it too: its replacements are rebuilt from the
    layer's weights and geometry only, so the activation is re-applied as a
    separate `keras.layers.Activation` after each replaced layer.
  - `Conv2d(..., padding="same")`: the hex grid is always padded so the
    output size depends only on the stride, so only `"same"` is accepted;
    anything else raises a `ValueError` instead of being ignored. Not added
    to `Conv3d`, where `depth_padding` already covers the depth axis.
  - `MaxPool2d`/`MaxPool3d` take the window size as `pool_size`, the Keras
    name. `kernel_size` is still accepted as an alias, so existing code and
    saved 0.4.x models keep working; passing both raises. `get_config()` now
    writes `pool_size`. `MaxPool2d` also accepts `padding="same"`, and its
    arguments follow the `keras.layers.MaxPool2D` order -- `pool_size,
    strides, padding, name` -- so positional calls mean the same thing too.
- **`AvgPool2d`**: hexagonal average pooling over the same window as
  `MaxPool2d`. Each output cell is the mean of the window's cells that lie
  inside the grid: positions past the grid border count in neither the sum
  nor the divisor, like `keras.layers.AveragePooling2D(padding="same")`,
  while empty cells inside the grid count as zeros, as they do for `Conv2d`
  and `MaxPool2d`. It has no hls4ml export path yet: `patch_model_for_hls`
  raises `NotImplementedError` on it rather than letting it reach hls4ml as
  an unknown layer.

### Changed

- **`MaxPool2d`'s third positional argument is now `padding`, not `name`.**
  A 0.4.x call passing the name positionally, such as
  `MaxPool2d(2, 2, "pool1")`, now raises a `ValueError` instead of naming the
  layer. Pass `name=` as a keyword.
- **`Conv2d`/`Conv3d` require `use_bias` and `debug` to be booleans** (a
  numpy bool is fine) and raise a `TypeError` otherwise. `use_bias` sits in
  the fourth positional slot, where `keras.layers.Conv2D` takes `padding`, so
  a Keras-style `Conv2d(8, 1, 1, "same")` used to mean `use_bias="same"`,
  i.e. `True`, without any error.

### Fixed

- **A convolution with a stride at least as large as the grid width
  crashed** on the TensorFlow and torch backends -- a stride-2 `Conv2d` on a
  grid only 1 or 2 columns wide, or a stride-1 `Conv2d`/`Conv3d` on a single
  column (where a ResNet block lands right after a stride-2 layer). Upstream
  HexagDLy fails the same way. The even columns of such an output are empty,
  but were computed anyway from a zero-width input; they are now skipped.
  This shows up at the end of deep networks: five or six stride-2
  downsamplings take a 39-55-column camera image to 2 or 1 columns. Every
  other configuration computes exactly as before. The jax backend already
  handled these cases, and its results are unchanged.

## [0.4.2] - 2026-08-28

### Fixed

- `__version__` was a hardcoded string that drifted stale across three
  releases in a row -- it still read `"0.3.0"` as of 0.4.1, despite
  `pyproject.toml`'s actual package version having moved to 0.3.1, 0.4.0,
  and 0.4.1 in the meantime. Anyone checking `keras_hexagdly.__version__`
  at runtime to detect what's installed got a wrong answer.
  Now derived from the installed package's own metadata
  (`importlib.metadata.version("keras-hexagdly")`, set from
  `pyproject.toml`'s `version` at build time) instead of a second,
  manually-synced constant, so it can't drift out of sync again. Falls
  back to `"unknown"` if run from source without an installed/editable
  package.

## [0.4.1] - 2026-08-27

### Fixed

- **`Conv2d`/`Conv3d`'s default (He-normal) weight initialization was
  scaled too large, and grew larger with `kernel_size`.** Both layers split
  each output channel's receptive field across several separately-shaped
  sub-kernel tensors (`kernel0`, `kernel1`, ... -- or, with
  `share_neighbors`, one shared value per hex ring) whose contributions are
  *summed* to produce the output. Calling `keras.initializers.HeNormal()`
  on each piece independently -- as this port originally did, faithfully
  mirroring upstream `hexagdly`'s (and `pytorch_hexagdly`'s) per-sub-kernel
  `kaiming_normal_()` calls -- calibrates each piece's variance as if it
  alone were the entire receptive field. The summed output ends up with
  variance inflated by roughly the number of sub-kernels: measured output
  std for a single untrained layer, on unit-variance input, was ~1.4x too
  high at `kernel_size=1` (2 sub-kernels) growing to ~1.9x too high at
  `kernel_size=3` (4 sub-kernels), instead of holding constant near the
  He-init target of `sqrt(2)` regardless of kernel size.
  - This is a real, upstream-inherited bug, not a deliberate design choice
    -- `Conv2d_CustomKernel`/`Conv3d_CustomKernel` are unaffected (they
    require explicit caller-supplied `sub_kernels`, no default init path).
  - Fixed by initializing from one pool sized by the *combined* fan-in
    across all of a layer's sub-kernels/rings, then slicing pieces off it --
    see `_combined_fan_in_pool` in `layers.py`. `debug=True` (all-ones
    init) is unaffected.
  - Practical symptom this explains: a hex conv stack noticeably
    out-scaling an equivalent square `keras.layers.Conv2D` stack's
    activations before training even starts, forcing a lower learning rate
    or `batchnorm=True` just to make the two comparable -- that workaround
    is no longer necessary for this specific cause (though it may still be
    good practice for other reasons).
  - No change to `get_config()`/`from_config()`, weight *shapes*, or the
    convolution math itself -- only the *values* a freshly-built layer's
    weights start at. A `.keras` file saved from a previously-built (and
    at least partially trained) model is unaffected on load, since loading
    restores the saved weight values rather than re-running the
    initializer.

## [0.4.0] - 2026-08-27

### Removed (BREAKING)

- **The original PyTorch-hexagdly-style constructor names are gone**:
  `in_channels`, `out_channels`, `stride`, `bias` no longer exist on
  `Conv2d`/`Conv3d`/`Conv2d_CustomKernel`/`Conv3d_CustomKernel`/
  `MaxPool2d`/`MaxPool3d` (`Conv2d_CustomKernel`/`Conv3d_CustomKernel`'s
  `bias` -- the bias *array*, not a boolean flag -- is unaffected, it was
  never part of this rename). Passing any of the removed names now raises
  (`TypeError`/`ValueError`), not a warning.
  - `filters`/`kernel_size` are now genuinely positional, matching
    `keras.layers.Conv2D(filters, kernel_size, strides=..., ...)` exactly.
    This is why the old names had to go rather than staying deprecated
    forever: keeping the old 2-positional `Conv2d(in_channels,
    out_channels, ...)` form alive indefinitely would have meant
    `Conv2d(32, 3, strides=2)` -- written the way anyone coming from
    `keras.layers.Conv2D` naturally would -- kept silently building the
    wrong layer (`in_channels=32, out_channels=3, kernel_size=1`) instead of
    raising. That silent-misinterpretation risk, not just naming taste, is
    the actual reason for this breaking change.
  - `in_channels` is not reintroduced under any name -- it's always inferred
    from the input in `build()`, like every standard Keras layer.
  - `get_config()`/`from_config()` only ever produced/consumed the new names
    as of 0.3.1, so this release does not change what a `.keras` file
    written by 0.3.1 (or this version) contains -- only files written by
    0.3.0 or earlier (using the pre-rename names) can no longer be loaded.
    If you need those, pin `keras-hexagdly==0.3.1` (the one release where
    both name sets were accepted) to re-save them under the new names, then
    upgrade.

## [0.3.1] - 2026-08-27

**Transitional release.** This version accepts both the old and new
constructor names side by side (see below). The *next* release removes the
old PyTorch-hexagdly-style names entirely -- pin `keras-hexagdly==0.3.1` if
you specifically need the old names to keep working a little longer;
otherwise migrate to the new names now, since this is the only version
where both work.

### Added

- **Keras-native constructor keyword names**: `Conv2d`/`Conv3d` now accept
  `filters` (alias for `out_channels`), and `Conv2d`/`Conv3d`/
  `Conv2d_CustomKernel`/`Conv3d_CustomKernel`/`MaxPool2d`/`MaxPool3d` all
  accept `strides` (alias for `stride`); `Conv2d`/`Conv3d` additionally
  accept `use_bias` (alias for `bias`). These are the preferred names going
  forward and match standard Keras layer conventions
  (`keras.layers.Conv2D(filters=..., strides=..., use_bias=...)`).
  `get_config()` now emits these new names.

### Changed

- **Deprecated the original PyTorch-hexagdly-style constructor names**:
  `stride` and `bias` (on `Conv2d`/`Conv3d`; `Conv2d_CustomKernel`/
  `Conv3d_CustomKernel`'s `bias` is the bias *array*, not a boolean flag, and
  is unaffected/not deprecated) now emit a `DeprecationWarning` pointing at
  `strides`/`use_bias`. `in_channels`/`out_channels` are unaffected (not
  deprecated in this release) -- `filters` is simply the additional
  Keras-idiomatic alias for `out_channels`. All old names keep working
  exactly as before in this release, including loading models saved by
  previous versions (whose serialized config uses the old key names) --
  see `tests/test_param_rename_backcompat.py`.

### Changed

- **Restructured the gather-strategy HLS kernels around hls4ml's proven conv2d
  pattern** (`nnet_hex_ring_mac.h`, `nnet_hex_gather.h` and `_3d` variants,
  `hls4ml_handler.py`). The previous ring MAC flattened all
  `n_out*k*Cin*Cout` products into one fully-unrolled loop with runtime
  div/mod index decoding and runtime-addressed accumulators, which made Vitis
  HLS build thousands of multipliers, emit II violations (HLS 200-882/885),
  spend hours in scheduling/binding, and run out of memory (>48 GB) at
  20×20-grid scale. The new structure mirrors `conv_2d_latency_cl`:
  a sequential partition loop pipelined at `II=ReuseFactor` over
  `n_partitions = n_out / n_pixels` iterations, an unrolled pixel loop over
  `n_pixels` concurrent output pixels, and per-pixel Product/Accum stages in
  which every mult/accum/weight index is a compile-time constant (no
  crossbars, no div/mod). `multiplier_limit` is now per-partition
  (`ceil(n_pixels*k*Cin*Cout / ReuseFactor)`), satisfiable by construction.

- **Neighbor index tables and ring maps are now compile-time constants.**
  `HexGather`'s `(N_out, K)` table and `HexRingMAC`'s ring map are emitted as
  `static const` arrays in the generated `parameters.h` (the same trick
  hls4ml's im2col codegen uses) instead of runtime weight ROMs. The
  fully-unrolled gather constant-folds to pure wiring with literal zeros at
  border slots; ring weight sharing costs no runtime indirection
  (`row_map[k]`, identity when `share_neighbors=False`). The `idx*`/`ridx*`
  weight arrays and their derived integer types no longer exist.

### Added

- **`ParallelizationFactor` support for `HexRingMAC` / `HexRingMAC3D`**
  (per-layer or model-wide hls4ml config), mirroring hls4ml's conv handling:
  `n_pixels = PF` output pixels are computed concurrently and hardware cost
  scales with `PF * k * Cin * Cout / ReuseFactor` instead of the whole image.
  Defaults to 1 (one output pixel per II); invalid values clamp to the
  closest divisor of `n_out` with a warning. Latency scales as
  `n_partitions * ReuseFactor` cycles per layer.
  A rolled partition loop requires dataflow pipeline style (set
  `cfg['Model']['PipelineStyle'] = 'dataflow'`, or `Strategy = 'Resource'`):
  under 'pipeline' style the top-level `#pragma HLS PIPELINE` force-unrolls
  every layer loop (same physics as hls4ml conv, cf. their
  `SetPipelineStyle` pass), so the emitted config falls back to
  `n_pixels = n_out` with a whole-layer multiplier cap and a warning.

## [0.2.0] - 2026-06-30

### Added

- `src/keras_hexagdly/indexed.py`: neighbor-table builder and indexed forward
  references for all four layer types (`Conv2d`, `MaxPool2d`, `Conv3d`,
  `MaxPool3d`). Derives the `(N_out, K)` neighbor table empirically by firing
  single-pixel impulses through the existing `call()` path — no coordinate
  assumptions, exact by construction. Provides NumPy reference implementations
  (`indexed_conv2d_forward`, `indexed_maxpool2d_forward`, etc.) that match
  `call()` bit-for-bit and double as C-simulation oracles.

- `src/keras_hexagdly/hls4ml_ext.py`: hls4ml export bridge.
  `patch_model_for_hls(model, strategy="slotwise")` returns a new Keras model
  where every hex layer is replaced by stock hls4ml-native ops — no custom HLS
  layer needed. Two strategies selectable via the `strategy` parameter:
  - `"slotwise"` (default): K separate gather+MAC `EinsumDense` pairs per
    layer, summed via `Add`. Largest static array is `(N_out, N_in)` per slot
    — safe for Vitis HLS RTL synthesis at typical camera sizes.
  - `"folded"`: one `EinsumDense` with the full `(N_in,Cin,N_out,Cout)` kernel.
    Simpler graph; fine for C-simulation but causes Vitis HLS clang to segfault
    during synthesis on grids larger than ~9×9.
  - `"gather"`: `NotImplementedError` placeholder for a future sparse-gather
    custom layer that scales to full camera size (e.g. DigiCam N=1296).
  `MaxPool2d` → `EinsumDense` gather + `MaxPooling1D(K,K)` (strategy-agnostic).
  `MaxPool3d` → `NotImplementedError` (pending).
  The original model and its weights are never modified; `hls4ml` is an
  optional dependency, never imported at training time.

- `tests/test_indexed_equivalence.py`: 75 equivalence tests covering all layer
  types, kernel sizes (1–3), strides (1–2), both weight modes
  (`share_neighbors=True/False`), border pixels, and all-negative inputs.

- `tests/test_hls4ml_ext.py`: 29 patch-model tests in two tiers — Keras
  float32 equivalence (always runs) and hls4ml C-simulation via g++
  (auto-skipped when hls4ml is not installed).

- `notebooks/keras_hexagdly_hls4ml_export.ipynb`: end-to-end notebook showing
  training a hex CNN, patching it for hls4ml, verifying bit-identical outputs,
  running C-simulation, and (optionally) full RTL synthesis with Vitis HLS
  2024.2 — including the PATH/shim setup for the `vivado_hls` → `vitis_hls`
  rename.

### Changed

- Build backend switched from `setuptools` to `hatchling` to fix a long-standing
  `UNKNOWN.egg-info` issue: `pip install -e .` now creates a proper editable
  install without polluting the repo root with a stale egg-info directory.

## [0.1.1] - 2026-06-30

The package code is unchanged from 0.1.0 (same API, same outputs); this release
is packaging metadata plus repo-only additions.

### Changed

- Packaging metadata: expanded PyPI keywords, classifiers, and project URLs, and
  a more search-friendly one-line description, to improve discoverability.

### Added

- `notebooks/cta_lst_hexconv_gamma_vs_noise.ipynb`: a self-contained example
  training a hexagonal 3-D CNN on real CTA LST camera geometry and simulated
  events (streamed from `ctapipe`) to separate gamma showers from night-sky
  background, including a visualisation of the learned kernels.

### Fixed

- Example notebooks now read layer outputs via `keras.ops.convert_to_numpy`
  instead of `np.asarray`, so they no longer crash on a GPU-resident tensor
  under the torch/jax backends.

## [0.1.0] - 2026-06-29

Initial release: a Keras 3 port of [HexagDLy](https://github.com/ai4iacts/hexagdly).

### Added

- `Conv2d`, `Conv3d`, `Conv2d_CustomKernel`, `Conv3d_CustomKernel`,
  `MaxPool2d`, `MaxPool3d` -- channels-last Keras 3 layers, numerically
  faithful to upstream PyTorch HexagDLy on every backend (tensorflow, torch,
  jax).
- `share_neighbors`: ties hexagonal kernel weights by ring instead of giving
  every cell an independent weight. New, no upstream equivalent.
- `depth_padding="same"` on `Conv3d`: zero-pads the depth/time axis so the
  temporal kernel is centred and output depth equals input depth, instead of
  upstream's `"valid"`-only behaviour. New, no upstream equivalent.
- `in_channels` inference from the input shape (the `Conv2d(out_channels,
  ...)` call form), in addition to the original explicit
  `Conv2d(in_channels, out_channels, ...)` form.
- Input validation: `kernel_size`/`stride` must be positive integers (or a
  2-tuple for the 3D layers), checked at construction time with a clear
  `ValueError` instead of failing deep inside the first call.
- Five example notebooks ported from upstream's own notebooks, plus the two
  new features.
- Test suite: hand-verified standalone tests, cross-checks against the
  upstream PyPI `hexagdly` package (including forward-compatibility tests
  that activate automatically if upstream ever adds `share_neighbors`/
  `depth_padding`), edge cases, `get_config`/`from_config` and
  `model.save`/`load_model` round-trips, and mixed-precision checks --
  verified on all three Keras 3 backends.
- Speed benchmarks against upstream (`benchmarks/`), eager and compiled.
- GitHub Actions CI (test matrix across Python/backend versions + ruff).
