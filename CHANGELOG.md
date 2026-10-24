# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.7.0] - 2026-10-23

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
- **A notebook** (`notebooks/keras_hexagdly_qat_hgq2_to_hls4ml.ipynb`) taking a hexagonal CNN from
  data to a Vitis HLS synthesis: float reference, QAT with EBOPs, `freeze`, hls4ml, C simulation, and a
  resource comparison with a uniform-precision export.
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
  Keras); `"types"` drops them, their precision being in `types`, for hls4ml `io_stream` (per-layer
  activation quantizers only: hls4ml handles per-element ones in `io_parallel` only). Per-layer methods:
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
  TensorFlow backend (HGQ2 cannot do it), and `patch_model_for_hls` on a model frozen with the default
  `activation_quantizers="keep"` (the standalone quantizers cannot be converted by hls4ml: freeze with
  `"types"`).
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
