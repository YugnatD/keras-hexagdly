# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
