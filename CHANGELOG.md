# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

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
