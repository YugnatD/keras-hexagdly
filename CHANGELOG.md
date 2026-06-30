# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
