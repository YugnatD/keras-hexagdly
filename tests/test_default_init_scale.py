"""Tests for the combined-fan-in default initializer fix.

Conv2d/Conv3d split each output channel's receptive field across several
separately-shaped sub-kernel tensors (kernel0, kernel1, ... -- or, with
share_neighbors, one shared value per ring) whose contributions are SUMMED
to produce the output. Calling He-normal on each piece independently --
as upstream hexagdly, pytorch_hexagdly, and this port originally did --
calibrates each piece's variance as if it alone were the entire receptive
field, so the summed output ends up with variance inflated by roughly the
number of pieces (verified: output std ended up growing with kernel_size/
share_neighbors group count instead of staying constant). See
_combined_fan_in_pool in layers.py for the fix and full explanation.

These tests check the OBSERVABLE property that actually matters: for a
freshly-built (untrained) layer, the ratio of output std to input std
should stay close to sqrt(2) (He-init's variance-doubling target for a
ReLU-facing layer) and, critically, should NOT grow with kernel_size or
with the number of rings share_neighbors introduces -- that growth was
the actual bug symptom.
"""

import keras
import numpy as np
import pytest

import keras_hexagdly as hgly

# He-normal targets Var(output) = 2 * Var(input) for a correctly-scaled
# layer; std ratio = sqrt(2). Real single-layer measurements land close to
# this (see the investigation that found this bug: ratios of 1.35-1.41
# across kernel_size 1-3 and both share_neighbors settings, vs. the buggy
# code's 1.95-2.75 that grew with kernel_size). Allow a generous band for
# sampling noise while still being tight enough to catch the ~1.4x-per-
# sub-kernel bug (which pushes the ratio well outside this range already
# at kernel_size=1).
EXPECTED_RATIO = np.sqrt(2.0)
RATIO_TOLERANCE = 0.25


def _output_input_std_ratio(layer, x):
    y = keras.ops.convert_to_numpy(layer(keras.ops.convert_to_tensor(x)))
    return float(y.std() / x.std())


@pytest.fixture
def x():
    # Zero-mean, moderate sample size for a stable std estimate; channel
    # count and spatial size chosen to keep the test fast.
    rng = np.random.default_rng(0)
    return rng.standard_normal((256, 17, 17, 4)).astype("float32")


@pytest.mark.parametrize("kernel_size", [1, 2, 3])
def test_conv2d_output_scale_independent_of_kernel_size(x, kernel_size):
    """The core regression check: the std ratio must stay near sqrt(2)
    regardless of kernel_size (i.e. regardless of how many sub-kernels the
    layer is internally split across)."""
    layer = hgly.Conv2d(16, kernel_size=kernel_size, use_bias=False)
    ratio = _output_input_std_ratio(layer, x)
    assert abs(ratio - EXPECTED_RATIO) < RATIO_TOLERANCE, (
        f"kernel_size={kernel_size}: output/input std ratio {ratio:.3f} is far "
        f"from the He-init target {EXPECTED_RATIO:.3f} -- looks like the "
        "combined-fan-in fix regressed."
    )


@pytest.mark.parametrize("kernel_size", [1, 2])
def test_conv2d_share_neighbors_output_scale_independent_of_kernel_size(x, kernel_size):
    """Same property must hold for the share_neighbors=ring weight path,
    which uses a separate init call (ring_weights) from the non-shared path."""
    layer = hgly.Conv2d(16, kernel_size=kernel_size, use_bias=False, share_neighbors=True)
    ratio = _output_input_std_ratio(layer, x)
    assert abs(ratio - EXPECTED_RATIO) < RATIO_TOLERANCE, (
        f"kernel_size={kernel_size} share_neighbors=ring: ratio {ratio:.3f} is "
        f"far from {EXPECTED_RATIO:.3f}."
    )


def test_conv2d_ratio_does_not_grow_with_kernel_size(x):
    """Direct check against the actual bug symptom: before the fix, the
    ratio grew roughly as sqrt(number of sub-kernels) = sqrt(kernel_size+1).
    Build kernel_size=1 (2 sub-kernels) and kernel_size=3 (4 sub-kernels)
    and confirm they land close together, not ~sqrt(2) apart."""
    ratios = []
    for kernel_size in (1, 3):
        layer = hgly.Conv2d(16, kernel_size=kernel_size, use_bias=False)
        ratios.append(_output_input_std_ratio(layer, x))
    assert abs(ratios[0] - ratios[1]) < RATIO_TOLERANCE, (
        f"kernel_size=1 ratio={ratios[0]:.3f} vs kernel_size=3 ratio={ratios[1]:.3f} "
        "-- growing with kernel_size is exactly the pre-fix bug symptom."
    )


def test_conv3d_output_scale_matches_conv2d_target():
    """Conv3d has its own build() with the same combined-fan-in fix
    (including the extra depth_size factor in total_taps) -- check it
    independently rather than assuming Conv2d's fix covers it."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((32, 3, 9, 9, 4)).astype("float32")
    layer = hgly.Conv3d(16, kernel_size=1, use_bias=False)
    ratio = _output_input_std_ratio(layer, x)
    assert abs(ratio - EXPECTED_RATIO) < RATIO_TOLERANCE, (
        f"Conv3d kernel_size=1: ratio {ratio:.3f} far from {EXPECTED_RATIO:.3f}."
    )


def test_debug_mode_still_all_ones():
    """debug=True must still produce an all-ones kernel (unaffected by the
    combined-fan-in pool, which for Ones() is trivially still all ones
    wherever it's sliced from)."""
    layer = hgly.Conv2d(4, kernel_size=2, use_bias=False, debug=True)
    _ = layer(keras.ops.convert_to_tensor(np.zeros((1, 9, 9, 2), dtype="float32")))
    for w in layer.get_weights():
        assert np.all(keras.ops.convert_to_numpy(w) == 1.0)
