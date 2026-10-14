"""The accumulator of an hls4ml hex convolution must hold weight * input sums, not weight sums."""
from types import SimpleNamespace

import pytest

pytest.importorskip("hls4ml")
from hls4ml.model.types import FixedPrecisionType, IntegerPrecisionType

from keras_hexagdly.hls4ml_handler import _accum_type_str


def node(weight_precision, input_precision=None, bias_precision=None):
    """What _accum_type_str reads: the weight, bias and input variable types."""
    weights = SimpleNamespace(type=SimpleNamespace(precision=weight_precision, name="weight_t"))
    bias = SimpleNamespace(type=SimpleNamespace(precision=bias_precision or weight_precision))
    variable = SimpleNamespace(type=SimpleNamespace(precision=input_precision))
    return SimpleNamespace(
        get_weights=lambda name: bias if name == "mac_bias" else weights,
        get_input_variable=(lambda: variable) if input_precision is not None else (lambda: 1 / 0),
    )


def test_input_integer_bits_are_added_to_the_sum_growth():
    # 112 terms -> 7 bits for the sum; the input <18,7> brings 7 more integer bits
    accum = _accum_type_str(node(FixedPrecisionType(16, 3), FixedPrecisionType(18, 7)), 112)
    assert accum == "ap_fixed<30, 17>"


def test_a_larger_input_gives_a_wider_accumulator():
    narrow = node(FixedPrecisionType(16, 3), FixedPrecisionType(10, 2))
    wide = node(FixedPrecisionType(16, 3), FixedPrecisionType(20, 11))
    assert _accum_type_str(wide, 56) == "ap_fixed<33, 20>"
    assert _accum_type_str(narrow, 56) == "ap_fixed<24, 11>"


def test_integer_and_unsigned_input_types():
    integer_input = node(FixedPrecisionType(16, 3), IntegerPrecisionType(8, signed=False))
    assert _accum_type_str(integer_input, 7) == "ap_fixed<27, 14>"
    unsigned_weights = node(FixedPrecisionType(16, 3, signed=False), FixedPrecisionType(8, 4))
    assert _accum_type_str(unsigned_weights, 8) == "ap_ufixed<23, 10>"


def test_unreadable_input_type_falls_back_to_the_weight_growth():
    assert _accum_type_str(node(FixedPrecisionType(16, 3)), 112) == "ap_fixed<23, 10>"


def test_non_fixed_weight_type_keeps_the_old_fallback():
    assert _accum_type_str(node(object(), FixedPrecisionType(18, 7)), 112) == "weight_t"


def test_finer_input_fraction_is_kept():
    # weights <8,3> have 5 fractional bits, the input <18,2> has 16: the data must not be cut to 5
    finer = node(FixedPrecisionType(8, 3), FixedPrecisionType(18, 2))
    assert _accum_type_str(finer, 7) == "ap_fixed<24, 8>"
    # a coarser input does not shrink the weights' fractional bits
    coarser = node(FixedPrecisionType(16, 3), FixedPrecisionType(6, 4))
    assert _accum_type_str(coarser, 7) == "ap_fixed<23, 10>"


def test_wide_bias_widens_the_integer_part():
    # products need 3 + 2 + 3 = 8 integer bits, the bias 12: the accumulator is seeded with it
    wide_bias = node(
        FixedPrecisionType(16, 3), FixedPrecisionType(4, 2), FixedPrecisionType(24, 12))
    assert _accum_type_str(wide_bias, 7) == "ap_fixed<25, 12>"
