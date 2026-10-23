"""The accumulator of an hls4ml hex convolution must hold weight * input sums, not weight sums."""

from types import SimpleNamespace

import pytest

pytest.importorskip("hls4ml")
from hls4ml.model.types import FixedPrecisionType, IntegerPrecisionType

from keras_hexagdly.hls4ml_handler import _accum_type_str


def node(weight_precision, input_precision=None, bias_precision=None, *, model=None, layer=None):
    """What _accum_type_str reads: the weight, bias and input variable types, and the config.

    ``model`` / ``layer`` are the ``Model`` section and the layer's own ``LayerName`` entry of the
    hls4ml config; None means no hls4ml model at all (a stand-alone call)."""
    weights = SimpleNamespace(type=SimpleNamespace(precision=weight_precision, name="weight_t"))
    bias = SimpleNamespace(type=SimpleNamespace(precision=bias_precision or weight_precision))
    variable = SimpleNamespace(type=SimpleNamespace(precision=input_precision))
    config = None
    if model is not None or layer is not None:
        config = SimpleNamespace(
            get_layer_config_value=lambda _node, key, default=None: (layer or {}).get(key, default),
            get_config_value=lambda key, default=None: {"HLSConfig": {"Model": model or {}}}.get(
                key, default
            ),
        )
    return SimpleNamespace(
        get_weights=lambda name: bias if name == "mac_bias" else weights,
        get_input_variable=(lambda: variable) if input_precision is not None else (lambda: 1 / 0),
        model=None if config is None else SimpleNamespace(config=config),
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
        FixedPrecisionType(16, 3), FixedPrecisionType(4, 2), FixedPrecisionType(24, 12)
    )
    assert _accum_type_str(wide_bias, 7) == "ap_fixed<25, 12>"


# ---- ExactAccumulator / AccumPrecision ----
WEIGHTS, INPUT = FixedPrecisionType(16, 3), FixedPrecisionType(10, 2)  # 13 and 8 fractional bits


def test_by_default_the_accumulator_keeps_the_fractional_bits_of_the_finer_operand():
    assert _accum_type_str(node(WEIGHTS, INPUT), 56) == "ap_fixed<24, 11>"  # 13 fractional
    assert _accum_type_str(node(WEIGHTS, INPUT, model={}, layer={}), 56) == "ap_fixed<24, 11>"


def test_exact_accumulator_keeps_every_bit_of_the_product():
    exact = _accum_type_str(node(WEIGHTS, INPUT, model={"ExactAccumulator": True}), 56)
    assert exact == "ap_fixed<33, 12>"  # 13 + 8 = 21 fractional bits; 11 integer bits + 1 carry


def test_exact_accumulator_can_be_set_on_one_layer_and_that_wins_over_the_model():
    on_layer = node(WEIGHTS, INPUT, layer={"ExactAccumulator": True})
    assert _accum_type_str(on_layer, 56) == "ap_fixed<33, 12>"
    off_layer = node(
        WEIGHTS, INPUT, model={"ExactAccumulator": True}, layer={"ExactAccumulator": False}
    )
    assert _accum_type_str(off_layer, 56) == "ap_fixed<24, 11>"


def test_accum_precision_wins_over_exact_accumulator():
    both = node(
        WEIGHTS,
        INPUT,
        model={"ExactAccumulator": True},
        layer={"AccumPrecision": "ap_fixed<40,12>"},
    )
    assert _accum_type_str(both, 56) == "ap_fixed<40,12>"


def test_exact_accumulator_without_a_readable_input_type_only_adds_the_carry_bit():
    unknown_input = node(WEIGHTS, model={"ExactAccumulator": True})
    # products: 3 + 0 + 6 = 9 integer bits, 13 fractional; the carry of the bias addition makes 10
    assert _accum_type_str(unknown_input, 56) == "ap_fixed<23, 10>"


def test_exact_accumulator_keeps_the_fractional_bits_of_a_finer_bias():
    fine_bias = FixedPrecisionType(30, 4)  # 26 fractional bits, finer than the 21 of the product
    exact = node(WEIGHTS, INPUT, fine_bias, model={"ExactAccumulator": True})
    assert _accum_type_str(exact, 56) == "ap_fixed<38, 12>"  # 26 fractional, max(11, 4) + 1 integer
    default = node(WEIGHTS, INPUT, fine_bias)
    assert _accum_type_str(default, 56) == "ap_fixed<24, 11>"  # unchanged: the bias is truncated


def test_the_default_accumulator_has_no_carry_bit():
    """Existing exports must not change: the carry bit belongs to the exact accumulator only."""
    assert _accum_type_str(node(WEIGHTS, INPUT, model={"ExactAccumulator": False}), 56) == (
        "ap_fixed<24, 11>"
    )
