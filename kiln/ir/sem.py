from __future__ import annotations


from typing import Final, Sequence


__all__ = (
    "INT8_MIN",
    "INT8_MAX",
    "INT32_MIN",
    "INT32_MAX",
    "clamp_int8",
    "saturating_rounding_doubling_high_mul",
    "rounding_divide_by_pot",
    "requantize",
    "lut_lookup",
)


INT8_MIN: Final = -(1 << 7)

INT8_MAX: Final = (1 << 7) - 1


INT32_MIN: Final = -(1 << 31)

INT32_MAX: Final = (1 << 31) - 1


_Q31_SCALE: Final = 1 << 31

_Q31_ROUND: Final = 1 << 30

_MAX_SHIFT: Final = 31

_LUT_SIZE: Final = 1 << 8


def _require_int8(value: int) -> int:

    if not INT8_MIN <= value <= INT8_MAX:
        raise ValueError(f"int8 overflow: {value}")

    return value


def _require_int32(value: int) -> int:

    if not INT32_MIN <= value <= INT32_MAX:
        raise ValueError(f"int32 overflow: {value}")

    return value


def _require_shift(exponent: int) -> int:

    if not 0 <= exponent <= _MAX_SHIFT:
        raise ValueError(f"invalid shift: {exponent}")

    return exponent


def _divide_toward_zero(numerator: int, denominator: int) -> int:

    magnitude = abs(numerator) // denominator

    return -magnitude if numerator < 0 else magnitude


def _clamp(value: int, low: int, high: int) -> int:

    return max(low, min(high, value))


def clamp_int8(value: int) -> int:

    return _clamp(value, INT8_MIN, INT8_MAX)


def saturating_rounding_doubling_high_mul(a: int, b: int) -> int:

    _require_int32(a)

    _require_int32(b)

    if a == INT32_MIN and b == INT32_MIN:
        return INT32_MAX

    product = a * b

    rounding = _Q31_ROUND if product >= 0 else 1 - _Q31_ROUND

    result = _divide_toward_zero(product + rounding, _Q31_SCALE)

    return _clamp(result, INT32_MIN, INT32_MAX)


def rounding_divide_by_pot(value: int, exponent: int) -> int:

    _require_int32(value)

    _require_shift(exponent)

    if exponent == 0:
        return value

    magnitude = (abs(value) + (1 << (exponent - 1))) >> exponent

    return -magnitude if value < 0 else magnitude


def requantize(accum: int, multiplier: int, shift: int) -> int:

    _require_int32(accum)

    _require_int32(multiplier)

    _require_shift(shift)

    scaled = saturating_rounding_doubling_high_mul(accum, multiplier)

    shifted = rounding_divide_by_pot(scaled, shift)

    return clamp_int8(shifted)


def lut_lookup(value: int, table: Sequence[int]) -> int:

    _require_int8(value)

    if len(table) != _LUT_SIZE:
        raise ValueError(f"invalid LUT size: {len(table)}")

    result = table[value - INT8_MIN]

    return _require_int8(result)
