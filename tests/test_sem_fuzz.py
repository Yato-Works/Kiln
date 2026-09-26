from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest
from hypothesis import example, given, settings, strategies as st

from kiln.ir.sem import (
    INT8_MAX,
    INT8_MIN,
    INT32_MAX,
    INT32_MIN,
    clamp_int8,
    lut_lookup,
    requantize,
    rounding_divide_by_pot,
    saturating_rounding_doubling_high_mul,
)


@pytest.fixture(scope="module")
def sem_c_binary():
    """Compiles the C fixed-point math helpers to test bit-exact equivalence with sem.py."""
    c_source = """
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

static inline int32_t _clamp_int32(int64_t value) {
    if (value < INT32_MIN) return INT32_MIN;
    if (value > INT32_MAX) return INT32_MAX;
    return (int32_t)value;
}

static inline int32_t _srdhm(int32_t a, int32_t b) {
    if (a == INT32_MIN && b == INT32_MIN) return INT32_MAX;
    int64_t product = (int64_t)a * (int64_t)b;
    int64_t nudge = product >= 0 ? (int64_t)(1 << 30) : 1 - (int64_t)(1 << 30);
    int64_t result = (product + nudge) / (int64_t)(1LL << 31);
    return _clamp_int32(result);
}

static inline int32_t _round_divide_by_pot(int32_t value, int exponent) {
    if (exponent == 0) return value;
    int64_t magnitude = value < 0 ? -(int64_t)value : (int64_t)value;
    magnitude = (magnitude + (int64_t)(1LL << (exponent - 1))) >> exponent;
    return value < 0 ? -(int32_t)magnitude : (int32_t)magnitude;
}

static inline int8_t _requantize(int32_t accum, int32_t multiplier, int shift) {
    int32_t scaled = _srdhm(accum, multiplier);
    int32_t shifted = _round_divide_by_pot(scaled, shift);
    if (shifted < INT8_MIN) return INT8_MIN;
    if (shifted > INT8_MAX) return INT8_MAX;
    return (int8_t)shifted;
}

int main(void) {
    int cmd;
    // cmd 1: srdhm (a, b) -> int32
    // cmd 2: pot (val, exp) -> int32
    // cmd 3: requantize (accum, multiplier, shift) -> int8
    while (scanf("%d", &cmd) == 1) {
        if (cmd == 1) {
            long long a, b;
            if (scanf("%lld %lld", &a, &b) != 2) return 0;
            int32_t res = _srdhm((int32_t)a, (int32_t)b);
            printf("%d\\n", (int)res);
        } else if (cmd == 2) {
            long long v;
            int e;
            if (scanf("%lld %d", &v, &e) != 2) return 0;
            int32_t res = _round_divide_by_pot((int32_t)v, e);
            printf("%d\\n", (int)res);
        } else if (cmd == 3) {
            long long acc, mul;
            int s;
            if (scanf("%lld %lld %d", &acc, &mul, &s) != 3) return 0;
            int8_t res = _requantize((int32_t)acc, (int32_t)mul, s);
            printf("%d\\n", (int)res);
        }
        fflush(stdout);
    }
    return 0;
}
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "sem_test.c"
        exe = Path(tmpdir) / "sem_test.exe"
        src.write_text(c_source, encoding="utf-8")
        subprocess.run(["gcc", "-O3", "-Wall", "-Werror", str(src), "-o", str(exe)], check=True)

        proc = subprocess.Popen([str(exe)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        yield proc
        proc.stdin.close()
        proc.terminate()
        proc.wait()


class TestSemanticsFuzz:
    """Extreme edge case & property-based tests for fixed-point math in sem.py vs C99."""

    @settings(max_examples=300)
    @given(
        a=st.integers(INT32_MIN, INT32_MAX),
        b=st.integers(INT32_MIN, INT32_MAX),
    )
    @example(a=INT32_MIN, b=INT32_MIN)  # Critical saturating branch
    @example(a=INT32_MIN, b=INT32_MAX)
    @example(a=INT32_MAX, b=INT32_MAX)
    @example(a=0, b=INT32_MIN)
    @example(a=0, b=0)
    @example(a=1 << 30, b=1 << 30)
    def test_srdhm_extreme_boundaries(self, sem_c_binary, a: int, b: int) -> None:
        py_res = saturating_rounding_doubling_high_mul(a, b)
        sem_c_binary.stdin.write(f"1 {a} {b}\n")
        sem_c_binary.stdin.flush()
        c_res = int(sem_c_binary.stdout.readline().strip())
        assert py_res == c_res, f"SRDHM mismatch: a={a}, b={b} -> py={py_res}, c={c_res}"

    @settings(max_examples=200)
    @given(
        value=st.integers(INT32_MIN, INT32_MAX),
        exponent=st.integers(0, 31),
    )
    @example(value=INT32_MIN, exponent=0)
    @example(value=INT32_MIN, exponent=31)
    @example(value=INT32_MAX, exponent=0)
    @example(value=INT32_MAX, exponent=31)
    @example(value=-1, exponent=1)
    @example(value=1, exponent=1)
    def test_pot_divide_boundaries(self, sem_c_binary, value: int, exponent: int) -> None:
        py_res = rounding_divide_by_pot(value, exponent)
        sem_c_binary.stdin.write(f"2 {value} {exponent}\n")
        sem_c_binary.stdin.flush()
        c_res = int(sem_c_binary.stdout.readline().strip())
        assert py_res == c_res, f"POT mismatch: val={value}, exp={exponent} -> py={py_res}, c={c_res}"

    @settings(max_examples=300)
    @given(
        accum=st.integers(INT32_MIN, INT32_MAX),
        multiplier=st.integers(1 << 30, (1 << 31) - 1),
        shift=st.integers(0, 31),
    )
    @example(accum=INT32_MIN, multiplier=1 << 30, shift=0)
    @example(accum=INT32_MAX, multiplier=(1 << 31) - 1, shift=0)
    @example(accum=INT32_MIN, multiplier=(1 << 31) - 1, shift=31)
    @example(accum=0, multiplier=1 << 30, shift=1)
    def test_requantize_boundaries(self, sem_c_binary, accum: int, multiplier: int, shift: int) -> None:
        py_res = requantize(accum, multiplier, shift)
        sem_c_binary.stdin.write(f"3 {accum} {multiplier} {shift}\n")
        sem_c_binary.stdin.flush()
        c_res = int(sem_c_binary.stdout.readline().strip())
        assert py_res == c_res, f"Requantize mismatch: acc={accum}, mul={multiplier}, shift={shift} -> py={py_res}, c={c_res}"