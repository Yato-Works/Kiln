from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from hypothesis import example, given, settings, strategies as st

from kiln.backend.c99.emit_c import emit_c
from kiln.ir.kir import DType, Graph, Node, OpType, QuantParam, TensorDesc
from kiln.ref.ref import run_ref


def _has_gcc() -> bool:
    return shutil.which("gcc") is not None


@pytest.fixture(scope="module")
def pipeline_c_binary():
    """Compiles a complete 7-operation Kiln pipeline into a persistent executable for fuzzing."""
    if not _has_gcc():
        pytest.skip("gcc compiler is required for C99 fuzz tests")

    qp = QuantParam(multiplier=1073741824, shift=1)

    tensors = {
        "in": TensorDesc("in", shape=(2,), dtype=DType.UINT8),
        "emb": TensorDesc("emb", shape=(2, 8), dtype=DType.INT8),
        "conv": TensorDesc("conv", shape=(2, 8), dtype=DType.INT8, quant=qp),
        "lut": TensorDesc("lut", shape=(2, 8), dtype=DType.INT8),
        "sum": TensorDesc("sum", shape=(2, 8), dtype=DType.INT8, quant=qp),
        "fc": TensorDesc("fc", shape=(2, 4), dtype=DType.INT8, quant=qp),
        "pool": TensorDesc("pool", shape=(2, 1), dtype=DType.INT8),
        "out": TensorDesc("out", shape=(), dtype=DType.INT8),
    }

    w_emb = [[(i * 3 + j * 5) % 30 - 15 for j in range(8)] for i in range(256)]
    w_conv = [[[(oc * 7 + ic * 3 + k) % 15 - 7 for k in range(3)] for ic in range(2)] for oc in range(2)]
    lut_table = [((i - 128) // 3) for i in range(256)]
    w_fc = [[(o * 11 + f * 7) % 20 - 10 for f in range(8)] for o in range(4)]

    weights = {"w_emb": w_emb, "w_conv": w_conv, "w_fc": w_fc}

    nodes = [
        Node(OpType.EMBEDDING, "op_emb", inputs=["in"], outputs=["emb"], attrs={"weight": "w_emb"}),
        Node(OpType.CONV1D, "op_conv", inputs=["emb"], outputs=["conv"], attrs={"weight": "w_conv", "stride": 1, "padding": 1}),
        Node(OpType.LUT, "op_lut", inputs=["conv"], outputs=["lut"], attrs={"table": lut_table}),
        Node(OpType.ADD, "op_add", inputs=["conv", "lut"], outputs=["sum"]),
        Node(OpType.MATMUL, "op_fc", inputs=["sum"], outputs=["fc"], attrs={"weight": "w_fc"}),
        Node(OpType.MAXPOOL1D, "op_pool", inputs=["fc"], outputs=["pool"], attrs={"kernel_size": 4, "stride": 4, "padding": 0}),
        Node(OpType.ARGMAX, "op_argmax", inputs=["pool"], outputs=["out"]),
    ]

    graph = Graph(inputs=["in"], outputs=["out"], nodes=nodes, tensors=tensors)
    c_code = emit_c(graph, weights, runtime_header=None)

    # Fast IO harness reading (b0, b1) per line until EOF
    harness = c_code + """
#include <stdio.h>
#include <stdint.h>

int main(void) {
    int b0, b1;
    while (scanf("%d %d", &b0, &b1) == 2) {
        uint8_t input[2];
        input[0] = (uint8_t)b0;
        input[1] = (uint8_t)b1;
        int8_t res = kiln_predict(input, 2);
        printf("%d\\n", (int)res);
        fflush(stdout);
    }
    return 0;
}
"""
    tmpdir = tempfile.mkdtemp()
    src_path = Path(tmpdir) / "fuzz_model.c"
    exe_path = Path(tmpdir) / "fuzz_model.exe"
    src_path.write_text(harness, encoding="utf-8")

    compile_cmd = ["gcc", "-O3", "-Wall", "-Wextra", "-Werror", "-pedantic", str(src_path), "-o", str(exe_path)]
    subprocess.run(compile_cmd, check=True)

    proc = subprocess.Popen(
        [str(exe_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    yield (proc, graph, weights)

    proc.stdin.close()
    proc.terminate()
    proc.wait()
    shutil.rmtree(tmpdir, ignore_errors=True)


def _eval_c_and_ref(pipeline_c_binary, b0: int, b1: int) -> tuple[int, int]:
    proc, graph, weights = pipeline_c_binary
    proc.stdin.write(f"{b0} {b1}\n")
    proc.stdin.flush()
    c_out_line = proc.stdout.readline()
    c_pred = int(c_out_line.strip())

    ref_out = run_ref(graph, {"in": [b0, b1]}, weights)
    ref_pred = ref_out["out"][0]
    return c_pred, ref_pred


@pytest.mark.skipif(not _has_gcc(), reason="gcc compiler is required for C99 fuzz tests")
class TestFuzzDifferential:
    """Rigorous differential fuzzing with Hypothesis against GCC -O3 binary."""

    @settings(max_examples=250)
    @given(b0=st.integers(0, 255), b1=st.integers(0, 255))
    @example(b0=0, b1=0)
    @example(b0=255, b1=255)
    @example(b0=127, b1=128)
    @example(b0=128, b1=127)
    @example(b0=0x55, b1=0xAA)
    @example(b0=0xAA, b1=0x55)
    @example(b0=1, b1=0)
    @example(b0=0, b1=1)
    def test_hypothesis_random_fuzz(self, pipeline_c_binary, b0: int, b1: int) -> None:
        """Tests 150 arbitrary random byte combinations."""
        c_pred, ref_pred = _eval_c_and_ref(pipeline_c_binary, b0, b1)
        assert c_pred == ref_pred, f"Bit mismatch for inputs ({b0}, {b1}): C={c_pred}, Ref={ref_pred}"

    def test_extreme_edge_cases(self, pipeline_c_binary) -> None:
        """Exhaustively tests all boundary conditions and pathological patterns."""
        edge_patterns = [
            # All 0x00 and all 0xFF
            (0x00, 0x00),
            (0xFF, 0xFF),
            (0x00, 0xFF),
            (0xFF, 0x00),
            # Alternating bit patterns (0x55, 0xAA)
            (0x55, 0x55),
            (0xAA, 0xAA),
            (0x55, 0xAA),
            (0xAA, 0x55),
            # Sparse (one byte zero, one byte non-zero)
            (0x01, 0x00),
            (0x00, 0x01),
            (0x80, 0x00),
            (0x00, 0x80),
            (0x7F, 0x00),
            (0x00, 0x7F),
            # Boundary values
            (127, 128),
            (128, 127),
            (64, 192),
            (192, 64),
        ]
        for b0, b1 in edge_patterns:
            c_pred, ref_pred = _eval_c_and_ref(pipeline_c_binary, b0, b1)
            assert c_pred == ref_pred, f"Bit mismatch for edge pattern ({b0:#04x}, {b1:#04x}): C={c_pred}, Ref={ref_pred}"