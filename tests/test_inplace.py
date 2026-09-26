"""Tests for in-place elementwise fusion and zero-copy graph inputs.

Both memory reductions are planner-level features (`kiln.ir.memory`) that must
never change numerics, so this suite checks the safety rules, the arena
arithmetic and one compiled bit-exact differential run.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile

from pathlib import Path

import pytest

from kiln.backend.c99.emit_c import emit_c
from kiln.budget.report import audit_graph
from kiln.ir.kir import DType, Graph, Node, OpType, QuantParam, TensorDesc
from kiln.ir.memory import INPLACE_OPS, plan_graph_memory, plan_inplace_aliases
from kiln.ir.plan import allocate_arena, calculate_liveness
from kiln.ref.ref import run_ref

QUANT = QuantParam(multiplier=1073741824, shift=1)

RELU_TABLE = [max(0, index) for index in range(-128, 128)]

WEIGHTS = {
    "w0": [
        [[(oc * 5 + ic * 3 + k) % 9 - 4 for k in range(3)] for ic in range(4)]
        for oc in range(4)
    ],
    "w1": [
        [[(oc * 7 + ic * 5 + k) % 11 - 5 for k in range(3)] for ic in range(4)]
        for oc in range(8)
    ],
}

#: Element count of the two-stage toy graph input (4 channels x 24 frames).
LENGTH = 96


def _fused_graph() -> Graph:
    """Two fused stages: CONV1D -> LUT -> MAXPOOL1D, repeated, then ARGMAX."""
    tensors = {
        "in": TensorDesc("in", shape=(4, 24), dtype=DType.INT8),
        "conv": TensorDesc("conv", shape=(4, 24), dtype=DType.INT8, quant=QUANT),
        "relu": TensorDesc("relu", shape=(4, 24), dtype=DType.INT8),
        "pool": TensorDesc("pool", shape=(4, 12), dtype=DType.INT8),
        "conv2": TensorDesc("conv2", shape=(8, 12), dtype=DType.INT8, quant=QUANT),
        "relu2": TensorDesc("relu2", shape=(8, 12), dtype=DType.INT8),
        "pool2": TensorDesc("pool2", shape=(8, 1), dtype=DType.INT8),
        "out": TensorDesc("out", shape=(), dtype=DType.INT8),
    }
    nodes = [
        Node(OpType.CONV1D, "op0", inputs=["in"], outputs=["conv"], attrs={"weight": "w0", "stride": 1, "padding": 1}),
        Node(OpType.LUT, "op1", inputs=["conv"], outputs=["relu"], attrs={"table": RELU_TABLE}),
        Node(OpType.MAXPOOL1D, "op2", inputs=["relu"], outputs=["pool"], attrs={"kernel_size": 2, "stride": 2, "padding": 0}),
        Node(OpType.CONV1D, "op3", inputs=["pool"], outputs=["conv2"], attrs={"weight": "w1", "stride": 1, "padding": 1}),
        Node(OpType.LUT, "op4", inputs=["conv2"], outputs=["relu2"], attrs={"table": RELU_TABLE}),
        Node(OpType.MAXPOOL1D, "op5", inputs=["relu2"], outputs=["pool2"], attrs={"kernel_size": 12, "stride": 12, "padding": 0}),
        Node(OpType.ARGMAX, "op6", inputs=["pool2"], outputs=["out"]),
    ]
    return Graph(inputs=["in"], outputs=["out"], nodes=nodes, tensors=tensors)


def _branching_graph() -> Graph:
    """``conv`` feeds both a LUT and an ADD, so it must not be overwritten."""
    tensors = {
        "in": TensorDesc("in", shape=(4, 16), dtype=DType.INT8),
        "conv": TensorDesc("conv", shape=(4, 16), dtype=DType.INT8, quant=QUANT),
        "relu": TensorDesc("relu", shape=(4, 16), dtype=DType.INT8),
        "sum": TensorDesc("sum", shape=(4, 16), dtype=DType.INT8, quant=QUANT),
        "out": TensorDesc("out", shape=(), dtype=DType.INT8),
    }
    nodes = [
        Node(OpType.CONV1D, "op0", inputs=["in"], outputs=["conv"], attrs={"weight": "w", "stride": 1, "padding": 1}),
        Node(OpType.LUT, "op1", inputs=["conv"], outputs=["relu"], attrs={"table": RELU_TABLE}),
        Node(OpType.ADD, "op2", inputs=["conv", "relu"], outputs=["sum"]),
        Node(OpType.ARGMAX, "op3", inputs=["sum"], outputs=["out"]),
    ]
    return Graph(inputs=["in"], outputs=["out"], nodes=nodes, tensors=tensors)


def test_lut_only_is_an_inplace_op() -> None:
    assert INPLACE_OPS == frozenset({OpType.LUT})


def test_single_consumer_lut_aliases_its_input() -> None:
    assert dict(plan_inplace_aliases(_fused_graph())) == {"relu": "conv", "relu2": "conv2"}


def test_branching_input_is_never_aliased() -> None:
    assert dict(plan_inplace_aliases(_branching_graph())) == {}


def test_graph_input_consumer_is_never_aliased() -> None:
    """A LUT reading the graph input must not clobber the caller's buffer."""
    tensors = {
        "in": TensorDesc("in", shape=(4, 16), dtype=DType.INT8),
        "relu": TensorDesc("relu", shape=(4, 16), dtype=DType.INT8),
        "conv": TensorDesc("conv", shape=(4, 16), dtype=DType.INT8, quant=QUANT),
        "out": TensorDesc("out", shape=(), dtype=DType.INT8),
    }
    nodes = [
        Node(OpType.LUT, "op0", inputs=["in"], outputs=["relu"], attrs={"table": RELU_TABLE}),
        Node(OpType.CONV1D, "op1", inputs=["relu"], outputs=["conv"], attrs={"weight": "w0", "stride": 1, "padding": 1}),
        Node(OpType.ARGMAX, "op2", inputs=["conv"], outputs=["out"]),
    ]
    input_fed = Graph(inputs=["in"], outputs=["out"], nodes=nodes, tensors=tensors)
    assert dict(plan_inplace_aliases(input_fed)) == {}


def test_differently_sized_lut_is_never_aliased() -> None:
    """Sizes must match exactly: a shrinking LUT output cannot share bytes."""
    tensors = {
        "in": TensorDesc("in", shape=(4, 16), dtype=DType.INT8),
        "conv": TensorDesc("conv", shape=(4, 16), dtype=DType.INT8, quant=QUANT),
        "relu": TensorDesc("relu", shape=(4, 8), dtype=DType.INT8),
        "out": TensorDesc("out", shape=(), dtype=DType.INT8),
    }
    nodes = [
        Node(OpType.CONV1D, "op0", inputs=["in"], outputs=["conv"], attrs={"weight": "w0", "stride": 1, "padding": 1}),
        Node(OpType.LUT, "op1", inputs=["conv"], outputs=["relu"], attrs={"table": RELU_TABLE}),
        Node(OpType.ARGMAX, "op2", inputs=["relu"], outputs=["out"]),
    ]
    shrunk = Graph(inputs=["in"], outputs=["out"], nodes=nodes, tensors=tensors)
    assert dict(plan_inplace_aliases(shrunk)) == {}


def test_arena_shrinks_and_offsets_are_shared() -> None:
    graph = _fused_graph()
    liveness = calculate_liveness(graph.to_op_steps(), graph_outputs=graph.outputs)
    tensors = graph.to_tensor_infos()
    plain = allocate_arena(tensors, liveness)
    fused = allocate_arena(tensors, liveness, aliases=plan_inplace_aliases(graph))

    # plain: in(96) | conv(96) | relu(96) | ... -> 240 B
    # fused: conv+relu and conv2+relu2 each share one block -> 192 B
    assert plain.total_arena_bytes == 240
    assert fused.total_arena_bytes == 192
    assert fused.offsets["relu"] == fused.offsets["conv"]
    assert fused.offsets["relu2"] == fused.offsets["conv2"]


def test_zero_copy_inputs_leave_the_arena() -> None:
    plan = plan_graph_memory(_fused_graph())
    # identical to the fused plan minus the 96 B input copy: 192 -> 144 B
    assert "in" not in plan.offsets
    assert plan.total_arena_bytes == 144


def test_audit_matches_the_emitted_arena_size() -> None:
    graph = _fused_graph()
    code = emit_c(graph, WEIGHTS, runtime_header=None)
    report = audit_graph(graph, WEIGHTS)
    assert f"#define ARENA_SIZE {report.arena_ram_bytes}" in code


def test_emitted_c_reads_the_input_in_place() -> None:
    code = emit_c(_fused_graph(), WEIGHTS, runtime_header=None)
    assert "const int8_t *t_in = (const int8_t *)(const void *)input_bytes;" in code
    assert "input_bytes[i]" not in code


@pytest.mark.parametrize(
    ("aliases", "error", "match"),
    [
        ({"relu": "missing"}, KeyError, "unknown tensor"),
        ({"missing": "conv"}, KeyError, "unknown tensor"),
        ({"relu": "relu"}, ValueError, "self alias"),
        ({"conv": "relu", "relu": "conv"}, ValueError, "cyclic alias"),
    ],
)
def test_unsafe_alias_requests_are_rejected(aliases, error, match) -> None:
    graph = _fused_graph()
    liveness = calculate_liveness(graph.to_op_steps(), graph_outputs=graph.outputs)
    with pytest.raises(error, match=match):
        allocate_arena(graph.to_tensor_infos(), liveness, aliases=aliases)


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc required for C99 parity")
def test_inplace_fusion_is_bit_exact_against_the_reference() -> None:
    """The fused in-place LUT must produce identical C and reference results."""
    graph = _fused_graph()
    code = emit_c(graph, WEIGHTS, runtime_header=None)
    harness = code + f"""
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {{
    if (argc < {LENGTH + 1}) return 1;
    int8_t input[{LENGTH}];
    for (int i = 0; i < {LENGTH}; ++i) {{
        input[i] = (int8_t)atoi(argv[i + 1]);
    }}
    printf("%d\\n", (int)kiln_predict((const uint8_t *)input, {LENGTH}));
    return 0;
}}
"""

    inputs = [
        [((index * 37 + slot * 11) % 255) - 127 for slot in range(LENGTH)]
        for index in range(16)
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        source = Path(tmpdir) / "fused.c"
        binary = Path(tmpdir) / "fused.exe"
        source.write_text(harness, encoding="utf-8")
        subprocess.run(
            ["gcc", "-O3", "-Wall", "-Wextra", "-Werror", "-pedantic", str(source), "-o", str(binary)],
            check=True,
        )

        compiled = []
        for sample in inputs:
            result = subprocess.run(
                [str(binary), *[str(value) for value in sample]],
                capture_output=True,
                text=True,
                check=True,
            )
            compiled.append(int(result.stdout.strip()))

    reference = [
        run_ref(graph, {"in": sample}, WEIGHTS)["out"][0]
        for sample in inputs
    ]
    assert compiled == reference
