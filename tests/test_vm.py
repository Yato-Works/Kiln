"""Tests for the portable KIR v1 virtual target."""

from __future__ import annotations

from pathlib import Path

import pytest

from kiln.cli import main
from kiln.ir.kir import DType, Graph, Node, OpType, TensorDesc
from kiln.ir.serialization import dumps_bundle
from kiln.ref.ref import run_ref
from kiln.vm import VMError, VirtualMachine


def _graph() -> Graph:
    return Graph(
        inputs=["x"],
        outputs=["label"],
        tensors={
            "x": TensorDesc("x", (2,), DType.INT8),
            "scores": TensorDesc("scores", (2,), DType.INT8),
            "label": TensorDesc("label", (), DType.INT8),
        },
        nodes=[
            Node(OpType.LUT, "lut", ["x"], ["scores"], {"table": list(range(-128, 128))}),
            Node(OpType.ARGMAX, "argmax", ["scores"], ["label"]),
        ],
    )


def test_vm_loads_bundle_and_reports_deterministic_metrics(tmp_path: Path) -> None:
    bundle = tmp_path / "model.kir.json"
    bundle.write_text(dumps_bundle(_graph(), {}), encoding="utf-8")

    vm = VirtualMachine.from_bundle(bundle)
    first = vm.run({"x": [-2, 7]})
    second = vm.run({"x": [-2, 7]})

    assert first == second
    assert first.outputs == {"label": [1]}
    assert first.metrics.instructions == 2
    assert first.metrics.cycles > 0
    assert first.metrics.allocation_bytes == 0
    assert first.metrics.int32_overflow_free is True
    assert [event.op for event in first.trace] == ["lut", "argmax"]


def test_vm_rejects_invalid_input_domain(tmp_path: Path) -> None:
    bundle = tmp_path / "model.kir.json"
    bundle.write_text(dumps_bundle(_graph(), {}), encoding="utf-8")
    vm = VirtualMachine.from_bundle(bundle)

    with pytest.raises(VMError, match="outside"):
        vm.run({"x": [-2, 128]})


def test_vm_matches_the_independent_reference_on_the_full_instruction_path() -> None:
    from examples.hazard.graph import build_bag_graph, random_weights

    graph = build_bag_graph(tokens=4, dim=2, labels=3)
    weights = random_weights(dim=2, labels=3, seed=7)
    inputs = {"tokens": [0, 3, 9, 255]}

    vm_result = VirtualMachine(graph, weights).run(inputs)
    reference = run_ref(graph, inputs, weights)

    assert vm_result.outputs == reference
    assert [event.op for event in vm_result.trace] == [
        "embedding",
        "matmul",
        "maxpool1d",
        "argmax",
    ]


def test_vm_cli_accepts_single_input_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = tmp_path / "model.kir.json"
    inputs = tmp_path / "input.json"
    bundle.write_text(dumps_bundle(_graph(), {}), encoding="utf-8")
    inputs.write_text("[-2, 7]", encoding="utf-8")

    assert main(["vm", str(bundle), "--input", str(inputs), "--json", "--trace"]) == 0
    output = capsys.readouterr().out
    assert '"outputs": {' in output
    assert '"label": [' in output
    assert '"op": "argmax"' in output
