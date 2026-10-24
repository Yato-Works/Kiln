"""Regression tests for the shipped, PyTorch-free Hazard KIR artifact."""

from __future__ import annotations

import json
from pathlib import Path

from examples.hazard.graph import build_hazard_model
from kiln.ir.serialization import dumps_bundle
from kiln.ref.ref import run_ref
from kiln.vm import VirtualMachine

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "examples" / "hazard" / "hazard.kir.json"
EXAMPLE_INPUT = ROOT / "examples" / "hazard" / "data" / "example_tokens.json"


def test_shipped_hazard_bundle_is_reproducible() -> None:
    graph, weights = build_hazard_model()

    assert BUNDLE.read_text(encoding="utf-8") == dumps_bundle(graph, weights) + "\n"


def test_shipped_hazard_bundle_runs_on_the_vm_and_matches_reference() -> None:
    vm = VirtualMachine.from_bundle(BUNDLE)
    tokens = json.loads(EXAMPLE_INPUT.read_text(encoding="utf-8"))
    inputs = {graph_input: tokens for graph_input in vm.graph.inputs}

    result = vm.run(inputs)
    reference = run_ref(vm.graph, inputs, vm.weights)

    assert result.outputs == reference == {"label": [1]}
    assert result.metrics.arena_bytes == 1344
    assert result.metrics.rom_bytes == 2096
    assert result.metrics.allocation_bytes == 0
    assert result.metrics.int32_overflow_free is True
