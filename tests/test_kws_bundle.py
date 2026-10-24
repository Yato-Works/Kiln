"""Regression tests for the shipped, portable KWS KIR artifact."""

from __future__ import annotations

import json
from pathlib import Path

from examples.kws.kws_graph import build_kws_model_graph
from kiln.ir.serialization import dumps_bundle
from kiln.ref.ref import run_ref
from kiln.vm import VirtualMachine

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "examples" / "kws" / "kws.kir.json"
EXAMPLE_INPUT = ROOT / "examples" / "kws" / "example_input.json"


def test_shipped_kws_bundle_is_reproducible() -> None:
    lowered = build_kws_model_graph()

    assert (
        BUNDLE.read_text(encoding="utf-8")
        == dumps_bundle(lowered.graph, lowered.weights, indent=None) + "\n"
    )


def test_shipped_kws_bundle_runs_on_the_vm_and_matches_reference() -> None:
    vm = VirtualMachine.from_bundle(BUNDLE)
    features = json.loads(EXAMPLE_INPUT.read_text(encoding="utf-8"))

    result = vm.run({"graph_in": features})
    reference = run_ref(vm.graph, {"graph_in": features}, vm.weights)

    assert result.outputs == reference == {"graph_out": [0]}
    assert result.metrics.arena_bytes == 1992
    assert result.metrics.rom_bytes == 39614
    assert result.metrics.allocation_bytes == 0
    assert result.metrics.int32_overflow_free is True
