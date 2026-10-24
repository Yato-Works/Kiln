"""Export the shipped Hazard student as a portable KIR v1 bundle.

The generated JSON is the hand-off artifact for users who want to run, inspect,
compile, or link the model without importing PyTorch. The exporter itself uses
the checked-in graph factory and weight artifact, so regeneration is deterministic.

Run from the repository root::

    python -m examples.hazard.export_bundle
    python -m kiln validate examples/hazard/hazard.kir.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from examples.hazard.graph import build_hazard_model
from kiln.ir.serialization import dumps_bundle


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "hazard.kir.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output KIR bundle path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    graph, weights = build_hazard_model()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dumps_bundle(graph, weights) + "\n", encoding="utf-8")
    print(f"[SUCCESS] Exported hazard student -> {args.output}")
    print(
        f"           tensors={len(graph.tensors)}, nodes={len(graph.nodes)}, weights={len(weights)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
