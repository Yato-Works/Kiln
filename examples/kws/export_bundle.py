"""Export the shipped KWS checkpoint as a portable KIR v1 bundle.

The exporter performs the same validation-split calibration as the Phase 1
benchmarks, then writes the graph and quantized weights without PyTorch class
names. Consumers of the generated bundle can validate, run, compile, and link
it without importing the PyTorch frontend.

Run from the repository root::

    python -m examples.kws.export_bundle
    python -m kiln validate examples/kws/kws.kir.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from examples.kws.benchmark import load_splits
from examples.kws.kws_graph import build_kws_model_graph
from kiln.ir.serialization import dumps_bundle


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "kws.kir.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output KIR bundle path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="optional QAT checkpoint override (default: the shipped checkpoint)",
    )
    args = parser.parse_args()

    specs, _, (_, validation_indices, _) = load_splits()
    lowered = build_kws_model_graph(args.weights)
    # build_kws_model_graph calibrates from the frozen validation split itself;
    # loading the split here makes the source data dependency explicit in the
    # exporter and catches a missing calibration corpus before writing output.
    if len(validation_indices) == 0 or len(specs) == 0:
        raise RuntimeError("KWS calibration corpus is empty")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        dumps_bundle(lowered.graph, lowered.weights, indent=None) + "\n", encoding="utf-8"
    )
    print(f"[SUCCESS] Exported KWS checkpoint -> {args.output}")
    print(
        f"           tensors={len(lowered.graph.tensors)}, nodes={len(lowered.graph.nodes)}, "
        f"weights={len(lowered.weights)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
