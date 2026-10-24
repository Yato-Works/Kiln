"""Run the shipped KWS KIR bundle on quantized log-mel features.

The input is a JSON array of exactly 490 signed int8 values (10 mel bins x 49
frames), not raw PCM audio. Feature extraction is intentionally a separate
frontend contract; the companion C frontend handles PCM on supported builds.

Example::

    python -m examples.kws.predict --input examples/kws/example_input.json --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from examples.kws.labels import KWS_CLASSES
from kiln.vm import VirtualMachine

DEFAULT_BUNDLE = Path(__file__).resolve().parent / "kws.kir.json"


def _load_input(path: Path) -> list[int]:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("graph_in")
    if not isinstance(raw, list) or not all(isinstance(value, int) for value in raw):
        raise ValueError("input JSON must be an integer array or an object with 'graph_in'")
    return raw


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="quantized log-mel feature JSON")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE, help="KIR bundle path")
    parser.add_argument("--json", action="store_true", help="output machine-readable JSON")
    args = parser.parse_args(argv)

    try:
        features = _load_input(args.input)
        result = VirtualMachine.from_bundle(args.bundle).run({"graph_in": features})
        label_id = int(result.outputs["graph_out"][0])
        label = KWS_CLASSES[label_id]
    except (OSError, IndexError, KeyError, TypeError, ValueError) as exc:
        if args.json:
            print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        else:
            print(f"[ERROR] KWS prediction failed: {exc}")
        return 1

    payload = {
        "status": "ok",
        "label": label,
        "label_id": label_id,
        "feature_count": len(features),
        "metrics": {
            "cycles": result.metrics.cycles,
            "arena_bytes": result.metrics.arena_bytes,
            "rom_bytes": result.metrics.rom_bytes,
            "allocation_bytes": result.metrics.allocation_bytes,
        },
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"prediction: {label} ({label_id})")
        print(f"features: {len(features)} int8 values")
        print(f"cycles: {result.metrics.cycles}")
        print(f"arena: {result.metrics.arena_bytes} B; ROM: {result.metrics.rom_bytes} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
