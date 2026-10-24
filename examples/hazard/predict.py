"""Run the shipped Hazard KIR bundle on a source snippet.

This is the user-facing, torch-free path. The tokenizer is the normative
Hazard contract; the VM consumes the resulting KIR input without importing the
training or PyTorch frontend code.

Examples::

    python -m examples.hazard.predict --text "print(eval(user_input))"
    python -m examples.hazard.predict --file snippet.py --json
    Get-Content snippet.py | python -m examples.hazard.predict --stdin
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from examples.hazard.labels import HAZARD_CLASSES
from examples.hazard.tokenizer import tokenize
from kiln.vm import VirtualMachine

DEFAULT_BUNDLE = Path(__file__).resolve().parent / "hazard.kir.json"


def _read_source(args: argparse.Namespace) -> str:
    selected = sum(value is not None for value in (args.text, args.file)) + int(args.stdin)
    if selected != 1:
        raise ValueError("choose exactly one of --text, --file, or --stdin")
    if args.text is not None:
        return args.text
    if args.file is not None:
        return args.file.read_text(encoding="utf-8")
    return sys.stdin.read()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="source snippet to classify")
    source.add_argument("--file", type=Path, help="UTF-8 source file to classify")
    source.add_argument("--stdin", action="store_true", help="read the source snippet from stdin")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE, help="KIR bundle path")
    parser.add_argument("--json", action="store_true", help="output machine-readable JSON")
    args = parser.parse_args(argv)

    try:
        snippet = _read_source(args)
        tokens = list(tokenize(snippet))
        result = VirtualMachine.from_bundle(args.bundle).run({"tokens": tokens})
        label_id = int(result.outputs["label"][0])
        label = HAZARD_CLASSES[label_id]
    except (OSError, IndexError, KeyError, TypeError, ValueError) as exc:
        if args.json:
            print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        else:
            print(f"[ERROR] Hazard prediction failed: {exc}", file=sys.stderr)
        return 1

    payload = {
        "status": "ok",
        "label": label,
        "label_id": label_id,
        "token_count": len(tokens),
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
        print(f"tokens: {len(tokens)}")
        print(f"cycles: {result.metrics.cycles}")
        print(f"arena: {result.metrics.arena_bytes} B; ROM: {result.metrics.rom_bytes} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
