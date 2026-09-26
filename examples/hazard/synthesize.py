"""Phase 2 M2.1: Dataset synthesis, curation, and manifest pipeline.

This module provides the synthesis harness and corpus management conforming to
Design Gates G2 (Frozen corpus) and G3 (Real held-out set):
- Deterministic curation of synthetic Python code snippets across 6 hazard classes.
- Real held-out evaluation set never seen during training or validation.
- Canonical JSON and SHA-256 manifest generation for bit-exact reproducibility.
- OpenRouter API synthesis integration for iterative expansion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from examples.hazard.graph import DEFAULT_LABELS, HAZARD_CLASSES
from examples.hazard.tokenizer import TOKEN_COUNT, WINDOW_BYTES, tokenize

__all__ = (
    "CORPUS_PATH",
    "HELD_OUT_PATH",
    "MANIFEST_PATH",
    "CorpusManifest",
    "HazardSample",
    "build_corpus",
    "load_corpus",
    "load_held_out",
    "save_dataset",
    "verify_manifest",
)

DATA_DIR: Final = Path(__file__).resolve().parent / "data"
CORPUS_PATH: Final = DATA_DIR / "hazard_corpus.json"
HELD_OUT_PATH: Final = DATA_DIR / "held_out_real.json"
MANIFEST_PATH: Final = DATA_DIR / "manifest.json"
SEEDS_PATH: Final = DATA_DIR / "seeds.json"
HELD_OUT_RAW_PATH: Final = DATA_DIR / "held_out_raw.json"


@dataclass(frozen=True, slots=True)
class HazardSample:
    """A single labeled code snippet in the hazard dataset."""

    snippet: str
    label: str
    label_id: int

    def validate(self) -> None:
        """Asserts that the sample conforms strictly to the M2.0 contract."""
        assert self.label in HAZARD_CLASSES, f"Unknown class: {self.label}"
        assert HAZARD_CLASSES[self.label_id] == self.label, "Mismatched label_id"
        raw_bytes = self.snippet.encode("ascii")
        assert len(raw_bytes) <= 80, f"Snippet too long ({len(raw_bytes)} B > 80): {self.snippet}"
        tokens = list(tokenize(raw_bytes))
        assert len(tokens) == TOKEN_COUNT, f"Expected {TOKEN_COUNT} tokens, got {len(tokens)}"


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    """Design Gate G2 manifest ensuring bit-exact corpus provenance."""

    corpus_sha256: str
    held_out_sha256: str
    num_samples: int
    num_held_out: int
    num_classes: int
    classes: list[str]
    token_count: int
    window_bytes: int
    split_seed: int
    teacher_model: str


def _compute_sha256(data: bytes) -> str:
    """Computes SHA-256 digest hex string."""
    return hashlib.sha256(data).hexdigest()


def _canonical_json(obj: Any) -> str:
    """Produces sorted, deterministic JSON with Unix newlines."""
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def build_corpus(*, seed: int = 1337) -> list[HazardSample]:
    """Builds and validates the deterministic 300-sample corpus from seeds.json."""
    if not SEEDS_PATH.exists():
        raise FileNotFoundError(f"Missing seeds template file: {SEEDS_PATH}")
    seeds_data: dict[str, list[str]] = json.loads(SEEDS_PATH.read_text(encoding="utf-8"))

    held_out_raw = json.loads(HELD_OUT_RAW_PATH.read_text(encoding="utf-8"))
    held_out_snippets = {item["snippet"] for item in held_out_raw}

    samples: list[HazardSample] = []
    class_to_id = {cls: idx for idx, cls in enumerate(HAZARD_CLASSES)}

    for label, snippets in seeds_data.items():
        label_id = class_to_id[label]
        for snippet in snippets:
            assert snippet not in held_out_snippets, f"Leakage detected: {snippet} in held-out set!"
            sample = HazardSample(snippet=snippet, label=label, label_id=label_id)
            sample.validate()
            samples.append(sample)

    rng = random.Random(seed)
    rng.shuffle(samples)
    return samples


def save_dataset() -> CorpusManifest:
    """Builds, validates, writes dataset files and saves the frozen manifest."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    corpus_samples = build_corpus()
    corpus_dicts = [asdict(s) for s in corpus_samples]
    corpus_json_text = _canonical_json(corpus_dicts)
    corpus_bytes = corpus_json_text.encode("utf-8")
    CORPUS_PATH.write_bytes(corpus_bytes)
    corpus_hash = _compute_sha256(corpus_bytes)

    held_out_raw = json.loads(HELD_OUT_RAW_PATH.read_text(encoding="utf-8"))
    held_out_samples = []
    class_to_id = {cls: idx for idx, cls in enumerate(HAZARD_CLASSES)}
    for item in held_out_raw:
        sample = HazardSample(
            snippet=item["snippet"],
            label=item["label"],
            label_id=class_to_id[item["label"]],
        )
        sample.validate()
        held_out_samples.append(asdict(sample))

    held_out_json_text = _canonical_json(held_out_samples)
    held_out_bytes = held_out_json_text.encode("utf-8")
    HELD_OUT_PATH.write_bytes(held_out_bytes)
    held_out_hash = _compute_sha256(held_out_bytes)

    manifest = CorpusManifest(
        corpus_sha256=corpus_hash,
        held_out_sha256=held_out_hash,
        num_samples=len(corpus_samples),
        num_held_out=len(held_out_samples),
        num_classes=DEFAULT_LABELS,
        classes=list(HAZARD_CLASSES),
        token_count=TOKEN_COUNT,
        window_bytes=WINDOW_BYTES,
        split_seed=1337,
        teacher_model="meta-llama/llama-3.3-70b-instruct",
    )
    manifest_bytes = _canonical_json(asdict(manifest)).encode("utf-8")
    MANIFEST_PATH.write_bytes(manifest_bytes)
    return manifest


def load_corpus() -> list[HazardSample]:
    """Loads and validates the frozen corpus from disk."""
    if not CORPUS_PATH.exists():
        save_dataset()
    data = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    samples = [HazardSample(**item) for item in data]
    for s in samples:
        s.validate()
    return samples


def load_held_out() -> list[HazardSample]:
    """Loads and validates the real held-out evaluation set from disk."""
    if not HELD_OUT_PATH.exists():
        save_dataset()
    data = json.loads(HELD_OUT_PATH.read_text(encoding="utf-8"))
    samples = [HazardSample(**item) for item in data]
    for s in samples:
        s.validate()
    return samples


def verify_manifest() -> bool:
    """Verifies that the files on disk match the recorded SHA-256 digests in manifest.json."""
    if not MANIFEST_PATH.exists() or not CORPUS_PATH.exists() or not HELD_OUT_PATH.exists():
        return False
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    current_corpus_hash = _compute_sha256(CORPUS_PATH.read_bytes())
    current_held_out_hash = _compute_sha256(HELD_OUT_PATH.read_bytes())
    return (
        current_corpus_hash == manifest["corpus_sha256"]
        and current_held_out_hash == manifest["held_out_sha256"]
    )


def query_openrouter_snippets(
    cls_name: str,
    count: int = 10,
    model: str = "meta-llama/llama-3.3-70b-instruct",
    api_key: str | None = None,
) -> list[str]:
    """Query OpenRouter to generate new snippet candidates for active distillation."""
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/smily/Kiln",
        "X-Title": "Kiln Hazard Distillation",
    }
    prompt = f"""Generate {count} distinct, realistic Python single-line code snippets for vulnerability class: "{cls_name}".
Guidelines:
- Each snippet must be 1 line of valid Python, <= 70 chars.
- ASCII characters only.
- Output JSON array of strings: ["code1", "code2", ...] without markdown."""

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Output valid JSON array of strings only."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 500,
        "temperature": 0.7,
    }
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        raw = res["choices"][0]["message"]["content"].strip()
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiln Hazard Dataset Synthesis & Verification")
    parser.add_argument("--save", action="store_true", help="Generate and save dataset files")
    parser.add_argument("--verify", action="store_true", help="Verify manifest hashes")
    args = parser.parse_args()

    if args.save:
        manifest = save_dataset()
        print(f"Dataset generated successfully.")
        print(f"  Corpus SHA-256  : {manifest.corpus_sha256}")
        print(f"  Held-out SHA-256: {manifest.held_out_sha256}")
        print(f"  Samples         : {manifest.num_samples} (training pool)")
        print(f"  Held-out Real   : {manifest.num_held_out} (G3 control)")
    elif args.verify:
        ok = verify_manifest()
        print(f"Manifest verification: {'PASS' if ok else 'FAIL'}")
        if not ok:
            exit(1)
    else:
        manifest = save_dataset()
        print(f"Corpus built and verified. Samples: {manifest.num_samples}, SHA256: {manifest.corpus_sha256[:16]}...")


if __name__ == "__main__":
    main()

