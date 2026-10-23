"""Phase 2 M2.1 acceptance tests: dataset manifest, provenance, and taxonomy invariants.

Design Gates verified:
- G2 (Frozen corpus): corpus and held-out real sets are checked-in artifacts
  whose bit-exact SHA-256 digests match manifest.json.
- G3 (Real held-out set): evaluated against real, never-synthesized code with zero leakage.
- Single-label 6-class security taxonomy contract.
"""

from __future__ import annotations

import json


from examples.hazard.graph import DEFAULT_LABELS, HAZARD_CLASSES
from examples.hazard.synthesize import (
    MANIFEST_PATH,
    load_corpus,
    load_held_out,
    verify_manifest,
)
from examples.hazard.tokenizer import TOKEN_COUNT, WINDOW_BYTES, tokenize


def test_manifest_verifies_bit_exactly() -> None:
    """Design Gate G2: manifest hash reproduces the corpus bit-exactly."""
    assert verify_manifest(), "manifest.json digests do not match files on disk"


def test_manifest_metadata_and_shapes() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["num_classes"] == DEFAULT_LABELS == len(HAZARD_CLASSES)
    assert manifest["classes"] == list(HAZARD_CLASSES)
    assert manifest["num_samples"] == 300
    assert manifest["num_held_out"] == 60
    assert manifest["token_count"] == TOKEN_COUNT == 96
    assert manifest["window_bytes"] == WINDOW_BYTES == 48
    assert manifest["split_seed"] == 1337


def test_corpus_samples_conformance() -> None:
    corpus = load_corpus()
    assert len(corpus) == 300

    class_counts = {cls: 0 for cls in HAZARD_CLASSES}
    for sample in corpus:
        sample.validate()
        class_counts[sample.label] += 1
        raw = sample.snippet.encode("ascii")
        assert len(raw) <= 80
        toks = list(tokenize(raw))
        assert len(toks) == TOKEN_COUNT

    # Perfectly balanced classes (50 per class)
    for cls, count in class_counts.items():
        assert count == 50, f"Class {cls} has {count} samples != 50"


def test_real_held_out_samples_conformance() -> None:
    held_out = load_held_out()
    assert len(held_out) == 60

    class_counts = {cls: 0 for cls in HAZARD_CLASSES}
    for sample in held_out:
        sample.validate()
        class_counts[sample.label] += 1
        raw = sample.snippet.encode("ascii")
        assert len(raw) <= 80
        toks = list(tokenize(raw))
        assert len(toks) == TOKEN_COUNT

    # 10 per class
    for cls, count in class_counts.items():
        assert count == 10, f"Class {cls} has {count} samples != 10"


def test_gate_g3_zero_leakage_between_corpus_and_held_out() -> None:
    """Design Gate G3: strictly zero overlap between training corpus and real held-out set."""
    corpus = load_corpus()
    held_out = load_held_out()

    corpus_snippets = {s.snippet for s in corpus}
    held_out_snippets = {s.snippet for s in held_out}

    overlap = corpus_snippets.intersection(held_out_snippets)
    assert len(overlap) == 0, f"Data leakage detected! Overlapping snippets: {overlap}"
