"""M2.2 acceptance: the exported int8 student — contract, budget, bit-exactness.

Design Gates verified:
- G5 (selection by validation): the shipped metrics come from the frozen 1337
  split, and the artifact records how the checkpoint was selected.
- Budget: the *trained* weights pass arena < 2 KB / ROM < 40 KB.
- Bit-exactness: the independent int64 PyTorch forward, ``kiln.ref.run_ref``
  and the gcc-compiled emitted C99 return identical labels for all 300 corpus
  and 60 real held-out samples.

The suite reads the checked-in artifact; it never re-trains.  Reproducing the
artifact itself is ``python -m examples.hazard.train`` (G6).
"""

from __future__ import annotations

import random
import shutil
from typing import Any

import pytest
import torch

from examples.hazard.graph import (
    DEFAULT_DIM,
    DEFAULT_LABELS,
    build_bag_graph,
    label_quant,
)
from examples.hazard.synthesize import load_corpus, load_held_out
from examples.hazard.tokenizer import (
    BIGRAM_BASE,
    BIGRAM_BUCKETS,
    TOKEN_COUNT,
    VOCAB_SIZE,
    fnv1a32,
    tokenize,
)
from examples.hazard.train import (
    SEED,
    SPACE_BIGRAM_ID,
    SPACE_BYTE_ID,
    first_argmax,
    load_student,
    predict_c,
    predict_int,
    predict_ref,
    requantize_int64,
    split_indices,
)
from kiln.budget.report import audit_graph
from kiln.ir.sem import requantize as sem_requantize


@pytest.fixture(scope="module")
def student() -> dict[str, Any]:
    return load_student()


@pytest.fixture(scope="module")
def w_dict(student: dict[str, Any]) -> dict[str, Any]:
    return {"w_emb": student["w_emb"], "w_mm": student["w_mm"]}


@pytest.fixture(scope="module")
def all_rows() -> list[list[int]]:
    """Every sample the bit-exact check must cover: 300 corpus + 60 held-out."""
    corpus = load_corpus()
    held_out = load_held_out()
    return [list(tokenize(sample.snippet)) for sample in corpus + held_out]


@pytest.fixture(scope="module")
def graph():
    return build_bag_graph()


@pytest.fixture(scope="module")
def reference_labels(graph, w_dict, all_rows) -> list[int]:
    return predict_ref(graph, w_dict, all_rows)


def test_artifact_contract(student: dict[str, Any]) -> None:
    assert student["format"] == "kiln-hazard-student"
    assert student["version"] == 1
    assert student["seed"] == SEED == 1337
    assert (student["dim"], student["labels"]) == (DEFAULT_DIM, DEFAULT_LABELS)
    assert (student["token_count"], student["vocab_size"]) == (TOKEN_COUNT, VOCAB_SIZE)

    w_emb, w_mm = student["w_emb"], student["w_mm"]
    assert len(w_emb) == VOCAB_SIZE
    assert all(len(row) == DEFAULT_DIM for row in w_emb)
    assert len(w_mm) == DEFAULT_LABELS
    assert all(len(row) == DEFAULT_DIM for row in w_mm)
    cells = [value for row in w_emb + w_mm for value in row]
    assert all(isinstance(value, int) for value in cells)
    assert min(cells) >= -128 and max(cells) <= 127


def test_label_quant_matches_the_shipped_artifact(student: dict[str, Any], graph) -> None:
    """``build_bag_graph()`` with no arguments must ship the calibrated scale."""
    quant = graph.tensors["logits"].quant
    assert (quant.multiplier, quant.shift) == (
        student["label_multiplier"],
        student["label_shift"],
    )
    default = label_quant()
    assert (default.multiplier, default.shift) == (quant.multiplier, quant.shift)


def test_pad_sentinels_are_the_documented_ids() -> None:
    assert SPACE_BYTE_ID == 0x20 - 0x1F == 1
    assert SPACE_BIGRAM_ID == BIGRAM_BASE + fnv1a32(b"  ") % BIGRAM_BUCKETS == 199


def test_split_is_the_frozen_1337_scheme() -> None:
    train, val, test = split_indices(300)
    assert (len(train), len(val), len(test)) == (210, 45, 45)
    assert sorted(train + val + test) == list(range(300))
    assert split_indices(300) == (train, val, test)  # deterministic


def test_argmax_breaks_ties_toward_the_first_index() -> None:
    rows = torch.tensor([[1.0, 1.0, -0.5], [-1.0, 0.0, 0.0]], dtype=torch.float32)
    assert first_argmax(rows) == [0, 1]  # ties resolve to the lowest index


def test_requantize_port_is_bit_exact_against_kiln_sem(student: dict[str, Any]) -> None:
    """The training-time int64 port must equal ``kiln.ir.sem.requantize``.

    This is the guard that keeps QAT optimising the arithmetic the emitted C
    actually runs: SRDHM rounding, toward-zero division, power-of-two rounding
    and int8 saturation, across extreme and boundary accumulators.
    """
    pairs = [
        (student["label_multiplier"], student["label_shift"]),
        (1, 0),
        (715827883, 1),  # ~0.5 in Q31
        (2**31 - 1, 31),
        (1073741824, 30),
    ]
    accumulators = [
        -(2**31),
        -(2**31) + 1,
        -129072,
        -5080,
        -127,
        -1,
        0,
        1,
        127,
        5080,
        129072,
        2**31 - 1,
    ]
    generator = random.Random(SEED)
    accumulators += [generator.randint(-(2**31), 2**31 - 1) for _ in range(150)]

    values = torch.tensor(accumulators, dtype=torch.int64)
    for multiplier, shift in pairs:
        expected = [sem_requantize(value, multiplier, shift) for value in accumulators]
        actual = requantize_int64(values, multiplier, shift).tolist()
        assert actual == expected, f"mismatch for multiplier={multiplier} shift={shift}"


def test_int_forward_is_bit_exact_with_kiln_reference(
    w_dict, all_rows, graph, reference_labels
) -> None:
    """M2.2 core: PyTorch int forward == ``kiln.ref`` on all 360 samples."""
    quant = graph.tensors["logits"].quant
    direct = predict_int(w_dict, all_rows, quant.multiplier, quant.shift)
    assert direct == reference_labels


def test_emitted_c_is_bit_exact_with_reference(w_dict, all_rows, graph, reference_labels) -> None:
    """M2.2 core: gcc-compiled emitted C == ``kiln.ref`` on all 360 samples."""
    if shutil.which("gcc") is None:
        pytest.skip("gcc is required for the emitted-C bit-exactness check")
    compiled = predict_c(graph, w_dict, all_rows)
    assert compiled is not None
    assert compiled == reference_labels


def test_budget_of_the_trained_weights(student, w_dict, graph) -> None:
    """M2.2 acceptance: arena/ROM audited on the *trained* weights."""
    report = audit_graph(graph, w_dict)
    assert report.arena_ram_bytes == TOKEN_COUNT * (DEFAULT_DIM + DEFAULT_LABELS) == 1344
    assert report.arena_ram_bytes < 2048
    assert report.total_rom_bytes == (VOCAB_SIZE + DEFAULT_LABELS) * DEFAULT_DIM == 2096
    assert report.total_rom_bytes < 40960
    assert report.total_macs == TOKEN_COUNT * DEFAULT_DIM * DEFAULT_LABELS == 4608
    # The artifact's own audit must agree with a fresh one.
    assert student["budget"]["arena_ram_bytes"] == report.arena_ram_bytes
    assert student["budget"]["total_rom_bytes"] == report.total_rom_bytes


def test_reported_metrics_clear_the_floor(student: dict[str, Any]) -> None:
    """G5: metrics reported for the frozen split, with sane floors.

    Measured on the shipped run: float test 68.9%, int8 test/held-out 73.3%,
    zero saturation; the floors keep ~6 points of slack for environment drift.
    """
    metrics = student["metrics"]
    assert metrics["int_val"] >= 65.0
    assert metrics["int_test"] >= 65.0
    assert metrics["int_held_out"] >= 65.0
    assert metrics["float_test"] >= 60.0
    assert metrics["float_int_agreement_test"] >= 95.0
    assert metrics["saturation_rate"] < 0.01
    # G5: training records val-only selection inputs, never test-driven ones.
    assert student["training"]["float_best_val_acc"] >= metrics["float_val"] - 1e-9
    assert student["training"]["qat_best_val_acc"] >= metrics["int_val"] - 1e-9
    assert student["training"]["qat"] is True
    assert student["training"]["tau"] == [1.0, 0.08]
