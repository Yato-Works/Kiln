"""Phase 2 M2.2: the int8 hazard student — train, quantise, verify, export.

Run from the repository root::

    python -m examples.hazard.train             # float + PTQ + QAT + verify + export
    python -m examples.hazard.train --no-qat    # float + post-training quantisation only

Pipeline (G1: features frozen first; G5: selection on validation only)
---------------------------------------------------------------------

1. **Data** — load the M2.1 frozen corpus and the real held-out set, tokenize
   with the M2.0 contract, split 210/45/45 with seed 1337 (the same scheme
   Phase 1's ``split_indices`` used).
2. **Float distillation** — ``BagClassifier`` (``nn.Embedding`` + bias-free
   ``nn.Linear``) trained with a smooth-max (logsumexp) surrogate whose tau
   anneals 1.0 -> 0.08.  Every *evaluation* uses the hard max-pool, because
   hard max-pool is what ``MAXPOOL1D(axis=0)`` computes in the emitted graph.
3. **Post-training int8** — both tables quantised per-tensor, symmetric,
   ``max|w| / 127`` (``kiln.frontend.pytorch.quantize_tensor``); the label
   requantisation scale is calibrated from the integer accumulators over
   train+val and compared against ``_LABEL_SCALE`` in ``examples/hazard/graph.py``.
4. **QAT** (on by default) — fine-tunes through the *exact* integer pipeline:
   an int64 port of ``kiln.ir.sem.requantize`` with straight-through gradients,
   hard max-pool, cross-entropy.  Checkpoints are selected on validation int8
   accuracy (G5), never on test or held-out.
5. **Verification** — the independent int64 PyTorch forward,
   ``kiln.ref.run_ref`` and (when gcc is present) the compiled emitted C99
   must return identical labels for all 300 corpus + 60 held-out samples.
6. **Export** — ``hazard_int8_weights.json`` plus the ``kiln.budget.report``
   audit (arena < 2 KB, ROM < 40 KB).

The last line of output is a machine-readable ``RESULT {json}`` line.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Final, Sequence

import torch
import torch.nn as nn

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
from kiln.backend.c99.emit_c import emit_c
from kiln.budget.report import audit_graph
from kiln.frontend.pytorch import compute_multiplier_and_shift, quantize_tensor
from kiln.ref.ref import run_ref

__all__ = (
    "ARTIFACT_PATH",
    "BagClassifier",
    "SEED",
    "accuracy",
    "calibrate_label_scale",
    "load_student",
    "predict_c",
    "predict_float",
    "predict_int",
    "predict_ref",
    "requantize_int64",
    "split_indices",
)

#: Where the trained student lands (checked in; loaded by the tests/benchmark).
ARTIFACT_PATH: Final = Path(__file__).resolve().parent / "hazard_int8_weights.json"

#: Frozen split seed (G2), shared with the corpus manifest.
SEED: Final = 1337

#: The two vocabulary rows the normal form fills with padding: the byte id of
#: a space and the ``"  "`` bigram bucket.  They start as strong negative
#: evidence so padding can never win the max-pool.
SPACE_BYTE_ID: Final = 0x20 - 0x1F
SPACE_BIGRAM_ID: Final = BIGRAM_BASE + fnv1a32(b"  ") % BIGRAM_BUCKETS
PAD_EMBEDDING_INIT: Final = -5.0

#: Smooth-max annealing of the float stage (tau -> ~hard max).
TAU_START: Final = 1.0
TAU_END: Final = 0.08

LR: Final = 0.03
WEIGHT_DECAY: Final = 1e-4
EPOCHS: Final = 180

QAT_LR: Final = 0.01
QAT_EPOCHS: Final = 40

#: gcc flags for the emitted C (same contract as tests/test_hazard_tokenizer).
GCC_FLAGS: Final = ("-std=c99", "-O3", "-Wall", "-Wextra", "-Werror", "-pedantic")

#: Harness appended to the emitted translation unit: one count-prefixed file
#: of token vectors in, one label per line out (the Phase 1a benchmark pattern).
_HARNESS: Final = """
#include <stdio.h>
#include <stdint.h>

int main(int argc, char **argv) {{
    if (argc < 2) return 1;
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 2;
    int32_t count = 0;
    if (fread(&count, sizeof(int32_t), 1, f) != 1) return 3;
    for (int32_t iter = 0; iter < count; ++iter) {{
        uint8_t input[{tokens}];
        if (fread(input, sizeof(uint8_t), {tokens}, f) != {tokens}) return 4;
        printf("%d\\n", (int)kiln_predict(input, {tokens}));
    }}
    fclose(f);
    return 0;
}}
"""


def split_indices(count: int) -> tuple[list[int], list[int], list[int]]:
    """The frozen 70/15/15 split (seed 1337), the Phase 1 scheme."""
    generator = torch.Generator().manual_seed(SEED)
    order = torch.randperm(count, generator=generator).tolist()
    train_end = int(count * 0.70)
    val_end = train_end + int(count * 0.15)
    return order[:train_end], order[train_end:val_end], order[val_end:]


class BagClassifier(nn.Module):
    """Float student: a token bag scored per class before the pooling op."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, DEFAULT_DIM)
        self.classifier = nn.Linear(DEFAULT_DIM, DEFAULT_LABELS, bias=False)
        with torch.no_grad():
            self.embedding.weight[SPACE_BYTE_ID] = PAD_EMBEDDING_INIT
            self.embedding.weight[SPACE_BIGRAM_ID] = PAD_EMBEDDING_INIT

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(N, TOKEN_COUNT)`` ids -> ``(N, TOKEN_COUNT, labels)`` scores."""
        return self.classifier(self.embedding(tokens))


def pooled_logits(logits: torch.Tensor, tau: float | None) -> torch.Tensor:
    """Max-pool over the token axis.

    ``tau=None`` is the deployed hard pool (``MAXPOOL1D(axis=0)``); a finite
    tau is the smooth-max surrogate the float stage trains with.
    """
    if tau is None:
        return logits.max(dim=1).values
    return tau * torch.logsumexp(logits / tau, dim=1)


def first_argmax(rows: torch.Tensor) -> list[int]:
    """The ``ARGMAX`` rule from ``kiln.ref``: the *first* index of the max."""
    width = rows.shape[-1]
    predictions: list[int] = []
    for row in rows:
        best = 0
        best_value = row[0].item()
        for index in range(1, width):
            value = row[index].item()
            if value > best_value:
                best, best_value = index, value
        predictions.append(best)
    return predictions


def accuracy(predictions: Sequence[int], labels: Sequence[int]) -> float:
    """Percent agreement between predicted and gold class ids."""
    if not labels:
        return 0.0
    hits = sum(int(predicted) == int(gold) for predicted, gold in zip(predictions, labels))
    return 100.0 * hits / len(labels)


def requantize_int64(
    accum: torch.Tensor,
    multiplier: int,
    shift: int,
    *,
    clamp: bool = True,
) -> torch.Tensor:
    """Bit-exact torch port of ``kiln.ir.sem.requantize`` (int32 math in int64).

    Written independently of ``kiln.ref`` so the two implementations actually
    check each other: SRDHM rounding, toward-zero division, power-of-two
    rounding, int8 saturation — the exact sequence the emitted C runs.
    ``clamp=False`` returns the pre-saturation value (used for diagnostics).
    """
    product = accum * int(multiplier)
    rounding = torch.where(product >= 0, 1 << 30, 1 - (1 << 30))
    scaled = torch.div(product + rounding, 1 << 31, rounding_mode="trunc")
    scaled = scaled.clamp(-(1 << 31), (1 << 31) - 1)
    if shift:
        magnitude = (scaled.abs() + (1 << (shift - 1))) >> shift
        scaled = torch.where(scaled < 0, -magnitude, magnitude)
    if not clamp:
        return scaled
    return scaled.clamp(-128, 127)


def fake_quantize(weight: torch.Tensor) -> torch.Tensor:
    """``quantize_tensor``'s per-tensor rule with a straight-through gradient.

    Scale arithmetic mirrors ``kiln.frontend.pytorch.quantize_tensor`` exactly
    (float64 ``max|w| / 127`` from ``.item()``) so the QAT forward and the
    exported table always agree cell-for-cell.
    """
    max_value = float(weight.abs().max().item())
    if max_value < 1e-7:
        return weight + (torch.zeros_like(weight) - weight).detach()
    scale = max_value / 127.0  # python float, same as quantize_tensor
    quantised = torch.round(weight / scale).clamp(-127, 127)
    return weight + (quantised - weight).detach()


class _RequantizeSTE(torch.autograd.Function):
    """Exact int requantise forward, ``scale``-proportional gradient backward."""

    @staticmethod
    def forward(ctx, accum: torch.Tensor, multiplier: int, shift: int, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return requantize_int64(accum.to(torch.int64), multiplier, shift).to(accum.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output * ctx.scale, None, None, None


def export_weights(model: BagClassifier) -> dict[str, Any]:
    """Per-tensor symmetric int8 export of both tables (the ``run_ref`` dict)."""
    w_emb, _ = quantize_tensor(model.embedding.weight.data)
    w_mm, _ = quantize_tensor(model.classifier.weight.data)
    return {"w_emb": w_emb, "w_mm": w_mm}


def _accumulators(w_dict: dict[str, Any], rows: Sequence[Sequence[int]]) -> torch.Tensor:
    """Integer ``MATMUL`` accumulators: ``(N, TOKEN_COUNT, labels)`` int64."""
    embedding = torch.tensor(w_dict["w_emb"], dtype=torch.int64)  # (vocab, dim)
    head = torch.tensor(w_dict["w_mm"], dtype=torch.int64)  # (labels, dim)
    tokens = torch.tensor(rows, dtype=torch.int64)  # (N, tokens)
    gathered = embedding[tokens]  # (N, tokens, dim)
    factors = head.view(1, 1, *head.shape)
    return (gathered.unsqueeze(2) * factors).sum(dim=3)


def predict_float(model: BagClassifier, rows: Sequence[Sequence[int]]) -> list[int]:
    """Hard-max-pool (deployed semantics) float predictions."""
    tokens = torch.tensor(rows, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        pooled = pooled_logits(model(tokens), tau=None)
    return first_argmax(pooled)


def predict_int(w_dict: dict[str, Any], rows: Sequence[Sequence[int]], multiplier: int, shift: int) -> list[int]:
    """The full int8 graph: ``EMBEDDING`` -> ``MATMUL`` -> requantise -> pool -> argmax."""
    logits = requantize_int64(_accumulators(w_dict, rows), multiplier, shift)
    return first_argmax(logits.max(dim=1).values)


def calibrate_label_scale(w_dict: dict[str, Any], rows: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Max-abs calibration of the label requantise (over train+val, G5).

    Returns the proposed ``(scale, multiplier, shift)``; compare against
    ``graph.tensors["logits"].quant`` to see if ``_LABEL_SCALE`` in
    ``examples/hazard/graph.py`` should be updated.
    """
    max_abs = int(_accumulators(w_dict, rows).abs().max())
    scale = 127.0 / max_abs
    multiplier, shift = compute_multiplier_and_shift(scale)
    return {"scale": scale, "multiplier": multiplier, "shift": shift, "max_abs_accum": max_abs}


def saturation_rate(w_dict: dict[str, Any], rows: Sequence[Sequence[int]], multiplier: int, shift: int) -> float:
    """Fraction of per-token logits the int8 clamp touches (0.0 is ideal)."""
    raw = requantize_int64(_accumulators(w_dict, rows), multiplier, shift, clamp=False)
    return float((raw.abs() > 127).float().mean())


def predict_ref(graph, w_dict: dict[str, Any], rows: Sequence[Sequence[int]]) -> list[int]:
    """Labels from ``kiln.ref.run_ref`` — the independent reference VM."""
    return [run_ref(graph, {"tokens": row}, w_dict)["label"][0] for row in rows]


def predict_c(graph, w_dict: dict[str, Any], rows: Sequence[Sequence[int]]) -> list[int] | None:
    """Labels from the compiled emitted C99, or ``None`` when gcc is absent."""
    if shutil.which("gcc") is None:
        return None
    source_text = emit_c(graph, w_dict, runtime_header=None) + _HARNESS.format(tokens=TOKEN_COUNT)
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "hazard_student.c"
        binary_path = Path(tmpdir) / "hazard_student.exe"
        source_path.write_text(source_text, encoding="utf-8")
        build = subprocess.run(
            ["gcc", *GCC_FLAGS, str(source_path), "-o", str(binary_path)],
            capture_output=True,
            text=True,
        )
        if build.returncode != 0:
            raise RuntimeError(f"emitted C failed to compile:\n{build.stderr}")
        payload_path = Path(tmpdir) / "inputs.bin"
        with payload_path.open("wb") as handle:
            handle.write(struct.pack("<i", len(rows)))
            for row in rows:
                handle.write(struct.pack(f"{TOKEN_COUNT}B", *row))
        run = subprocess.run([str(binary_path), str(payload_path)], capture_output=True, text=True)
        if run.returncode != 0:
            raise RuntimeError(f"C harness exited {run.returncode}: {run.stderr}")
    return [int(line) for line in run.stdout.split()]


def load_student(path: Path = ARTIFACT_PATH) -> dict[str, Any]:
    """Loads the exported int8 student artifact (weights + metadata)."""
    return json.loads(path.read_text(encoding="utf-8"))


def train_float(
    rows: Sequence[Sequence[int]],
    labels: Sequence[int],
    val_rows: Sequence[Sequence[int]],
    val_labels: Sequence[int],
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    verbose: bool = True,
) -> tuple[BagClassifier, dict[str, float]]:
    """Smooth-max annealed float distillation (G5: best hard-max val accuracy)."""
    torch.manual_seed(SEED)
    model = BagClassifier()
    tokens = torch.tensor(rows, dtype=torch.long)
    gold = torch.tensor(labels, dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_acc = -1.0
    best_state: dict[str, torch.Tensor] = {}
    final_loss = float("nan")
    for epoch in range(1, epochs + 1):
        progress = (epoch - 1) / max(epochs - 1, 1)
        tau = TAU_START * (TAU_END / TAU_START) ** progress
        model.train()
        optimizer.zero_grad()
        loss = criterion(pooled_logits(model(tokens), tau), gold)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
        val_acc = accuracy(predict_float(model, val_rows), val_labels)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if verbose and (epoch % 20 == 0 or epoch == epochs):
            print(f"  epoch {epoch:3d}/{epochs}  tau {tau:.3f}  loss {final_loss:.4f}  val {val_acc:5.1f}%")
    model.load_state_dict(best_state)
    return model, {"best_val_acc": best_acc, "final_loss": final_loss}


def train_qat(
    model: BagClassifier,
    rows: Sequence[Sequence[int]],
    labels: Sequence[int],
    val_rows: Sequence[Sequence[int]],
    val_labels: Sequence[int],
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    multiplier: int,
    shift: int,
    scale: float,
    verbose: bool = True,
) -> tuple[BagClassifier, dict[str, float]]:
    """QAT through the exact integer pipeline (G5: best validation int8 accuracy).

    Forward: fake-quantised tables -> int GEMM -> real requantise (STE) ->
    hard max-pool -> cross-entropy.  That is the deployed graph, gradient
    aside, so what validation measures is what the VM computes.
    """
    tokens = torch.tensor(rows, dtype=torch.long)
    gold = torch.tensor(labels, dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_acc = -1.0
    best_state: dict[str, torch.Tensor] = {}
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        embedding = fake_quantize(model.embedding.weight)
        head = fake_quantize(model.classifier.weight)
        gathered = embedding[tokens]  # (N, tokens, dim)
        accum = gathered @ head.T  # (N, tokens, labels)
        logits = _RequantizeSTE.apply(accum, multiplier, shift, scale)
        loss = criterion(pooled_logits(logits, tau=None), gold)
        loss.backward()
        optimizer.step()
        weights = export_weights(model)
        val_acc = accuracy(predict_int(weights, val_rows, multiplier, shift), val_labels)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if verbose and (epoch % 10 == 0 or epoch == epochs):
            print(f"  qat epoch {epoch:3d}/{epochs}  int val {val_acc:5.1f}%")
    model.load_state_dict(best_state)
    return model, {"best_val_acc": best_acc}


def _format_artifact(artifact: dict[str, Any]) -> str:
    """JSON with the two weight tables kept on one line each (reviewable diffs)."""
    meta = {key: value for key, value in artifact.items() if key not in ("w_emb", "w_mm")}
    meta_text = json.dumps(meta, indent=2).removesuffix("\n}")
    lines = [
        meta_text + ",",
        f'  "w_emb": {json.dumps(artifact["w_emb"], separators=(",", ":"))},',
        f'  "w_mm": {json.dumps(artifact["w_mm"], separators=(",", ":"))}',
        "}",
    ]
    return "\n".join(lines) + "\n"


def _banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 M2.2: train and export the int8 hazard student")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="float smooth-max training epochs")
    parser.add_argument("--qat-epochs", type=int, default=QAT_EPOCHS, help="quantisation-aware fine-tune epochs")
    parser.add_argument("--lr", type=float, default=LR, help="float-stage AdamW learning rate")
    parser.add_argument("--qat-lr", type=float, default=QAT_LR, help="QAT AdamW learning rate")
    parser.add_argument("--no-qat", action="store_true", help="skip QAT; deploy post-training quantisation")
    parser.add_argument("--no-gcc", action="store_true", help="skip the compiled emitted-C check")
    parser.add_argument("--no-export", action="store_true", help="verify only; do not write the artifact")
    args = parser.parse_args(argv)

    _banner("PHASE 2 M2.2: HAZARD STUDENT (int8, 6 classes, 96 tokens x dim 8)")

    # ---- data: frozen corpus, frozen split, real held-out never seen (G2/G5) ----
    corpus = load_corpus()
    held_out = load_held_out()
    all_rows = [list(tokenize(sample.snippet)) for sample in corpus]
    all_labels = [sample.label_id for sample in corpus]
    held_rows = [list(tokenize(sample.snippet)) for sample in held_out]
    held_labels = [sample.label_id for sample in held_out]
    train_index, val_index, test_index = split_indices(len(corpus))
    train_rows = [all_rows[i] for i in train_index]
    train_labels = [all_labels[i] for i in train_index]
    val_rows = [all_rows[i] for i in val_index]
    val_labels = [all_labels[i] for i in val_index]
    test_rows = [all_rows[i] for i in test_index]
    test_labels = [all_labels[i] for i in test_index]
    print(
        f"corpus {len(corpus)} -> train {len(train_rows)} / val {len(val_rows)} / test {len(test_rows)}"
        f"  + held-out {len(held_rows)} real snippets"
    )

    # ---- stage 1: float distillation with smooth-max annealing ----
    _banner(f"STAGE 1: FLOAT SMOOTH-MAX (lr {args.lr}, wd {WEIGHT_DECAY}, tau {TAU_START} -> {TAU_END})")
    model, float_history = train_float(
        train_rows, train_labels, val_rows, val_labels,
        epochs=args.epochs, lr=args.lr, weight_decay=WEIGHT_DECAY,
    )
    float_val = accuracy(predict_float(model, val_rows), val_labels)
    float_test = accuracy(predict_float(model, test_rows), test_labels)
    float_held = accuracy(predict_float(model, held_rows), held_labels)
    print(f"  hard-max float: val {float_val:.1f}%  test {float_test:.1f}%  held-out {float_held:.1f}%")

    # ---- stage 2: post-training int8 + label-scale calibration ----
    _banner("STAGE 2: POST-TRAINING INT8 + LABEL-SCALE CALIBRATION")
    weights = export_weights(model)
    calibration = calibrate_label_scale(weights, train_rows + val_rows)
    graph = build_bag_graph(label_scale=calibration["scale"])
    quant = graph.tensors["logits"].quant
    multiplier, shift = int(quant.multiplier), int(quant.shift)
    deployed_scale = calibration["scale"]
    default_quant = label_quant()
    if (multiplier, shift) != (default_quant.multiplier, default_quant.shift):
        default_scale = default_quant.multiplier / float(1 << 31) * (2.0 ** -default_quant.shift)
        print(
            "  WARNING: graph.py _LABEL_SCALE no longer matches this calibration: constant "
            f"{default_scale:.6g} vs calibrated {deployed_scale:.6g} "
            f"(max|accum| {calibration['max_abs_accum']}) - update _LABEL_SCALE to "
            f"{deployed_scale:.6g} in examples/hazard/graph.py and re-run."
        )
    else:
        print(f"  label scale {deployed_scale:.6g} == graph.py _LABEL_SCALE (max|accum| {calibration['max_abs_accum']})")
    pre_qat = {
        "val": accuracy(predict_int(weights, val_rows, multiplier, shift), val_labels),
        "test": accuracy(predict_int(weights, test_rows, multiplier, shift), test_labels),
        "held_out": accuracy(predict_int(weights, held_rows, multiplier, shift), held_labels),
        "saturation_rate": saturation_rate(weights, train_rows + val_rows, multiplier, shift),
    }
    print(
        f"  PTQ int8: val {pre_qat['val']:.1f}%  test {pre_qat['test']:.1f}%  "
        f"held-out {pre_qat['held_out']:.1f}%  saturating {pre_qat['saturation_rate'] * 100:.2f}% of logits"
    )

    # ---- stage 3: quantisation-aware fine-tuning (exact integer forward) ----
    qat_history: dict[str, float] = {}
    if args.no_qat:
        _banner("STAGE 3: QAT SKIPPED (--no-qat)")
    else:
        _banner(f"STAGE 3: QAT (lr {args.qat_lr}, {args.qat_epochs} epochs, exact requantise forward)")
        model, qat_history = train_qat(
            model, train_rows, train_labels, val_rows, val_labels,
            epochs=args.qat_epochs, lr=args.qat_lr, weight_decay=WEIGHT_DECAY,
            multiplier=multiplier, shift=shift, scale=deployed_scale,
        )
        weights = export_weights(model)
        recalibration = calibrate_label_scale(weights, train_rows + val_rows)
        if (recalibration["multiplier"], recalibration["shift"]) != (multiplier, shift):
            print(
                f"  note: post-QAT accumulators calibrate to {recalibration['scale']:.6g} "
                f"(deployed stays {deployed_scale:.6g}; ratio {recalibration['scale'] / deployed_scale:.3f})"
            )

    int_val = accuracy(predict_int(weights, val_rows, multiplier, shift), val_labels)
    int_test = accuracy(predict_int(weights, test_rows, multiplier, shift), test_labels)
    int_held = accuracy(predict_int(weights, held_rows, multiplier, shift), held_labels)
    saturation = saturation_rate(weights, train_rows + val_rows, multiplier, shift)
    float_test_preds = predict_float(model, test_rows)
    int_test_preds = predict_int(weights, test_rows, multiplier, shift)
    agreement_test = accuracy(int_test_preds, float_test_preds)
    print(
        f"  final int8: val {int_val:.1f}%  test {int_test:.1f}%  held-out {int_held:.1f}%"
        f"  saturating {saturation * 100:.2f}%  float-int agreement(test) {agreement_test:.1f}%"
    )

    # ---- stage 4: bit-exact PyTorch <-> kiln.ref <-> emitted C99 ----
    _banner("STAGE 4: BIT-EXACT VERIFICATION (pytorch int forward vs run_ref vs emitted C)")
    verification_rows = all_rows + held_rows
    direct_preds = predict_int(weights, verification_rows, multiplier, shift)
    ref_preds = predict_ref(graph, weights, verification_rows)
    if direct_preds != ref_preds:
        bad = [i for i, (a, b) in enumerate(zip(direct_preds, ref_preds)) if a != b]
        raise RuntimeError(f"pytorch int forward != kiln.ref on rows {bad[:8]}")
    print(f"  pytorch int forward == kiln.ref  on all {len(verification_rows)} samples")
    c_preds = None if args.no_gcc else predict_c(graph, weights, verification_rows)
    if c_preds is None:
        print("  emitted C99 check SKIPPED (gcc not available)")
    else:
        if c_preds != ref_preds:
            bad = [i for i, (a, b) in enumerate(zip(c_preds, ref_preds)) if a != b]
            raise RuntimeError(f"emitted C != kiln.ref on rows {bad[:8]}")
        print(f"  emitted C99 == kiln.ref          on all {len(verification_rows)} samples")

    # ---- stage 5: budget audit (arena < 2 KB, ROM < 40 KB) ----
    _banner("STAGE 5: BUDGET AUDIT + EXPORT")
    report = audit_graph(graph, weights)
    print(
        f"  arena {report.arena_ram_bytes} B (limit 2048)  "
        f"ROM {report.total_rom_bytes} B (limit 40960)  MACs {report.total_macs}"
    )
    if report.arena_ram_bytes >= 2048 or report.total_rom_bytes >= 40960:
        raise RuntimeError(
            f"budget exceeded: arena {report.arena_ram_bytes} B, ROM {report.total_rom_bytes} B"
        )

    artifact = {
        "format": "kiln-hazard-student",
        "version": 1,
        "seed": SEED,
        "dim": DEFAULT_DIM,
        "labels": DEFAULT_LABELS,
        "token_count": TOKEN_COUNT,
        "vocab_size": VOCAB_SIZE,
        "label_multiplier": multiplier,
        "label_shift": shift,
        "label_scale": deployed_scale,
        "calibration": calibration,
        "training": {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": WEIGHT_DECAY,
            "tau": [TAU_START, TAU_END],
            "qat": not args.no_qat,
            "qat_epochs": args.qat_epochs,
            "qat_lr": args.qat_lr,
            "float_best_val_acc": float_history["best_val_acc"],
            "qat_best_val_acc": qat_history.get("best_val_acc"),
        },
        "metrics": {
            "float_val": float_val,
            "float_test": float_test,
            "float_held_out": float_held,
            "ptq_val": pre_qat["val"],
            "ptq_test": pre_qat["test"],
            "ptq_held_out": pre_qat["held_out"],
            "int_val": int_val,
            "int_test": int_test,
            "int_held_out": int_held,
            "float_int_agreement_test": agreement_test,
            "saturation_rate": saturation,
        },
        "budget": {
            "arena_ram_bytes": report.arena_ram_bytes,
            "total_rom_bytes": report.total_rom_bytes,
            "total_macs": report.total_macs,
        },
        "w_emb": weights["w_emb"],
        "w_mm": weights["w_mm"],
    }
    if args.no_export:
        print("  export skipped (--no-export)")
    else:
        ARTIFACT_PATH.write_text(_format_artifact(artifact), encoding="utf-8")
        print(f"  wrote {ARTIFACT_PATH.name} ({ARTIFACT_PATH.stat().st_size} bytes)")

    result = {
        "artifact": ARTIFACT_PATH.name,
        "float_test": float_test,
        "int_test": int_test,
        "int_held_out": int_held,
        "arena": report.arena_ram_bytes,
        "rom": report.total_rom_bytes,
        "ref_agreement": 100.0,
        "c_agreement": None if c_preds is None else 100.0,
        "gcc": c_preds is not None,
    }
    print("RESULT " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
