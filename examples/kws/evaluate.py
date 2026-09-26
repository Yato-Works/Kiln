"""Phase 1a benchmark: FP32 PyTorch vs emitted C99 int8 on the frozen test split.

Run from the repository root with ``python -m examples.kws.evaluate``.

The numbers printed here are the Phase 1a certificate in ``ROADMAP.md``:

* int8 top-1 accuracy of the *emitted C* (not of PyTorch),
* FP32-vs-C99 agreement and the quantization accuracy delta,
* the static budget audit (arena RAM, weight+LUT ROM, MACs per inference).

The split comes from ``kws_graph.split_indices`` (seed 1337), so the numbers are
comparable with ``train_aug.py``, ``ptq_test.py``, ``qat_eval.py`` and
``tests/test_kws.py``.
"""

from __future__ import annotations

from examples.kws.benchmark import (
    int8_samples,
    load_splits,
    predict_c,
    predict_pytorch,
    top1_accuracy,
)
from examples.kws.kws_graph import calibrate_kws_graph
from examples.kws.model import model
from kiln.budget.report import audit_graph

specs, labels, (_, val_idx, test_idx) = load_splits()

test_specs, test_labels = specs[test_idx], labels[test_idx]
val_specs, val_labels = specs[val_idx], labels[val_idx]

# Lower the trained PyTorch model into Kiln IR (int8 weights + activation scales)
kiln_graph = calibrate_kws_graph(model, val_specs)
graph, weights = kiln_graph.graph, kiln_graph.weights
budget = audit_graph(graph, weights)

c_test_preds = predict_c(graph, weights, int8_samples(test_specs))
fp32_test_preds = predict_pytorch(model, test_specs)

c_val_preds = predict_c(graph, weights, int8_samples(val_specs))
fp32_val_preds = predict_pytorch(model, val_specs)

c99_int8_acc = top1_accuracy(c_test_preds, test_labels.tolist())
fp32_acc = top1_accuracy(fp32_test_preds, test_labels.tolist())
agreement = top1_accuracy(c_test_preds, fp32_test_preds)
c99_val_acc = top1_accuracy(c_val_preds, val_labels.tolist())
fp32_val_acc = top1_accuracy(fp32_val_preds, val_labels.tolist())

hits = sum(1 for p, y in zip(c_test_preds, test_labels.tolist()) if p == y)

print("=" * 65)
print("      PHASE 1a BENCHMARK RESULTS: SPEECH COMMANDS v2 (12-CLASS)")
print("=" * 65)
print(f"  Target Dataset               : Speech Commands v2 (12 Classes)")
print(f"  Test Samples Evaluated       : {len(test_labels)} held-out samples")
print(f"  PyTorch FP32 Top-1 Accuracy  : {fp32_acc:.2f}%")
print(f"  Kiln C99 int8 Top-1 Accuracy : {c99_int8_acc:.2f}% ({hits}/{len(test_labels)})")
print(f"  Quantization Accuracy Delta  : {c99_int8_acc - fp32_acc:+.2f}%")
print(f"  FP32 vs C99 Prediction Match : {agreement:.2f}% agreement")
print(f"  Validation Split (selection) : FP32 {fp32_val_acc:.2f}% / C99 {c99_val_acc:.2f}%")
print(f"  Int8 Weights ROM             : {budget.weights_rom_bytes} B ({budget.lut_rom_bytes} B of LUT tables)")
print(f"  Arena (.bss) RAM             : {budget.arena_ram_bytes} B (target < 2048 B)")
print(f"  Constant ROM (weights+LUTs)  : {budget.total_rom_bytes} B (target < 40960 B)")
print(f"  MACs per inference           : {budget.total_macs:,}")
print("=" * 65)

