"""Phase 1a shipped-checkpoint accuracy, measured through the emitted C99 graph.

``kws_qat_weights.pt`` is produced by ``train_aug.py`` (int8-domain training, the
recipe the deployed graph actually executes).  This script reports both the
int8 C accuracy and the FP32 accuracy of the same checkpoint on the frozen test
split, so the quantization delta of the *emitted artifact* is visible without
the full``evaluate.py`` parity table.

Run from the repository root with ``python -m examples.kws.qat_eval``.
"""

from __future__ import annotations

import os

from examples.kws.benchmark import (
    int8_samples,
    load_splits,
    predict_c,
    predict_pytorch,
    top1_accuracy,
)
from examples.kws.kws_graph import calibrate_kws_graph
from examples.kws.model import load_kws_model

WEIGHTS = os.path.join(os.path.dirname(__file__), "kws_qat_weights.pt")

specs, labels, (_, val_idx, test_idx) = load_splits()

model = load_kws_model(WEIGHTS)
lowered = calibrate_kws_graph(model, specs[val_idx])

targets = labels[test_idx].tolist()
c_accuracy = top1_accuracy(predict_c(lowered.graph, lowered.weights, int8_samples(specs[test_idx])), targets)
fp32_accuracy = top1_accuracy(predict_pytorch(model, specs[test_idx]), targets)

print(f"QAT (shipped) checkpoint: C99 int8 {c_accuracy:.2f}% | FP32 {fp32_accuracy:.2f}%")
print(f"  quantization delta : {c_accuracy - fp32_accuracy:+.2f}%")
print(f"  checkpoint         : {WEIGHTS}")
print(f"  evaluated          : {len(test_idx)} held-out samples")
