"""Phase 1a post-training quantization (PTQ) baseline.

``kws_trained_weights.pt`` comes from ``train.py``, which fits the model in the
**raw dB domain**.  Kiln's int8 contract is zero-point free: the deployed input
is ``q / 127`` with ``q = round((db + 40) / 40 * 127)``.  A dB-domain model
therefore pays a constant input offset the graph cannot represent, which is
exactly why the shipped recipe (``train_aug.py``) trains on ``q / 127``.

This script quantizes the dB-domain checkpoint, lowers it through the shared
``kws_graph`` builder and reports what the *emitted C* scores on the frozen test
split: the honest "what if we skipped QAT" number for the Phase 1a table.

Run from the repository root with ``python -m examples.kws.ptq_test``.
"""

from __future__ import annotations

import os

from examples.kws.benchmark import int8_samples, load_splits, predict_c, top1_accuracy
from examples.kws.kws_graph import calibrate_kws_graph
from examples.kws.model import load_kws_model

WEIGHTS = os.path.join(os.path.dirname(__file__), "kws_trained_weights.pt")

specs, labels, (_, val_idx, test_idx) = load_splits()

model = load_kws_model(WEIGHTS)
lowered = calibrate_kws_graph(model, specs[val_idx])

preds = predict_c(lowered.graph, lowered.weights, int8_samples(specs[test_idx]))
accuracy = top1_accuracy(preds, labels[test_idx].tolist())

print(f"PTQ baseline (dB-domain FP32 training -> quantized C): {accuracy:.2f}%")
print(f"  checkpoint : {WEIGHTS}")
print(f"  evaluated  : {len(test_idx)} held-out samples through the emitted C99 graph")
