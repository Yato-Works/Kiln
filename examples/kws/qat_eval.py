"""Phase 1a checkpoint accuracy, measured through the emitted C99 graph.

``kws_qat_weights.pt`` is produced by ``train_aug.py`` (int8-domain training, the
recipe the deployed graph actually executes).  This script reports both the
int8 C accuracy and the FP32 accuracy of the same checkpoint on the frozen test
split, so the quantization delta of the *emitted artifact* is visible without
the full``evaluate.py`` parity table.

Run from the repository root with ``python -m examples.kws.qat_eval`` (the
certificate layout prints 91.01% for the emitted C and 90.74% for FP32).  Any
architecture preset can be audited the same way, which is how the alternative
layouts in ``ROADMAP.md`` section 3 stay reproducible:

    python -m examples.kws.qat_eval --architecture previous \
        --weights examples/kws/kws_qat_weights_27_55.pt

reprints the previous 27/55 layout's certificate (89.10% int8 C / 88.69% FP32,
1,972 B of arena) from its kept checkpoint.
"""

from __future__ import annotations

import argparse
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
from examples.kws.train_aug import ARCHITECTURES

WEIGHTS = os.path.join(os.path.dirname(__file__), "kws_qat_weights.pt")

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--architecture", choices=sorted(ARCHITECTURES), default="shipped")
parser.add_argument("--weights", default=WEIGHTS, help="int8 checkpoint to audit")
arguments = parser.parse_args()

specs, labels, (_, val_idx, test_idx) = load_splits()

model = load_kws_model(arguments.weights, architecture=ARCHITECTURES[arguments.architecture])
lowered = calibrate_kws_graph(model, specs[val_idx])

targets = labels[test_idx].tolist()
c_accuracy = top1_accuracy(predict_c(lowered.graph, lowered.weights, int8_samples(specs[test_idx])), targets)
fp32_accuracy = top1_accuracy(predict_pytorch(model, specs[test_idx]), targets)

print(f"QAT ({arguments.architecture}) checkpoint: C99 int8 {c_accuracy:.2f}% | FP32 {fp32_accuracy:.2f}%")
print(f"  quantization delta : {c_accuracy - fp32_accuracy:+.2f}%")
print(f"  checkpoint         : {arguments.weights}")
print(f"  params             : {sum(p.numel() for p in model.parameters()):,}")
print(f"  evaluated          : {len(test_idx)} held-out samples")
