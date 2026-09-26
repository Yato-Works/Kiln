"""Legacy no-augmentation QAT baseline (25 epochs, no validation-driven output).

Kept because it is the recipe the original Phase 1a numbers were measured with,
but it is **superseded by** ``python -m examples.kws.train_aug``, which trains the
same int8-domain contract with the shipped schedule, model selection, optional
augmentation and a machine-readable ``RESULT {json}`` line.

Note: like ``train_aug.py``, this script writes ``examples/kws/kws_qat_weights.pt``
(the shipped checkpoint), so running it replaces the Phase 1a artifact.

The dB-domain counterpart is ``train.py`` (paired with ``ptq_test.py``).
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from examples.kws.model import KilnKWSModel, quantize_logmel_to_int8
from kiln.frontend.pytorch import quantize_tensor, compute_multiplier_and_shift
from kiln.ir.kir import DType, Graph, Node, OpType, QuantParam, TensorDesc
from kiln.backend.c99.emit_c import emit_c

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
data = torch.load("examples/kws/kws_processed_data.pt", weights_only=True)
specs = data["specs"]     # (4890, 10, 49)
labels = data["labels"]

# Pre-quantize inputs to int8 tensor scaled to [-1.0, 1.0]
# That is, train on q_in directly!
q_inputs = quantize_logmel_to_int8(specs).float() / 127.0 # strictly in [-1.0, 1.0]

torch.manual_seed(1337)
n_total = len(labels)
perm = torch.randperm(n_total)
n_train = int(n_total * 0.70)
n_val = int(n_total * 0.15)

train_idx = perm[:n_train]
val_idx = perm[n_train:n_train + n_val]
test_idx = perm[n_train + n_val:]

train_loader = DataLoader(TensorDataset(q_inputs[train_idx], labels[train_idx]), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(q_inputs[val_idx], labels[val_idx]), batch_size=64, shuffle=False)
test_loader = DataLoader(TensorDataset(q_inputs[test_idx], labels[test_idx]), batch_size=64, shuffle=False)

model = KilnKWSModel(num_classes=12).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

best_val_acc = 0.0
best_state = None

print("Training model directly on normalized int8 inputs [-1.0, 1.0]...")
for epoch in range(1, 26):
    model.train()
    correct = 0
    total = 0
    for x_b, y_b in train_loader:
        x_b = x_b.to(device)
        y_b = y_b.to(device)
        optimizer.zero_grad()
        out = model(x_b).squeeze(-1)
        loss = criterion(out, y_b)
        loss.backward()
        optimizer.step()
        preds = torch.argmax(out, dim=-1)
        correct += (preds == y_b).sum().item()
        total += len(y_b)
    scheduler.step()

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for x_v, y_v in val_loader:
            x_v = x_v.to(device)
            y_v = y_v.to(device)
            out_v = model(x_v).squeeze(-1)
            preds_v = torch.argmax(out_v, dim=-1)
            val_correct += (preds_v == y_v).sum().item()
            val_total += len(y_v)
    val_acc = val_correct / val_total * 100.0
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if epoch % 5 == 0 or epoch == 25:
        print(f"Epoch {epoch:2d}/25 | Train Acc: {correct/total*100:5.2f}% | Val Acc: {val_acc:5.2f}% | Best: {best_val_acc:5.2f}%")

model.load_state_dict(best_state)
model.eval().cpu()

# Evaluate PyTorch on test set
test_correct = 0
test_total = 0
with torch.no_grad():
    for x_t, y_t in test_loader:
        out_t = model(x_t).squeeze(-1)
        preds_t = torch.argmax(out_t, dim=-1)
        test_correct += (preds_t == y_t).sum().item()
        test_total += len(y_t)
print(f"\nPyTorch Test Accuracy on normalized int8 inputs: {test_correct/test_total*100:.2f}% ({test_correct}/{test_total})")

torch.save(best_state, "examples/kws/kws_qat_weights.pt")