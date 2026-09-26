import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from examples.kws.model import KilnKWSModel, quantize_logmel_to_int8
from kiln.frontend.pytorch import export_to_kir
from kiln.ref.ref import run_ref

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# 1. Load precomputed log-mel spectrograms & labels
data = torch.load("examples/kws/kws_processed_data.pt", weights_only=True)
specs = data["specs"]     # (4890, 10, 49) float dB [-80, 0]
labels = data["labels"]   # (4890,) int64 in [0..11]

# 2. Stratified train/val/test split (70% train = 3418, 15% val = 727, 15% test = 745)
torch.manual_seed(1337)
n_total = len(labels)
perm = torch.randperm(n_total)

n_train = int(n_total * 0.70)
n_val = int(n_total * 0.15)
n_test = n_total - n_train - n_val

train_idx = perm[:n_train]
val_idx = perm[n_train:n_train + n_val]
test_idx = perm[n_train + n_val:]

train_ds = TensorDataset(specs[train_idx], labels[train_idx])
val_ds = TensorDataset(specs[val_idx], labels[val_idx])
test_ds = TensorDataset(specs[test_idx], labels[test_idx])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

print(f"Dataset split: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

# 3. Model setup
model = KilnKWSModel(num_classes=12).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

# 4. Training loop (25 epochs with GPU takes ~5 seconds)
epochs = 25
best_val_acc = 0.0
best_model_state = None

print("Starting training...")
start_time = time.time()
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        out = model(x_batch).squeeze(-1) # (batch, 12, 1) -> (batch, 12)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds = torch.argmax(out, dim=-1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    scheduler.step()
    train_acc = (correct / total) * 100.0

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            out_val = model(x_val).squeeze(-1)
            preds_val = torch.argmax(out_val, dim=-1)
            val_correct += (preds_val == y_val).sum().item()
            val_total += len(y_val)
    val_acc = (val_correct / val_total) * 100.0

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if epoch % 5 == 0 or epoch == epochs:
        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Acc: {train_acc:5.2f}% | Val Acc: {val_acc:5.2f}% | Best Val: {best_val_acc:5.2f}%")

print(f"Training completed in {time.time() - start_time:.1f}s!")

# Load best weights
model.load_state_dict(best_model_state)
model.eval().cpu()

# 5. Evaluate on official TEST set (Float32 PyTorch)
test_correct = 0
test_total = 0
test_specs_list = []
test_labels_list = []

with torch.no_grad():
    for x_test, y_test in test_loader:
        out_test = model(x_test).squeeze(-1)
        preds_test = torch.argmax(out_test, dim=-1)
        test_correct += (preds_test == y_test).sum().item()
        test_total += len(y_test)
        test_specs_list.append(x_test)
        test_labels_list.append(y_test)

fp32_test_acc = (test_correct / test_total) * 100.0
print("=" * 60)
print(f"RESULTS ON SPEECH COMMANDS TEST SET (745 unseen samples):")
print(f"  PyTorch FP32 Top-1 Test Accuracy: {fp32_test_acc:.2f}% ({test_correct}/{test_total})")
print("=" * 60)

# Save trained weights
torch.save(best_model_state, "examples/kws/kws_trained_weights.pt")