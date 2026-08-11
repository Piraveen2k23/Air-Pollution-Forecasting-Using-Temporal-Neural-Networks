"""
train_aqi_classifier.py  -  Extended Task 4: AQI Classification

Maps PM2.5 ug/m3 to 4 AQI classes (Chinese standard) and trains an LSTM
classifier using cross-entropy loss.

Chinese AQI PM2.5 breakpoints (4-class for task spec):
  Class 0 - Good:        0  -  50 ug/m3
  Class 1 - Moderate:   50  - 100 ug/m3
  Class 2 - Unhealthy: 100  - 150 ug/m3
  Class 3 - Hazardous: >150 ug/m3

Architecture:
  Same LSTM backbone (3 layers, 256 hidden, dropout=0.2)
  Final linear layer: hidden_size -> 4  (4-class softmax)
  Loss: CrossEntropyLoss (handles class imbalance with class_weight)

Outputs:
  best_aqi_classifier.pt   -- classifier weights
  aqi_results.pkl          -- accuracy, f1, confusion_matrix, class_names, history
"""

import numpy as np
import pickle
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)

HIDDEN_SIZE   = 256
NUM_LAYERS    = 3
DROPOUT       = 0.2
BATCH_SIZE    = 512
LEARNING_RATE = 1e-3
NUM_EPOCHS    = 30
PATIENCE      = 5
NUM_CLASSES   = 4

CLASS_NAMES   = ["Good\n(0-50)", "Moderate\n(50-100)",
                 "Unhealthy\n(100-150)", "Hazardous\n(>150)"]

AQI_BREAKPOINTS = [0, 50, 100, 150, float("inf")]

def pm25_to_aqi_class(pm25_array):
    """Map PM2.5 ug/m3 values to integer class labels 0-3."""
    labels = np.zeros(len(pm25_array), dtype=np.int64)
    for i, (lo, hi) in enumerate(zip(AQI_BREAKPOINTS[:-1], AQI_BREAKPOINTS[1:])):
        mask = (pm25_array >= lo) & (pm25_array < hi)
        labels[mask] = i
    return labels

# ─── Load data ────────────────────────────────────────────────────────────────
print("=" * 65)
print(" Loading data ...")
print("=" * 65)

data = np.load("data/processed/windows.npz", allow_pickle=True)
X_train   = data["X_train"]
y_train_r = data["y_train_raw"]
X_val     = data["X_val"]
y_val_r   = data["y_val_raw"]

with open("data/processed/fitted_preprocessing.pkl", "rb") as f:
    fitted = pickle.load(f)

# Convert regression targets to class labels
y_train_cls = pm25_to_aqi_class(y_train_r)
y_val_cls   = pm25_to_aqi_class(y_val_r)

# Class distribution
print("\n  Training class distribution:")
for i, name in enumerate(CLASS_NAMES):
    n = (y_train_cls == i).sum()
    print(f"    Class {i} ({name.replace(chr(10),' ')}): {n:,}  ({100*n/len(y_train_cls):.1f}%)")

print("\n  Validation class distribution:")
for i, name in enumerate(CLASS_NAMES):
    n = (y_val_cls == i).sum()
    print(f"    Class {i} ({name.replace(chr(10),' ')}): {n:,}  ({100*n/len(y_val_cls):.1f}%)")

# ─── Class weights to handle imbalance ────────────────────────────────────────
class_counts  = np.bincount(y_train_cls, minlength=NUM_CLASSES).astype(float)
class_weights = 1.0 / (class_counts / class_counts.sum())
class_weights = class_weights / class_weights.sum() * NUM_CLASSES
print(f"\n  Class weights (inverse frequency): {class_weights.round(3)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

# ─── DataLoaders ──────────────────────────────────────────────────────────────
X_train_t   = torch.tensor(X_train,     dtype=torch.float32)
y_train_t   = torch.tensor(y_train_cls, dtype=torch.long)
X_val_t     = torch.tensor(X_val,       dtype=torch.float32)
y_val_t     = torch.tensor(y_val_cls,   dtype=torch.long)
weights_t   = torch.tensor(class_weights, dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

INPUT_SIZE = X_train.shape[2]

# ─── Model ────────────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    """
    LSTM backbone with a 4-class classification head.
    Same LSTM layers as regression model; only the final linear differs.
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _  = self.lstm(x)
        last    = self.dropout(out[:, -1, :])
        return self.fc(last)   # (batch, num_classes)  -- raw logits

model = LSTMClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, NUM_CLASSES).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  LSTMClassifier parameters: {total_params:,}")

loss_fn   = nn.CrossEntropyLoss(weight=weights_t)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3)  # mode=max: we want accuracy UP

# ─── Eval helper ──────────────────────────────────────────────────────────────
def evaluate_acc(model, X_tensor, y_true_np):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), 1024):
            logits = model(X_tensor[i:i+1024].to(device))
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
    y_pred = np.concatenate(all_preds)
    acc    = float(accuracy_score(y_true_np, y_pred))
    return acc, y_pred

# ─── Training ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f" Training AQI Classifier for up to {NUM_EPOCHS} epochs ...")
print("=" * 65)
print(f"  {'Epoch':>5}  {'Train Loss':>12}  {'Val Acc%':>10}  {'Time':>8}  Status")
print(f"  {'-'*5}  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*10}")

best_acc          = 0.0
epochs_no_improve = 0
history           = {"train_loss": [], "val_acc": []}
best_state        = None

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss   = loss_fn(logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * len(X_batch)

    avg_loss = running_loss / len(X_train_t)
    val_acc, _ = evaluate_acc(model, X_val_t, y_val_cls)
    scheduler.step(val_acc)
    elapsed = time.time() - t0
    history["train_loss"].append(avg_loss)
    history["val_acc"].append(val_acc * 100)

    if val_acc > best_acc:
        best_acc   = val_acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        torch.save(model.state_dict(), "models/best_aqi_classifier.pt")
        epochs_no_improve = 0
        status = "<-- best"
    else:
        epochs_no_improve += 1
        status = f"no improve ({epochs_no_improve}/{PATIENCE})"

    print(f"  {epoch:>5}  {avg_loss:>12.6f}  {val_acc*100:>10.2f}  {elapsed:>7.1f}s  {status}")

    if epochs_no_improve >= PATIENCE:
        print(f"\n  Early stopping triggered after epoch {epoch}.")
        break

# ─── Final evaluation ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" Final Evaluation with best weights ...")
print("=" * 65)

model.load_state_dict(torch.load("models/best_aqi_classifier.pt", weights_only=True))
best_acc_final, y_pred_val = evaluate_acc(model, X_val_t, y_val_cls)

print(f"\n  Overall Accuracy: {best_acc_final*100:.2f}%")
print("\n  Classification Report:")
clean_names = [n.replace("\n", " ") for n in CLASS_NAMES]
print(classification_report(y_val_cls, y_pred_val, target_names=clean_names))

cm = confusion_matrix(y_val_cls, y_pred_val)
print("  Confusion Matrix:")
print(cm)

f1_per_class = f1_score(y_val_cls, y_pred_val, average=None)
f1_macro     = f1_score(y_val_cls, y_pred_val, average="macro")
f1_weighted  = f1_score(y_val_cls, y_pred_val, average="weighted")

print(f"\n  F1 per class: {dict(zip(clean_names, f1_per_class.round(4)))}")
print(f"  Macro F1: {f1_macro:.4f}   Weighted F1: {f1_weighted:.4f}")

# ─── Save results ─────────────────────────────────────────────────────────────
results = {
    "accuracy":        best_acc_final,
    "f1_per_class":    f1_per_class.tolist(),
    "f1_macro":        float(f1_macro),
    "f1_weighted":     float(f1_weighted),
    "confusion_matrix": cm.tolist(),
    "class_names":     CLASS_NAMES,
    "y_val_true":      y_val_cls.tolist(),
    "y_val_pred":      y_pred_val.tolist(),
    "history":         history,
    "aqi_breakpoints": AQI_BREAKPOINTS[:-1],
}
with open("results/aqi_results.pkl", "wb") as f:
    pickle.dump(results, f)

print(f"\n  Saved: best_aqi_classifier.pt")
print(f"         aqi_results.pkl")
print("\n" + "=" * 65)
print(f"  DONE!  Best Val Accuracy = {best_acc_final*100:.2f}%  |  Macro F1 = {f1_macro:.4f}")
print("=" * 65)
