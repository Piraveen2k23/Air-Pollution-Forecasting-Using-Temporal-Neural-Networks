"""
train_gru.py  -  Extended Task 2A: Train a GRU forecaster for 1-hour-ahead PM2.5 prediction.

Architecture: identical to the Stacked LSTM (3 layers, 256 hidden, dropout=0.2)
for a fair apples-to-apples comparison.

GRU vs LSTM:
  - GRU uses two gates (reset, update) vs LSTM's three (input, forget, output)
  - GRU has no separate cell state -> fewer parameters, often trains faster
  - In practice, performance is usually within a few percent of LSTM

Outputs:
  best_gru.pt       -- model weights at best validation RMSE
  gru_results.pkl   -- dict with val_rmse, history, val_pred_raw, y_val_raw
"""

import numpy as np
import pickle
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HIDDEN_SIZE   = 256
NUM_LAYERS    = 3
DROPOUT       = 0.2
BATCH_SIZE    = 512
LEARNING_RATE = 1e-3
NUM_EPOCHS    = 30
PATIENCE      = 5

print("=" * 65)
print(" Loading data from windows.npz ...")
print("=" * 65)

data = np.load("windows.npz", allow_pickle=True)
X_train      = data["X_train"]
y_train_norm = data["y_train_norm"]
y_train_raw  = data["y_train_raw"]
X_val        = data["X_val"]
y_val_norm   = data["y_val_norm"]
y_val_raw    = data["y_val_raw"]

with open("fitted_preprocessing.pkl", "rb") as f:
    fitted = pickle.load(f)
mean_pm25 = float(fitted["means"]["PM2.5"])
std_pm25  = float(fitted["stds"]["PM2.5"])

INPUT_SIZE = X_train.shape[2]
print(f"  X_train : {X_train.shape}")
print(f"  X_val   : {X_val.shape}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train_norm, dtype=torch.float32)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val_norm, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

class GRUForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        last   = self.dropout(out[:, -1, :])
        return self.fc(last).squeeze(-1)

model = GRUForecaster(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  GRU parameters: {total_params:,}")

loss_fn   = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

def evaluate_rmse(model, X_tensor, y_raw, batch_size=1024):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i + batch_size].to(device)
            pred  = model(batch).cpu().numpy()
            all_preds.append(pred)
    pred_norm = np.concatenate(all_preds)
    pred_raw  = pred_norm * std_pm25 + mean_pm25
    return float(np.sqrt(np.mean((pred_raw - y_raw) ** 2))), pred_raw

print("\n" + "=" * 65)
print(f" Training GRU for up to {NUM_EPOCHS} epochs (patience={PATIENCE}) ...")
print("=" * 65)
print(f"  {'Epoch':>5}  {'Train Loss':>12}  {'Val RMSE':>10}  {'Time':>8}  Status")
print(f"  {'-'*5}  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*10}")

best_val_rmse     = float("inf")
epochs_no_improve = 0
history           = {"train_loss": [], "val_rmse": []}

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * len(X_batch)
    avg_train_loss = running_loss / len(X_train_t)
    val_rmse, _   = evaluate_rmse(model, X_val_t, y_val_raw)
    scheduler.step(val_rmse)
    elapsed = time.time() - t0
    history["train_loss"].append(avg_train_loss)
    history["val_rmse"].append(val_rmse)
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save(model.state_dict(), "best_gru.pt")
        epochs_no_improve = 0
        status = "<-- best"
    else:
        epochs_no_improve += 1
        status = f"no improve ({epochs_no_improve}/{PATIENCE})"
    print(f"  {epoch:>5}  {avg_train_loss:>12.6f}  {val_rmse:>10.3f}  {elapsed:>7.1f}s  {status}")
    if epochs_no_improve >= PATIENCE:
        print(f"\n  Early stopping after epoch {epoch}.")
        break

model.load_state_dict(torch.load("best_gru.pt", weights_only=True))
val_rmse, val_pred_raw = evaluate_rmse(model, X_val_t, y_val_raw)
val_mae = float(np.mean(np.abs(val_pred_raw - y_val_raw)))

try:
    with open("lstm_results.pkl", "rb") as f:
        lstm_res = pickle.load(f)
    lstm_val_rmse = lstm_res["val_rmse"]
    print(f"\n  LSTM Val RMSE: {lstm_val_rmse:.3f}   GRU Val RMSE: {val_rmse:.3f}")
    diff = lstm_val_rmse - val_rmse
    print(f"  GRU vs LSTM: {'+' if diff>0 else ''}{diff:.3f} ({'GRU better' if diff>0 else 'LSTM better'})")
except FileNotFoundError:
    print(f"  GRU Val RMSE: {val_rmse:.3f}  Val MAE: {val_mae:.3f}")

results = {
    "val_rmse": val_rmse, "val_mae": val_mae,
    "val_pred_raw": val_pred_raw, "y_val_raw": y_val_raw,
    "history": history, "best_epoch": int(np.argmin(history["val_rmse"])) + 1,
}
with open("gru_results.pkl", "wb") as f:
    pickle.dump(results, f)

print(f"\n  Saved: best_gru.pt  gru_results.pkl")
print(f"  DONE! Best GRU Val RMSE = {best_val_rmse:.3f} ug/m3")
