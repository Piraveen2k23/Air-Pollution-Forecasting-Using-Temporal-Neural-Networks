"""
feature_importance.py  -  Extended Task 3: Investigate the value of auxiliary
meteorological features for PM2.5 forecasting.

Approach:
  1. Train Pollution-Only LSTM: features = PM2.5,PM10,SO2,NO2,CO,O3 + calendar + station
  2. Compare to Full-Feature LSTM (already trained, loaded from lstm_results.pkl)
  3. Quantify % RMSE reduction from adding weather variables
  4. Fit Ridge regression on both feature sets; extract & rank coefficients

Outputs:
  pollution_only_results.pkl       -- val_rmse, history
  feature_importance_results.pkl   -- ridge_coefs, full_rmse, poll_only_rmse
"""

import numpy as np
import pickle
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge

HIDDEN_SIZE   = 256
NUM_LAYERS    = 3
DROPOUT       = 0.2
BATCH_SIZE    = 512
LEARNING_RATE = 1e-3
NUM_EPOCHS    = 30
PATIENCE      = 5

# ─── Load full windows ────────────────────────────────────────────────────────
print("=" * 65)
print(" Loading data ...")
print("=" * 65)

data = np.load("windows.npz", allow_pickle=True)
X_train_full = data["X_train"]      # (N, 24, 29)
y_train_norm = data["y_train_norm"]
y_train_raw  = data["y_train_raw"]
X_val_full   = data["X_val"]
y_val_norm   = data["y_val_norm"]
y_val_raw    = data["y_val_raw"]

with open("fitted_preprocessing.pkl", "rb") as f:
    fitted = pickle.load(f)
mean_pm25 = float(fitted["means"]["PM2.5"])
std_pm25  = float(fitted["stds"]["PM2.5"])

with open("feature_cols.pkl", "rb") as f:
    meta = pickle.load(f)
feature_cols = meta["feature_cols"]
print(f"  Full feature set ({len(feature_cols)}): {feature_cols}")

# ─── Build pollution-only feature index mask ──────────────────────────────────
POLLUTANTS_NORM = ["PM2.5_norm", "PM10_norm", "SO2_norm", "NO2_norm", "CO_norm", "O3_norm"]
CALENDAR        = ["hour_sin", "hour_cos", "month_sin", "month_cos"]
STATION_COLS    = [c for c in feature_cols if c.startswith("st_")]

poll_only_cols = POLLUTANTS_NORM + CALENDAR + STATION_COLS
poll_only_idx  = [feature_cols.index(c) for c in poll_only_cols]

print(f"\n  Pollution-only feature set ({len(poll_only_cols)}): {poll_only_cols}")

X_train_poll = X_train_full[:, :, poll_only_idx]   # (N, 24, n_poll)
X_val_poll   = X_val_full[:, :, poll_only_idx]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

# ─── LSTM model ───────────────────────────────────────────────────────────────
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :])).squeeze(-1)

def train_lstm(X_tr, yn_tr, yr_tr, X_v, yr_v, input_size, label=""):
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(yn_tr, dtype=torch.float32)
    Xv = torch.tensor(X_v,  dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=0)
    model = LSTMForecaster(input_size, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    sch   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    lf    = nn.MSELoss()
    best_rmse = float("inf")
    history   = {"train_loss": [], "val_rmse": []}
    no_imp    = 0
    best_state= None
    print(f"  {'Epoch':>5}  {'Train Loss':>12}  {'Val RMSE':>10}  {'Time':>8}  Status")
    print(f"  {'-'*5}  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*10}")
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        model.train()
        rl = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = lf(model(Xb), yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            rl += loss.item() * len(Xb)
        avg_loss = rl / len(Xt)
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(Xv), 1024):
                preds.append(model(Xv[i:i+1024].to(device)).cpu().numpy())
        pred_raw = np.concatenate(preds) * std_pm25 + mean_pm25
        rmse = float(np.sqrt(np.mean((pred_raw - yr_v) ** 2)))
        sch.step(rmse)
        elapsed = time.time() - t0
        history["train_loss"].append(avg_loss)
        history["val_rmse"].append(rmse)
        if rmse < best_rmse:
            best_rmse = rmse; best_state = {k: v.clone() for k, v in model.state_dict().items()}; no_imp = 0; status = "<-- best"
        else:
            no_imp += 1; status = f"no improve ({no_imp}/{PATIENCE})"
        print(f"  {epoch:>5}  {avg_loss:>12.6f}  {rmse:>10.3f}  {elapsed:>7.1f}s  {status}")
        if no_imp >= PATIENCE:
            print(f"\n  Early stopping after epoch {epoch}.")
            break
    model.load_state_dict(best_state)
    return model, best_rmse, history, pred_raw

# ─── Train pollution-only LSTM ────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" Training POLLUTION-ONLY LSTM ...")
print("=" * 65)
model_poll, poll_val_rmse, poll_hist, poll_pred = train_lstm(
    X_train_poll, y_train_norm, y_train_raw,
    X_val_poll, y_val_raw,
    input_size=len(poll_only_cols), label="Pollution-Only")

with open("pollution_only_results.pkl", "wb") as f:
    pickle.dump({"val_rmse": poll_val_rmse, "history": poll_hist,
                 "val_pred_raw": poll_pred, "y_val_raw": y_val_raw,
                 "feature_cols": poll_only_cols}, f)
print(f"\n  Pollution-Only LSTM Val RMSE: {poll_val_rmse:.3f}")

# ─── Load full LSTM results for comparison ────────────────────────────────────
try:
    with open("lstm_results.pkl", "rb") as f:
        lstm_res = pickle.load(f)
    full_val_rmse = lstm_res["val_rmse"]
except FileNotFoundError:
    full_val_rmse = None

print("\n" + "=" * 65)
print(" FEATURE IMPORTANCE: Full vs Pollution-Only comparison")
print("=" * 65)
print(f"  Pollution-Only LSTM  Val RMSE: {poll_val_rmse:.3f}")
if full_val_rmse:
    pct = (poll_val_rmse - full_val_rmse) / poll_val_rmse * 100
    print(f"  Full-Feature   LSTM  Val RMSE: {full_val_rmse:.3f}")
    print(f"  Weather features give: {pct:.1f}% RMSE reduction")

# ─── Ridge regression feature importance ─────────────────────────────────────
print("\n" + "=" * 65)
print(" Fitting Ridge regression for feature importance ...")
print("=" * 65)

# Use last timestep features as the Ridge input (same as baselines.py)
X_tr_last_full = X_train_full[:, -1, :]   # (N, 29)
X_v_last_full  = X_val_full[:, -1, :]

ridge_full = Ridge(alpha=1.0)
ridge_full.fit(X_tr_last_full, y_train_norm)
ridge_pred = ridge_full.predict(X_v_last_full) * std_pm25 + mean_pm25
ridge_rmse = float(np.sqrt(np.mean((ridge_pred - y_val_raw) ** 2)))
print(f"  Ridge (full features) Val RMSE: {ridge_rmse:.3f}")

# Extract and sort coefficients
coefs = ridge_full.coef_
sorted_idx = np.argsort(np.abs(coefs))[::-1]
top_features = [(feature_cols[i], float(coefs[i])) for i in sorted_idx]

print("\n  Top-15 feature importances (by |coef|):")
print(f"  {'Feature':<25} {'Coef':>10}")
for name, c in top_features[:15]:
    print(f"  {name:<25} {c:>10.4f}")

# ─── Save results ─────────────────────────────────────────────────────────────
fi_results = {
    "poll_only_val_rmse": poll_val_rmse,
    "full_val_rmse":      full_val_rmse,
    "ridge_rmse":         ridge_rmse,
    "ridge_coefs":        list(zip(feature_cols, coefs.tolist())),
    "top_features":       top_features,
    "weather_gain_pct":   (poll_val_rmse - full_val_rmse) / poll_val_rmse * 100 if full_val_rmse else None,
}
with open("feature_importance_results.pkl", "wb") as f:
    pickle.dump(fi_results, f)

print("\n  Saved: pollution_only_results.pkl")
print("         feature_importance_results.pkl")
