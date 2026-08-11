"""
window_ablation.py  -  Extended Task 2B: Investigate the effect of look-back
window size on PM2.5 forecasting accuracy.

Trains a lightweight LSTM (1 layer, 128 hidden, max 20 epochs) for each
window size to isolate the effect of the look-back horizon on validation RMSE.

Window sizes tested: 6, 12, 18, 24, 36, 48 hours

Outputs:
  window_ablation_results.pkl  --  {window_size: {"val_rmse": float, "history": ...}}
"""

import numpy as np
import pickle
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# Ablation-only model: smaller so 6 runs finish in reasonable time
HIDDEN_SIZE   = 128
NUM_LAYERS    = 1
DROPOUT       = 0.1
BATCH_SIZE    = 512
LEARNING_RATE = 1e-3
NUM_EPOCHS    = 20
PATIENCE      = 4

WINDOW_SIZES = [6, 12, 18, 24, 36, 48]

# ─── Load raw cleaned data (we rebuild windows per size) ──────────────────────
print("=" * 65)
print(" Loading cleaned data for window ablation ...")
print("=" * 65)

train_df = pd.read_pickle("train_clean.pkl")
val_df   = pd.read_pickle("val_clean.pkl")

with open("fitted_preprocessing.pkl", "rb") as f:
    fitted = pickle.load(f)
mean_pm25 = float(fitted["means"]["PM2.5"])
std_pm25  = float(fitted["stds"]["PM2.5"])

with open("feature_cols.pkl", "rb") as f:
    meta = pickle.load(f)
feature_cols  = meta["feature_cols"]
station_list  = meta["station_list"]

# Add cyclical calendar features (already done in train_clean if windowing.py was run)
for df in [train_df, val_df]:
    if "hour_sin" not in df.columns:
        df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
        df["month_sin"]  = np.sin(2 * np.pi * (df["month"] - 1) / 12)
        df["month_cos"]  = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    # Station one-hots
    for s in station_list:
        col = f"st_{s}"
        if col not in df.columns:
            df[col] = (df["station"] == s).astype(np.float32)

print(f"  Feature columns: {len(feature_cols)}")
print(f"  Stations: {station_list}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}\n")

# ─── Helpers ──────────────────────────────────────────────────────────────────
def build_windows(df, window):
    Xs, y_norms, y_raws = [], [], []
    for station in station_list:
        sub = df[df["station"] == station].sort_values("datetime")
        arr = sub[feature_cols].values.astype(np.float32)
        yn  = sub["PM2.5_norm"].values.astype(np.float32)
        yr  = sub["PM2.5"].values.astype(np.float32)
        T   = arr.shape[0]
        if T <= window:
            continue
        wins = np.lib.stride_tricks.sliding_window_view(arr, window_shape=window, axis=0)
        wins = wins.transpose(0, 2, 1)  # (T-window+1, window, F)
        n = wins.shape[0] - 1
        Xs.append(wins[:n])
        y_norms.append(yn[window:window + n])
        y_raws.append(yr[window:window + n])
    return (np.concatenate(Xs), np.concatenate(y_norms), np.concatenate(y_raws))


class SmallLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers=1,
                               batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :])).squeeze(-1)


def train_and_eval(X_tr, yn_tr, yr_tr, X_v, yn_v, yr_v):
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(yn_tr, dtype=torch.float32)
    Xv = torch.tensor(X_v,  dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=0)
    m   = SmallLSTM(X_tr.shape[2], HIDDEN_SIZE, DROPOUT).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=LEARNING_RATE)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    lf  = nn.MSELoss()
    best_rmse = float("inf")
    hist = []
    no_imp = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        m.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = lf(m(Xb), yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        m.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(Xv), 1024):
                preds.append(m(Xv[i:i+1024].to(device)).cpu().numpy())
        pred_raw = np.concatenate(preds) * std_pm25 + mean_pm25
        rmse = float(np.sqrt(np.mean((pred_raw - yr_v) ** 2)))
        sch.step(rmse)
        hist.append(rmse)
        if rmse < best_rmse:
            best_rmse = rmse; no_imp = 0
        else:
            no_imp += 1
        if no_imp >= PATIENCE:
            break
    return best_rmse, hist

# ─── Main ablation loop ───────────────────────────────────────────────────────
ablation_results = {}

for w in WINDOW_SIZES:
    print(f"\n{'=' * 65}")
    print(f" Window = {w}h -- building windows ...")
    t0 = time.time()
    X_tr, yn_tr, yr_tr = build_windows(train_df, w)
    X_v,  yn_v,  yr_v  = build_windows(val_df,   w)
    print(f"  Train windows: {X_tr.shape[0]:,}   Val windows: {X_v.shape[0]:,}")
    print(f"  Training SmallLSTM ...")
    best_rmse, hist = train_and_eval(X_tr, yn_tr, yr_tr, X_v, yn_v, yr_v)
    elapsed = time.time() - t0
    ablation_results[w] = {"val_rmse": best_rmse, "history": hist}
    print(f"  Window {w}h -> Best Val RMSE: {best_rmse:.3f}  ({elapsed:.0f}s)")

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" WINDOW ABLATION SUMMARY")
print("=" * 65)
print(f"  {'Window':>8}  {'Val RMSE':>10}")
for w in WINDOW_SIZES:
    rmse = ablation_results[w]["val_rmse"]
    best_mark = " <-- BEST" if rmse == min(ablation_results[w2]["val_rmse"] for w2 in WINDOW_SIZES) else ""
    print(f"  {str(w)+'h':>8}  {rmse:>10.3f}{best_mark}")

with open("window_ablation_results.pkl", "wb") as f:
    pickle.dump(ablation_results, f)

print("\n  Saved: window_ablation_results.pkl")
