"""
train_lstm_v2.py  —  Improved LSTM with:
  - Wider hidden state (256 units, 3 layers)
  - Attention mechanism over ALL sequence outputs (not just last timestep)
  - Huber loss (robust to PM2.5 spike outliers)
  - AdamW optimizer with weight decay (better generalization)
  - ReduceLROnPlateau scheduler (CPU-friendly — halves LR when stuck)
  - CPU-optimized: 1 training seed, patience=5

Uses windows_v2.npz (48h window + 46 lag/rolling features).
Run AFTER feature_engineering.py.

Output files:
  best_lstm_v2_seed0.pt    ← trained model weights
  lstm_v2_results.pkl      ← val predictions + metrics
"""

import numpy as np
import pickle
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS  (CPU-optimized)
# ─────────────────────────────────────────────────────────────────────────────
HIDDEN_SIZE   = 256     # wider than v1 (was 128) — more memory
NUM_LAYERS    = 3       # deeper than v1 (was 2)
DROPOUT       = 0.3
BATCH_SIZE    = 512     # good balance for CPU
LEARNING_RATE = 1e-3
NUM_EPOCHS    = 50      # max epochs; early stopping will trigger sooner
PATIENCE      = 5       # stop if val RMSE doesn't improve for 5 epochs
SEED_COUNT    = 1       # CPU: train 1 seed only (saves 2/3 of time vs 3 seeds)
HUBER_DELTA   = 1.0     # Huber loss delta — transition between L1 and L2 regime

WINDOWS_FILE  = "data/processed/windows_v2.npz"
FITTED_PATH   = "data/processed/fitted_preprocessing.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print(" LSTM v2 — Attention + Huber loss  (48h window, CPU build)")
print("=" * 65)

print(f"\n[1/4] Loading {WINDOWS_FILE} ...")
data = np.load(WINDOWS_FILE, allow_pickle=True)
X_train      = data["X_train"]
y_train_norm = data["y_train_norm"]
y_train_raw  = data["y_train_raw"]
X_val        = data["X_val"]
y_val_norm   = data["y_val_norm"]
y_val_raw    = data["y_val_raw"]

with open(FITTED_PATH, "rb") as f:
    fitted = pickle.load(f)
mean_pm25 = float(fitted["means"]["PM2.5"])
std_pm25  = float(fitted["stds"]["PM2.5"])

INPUT_SIZE = X_train.shape[2]   # 46 features
SEQ_LEN    = X_train.shape[1]   # 48 hours

print(f"  X_train : {X_train.shape}  ({X_train.shape[0]:,} windows)")
print(f"  X_val   : {X_val.shape}  ({X_val.shape[0]:,} windows)")
print(f"  Features: {INPUT_SIZE}  |  Sequence: {SEQ_LEN}h")
print(f"  PM2.5   : mean={mean_pm25:.2f}, std={std_pm25:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Build DataLoaders
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/4] Building DataLoaders ...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train_norm, dtype=torch.float32)
X_val_t   = torch.tensor(X_val, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False,   # False on CPU
)
print(f"  Train batches / epoch: {len(train_loader)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Model: LSTM + Temporal Attention
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/4] Defining LSTM v2 with attention ...")

class TemporalAttention(nn.Module):
    """
    Additive attention over the 48 LSTM output vectors.

    The model learns to weight each hour:
      - Recent high-PM2.5 hours → high weight
      - Stale calm hours        → low weight

    Output: a single context vector (weighted average over all 48 hours)
    instead of just taking the last hour's output.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch, seq_len, hidden)
        scores  = self.score(lstm_out)              # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)      # (batch, seq_len, 1)
        context = (weights * lstm_out).sum(dim=1)   # (batch, hidden)
        return context


class LSTMv2Forecaster(nn.Module):
    """
    Improved LSTM for PM2.5 forecasting.

    Architecture:
        Input  (batch, 48, 46)
           ↓
        LSTM × 3 layers, hidden=256
           ↓  (batch, 48, 256) — outputs at ALL timesteps
        TemporalAttention
           ↓  (batch, 256) — weighted sum over 48 hours
        Dropout(0.3)
           ↓
        Linear(256 → 64) + ReLU
        Linear(64 → 1)
           ↓
        Predicted PM2.5_norm
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (batch, 48, 46)
        out, _  = self.lstm(x)           # (batch, 48, 256)
        context = self.attention(out)    # (batch, 256)
        context = self.dropout(context)
        return self.head(context).squeeze(-1)   # (batch,)


def evaluate_rmse(model, X_t, y_raw, batch_size=2048):
    """Run model in eval mode, return RMSE in real μg/m³."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            p = model(X_t[i:i + batch_size].to(device)).cpu().numpy()
            preds.append(p)
    preds_norm = np.concatenate(preds)
    preds_raw  = preds_norm * std_pm25 + mean_pm25
    rmse = float(np.sqrt(np.mean((preds_raw - y_raw) ** 2)))
    return rmse, preds_raw


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Training loop
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[4/4] Training ...")
print(f"  Config: hidden={HIDDEN_SIZE}, layers={NUM_LAYERS}, dropout={DROPOUT}")
print(f"          window={SEQ_LEN}h, features={INPUT_SIZE}")
print(f"          loss=Huber(delta={HUBER_DELTA}), patience={PATIENCE}")

all_seed_val_preds = []

for seed in range(SEED_COUNT):
    print(f"\n{'─' * 65}")
    print(f"  SEED {seed}")
    print(f"{'─' * 65}")

    torch.manual_seed(seed * 42)
    np.random.seed(seed * 42)

    model = LSTMv2Forecaster(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,}")

    loss_fn   = nn.HuberLoss(delta=HUBER_DELTA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # ReduceLROnPlateau: halves LR if val RMSE doesn't improve for 3 epochs
    # Works better than CosineAnnealing on CPU where epochs take longer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_rmse     = float("inf")
    epochs_no_improve = 0
    best_weights      = None

    print(f"\n  {'Epoch':>5}  {'Train Loss':>12}  {'Val RMSE':>10}  {'Time':>8}  {'LR':>8}  Status")
    print(f"  {'-'*5}  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*12}")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * len(X_batch)

        avg_loss = running_loss / len(X_train_t)
        val_rmse, val_preds = evaluate_rmse(model, X_val_t, y_val_raw)
        scheduler.step(val_rmse)   # pass val RMSE to ReduceLROnPlateau
        elapsed  = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_weights  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            status = "<-- BEST"
        else:
            epochs_no_improve += 1
            status = f"({epochs_no_improve}/{PATIENCE})"

        print(f"  {epoch:>5}  {avg_loss:>12.6f}  {val_rmse:>10.3f}  {elapsed:>7.1f}s  {current_lr:>8.6f}  {status}")

        if epochs_no_improve >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}.")
            break

    # Save best weights
    save_path = f"models/best_lstm_v2_seed{seed}.pt"
    torch.save(best_weights, save_path)
    print(f"\n  Seed {seed} best val RMSE: {best_val_rmse:.3f} → saved {save_path}")

    # Load best and get predictions for results file
    model.load_state_dict(best_weights)
    _, best_val_preds = evaluate_rmse(model, X_val_t, y_val_raw)
    all_seed_val_preds.append(best_val_preds)


# ─────────────────────────────────────────────────────────────────────────────
# Final summary + save results
# ─────────────────────────────────────────────────────────────────────────────
ensemble_preds = np.mean(all_seed_val_preds, axis=0)
ensemble_rmse  = float(np.sqrt(np.mean((ensemble_preds - y_val_raw) ** 2)))

print("\n" + "=" * 65)
print("  RESULTS")
print("=" * 65)
print(f"  Val RMSE (seed 0) : {float(np.sqrt(np.mean((all_seed_val_preds[0] - y_val_raw)**2))):.3f} μg/m³")
print(f"\n  Previous best     : 14.878 (LSTM 70% + TCN 30% on 24h windows)")
print("=" * 65)

with open("results/lstm_v2_results.pkl", "wb") as f:
    pickle.dump({
        "val_pred_raw":     all_seed_val_preds[0],
        "val_pred_ensemble": ensemble_preds,
        "y_val_raw":        y_val_raw,
        "mean_pm25":        mean_pm25,
        "std_pm25":         std_pm25,
        "hyperparams": {
            "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS,
            "dropout":     DROPOUT,     "huber_delta": HUBER_DELTA,
            "window":      SEQ_LEN,     "n_features":  INPUT_SIZE,
        },
    }, f)

print(f"\n  Saved: best_lstm_v2_seed0.pt")
print(f"         lstm_v2_results.pkl")
print(f"\n  Next: python generate_lstm_v2_submission.py")
