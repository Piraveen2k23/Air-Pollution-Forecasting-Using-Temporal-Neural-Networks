"""
train_tcn.py  —  Step 5B: Train a TCN to forecast PM2.5 one hour ahead.

HOW A TCN WORKS (quick recap):
  Instead of reading the 24-hour sequence step-by-step (like LSTM),
  a TCN slides small convolution filters over the time axis — similar to
  how image filters detect edges or textures, but applied to time.

  The key tricks that make it work for forecasting:
    1. Causal convolution  — filter at time t only sees t and earlier (never future)
    2. Dilated convolution — filter skips steps to see far back with few parameters
    3. Residual connection — skip connections to prevent vanishing gradients

  Dilation schedule (kernel=3, levels=4):
    Block 1 (dilation=1): sees up to 3 timesteps back
    Block 2 (dilation=2): sees up to 7 timesteps back
    Block 3 (dilation=4): sees up to 15 timesteps back
    Block 4 (dilation=8): sees up to 31 timesteps back  ← covers all 24h!

Key terms:
  - Conv1d        : 1D convolution (filter sliding over the time axis)
  - dilation      : how many steps the filter skips between the values it reads
  - causal        : only looks at the past, never the future
  - BatchNorm     : normalizes the layer outputs to stabilize training
  - residual      : output = layer_output + original_input (skip connection)
  - receptive field: how many total timesteps a layer can "see"
"""

import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
NUM_CHANNELS  = 256     # number of convolution filters in each layer (width of network)
KERNEL_SIZE   = 5      # how many consecutive (or dilated) timesteps each filter covers
NUM_LEVELS    = 4      # number of TCN blocks — dilation doubles each block: 1,2,4,8
DROPOUT       = 0.3    # fraction of neurons randomly zeroed (prevents overfitting)
BATCH_SIZE    = 512    # examples per gradient update
LEARNING_RATE = 1e-3   # step size for weight updates
NUM_EPOCHS    = 30     # max training passes through the full data
PATIENCE      = 5      # stop early if val RMSE doesn't improve for this many epochs
EVAL_BATCH    = 2048   # batch size for evaluation (bigger = faster, uses more RAM)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load pre-built windows + preprocessing stats
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print(" STEP 1: Loading data from windows.npz ...")
print("=" * 65)

data = np.load("windows.npz", allow_pickle=True)
X_train      = data["X_train"]        # (262944, 24, 29)
y_train_norm = data["y_train_norm"]   # (262944,)
y_train_raw  = data["y_train_raw"]    # (262944,)
X_val        = data["X_val"]          # (52128,  24, 29)
y_val_norm   = data["y_val_norm"]
y_val_raw    = data["y_val_raw"]

with open("fitted_preprocessing.pkl", "rb") as f:
    fitted = pickle.load(f)
mean_pm25 = float(fitted["means"]["PM2.5"])
std_pm25  = float(fitted["stds"]["PM2.5"])

INPUT_SIZE = X_train.shape[2]   # 29

print(f"  X_train : {X_train.shape}  ({X_train.shape[0]:,} windows)")
print(f"  X_val   : {X_val.shape}  ({X_val.shape[0]:,} windows)")
print(f"  Features: {INPUT_SIZE}")

# Receptive field check — make sure we can see all 24 hours
rf = 1 + (KERNEL_SIZE - 1) * sum(2**i for i in range(NUM_LEVELS))
print(f"\n  TCN receptive field with kernel={KERNEL_SIZE}, levels={NUM_LEVELS}: {rf} timesteps")
print(f"  Window length: 24 timesteps  →  {'OK, covers full window!' if rf >= 24 else 'WARNING: too small!'}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Convert to PyTorch tensors + DataLoaders
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" STEP 2: Wrapping data in PyTorch DataLoaders ...")
print("=" * 65)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train_norm, dtype=torch.float32)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val_norm, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
print(f"  Train batches per epoch: {len(train_loader)}")
print(f"  Val   batches (eval)   : {len(X_val_t) // EVAL_BATCH + 1}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Define the TCN Model
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" STEP 3: Defining the TCN model ...")
print("=" * 65)


class CausalConv1d(nn.Module):
    """
    A single 1D convolution that is strictly causal (only looks at past).

    How causal padding works:
      - PyTorch's Conv1d with padding=p adds p zeros to BOTH left and right sides.
      - For causal convolution we need zeros ONLY on the LEFT.
      - Fix: use padding=p (pads both sides), then CHOP off the rightmost p elements.
      - This ensures: output at time t only depends on input at times <= t.

    padding_amount = (kernel_size - 1) * dilation
    This is exactly what's needed so output length == input length.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding_amount = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation,
            padding=self.padding_amount,   # pads both sides
        )

    def forward(self, x):
        # x shape: (batch, channels, time)
        out = self.conv(x)
        # Chop off rightmost self.padding_amount elements (future-looking padding)
        if self.padding_amount > 0:
            out = out[:, :, :-self.padding_amount]
        return out


class TCNBlock(nn.Module):
    """
    One TCN block = two causal conv layers + BatchNorm + ReLU + Dropout + residual.

    Two convolutions per block is standard (same as ResNet blocks) —
    it gives the block more representational power than a single conv.

    BatchNorm: normalizes the activations across the batch dimension.
               Makes training much more stable (less sensitive to learning rate).

    Residual: if in_channels != out_channels, we use a 1×1 conv to match dimensions
              before adding. (1×1 conv = convolution with kernel_size=1 = just a linear
              projection of the channel dimension, no temporal mixing)
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()

        # Two causal conv layers
        self.causal1 = CausalConv1d(in_channels,  out_channels, kernel_size, dilation)
        self.causal2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)

        # Batch normalization (normalizes over batch dim to stabilize training)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Residual projection: match channel dims if they differ
        # (only needed for the very first block where in_channels=input_size=29)
        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x):
        # x shape: (batch, in_channels, time)

        # First causal conv → norm → relu → dropout
        out = self.causal1(x)
        out = self.dropout(F.relu(self.norm1(out)))

        # Second causal conv → norm → relu → dropout
        out = self.causal2(out)
        out = self.dropout(F.relu(self.norm2(out)))

        # Add residual: output = learned_features + original_input
        # This is the "skip connection" — it helps gradients flow backward
        return out + self.residual_proj(x)


class TCNForecaster(nn.Module):
    """
    Full TCN model for PM2.5 forecasting.

    Architecture:
      Input  (batch, 24, 29)
         ↓  transpose to (batch, 29, 24)   [Conv1d is channels-first]
         ↓
      TCN Block (dilation=1)  → sees 3 timesteps
         ↓
      TCN Block (dilation=2)  → sees 7 timesteps
         ↓
      TCN Block (dilation=4)  → sees 15 timesteps
         ↓
      TCN Block (dilation=8)  → sees 31 timesteps ✓ covers all 24h
         ↓  (batch, num_channels, 24)
      Take last timestep  → (batch, num_channels)
         ↓
      Linear (num_channels → 1)
         ↓
      Predicted PM2.5_norm
    """

    def __init__(self, input_size, num_channels, kernel_size, num_levels, dropout):
        super().__init__()

        # Build num_levels TCN blocks with doubling dilation: 1, 2, 4, 8, ...
        blocks = []
        for i in range(num_levels):
            dilation  = 2 ** i                              # 1, 2, 4, 8
            in_ch     = input_size if i == 0 else num_channels
            blocks.append(TCNBlock(in_ch, num_channels, kernel_size, dilation, dropout))

        self.tcn = nn.Sequential(*blocks)

        # Final prediction head: channels → 1 value
        self.fc = nn.Linear(num_channels, 1)

    def forward(self, x):
        # x: (batch, 24, 29)  ← time-first from DataLoader

        # Conv1d expects (batch, channels, time) → transpose
        x = x.transpose(1, 2)               # → (batch, 29, 24)

        # Pass through all TCN blocks
        out = self.tcn(x)                   # → (batch, num_channels, 24)

        # Take ONLY the last timestep (has seen all previous timesteps via dilation)
        last = out[:, :, -1]                # → (batch, num_channels)

        # Project to single prediction
        return self.fc(last).squeeze(-1)    # → (batch,)


model = TCNForecaster(
    input_size=INPUT_SIZE,
    num_channels=NUM_CHANNELS,
    kernel_size=KERNEL_SIZE,
    num_levels=NUM_LEVELS,
    dropout=DROPOUT,
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Model architecture:\n{model}")
print(f"\n  Total trainable parameters: {total_params:,}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Loss function and optimizer
# ─────────────────────────────────────────────────────────────────────────────
loss_fn   = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: batched evaluation  (avoids OOM errors on large datasets)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_rmse_batched(model, X_tensor, y_raw, batch_size=EVAL_BATCH):
    """
    Run the model in batches and return RMSE in real μg/m3 units.

    Why batched? If we pass ALL 262k training windows at once, PyTorch tries
    to allocate a huge tensor all at once → runs out of RAM.
    Processing in smaller chunks avoids that problem entirely.
    """
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i + batch_size].to(device)
            pred  = model(batch).cpu().numpy()
            all_preds.append(pred)
    preds_norm = np.concatenate(all_preds)
    preds_raw  = preds_norm * std_pm25 + mean_pm25
    rmse = float(np.sqrt(np.mean((preds_raw - y_raw) ** 2)))
    return rmse, preds_raw

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Training loop
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f" STEP 5: Training for up to {NUM_EPOCHS} epochs ...")
print(f"         (early stopping after {PATIENCE} epochs of no improvement)")
print("=" * 65)
print(f"  {'Epoch':>5}  {'Train Loss':>12}  {'Val RMSE':>10}  {'Time':>8}  {'Status'}")
print(f"  {'-'*5}  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*15}")

best_val_rmse     = float("inf")
epochs_no_improve = 0
history           = {"train_loss": [], "val_rmse": []}

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()

    # ── Training phase ──────────────────────────────────────────────────────
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        pred = model(X_batch)                    # forward pass
        loss = loss_fn(pred, y_batch)            # compute MSE loss

        optimizer.zero_grad()                    # clear old gradients
        loss.backward()                          # compute new gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # clip for stability
        optimizer.step()                         # update weights

        running_loss += loss.item() * len(X_batch)

    avg_train_loss = running_loss / len(X_train_t)

    # ── Validation phase (batched!) ──────────────────────────────────────────
    val_rmse, _ = evaluate_rmse_batched(model, X_val_t, y_val_raw)

    scheduler.step(val_rmse)
    elapsed = time.time() - t0
    history["train_loss"].append(avg_train_loss)
    history["val_rmse"].append(val_rmse)

    # ── Early stopping + model saving ────────────────────────────────────────
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save(model.state_dict(), "best_tcn.pt")
        epochs_no_improve = 0
        status = "<-- best"
    else:
        epochs_no_improve += 1
        status = f"no improve ({epochs_no_improve}/{PATIENCE})"

    print(f"  {epoch:>5}  {avg_train_loss:>12.6f}  {val_rmse:>10.3f}  {elapsed:>7.1f}s  {status}")

    if epochs_no_improve >= PATIENCE:
        print(f"\n  Early stopping triggered after epoch {epoch}.")
        break

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Final evaluation and comparison
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" STEP 6: Loading best TCN and comparing to LSTM + baselines ...")
print("=" * 65)

model.load_state_dict(torch.load("best_tcn.pt", weights_only=True))

train_rmse, _            = evaluate_rmse_batched(model, X_train_t, y_train_raw)
val_rmse,   val_pred_raw = evaluate_rmse_batched(model, X_val_t,   y_val_raw)
train_mae = float(np.mean(np.abs(evaluate_rmse_batched(model, X_train_t, y_train_raw)[1] - y_train_raw)))
val_mae   = float(np.mean(np.abs(val_pred_raw - y_val_raw)))

# Load baseline results
try:
    with open("baseline_results.pkl", "rb") as f:
        baseline = pickle.load(f)
    persist_rmse = baseline["results"]["persistence_val"][0]
    persist_mae  = baseline["results"]["persistence_val"][1]
    ridge_rmse   = baseline["results"]["ridge_val"][0]
    ridge_mae    = baseline["results"]["ridge_val"][1]
    has_baseline = True
except FileNotFoundError:
    has_baseline = False

# Load LSTM results for comparison
try:
    with open("lstm_results.pkl", "rb") as f:
        lstm_res = pickle.load(f)
    lstm_rmse = lstm_res["val_rmse"]
    lstm_mae  = lstm_res["val_mae"]
    has_lstm = True
except FileNotFoundError:
    has_lstm = False

print(f"\n  {'Model':<28}  {'Val RMSE':>10}  {'Val MAE':>10}")
print(f"  {'-'*28}  {'-'*10}  {'-'*10}")
if has_baseline:
    print(f"  {'Persistence (baseline)':<28}  {persist_rmse:>10.3f}  {persist_mae:>10.3f}")
    print(f"  {'Ridge (baseline)':<28}  {ridge_rmse:>10.3f}  {ridge_mae:>10.3f}")
if has_lstm:
    print(f"  {'LSTM':<28}  {lstm_rmse:>10.3f}  {lstm_mae:>10.3f}")
print(f"  {'TCN (ours)':<28}  {val_rmse:>10.3f}  {val_mae:>10.3f}")

if has_baseline:
    print(f"\n  Beat Persistence? {'YES' if val_rmse < persist_rmse else 'NO'}  "
          f"(diff: {persist_rmse - val_rmse:+.3f})")
    print(f"  Beat Ridge?       {'YES' if val_rmse < ridge_rmse else 'NO'}  "
          f"(diff: {ridge_rmse - val_rmse:+.3f})")
if has_lstm:
    print(f"  Beat LSTM?        {'YES' if val_rmse < lstm_rmse else 'NO'}  "
          f"(diff: {lstm_rmse - val_rmse:+.3f})")

# ── Save TCN results ─────────────────────────────────────────────────────────
results = {
    "val_rmse":    val_rmse,
    "val_mae":     val_mae,
    "train_rmse":  train_rmse,
    "train_mae":   train_mae,
    "val_pred_raw": val_pred_raw,
    "y_val_raw":   y_val_raw,
    "history":     history,
    "hyperparams": {
        "num_channels":  NUM_CHANNELS,
        "kernel_size":   KERNEL_SIZE,
        "num_levels":    NUM_LEVELS,
        "dropout":       DROPOUT,
        "batch_size":    BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
    },
}
with open("tcn_results.pkl", "wb") as f:
    pickle.dump(results, f)

print(f"\n  Saved: best_tcn.pt       (model weights)")
print(f"         tcn_results.pkl    (metrics + predictions)")
print("\n" + "=" * 65)
print(f"  DONE!  Best Val RMSE = {best_val_rmse:.3f} ug/m3")
print("=" * 65)
