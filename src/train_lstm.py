"""
train_lstm.py  —  Step 5A: Train an LSTM to forecast PM2.5 one hour ahead.

HOW THIS SCRIPT WORKS (plain English):
  1. Load the pre-built windows (262,944 training examples, each = 24 hours of data)
  2. Define the LSTM model (a neural network with memory)
  3. Train it: show it batches of examples, compare predictions to truth, adjust weights
  4. After every epoch (full pass through data), check performance on the val set
  5. Save the best model (lowest val RMSE) to disk
  6. Print a final comparison against the baseline scores

Key terms used below:
  - epoch      : one full pass through ALL training data
  - batch      : small chunk of training examples processed at once (e.g., 512)
  - loss       : a number measuring how wrong the predictions are (we want it to go DOWN)
  - gradient   : direction to nudge each weight to reduce loss
  - optimizer  : algorithm that does the nudging (we use Adam — very popular)
  - dropout    : randomly zero out some neurons during training to prevent overfitting
  - hidden_size: how many memory units the LSTM has (bigger = more capacity, slower)
"""

import numpy as np
import pickle
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS  — settings you can change to experiment
# ─────────────────────────────────────────────────────────────────────────────
HIDDEN_SIZE  = 256    # number of memory units in each LSTM layer
NUM_LAYERS   = 3      # how many LSTM layers to stack (depth of the network)
DROPOUT      = 0.2    # 20% of neurons randomly disabled each step (prevents overfitting)
BATCH_SIZE   = 512    # examples processed per gradient update
LEARNING_RATE = 1e-3  # step size for weight updates (0.001 is a good default)
NUM_EPOCHS   = 30     # how many times to loop through all training data
PATIENCE     = 5      # stop early if val RMSE doesn't improve for this many epochs

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load the pre-built windows and preprocessing stats
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print(" STEP 1: Loading data from windows.npz ...")
print("=" * 65)

data = np.load("data/processed/windows.npz", allow_pickle=True)

# X_train shape: (262944, 24, 29)  — 262k examples, 24 hours, 29 features each
# y_train_norm  — normalized PM2.5 targets (what the model directly predicts)
# y_train_raw   — actual μg/m³ targets (used only for final RMSE reporting)
X_train      = data["X_train"]        # shape (N_train, 24, 29)
y_train_norm = data["y_train_norm"]   # shape (N_train,)
y_train_raw  = data["y_train_raw"]    # shape (N_train,)

X_val        = data["X_val"]          # shape (N_val, 24, 29)
y_val_norm   = data["y_val_norm"]     # shape (N_val,)
y_val_raw    = data["y_val_raw"]      # shape (N_val,)

# Load the mean and std of PM2.5 (from training data only).
# We need these to convert normalized predictions → real μg/m³ for RMSE.
with open("data/processed/fitted_preprocessing.pkl", "rb") as f:
    fitted = pickle.load(f)
mean_pm25 = float(fitted["means"]["PM2.5"])
std_pm25  = float(fitted["stds"]["PM2.5"])

INPUT_SIZE = X_train.shape[2]   # 29 features

print(f"  X_train : {X_train.shape}  ({X_train.shape[0]:,} windows)")
print(f"  X_val   : {X_val.shape}  ({X_val.shape[0]:,} windows)")
print(f"  Features: {INPUT_SIZE}")
print(f"  PM2.5 mean={mean_pm25:.2f}, std={std_pm25:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Convert numpy arrays → PyTorch Tensors and build DataLoaders
# ─────────────────────────────────────────────────────────────────────────────
# PyTorch works with "Tensors" (think: multi-dimensional arrays with GPU support).
# DataLoader wraps a dataset and automatically feeds shuffled batches during training.

print("\n" + "=" * 65)
print(" STEP 2: Wrapping data in PyTorch DataLoaders ...")
print("=" * 65)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}  (CPU training — will take a few minutes per epoch)")

# Convert to float32 tensors (standard precision for neural networks)
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train_norm, dtype=torch.float32)

X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val_norm, dtype=torch.float32)

# TensorDataset pairs up inputs and targets
# DataLoader slices them into batches and shuffles training data each epoch
train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=BATCH_SIZE,
    shuffle=True,     # shuffle order each epoch so the model doesn't memorize order
    num_workers=0,
)
val_loader = DataLoader(
    TensorDataset(X_val_t, y_val_t),
    batch_size=BATCH_SIZE * 2,   # can use bigger batches for validation (no gradients)
    shuffle=False,
    num_workers=0,
)
print(f"  Train batches per epoch: {len(train_loader)}")
print(f"  Val   batches per epoch: {len(val_loader)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Define the LSTM Model
# ─────────────────────────────────────────────────────────────────────────────
# In PyTorch, you define a model as a Python class that inherits from nn.Module.
# Two methods are required:
#   __init__ : define the layers (called once at creation)
#   forward  : define how data flows through the layers (called every batch)

print("\n" + "=" * 65)
print(" STEP 3: Defining the LSTM model ...")
print("=" * 65)

class LSTMForecaster(nn.Module):
    """
    LSTM network for 1-hour-ahead PM2.5 forecasting.

    Architecture:
        Input  (batch, 24, 29)
           ↓
        LSTM × NUM_LAYERS   — reads the 24-hour sequence, builds a memory state
           ↓  (batch, 24, hidden_size)
        Take last timestep   — only the final hidden state carries the full context
           ↓  (batch, hidden_size)
        Dropout              — randomly zero some units to prevent overfitting
           ↓
        Linear (hidden → 1) — squash hidden state to one prediction number
           ↓  (batch,)
        Output: predicted PM2.5_norm
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()

        self.hidden_size = hidden_size

        # The LSTM layer(s).
        # batch_first=True means input shape is (batch, seq_len, features)
        # dropout applies between LSTM layers (not after the last one)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Dropout applied after the LSTM's last layer output
        self.dropout = nn.Dropout(dropout)

        # A simple linear (fully connected) layer: hidden_size → 1 number
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, 24, 29)

        # Pass through LSTM.
        # out shape: (batch_size, 24, hidden_size)
        out, _ = self.lstm(x)

        # Take the last timestep's output
        last = out[:, -1, :]

        # Apply dropout for regularization
        last = self.dropout(last)

        # Linear layer: hidden_size → 1
        # squeeze(-1) removes the trailing dimension: (batch, 1) → (batch,)
        return self.fc(last).squeeze(-1)


# Create an instance of the model and move it to the device (CPU here)
model = LSTMForecaster(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(device)

# Count how many learnable parameters the model has
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Model architecture:\n{model}")
print(f"\n  Total trainable parameters: {total_params:,}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Define Loss Function and Optimizer
# ─────────────────────────────────────────────────────────────────────────────
# Loss function: MSELoss (Mean Squared Error)
#   loss = mean( (predicted - actual)^2 )
#   We train on the NORMALIZED targets so the loss values are comparable.
#
# Optimizer: Adam
#   Adam = "Adaptive Moment Estimation". It adjusts the learning rate for each
#   weight individually based on recent gradient history. Very popular default.

loss_fn   = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# Learning rate scheduler: if val loss plateaus, reduce LR by half.
# This helps squeeze out the last few percent of performance.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Training Loop
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print(f" STEP 5: Training for up to {NUM_EPOCHS} epochs ...")
print(f"         (early stopping after {PATIENCE} epochs of no improvement)")
print("=" * 65)
print(f"  {'Epoch':>5}  {'Train Loss':>12}  {'Val RMSE':>10}  {'Time':>8}  {'Status'}")
print(f"  {'-'*5}  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*10}")

# Helper: compute RMSE on validation set in real μg/m³ units
def evaluate_rmse(model, X_tensor, y_raw, batch_size=1024):
    """Run model on full X_tensor in batches (no gradient), return RMSE vs y_raw (real units)."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i + batch_size].to(device)
            pred = model(batch).cpu().numpy()
            all_preds.append(pred)
    pred_norm = np.concatenate(all_preds)
    pred_raw = pred_norm * std_pm25 + mean_pm25    # un-normalize
    rmse = float(np.sqrt(np.mean((pred_raw - y_raw) ** 2)))
    return rmse, pred_raw

best_val_rmse   = float("inf")
epochs_no_improve = 0
history = {"train_loss": [], "val_rmse": []}

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()

    # ── Training phase ──────────────────────────────────────────────────────
    model.train()   # tells PyTorch: enable dropout, compute gradients
    running_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # 1. Forward pass: feed input through the model to get predictions
        pred = model(X_batch)

        # 2. Compute loss: how wrong are the predictions?
        loss = loss_fn(pred, y_batch)

        # 3. Backward pass: compute gradients (which direction to update each weight)
        optimizer.zero_grad()   # clear gradients from previous batch
        loss.backward()         # compute new gradients

        # 4. Gradient clipping: prevent gradients from exploding (common in RNNs)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 5. Update weights using the optimizer (Adam)
        optimizer.step()

        running_loss += loss.item() * len(X_batch)

    avg_train_loss = running_loss / len(X_train_t)

    # ── Validation phase ─────────────────────────────────────────────────────
    val_rmse, _ = evaluate_rmse(model, X_val_t, y_val_raw)

    # Step the learning rate scheduler based on val RMSE
    scheduler.step(val_rmse)

    elapsed = time.time() - t0
    history["train_loss"].append(avg_train_loss)
    history["val_rmse"].append(val_rmse)

    # ── Early stopping + best model saving ───────────────────────────────────
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save(model.state_dict(), "models/best_lstm.pt")   # save weights to disk
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
# STEP 6 — Load Best Model and Final Evaluation
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" STEP 6: Loading best model and evaluating ...")
print("=" * 65)

model.load_state_dict(torch.load("models/best_lstm.pt", weights_only=True))

train_rmse, train_pred_raw = evaluate_rmse(model, X_train_t, y_train_raw)
val_rmse,   val_pred_raw   = evaluate_rmse(model, X_val_t,   y_val_raw)

train_mae = float(np.mean(np.abs(train_pred_raw - y_train_raw)))
val_mae   = float(np.mean(np.abs(val_pred_raw   - y_val_raw)))

# ── Load baseline results to compare ────────────────────────────────────────
try:
    with open("results/baseline_results.pkl", "rb") as f:
        baseline = pickle.load(f)
    persist_val_rmse = baseline["results"]["persistence_val"][0]
    ridge_val_rmse   = baseline["results"]["ridge_val"][0]
    persist_val_mae  = baseline["results"]["persistence_val"][1]
    ridge_val_mae    = baseline["results"]["ridge_val"][1]
    has_baseline = True
except FileNotFoundError:
    has_baseline = False

print(f"\n  {'Model':<25}  {'Val RMSE':>10}  {'Val MAE':>10}")
print(f"  {'-'*25}  {'-'*10}  {'-'*10}")
if has_baseline:
    print(f"  {'Persistence (baseline)':<25}  {persist_val_rmse:>10.3f}  {persist_val_mae:>10.3f}")
    print(f"  {'Ridge (baseline)':<25}  {ridge_val_rmse:>10.3f}  {ridge_val_mae:>10.3f}")
print(f"  {'LSTM (ours)':<25}  {val_rmse:>10.3f}  {val_mae:>10.3f}")

if has_baseline:
    beat_persist = val_rmse < persist_val_rmse
    beat_ridge   = val_rmse < ridge_val_rmse
    print(f"\n  Beat Persistence? {'YES' if beat_persist else 'NO'}  "
          f"(improvement: {persist_val_rmse - val_rmse:+.3f} RMSE)")
    print(f"  Beat Ridge?       {'YES' if beat_ridge   else 'NO'}  "
          f"(improvement: {ridge_val_rmse   - val_rmse:+.3f} RMSE)")

# ── Save results for later comparison (with TCN) ─────────────────────────────
results = {
    "val_rmse":      val_rmse,
    "val_mae":       val_mae,
    "train_rmse":    train_rmse,
    "train_mae":     train_mae,
    "val_pred_raw":  val_pred_raw,
    "y_val_raw":     y_val_raw,
    "history":       history,
    "hyperparams": {
        "hidden_size":   HIDDEN_SIZE,
        "num_layers":    NUM_LAYERS,
        "dropout":       DROPOUT,
        "batch_size":    BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs":    NUM_EPOCHS,
    },
}
with open("results/lstm_results.pkl", "wb") as f:
    pickle.dump(results, f)

print(f"\n  Saved: best_lstm.pt          (model weights)")
print(f"         lstm_results.pkl       (metrics + predictions for comparison)")
print("\n" + "=" * 65)
print(f"  DONE!  Best Val RMSE = {best_val_rmse:.3f} ug/m3")
print("=" * 65)
