"""
CO5420 - Air Pollution Forecasting: Best LSTM Model (single-model, tuned)

This is a focused, single-architecture version of the pipeline built around
everything we found that helps LSTM specifically:

  - PER-STATION normalisation (each of the 12 stations gets its own mean/std
    for every numeric feature, instead of one global mean/std). Stations have
    very different baseline pollution levels, so this is the single biggest
    lever for a multi-station model.
  - Rolling mean/std + first-difference features for PM2.5, PM10, TEMP, WSPM,
    computed causally within each 24-hour window (so they reconstruct
    identically from test.csv's lagged columns at prediction time).
  - Cyclical (sin/cos) encoding of wind direction and calendar time.
  - Station one-hot identity (kept even with per-station normalisation, since
    it can still help the model learn station-specific volatility/seasonal
    patterns beyond just the mean level).
  - A deeper/wider LSTM (128 hidden units, 2 layers) than the earlier
    CPU-limited version, with early stopping + LR scheduling.

Usage:
    python best_lstm.py --data train_raw.csv --epochs 40 --patience 8 \
        --hidden_size 128 --num_layers 2 --batch_size 256 \
        --save_path best_lstm.pt
"""

import argparse
import copy
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

RANDOM_SEED = 42
WINDOW_SIZE = 24
POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
WEATHER = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
NUMERIC_FEATURES = POLLUTANTS + WEATHER
WIND_DIR_COL = "wd"
STATION_COL = "station"
TARGET_COL = "PM2.5"
VAL_FRACTION = 0.15

# Rolling/diff features - all derivable from ONLY the values inside a single
# 24-hour window, so predict_best_lstm.py can reconstruct them from test.csv.
ROLLING_SOURCE_COLS = {
    "PM2.5_roll_mean_3": ("PM2.5", "mean", 3),
    "PM2.5_roll_mean_6": ("PM2.5", "mean", 6),
    "PM2.5_roll_mean_12": ("PM2.5", "mean", 12),
    "PM2.5_roll_std_6": ("PM2.5", "std", 6),
    "PM2.5_diff_1": ("PM2.5", "diff", 1),
    "PM10_roll_mean_6": ("PM10", "mean", 6),
    "TEMP_roll_mean_6": ("TEMP", "mean", 6),
    "WSPM_roll_mean_6": ("WSPM", "mean", 6),
}
EXTRA_TEMPORAL_FEATURES = list(ROLLING_SOURCE_COLS.keys())
NORMALISED_COLS = NUMERIC_FEATURES + EXTRA_TEMPORAL_FEATURES  # all get per-station z-score

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

COMPASS_TO_DEG = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5,
    "SE": 135, "SSE": 157.5, "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}


# ---------------------------------------------------------------------------
# 1. Load, impute, engineer features (PER-STATION normalisation)
# ---------------------------------------------------------------------------
def load_and_preprocess(csv_path: str):
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.sort_values([STATION_COL, "datetime"]).reset_index(drop=True)

    # --- Missing value imputation (per station) ---
    for col in NUMERIC_FEATURES:
        df[col] = df.groupby(STATION_COL)[col].transform(
            lambda s: s.interpolate(limit_direction="both"))
        df[col] = df.groupby(STATION_COL)[col].transform(lambda s: s.ffill().bfill())
        df[col] = df[col].fillna(df.groupby(STATION_COL)[col].transform("mean"))
        df[col] = df[col].fillna(df[col].mean())

    # --- Wind direction: circular sin/cos encoding ---
    df[WIND_DIR_COL] = df.groupby(STATION_COL)[WIND_DIR_COL].transform(
        lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "N"))
    deg = df[WIND_DIR_COL].map(COMPASS_TO_DEG).fillna(0.0)
    df["wd_sin"] = np.sin(np.deg2rad(deg))
    df["wd_cos"] = np.cos(np.deg2rad(deg))

    # --- Cyclical calendar features ---
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    dow = df["datetime"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # --- Station identity (one-hot) ---
    station_dummies = pd.get_dummies(df[STATION_COL], prefix="st").astype(np.float32)
    df = pd.concat([df, station_dummies], axis=1)
    station_cols = list(station_dummies.columns)

    # --- Rolling/diff temporal features (computed BEFORE normalisation) ---
    for feat_name, (src_col, kind, k) in ROLLING_SOURCE_COLS.items():
        if kind == "mean":
            df[feat_name] = df.groupby(STATION_COL)[src_col].transform(
                lambda s, k=k: s.rolling(k, min_periods=1).mean())
        elif kind == "std":
            df[feat_name] = df.groupby(STATION_COL)[src_col].transform(
                lambda s, k=k: s.rolling(k, min_periods=1).std().fillna(0.0))
        elif kind == "diff":
            df[feat_name] = df.groupby(STATION_COL)[src_col].transform(
                lambda s: s.diff().fillna(0.0))

    # --- PER-STATION z-score normalisation ---
    # Each station gets its own mean/std for every numeric + engineered
    # feature. This is the key change from the global-normalisation version:
    # stations have very different baseline pollution levels, so normalising
    # everyone the same way blurs signal a per-station model could use.
    norm_stats = {}   # {station: {feature: (mu, sigma)}}
    stations = sorted(df[STATION_COL].unique().tolist())
    for station in stations:
        norm_stats[station] = {}
        mask = df[STATION_COL] == station
        for col in NORMALISED_COLS:
            mu = df.loc[mask, col].mean()
            sigma = df.loc[mask, col].std() + 1e-8
            norm_stats[station][col] = (float(mu), float(sigma))

    for col in NORMALISED_COLS:
        mu_series = df[STATION_COL].map(lambda s: norm_stats[s][col][0])
        sigma_series = df[STATION_COL].map(lambda s: norm_stats[s][col][1])
        df[col + "_norm"] = (df[col] - mu_series) / sigma_series

    feature_cols = (
        [c + "_norm" for c in NUMERIC_FEATURES]
        + ["wd_sin", "wd_cos", "hour_sin", "hour_cos",
           "month_sin", "month_cos", "dow_sin", "dow_cos"]
        + [c + "_norm" for c in EXTRA_TEMPORAL_FEATURES]
        + station_cols
    )
    df.attrs["norm_stats"] = norm_stats
    df.attrs["feature_cols"] = feature_cols
    df.attrs["stations"] = stations
    return df, feature_cols


# ---------------------------------------------------------------------------
# 2. Windowing with chronological (per-station) train/val split
# ---------------------------------------------------------------------------
def build_windows_split(df: pd.DataFrame, feature_cols, window_size=WINDOW_SIZE,
                         val_fraction=VAL_FRACTION):
    X_train, y_train, X_val, y_val = [], [], [], []
    for _, group in df.groupby(STATION_COL):
        group = group.reset_index(drop=True)
        feats = group[feature_cols].values.astype(np.float32)
        target = group[TARGET_COL].values.astype(np.float32)
        n = len(group)
        split_point = int(n * (1 - val_fraction))
        for start in range(n - window_size):
            end = start + window_size
            target_idx = end
            x_win, y_ = feats[start:end], target[target_idx]
            if target_idx < split_point:
                X_train.append(x_win); y_train.append(y_)
            else:
                X_val.append(x_win); y_val.append(y_)
    return (np.stack(X_train), np.array(y_train, dtype=np.float32),
            np.stack(X_val), np.array(y_val, dtype=np.float32))


class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def persistence_baseline(df: pd.DataFrame, val_fraction=VAL_FRACTION):
    preds, actuals = [], []
    for _, group in df.groupby(STATION_COL):
        pm = group[TARGET_COL].values
        n = len(pm)
        split_point = int(n * (1 - val_fraction))
        for i in range(max(split_point, 1), n):
            preds.append(pm[i - 1]); actuals.append(pm[i])
    rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
    mae = float(mean_absolute_error(actuals, preds))
    return rmse, mae


# ---------------------------------------------------------------------------
# 3. Model: the one best LSTM
# ---------------------------------------------------------------------------
class BestLSTM(nn.Module):
    def __init__(self, n_features, hidden_size=128, num_layers=2, dropout=0.25):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)


# ---------------------------------------------------------------------------
# 4. Training with early stopping + LR scheduling
# ---------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs, lr, device, patience):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2)
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        val_rmse, val_mae = evaluate_model(model, val_loader, device)
        scheduler.step(val_rmse)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:2d}/{epochs} | train MSE: {train_loss:8.3f} "
              f"| val RMSE: {val_rmse:7.3f} | val MAE: {val_mae:7.3f} | lr: {lr_now:.2e}")

        if val_rmse < best_val_rmse - 1e-4:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (best val RMSE: {best_val_rmse:.3f})")
                break

    model.load_state_dict(best_state)
    return model, best_val_rmse


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    preds_all, targets_all = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        preds_all.append(model(xb).cpu().numpy())
        targets_all.append(yb.numpy())
    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    rmse = float(np.sqrt(mean_squared_error(targets_all, preds_all)))
    mae = float(mean_absolute_error(targets_all, preds_all))
    return rmse, mae


# ---------------------------------------------------------------------------
# 5. Orchestration
# ---------------------------------------------------------------------------
def main(csv_path, epochs, batch_size, patience, hidden_size, num_layers,
         lr, save_path, meta_path, resume_from=None, cache_windows=None):
    import os
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if cache_windows and os.path.exists(cache_windows):
        print("Loading cached windows + metadata...")
        cache = np.load(cache_windows)
        X_train, y_train, X_val, y_val = (cache["X_train"], cache["y_train"],
                                           cache["X_val"], cache["y_val"])
        with open(meta_path) as f:
            meta = json.load(f)
        norm_stats, feature_cols, stations = meta["norm_stats"], meta["feature_cols"], meta["stations"]
        persistence_rmse, persistence_mae = meta["persistence_rmse"], meta["persistence_mae"]
    else:
        print("Loading and preprocessing (per-station normalisation)...")
        df, feature_cols = load_and_preprocess(csv_path)
        norm_stats = df.attrs["norm_stats"]
        stations = df.attrs["stations"]

        print("Computing persistence baseline...")
        persistence_rmse, persistence_mae = persistence_baseline(df)
        print(f"    Persistence RMSE: {persistence_rmse:.3f} | MAE: {persistence_mae:.3f}")

        print("Building windows...")
        X_train, y_train, X_val, y_val = build_windows_split(df, feature_cols)

        if cache_windows:
            np.savez(cache_windows, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
            with open(meta_path, "w") as f:
                json.dump({"norm_stats": norm_stats, "feature_cols": feature_cols,
                           "stations": stations, "persistence_rmse": persistence_rmse,
                           "persistence_mae": persistence_mae}, f)
            print(f"Cached windows to {cache_windows}, metadata to {meta_path}")

    n_features = X_train.shape[2]
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | features: {n_features}")

    train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WindowDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    model = BestLSTM(n_features, hidden_size=hidden_size, num_layers=num_layers)
    if resume_from and os.path.exists(resume_from):
        ckpt = torch.load(resume_from, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        print(f"Resumed weights from {resume_from}")

    model, val_rmse = train_model(model, train_loader, val_loader, epochs, lr, device, patience)
    val_mae = evaluate_model(model, val_loader, device)[1]

    print(f"\n===== Final LSTM: val RMSE {val_rmse:.3f} | val MAE {val_mae:.3f} =====")
    print(f"(persistence baseline was RMSE {persistence_rmse:.3f} for reference)")

    torch.save({
        "state_dict": model.state_dict(),
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "feature_cols": feature_cols,
        "window_size": WINDOW_SIZE,
        "val_rmse": val_rmse,
        "val_mae": val_mae,
    }, save_path)
    print(f"Saved model to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", default="best_lstm.pt")
    parser.add_argument("--meta_path", default="preprocessing_meta.json")
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--cache_windows", default=None)
    args = parser.parse_args()
    main(args.data, args.epochs, args.batch_size, args.patience, args.hidden_size,
         args.num_layers, args.lr, args.save_path, args.meta_path,
         args.resume_from, args.cache_windows)
