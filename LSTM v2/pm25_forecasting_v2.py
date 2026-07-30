"""
CO5420 - Air Pollution Forecasting Using Temporal Neural Networks
Improved pipeline: baselines + FFN + LSTM + GRU for 1-hour-ahead PM2.5 prediction.

Feature engineering improvements over the first draft:
  - Cyclical (sin/cos) encoding of hour-of-day, day-of-week, month, and wind
    direction, instead of raw integers / arbitrary category codes.
  - Station identity as a one-hot feature so the model can learn per-station
    baseline pollution levels.
  - Chronological (not random) train/validation split, per station, so
    validation genuinely tests forecasting into the future.
  - Early stopping + LR scheduling so training time is spent where it helps.
  - Reports both RMSE (competition metric) and MAE (required in the report).

Usage:
    python pm25_forecasting_v2.py --data train_raw.csv --epochs 30
"""

import argparse
import copy
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
# PM2.5 gets the richest treatment (it's the target); PM10, TEMP, WSPM get a
# lighter rolling-mean treatment since they're informative but secondary.
EXTRA_TEMPORAL_FEATURES = [
    "PM2.5_roll_mean_3", "PM2.5_roll_mean_6", "PM2.5_roll_mean_12",
    "PM2.5_roll_std_6", "PM2.5_diff_1",
    "PM10_roll_mean_6", "TEMP_roll_mean_6", "WSPM_roll_mean_6",
]
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
WIND_DIR_COL = "wd"
STATION_COL = "station"
TARGET_COL = "PM2.5"
VAL_FRACTION = 0.15   # last 15% of each station's timeline held out, chronologically

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 16-point compass -> degrees, used for a proper circular sin/cos encoding
COMPASS_TO_DEG = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5,
    "SE": 135, "SSE": 157.5, "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}


# ---------------------------------------------------------------------------
# 1. Load, impute, engineer features
# ---------------------------------------------------------------------------
def load_and_preprocess(csv_path: str):
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.sort_values([STATION_COL, "datetime"]).reset_index(drop=True)

    # --- Missing value imputation ---
    # Short gaps: linear interpolation along time, per station (physically
    # sensible for pollutant/weather series). Remaining edge gaps: ffill/bfill.
    # Anything still missing (e.g. a whole run at the very start): station mean,
    # then global mean as a last resort.
    for col in NUMERIC_FEATURES:
        df[col] = df.groupby(STATION_COL)[col].transform(
            lambda s: s.interpolate(limit_direction="both")
        )
        df[col] = df.groupby(STATION_COL)[col].transform(lambda s: s.ffill().bfill())
        df[col] = df[col].fillna(df.groupby(STATION_COL)[col].transform("mean"))
        df[col] = df[col].fillna(df[col].mean())

    # Wind direction: fill missing with the station's most frequent direction,
    # then convert to an angle and encode circularly (0 deg and 360 deg are
    # the same direction, which a raw category index can't express).
    df[WIND_DIR_COL] = df.groupby(STATION_COL)[WIND_DIR_COL].transform(
        lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "N")
    )
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

    # --- Extra temporal features (rolling stats / diffs) ---
    # Deliberately computed only from information available WITHIN a single
    # 24-hour window (rolling stats with min_periods=1, and a first
    # difference), so predict_submission.py can reconstruct them identically
    # from test.csv's lagged columns without needing data before the window.
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

    # --- Normalise numeric features (z-score) ---
    norm_stats = {}
    for col in NUMERIC_FEATURES + EXTRA_TEMPORAL_FEATURES:
        mu, sigma = df[col].mean(), df[col].std() + 1e-8
        df[col + "_norm"] = (df[col] - mu) / sigma
        norm_stats[col] = (float(mu), float(sigma))

    feature_cols = (
        [c + "_norm" for c in NUMERIC_FEATURES]
        + ["wd_sin", "wd_cos", "hour_sin", "hour_cos",
           "month_sin", "month_cos", "dow_sin", "dow_cos"]
        + [c + "_norm" for c in EXTRA_TEMPORAL_FEATURES]
        + station_cols
    )
    df.attrs["norm_stats"] = norm_stats
    df.attrs["feature_cols"] = feature_cols
    return df, feature_cols


# ---------------------------------------------------------------------------
# 2. Windowing with a chronological (per-station) train/val split
# ---------------------------------------------------------------------------
def build_windows_split(df: pd.DataFrame, feature_cols, window_size=WINDOW_SIZE,
                         val_fraction=VAL_FRACTION):
    X_train, y_train, X_val, y_val = [], [], [], []

    for _, group in df.groupby(STATION_COL):
        group = group.reset_index(drop=True)
        feats = group[feature_cols].values.astype(np.float32)
        target = group[TARGET_COL].values.astype(np.float32)
        n = len(group)

        # Split this station's timeline chronologically BEFORE windowing,
        # so no validation window's target (or input) leaks from the future
        # into training, and validation genuinely tests forecasting ability.
        split_point = int(n * (1 - val_fraction))

        for start in range(n - window_size):
            end = start + window_size
            target_idx = end
            x_win, y_val_ = feats[start:end], target[target_idx]
            if target_idx < split_point:
                X_train.append(x_win)
                y_train.append(y_val_)
            else:
                X_val.append(x_win)
                y_val.append(y_val_)

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


# ---------------------------------------------------------------------------
# 3. Baselines
# ---------------------------------------------------------------------------
def persistence_baseline(df: pd.DataFrame, val_fraction=VAL_FRACTION):
    """Predict next hour = current hour's PM2.5, evaluated only on the same
    chronological validation region used for the neural models (fair comparison)."""
    preds, actuals = [], []
    for _, group in df.groupby(STATION_COL):
        pm = group[TARGET_COL].values
        n = len(pm)
        split_point = int(n * (1 - val_fraction))
        # target index i uses pm[i-1] as the persistence prediction
        for i in range(max(split_point, 1), n):
            preds.append(pm[i - 1])
            actuals.append(pm[i])
    rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
    mae = float(mean_absolute_error(actuals, preds))
    return rmse, mae


# ---------------------------------------------------------------------------
# 4. Models
# ---------------------------------------------------------------------------
class FeedForwardBaseline(nn.Module):
    def __init__(self, window_size, n_features, hidden=256):
        super().__init__()
        in_dim = window_size * n_features
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RecurrentForecaster(nn.Module):
    """Shared LSTM/GRU forecaster - cell type is selectable."""
    def __init__(self, n_features, hidden_size=96, num_layers=2,
                 dropout=0.2, cell="lstm"):
        super().__init__()
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.cell = cell
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, hidden = self.rnn(x)
        h_n = hidden[0] if self.cell == "lstm" else hidden
        last_hidden = h_n[-1]
        return self.head(last_hidden).squeeze(-1)


# ---------------------------------------------------------------------------
# 5. Training with early stopping + LR scheduling
# ---------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs=30, lr=1e-3,
                 device="cpu", patience=5, model_name="model"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
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
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[{model_name}] epoch {epoch:2d}/{epochs} | train MSE: {train_loss:8.3f} "
              f"| val RMSE: {val_rmse:7.3f} | val MAE: {val_mae:7.3f} | lr: {current_lr:.2e}")

        if val_rmse < best_val_rmse - 1e-4:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[{model_name}] early stopping at epoch {epoch} "
                      f"(best val RMSE: {best_val_rmse:.3f})")
                break

    model.load_state_dict(best_state)
    return model, best_val_rmse


@torch.no_grad()
def evaluate_model(model, loader, device="cpu"):
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
# 6. Orchestration
# ---------------------------------------------------------------------------
def main(csv_path, epochs=30, batch_size=256, patience=5, save_path=None,
         hidden_size=96, num_layers=2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    print("[1/6] Loading and preprocessing data...")
    df, feature_cols = load_and_preprocess(csv_path)
    print(f"    {len(feature_cols)} features per timestep: {feature_cols[:8]} ... (+ station one-hot)")

    print("[2/6] Persistence baseline (on chronological validation region)...")
    p_rmse, p_mae = persistence_baseline(df)
    print(f"    Persistence  RMSE: {p_rmse:.3f} | MAE: {p_mae:.3f}")

    print("[3/6] Building 24-hour windows with chronological split...")
    X_train, y_train, X_val, y_val = build_windows_split(df, feature_cols)
    n_features = X_train.shape[2]
    print(f"    Train windows: {X_train.shape} | Val windows: {X_val.shape}")

    train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WindowDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    print("\n[4/6] Training feedforward baseline...")
    ffn = FeedForwardBaseline(WINDOW_SIZE, n_features)
    ffn, ffn_rmse = train_model(ffn, train_loader, val_loader, epochs=epochs,
                                 device=device, patience=patience, model_name="FFN")
    ffn_mae = evaluate_model(ffn, val_loader, device)[1]

    print("\n[5/6] Training LSTM forecaster...")
    lstm = RecurrentForecaster(n_features, hidden_size=hidden_size, num_layers=num_layers, cell="lstm")
    lstm, lstm_rmse = train_model(lstm, train_loader, val_loader, epochs=epochs,
                                    device=device, patience=patience, model_name="LSTM")
    lstm_mae = evaluate_model(lstm, val_loader, device)[1]

    print("\n[6/6] Training GRU forecaster...")
    gru = RecurrentForecaster(n_features, hidden_size=hidden_size, num_layers=num_layers, cell="gru")
    gru, gru_rmse = train_model(gru, train_loader, val_loader, epochs=epochs,
                                  device=device, patience=patience, model_name="GRU")
    gru_mae = evaluate_model(gru, val_loader, device)[1]

    print("\n===== Summary (chronological validation, lower is better) =====")
    print(f"{'Model':<14}{'RMSE':>10}{'MAE':>10}")
    print(f"{'Persistence':<14}{p_rmse:>10.3f}{p_mae:>10.3f}")
    print(f"{'FeedForward':<14}{ffn_rmse:>10.3f}{ffn_mae:>10.3f}")
    print(f"{'LSTM':<14}{lstm_rmse:>10.3f}{lstm_mae:>10.3f}")
    print(f"{'GRU':<14}{gru_rmse:>10.3f}{gru_mae:>10.3f}")

    results = {"persistence": (p_rmse, p_mae), "ffn": (ffn_rmse, ffn_mae),
               "lstm": (lstm_rmse, lstm_mae), "gru": (gru_rmse, gru_mae)}
    best_name = min(["ffn", "lstm", "gru"], key=lambda k: results[k][0])
    best_model = {"ffn": ffn, "lstm": lstm, "gru": gru}[best_name]
    print(f"\nBest model: {best_name.upper()} (val RMSE {results[best_name][0]:.3f})")

    if save_path:
        torch.save({
            "model_name": best_name,
            "state_dict": best_model.state_dict(),
            "feature_cols": feature_cols,
            "norm_stats": df.attrs["norm_stats"],
            "window_size": WINDOW_SIZE,
        }, save_path)
        print(f"Saved best model checkpoint to {save_path}")

    return results, best_name, best_model, df, feature_cols


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--save_path", type=str, default="best_model.pt")
    parser.add_argument("--hidden_size", type=int, default=96)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--model_type", type=str, default="all",
                         choices=["all", "ffn", "lstm", "gru"],
                         help="Train only one model type (for splitting long runs across calls)")
    parser.add_argument("--cache_windows", type=str, default=None,
                         help="Path to .npz cache of pre-built windows, to skip rebuilding each call")
    parser.add_argument("--results_json", type=str, default="results.json",
                         help="Append this run's result to a JSON file")
    parser.add_argument("--resume_from", type=str, default=None,
                         help="Checkpoint path to warm-start model weights from")
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if args.model_type == "all":
        main(args.data, epochs=args.epochs, batch_size=args.batch_size,
             patience=args.patience, save_path=args.save_path,
             hidden_size=args.hidden_size, num_layers=args.num_layers)
    else:
        import json, os, time
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        if args.cache_windows and os.path.exists(args.cache_windows):
            print("Loading cached windows...")
            cache = np.load(args.cache_windows)
            X_train, y_train, X_val, y_val = (cache["X_train"], cache["y_train"],
                                               cache["X_val"], cache["y_val"])
        else:
            print("Loading, preprocessing, windowing (will cache for next runs)...")
            df, feature_cols = load_and_preprocess(args.data)
            X_train, y_train, X_val, y_val = build_windows_split(df, feature_cols)
            if args.cache_windows:
                np.savez(args.cache_windows, X_train=X_train, y_train=y_train,
                         X_val=X_val, y_val=y_val)
                print(f"Cached windows to {args.cache_windows}")

        n_features = X_train.shape[2]
        print(f"Train: {X_train.shape} | Val: {X_val.shape}")
        train_loader = DataLoader(WindowDataset(X_train, y_train),
                                   batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(WindowDataset(X_val, y_val),
                                 batch_size=args.batch_size, shuffle=False)

        if args.model_type == "ffn":
            model = FeedForwardBaseline(WINDOW_SIZE, n_features)
        else:
            model = RecurrentForecaster(n_features, hidden_size=args.hidden_size,
                                         num_layers=args.num_layers, cell=args.model_type)

        if args.resume_from and os.path.exists(args.resume_from):
            ckpt = torch.load(args.resume_from, map_location="cpu")
            model.load_state_dict(ckpt["state_dict"])
            print(f"Resumed weights from {args.resume_from}")

        t0 = time.time()
        model, val_rmse = train_model(model, train_loader, val_loader, epochs=args.epochs,
                                       lr=args.lr, device=device, patience=args.patience,
                                       model_name=args.model_type.upper())
        val_mae = evaluate_model(model, val_loader, device)[1]
        elapsed = time.time() - t0

        ckpt_path = args.save_path.replace(".pt", f"_{args.model_type}.pt")
        torch.save({"model_name": args.model_type, "state_dict": model.state_dict(),
                    "hidden_size": args.hidden_size, "num_layers": args.num_layers,
                    "window_size": WINDOW_SIZE}, ckpt_path)
        print(f"Saved {args.model_type} checkpoint to {ckpt_path} (took {elapsed:.0f}s)")

        results = {}
        if os.path.exists(args.results_json):
            with open(args.results_json) as f:
                results = json.load(f)
        results[args.model_type] = {"rmse": val_rmse, "mae": val_mae, "seconds": elapsed}
        with open(args.results_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Updated {args.results_json}: {results}")
