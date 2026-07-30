"""
CO5420 - Generate Kaggle submission predictions from test.csv

test.csv (per the competition description) contains one row per 24-hour
input window, with columns flattened as <feature>_lag_24 ... <feature>_lag_1
(oldest to most recent), plus `id` and `station`.

This script:
  1. Re-derives normalisation stats + feature ordering from train_raw.csv
     (must match what the model was trained on - re-running
     load_and_preprocess on the same training file guarantees this).
  2. Reshapes each test row's lag columns back into a (24, n_features)
     sequence per the SAME feature order used in training.
  3. Loads the trained model checkpoint and runs inference.
  4. Writes id,PM2.5 to submission.csv in the exact sample_submission.csv format.

Usage:
    python predict_submission.py \
        --train_csv train_raw.csv \
        --test_csv test.csv \
        --checkpoint best_model_ffn.pt \
        --model_type ffn \
        --out submission.csv
"""

import argparse
import numpy as np
import pandas as pd
import torch

from pm25_forecasting_v2 import (
    load_and_preprocess, FeedForwardBaseline, RecurrentForecaster,
    NUMERIC_FEATURES, EXTRA_TEMPORAL_FEATURES, ROLLING_SOURCE_COLS,
    WINDOW_SIZE, COMPASS_TO_DEG,
)


def build_test_windows(test_df: pd.DataFrame, norm_stats, feature_cols, stations,
                        window_size=WINDOW_SIZE):
    """Reconstruct (N, window_size, n_features) arrays from test.csv's
    flattened <feature>_lag_K columns, applying the SAME normalisation,
    rolling/diff feature engineering, and station one-hot scheme used
    during training.

    Rolling/diff features are computed causally WITHIN each test window
    only (no data before lag_24 is available), matching how they were
    engineered in load_and_preprocess (min_periods=1 rolling)."""
    n = len(test_df)
    n_features = len(feature_cols)
    X = np.zeros((n, window_size, n_features), dtype=np.float32)
    station_cols = [c for c in feature_cols if c.startswith("st_")]

    # --- First pass: extract raw sequences for every source column that
    # a rolling/diff feature depends on (PM2.5, PM10, TEMP, WSPM, ...) ---
    source_cols = sorted(set(src for src, _, _ in ROLLING_SOURCE_COLS.values()))
    raw_seq = {}
    for src_col in source_cols:
        mu, _ = norm_stats[src_col]
        arr = np.zeros((n, window_size), dtype=np.float64)
        for t in range(window_size):
            lag = window_size - t
            col = f"{src_col}_lag_{lag}"
            arr[:, t] = test_df[col].astype(float).fillna(mu).values
        raw_seq[src_col] = arr

    # Causal rolling mean/std (min_periods=1, matches training) / first diff,
    # computed per row across the window's time axis, for every extra feature.
    extra_raw = {}
    for feat_name, (src_col, kind, k) in ROLLING_SOURCE_COLS.items():
        seq = raw_seq[src_col]
        out = np.zeros_like(seq)
        for t in range(window_size):
            if kind == "mean":
                out[:, t] = seq[:, max(0, t - k + 1):t + 1].mean(axis=1)
            elif kind == "std":
                sl = seq[:, max(0, t - k + 1):t + 1]
                out[:, t] = sl.std(axis=1) if sl.shape[1] > 1 else 0.0
            elif kind == "diff":
                out[:, t] = seq[:, t] - seq[:, t - 1] if t > 0 else 0.0
        extra_raw[feat_name] = out

    # --- Second pass: fill X in the exact column order of feature_cols ---
    for t in range(window_size):
        lag = window_size - t
        col_idx = 0

        for feat in NUMERIC_FEATURES:
            mu, sigma = norm_stats[feat]
            raw_col = f"{feat}_lag_{lag}"
            vals = test_df[raw_col].astype(float).fillna(mu)
            X[:, t, col_idx] = ((vals - mu) / sigma).values
            col_idx += 1

        wd_col = f"wd_lag_{lag}"
        if wd_col in test_df.columns:
            deg = test_df[wd_col].map(COMPASS_TO_DEG).fillna(0.0)
        else:
            deg = pd.Series(0.0, index=test_df.index)
        X[:, t, col_idx] = np.sin(np.deg2rad(deg)).values; col_idx += 1
        X[:, t, col_idx] = np.cos(np.deg2rad(deg)).values; col_idx += 1

        for cal_col, period in [("hour", 24), ("month", 12)]:
            lag_col = f"{cal_col}_lag_{lag}"
            if lag_col in test_df.columns:
                raw = test_df[lag_col].astype(float)
                shift = 1 if cal_col == "month" else 0
                ang = 2 * np.pi * (raw - shift) / period
                X[:, t, col_idx] = np.sin(ang).values; col_idx += 1
                X[:, t, col_idx] = np.cos(ang).values; col_idx += 1
            else:
                col_idx += 2

        dow_lag_col = f"dow_lag_{lag}"
        if dow_lag_col in test_df.columns:
            raw = test_df[dow_lag_col].astype(float)
            ang = 2 * np.pi * raw / 7
            X[:, t, col_idx] = np.sin(ang).values; col_idx += 1
            X[:, t, col_idx] = np.cos(ang).values; col_idx += 1
        else:
            col_idx += 2

        for feat in EXTRA_TEMPORAL_FEATURES:
            mu, sigma = norm_stats[feat]
            X[:, t, col_idx] = (extra_raw[feat][:, t] - mu) / sigma
            col_idx += 1

        for st_col in station_cols:
            st_name = st_col[len("st_"):]
            X[:, t, col_idx] = (test_df["station"] == st_name).astype(np.float32).values
            col_idx += 1

    return X


def main(train_csv, test_csv, checkpoint_path, model_type, out_path,
         hidden_size=64, num_layers=1):
    print("Re-deriving normalisation stats + feature order from training data...")
    train_df, feature_cols = load_and_preprocess(train_csv)
    norm_stats = train_df.attrs["norm_stats"]
    stations = sorted(train_df["station"].unique().tolist())

    print(f"Loading test data from {test_csv} ...")
    test_df = pd.read_csv(test_csv)

    print("Building test windows (matching training feature order)...")
    X_test = build_test_windows(test_df, norm_stats, feature_cols, stations)
    n_features = X_test.shape[2]

    print(f"Loading model checkpoint: {checkpoint_path} ({model_type})")
    if model_type == "ffn":
        model = FeedForwardBaseline(WINDOW_SIZE, n_features)
    else:
        model = RecurrentForecaster(n_features, hidden_size=hidden_size,
                                     num_layers=num_layers, cell=model_type)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print("Running inference...")
    with torch.no_grad():
        preds = model(torch.from_numpy(X_test)).numpy()

    submission = pd.DataFrame({"id": test_df["id"], "PM2.5": preds})
    submission.to_csv(out_path, index=False)
    print(f"Wrote {len(submission)} predictions to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", required=True, help="train_raw.csv (to re-derive norm stats)")
    parser.add_argument("--test_csv", required=True, help="test.csv (flattened lag windows)")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model .pt file")
    parser.add_argument("--model_type", choices=["ffn", "lstm", "gru"], default="ffn")
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()
    main(args.train_csv, args.test_csv, args.checkpoint, args.model_type,
         args.out, args.hidden_size, args.num_layers)
