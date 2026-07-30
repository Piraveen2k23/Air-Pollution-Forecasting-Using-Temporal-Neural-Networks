"""
CO5420 - Generate Kaggle submission from the best LSTM model (best_lstm.py).

Uses the SAME per-station normalisation, rolling/diff features, and feature
order that best_lstm.py trained with (loaded from preprocessing_meta.json,
produced during training).

Usage:
    python predict_best_lstm.py \
        --meta_path preprocessing_meta.json \
        --test_csv test.csv \
        --checkpoint best_lstm.pt \
        --out submission_lstm.csv
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch

from best_lstm import BestLSTM, WINDOW_SIZE, NUMERIC_FEATURES, \
    EXTRA_TEMPORAL_FEATURES, ROLLING_SOURCE_COLS, COMPASS_TO_DEG


def build_test_windows(test_df, norm_stats, feature_cols, window_size=WINDOW_SIZE):
    """norm_stats here is PER-STATION: {station: {feature: (mu, sigma)}}.
    Each test row's own `station` column picks which stats to apply."""
    n = len(test_df)
    n_features = len(feature_cols)
    X = np.zeros((n, window_size, n_features), dtype=np.float32)
    station_cols = [c for c in feature_cols if c.startswith("st_")]
    station_names = [c[len("st_"):] for c in station_cols]

    # Per-row mu/sigma arrays for every normalised column, looked up via station.
    def mu_sigma_arrays(col):
        mus = test_df[STATION_COL_GUESS].map(lambda s: norm_stats[s][col][0]).values
        sigmas = test_df[STATION_COL_GUESS].map(lambda s: norm_stats[s][col][1]).values
        return mus, sigmas

    STATION_COL_GUESS = "station"

    # --- Raw sequences for rolling-feature source columns ---
    source_cols = sorted(set(src for src, _, _ in ROLLING_SOURCE_COLS.values()))
    raw_seq = {}
    for src_col in source_cols:
        arr = np.zeros((n, window_size), dtype=np.float64)
        mus = test_df[STATION_COL_GUESS].map(lambda s: norm_stats[s][src_col][0]).values
        for t in range(window_size):
            lag = window_size - t
            col = f"{src_col}_lag_{lag}"
            vals = test_df[col].astype(float).values.copy()
            nan_mask = np.isnan(vals)
            vals[nan_mask] = mus[nan_mask]
            arr[:, t] = vals
        raw_seq[src_col] = arr

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

    for t in range(window_size):
        lag = window_size - t
        col_idx = 0

        for feat in NUMERIC_FEATURES:
            mus, sigmas = mu_sigma_arrays(feat)
            raw_col = f"{feat}_lag_{lag}"
            vals = test_df[raw_col].astype(float).values.copy()
            nan_mask = np.isnan(vals)
            vals[nan_mask] = mus[nan_mask]
            X[:, t, col_idx] = (vals - mus) / sigmas
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
            mus, sigmas = mu_sigma_arrays(feat)
            X[:, t, col_idx] = (extra_raw[feat][:, t] - mus) / sigmas
            col_idx += 1

        for st_col, st_name in zip(station_cols, station_names):
            X[:, t, col_idx] = (test_df["station"] == st_name).astype(np.float32).values
            col_idx += 1

    return X


def main(meta_path, test_csv, checkpoint_path, out_path):
    print(f"Loading preprocessing metadata from {meta_path} ...")
    with open(meta_path) as f:
        meta = json.load(f)
    norm_stats = meta["norm_stats"]
    feature_cols = meta["feature_cols"]

    print(f"Loading test data from {test_csv} ...")
    test_df = pd.read_csv(test_csv)

    print("Building test windows (per-station normalisation)...")
    X_test = build_test_windows(test_df, norm_stats, feature_cols)
    n_features = X_test.shape[2]

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = BestLSTM(n_features, hidden_size=ckpt["hidden_size"], num_layers=ckpt["num_layers"])
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
    parser.add_argument("--meta_path", default="preprocessing_meta.json")
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="submission_lstm.csv")
    args = parser.parse_args()
    main(args.meta_path, args.test_csv, args.checkpoint, args.out)
