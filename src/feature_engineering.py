"""
feature_engineering.py  —  Build enriched 48-hour windows with lag/rolling features.

WHAT THIS SCRIPT DOES:
  Reads the already-cleaned DataFrames (train_clean.pkl, val_clean.pkl),
  adds 17 new engineered features (lags, rolling stats, day-of-week),
  then builds sliding 48-hour windows and saves them to windows_v2.npz.

CURRENT INPUT (windows.npz):
  X_train shape: (262944, 24, 29)   — 24 hours, 29 features per hour

NEW OUTPUT (windows_v2.npz):
  X_train shape: (~250000, 48, 46)  — 48 hours, 46 features per hour

NEW FEATURES ADDED (17 total):
  Lag PM2.5:      pm25_lag1, lag2, lag3, lag6, lag12, lag24
  Rolling mean:   pm25_rmean3, rmean6, rmean12, rmean24
  Rolling std:    pm25_rstd3, rstd6, rstd12
  Rolling max:    pm25_rmax6, rmax24
  Day-of-week:    dow_sin, dow_cos, is_weekend

Run order:
    1. python feature_engineering.py   ← this script
    2. python train_lstm_v2.py
    3. python generate_lstm_v2_submission.py
"""

import numpy as np
import pandas as pd
import pickle

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
WINDOW      = 48          # 48-hour context window (was 24)
OUTPUT_NPZ  = "data/processed/windows_v2.npz"
OUTPUT_PKL  = "feature_cols_v2.pkl"

POLLUTANTS_NORM = ['PM2.5_norm', 'PM10_norm', 'SO2_norm', 'NO2_norm', 'CO_norm', 'O3_norm']
WEATHER_NORM    = ['TEMP_norm', 'PRES_norm', 'DEWP_norm', 'RAIN_norm', 'WSPM_norm']
WIND_COLS       = ['wd_sin', 'wd_cos']

print("=" * 65)
print(" FEATURE ENGINEERING v2")
print(f" 48h windows + lag/rolling features → {OUTPUT_NPZ}")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load cleaned DataFrames
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/5] Loading train_clean.pkl and val_clean.pkl ...")
train_df = pd.read_pickle("data/processed/train_clean.pkl")
val_df   = pd.read_pickle("data/processed/val_clean.pkl")

print(f"  train_df : {train_df.shape}")
print(f"  val_df   : {val_df.shape}")
print(f"  Columns  : {list(train_df.columns[:8])} ...")

station_list = sorted(train_df['station'].unique())
print(f"  Stations : {station_list}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Add engineered features per-station (no cross-station leakage)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Adding lag / rolling / calendar features ...")

def add_features(df: pd.DataFrame, station_list: list):
    """
    Adds 17 new features to the DataFrame.
    All lag/rolling are computed PER STATION to avoid leakage.
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])

    # ── Calendar cyclical ────────────────────────────────────────────────────
    df['hour_sin']   = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']   = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin']  = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos']  = np.cos(2 * np.pi * (df['month'] - 1) / 12)

    # Day-of-week cyclical + weekend flag
    dow = df['datetime'].dt.dayofweek   # 0=Mon … 6=Sun
    df['dow_sin']    = np.sin(2 * np.pi * dow / 7)
    df['dow_cos']    = np.cos(2 * np.pi * dow / 7)
    df['is_weekend'] = (dow >= 5).astype(np.float32)

    # ── Column names for new features ───────────────────────────────────────
    LAG_HOURS    = [1, 2, 3, 6, 12, 24]
    ROLL_MEAN_W  = [3, 6, 12, 24]
    ROLL_STD_W   = [3, 6, 12]
    ROLL_MAX_W   = [6, 24]

    new_cols = (
        [f'pm25_lag{h}'    for h in LAG_HOURS]   +
        [f'pm25_rmean{w}'  for w in ROLL_MEAN_W] +
        [f'pm25_rstd{w}'   for w in ROLL_STD_W]  +
        [f'pm25_rmax{w}'   for w in ROLL_MAX_W]
    )

    # Initialise all new columns with 0 (will be filled per-station below)
    for col in new_cols:
        df[col] = 0.0

    # ── Per-station computation ──────────────────────────────────────────────
    for station in station_list:
        mask = df['station'] == station
        pm25 = df.loc[mask, 'PM2.5_norm']

        # Lag features — shift by h hours within the station series
        for h in LAG_HOURS:
            df.loc[mask, f'pm25_lag{h}'] = pm25.shift(h).fillna(0).values

        # Rolling mean — average over past w hours
        for w in ROLL_MEAN_W:
            df.loc[mask, f'pm25_rmean{w}'] = (
                pm25.rolling(w, min_periods=1).mean().values
            )

        # Rolling std — how volatile the last w hours were
        for w in ROLL_STD_W:
            df.loc[mask, f'pm25_rstd{w}'] = (
                pm25.rolling(w, min_periods=2).std().fillna(0).values
            )

        # Rolling max — worst value in the last w hours
        for w in ROLL_MAX_W:
            df.loc[mask, f'pm25_rmax{w}'] = (
                pm25.rolling(w, min_periods=1).max().values
            )

    return df, new_cols


train_df, new_feat_cols = add_features(train_df, station_list)
val_df,   _             = add_features(val_df,   station_list)

print(f"  New features added: {len(new_feat_cols)}")
print(f"  Example new cols  : {new_feat_cols[:6]}")

# ── Station one-hot ──────────────────────────────────────────────────────────
for s in station_list:
    train_df[f'st_{s}'] = (train_df['station'] == s).astype(np.float32)
    val_df[f'st_{s}']   = (val_df['station']   == s).astype(np.float32)
station_onehot_cols = [f'st_{s}' for s in station_list]

# ── Final feature list (old base + new lag/rolling + station one-hot) ────────
base_feats = (POLLUTANTS_NORM + WEATHER_NORM + WIND_COLS +
              ['hour_sin', 'hour_cos', 'month_sin', 'month_cos',
               'dow_sin', 'dow_cos', 'is_weekend'])
feature_cols_v2 = base_feats + new_feat_cols + station_onehot_cols

print(f"\n  TOTAL features: {len(feature_cols_v2)}  (was 29)")
print(f"    Base features : {len(base_feats)}")
print(f"    Lag/rolling   : {len(new_feat_cols)}")
print(f"    Station OH    : {len(station_onehot_cols)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Build 48-hour sliding windows
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[3/5] Building {WINDOW}-hour sliding windows ...")

def build_windows(df, station_list, feature_cols, window=WINDOW):
    Xs, y_norms, y_raws, dts, stations_out = [], [], [], [], []
    for station in station_list:
        sub = df[df['station'] == station].sort_values('datetime').reset_index(drop=True)
        arr   = sub[feature_cols].values.astype(np.float32)
        tnorm = sub['PM2.5_norm'].values.astype(np.float32)
        traw  = sub['PM2.5'].values.astype(np.float32)
        dtime = sub['datetime'].values

        T = arr.shape[0]
        if T <= window:
            print(f"  WARNING: station {station} too short ({T} rows), skipping")
            continue

        # sliding_window_view gives shape (T-window+1, F, window), transpose → (T-window+1, window, F)
        wins = np.lib.stride_tricks.sliding_window_view(arr, window_shape=window, axis=0)
        wins = wins.transpose(0, 2, 1)

        n = wins.shape[0] - 1   # drop last (no next-hour target available)
        Xs.append(wins[:n])
        y_norms.append(tnorm[window:window + n])
        y_raws.append(traw[window:window + n])
        dts.append(dtime[window:window + n])
        stations_out.append(np.full(n, station))

    return (np.concatenate(Xs),
            np.concatenate(y_norms),
            np.concatenate(y_raws),
            np.concatenate(dts),
            np.concatenate(stations_out))


X_train, y_train_norm, y_train_raw, dt_train, st_train = build_windows(
    train_df, station_list, feature_cols_v2)
X_val, y_val_norm, y_val_raw, dt_val, st_val = build_windows(
    val_df, station_list, feature_cols_v2)

print(f"  X_train : {X_train.shape}  ({X_train.shape[0]:,} windows)")
print(f"  X_val   : {X_val.shape}  ({X_val.shape[0]:,} windows)")

nan_train = np.isnan(X_train).sum()
nan_val   = np.isnan(X_val).sum()
print(f"  NaNs in X_train: {nan_train}  X_val: {nan_val}")
if nan_train > 0 or nan_val > 0:
    print("  Filling any residual NaNs with 0 ...")
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val   = np.nan_to_num(X_val,   nan=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Save windows_v2.npz
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[4/5] Saving {OUTPUT_NPZ} ...")
np.savez_compressed(
    OUTPUT_NPZ,
    X_train=X_train, y_train_norm=y_train_norm, y_train_raw=y_train_raw,
    dt_train=dt_train.astype('datetime64[ns]').astype(np.int64), st_train=st_train,
    X_val=X_val,   y_val_norm=y_val_norm,   y_val_raw=y_val_raw,
    dt_val=dt_val.astype('datetime64[ns]').astype(np.int64),     st_val=st_val,
)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Save feature_cols_v2.pkl
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[5/5] Saving {OUTPUT_PKL} ...")
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump({
        "feature_cols": feature_cols_v2,
        "station_list": station_list,
        "window":       WINDOW,
        "new_feat_cols": new_feat_cols,
    }, f)

print("\n" + "=" * 65)
print("  DONE!")
print(f"  {OUTPUT_NPZ}  →  {X_train.shape[0]:,} train + {X_val.shape[0]:,} val windows")
print(f"  {OUTPUT_PKL}  →  {len(feature_cols_v2)} features")
print(f"\n  Next: python train_lstm_v2.py")
print("=" * 65)
