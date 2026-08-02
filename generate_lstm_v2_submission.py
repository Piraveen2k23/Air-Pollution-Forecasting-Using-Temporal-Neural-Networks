"""
generate_lstm_v2_submission.py  —  Generate Kaggle submission from LSTM v2 model.

WHAT THIS DOES:
  1. Loads test_raw.csv
  2. Applies the SAME feature engineering as feature_engineering.py
     (lag features, rolling stats, day-of-week, 48h windows)
  3. Loads best_lstm_v2_seed0.pt
  4. Runs predictions
  5. Matches predictions to test.csv row IDs
  6. Saves submission_lstm_v2.csv

IMPORTANT: Must be run AFTER train_lstm_v2.py has produced best_lstm_v2_seed0.pt
"""

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS — must match train_lstm_v2.py exactly
# ─────────────────────────────────────────────────────────────────────────────
WINDOW        = 48
HIDDEN_SIZE   = 256
NUM_LAYERS    = 3
DROPOUT       = 0.3

MODEL_PATH    = "best_lstm_v2_seed0.pt"
FITTED_PATH   = "fitted_preprocessing.pkl"
FEATURE_PATH  = "feature_cols_v2.pkl"
OUTPUT_PATH   = "submission_lstm_v2.csv"

TEST_RAW_CSV  = (r"c:\Users\abdul\.cache\kagglehub\competitions"
                 r"\co-5420-air-pollution-forecasting-using-temporal-n-ns"
                 r"\CO5420-AirPollution\public\test_raw.csv")
TEST_CSV      = (r"c:\Users\abdul\.cache\kagglehub\competitions"
                 r"\co-5420-air-pollution-forecasting-using-temporal-n-ns"
                 r"\CO5420-AirPollution\public\test.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Model definition — must match train_lstm_v2.py exactly
# ─────────────────────────────────────────────────────────────────────────────
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_out):
        scores  = self.score(lstm_out)
        weights = torch.softmax(scores, dim=1)
        return (weights * lstm_out).sum(dim=1)


class LSTMv2Forecaster(nn.Module):
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
        out, _  = self.lstm(x)
        context = self.attention(out)
        context = self.dropout(context)
        return self.head(context).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing constants (must match preprocessing.py)
# ─────────────────────────────────────────────────────────────────────────────
POLLUTANTS   = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
WEATHER      = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
NUMERIC_COLS = POLLUTANTS + WEATHER

print("=" * 60)
print(f" LSTM v2 SUBMISSION GENERATOR  (WINDOW={WINDOW}h)")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load everything
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/7] Loading test data and fitted parameters ...")
test_raw = pd.read_csv(TEST_RAW_CSV)
test_raw['datetime'] = pd.to_datetime(test_raw[['year', 'month', 'day', 'hour']])
test_raw = test_raw.sort_values(['station', 'datetime']).reset_index(drop=True)

with open(FITTED_PATH, 'rb') as f:
    fitted = pickle.load(f)
seasonal_lookup = fitted['seasonal_lookup']
global_medians  = fitted['global_medians']
wd_mode         = fitted['wd_mode']
means           = fitted['means']
stds            = fitted['stds']
deg_map         = fitted['deg_map']
mean_pm25       = float(means['PM2.5'])
std_pm25        = float(stds['PM2.5'])

with open(FEATURE_PATH, 'rb') as f:
    meta = pickle.load(f)
feature_cols = meta['feature_cols']
station_list = meta['station_list']
INPUT_SIZE   = len(feature_cols)

test_structured = pd.read_csv(TEST_CSV)
print(f"  test_raw rows  : {len(test_raw):,}")
print(f"  submission rows: {len(test_structured):,}")
print(f"  Features       : {INPUT_SIZE}  |  Window: {WINDOW}h")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Preprocess test data (same transforms as training)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/7] Preprocessing test data ...")
df = test_raw.copy()
df['month'] = df['datetime'].dt.month
df['hour']  = df['datetime'].dt.hour

# Local interpolation
for _, idx in df.groupby('station').groups.items():
    df.loc[idx, NUMERIC_COLS] = df.loc[idx, NUMERIC_COLS].interpolate(
        method='linear', limit=6, limit_direction='both')

# Seasonal fill
merged = df.merge(seasonal_lookup, on=['station', 'month', 'hour'],
                  suffixes=('', '_ssn'), how='left')
for c in NUMERIC_COLS:
    ssn = f'{c}_ssn'
    if ssn in merged.columns:
        df[c] = np.where(df[c].isna(), merged[ssn].values, df[c].values)

# Global median fill
df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(global_medians)

# Wind direction → sin/cos
df['wd'] = df['wd'].fillna(df['station'].map(wd_mode))
rad = np.deg2rad(df['wd'].map(deg_map))
df['wd_sin'] = np.sin(rad)
df['wd_cos'] = np.cos(rad)

# Normalize
for c in NUMERIC_COLS:
    df[c + '_norm'] = (df[c] - means[c]) / stds[c]

# Calendar cyclical
df['hour_sin']  = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos']  = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)

# Day-of-week cyclical + weekend
dow = df['datetime'].dt.dayofweek
df['dow_sin']    = np.sin(2 * np.pi * dow / 7)
df['dow_cos']    = np.cos(2 * np.pi * dow / 7)
df['is_weekend'] = (dow >= 5).astype(np.float32)

# Station one-hot
for s in station_list:
    df[f'st_{s}'] = (df['station'] == s).astype(np.float32)

print("\n[3/7] Adding lag/rolling features to test data ...")
LAG_HOURS   = [1, 2, 3, 6, 12, 24]
ROLL_MEAN_W = [3, 6, 12, 24]
ROLL_STD_W  = [3, 6, 12]
ROLL_MAX_W  = [6, 24]

new_lag_cols = (
    [f'pm25_lag{h}'   for h in LAG_HOURS]  +
    [f'pm25_rmean{w}' for w in ROLL_MEAN_W] +
    [f'pm25_rstd{w}'  for w in ROLL_STD_W]  +
    [f'pm25_rmax{w}'  for w in ROLL_MAX_W]
)
for col in new_lag_cols:
    df[col] = 0.0

for station in station_list:
    mask = df['station'] == station
    pm25 = df.loc[mask, 'PM2.5_norm']
    for h in LAG_HOURS:
        df.loc[mask, f'pm25_lag{h}'] = pm25.shift(h).fillna(0).values
    for w in ROLL_MEAN_W:
        df.loc[mask, f'pm25_rmean{w}'] = pm25.rolling(w, min_periods=1).mean().values
    for w in ROLL_STD_W:
        df.loc[mask, f'pm25_rstd{w}'] = pm25.rolling(w, min_periods=2).std().fillna(0).values
    for w in ROLL_MAX_W:
        df.loc[mask, f'pm25_rmax{w}'] = pm25.rolling(w, min_periods=1).max().values

# Final NaN check
nan_count = df[feature_cols].isna().sum().sum()
print(f"  NaNs after preprocessing: {nan_count}")
if nan_count > 0:
    df[feature_cols] = df[feature_cols].fillna(0)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Build 48-hour sliding windows from test data
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[4/7] Building {WINDOW}-hour sliding windows ...")
Xs, window_info = [], []

for station in station_list:
    sub = df[df['station'] == station].sort_values('datetime').reset_index(drop=True)
    arr = sub[feature_cols].values.astype(np.float32)
    dts = sub['datetime'].values

    T = len(arr)
    if T <= WINDOW:
        print(f"  WARNING: station {station} too short, skipping")
        continue

    wins = np.lib.stride_tricks.sliding_window_view(arr, window_shape=WINDOW, axis=0)
    wins = wins.transpose(0, 2, 1)

    n = wins.shape[0] - 1
    Xs.append(wins[:n])
    target_dts = dts[WINDOW:WINDOW + n]
    window_info.extend([(station, pd.Timestamp(dt).value) for dt in target_dts])

X_test_all = np.concatenate(Xs, axis=0)
lookup = {key: i for i, key in enumerate(window_info)}
print(f"  Total test windows: {X_test_all.shape[0]:,}  shape={X_test_all.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Load LSTM v2 and predict
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[5/7] Loading {MODEL_PATH} and running predictions ...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = LSTMv2Forecaster(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

X_tensor   = torch.tensor(X_test_all, dtype=torch.float32)
preds_norm = []
BATCH      = 1024

with torch.no_grad():
    for i in range(0, len(X_tensor), BATCH):
        out = model(X_tensor[i:i + BATCH].to(device)).cpu().numpy()
        preds_norm.append(out)

preds_norm = np.concatenate(preds_norm)
preds_raw  = np.clip(preds_norm * std_pm25 + mean_pm25, 0, None)

print(f"  Predictions: min={preds_raw.min():.1f}  "
      f"max={preds_raw.max():.1f}  mean={preds_raw.mean():.1f} μg/m³")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Match predictions to submission IDs
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/7] Matching predictions to submission IDs ...")
cols       = list(test_structured.columns)
year_cols  = [c for c in cols if c.startswith('year')]
month_cols = [c for c in cols if c.startswith('month')]
day_cols   = [c for c in cols if c.startswith('day')]
hour_cols  = [c for c in cols if c.startswith('hour')]

last_dt   = pd.to_datetime({
    'year':  test_structured[year_cols[-1]],
    'month': test_structured[month_cols[-1]],
    'day':   test_structured[day_cols[-1]],
    'hour':  test_structured[hour_cols[-1]],
})
target_dt = last_dt + pd.Timedelta(hours=1)

pm25_preds = []
not_found  = 0

for i in range(len(test_structured)):
    st  = test_structured['station'].iloc[i]
    key = (st, target_dt.iloc[i].value)
    idx = lookup.get(key)
    if idx is not None:
        pm25_preds.append(float(preds_raw[idx]))
    else:
        pm25_preds.append(float(mean_pm25))
        not_found += 1

matched = len(pm25_preds) - not_found
print(f"  Matched: {matched} / {len(pm25_preds)}", end="")
if not_found:
    print(f"  ({not_found} used fallback mean)")
else:
    print("  (all matched!)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Write submission CSV
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[7/7] Writing {OUTPUT_PATH} ...")
submission = pd.DataFrame({
    'id':    test_structured['id'].values,
    'PM2.5': pm25_preds,
})
submission.to_csv(OUTPUT_PATH, index=False)

print(f"\n  Rows  : {len(submission):,}")
print(f"  PM2.5 : min={submission['PM2.5'].min():.1f}  "
      f"max={submission['PM2.5'].max():.1f}  "
      f"mean={submission['PM2.5'].mean():.1f}")
print(f"\n  First 5 rows:")
print(submission.head().to_string(index=False))

print("\n" + "=" * 60)
print(f"  DONE!  Upload '{OUTPUT_PATH}' to Kaggle.")
print(f"  Target: beat 14.878 → aim for < 13.0")
print("=" * 60)
