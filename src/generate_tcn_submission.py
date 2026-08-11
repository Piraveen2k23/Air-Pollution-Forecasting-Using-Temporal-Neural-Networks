"""
generate_tcn_submission.py  —  Generate Kaggle submission using the trained TCN.

IMPORTANT: best_tcn.pt was trained on 48-hour windows (WINDOW=48).
           This script builds 48-hour windows from test_raw.csv to match.
           If you retrain the TCN on 24h windows, change WINDOW=24 below.
"""

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS — match whatever was used during training
# ─────────────────────────────────────────────────────────────────────────────
WINDOW        = 24     # must match what windowing.py used when training

NUM_CHANNELS  = 256    # updated to match best_tcn.pt
KERNEL_SIZE   = 5
NUM_LEVELS    = 4
DROPOUT       = 0.3    # updated to match best_tcn.pt (was 0.2)

MODEL_PATH    = "models/best_tcn.pt"
FITTED_PATH   = "data/processed/fitted_preprocessing.pkl"
FEATURE_PATH  = "data/processed/feature_cols.pkl"
OUTPUT_PATH   = "submissions/submission_tcn.csv"

TEST_RAW_CSV  = "data/raw/test_raw.csv"
TEST_CSV      = "data/raw/test.csv"

# ─────────────────────────────────────────────────────────────────────────────
# TCN model definition — must match train_tcn.py exactly
# ─────────────────────────────────────────────────────────────────────────────
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding_amount = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding_amount)

    def forward(self, x):
        out = self.conv(x)
        if self.padding_amount > 0:
            out = out[:, :, :-self.padding_amount]
        return out


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.causal1 = CausalConv1d(in_channels,  out_channels, kernel_size, dilation)
        self.causal2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm1   = nn.BatchNorm1d(out_channels)
        self.norm2   = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual_proj = (nn.Conv1d(in_channels, out_channels, kernel_size=1)
                              if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        out = self.causal1(x)
        out = self.dropout(F.relu(self.norm1(out)))
        out = self.causal2(out)
        out = self.dropout(F.relu(self.norm2(out)))
        return out + self.residual_proj(x)


class TCNForecaster(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, num_levels, dropout):
        super().__init__()
        blocks = []
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch    = input_size if i == 0 else num_channels
            blocks.append(TCNBlock(in_ch, num_channels, kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*blocks)
        self.fc  = nn.Linear(num_channels, 1)

    def forward(self, x):
        x   = x.transpose(1, 2)       # (batch, features, time)
        out = self.tcn(x)              # (batch, num_channels, time)
        return self.fc(out[:, :, -1]).squeeze(-1)   # last timestep → scalar

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing constants
# ─────────────────────────────────────────────────────────────────────────────
POLLUTANTS   = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
WEATHER      = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
NUMERIC_COLS = POLLUTANTS + WEATHER

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load everything
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print(f" TCN SUBMISSION GENERATOR  (WINDOW={WINDOW}h)")
print("=" * 60)

print("\n[1/6] Loading test data and fitted parameters ...")
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
print(f"  Features       : {INPUT_SIZE}")
print(f"  Window size    : {WINDOW}h")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Preprocess test data (same transforms as training)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/6] Preprocessing test data ...")

df = test_raw.copy()
df['month'] = df['datetime'].dt.month
df['hour']  = df['datetime'].dt.hour

# Local interpolation
for _, idx in df.groupby('station').groups.items():
    df.loc[idx, NUMERIC_COLS] = df.loc[idx, NUMERIC_COLS].interpolate(
        method='linear', limit=6, limit_direction='both'
    )

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

# Station one-hot
for s in station_list:
    df[f'st_{s}'] = (df['station'] == s).astype(np.float32)

# Final NaN check
nan_count = df[feature_cols].isna().sum().sum()
print(f"  NaNs after preprocessing: {nan_count}")
if nan_count > 0:
    df[feature_cols] = df[feature_cols].fillna(0)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Build sliding windows from test data
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[3/6] Building {WINDOW}-hour sliding windows ...")

Xs            = []
window_info   = []   # (station, target_datetime_ns)

for station in station_list:
    sub = df[df['station'] == station].sort_values('datetime').reset_index(drop=True)
    arr = sub[feature_cols].values.astype(np.float32)
    dts = sub['datetime'].values

    T = len(arr)
    if T <= WINDOW:
        print(f"  WARNING: station {station} has only {T} rows, skipping")
        continue

    windows = np.lib.stride_tricks.sliding_window_view(arr, window_shape=WINDOW, axis=0)
    windows = windows.transpose(0, 2, 1)    # (T-WINDOW+1, WINDOW, F)

    n = windows.shape[0] - 1               # drop last (no target hour)
    Xs.append(windows[:n])
    target_dts = dts[WINDOW:WINDOW + n]
    window_info.extend([(station, pd.Timestamp(dt).value) for dt in target_dts])

X_test_all = np.concatenate(Xs, axis=0)
lookup = {key: i for i, key in enumerate(window_info)}
print(f"  Total test windows: {X_test_all.shape[0]:,}  shape={X_test_all.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Load TCN and run predictions
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[4/6] Loading {MODEL_PATH} and running predictions ...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = TCNForecaster(INPUT_SIZE, NUM_CHANNELS, KERNEL_SIZE, NUM_LEVELS, DROPOUT).to(device)
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
      f"max={preds_raw.max():.1f}  mean={preds_raw.mean():.1f} ug/m3")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Match predictions to submission IDs
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] Matching predictions to submission IDs ...")

# Derive target datetime from the LAST context hour in test.csv + 1h
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
        pm25_preds.append(float(mean_pm25))   # fallback
        not_found += 1

matched = len(pm25_preds) - not_found
print(f"  Matched: {matched} / {len(pm25_preds)}", end="")
if not_found:
    print(f"  ({not_found} used fallback mean)")
else:
    print("  (all matched!)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Write submission CSV
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[6/6] Writing {OUTPUT_PATH} ...")

submission = pd.DataFrame({
    'id':    test_structured['id'].values,
    'PM2.5': pm25_preds,
})
submission.to_csv(OUTPUT_PATH, index=False)

print(f"\n  Rows   : {len(submission):,}")
print(f"  PM2.5  : min={submission['PM2.5'].min():.1f}  "
      f"max={submission['PM2.5'].max():.1f}  "
      f"mean={submission['PM2.5'].mean():.1f}")
print(f"\n  First 5 rows:")
print(submission.head().to_string(index=False))

print("\n" + "=" * 60)
print(f"  DONE!  Upload '{OUTPUT_PATH}' to Kaggle.")
print("=" * 60)
