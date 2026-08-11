"""
generate_submission.py — Step 8: Generate Kaggle submission using the trained LSTM.

HOW IT WORKS:
  The test set (test.csv) contains 4,103 rows. Each row has an ID (e.g. test_00001)
  and 24 hours of pre-built context (all features for hours t-23 through t-1).
  We feed each row's 24-hour window into the trained LSTM to predict PM2.5 at hour t.

  Pipeline:
    1. Load test.csv  (the IDs + raw 24-hour windows)
    2. Apply the SAME preprocessing transforms used on training data:
         - Fill any missing values with seasonal lookup / global median
         - Encode wind direction as sin/cos
         - Normalize with training means/stds
    3. Reshape into (N, 24, 29) tensors  [same shape the LSTM expects]
    4. Run the LSTM model  (no training, just forward pass)
    5. Un-normalize predictions back to real μg/m3
    6. Write submission.csv  matching the sample_submission.csv format
"""

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
TEST_CSV      = "data/raw/test_raw.csv"
SAMPLE_SUB    = "submissions/sample_submission.csv"
MODEL_PATH    = "models/best_lstm.pt"
FITTED_PATH   = "data/processed/fitted_preprocessing.pkl"
FEATURE_PATH  = "data/processed/feature_cols.pkl"
OUTPUT_PATH   = "submissions/submission.csv"

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.fc(last).squeeze(-1)

# ─────────────────────────────────────────────────────────────────────────────
# Constants  (must match preprocessing.py and windowing.py)
# ─────────────────────────────────────────────────────────────────────────────
POLLUTANTS   = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
WEATHER      = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
NUMERIC_COLS = POLLUTANTS + WEATHER
WINDOW       = 24
HIDDEN_SIZE  = 256
NUM_LAYERS   = 3
DROPOUT      = 0.2

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load test data, fitted preprocessing params, feature list
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print(" STEP 1: Loading test data and preprocessing params ...")
print("=" * 60)

test_raw = pd.read_csv(TEST_CSV)
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
feature_cols  = meta['feature_cols']
station_list  = meta['station_list']
INPUT_SIZE    = len(feature_cols)

sample_sub = pd.read_csv(SAMPLE_SUB)

print(f"  Test rows        : {len(test_raw):,}")
print(f"  Stations         : {sorted(test_raw['station'].unique())}")
print(f"  Date range       : {test_raw['datetime'].min()} -> {test_raw['datetime'].max()}")
print(f"  Submission rows  : {len(sample_sub):,}")
print(f"  Features expected: {INPUT_SIZE}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Apply the same preprocessing pipeline as training
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(" STEP 2: Preprocessing test data ...")
print("=" * 60)

df = test_raw.copy()

# 2a. Local linear interpolation (within each station, up to 6-hour gaps)
for _, idx in df.groupby('station').groups.items():
    df.loc[idx, NUMERIC_COLS] = df.loc[idx, NUMERIC_COLS].interpolate(
        method='linear', limit=6, limit_direction='both'
    )

# 2b. Seasonal fallback using training lookup table
df['month'] = df['datetime'].dt.month
df['hour']  = df['datetime'].dt.hour
merged = df.merge(seasonal_lookup, on=['station', 'month', 'hour'],
                  suffixes=('', '_ssn'), how='left')
for c in NUMERIC_COLS:
    ssn_col = f'{c}_ssn'
    if ssn_col in merged.columns:
        df[c] = np.where(df[c].isna(), merged[ssn_col].values, df[c].values)

# 2c. Global median fallback
df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(global_medians)

# 2d. Wind direction: fill missing, encode as sin/cos
df['wd'] = df['wd'].fillna(df['station'].map(wd_mode))
deg = df['wd'].map(deg_map)
rad = np.deg2rad(deg)
df['wd_sin'] = np.sin(rad)
df['wd_cos'] = np.cos(rad)

# 2e. Normalize all numeric columns (using training means/stds)
for c in NUMERIC_COLS:
    df[c + '_norm'] = (df[c] - means[c]) / stds[c]

# 2f. Calendar cyclical features
df['hour_sin']   = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos']   = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin']  = np.sin(2 * np.pi * (df['month'] - 1) / 12)
df['month_cos']  = np.cos(2 * np.pi * (df['month'] - 1) / 12)

# 2g. Station one-hot columns
for s in station_list:
    df[f'st_{s}'] = (df['station'] == s).astype(np.float32)

nan_check = df[feature_cols].isna().sum().sum()
print(f"  NaNs after preprocessing: {nan_check}")
if nan_check > 0:
    print("  WARNING: filling remaining NaNs with 0")
    df[feature_cols] = df[feature_cols].fillna(0)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Build sliding windows for the test set
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(" STEP 3: Building 24-hour sliding windows ...")
print("=" * 60)

# We need to know the datetime of the TARGET hour (what we're predicting).
# The sample_submission has 4,103 IDs. We'll build all windows from the test
# data, tag each window with its target datetime + station, then match to the
# submission IDs.

# Load test.csv (the structured file with IDs and context columns)
test_structured = pd.read_csv("data/raw/test.csv")
print(f"  test.csv shape: {test_structured.shape}")
print(f"  test.csv columns (first 5): {list(test_structured.columns[:5])}")

# The test.csv already encodes the 24-hour windows.
# Let's check whether we can use test_raw.csv to reconstruct windows
# by station, since test_raw has clean station/datetime structure.

Xs, ids_out = [], []
all_window_info = []   # (station, target_datetime) for each window

for station in station_list:
    sub_df = df[df['station'] == station].sort_values('datetime').reset_index(drop=True)
    arr = sub_df[feature_cols].values.astype(np.float32)
    datetimes = sub_df['datetime'].values

    T = len(arr)
    if T <= WINDOW:
        continue

    windows = np.lib.stride_tricks.sliding_window_view(arr, window_shape=WINDOW, axis=0)
    windows = windows.transpose(0, 2, 1)   # (T-WINDOW+1, WINDOW, F)

    n_windows = windows.shape[0] - 1
    X = windows[:n_windows]
    target_dts = datetimes[WINDOW:WINDOW + n_windows]

    Xs.append(X)
    all_window_info.extend([(station, dt) for dt in target_dts])

X_test_all = np.concatenate(Xs, axis=0)
print(f"  Total test windows built: {X_test_all.shape[0]:,}")

# Build a lookup: (station, datetime_ns) -> window index
lookup = {(s, pd.Timestamp(dt).value): i for i, (s, dt) in enumerate(all_window_info)}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Run the LSTM on all test windows
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(" STEP 4: Running LSTM model on test data ...")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMForecaster(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

X_tensor = torch.tensor(X_test_all, dtype=torch.float32).to(device)

BATCH = 1024
preds_norm = []
with torch.no_grad():
    for i in range(0, len(X_tensor), BATCH):
        batch = X_tensor[i:i + BATCH]
        out = model(batch).cpu().numpy()
        preds_norm.append(out)

preds_norm = np.concatenate(preds_norm)
preds_raw  = preds_norm * std_pm25 + mean_pm25
preds_raw  = np.clip(preds_raw, 0, None)   # PM2.5 cannot be negative

print(f"  Predictions: min={preds_raw.min():.1f}  max={preds_raw.max():.1f}  mean={preds_raw.mean():.1f} ug/m3")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Match predictions to submission IDs
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(" STEP 5: Matching predictions to submission IDs ...")
print("=" * 60)

# test.csv has columns: id, station, plus hour-by-hour data for hours t-23..t-1.
# The target hour corresponds to the hour AFTER the last context hour.
# We can derive target datetime from the last context hour's year/month/day/hour + 1h.

# Find columns for the LAST context hour (highest suffix index)
cols = list(test_structured.columns)

# Identify the year/month/day/hour columns for the last (most recent) context hour.
# The pattern is: year, month, day, hour appear multiple times — the last set is newest.
year_cols  = [c for c in cols if c.startswith('year')]
month_cols = [c for c in cols if c.startswith('month')]
day_cols   = [c for c in cols if c.startswith('day')]
hour_cols  = [c for c in cols if c.startswith('hour')]

last_year  = test_structured[year_cols[-1]]
last_month = test_structured[month_cols[-1]]
last_day   = test_structured[day_cols[-1]]
last_hour  = test_structured[hour_cols[-1]]

# Target = context_last + 1 hour
last_dt  = pd.to_datetime({'year': last_year, 'month': last_month, 'day': last_day, 'hour': last_hour})
target_dt = last_dt + pd.Timedelta(hours=1)

# Station column
station_col = test_structured['station']

print(f"  test.csv rows          : {len(test_structured)}")
print(f"  Target datetime range  : {target_dt.min()} -> {target_dt.max()}")

# Match each submission row to a window prediction
pm25_preds = []
not_found  = 0

for i, row_id in enumerate(test_structured['id']):
    st  = station_col.iloc[i]
    tdt = target_dt.iloc[i]
    key = (st, tdt.value)
    idx = lookup.get(key, None)
    if idx is not None:
        pm25_preds.append(float(preds_raw[idx]))
    else:
        # Fallback: use global mean if no matching window (edge case)
        pm25_preds.append(float(mean_pm25))
        not_found += 1

print(f"  Matched windows : {len(pm25_preds) - not_found} / {len(pm25_preds)}")
if not_found > 0:
    print(f"  WARNING: {not_found} IDs had no matching window — used global mean fallback")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Write submission.csv
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(" STEP 6: Writing submission.csv ...")
print("=" * 60)

submission = pd.DataFrame({
    'id':    test_structured['id'].values,
    'PM2.5': pm25_preds,
})

submission.to_csv(OUTPUT_PATH, index=False)

print(f"  Rows written  : {len(submission)}")
print(f"  PM2.5 stats   : min={submission['PM2.5'].min():.1f}  "
      f"max={submission['PM2.5'].max():.1f}  "
      f"mean={submission['PM2.5'].mean():.1f}")
print(f"\n  Saved to: {OUTPUT_PATH}")
print("\n  First 5 rows of submission:")
print(submission.head().to_string(index=False))
print("\n" + "=" * 60)
print("  DONE! Upload submission.csv to Kaggle.")
print("=" * 60)
