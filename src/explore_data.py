"""
Quick data explorer — shows what the preprocessed data looks like.
"""
import pandas as pd
import numpy as np

# ── Load preprocessed files ──────────────────────────────────────────────────
train_df = pd.read_pickle('data/processed/train_clean.pkl')
val_df   = pd.read_pickle('data/processed/val_clean.pkl')

print("=" * 70)
print("  PREPROCESSED DATA OVERVIEW")
print("=" * 70)

# ── Basic shape ──────────────────────────────────────────────────────────────
print(f"\nTrain set : {len(train_df):,} rows  |  {train_df.shape[1]} columns")
print(f"Val   set : {len(val_df):,} rows  |  {val_df.shape[1]} columns")

print(f"\nTrain date range: {train_df['datetime'].min()} -> {train_df['datetime'].max()}")
print(f"Val   date range: {val_df['datetime'].min()} -> {val_df['datetime'].max()}")

# ── Stations ─────────────────────────────────────────────────────────────────
print(f"\nStations ({train_df['station'].nunique()} total): {sorted(train_df['station'].unique())}")

# ── Columns ──────────────────────────────────────────────────────────────────
print(f"\nAll columns ({len(train_df.columns)}):")
for i, col in enumerate(train_df.columns, 1):
    print(f"   {i:2d}. {col}")

# ── Sample rows ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FIRST 5 ROWS (Train) - Key Columns Only")
print("=" * 70)

key_cols = ['station', 'datetime', 'PM2.5', 'PM2.5_norm', 'TEMP', 'TEMP_norm',
            'wd_sin', 'wd_cos']
print(train_df[key_cols].head(5).to_string(index=False))

# ── PM2.5 stats ───────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PM2.5 STATISTICS (Raw ug/m3)")
print("=" * 70)
print(f"{'Metric':<15} {'Train':>12} {'Val':>12}")
print("-" * 40)
for label, func in [("Mean", "mean"), ("Std Dev", "std"), ("Min", "min"),
                    ("25th pct", lambda x: x.quantile(0.25)),
                    ("Median", "median"),
                    ("75th pct", lambda x: x.quantile(0.75)),
                    ("Max", "max")]:
    if callable(func):
        tr = func(train_df['PM2.5'])
        va = func(val_df['PM2.5'])
    else:
        tr = getattr(train_df['PM2.5'], func)()
        va = getattr(val_df['PM2.5'], func)()
    print(f"{label:<15} {tr:>12.2f} {va:>12.2f}")

# ── Missing values check ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  MISSING VALUES AFTER PREPROCESSING")
print("=" * 70)
numeric_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3',
                'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
train_nans = train_df[numeric_cols].isna().sum()
val_nans   = val_df[numeric_cols].isna().sum()
print(f"{'Column':<12} {'Train NaNs':>12} {'Val NaNs':>10}")
print("-" * 36)
for col in numeric_cols:
    print(f"{col:<12} {train_nans[col]:>12} {val_nans[col]:>10}")

# ── Normalized columns check ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  NORMALIZED COLUMNS (should be ~mean=0, std=1 on train)")
print("=" * 70)
norm_cols = ['PM2.5_norm', 'TEMP_norm', 'CO_norm', 'O3_norm']
print(f"{'Column':<15} {'Train Mean':>12} {'Train Std':>10} {'Val Mean':>10} {'Val Std':>10}")
print("-" * 60)
for col in norm_cols:
    print(f"{col:<15} {train_df[col].mean():>12.4f} {train_df[col].std():>10.4f}"
          f" {val_df[col].mean():>10.4f} {val_df[col].std():>10.4f}")

# ── Windows info ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  WINDOWS (model-ready arrays from windows.npz)")
print("=" * 70)
try:
    data = np.load('data/processed/windows.npz', allow_pickle=True)
    X_train = data['X_train']
    X_val   = data['X_val']
    y_train = data['y_train_raw']
    y_val   = data['y_val_raw']
    print(f"X_train shape : {X_train.shape}  -> {X_train.shape[0]:,} windows x {X_train.shape[1]} timesteps x {X_train.shape[2]} features")
    print(f"X_val   shape : {X_val.shape}  -> {X_val.shape[0]:,} windows x {X_val.shape[1]} timesteps x {X_val.shape[2]} features")
    print(f"\ny_train (PM2.5 targets): min={y_train.min():.1f}  max={y_train.max():.1f}  mean={y_train.mean():.1f}")
    print(f"y_val   (PM2.5 targets): min={y_val.min():.1f}  max={y_val.max():.1f}  mean={y_val.mean():.1f}")
    print(f"\nEach training example = a {X_train.shape[1]}-hour sequence of {X_train.shape[2]} features -> 1 PM2.5 value to predict")
except FileNotFoundError:
    print("windows.npz not found — run windowing.py first.")

print("\n" + "=" * 70)
print("  Done!")
print("=" * 70)
