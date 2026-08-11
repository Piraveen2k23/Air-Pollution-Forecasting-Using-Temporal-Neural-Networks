"""
Step 2 — Preprocessing pipeline for the Beijing PM2.5 forecasting task.

Design principles:
 - Chronological split FIRST, before any statistic is computed, so nothing
   from the validation period leaks into training statistics.
 - All fitted values (seasonal lookup tables, wind-direction modes, means/stds)
   are computed on the TRAIN partition only, then reused unchanged on validation
   (and later on the real Kaggle test set).
 - Missing-value strategy is two-tier, based on the EDA finding that most gaps
   are short (a few hours) but a handful are very long (up to ~340 hours):
     1) local linear interpolation, capped at a 6-hour limit
     2) seasonal fallback: mean for that (station, month, hour-of-day) from train
     3) final fallback: global train median (only needed for edge rows)
"""
import pandas as pd
import numpy as np
import pickle

RAW_PATH = "data/raw/train_raw.csv"
VAL_START = "2015-09-01"   # last ~6 months of the 3-year train file held out for validation

POLLUTANTS = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
WEATHER = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
NUMERIC_COLS = POLLUTANTS + WEATHER

COMPASS = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
DEG_MAP = {d: i * 22.5 for i, d in enumerate(COMPASS)}


def load_and_split(path=RAW_PATH, val_start=VAL_START):
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df = df.sort_values(['station', 'datetime']).reset_index(drop=True)
    train_df = df[df['datetime'] < val_start].copy()
    val_df = df[df['datetime'] >= val_start].copy()
    return train_df, val_df


def interpolate_within_station(data, cols, limit=6):
    data = data.copy()
    for _, idx in data.groupby('station').groups.items():
        data.loc[idx, cols] = data.loc[idx, cols].interpolate(
            method='linear', limit=limit, limit_direction='both'
        )
    return data


def fit_seasonal_lookup(train_df, cols):
    return train_df.groupby(['station', 'month', 'hour'])[cols].mean()


def apply_seasonal_fill(data, lookup, cols):
    data = data.copy()
    merged = data.merge(lookup, on=['station', 'month', 'hour'], suffixes=('', '_ssn'), how='left')
    for c in cols:
        data[c] = np.where(data[c].isna(), merged[f'{c}_ssn'].values, data[c].values)
    return data


def fit_wd_mode(train_df):
    return train_df.groupby('station')['wd'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'N')


def encode_wind(data, wd_mode):
    data = data.copy()
    data['wd'] = data['wd'].fillna(data['station'].map(wd_mode))
    deg = data['wd'].map(DEG_MAP)
    rad = np.deg2rad(deg)
    data['wd_sin'] = np.sin(rad)
    data['wd_cos'] = np.cos(rad)
    return data


def fit_scaler(train_df, cols):
    return train_df[cols].mean(), train_df[cols].std()


def apply_scaler(data, means, stds, cols):
    data = data.copy()
    for c in cols:
        data[c + '_norm'] = (data[c] - means[c]) / stds[c]
    return data


def run_pipeline():
    train_df, val_df = load_and_split()

    # 1) local interpolation (each partition uses only its own rows)
    train_df = interpolate_within_station(train_df, NUMERIC_COLS)
    val_df = interpolate_within_station(val_df, NUMERIC_COLS)

    # 2) seasonal fallback fit on TRAIN only, applied to both
    seasonal_lookup = fit_seasonal_lookup(train_df, NUMERIC_COLS)
    train_df = apply_seasonal_fill(train_df, seasonal_lookup, NUMERIC_COLS)
    val_df = apply_seasonal_fill(val_df, seasonal_lookup, NUMERIC_COLS)

    # 3) final fallback: global train median (catches rare edge cases)
    global_medians = train_df[NUMERIC_COLS].median()
    train_df[NUMERIC_COLS] = train_df[NUMERIC_COLS].fillna(global_medians)
    val_df[NUMERIC_COLS] = val_df[NUMERIC_COLS].fillna(global_medians)

    # 4) wind direction: fill missing category, then cyclical encode
    wd_mode = fit_wd_mode(train_df)
    train_df = encode_wind(train_df, wd_mode)
    val_df = encode_wind(val_df, wd_mode)

    # 5) normalization fit on TRAIN only
    means, stds = fit_scaler(train_df, NUMERIC_COLS)
    train_df = apply_scaler(train_df, means, stds, NUMERIC_COLS)
    val_df = apply_scaler(val_df, means, stds, NUMERIC_COLS)

    # persist fitted parameters for reuse later on test.csv
    fitted = {
        'seasonal_lookup': seasonal_lookup,
        'global_medians': global_medians,
        'wd_mode': wd_mode,
        'means': means,
        'stds': stds,
        'deg_map': DEG_MAP,
    }
    with open('data/processed/fitted_preprocessing.pkl', 'wb') as f:
        pickle.dump(fitted, f)

    return train_df, val_df, fitted


if __name__ == '__main__':
    train_df, val_df, fitted = run_pipeline()

    print("Train rows:", len(train_df), " | Val rows:", len(val_df))
    print("Train date range:", train_df['datetime'].min(), "to", train_df['datetime'].max())
    print("Val date range:  ", val_df['datetime'].min(), "to", val_df['datetime'].max())
    print()
    print("Remaining NaNs in train (numeric cols):", train_df[NUMERIC_COLS].isna().sum().sum())
    print("Remaining NaNs in val   (numeric cols):", val_df[NUMERIC_COLS].isna().sum().sum())
    print("Remaining NaNs in wd_sin/cos (train/val):",
          train_df[['wd_sin','wd_cos']].isna().sum().sum(),
          val_df[['wd_sin','wd_cos']].isna().sum().sum())
    print()
    print("PM2.5_norm stats (train):", train_df['PM2.5_norm'].mean().round(4), train_df['PM2.5_norm'].std().round(4))
    print("PM2.5_norm stats (val):  ", val_df['PM2.5_norm'].mean().round(4), val_df['PM2.5_norm'].std().round(4))

    train_df.to_pickle('data/processed/train_clean.pkl')
    val_df.to_pickle('data/processed/val_clean.pkl')
    print("\nSaved train_clean.pkl, val_clean.pkl, fitted_preprocessing.pkl")
