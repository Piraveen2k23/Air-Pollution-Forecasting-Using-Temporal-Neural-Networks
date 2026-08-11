"""
Step 3 — Build 24-hour input windows -> 1-hour-ahead PM2.5 target.

Design notes:
 - Windows are built PER STATION (never sliding across a station boundary).
 - Windows are built PER PARTITION (train windows only use train rows, val
   windows only use val rows) to keep the split boundary clean and simple.
   This costs ~23 samples per station at the start of validation (negligible:
   ~276 out of 52k rows) in exchange for zero risk of leakage/confusion.
 - Adds cyclical hour-of-day / month-of-year encodings (deterministic, no
   fitting needed) so the model sees hour 23 and hour 0 as neighbors, not
   numerically distant.
 - Adds a one-hot station indicator, repeated across all 24 timesteps, so a
   single pooled model (trained on all 12 stations at once) can still learn
   station-specific baseline levels instead of averaging them away.
 - Uses np.lib.stride_tricks.sliding_window_view for fast, vectorized window
   construction instead of a slow per-row Python loop.
"""
import numpy as np
import pandas as pd
import pickle

WINDOW = 24

POLLUTANTS_NORM = ['PM2.5_norm', 'PM10_norm', 'SO2_norm', 'NO2_norm', 'CO_norm', 'O3_norm']
WEATHER_NORM = ['TEMP_norm', 'PRES_norm', 'DEWP_norm', 'RAIN_norm', 'WSPM_norm']
WIND_COLS = ['wd_sin', 'wd_cos']


def add_calendar_cyclical(df):
    df = df.copy()
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
    return df


def make_windows_for_station(df_station, feature_cols, window=WINDOW):
    arr = df_station[feature_cols].values.astype(np.float32)          # (T, F)
    target_norm = df_station['PM2.5_norm'].values.astype(np.float32)
    target_raw = df_station['PM2.5'].values.astype(np.float32)
    datetimes = df_station['datetime'].values

    T = arr.shape[0]
    if T <= window:
        return None

    windows = np.lib.stride_tricks.sliding_window_view(arr, window_shape=window, axis=0)
    windows = windows.transpose(0, 2, 1)                              # (T-window+1, window, F)

    n_windows = windows.shape[0] - 1        # drop the last, which has no next-hour target
    X = windows[:n_windows]                 # hours [i, i+window) as input
    y_norm = target_norm[window:window + n_windows]   # the hour right after each window
    y_raw = target_raw[window:window + n_windows]
    target_dt = datetimes[window:window + n_windows]
    return X, y_norm, y_raw, target_dt


def build_dataset(df, station_list, feature_cols):
    Xs, y_norms, y_raws, dts, stations_out = [], [], [], [], []
    for station in station_list:
        sub = df[df['station'] == station].sort_values('datetime')
        result = make_windows_for_station(sub, feature_cols)
        if result is None:
            continue
        X, y_norm, y_raw, target_dt = result
        Xs.append(X)
        y_norms.append(y_norm)
        y_raws.append(y_raw)
        dts.append(target_dt)
        stations_out.append(np.full(len(y_norm), station))
    X_all = np.concatenate(Xs, axis=0)
    y_norm_all = np.concatenate(y_norms, axis=0)
    y_raw_all = np.concatenate(y_raws, axis=0)
    dt_all = np.concatenate(dts, axis=0)
    station_all = np.concatenate(stations_out, axis=0)
    return X_all, y_norm_all, y_raw_all, dt_all, station_all


def run():
    train_df = pd.read_pickle('data/processed/train_clean.pkl')
    val_df = pd.read_pickle('data/processed/val_clean.pkl')

    train_df = add_calendar_cyclical(train_df)
    val_df = add_calendar_cyclical(val_df)

    station_list = sorted(train_df['station'].unique())

    # one-hot station columns, fit on train's station list (fixed vocabulary)
    for s in station_list:
        train_df[f'st_{s}'] = (train_df['station'] == s).astype(np.float32)
        val_df[f'st_{s}'] = (val_df['station'] == s).astype(np.float32)
    station_onehot_cols = [f'st_{s}' for s in station_list]

    feature_cols = (POLLUTANTS_NORM + WEATHER_NORM + WIND_COLS +
                     ['hour_sin', 'hour_cos', 'month_sin', 'month_cos'] +
                     station_onehot_cols)

    X_train, y_train_norm, y_train_raw, dt_train, st_train = build_dataset(train_df, station_list, feature_cols)
    X_val, y_val_norm, y_val_raw, dt_val, st_val = build_dataset(val_df, station_list, feature_cols)

    print("Feature columns (%d):" % len(feature_cols), feature_cols)
    print()
    print("X_train:", X_train.shape, "y_train_norm:", y_train_norm.shape)
    print("X_val:  ", X_val.shape, "y_val_norm:", y_val_norm.shape)
    print()
    print("Any NaNs in X_train:", np.isnan(X_train).sum())
    print("Any NaNs in X_val:  ", np.isnan(X_val).sum())
    print("Train windows per station:")
    print(pd.Series(st_train).value_counts())

    np.savez_compressed(
        'data/processed/windows.npz',
        X_train=X_train, y_train_norm=y_train_norm, y_train_raw=y_train_raw,
        dt_train=dt_train.astype('datetime64[ns]').astype(np.int64), st_train=st_train,
        X_val=X_val, y_val_norm=y_val_norm, y_val_raw=y_val_raw,
        dt_val=dt_val.astype('datetime64[ns]').astype(np.int64), st_val=st_val,
    )
    with open('data/processed/feature_cols.pkl', 'wb') as f:
        pickle.dump({'feature_cols': feature_cols, 'station_list': station_list}, f)

    print("\nSaved windows.npz, feature_cols.pkl")
    return X_train, y_train_norm, y_train_raw, dt_train, st_train, X_val, y_val_norm, y_val_raw, dt_val, st_val


if __name__ == '__main__':
    run()
