"""
Step 4 — Baseline models, evaluated with RMSE/MAE on the validation split
built in Step 3. Every later model (LSTM, TCN) has to beat these to be
worth using.

Baseline 1: Persistence — predict that PM2.5 one hour from now equals the
            most recently observed PM2.5 value. This is the standard
            "naive" forecasting baseline named in the project proposal.
Baseline 2: Ridge regression on the most recent hour's full feature vector
            (all pollutants + weather + wind + calendar + station one-hot
            at time t) — tests whether a simple *linear* combination of the
            current state beats pure persistence, without yet using any
            temporal/sequence structure.
"""
import numpy as np
import pickle
from sklearn.linear_model import Ridge

data = np.load('windows.npz', allow_pickle=True)
with open('feature_cols.pkl', 'rb') as f:
    meta = pickle.load(f)
feature_cols = meta['feature_cols']
pm25_idx = feature_cols.index('PM2.5_norm')

with open('fitted_preprocessing.pkl', 'rb') as f:
    fitted = pickle.load(f)
mean_pm25 = fitted['means']['PM2.5']
std_pm25 = fitted['stds']['PM2.5']


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def report(name, y_true, y_pred):
    r, m = rmse(y_true, y_pred), mae(y_true, y_pred)
    print(f"{name:35s}  RMSE={r:7.3f}  MAE={m:7.3f}")
    return r, m


X_train, X_val = data['X_train'], data['X_val']
y_train_raw, y_val_raw = data['y_train_raw'], data['y_val_raw']
y_train_norm, y_val_norm = data['y_train_norm'], data['y_val_norm']

results = {}

# ---- Baseline 1: Persistence ----
persist_train_pred = X_train[:, -1, pm25_idx] * std_pm25 + mean_pm25
persist_val_pred = X_val[:, -1, pm25_idx] * std_pm25 + mean_pm25

print("=== Baseline 1: Persistence (last observed PM2.5) ===")
results['persistence_train'] = report("Train", y_train_raw, persist_train_pred)
results['persistence_val'] = report("Validation", y_val_raw, persist_val_pred)
print()

# ---- Baseline 2: Ridge regression on last-hour feature vector ----
X_train_last = X_train[:, -1, :]
X_val_last = X_val[:, -1, :]

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_last, y_train_norm)

ridge_train_pred_norm = ridge.predict(X_train_last)
ridge_val_pred_norm = ridge.predict(X_val_last)
ridge_train_pred = ridge_train_pred_norm * std_pm25 + mean_pm25
ridge_val_pred = ridge_val_pred_norm * std_pm25 + mean_pm25

print("=== Baseline 2: Ridge regression (last-hour features -> next hour) ===")
results['ridge_train'] = report("Train", y_train_raw, ridge_train_pred)
results['ridge_val'] = report("Validation", y_val_raw, ridge_val_pred)

# save for later comparison against LSTM/TCN
with open('baseline_results.pkl', 'wb') as f:
    pickle.dump({
        'results': results,
        'persist_val_pred': persist_val_pred,
        'ridge_val_pred': ridge_val_pred,
        'y_val_raw': y_val_raw,
    }, f)
print("\nSaved baseline_results.pkl")