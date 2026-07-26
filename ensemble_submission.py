"""
ensemble_submission.py  —  Combine LSTM + TCN predictions into one submission.

HOW IT WORKS:
  Reads two already-generated submission CSVs, blends their PM2.5 predictions
  using different weights, and saves one file per weight combination so you can
  upload all of them to Kaggle and pick the best score.

  Formula:  final = w_lstm × lstm_pred + (1 - w_lstm) × tcn_pred

  Since LSTM scores better on Kaggle (14.935 vs 15.381), we try several
  weight combinations that give more influence to the LSTM.
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# INPUT FILES — point these to your best submissions
# ─────────────────────────────────────────────────────────────────────────────
LSTM_CSV = "submission.csv"            # Kaggle: 14.935 (best so far)
TCN_CSV  = "submission_tcn.csv"        # Kaggle: 15.381

# ─────────────────────────────────────────────────────────────────────────────
# Load both submission files
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  ENSEMBLE SUBMISSION GENERATOR")
print("=" * 55)

lstm = pd.read_csv(LSTM_CSV)
tcn  = pd.read_csv(TCN_CSV)

# Safety check — IDs must be in the same order
assert list(lstm['id']) == list(tcn['id']), \
    "ERROR: submission files have different IDs or order!"

print(f"\n  LSTM  predictions loaded : {len(lstm):,} rows")
print(f"  TCN   predictions loaded : {len(tcn):,} rows")
print(f"\n  LSTM  PM2.5 mean: {lstm['PM2.5'].mean():.2f}")
print(f"  TCN   PM2.5 mean: {tcn['PM2.5'].mean():.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# Generate multiple weight combinations and save each as a separate CSV
# ─────────────────────────────────────────────────────────────────────────────
# w_lstm = fraction of LSTM in the blend
# Since LSTM is stronger on Kaggle, we try heavier LSTM weights too
weights = [
    (0.50, 0.50),   # 50/50  — pure average
    (0.60, 0.40),   # 60% LSTM, 40% TCN
    (0.70, 0.30),   # 70% LSTM, 30% TCN
    (0.75, 0.25),   # 75% LSTM, 25% TCN
    (0.80, 0.20),   # 80% LSTM, 20% TCN
]

print(f"\n  {'File':<40}  {'Mean PM2.5':>12}  {'LSTM%':>6}  {'TCN%':>6}")
print(f"  {'-'*40}  {'-'*12}  {'-'*6}  {'-'*6}")

# also show the originals for reference
print(f"  {'submission.csv (LSTM original)':<40}  {lstm['PM2.5'].mean():>12.2f}  {'100%':>6}  {'  0%':>6}")
print(f"  {'submission_tcn.csv (TCN original)':<40}  {tcn['PM2.5'].mean():>12.2f}  {'  0%':>6}  {'100%':>6}")
print(f"  {'-'*40}  {'-'*12}  {'-'*6}  {'-'*6}")

generated_files = []
for w_lstm, w_tcn in weights:
    # Weighted blend
    blended = w_lstm * lstm['PM2.5'].values + w_tcn * tcn['PM2.5'].values
    blended = np.clip(blended, 0, None)    # PM2.5 can't be negative

    # Build output dataframe
    out = pd.DataFrame({'id': lstm['id'], 'PM2.5': blended})

    # Filename encodes the weights so you know what you're uploading
    fname = f"submission_ensemble_lstm{int(w_lstm*100)}_tcn{int(w_tcn*100)}.csv"
    out.to_csv(fname, index=False)
    generated_files.append(fname)

    print(f"  {fname:<40}  {blended.mean():>12.2f}  {int(w_lstm*100):>5}%  {int(w_tcn*100):>5}%")

print(f"\n  Generated {len(generated_files)} ensemble files.")

# ─────────────────────────────────────────────────────────────────────────────
# Recommendation
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  UPLOAD ORDER (try all, pick best Kaggle score)")
print("=" * 55)
print("""
  Strategy: upload from heaviest LSTM weight downward.
  Since LSTM already wins (14.935), the ensemble only
  helps if TCN adds complementary information.

  Recommended upload order:
    1. submission_ensemble_lstm70_tcn30.csv  ← start here
    2. submission_ensemble_lstm60_tcn40.csv
    3. submission_ensemble_lstm50_tcn50.csv
    4. submission_ensemble_lstm80_tcn20.csv

  If any ensemble beats 14.935 → that's your new best!
  If none beat it → stick with submission.csv (LSTM alone)
""")
