# Team Division — PM2.5 Air Pollution Forecasting

> 5 parts · 1 per team member · each part has clear files, deliverables, and dependencies

---

## How the Parts Connect

```
Part 1 (Data & Preprocessing)
    ↓ produces: train_clean.pkl, val_clean.pkl, fitted_preprocessing.pkl
Part 2 (EDA & Baselines)
    ↓ produces: baseline_results.pkl, EDA report
Part 3 (Deep Learning Models — LSTM & TCN)
    ↓ produces: best_lstm.pt, best_tcn.pt, submission CSVs
Part 4 (Feature Engineering & Model Improvement)
    ↓ produces: windows_v2.npz, best_lstm_v2_seed0.pt, submission_lstm_v2.csv
Part 5 (Ensemble & Final Submission)
    ↓ produces: final submission CSV uploaded to Kaggle
```

All parts depend on Part 1 finishing first.
Parts 2, 3, and 4 can run in parallel after Part 1 is done.
Part 5 waits for Parts 3 and 4.

---

## Part 1 — Data & Preprocessing

**Difficulty:** ⭐⭐ Medium
**Role:** Data Engineer / Preprocessing Lead

### What You Own
| File | Action |
|---|---|
| `preprocessing.py` | Your main script — understand every line |
| `explore_data.py` | Run this first to understand the raw data |
| `train_raw.csv` | Your input |
| `train_clean.pkl` | Your output (consumed by everyone else) |
| `val_clean.pkl` | Your output |
| `fitted_preprocessing.pkl` | Your output (CRITICAL — used by submission scripts) |

### What You Do
1. **Understand the raw data** — run `explore_data.py`, look at the 12 stations,
   check how many missing values exist per column and per station
2. **Run the preprocessing pipeline** — `python preprocessing.py`
3. **Verify the outputs** — check that:
   - No NaNs remain in numeric columns
   - PM2.5_norm has mean ≈ 0, std ≈ 1 on training data
   - Val data uses training statistics (not its own)
4. **Document your decisions** — write a short section in the report explaining
   WHY you chose linear interpolation with 6h limit, seasonal fallback, etc.

### Key Decisions Already Made (understand these)
- **Chronological split at 2015-09-01** — last ~6 months held out for validation
- **3-tier missing value strategy:** interpolate → seasonal fallback → global median
- **Wind direction** encoded as sin/cos to preserve circular distance
- **Normalization** fitted on training data only — NEVER on val or test

### Deliverables
- [ ] `train_clean.pkl` and `val_clean.pkl` saved
- [ ] `fitted_preprocessing.pkl` saved (has means, stds, seasonal lookup)
- [ ] Written summary: How many NaNs were filled? By which method?
- [ ] Written summary: Train date range vs Val date range

### Depends On
Nothing — this is the starting point.

### Blocks
Everyone else is waiting for your pkl files.

---

## Part 2 — Exploratory Data Analysis & Baselines

**Difficulty:** ⭐ Easy–Medium
**Role:** Analyst / Baseline Researcher

### What You Own
| File | Action |
|---|---|
| `explore_data.py` | Run and extend with your own analysis |
| `baselines.py` | Run and fully understand |
| `baseline_results.pkl` | Your output |
| `windows.npz` | You need this (Part 1 must finish first) |

### What You Do
1. **EDA (Exploratory Data Analysis)** — using `explore_data.py` as a starting point:
   - Plot PM2.5 over time for each station — are there seasonal trends?
   - Plot PM2.5 distribution — is it skewed? Are there extreme outliers?
   - Plot correlation: which features (TEMP, WSPM, RAIN) most relate to PM2.5?
   - Check: does PM2.5 differ significantly between weekdays and weekends?
   - Check: which station has the highest average PM2.5? Which the lowest?
2. **Run baselines** — `python baselines.py`
3. **Understand the scores** — Persistence should be ~23 RMSE, Ridge ~20 RMSE
4. **Report findings** — write the analysis section of the project report

### Key Concepts to Explain in Your Report
- **Why persistence is the minimum bar** — if our model can't beat "copy last value,"
  it's useless
- **Why Ridge is the next bar** — it proves that using multiple features linearly
  helps, but sequence memory is what's missing
- **What the data tells us** — seasonal trends, station differences, pollution patterns

### Deliverables
- [ ] `baseline_results.pkl` saved (needed by Parts 3 and 4 for comparison prints)
- [ ] EDA report section: at least 5 plots with explanations
- [ ] Written summary: persistence RMSE, ridge RMSE, and what they tell us
- [ ] Written answer: which station has the worst pollution? Why might that be?

### Depends On
- Part 1: needs `train_clean.pkl`, `val_clean.pkl`, `windows.npz`

### Blocks
- Part 5 (needs baseline numbers for the final comparison table)

---

## Part 3 — Deep Learning Models (LSTM & TCN)

**Difficulty:** ⭐⭐⭐ Hard
**Role:** ML Engineer — Sequence Models

### What You Own
| File | Action |
|---|---|
| `train_lstm.py` | Train LSTM v1 — understand every line |
| `train_tcn.py` | Train TCN — understand the architecture |
| `windowing.py` | Understand how windows are built |
| `generate_submission.py` | Generate LSTM submission |
| `generate_tcn_submission.py` | Generate TCN submission |
| `best_lstm.pt` | Your output |
| `best_tcn.pt` | Your output |
| `submission.csv` | Your output |
| `submission_tcn.csv` | Your output |

### What You Do
1. **Run windowing** — `python windowing.py` (produces `windows.npz`)
2. **Train LSTM v1** — `python train_lstm.py`
   - Watch the val RMSE drop each epoch
   - Understand: hidden_size, num_layers, dropout, patience
3. **Train TCN** — `python train_tcn.py`
   - Understand: how dilated causal convolutions work
   - Understand: why the receptive field must cover all 24 hours
4. **Generate both submissions** — run both generate scripts
5. **Document the architectures** — write the model section of the report

### Key Concepts to Explain in Your Report
- **Why LSTM over FFNN?** — sequence memory vs snapshot
- **Why dropout?** — prevent overfitting
- **What is early stopping?** — stop when val RMSE stops improving (patience=5)
- **What is gradient clipping?** — prevents exploding gradients in RNNs
- **How does TCN's dilated convolution work?** — and what is its receptive field?

### Deliverables
- [ ] `best_lstm.pt` trained and saved
- [ ] `best_tcn.pt` trained and saved
- [ ] `submission.csv` generated (LSTM predictions)
- [ ] `submission_tcn.csv` generated (TCN predictions)
- [ ] `lstm_results.pkl` and `tcn_results.pkl` saved
- [ ] Written summary: final val RMSE for LSTM and TCN, training time

### Depends On
- Part 1: needs `train_clean.pkl`, `val_clean.pkl`, `fitted_preprocessing.pkl`
- Part 2: needs `baseline_results.pkl` (for comparison printing)
- `windowing.py` must be run first to produce `windows.npz`

### Blocks
- Part 5: needs `submission.csv` and `submission_tcn.csv`

---

## Part 4 — Feature Engineering & Model Improvement

**Difficulty:** ⭐⭐⭐⭐ Hard–Advanced
**Role:** ML Engineer — Feature Engineering & Advanced Models

### What You Own
| File | Action |
|---|---|
| `feature_engineering.py` | Your main script — build 47-feature, 48h windows |
| `train_lstm_v2.py` | Train improved LSTM (attention + Huber loss) |
| `generate_lstm_v2_submission.py` | Generate improved submission |
| `windows_v2.npz` | Your output |
| `feature_cols_v2.pkl` | Your output |
| `best_lstm_v2_seed0.pt` | Your output |
| `submission_lstm_v2.csv` | Your output |
| `LSTM v3/best_lstm.py` | Study this — your friend's per-station normalization idea |

### What You Do
1. **Run feature engineering** — `python feature_engineering.py`
   - Understand every new feature: lag1, lag6, rolling mean, rolling std, rolling max
   - Verify: `windows_v2.npz` has shape `(~262k, 48, 47)`
2. **Train LSTM v2** — `python train_lstm_v2.py`
   - Monitor val RMSE — target: drop below 13.5 on validation
   - Understand: what attention does differently from just taking the last step
   - Understand: why Huber loss is better than MSE for spiky data
3. **Generate submission** — `python generate_lstm_v2_submission.py`
4. **Study LSTM v3** — read `LSTM v3/best_lstm.py` and understand per-station normalization
5. **Document improvements** — write the "improvements" section of the report

### Key Concepts to Explain in Your Report
- **Why lag features help** — explicit vs implicit sequence information
- **What rolling statistics capture** — trend, volatility, extreme values
- **How temporal attention works** — weighted average over all timesteps vs just the last
- **Why Huber loss is better than MSE for PM2.5** — outlier robustness
- **What per-station normalization solves** (from LSTM v3)

### Deliverables
- [ ] `windows_v2.npz` built (47 features, 48h window)
- [ ] `best_lstm_v2_seed0.pt` trained
- [ ] `submission_lstm_v2.csv` generated
- [ ] `lstm_v2_results.pkl` saved
- [ ] Written summary: val RMSE of LSTM v2 vs LSTM v1 — how much did it improve?
- [ ] Written explanation of each new feature category (lags, rolling stats, dow)

### Depends On
- Part 1: needs `train_clean.pkl`, `val_clean.pkl`, `fitted_preprocessing.pkl`
- Part 2: needs `baseline_results.pkl` (for comparison)

### Blocks
- Part 5: needs `submission_lstm_v2.csv`

---

## Part 5 — Ensemble & Final Submission

**Difficulty:** ⭐⭐ Medium
**Role:** Integration Lead / Results Analyst

### What You Own
| File | Action |
|---|---|
| `ensemble_submission.py` | Your main script |
| `submission_ensemble_lstm70_tcn30.csv` | Current best — you own the improvement |
| All final `submission_*.csv` files | You compare, select, and upload |

### What You Do
1. **Wait for Parts 3 and 4** to produce their submission CSVs
2. **Run the ensemble script** — `python ensemble_submission.py`
   - This blends LSTM and TCN predictions with various weights
   - Try more combinations: 65/35, 72/28, etc.
3. **Try blending LSTM v2 with TCN** — edit `ensemble_submission.py` to use
   `submission_lstm_v2.csv` instead of `submission.csv`
4. **Compare all submissions** — make a table of every CSV and its Kaggle score
5. **Upload to Kaggle** in order (most promising first) and record scores
6. **Compile the final report** — collect everyone's written sections into one document

### Extended Tasks (if time allows)
- Try a 3-way blend: LSTM v1 + TCN + LSTM v2
- Try blending with FFNN and GRU submissions from your friend's folders
- Try learned weights: fit a tiny linear model on val set predictions to find optimal blend

### Key Concepts to Explain in Your Report
- **Why ensembling works** — model diversity, error cancellation
- **Why fixed weights beat equal-weight averaging** — models have different strengths
- **How you found the best blend** — grid search over weight combinations

### Deliverables
- [ ] Table of all Kaggle submission scores with blend weights
- [ ] Final best submission CSV uploaded to Kaggle
- [ ] Combined project report (collect sections from all 5 members)
- [ ] Final comparison table: Baseline → LSTM → TCN → Ensemble → LSTM v2

### Depends On
- Part 3: needs `submission.csv` and `submission_tcn.csv`
- Part 4: needs `submission_lstm_v2.csv`
- Part 2: needs baseline RMSE numbers for the final table

---

## Summary Table

| Part | Role | Key Output | Difficulty | Can Start |
|---|---|---|---|---|
| **1** | Data & Preprocessing | `train_clean.pkl`, `fitted_preprocessing.pkl` | ⭐⭐ | Immediately |
| **2** | EDA & Baselines | `baseline_results.pkl` + EDA report | ⭐ | After Part 1 |
| **3** | LSTM & TCN | `best_lstm.pt`, `best_tcn.pt`, submission CSVs | ⭐⭐⭐ | After Part 1 |
| **4** | Feature Eng. & LSTM v2 | `windows_v2.npz`, `submission_lstm_v2.csv` | ⭐⭐⭐⭐ | After Part 1 |
| **5** | Ensemble & Report | Final submission + project report | ⭐⭐ | After Parts 3 & 4 |

---

## Shared Rules for the Team

> [!IMPORTANT]
> **Never modify `fitted_preprocessing.pkl`** after Part 1 finishes.
> All models use this file to normalize test data. Changing it breaks every submission.

> [!IMPORTANT]
> **Never split data randomly.** Always use the same chronological split:
> train = before 2015-09-01, val = after 2015-09-01.

> [!TIP]
> **Check in your results files to Git** (`*.pkl`, `*.pt`, `*.csv`) after
> each run so teammates can use your outputs without retraining everything.

> [!TIP]
> **Coordinate on Kaggle submission slots** — Kaggle limits daily submissions.
> Parts 3, 4, and 5 should agree on a submission order before uploading.
