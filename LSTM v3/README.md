# CO5420 - Best LSTM Model for PM2.5 Forecasting

One focused, tuned LSTM model - built with everything we identified that
should genuinely help, rather than a multi-model comparison.

## Files
- `best_lstm.py` - training script
- `predict_best_lstm.py` - generates a Kaggle `submission.csv` from a
  trained checkpoint once `test.csv` is available
- `best_lstm_demo.pt` - a checkpoint trained on a 2-station subset (demo
  only - retrain on the full 12-station `train_raw.csv` for your real run)

## What makes this "the best" version
1. **Per-station normalisation** (the main upgrade). Each of the 12 Beijing
   stations gets its own mean/std for every numeric feature, instead of one
   global mean/std shared across all stations. Stations have genuinely
   different baseline pollution levels, so this should be the single biggest
   lever for a multi-station model - more impactful than architecture tweaks.
2. **Rolling/diff temporal features**: rolling mean (3h/6h/12h) and rolling
   std (6h) of PM2.5, plus its first difference; rolling mean (6h) of PM10,
   TEMP, and WSPM. All computed causally *within* each 24-hour window only,
   so they reconstruct identically from `test.csv`'s lagged columns.
3. **Cyclical encoding** of wind direction (16-point compass -> sin/cos) and
   calendar time (hour/day-of-week/month -> sin/cos).
4. **Station one-hot identity**, kept alongside per-station normalisation so
   the model can still learn station-specific volatility/seasonal patterns
   beyond just the mean pollution level.
5. **A deeper/wider LSTM**: 128 hidden units, 2 layers (vs. 64/1 in the
   original CPU-limited version), with early stopping and
   `ReduceLROnPlateau` learning-rate scheduling.

## Demo result (2-station subset only - NOT your final number)
- Persistence baseline: RMSE 23.29
- This LSTM: **RMSE 22.53, MAE 10.49**

This was trained on only 2 of 12 stations due to sandbox limits. Per-station
normalisation should matter more with the full dataset (more stations with
more varied baselines to separate). Retrain on the full `train_raw.csv` to
get your real number.

## How to train (recommended: Colab with GPU)
```bash
python best_lstm.py --data train_raw.csv \
  --epochs 40 --patience 8 \
  --hidden_size 128 --num_layers 2 --batch_size 256 \
  --save_path best_lstm.pt --meta_path preprocessing_meta.json \
  --cache_windows windows_cache.npz
```
- `--cache_windows` and `--meta_path` let you resume training later without
  re-running preprocessing:
```bash
python best_lstm.py --data train_raw.csv \
  --epochs 15 --patience 8 --hidden_size 128 --num_layers 2 \
  --batch_size 256 --lr 3e-4 \
  --save_path best_lstm.pt --meta_path preprocessing_meta.json \
  --cache_windows windows_cache.npz --resume_from best_lstm.pt
```
Lower `--lr` on resumed runs once RMSE plateaus (mimics what
`ReduceLROnPlateau` would have done with a longer single run).

## How to predict once test.csv is released
```bash
python predict_best_lstm.py \
  --meta_path preprocessing_meta.json \
  --test_csv test.csv \
  --checkpoint best_lstm.pt \
  --out submission_lstm.csv
```
`preprocessing_meta.json` (created during training) stores the per-station
normalisation stats and feature order needed to process `test.csv`
consistently - keep it alongside your checkpoint.

## Caveat on test.csv column names
`predict_best_lstm.py` assumes `test.csv` provides `<feature>_lag_K` columns
for every numeric feature, plus (if available) `wd_lag_K`, `hour_lag_K`,
`month_lag_K`, `dow_lag_K` for wind/calendar. Check the real file's columns
once released and adjust the column-name patterns in `build_test_windows`
if they differ.

## Honest note on accuracy expectations
No model is "unbeatable" - PM2.5 has genuine hour-to-hour randomness no
model can predict, and other teams may have more compute/tuning time. This
version incorporates the most promising levers we identified (per-station
normalisation especially), but the actual gap to any specific competitor's
score depends on factors outside a single script - ensembling multiple
seeds, more extensive hyperparameter search, or additional model types
(e.g. gradient boosting) are the next things to try if you have more time.
