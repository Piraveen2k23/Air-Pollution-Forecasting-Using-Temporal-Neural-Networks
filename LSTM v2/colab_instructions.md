# Running the PM2.5 Forecasting Pipeline in Google Colab

This guide trains FFN, LSTM, and GRU models on `train_raw.csv`, generates
predictions for `test.csv`, and downloads all three submission files.

## ⚠️ Important: features have changed
`pm25_forecasting_v2.py` now includes extra rolling-mean/rolling-std/
first-difference PM2.5 features (26 features total). If you have OLD
checkpoint files from an earlier run, they will NOT load with this version
- you must train fresh checkpoints using the commands below.

## Prerequisites
- `train_raw.csv` (training data)
- `test.csv` (Kaggle test data - released Week 12)
- `pm25_forecasting_v2.py`
- `predict_submission.py`

**Before you start:** go to `Runtime` -> `Change runtime type` -> select
`GPU`. Training on CPU is much slower, and the LSTM/GRU below use a bigger
architecture than a CPU can comfortably handle in reasonable time.

---

## Cell 1 - Upload your files
```python
from google.colab import files
uploaded = files.upload()  # select: train_raw.csv, test.csv, pm25_forecasting_v2.py, predict_submission.py
```

## Cell 2 - Install dependencies
```python
!pip install torch pandas numpy scikit-learn -q
```

## Cell 3 - Train FFN
```python
!python pm25_forecasting_v2.py --data train_raw.csv --model_type ffn \
  --epochs 30 --patience 6 --batch_size 512 \
  --cache_windows windows_cache.npz \
  --save_path best_model.pt \
  --results_json results.json
```

## Cell 4 - Train LSTM (bigger architecture, more epochs)
(reuses the cached windows from Cell 3, so preprocessing is skipped)
```python
!python pm25_forecasting_v2.py --data train_raw.csv --model_type lstm \
  --epochs 40 --patience 8 --batch_size 256 \
  --hidden_size 128 --num_layers 2 \
  --cache_windows windows_cache.npz \
  --save_path best_model.pt \
  --results_json results.json
```
> This exact config (hidden_size 128, num_layers 2, batch_size 256) was
> validated on a 2-station subset of the real data and reached RMSE 22.66 /
> MAE 10.32 after ~20 epochs total (with a couple of learning-rate step-downs
> via `--resume_from` + lower `--lr`, matching how ReduceLROnPlateau works).
> On the full 12-station dataset with GPU, expect this to train faster per
> epoch and likely reach a lower RMSE with the extra data.

## Cell 5 - Train GRU (bigger architecture, more epochs)
```python
!python pm25_forecasting_v2.py --data train_raw.csv --model_type gru \
  --epochs 40 --patience 8 --batch_size 512 \
  --hidden_size 128 --num_layers 2 \
  --cache_windows windows_cache.npz \
  --save_path best_model.pt \
  --results_json results.json
```

> `hidden_size 128 --num_layers 2` is a meaningfully bigger LSTM/GRU than the
> CPU-limited version trained before (which used `hidden_size 64,
> num_layers 1`). Combined with the new rolling/diff features, this should
> help the recurrent models close the gap with (or beat) the feedforward net.
> If training seems slow or unstable, try `hidden_size 96` as a middle ground.

## Cell 6 - Check validation results before predicting
```python
import json
print(json.load(open("results.json")))
```
Compare RMSE across `ffn`, `lstm`, `gru`. Lower is better (this is the
Kaggle scoring metric). Also check MAE for your report.

## Cell 7 - Generate predictions for all three models
```python
!python predict_submission.py --train_csv train_raw.csv --test_csv test.csv \
  --checkpoint best_model_ffn.pt --model_type ffn --out submission_ffn.csv

!python predict_submission.py --train_csv train_raw.csv --test_csv test.csv \
  --checkpoint best_model_lstm.pt --model_type lstm --hidden_size 128 --num_layers 2 \
  --out submission_lstm.csv

!python predict_submission.py --train_csv train_raw.csv --test_csv test.csv \
  --checkpoint best_model_gru.pt --model_type gru --hidden_size 128 --num_layers 2 \
  --out submission_gru.csv
```

## Cell 8 (optional) - Ensemble the three predictions
Averaging predictions from multiple models often beats any single model.
```python
import pandas as pd

ffn = pd.read_csv("submission_ffn.csv")
lstm = pd.read_csv("submission_lstm.csv")
gru = pd.read_csv("submission_gru.csv")

ensemble = ffn.copy()
ensemble["PM2.5"] = (ffn["PM2.5"] + lstm["PM2.5"] + gru["PM2.5"]) / 3
ensemble.to_csv("submission_ensemble.csv", index=False)
```

## Cell 9 - Download your submission CSV(s)
```python
from google.colab import files
files.download("submission_lstm.csv")   # or whichever scored best in results.json
# files.download("submission_ensemble.csv")
```

---

## Important notes

- **`--hidden_size` and `--num_layers` must match** between training (Cell
  4/5) and prediction (Cell 7) for LSTM/GRU, or loading the checkpoint will
  fail with a shape mismatch error.
- Each `pm25_forecasting_v2.py --model_type X` run overwrites `best_model.pt`
  but also saves a model-specific copy (e.g. `best_model_ffn.pt`), so you
  keep all three checkpoints even after training all of them in sequence.
- Only **submit one** CSV to Kaggle - whichever model (or the ensemble) has
  the lowest RMSE in `results.json` / makes sense for your report - but keep
  all files for your project report's model comparison section.
- If Colab disconnects mid-training, you can resume a specific model from
  its last saved checkpoint:
  ```python
  !python pm25_forecasting_v2.py --data train_raw.csv --model_type lstm \
    --epochs 15 --patience 8 --batch_size 512 \
    --hidden_size 128 --num_layers 2 --lr 3e-4 \
    --cache_windows windows_cache.npz \
    --resume_from best_model_lstm.pt \
    --save_path best_model.pt \
    --results_json results.json
  ```
