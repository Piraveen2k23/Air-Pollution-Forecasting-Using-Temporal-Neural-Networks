# Complete Project Guide — PM2.5 Air Pollution Forecasting
### From Raw Data to Kaggle Submission — Everything Explained

> **Goal:** Predict PM2.5 (fine particle pollution) 1 hour into the future,
> for 12 weather stations in Beijing, using historical pollution + weather readings.
> **Metric:** RMSE (Root Mean Squared Error) in μg/m³ — lower is better.

---

## Table of Contents

1. [What Is The Problem?](#1-what-is-the-problem)
2. [The Data](#2-the-data)
3. [Step 1 — Preprocessing](#3-step-1--preprocessing)
4. [Step 2 — Windowing](#4-step-2--windowing)
5. [Step 3 — Baseline Models](#5-step-3--baseline-models)
6. [Step 4 — Neural Network Models](#6-step-4--neural-network-models)
7. [Step 5 — Feature Engineering v2](#7-step-5--feature-engineering-v2)
8. [Step 6 — Ensembling](#8-step-6--ensembling)
9. [Step 7 — Generating the Submission](#9-step-7--generating-the-submission)
10. [Model Comparison Table](#10-model-comparison-table)
11. [Run Order Cheat Sheet](#11-run-order-cheat-sheet)
12. [File Map](#12-file-map)

---

## 1. What Is The Problem?

Beijing has 12 air quality monitoring stations spread across the city.
Each station records pollution and weather readings **every hour**.

**Our task:** Given the last 24 hours of readings at a station,
predict what PM2.5 will be **1 hour later**.

```
  Hour 1  Hour 2  ...  Hour 24   →   Predict PM2.5 at Hour 25
 [data]  [data]       [data]     →   [single number]
```

**Why does this matter?**
PM2.5 (particles smaller than 2.5 micrometres) are the most dangerous
air pollutant — they enter the bloodstream directly through the lungs.
A reliable 1-hour forecast lets people, hospitals, and governments
take action before dangerous levels are reached.

**Real-life analogy:** Imagine you're a doctor watching a patient's heart rate
every minute. You want to predict if their heart rate will be dangerously high
in the next hour. You use all the readings from the past 24 hours as context.
That's exactly what we're doing — just with air pollution instead of heart rate.

---

## 2. The Data

### Files
| File | What it is |
|---|---|
| `train_raw.csv` | 3 years of hourly readings (2013–2016) across 12 stations |
| `test_raw.csv` | Same stations, recent hours — **no PM2.5 label** |
| `test.csv` | Structured test format Kaggle expects predictions for |
| `sample_submission.csv` | What the submission CSV must look like |

### Features in the raw data
| Column | What it measures |
|---|---|
| `PM2.5` | Fine particles — **this is what we predict** |
| `PM10` | Coarser particles |
| `SO2` | Sulfur dioxide (from coal burning) |
| `NO2` | Nitrogen dioxide (from traffic) |
| `CO` | Carbon monoxide (from combustion) |
| `O3` | Ozone |
| `TEMP` | Temperature (°C) |
| `PRES` | Air pressure (hPa) |
| `DEWP` | Dew point — humidity indicator (°C) |
| `RAIN` | Rainfall in mm — rain washes PM2.5 out of air |
| `WSPM` | Wind speed (m/s) |
| `wd` | Wind direction (N, NE, SW, etc.) |

### The 12 Stations
`Aotizhongxin`, `Changping`, `Dingling`, `Dongsi`, `Guanyuan`, `Gucheng`,
`Huairou`, `Nongzhanguan`, `Shunyi`, `Tiantan`, `Wanliu`, `Wanshouxigong`

We train ONE model on all 12 stations together (using a station one-hot flag
so the model knows which station it is looking at).

---

## 3. Step 1 — Preprocessing

**Script:** `preprocessing.py`
**Output:** `train_clean.pkl`, `val_clean.pkl`, `fitted_preprocessing.pkl`

### 3.1 Chronological Train/Val Split

We split time — not rows at random.

```
2013-03-01 ──────────────────────── 2015-08-31   TRAIN (~80%)
2015-09-01 ──────────────────────── 2016-02-28   VAL  (~20%)
```

**Why chronological?**
If you split randomly, "future" data leaks into training.
Your model learns from 2016 data while being tested on 2015 data — which
would never happen in real life. This is called **data leakage** and makes
your model look artificially good.

**Real-life analogy:** Studying for an exam by reading tomorrow's exam paper
first, then doing practice questions. Your practice score would be
misleadingly high. In ML, we always test on data the model has never seen —
and that data must be *later in time*, not just randomly held out.

---

### 3.2 Handling Missing Values

PM2.5 sensors go offline, transmitters fail, equipment gets serviced.
The dataset has missing readings. We fill them in 3 tiers:

#### Tier 1 — Local linear interpolation (up to 6-hour gaps)
```
Hour 10: PM2.5 = 45.0
Hour 11: PM2.5 = ???   ← draw a straight line between 45 and 60
Hour 12: PM2.5 = 60.0
```
**Analogy:** Your thermometer breaks for 2 hours. A safe guess is to
linearly interpolate between the last good reading and the next good one.

#### Tier 2 — Seasonal lookup (for gaps longer than 6 hours)
We compute the average PM2.5 for every `(station, month, hour-of-day)`
combination from training data. If a gap is too long for interpolation,
use the historical average for that station at that time.

```
E.g.: Dongsi station + March + 8am → average PM2.5 = 87.3 μg/m³
```

**Analogy:** Your thermometer breaks for 2 days. Look up historical
records: "In Beijing in March at 8am, it's typically 5°C." Use that.

#### Tier 3 — Global median (last resort)
If a station has NO historical average for that month/hour,
use the overall median across all training data.

> [!IMPORTANT]
> All lookup tables (seasonal averages, means, stds) are computed from
> **training data only**. Applying them to validation/test data is fine.
> Computing them on ALL data would leak future information into training.

---

### 3.3 Wind Direction Encoding

Wind direction is a compass string: `N`, `NE`, `SSW`, etc.

**Problem:** If you convert to numbers (N=0, NE=1, ..., NNW=15),
the model thinks N and NNW are very different (0 vs 15) when they're
only 22.5° apart on a circle.

**Solution:** Sin/cos cyclical encoding:
```
N   → sin(0°) = 0.0,  cos(0°) = 1.0
E   → sin(90°)= 1.0,  cos(90°)= 0.0
S   → sin(180°)=0.0,  cos(180°)=-1.0
```

**Analogy:** Clock hands. 11:55pm and 12:05am are 10 minutes apart,
but as raw numbers (2355 vs 5) they look far apart. Sin/cos wraps the circle.

---

### 3.4 Normalization (Z-score)

Raw PM2.5 ranges from 0–900 μg/m³. Temperature is −20 to +40°C.
These different scales confuse neural networks.

We normalize every numeric feature:
```
PM2.5_norm = (PM2.5 - mean_PM2.5) / std_PM2.5
```

After this, every feature has mean ≈ 0 and std ≈ 1.

**Analogy:** Comparing student heights (150–200cm) with exam scores (0–100).
You can't just average them — they're on different scales. Normalization
puts everything on the same footing.

**Key rule:** Compute mean and std from **training data only**, then apply
those same numbers to val and test data.

---

### 3.5 Calendar Cyclical Encoding

Hour-of-day (0–23) encoded as sin/cos so hour 23 and hour 0 are adjacent:
```
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```
Same for month (0–11) and day-of-week (0–6).

---

### 3.6 Station One-Hot Encoding

We tell the model which of the 12 stations it is looking at:
```
Aotizhongxin → [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Changping    → [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

---

## 4. Step 2 — Windowing

**Script:** `windowing.py`
**Output:** `windows.npz`

### The Concept

We convert the time series into individual training examples using a
**sliding window**.

```
Raw time series (one station):
  t=1:  [PM2.5=45, TEMP=5, ...]
  t=2:  [PM2.5=48, TEMP=5, ...]
  ...
  t=24: [PM2.5=60, TEMP=6, ...]
  t=25: [PM2.5=65, ...]          ← TARGET

Window 1: Input = rows 1..24  →  Target = PM2.5 at t=25
Window 2: Input = rows 2..25  →  Target = PM2.5 at t=26
Window 3: Input = rows 3..26  →  Target = PM2.5 at t=27
```

The window slides forward 1 hour at a time, generating one training
example per slide.

**Analogy:** Teaching someone to predict traffic by showing them dashcam
clips. Each clip is 24 hours long. The question is: "What does traffic
look like in the hour right after this clip?" Slide the clip forward 1
hour to make the next training example.

### Result

```
X_train shape: (262,944 , 24 , 29)
                ↑          ↑    ↑
           examples    hours  features
```

- **262,944 examples** — 12 stations × ~3 years of hourly data
- **24 timesteps** — 24 hours of context per example
- **29 features** — all normalized columns at each timestep

---

## 5. Step 3 — Baseline Models

**Script:** `baselines.py`

Before training complex neural networks, we establish benchmarks:
**"What does a completely dumb model score?"**

If your neural network barely beats a dumb model, it's not worth the effort.

---

### Baseline 1 — Persistence (The "Copy-Paste" Predictor)

**Rule:** PM2.5 next hour = PM2.5 right now.
```python
prediction[t+1] = PM2.5[t]   # just copy the current value forward
```

**Why this is reasonable:** Air pollution changes slowly most of the time.
If PM2.5 is 60 μg/m³ now, it will probably be around 60 in the next hour.

**Why this is still dumb:** It has no awareness of trends.
If pollution is rising steadily, persistence will always be wrong by
the same amount — it can't "see" the rising trend.

**Analogy:** Predicting tomorrow's weather by saying "it will be exactly
like today." Works fine for stable weather, fails badly when a storm rolls in.

**Score:** RMSE ≈ 23 μg/m³

---

### Baseline 2 — Ridge Regression (Smart Linear Model)

**Rule:** Fit a linear equation using the last hour's 29 features:
```
PM2.5_next = w1×PM2.5 + w2×TEMP + w3×WSPM + ... + bias
```

Ridge adds an L2 penalty (sum of squared weights) to prevent overfitting
when there are many features. The penalty forces weights to stay small
unless a feature is genuinely useful.

**Why better than persistence:** Uses wind, temperature, and other signals —
not just the current PM2.5 value.

**Why still limited:** Only looks at ONE hour. No memory of past trends.
A linear model cannot learn "if PM2.5 was rising for 3 hours, it will
likely keep rising."

**Analogy:** A doctor predicting heart rate using only the most recent
reading + a few vital signs — but with no knowledge of the trend over
the past hours.

**Score:** RMSE ≈ 20 μg/m³

**Key conclusion:** Any model we train MUST beat ~20 RMSE to be useful.
If it doesn't, it's not learning anything a simple linear model can't do.

---

## 6. Step 4 — Neural Network Models

### 6.1 FFNN — Feed-Forward Neural Network

**Folder:** `FFNN/`

The simplest neural network — no memory, no sequence.
Takes only the last hour's 29 features and maps them to a prediction.

```
[29 features at hour t] → [Dense layers] → [PM2.5 at t+1]
```

**Limitation:** Like Ridge, it ignores the 24-hour history. It only sees
a snapshot of the current moment.

**Result:** RMSE ≈ 20.8 — barely better than Ridge because it ignores history.

---

### 6.2 GRU — Gated Recurrent Unit

**Folder:** `GRU/`

GRU reads the 24-hour sequence step by step and carries a **hidden state**
(memory) forward.

It has two gates:
- **Update gate:** How much of the old memory to carry forward
- **Reset gate:** How much to forget when computing new memory

**Why simpler than LSTM:** Fewer parameters → faster training.
Slightly less powerful for long sequences.

**Result:** RMSE ≈ 21.7

---

### 6.3 LSTM — Long Short-Term Memory

**Script:** `train_lstm.py`

The workhorse of sequence modelling. Reads 24 hours step by step.
Has TWO memory pathways:
- **Cell state** — long-term memory (slow to change)
- **Hidden state** — short-term working memory

```
Hour 1 → [LSTM cell] → memory h1 ─┐
Hour 2 → [LSTM cell] → memory h2 ─┤
...                                ─┤
Hour 24→ [LSTM cell] → memory h24 → [Linear layer] → Prediction
```

**4 internal gates:**
| Gate | Role | Analogy |
|---|---|---|
| Forget gate | What old memory to erase | "I can forget this old fact" |
| Input gate | What new info to write | "Write this down" |
| Cell state | The actual long-term memory | The notebook |
| Output gate | What to read out as the prediction | "Read this page now" |

**Architecture:**
- `hidden_size = 128` — 128 memory units
- `num_layers = 2` — stack 2 LSTM layers
- `dropout = 0.2` — randomly zero 20% of neurons to prevent overfitting
- Loss = MSE (Mean Squared Error)

**Result:** Kaggle RMSE ≈ 14.9

---

### 6.4 TCN — Temporal Convolutional Network

**Script:** `train_tcn.py`

Instead of reading the sequence step-by-step (LSTM), TCN uses
1D convolution filters that slide across the time axis.

**Key tricks:**
- **Causal convolution:** Filter at time t only sees t and earlier (no future)
- **Dilated convolution:** Filters skip with exponentially growing gaps:
  ```
  Block 1 (dilation=1): sees  3 hours back
  Block 2 (dilation=2): sees  7 hours back
  Block 3 (dilation=4): sees 15 hours back
  Block 4 (dilation=8): sees 31 hours back ← covers full 24h!
  ```
- **Residual connections:** Skip connections keep gradients flowing

**Analogy:** Instead of reading a book word-by-word (LSTM), TCN scans
with a magnifying glass that looks at every 2nd word, then every 4th,
then every 8th — capturing both short and long patterns in one pass.

**Advantage:** Fully parallelizable (faster than LSTM on GPU).

**Result:** Kaggle RMSE ≈ 15.4

---

### 6.5 LSTM v2 — Attention + Huber Loss + 48h Window

**Script:** `train_lstm_v2.py`

Three upgrades over LSTM v1:

#### Upgrade A — Temporal Attention (Stop wasting 47 hours of context)

Standard LSTM throws away everything except the last step:
```python
last = out[:, -1, :]   # discards outputs from hours 1..23!
```

Attention learns a weighted average over ALL timesteps:
```
Hour 1  output × weight 0.01  ─┐
Hour 12 output × weight 0.30  ─┤ → weighted sum → context vector
Hour 48 output × weight 0.35  ─┘
```

The model learns to focus on the most predictive hours.

**Analogy:** A judge evaluating a case doesn't only read the last
witness statement. They weigh ALL statements — giving more weight
to the most credible ones. Attention does this over time.

#### Upgrade B — Huber Loss (robust to PM2.5 spikes)

```
MSE   error of 200 μg/m³: loss = 200²   = 40,000  ← dominates training!
Huber error of 200 μg/m³: loss = 1×200 − 0.5 = manageable
```

Rare dust storms cause PM2.5 spikes of 300–500 μg/m³. MSE gets obsessed
trying to predict these rare extremes, which hurts accuracy on normal days.
Huber is a hybrid: smooth like MSE for small errors, bounded like MAE for big ones.

#### Upgrade C — 48h window instead of 24h

PM2.5 patterns often span 2 days (factory cycles, traffic patterns).
More context = better predictions.

---

### 6.6 LSTM v3 — Per-Station Normalization

**Folder:** `LSTM v3/`

**The key insight:** The original model uses one global mean/std for PM2.5
across ALL 12 stations. But stations have very different baseline levels:
- Dingling (suburban): average ~50 μg/m³
- Dongsi (busy urban): average ~120 μg/m³

Using global stats confuses the model — a normalized value of "1.0"
means very different things at different stations.

**Fix:** Compute separate mean/std **for each station separately**.
This is argued to be the **single biggest improvement** for a multi-station model.

---

## 7. Step 5 — Feature Engineering v2

**Script:** `feature_engineering.py`
**Output:** `windows_v2.npz` (47 features per timestep, 48h window)

We add 17 new columns that give the model explicit knowledge of trends,
instead of making it infer trends from raw values.

### Lag Features (6 new columns)
```
pm25_lag1   = PM2.5 value 1 hour ago
pm25_lag6   = PM2.5 value 6 hours ago
pm25_lag24  = PM2.5 value 24 hours ago (same hour yesterday)
...
```

**Why:** The model already sees PM2.5 across the sequence, but lags make
the relationship between past and present *explicit*. Much easier to learn from.

**Analogy:** Difference between telling a student "here are last 24 test
scores" vs "you improved by 5 points each test." Same info, but the
second is much easier to act on.

### Rolling Mean (4 new columns)
```
pm25_rmean6  = average PM2.5 over last 6h
pm25_rmean24 = average PM2.5 over last 24h
```

**Why:** Smooths out noise. If the 6h rolling average is rising,
pollution is building — a strong signal for the future.

### Rolling Std (3 new columns)
```
pm25_rstd3  = how much PM2.5 fluctuated in the last 3h
pm25_rstd6  = how much PM2.5 fluctuated in the last 6h
```

**Why:** High std = unstable, unpredictable conditions. This is hard for
LSTM to infer on its own.

### Rolling Max (2 new columns)
```
pm25_rmax6   = worst PM2.5 in the last 6h
pm25_rmax24  = worst PM2.5 in the last 24h
```

**Why:** Captures whether there was a recent spike that might be continuing.

### Day-of-Week Features (3 new columns)
```
dow_sin, dow_cos, is_weekend
```

**Why:** Traffic (and thus PM2.5) differs between weekdays and weekends.
Monday morning rush hour ≠ Sunday morning.

---

## 8. Step 6 — Ensembling

**Script:** `ensemble_submission.py`

**Ensembling** = combining predictions from multiple models.

**Why it works:** Each model makes different kinds of errors.
LSTM might overestimate in windy conditions. TCN might underestimate in winter.
Averaging cancels some of these errors.

**Analogy:** If you ask 5 doctors to independently diagnose the same patient,
their combined opinion is more reliable than any single one.

### Our approach — weighted average:
```
final = w_LSTM × lstm_prediction + w_TCN × tcn_prediction
```

| Blend | Kaggle RMSE |
|---|---|
| LSTM 100% | 14.935 |
| LSTM 80% + TCN 20% | 14.879 |
| **LSTM 70% + TCN 30%** | **14.878 ← current best** |
| LSTM 60% + TCN 40% | 14.897 |
| LSTM 50% + TCN 50% | 14.933 |

---

## 9. Step 7 — Generating the Submission

**Scripts:** `generate_submission.py`, `generate_tcn_submission.py`,
`generate_lstm_v2_submission.py`

### The Process

```
1. Load test_raw.csv
2. Apply EXACT SAME preprocessing as training
   (using saved fitted_preprocessing.pkl — same means, stds, lookup tables)
3. Build sliding windows from test data
4. Run trained model → normalized predictions
5. Un-normalize: PM2.5_pred = pred_norm × std_PM2.5 + mean_PM2.5
6. Clip to 0 (PM2.5 can't be negative)
7. Match each prediction to correct row ID in test.csv
8. Save as CSV → upload to Kaggle
```

**Why the same preprocessing matters:**
If training used mean=80 but the test pipeline used a different mean,
every prediction would be systematically wrong.
The `fitted_preprocessing.pkl` file locks in training statistics forever.

---

## 10. Model Comparison Table

| Model | Architecture | Window | Features | Kaggle RMSE |
|---|---|---|---|---|
| Persistence (baseline) | Copy last value | — | 1 | ~23 |
| Ridge (baseline) | Linear | Last hour | 29 | ~20 |
| FFNN | Dense layers | Last hour | 29 | ~20.8 |
| GRU | Recurrent | 24h | 29 | ~21.7 |
| LSTM v1 | Recurrent, 2 layers | 24h | 29 | 14.935 |
| TCN | Dilated causal conv | 24h | 29 | 15.381 |
| **LSTM v1 + TCN (70/30)** | **Ensemble** | **24h** | **29** | **14.878** |
| LSTM v2 | LSTM + Attention | 48h | 47 | In training |
| LSTM v3 | Per-station norm | 24h | ~35 | Pending |

---

## 11. Run Order Cheat Sheet

```bash
# ── Original Pipeline ──────────────────────────────────────────
python preprocessing.py          # clean data, fill NaNs, normalize
python windowing.py              # build 24h windows (29 features)
python baselines.py              # check persistence ~23, ridge ~20

python train_lstm.py             # train LSTM v1
python train_tcn.py              # train TCN

python generate_submission.py        # LSTM → submission.csv
python generate_tcn_submission.py    # TCN  → submission_tcn.csv
python ensemble_submission.py        # blend → submission_ensemble_lstm70_tcn30.csv

# ── Improved v2 Pipeline ───────────────────────────────────────
python feature_engineering.py        # 48h windows + 47 features
python train_lstm_v2.py              # LSTM + attention + Huber loss
python generate_lstm_v2_submission.py # → submission_lstm_v2.csv
```

---

## 12. File Map

```
project/
│
├── preprocessing.py              Step 1: clean data, fill NaNs, normalize
├── windowing.py                  Step 2: build 24h sliding windows (29 features)
├── feature_engineering.py        Step 2v2: build 48h windows (47 features)
├── baselines.py                  Step 3: persistence + ridge benchmarks
│
├── train_lstm.py                 Train LSTM v1 (24h, MSE loss)
├── train_tcn.py                  Train TCN (24h, causal dilated conv)
├── train_lstm_v2.py              Train LSTM v2 (48h, attention, Huber loss)
│
├── generate_submission.py        LSTM v1  → submission.csv
├── generate_tcn_submission.py    TCN      → submission_tcn.csv
├── generate_lstm_v2_submission.py LSTM v2 → submission_lstm_v2.csv
├── ensemble_submission.py        Blend    → submission_ensemble_*.csv
│
├── FFNN/                         Friend's FFNN model + results
├── GRU/                          Friend's GRU model + results
├── LSTM/                         Friend's LSTM baseline
├── LSTM v2/                      Friend's LSTM v2 (Colab version)
├── LSTM v3/                      Friend's best LSTM (per-station normalization)
│
├── fitted_preprocessing.pkl      CRITICAL: saved normalization stats
├── feature_cols.pkl              Feature list for 24h windows
├── feature_cols_v2.pkl           Feature list for 48h windows
├── windows.npz                   24h windows (262k training examples)
├── windows_v2.npz                48h windows with lag features
│
├── best_lstm.pt                  Trained LSTM v1 weights
├── best_tcn.pt                   Trained TCN weights
│
└── submission_ensemble_lstm70_tcn30.csv   Current best submission (14.878)
```

---

> [!TIP]
> **Key lesson from this project:**
> The biggest improvements came from **better features** (lag/rolling features),
> **better data handling** (per-station normalization, Huber loss for spikes),
> and **ensembling** — not from making the neural network architecture fancier.
>
> **Always improve your data and features before improving your model.**
