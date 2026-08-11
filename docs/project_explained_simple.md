# Air Pollution Forecasting — Explained Simply, Step by Step

> Think of this document as a **guided tour** of the project.  
> No jargon. No rushing. One idea at a time.

---

## 🤔 First — What Is This Project Trying to Do?

Imagine you live in Beijing. Every hour, sensors around the city measure **how dirty the air is**. The main pollution number they track is called **PM2.5** — tiny particles in the air that are harmful to breathe.

**The question this project answers is:**

> *"Given the last 24 hours of air quality readings, can a neural network predict what the PM2.5 will be in the NEXT hour?"*

That's it. That is the entire project goal.

- **Input**: 24 hours of past sensor readings (pollution + weather)
- **Output**: 1 number — predicted PM2.5 for the next hour
- **Unit**: μg/m³ (micrograms per cubic meter — standard air quality unit)

---

## 📦 The Dataset — What Data Do We Have?

The data comes from **12 monitoring stations** spread across Beijing. Each station records measurements **every single hour** from **March 2013 to February 2017** — that's 4 years of hourly data.

### What Each Row in the CSV Looks Like

Every row = **one hour at one station**. The columns are:

| Column | What It Measures |
|---|---|
| `PM2.5` | Fine dust particles (our **target** — what we want to predict) |
| `PM10` | Coarser dust particles |
| `SO2` | Sulfur dioxide (from burning coal) |
| `NO2` | Nitrogen dioxide (from cars) |
| `CO` | Carbon monoxide |
| `O3` | Ozone |
| `TEMP` | Temperature (°C) |
| `PRES` | Atmospheric pressure (hPa) |
| `DEWP` | Dew point (how humid it is) |
| `RAIN` | Rainfall (mm) |
| `WSPM` | Wind speed (m/s) |
| `wd` | Wind direction (N, NNE, NE, …, NNW — 16 directions) |

> [!NOTE]
> We use ALL these columns as inputs to help predict PM2.5. Wind, rain, and temperature all affect how pollution spreads or clears.

---

## 🗂️ The Project Files — What Each One Does

Before diving into code, here's what every file on disk is for:

```
project/
│
├── data/
│   ├── raw/
│   │   ├── 📄 train_raw.csv          ← Raw training data (4 years of hourly readings)
│   │   └── 📄 test_raw.csv           ← Test data (Kaggle gives us windows to predict on)
│   └── processed/
│       ├── 📦 train_clean.pkl        ← Cleaned training data (saved for reuse)
│       ├── 📦 val_clean.pkl          ← Cleaned validation data
│       ├── 📦 fitted_preprocessing.pkl ← The "recipe" used to clean data
│       ├── 📦 windows.npz            ← 262,000 ready-to-use training examples
│       └── 📦 feature_cols.pkl       ← Names of all 29 input features
│
├── src/ 
│   ├── (Core Pipeline - run these in order)
│   ├── 🐍 explore_data.py        ← Step 1: Look at the data
│   ├── 🐍 preprocessing.py       ← Step 2: Clean the data
│   ├── 🐍 windowing.py           ← Step 3: Package data for the neural network
│   ├── 🐍 baselines.py           ← Step 4: Simple "dumb" benchmarks to beat
│   ├── 🐍 train_lstm.py          ← Step 5A: Train the LSTM neural network
│   ├── 🐍 train_tcn.py           ← Step 5B: Train the TCN neural network
│   ├── 🐍 generate_submission.py     ← Step 6A: Use LSTM to predict test data
│   ├── 🐍 generate_tcn_submission.py ← Step 6B: Use TCN to predict test data
│   ├── 🐍 ensemble_submission.py     ← Step 7: Combine both predictions
│   │
│   ├── (Advanced "V2" Upgrades)
│   ├── 🐍 feature_engineering.py     ← V2: Advanced feature engineering (lags, rolling stats)
│   ├── 🐍 train_lstm_v2.py           ← V2: Train improved LSTM with attention and Huber loss
│   ├── 🐍 generate_lstm_v2_submission.py ← V2: Use improved LSTM to predict test data
│   │
│   ├── (Extra Experiments & Analysis)
│   ├── 🐍 extended_graphs.py         ← Extra: Deeper visual analysis of the data
│   ├── 🐍 feature_importance.py      ← Extra: Analyzes which features matter most
│   ├── 🐍 poster_graphs.py           ← Extra: Generates graphs for presentation poster
│   ├── 🐍 train_aqi_classifier.py    ← Extra: Predict AQI category instead of raw PM2.5
│   ├── 🐍 train_gru.py               ← Extra: Train a GRU (simpler RNN) model
│   └── 🐍 window_ablation.py         ← Extra: Experiments testing different window sizes
│
├── models/
│   ├── 📦 best_lstm.pt           ← Saved LSTM brain (best weights found)
│   └── 📦 best_tcn.pt            ← Saved TCN brain (best weights found)
│
├── results/
│   ├── 📦 baseline_results.pkl   ← Scores of the simple benchmarks
│   ├── 📦 lstm_results.pkl       ← LSTM performance numbers
│   └── 📦 tcn_results.pkl        ← TCN performance numbers
│
└── submissions/
    ├── 📄 sample_submission.csv  ← Kaggle shows us the format to submit predictions
    ├── 📄 submission.csv         ← LSTM predictions for Kaggle
    ├── 📄 submission_tcn.csv     ← TCN predictions for Kaggle
    └── 📄 submission_ensemble_*.csv ← Blended predictions for Kaggle
```

---

## 🔍 Step 1 — `explore_data.py` — "Look Before You Touch"

Before doing anything, a good data scientist **looks at the data** to understand what they're working with.

This script loads the cleaned data and answers questions like:
- How many rows do we have?
- What date range does it cover?
- What do the PM2.5 values look like? (min, max, average?)
- Are there any missing values?
- Do the normalized columns look right (should average to ~0)?

**Think of it like:** opening a new textbook and flipping through every page before you start studying — so you know what's coming.

---

## 🧹 Step 2 — `preprocessing.py` — "Cleaning the Data"

Raw data from sensors is messy. This script fixes it. Let's go through each problem it solves.

---

### Problem 1: Missing Values (Gaps in the Sensor Data)

Sensors sometimes break or go offline. When they do, there's no reading for that hour — just a blank (`NaN` in Python, meaning "Not a Number").

**How it fills the gaps — three tiers:**

#### Tier 1: Interpolation (for short gaps, ≤ 6 hours)
If there are only a few missing hours in a row, we just **draw a straight line** between the last known value and the next known value.

```
Hour 10: PM2.5 = 40
Hour 11: PM2.5 = ??? (missing)
Hour 12: PM2.5 = ??? (missing)
Hour 13: PM2.5 = 70

→ Fill: Hour 11 = 50, Hour 12 = 60  (evenly spaced between 40 and 70)
```

#### Tier 2: Seasonal Average (for longer gaps)
If a gap is too long for interpolation, use the **historical average for that station, that month, and that hour of day** (all computed from training data only).

```
Hour 11 in June at Station A is still missing?
→ Look up: "What is the average PM2.5 at Station A, in June, at 11am, across all years?"
→ Use that value.
```

#### Tier 3: Global Median (last resort)
If even the seasonal average can't be found (very rare edge cases), use the **single middle value of all training PM2.5 readings**.

> [!IMPORTANT]
> **The Split Happens FIRST.** We divide the data into training (before Sep 2015) and validation (Sep 2015 onward) **before** computing any averages or statistics. This prevents "cheating" — using future information to fill past gaps.

---

### Problem 2: Wind Direction is a String, Not a Number

The `wd` column has values like `"N"`, `"NE"`, `"SSW"` — 16 compass directions. Neural networks need **numbers**.

**Naive approach (wrong):** Just assign numbers 0–15. But then NNW (15) and N (0) would seem far apart, even though they're neighbors on a compass!

**Smart approach — Cyclical Encoding:**
1. Convert direction to degrees (N=0°, NNE=22.5°, NE=45°, …)
2. Compute `sin(angle)` and `cos(angle)`

```
N   →  0°  →  sin=0.0,  cos=1.0
E   → 90°  →  sin=1.0,  cos=0.0
S   →180°  →  sin=0.0,  cos=-1.0
W   →270°  →  sin=-1.0, cos=0.0
```

Now N (0°) and NNW (337.5°) are very close to each other numerically — just like they are on a real compass. ✅

---

### Problem 3: All Features Need to Be on the Same Scale

PM2.5 might range from 0 to 500 μg/m³.  
Temperature might range from -10 to 40 °C.  
Pressure might range from 990 to 1050 hPa.

If we feed these raw numbers to a neural network, the large-magnitude features (pressure) will dominate over small-magnitude features (temperature), even if temperature is more informative.

**Solution — Z-score normalization:**
```
normalized = (value - mean) / standard_deviation
```

After this, every column has approximately:
- Mean = 0
- Standard deviation = 1

The neural network now "sees" all features at the same scale.

---

### What Gets Saved After This Step
- `train_clean.pkl` — the full cleaned & normalized training DataFrame
- `val_clean.pkl` — the full cleaned & normalized validation DataFrame
- `fitted_preprocessing.pkl` — the "recipe book" (all the averages, std deviations, seasonal lookup tables) so we can apply the **exact same transformations** to the test set later

---

## 📦 Step 3 — `windowing.py` — "Packaging Data for the Neural Network"

Neural networks don't read CSVs. They need data in a specific shape. This step converts the table of rows into thousands of **training examples**.

### The Core Idea: Sliding Windows

Imagine a magnifying glass that slides along the timeline of a single station:

```
Time:   1  2  3  4  5  6  7  8  9  10  11  12 ...

Window 1:  [1  2  3  4  5  6  7  8  9  10  11  12  ...  24] → predict hour 25
Window 2:  [2  3  4  5  6  7  8  9  10  11  12  13  ...  25] → predict hour 26
Window 3:  [3  4  5  6  7  8  9  10  11  12  13  14  ...  26] → predict hour 27
...
```

Each window is **24 hours long**. The target is the **PM2.5 value at the very next hour** (hour 25, 26, 27, ...).

> [!NOTE]
> Windows never cross station boundaries. Station A's data and Station B's data are never mixed into the same window.

### What's in Each Window?

Each of the 24 timesteps has **29 features** (numbers describing that hour). So each training example is a 3D block of shape:

```
(24 timesteps, 29 features)
```

And the target is a single number (PM2.5 next hour).

### Extra Features Added During Windowing

The 29 features include 4 calendar features added here:

| Feature | How Computed | Why |
|---|---|---|
| `hour_sin` | `sin(2π × hour / 24)` | 11pm and midnight are neighbors |
| `hour_cos` | `cos(2π × hour / 24)` | same reason |
| `month_sin` | `sin(2π × (month-1) / 12)` | December and January are neighbors |
| `month_cos` | `cos(2π × (month-1) / 12)` | same reason |

And 12 station one-hot columns (one per station):

| Feature | Value for Station A | Value for Station B |
|---|---|---|
| `st_StationA` | 1.0 | 0.0 |
| `st_StationB` | 0.0 | 1.0 |
| … | … | … |

This lets a **single model** handle all 12 stations at once — it just tells the model "you're currently looking at station X."

### Final Counts
- **Training windows**: 262,944 examples
- **Validation windows**: 52,128 examples
- **Shape of each X**: (24, 29)
- **Shape of each y**: scalar (one PM2.5 number)

Saved as `windows.npz` — a compressed numpy file.

---

## 📏 Step 4 — `baselines.py` — "The 'Dumb' Models We Must Beat"

Before building a complex neural network, we ask: **"What's the simplest possible prediction?"** If our fancy model doesn't beat these, it's not actually useful.

### Baseline 1: Persistence
> "I predict PM2.5 next hour = PM2.5 right now."

No learning whatsoever. Just repeat the current value. This is surprisingly decent because pollution doesn't change dramatically in one hour.

```python
prediction = last_hour_PM2.5
```

### Baseline 2: Ridge Regression
> "I use ALL 29 features from the most recent hour and fit a linear equation."

Like: `PM2.5_next = a₁×PM2.5_now + a₂×TEMP_now + a₃×WIND_now + ...`

Ridge regression finds the best values of `a₁, a₂, a₃, …` using math (no neural network needed). It's more powerful than Persistence, but it only uses the **current snapshot** — not the full 24-hour history.

### Why Do We Need These?
If our neural network can't beat a simple "just repeat the current value" prediction, then the neural network is useless. These baselines set the **bar to clear**.

---

## 🧠 Step 5A — `train_lstm.py` — "Teaching the LSTM Brain"

### What is an LSTM?

**LSTM = Long Short-Term Memory** — a type of neural network designed for sequences (time-series, text, speech, etc.).

#### The Core Intuition

Think of reading a sentence word by word. As you read, you keep a mental "summary" of what came before. When you get to the last word, your brain has processed the entire sentence and can answer a question about it.

An LSTM does the same with our 24-hour sequence:

```
Hour 1 data → LSTM → memory updated
Hour 2 data → LSTM → memory updated
Hour 3 data → LSTM → memory updated
...
Hour 24 data → LSTM → final memory
                         ↓
                    "Based on everything I've seen, PM2.5 will be X"
```

The LSTM has **memory gates** that decide:
- What to **remember** from the past
- What to **forget** (irrelevant old info)
- What to **output** at each step

#### What the LSTM Looks Like in Code

```
Input  (batch of 512 examples, each is 24 hours × 29 features)
   ↓
LSTM Layer 1  (256 memory units)
   ↓
LSTM Layer 2  (256 memory units)
   ↓
LSTM Layer 3  (256 memory units)
   ↓ (we only take the output at the LAST hour — it summarizes all 24 hours)
Dropout       (randomly zero 20% of neurons — prevents memorizing training data)
   ↓
Linear Layer  (256 → 1 number)
   ↓
Output: predicted PM2.5_norm (normalized — we un-normalize later)
```

---

### The Training Loop — How the LSTM Learns

Training = showing the model thousands of examples and adjusting its internal numbers (called **weights**) so it gets better over time.

Each **epoch** = one full pass through all 262,944 training windows.

Here's what happens for each **batch** (512 examples at a time):

#### Step 1: Forward Pass
Feed the 512 windows through the LSTM. Get 512 predictions.

#### Step 2: Compute Loss
Compare predictions to the real answers. Use **MSE (Mean Squared Error)**:
```
loss = average of (prediction - actual)²
```
A high loss = bad predictions. We want loss to go DOWN.

#### Step 3: Backward Pass (Backpropagation)
Math magic: compute how much each weight contributed to the error. This gives us the **gradient** — the direction to nudge each weight.

#### Step 4: Gradient Clipping
Sometimes gradients become very large (explode). We cap them at `max_norm=1.0`. Think of it as a speed limiter on weight updates — prevents wild, unstable jumps.

#### Step 5: Optimizer Step (Adam)
The optimizer takes the gradients and updates every weight slightly. **Adam** is the most popular optimizer — it adapts the step size per weight based on recent history.

#### Step 6: Validation
After each epoch, run the model on the **validation set** (data it has NEVER trained on). Compute the **RMSE** in real μg/m³ units.
```
RMSE = √(average of (predicted_μg - actual_μg)²)
```
Lower RMSE = better predictions.

#### Step 7: Early Stopping
If the validation RMSE doesn't improve for **5 consecutive epochs**, stop training. This prevents wasting time and prevents the model from over-fitting.

#### Step 8: Save Best Model
Whenever validation RMSE improves, save the model weights to `best_lstm.pt`.

---

### Key Numbers (Hyperparameters)

| Setting | Value | What It Controls |
|---|---|---|
| Hidden Size | 256 | How many "memory cells" each LSTM layer has. Bigger = more capacity. |
| Num Layers | 3 | How many LSTM layers stacked on top of each other. Deeper = can learn more complex patterns. |
| Dropout | 0.2 | 20% of neurons randomly turned off during training. Forces the network not to rely on any single neuron. |
| Batch Size | 512 | How many examples to process before updating weights. Larger = more stable updates. |
| Learning Rate | 0.001 | How big each weight update step is. Too big → unstable. Too small → slow learning. |
| Max Epochs | 30 | Maximum number of full passes through the data. |
| Patience | 5 | Stop early if no improvement for 5 epochs in a row. |

---

## 🔁 Step 5B — `train_tcn.py` — "Teaching the TCN Brain"

### What is a TCN?

**TCN = Temporal Convolutional Network** — a completely different approach to sequence modeling.

Instead of reading the sequence step-by-step like LSTM, a TCN processes **all 24 hours simultaneously** using sliding filters (like how image recognition works, but on time instead of space).

#### The Analogy

Imagine looking at a photo. A convolution filter (small square) slides across the image, detecting edges, textures, or patterns everywhere at once — in parallel.

A TCN does the same thing along the **time axis** of our sequence.

#### The Two Key Tricks

**Trick 1: Causal Convolutions**

The filter at time *t* can ONLY see data from time *t* and EARLIER — never the future. This is essential: when predicting hour 25, we must not let the model peek at hour 26.

```
Normal conv:   sees past AND future  ← BAD for forecasting
Causal conv:   sees past ONLY        ← GOOD ✓
```

**Trick 2: Dilated Convolutions**

A regular filter with `kernel_size=5` sees 5 consecutive timesteps. But with **dilation**, the filter **skips steps**:

```
Dilation = 1: sees timesteps [t, t-1, t-2, t-3, t-4]       (5 steps)
Dilation = 2: sees timesteps [t, t-2, t-4, t-6, t-8]       (9 steps back)
Dilation = 4: sees timesteps [t, t-4, t-8, t-12, t-16]     (17 steps back)
Dilation = 8: sees timesteps [t, t-8, t-16, t-24, t-32]    (33 steps back)
```

By stacking 4 blocks with doubling dilation, the network can "see" up to 61 timesteps back using very few parameters. This covers our entire 24-hour window! ✅

#### Residual Connections (Skip Connections)

Each TCN block adds its **input directly to its output**:

```
output = learned_features(input) + input
```

This helps gradients flow backward during training without vanishing. It's the same idea as ResNet in image recognition.

---

### TCN Architecture Summary

```
Input (24 timesteps, 29 features)
  ↓ [transpose so time is the last axis for Conv1d]
TCN Block 1  (dilation=1)  — sees up to 5 timesteps
  ↓ + residual
TCN Block 2  (dilation=2)  — sees up to 13 timesteps
  ↓ + residual
TCN Block 3  (dilation=4)  — sees up to 29 timesteps
  ↓ + residual
TCN Block 4  (dilation=8)  — sees up to 61 timesteps ← covers full 24h window
  ↓ [take the output at the LAST timestep — it has seen everything]
Linear (256 → 1)
  ↓
Predicted PM2.5_norm
```

Each TCN block internally does:
```
Input
  → Causal Conv 1D → Batch Normalization → ReLU → Dropout
  → Causal Conv 1D → Batch Normalization → ReLU → Dropout
  + Input (residual)
  ↓
Output
```

**Batch Normalization** = normalizes the values between layers so training is more stable.

---

### LSTM vs TCN — The Key Difference

| | LSTM | TCN |
|---|---|---|
| **Processes sequence** | Step by step (left to right) | All at once (parallel) |
| **Memory mechanism** | Gated memory cells | Dilated convolutions |
| **Speed** | Slower (sequential) | Faster (parallel) |
| **Kaggle RMSE** | **14.935** (better) | 15.381 |

---

## 🔄 Step 5C — `train_gru.py` — "The Simpler RNN"

### What is a GRU?
**GRU = Gated Recurrent Unit.** It is a close cousin to the LSTM, but simpler.

Both LSTM and GRU read sequences step-by-step and maintain a hidden "memory". However, while an LSTM has 3 gates (Forget, Input, Output) and two separate memory channels (Cell State and Hidden State), the **GRU combines these into a simpler mechanism**.

### How it Works
A GRU only has 2 gates:
1. **The Reset Gate:** Decides how much of the past memory to *forget* before combining it with new data. (E.g., if wind changes drastically, ignore past pollution).
2. **The Update Gate:** Decides how much of the *newly computed* memory should overwrite the old memory.

### Why Try It?
Because it has fewer gates, a GRU has fewer parameters (weights) to learn. This means it **trains faster** and is less prone to over-fitting on small datasets.
- **Did it beat the LSTM?** No. Its Kaggle RMSE was ~21.7 (compared to LSTM's 14.9). This tells us that for Beijing air pollution, the extra complexity and long-term memory separation of the LSTM is mathematically necessary to capture complex weather interactions.

---

## 📊 Step 6 — Generating Predictions for the Test Set

### Why Do We Need a Separate Script for This?

The test set (`test_raw.csv`) is **different** from training data. It doesn't have 4 years of time-series — instead, Kaggle gives us **pre-built 24-hour windows** (already sliced) and asks us to predict PM2.5 for each.

### What These Scripts Do

**`generate_submission.py`** (LSTM version)  
**`generate_tcn_submission.py`** (TCN version)

Both do the same thing:

```
1. Load test_raw.csv
         ↓
2. Apply the EXACT same preprocessing as training:
   - Fill missing values using fitted_preprocessing.pkl (the saved recipe)
   - Encode wind direction as sin/cos
   - Normalize using training means and stds
         ↓
3. Reshape into (N, 24, 29) tensor
         ↓
4. Load the trained model (best_lstm.pt or best_tcn.pt)
         ↓
5. Forward pass (no training, just computing predictions)
         ↓
6. Un-normalize: prediction_real = prediction_norm × std_PM2.5 + mean_PM2.5
         ↓
7. Write submission.csv (id, PM2.5)
```

> [!IMPORTANT]
> We must use the **exact same** normalization parameters (means and stds) from training. If we re-compute them from the test set, the scale would be different and the model's predictions would be wrong.

---

## 🎯 Step 7 — `ensemble_submission.py` — "Combining Both Models"

### Why Ensemble?

Two models that make **different kinds of errors** can often combine to make a **better combined prediction** than either alone.

### The Formula
```
final_prediction = w_lstm × lstm_prediction + (1 - w_lstm) × tcn_prediction
```

Since LSTM scores better (14.935 vs 15.381), we generate multiple blends, all giving LSTM more weight:

| File | LSTM Weight | TCN Weight |
|---|---|---|
| `submission_ensemble_lstm50_tcn50.csv` | 50% | 50% |
| `submission_ensemble_lstm60_tcn40.csv` | 60% | 40% |
| `submission_ensemble_lstm70_tcn30.csv` | 70% | 30% |
| `submission_ensemble_lstm75_tcn25.csv` | 75% | 25% |
| `submission_ensemble_lstm80_tcn20.csv` | 80% | 20% |

All 5 are uploaded to Kaggle and we pick whichever scores best.

---

## 📈 How Performance Is Measured — RMSE Explained

**RMSE = Root Mean Squared Error**

```
RMSE = √(average of (predicted - actual)²)
```

For example, if we predict 60 μg/m³ but the actual value is 50 μg/m³:
- Error = 60 - 50 = 10
- Squared error = 100
- Do this for all predictions, average them, take the square root
- If RMSE = 14.9, it means on average our predictions are off by ~14.9 μg/m³

**Lower RMSE = Better Model.**

---

## 🔢 The 29 Features — Explained One by One

| # | Feature | Type | What It Represents |
|---|---|---|---|
| 1 | `PM2.5_norm` | Pollutant | Fine dust — this is also what we're predicting! |
| 2 | `PM10_norm` | Pollutant | Coarser dust — related to PM2.5 |
| 3 | `SO2_norm` | Pollutant | Sulfur dioxide — from burning coal |
| 4 | `NO2_norm` | Pollutant | Nitrogen dioxide — from traffic |
| 5 | `CO_norm` | Pollutant | Carbon monoxide — combustion |
| 6 | `O3_norm` | Pollutant | Ozone — forms in sunlight + pollution |
| 7 | `TEMP_norm` | Weather | Temperature — affects chemical reactions |
| 8 | `PRES_norm` | Weather | Pressure — affects how pollution disperses |
| 9 | `DEWP_norm` | Weather | Dew point — humidity-related |
| 10 | `RAIN_norm` | Weather | Rainfall — rain washes pollution away |
| 11 | `WSPM_norm` | Weather | Wind speed — wind disperses pollution |
| 12 | `wd_sin` | Wind | Sine of wind direction angle |
| 13 | `wd_cos` | Wind | Cosine of wind direction angle |
| 14 | `hour_sin` | Calendar | Sine of hour (0–23) |
| 15 | `hour_cos` | Calendar | Cosine of hour |
| 16 | `month_sin` | Calendar | Sine of month (1–12) |
| 17 | `month_cos` | Calendar | Cosine of month |
| 18–29 | `st_*` × 12 | Station | One-hot: which of the 12 stations this is |

---

## 🗺️ The Complete Flow — One Picture

```
RAW DATA (train_raw.csv)
       │
       ▼
┌─────────────────────────────────────────┐
│ preprocessing.py                        │
│  1. Split: before/after Sep 2015        │
│  2. Fill gaps (interp → seasonal → med) │
│  3. Encode wind direction as sin/cos    │
│  4. Normalize all numbers to mean=0     │
└────────────────┬────────────────────────┘
                 │  train_clean.pkl, val_clean.pkl
                 ▼
┌─────────────────────────────────────────┐
│ windowing.py                            │
│  For each station:                      │
│    slide a 24h window → predict hour 25 │
│  Add calendar sin/cos + station one-hot │
│  Result: 262,944 training examples      │
│          52,128 validation examples     │
│  Shape: (N, 24, 29) → scalar target     │
└────────────────┬────────────────────────┘
                 │  windows.npz
        ┌────────┴────────┐
        ▼                 ▼
┌──────────────┐  ┌──────────────┐
│ train_lstm.py│  │ train_tcn.py │
│  30 epochs   │  │  30 epochs   │
│  early stop  │  │  early stop  │
│  Adam + clip │  │  Adam + clip │
└──────┬───────┘  └──────┬───────┘
       │                 │
  best_lstm.pt      best_tcn.pt
       │                 │
       ▼                 ▼
┌─────────────────────────────────────────┐
│ generate_submission.py                  │
│ generate_tcn_submission.py              │
│  Apply same preprocessing to test set   │
│  Run model forward pass (no training)   │
│  Un-normalize predictions               │
│  Write CSV                              │
└────────────────┬────────────────────────┘
                 │  submission.csv, submission_tcn.csv
                 ▼
┌─────────────────────────────────────────┐
│ ensemble_submission.py                  │
│  Blend LSTM + TCN at 5 weight ratios    │
└────────────────┬────────────────────────┘
                 │
        📤 Upload to Kaggle!
```

---

## 🚀 Beyond the Core: V2 Upgrades and Extra Experiments

The core pipeline is great, but we wanted to push the boundaries and do extra research. The remaining files in `src/` fall into two categories: **V2 Upgrades** (making the model fundamentally smarter) and **Extra Analysis** (understanding the problem better).

---

### 1. The V2 Upgrades — Explained In Detail

#### Upgrade A: `feature_engineering.py` (Spoon-feeding Trends)
In the core pipeline, we give the model raw temperatures and wind speeds and say, "Figure out the trend yourself." In V2, we calculate the trends using math and explicitly feed them to the model as extra columns.

- **Lag Features:** We copy the PM2.5 column and shift it downward. This explicitly tells the model, *"Hey, at this exact moment, the pollution 1 hour ago was X, and 24 hours ago was Y."*
- **Rolling Averages (Means):** We calculate the average PM2.5 over the past 3 hours and 6 hours. If the 3-hour rolling average is way higher than the current hour, the AI mathematically instantly knows pollution is dropping.
- **Rolling Standard Deviation:** We calculate how wildly the pollution is swinging over the last 6 hours. High standard deviation = unpredictable storm conditions. Low standard deviation = stable day.

This script increases our features from 29 to 47.

#### Upgrade B: `train_lstm_v2.py` (Attention & Huber Loss)

We take the LSTM from Step 5A and add two major architectural upgrades:

**1. Temporal Attention Mechanism**
Standard LSTMs suffer from "forgetfulness". By the time they read hour 24, the memory of hour 1 has faded.
**Attention** fixes this. Instead of only relying on the final memory state at hour 24, the Attention layer looks at the LSTM outputs for *all 24 hours simultaneously*. It assigns a "weight" (a percentage of importance) to each hour. If hour 18 had a massive pollution spike that dictates the future, the Attention mechanism puts 80% of its focus on hour 18, and ignores the quiet hours. 

**2. Huber Loss (Ignoring Outliers)**
Normally, neural networks train using **MSE** (Mean Squared Error). MSE hates being wrong. 
If a freak dust storm hits and pollution spikes to 999 μg/m³ (an outlier), the model will guess 150. The error is 849. Because MSE squares the error ($849^2 = 720,801$), the loss explodes. The AI panics, destroys its current brain weights, and desperately tries to learn how to predict dust storms — ruining its ability to predict normal days.

**Huber Loss** fixes this. It acts exactly like MSE for small, normal errors. But if an error is massive (like the dust storm), it switches to a straight line (linear error). The AI essentially shrugs and says, *"That was a freak outlier, I'll take the penalty but I won't ruin my brain trying to learn it."* This makes training vastly more stable.

**`generate_lstm_v2_submission.py`** simply uses this upgraded brain and the upgraded 47 features to generate test set predictions.

---

### 2. Extra Analysis & Experiments

**`train_aqi_classifier.py`**:
Instead of predicting the exact raw number of PM2.5 (e.g., 145 μg/m³), this script trains a model to predict the **AQI Category** (e.g., "Good", "Unhealthy", "Hazardous"). It changes the problem from *Regression* (predicting a continuous line) to *Classification* (putting data into buckets).

**`window_ablation.py`**:
"Ablation" means taking parts away to see what breaks. This script trains the model multiple times using different window sizes (e.g., 6 hours, 12 hours, 24 hours, 48 hours). By charting the performance of each, we prove mathematically *why* 24 hours is the optimal sweet spot for context.

**`feature_importance.py`**:
Deep learning is often called a "black box" because it's hard to know what the AI is thinking. This script runs permutation tests to see which features the AI cares about the most. Does it care more about Temperature or Wind Speed? This helps explain the AI's logic to humans.

**`extended_graphs.py` & `poster_graphs.py`**:
These scripts don't do any machine learning. They analyze the final results and generate beautiful, high-quality visual charts for the project report and the final presentation poster.

---

## 🏁 How to Run Everything (In Order)

```bash
# Make sure you have the required packages
pip install numpy pandas scikit-learn torch

# Step 1 (optional but recommended): look at the data
python src/explore_data.py

# Step 2: clean and normalize
python src/preprocessing.py
# ✅ Creates data files in data/processed/

# Step 3: build windows
python src/windowing.py
# ✅ Creates windows.npz in data/processed/

# Step 4: compute baselines
python src/baselines.py
# ✅ Creates baseline_results.pkl in results/

# Step 5A: train LSTM (takes ~10–30 min on CPU)
python src/train_lstm.py
# ✅ Creates best_lstm.pt in models/ and lstm_results.pkl in results/

# Step 5B: train TCN (takes ~10–30 min on CPU)
python src/train_tcn.py
# ✅ Creates best_tcn.pt in models/ and tcn_results.pkl in results/

# Step 6A: generate LSTM predictions
python src/generate_submission.py
# ✅ Creates submission.csv in submissions/

# Step 6B: generate TCN predictions
python src/generate_tcn_submission.py
# ✅ Creates submission_tcn.csv in submissions/

# Step 7: generate ensemble predictions
python src/ensemble_submission.py
# ✅ Creates 5 ensemble CSV files in submissions/
```

> [!TIP]
> Steps 2, 3, 4 only need to run **once**. After that, you can re-train and re-generate as many times as you want without re-running preprocessing.

> [!WARNING]
> Training on CPU takes several minutes per epoch. If you have an NVIDIA GPU and CUDA installed, PyTorch will automatically use it — training will be 10–30× faster.
