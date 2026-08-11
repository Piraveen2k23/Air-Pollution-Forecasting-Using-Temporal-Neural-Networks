# 🌍 Air Pollution Forecasting: A Complete Beginner's Guide

Welcome to the **PM2.5 Air Pollution Forecasting Project**! 
If you know absolutely nothing about this project (or even machine learning), this guide will explain everything from A to Z in plain English.

---

## 1. What Are We Trying to Do?

**The Goal:** Predict PM2.5 levels **1 hour into the future** for 12 different weather stations across Beijing. 

**What is PM2.5?** 
PM2.5 refers to tiny pollution particles in the air (smaller than 2.5 micrometers). They are the most dangerous type of air pollution because they are small enough to enter the bloodstream through the lungs.

**Why does this matter?**
If we can accurately predict when PM2.5 levels will spike, people can stay indoors, hospitals can prepare, and governments can temporarily shut down polluting factories.

**The Analogy:** 
Imagine you are a doctor monitoring a patient's heart rate every minute. You want to predict if their heart rate will spike in the next hour by looking at the last 24 hours of their data. We are doing the exact same thing, but instead of heart rate, we are monitoring air pollution!

---

## 2. What Data Are We Using?

We are using 3 years of hourly readings (2013–2016) from 12 different air quality monitoring stations in Beijing.

For every single hour, we have data on:
- **Pollutants:** PM2.5 (what we are predicting), PM10, SO2, NO2, CO, O3.
- **Weather:** Temperature, Air Pressure, Humidity (Dew Point), Rainfall, Wind Speed, and Wind Direction.

---

## 3. How Does the Project Work? (The 5-Step Pipeline)

To teach a computer how to predict pollution, we pass the data through a 5-step "pipeline". 

### Step 1: Preprocessing (Cleaning the Data)
Real-world data is messy. Sometimes sensors break and leave blank spots (missing values) in the data. 
- We fill small gaps by connecting the dots (if it was 40 at 1 PM and 60 at 3 PM, we guess it was 50 at 2 PM).
- We fill large gaps by looking at historical averages ("What is the usual pollution for a March morning?").
- We also "normalize" the data, meaning we scale all numbers (temperatures, wind speeds, pollution) to a similar range so the AI doesn't get confused.

### Step 2: Windowing (Creating Dashcam Clips)
AI models need context. We can't just give it the current temperature and ask for next hour's pollution. 
We take the continuous timeline of data and chop it into **24-hour sliding windows**. 
Think of these as 24-hour "dashcam clips". We show the AI the clip and ask: *"Based on the last 24 hours, what will the pollution be in the 25th hour?"* We slide the window forward one hour at a time to create hundreds of thousands of examples.

### Step 3: Baselines (The "Dumb" Benchmarks)
Before building complex AI, we establish "dumb" baselines to see if our AI is actually learning anything useful.
- **Persistence Baseline:** This simply guesses that *the next hour's pollution will be exactly the same as right now*. 
- **Ridge Baseline:** A basic math equation that looks at the current weather and pollution to make a guess, but it has no memory of the past 24 hours.
*If our complex AI cannot beat these simple methods, it is useless!*

### Step 4: Neural Networks (The AI Brains)
We train different types of Deep Learning models to solve the problem:
- **FFNN (Feed-Forward):** A basic brain. It only looks at the most recent hour (no memory). It performs poorly.
- **GRU & LSTM:** These models have **memory**. They look at the 24-hour "dashcam clip" step-by-step, remembering important trends (like a rising temperature) and forgetting useless noise. LSTMs are the workhorse of this project and perform very well!
- **TCN (Temporal Convolutional Network):** Instead of reading the 24 hours step-by-step, it scans the entire 24-hour period at once using filters, finding patterns much faster.

### Step 5: Ensembling (Teamwork)
Different AI models make different mistakes. An LSTM might overestimate pollution during a storm, while a TCN might underestimate it. 
We take the predictions from the LSTM and the TCN and **average them together**. For example, blending the predictions using an **80% LSTM and 20% TCN (lstm80_tcn20)** ratio proved to be the winning combination, achieving the **best overall score on Kaggle**! This is called "ensembling." Just like getting a second opinion from another doctor, combining models gives us the most accurate final prediction.

---

## 4. How the Project is Organized

The project files are neatly organized into folders so you can easily find what you need:

- 📁 **`data/`**: Contains the raw downloaded CSV files and the cleaned, processed data files.
- 📁 **`src/`**: The Python scripts that run the 5-step pipeline.
- 📁 **`models/`**: The saved "brains" (weights) of the trained AI models.
- 📁 **`results/`**: Logs and evaluation scores of how well the models performed.
- 📁 **`submissions/`**: The final CSV files containing our predictions, ready to be uploaded to Kaggle.
- 📁 **`docs/`**: Technical guides and team task divisions.

---

## 5. How to Run the Code

To run the whole pipeline yourself from start to finish, you just run the scripts in the `src/` folder in this exact order:

1. **Clean the data:** `python src/preprocessing.py`
2. **Create the windows:** `python src/windowing.py`
3. **Run the dumb baselines:** `python src/baselines.py`
4. **Train the AI models:** 
   - `python src/train_lstm.py`
   - `python src/train_tcn.py`
5. **Generate the predictions:**
   - `python src/generate_submission.py`
   - `python src/generate_tcn_submission.py`
6. **Blend them together:** `python src/ensemble_submission.py`

*(Note: Advanced versions like `feature_engineering.py` and `train_lstm_v2.py` exist for building an even smarter model!)*

---
**Summary:** We clean messy weather data, chop it into 24-hour chunks, train memory-based AI models to find patterns, and combine their guesses to accurately predict dangerous air pollution spikes before they happen!
