# Air Pollution Forecasting Using Temporal Neural Networks

This project builds a time-series forecasting pipeline for air pollution prediction using temporal neural networks. It includes preprocessing, window generation, baseline evaluation, and experiment scripts for forecasting PM2.5 one hour ahead.

The project uses the Beijing Multi-Site Air Quality Dataset from the UCI Machine Learning Repository, which contains hourly measurements of pollutants (PM2.5, PM10, SO₂, NO₂, CO, O₃) and meteorological variables (temperature, pressure, humidity, wind direction and speed) from 12 monitoring stations in Beijing, covering March 2013 to February 2017.

## Project structure
- preprocessing.py: cleaning and feature engineering pipeline
- windowing.py: sliding-window dataset construction
- baselines.py: persistence and ridge baseline evaluation
- train_raw.csv: raw training data
- sample_submission.csv: submission template

## Setup
Install dependencies with:

```bash
pip install -r requirements.txt
```

Run the pipeline in order:

```bash
python preprocessing.py
python windowing.py
python baselines.py
```

## Notes
The scripts expect the training CSV files to be present in the project root.
