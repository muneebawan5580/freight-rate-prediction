# Freight Rate Prediction Challenge

This project is a machine learning solution for predicting freight rates based on historical load data. The goal is to train and validate a regression model, generate predictions for unseen validation loads, and forecast freight rates for a fixed December 2025 lane scenario.

## Project Overview

The dataset contains historical freight load information including:

- Pickup and delivery locations
- Pickup and delivery coordinates
- Distance
- Equipment type
- Weight
- Date
- Market index
- Quote signal
- Posted freight rate

The target variable is:

`posted_rate`

The final model is used to generate predictions for 12,000 unseen validation loads.

## Data Quality and Preprocessing

Initial exploratory data analysis was performed before model training.

Key observations:

- The training dataset contains 48,000 records.
- No duplicate rows were found.
- Missing values were present in `weight` and `market_index`.
- Negative weight values were identified and treated as sign errors by converting them to absolute values.
- Missing numerical values were handled using median imputation.
- Missing categorical values were handled using most-frequent imputation.
- Categorical features were transformed using One-Hot Encoding.
- Date was converted into `month`, `day`, and `day_of_week` features.

The `load_id` column was excluded from model training because it is an identifier rather than a predictive feature.

## Validation Strategy

A time-based validation strategy was used instead of a random train-test split.

Training data:

`January 2025 - September 2025`

Validation data:

`October 2025`

This approach was selected because freight rate prediction is a time-dependent problem. Training on earlier data and validating on later data provides a more realistic estimate of model performance on future loads and helps reduce temporal data leakage.

## Model Comparison

Several regression models were evaluated.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 138.92 | 651.62 | 0.8183 |
| Random Forest Regressor | 164.59 | 700.47 | 0.7900 |
| Gradient Boosting Regressor | 131.70 | 651.92 | 0.8181 |
| Tuned Gradient Boosting Regressor | **118.26** | **648.65** | **0.8199** |

The tuned Gradient Boosting Regressor was selected as the final model because it achieved the lowest Mean Absolute Error (MAE) on the October holdout set.

## Final Model

The selected Gradient Boosting model used:

- 200 estimators
- Learning rate: 0.05
- Maximum tree depth: 3
- Minimum samples per leaf: 5
- Random state: 42

After model selection, the final model was retrained using all 48,000 available historical records before generating predictions for the unseen validation dataset.

## Validation Predictions

The final model generated predictions for all 12,000 validation loads.

The output file is:

`validation_predictions.csv`

It contains exactly two columns:

- `load_id`
- `predicted_rate`

All predicted rates were checked to ensure that they were finite, positive, and contained no missing values.

## December 2025 Forecast

The model was also used to generate daily predicted freight rates for the fixed December scenario:

- Pickup: Lexington
- Delivery: Fort Wayne
- Distance: 360 miles
- Equipment: Dry Van
- Weight: 32,000 lb
- Dates: December 1-31, 2025

For model features that were not provided directly in the December input file, known lane coordinates and historical lane median values from the training data were used.

The final December predictions were approximately in the range of **$847-$854**, with an average predicted rate of approximately **$851.29**.

The scorer-generated December chart is included in the project results.

## Project Files

- `freight_rate_prediction.ipynb` - Data analysis, preprocessing, model training and prediction workflow
- `validation_predictions.csv` - Predictions for the 12,000 validation loads
- `december-chart-inputs.csv` - December 2025 fixed-lane predictions
- `score.py` - Provided validation/scoring script
- `requirements.txt` - Python dependencies

## Installation

Install the required Python packages using:

```bash
python -m pip install -r requirements.txt

