# Predicting-Agricultural-Crop-Production-by-Yield-using-Machine-Learning
Supervised ML pipeline predicting district-level Malaysian crop production across 11,000+ agrofood records (data.gov.my). Benchmarked 5 regression models (XGBoost, LightGBM, Random Forest, Decision Tree, CatBoost) with residual &amp; LOESS diagnostics, achieving R² = 0.997.

# Predicting Agricultural Crop Production by Yield using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![ML](https://img.shields.io/badge/Models-XGBoost%20%7C%20LightGBM%20%7C%20CatBoost-orange)

## Overview
An end-to-end supervised machine learning project that predicts district-level crop production (in tonnes) across Malaysia using official agrofood datasets from `data.gov.my`. By integrating historical crop production and planted area records, this project benchmarks five regression models to identify high-performing agricultural regions and uncover key yield drivers.

## Methodology & Pipeline
* **Data Wrangling & Integration:** Merged raw district production and planted area datasets (11,000+ entries), standardized categorical text, resolved duplicate keys via aggregation, and filtered invalid zero-area entries.
* **Feature Engineering:** Engineered yield ratios (`production / planted_area`), log-transformed skewed target and area variables (`log1p`), computed district-level mean yields, and applied one-hot encoding across 135 districts and 114 crop species.
* **Model Benchmarking & Tuning:** Trained and tuned five regression models using `RandomizedSearchCV` with cross-validation:
  * **XGBoost Regressor:** $R^2$: 0.997 | MAE: 13.38t | RMSE: 191.90t
  * **Decision Tree Regressor:** $R^2$: 0.996 | MAE: 10.53t | RMSE: 234.53t
  * **LightGBM Regressor:** $R^2$: 0.996 | MAE: 18.75t | RMSE: 241.95t
  * **Random Forest Regressor:** $R^2$: 0.994 | MAE: 12.12t | RMSE: 286.39t
  * **CatBoost Regressor:** $R^2$: 0.980 | MAE: 34.04t | RMSE: 526.83t
* **Diagnostics:** Conducted residual distribution analysis, LOESS smoothing, and feature importance comparisons to evaluate model reliability and heteroscedasticity at extreme production volumes.

## Tech Stack
* **Language:** Python
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `statsmodels`, `matplotlib`, `seaborn`
