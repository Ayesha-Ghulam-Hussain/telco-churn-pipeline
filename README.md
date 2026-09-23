# Telco Customer Churn — End-to-End Pipeline

Predicts which customers are likely to churn using the Telco Customer Churn dataset (~7,043 rows). Covers data cleaning, exploratory analysis, model training/tuning, evaluation, and a servable API.

## Results

- **Best model:** XGBoost (tuned via RandomizedSearchCV)
- **Test-set ROC-AUC:** 0.8482
- **Precision / Recall / F1 (churn class):** 0.67 / 0.53 / 0.59
- Churn is strongly driven by **contract type** (month-to-month churns far more than longer contracts), **low tenure**, and **higher monthly charges** — see charts below.

## Exploratory Data Analysis

**Churn rate by contract type** — month-to-month customers churn at a much higher rate than 1- or 2-year contracts.
![Churn by contract](images/churn_by_contract.png)

**Tenure distribution by churn status** — churners cluster heavily at low tenure (newer customers).
![Tenure distribution](images/tenure_distribution.png)

**Monthly charges by churn status** — churners tend to pay more per month.
![Monthly charges by churn](images/monthly_charges_by_churn.png)

**Correlation heatmap** — tenure correlates negatively with churn; charges correlate positively.
![Correlation heatmap](images/correlation_heatmap.png)

## Model Evaluation

**Confusion matrix** on the held-out test set.
![Confusion matrix](images/confusion_matrix.png)

**ROC curve** — AUC 0.8482, well above the random-guess diagonal.
![ROC curve](images/roc_curve.png)

## 1. Setup