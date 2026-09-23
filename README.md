# Telco Customer Churn — End-to-End Pipeline

## 1. Setup
python -m venv venv && source venv/Scripts/activate
pip install -r requirements.txt

## 2. Get the data
Download from Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place the CSV here:
data/WA_Fn-UseC_-Telco-Customer-Churn.csv

## 3. Explore the data (optional but recommended)
python src/01_eda.py

This surfaces the dataset's main gotcha: TotalCharges is stored as text
with blank strings for customers with tenure == 0, and confirms the
class imbalance (~27% churn).

## 4. Train the pipeline
python src/train.py

This will:
- Clean the data (fix TotalCharges, recode SeniorCitizen, drop dupes)
- Engineer two extra features (AvgChargesPerTenure, NumAddonServices)
- Split into stratified train/test (80/20)
- Cross-validate three candidate models (Logistic Regression, Random Forest, XGBoost) on ROC-AUC
- Evaluate all three on the held-out test set
- Run a RandomizedSearchCV hyperparameter search on the best candidate
- Refit the tuned pipeline on the full dataset
- Save the complete pipeline (preprocessing + model) to models/churn_pipeline.joblib
- Save all metrics to models/metrics.json

Because preprocessing lives inside the saved sklearn Pipeline, there's zero risk of train/serve skew.

## 5. Serve it
uvicorn src.serve:app --reload --port 8000

Then send a POST request to http://localhost:8000/predict with a customer's JSON data. It returns a churn probability and a Yes/No prediction.

## Notes / where to go next
- Imbalance is handled via class_weight="balanced" / tree weighting.
- Track prediction drift in production and retrain periodically.
- Add SHAP on the tuned model if per-prediction explanations are needed.