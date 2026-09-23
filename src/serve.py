"""
Step 11: Serve the trained pipeline behind a REST API.

Run: uvicorn src.serve:app --reload --port 8000
Then POST a customer record to /predict.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from preprocessing import engineer_features, get_feature_columns

MODEL_PATH = "models/churn_pipeline.joblib"

app = FastAPI(title="Churn Prediction API")
model = joblib.load(MODEL_PATH)


class CustomerRecord(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    gender: str
    SeniorCitizen: str          # "Yes" / "No"
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(record: CustomerRecord):
    df = pd.DataFrame([record.dict()])
    df = engineer_features(df)
    numeric_all, categorical_all = get_feature_columns()
    X = df[numeric_all + categorical_all]

    proba = model.predict_proba(X)[0, 1]
    pred = int(proba >= 0.5)

    return {
        "churn_probability": round(float(proba), 4),
        "churn_prediction": "Yes" if pred else "No",
    }