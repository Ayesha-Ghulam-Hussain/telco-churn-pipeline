"""
Step 2-4: Cleaning + feature engineering + preprocessing pipeline.

This module is imported by train.py and serve.py so the exact same
transformations are applied at train time and inference time — no
train/serve skew.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

TARGET = "Churn"
ID_COL = "customerID"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


def load_and_clean(path: str) -> pd.DataFrame:
    """Load raw CSV and fix known data-quality issues."""
    df = pd.read_csv(path)

    # TotalCharges is stored as text with blank strings for customers
    # with tenure == 0. Coerce to numeric; blanks become NaN, which the
    # imputer in the pipeline will handle.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # SeniorCitizen is 0/1 but semantically categorical — treat as such.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # Drop exact duplicate rows, if any.
    df = df.drop_duplicates()

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a couple of derived features that tend to help on this dataset."""
    df = df.copy()

    # Average monthly spend implied by total charges vs tenure — catches
    # customers whose recent bill differs a lot from their historical average.
    df["AvgChargesPerTenure"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    # Count of subscribed add-on services — a simple "engagement" signal.
    addon_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["NumAddonServices"] = (df[addon_cols] == "Yes").sum(axis=1)

    return df


def get_feature_columns():
    return NUMERIC_FEATURES + ["AvgChargesPerTenure", "NumAddonServices"], CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    numeric_all, categorical_all = get_feature_columns()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_all),
        ("cat", categorical_transformer, categorical_all),
    ])

    return preprocessor


def prepare_xy(df: pd.DataFrame):
    """Full pipeline from raw-loaded df to model-ready X, y."""
    df = engineer_features(df)
    y = (df[TARGET] == "Yes").astype(int)
    numeric_all, categorical_all = get_feature_columns()
    X = df[numeric_all + categorical_all]
    return X, y