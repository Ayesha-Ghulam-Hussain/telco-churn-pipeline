"""
Steps 5-10: split -> train candidate models -> evaluate -> tune winner -> save.

Run: python src/train.py
Produces: models/churn_pipeline.joblib, models/metrics.json
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
from xgboost import XGBClassifier

from preprocessing import load_and_clean, prepare_xy, build_preprocessor

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_OUT = "models/churn_pipeline.joblib"
METRICS_OUT = "models/metrics.json"
RANDOM_STATE = 42


def get_candidates():
    """Baseline candidate models. class_weight/scale_pos_weight handle imbalance."""
    return {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", random_state=RANDOM_STATE,
        ),
    }


def evaluate(pipe, X_test, y_test, name):
    proba = pipe.predict_proba(X_test)[:, 1]
    preds = pipe.predict(X_test)
    metrics = {
        "model": name,
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
    }
    print(f"\n--- {name} ---")
    print(metrics)
    print(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    return metrics


def main():
    # 1-4: load, clean, engineer, build X/y
    df = load_and_clean(DATA_PATH)
    X, y = prepare_xy(df)

    # 5: split — stratified because churn is imbalanced (~27% positive)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 6-7: compare candidates with cross-validated ROC-AUC on the train set
    all_metrics = []
    fitted = {}
    for name, model in get_candidates().items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        print(f"{name}: CV ROC-AUC = {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        all_metrics.append(evaluate(pipe, X_test, y_test, name))

    # pick winner by test ROC-AUC
    best_name = max(all_metrics, key=lambda m: m["roc_auc"])["model"]
    print(f"\nBest baseline model: {best_name}")

    # 9: light hyperparameter tuning on the winner
    if best_name == "xgboost":
        param_dist = {
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [3, 4, 5, 6],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__subsample": [0.7, 0.8, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 1.0],
        }
        base_pipe = Pipeline([("preprocessor", preprocessor),
                               ("model", XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE))])
    elif best_name == "random_forest":
        param_dist = {
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [None, 8, 12, 20],
            "model__min_samples_split": [2, 5, 10],
        }
        base_pipe = Pipeline([("preprocessor", preprocessor),
                               ("model", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE))])
    else:
        param_dist = {
            "model__C": [0.01, 0.1, 1, 10],
            "model__penalty": ["l2"],
        }
        base_pipe = Pipeline([("preprocessor", preprocessor),
                               ("model", LogisticRegression(max_iter=1000, class_weight="balanced",
                                                             random_state=RANDOM_STATE))])

    search = RandomizedSearchCV(
        base_pipe, param_dist, n_iter=20, scoring="roc_auc",
        cv=cv, random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    tuned_pipe = search.best_estimator_
    print("\nBest params:", search.best_params_)

    tuned_metrics = evaluate(tuned_pipe, X_test, y_test, f"{best_name}_tuned")

    # 10: refit on ALL data (train+test) with best params, then save
    final_pipe = search.best_estimator_
    final_pipe.fit(X, y)

    joblib.dump(final_pipe, MODEL_OUT)
    with open(METRICS_OUT, "w") as f:
        json.dump({
            "baseline_comparison": all_metrics,
            "tuned_best": tuned_metrics,
            "best_params": search.best_params_,
        }, f, indent=2)

    print(f"\nSaved pipeline -> {MODEL_OUT}")
    print(f"Saved metrics -> {METRICS_OUT}")


if __name__ == "__main__":
    main()