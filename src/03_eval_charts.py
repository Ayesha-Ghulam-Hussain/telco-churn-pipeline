"""
Generates two model-evaluation charts and saves them as PNGs:
  - images/confusion_matrix.png
  - images/roc_curve.png

Run: python src/03_eval_charts.py
Expects: models/churn_pipeline.joblib (from train.py)
         data/WA_Fn-UseC_-Telco-Customer-Churn.csv

Note: the saved pipeline was refit on the FULL dataset at the end of
train.py (see its Step 10). To produce an honest test-set evaluation
here, this script re-creates the exact same train/test split (same
random_state) and reports metrics on that held-out slice, matching
what train.py printed to your terminal during training.
"""
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score, RocCurveDisplay,
)

from preprocessing import load_and_clean, prepare_xy

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "models/churn_pipeline.joblib"
IMAGES_DIR = "images"
RANDOM_STATE = 42


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)

    df = load_and_clean(DATA_PATH)
    X, y = prepare_xy(df)

    # Same split as train.py, so this matches the metrics printed during training.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipe = joblib.load(MODEL_PATH)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix — Churn Prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved images/confusion_matrix.png")

    # --- ROC curve ---
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc).plot(ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    ax.set_title(f"ROC Curve — AUC = {auc:.4f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "roc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved images/roc_curve.png")

    print(f"\nTest-set ROC-AUC: {auc:.4f}")


if __name__ == "__main__":
    main()