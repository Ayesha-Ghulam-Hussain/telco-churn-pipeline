"""
Generates EDA charts for the Telco Customer Churn dataset and saves them
as PNG files in images/.

Run: python src/02_charts.py
Expects: data/WA_Fn-UseC_-Telco-Customer-Churn.csv
Produces: images/churn_by_contract.png, images/tenure_distribution.png,
          images/monthly_charges_by_churn.png, images/correlation_heatmap.png
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
IMAGES_DIR = "images"


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df = df.drop_duplicates()
    return df


def chart_churn_by_contract(df):
    """Bar chart: churn rate by contract type."""
    plt.figure(figsize=(7, 5))
    churn_rate = (
        df.assign(ChurnFlag=(df["Churn"] == "Yes").astype(int))
        .groupby("Contract")["ChurnFlag"].mean()
        .sort_values(ascending=False)
    )
    sns.barplot(x=churn_rate.index, y=churn_rate.values, hue=churn_rate.index,
                palette="Reds_r", legend=False)
    plt.title("Churn Rate by Contract Type")
    plt.ylabel("Churn Rate")
    plt.xlabel("Contract")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "churn_by_contract.png"), dpi=150, bbox_inches="tight")
    plt.close()


def chart_tenure_distribution(df):
    """Histogram: tenure distribution, split by churn status."""
    plt.figure(figsize=(7, 5))
    sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True, multiple="stack")
    plt.title("Customer Tenure Distribution by Churn Status")
    plt.xlabel("Tenure (months)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "tenure_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()


def chart_monthly_charges_by_churn(df):
    """Boxplot: monthly charges compared across churn status."""
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", hue="Churn",
                palette="Set2", legend=False)
    plt.title("Monthly Charges by Churn Status")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "monthly_charges_by_churn.png"), dpi=150, bbox_inches="tight")
    plt.close()


def chart_correlation_heatmap(df):
    """Heatmap of correlations between numeric features and churn."""
    plt.figure(figsize=(6, 5))
    corr_df = df.copy()
    corr_df["Churn"] = (corr_df["Churn"] == "Yes").astype(int)
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    corr = corr_df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "correlation_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    df = load_and_clean(DATA_PATH)

    chart_churn_by_contract(df)
    print("Saved images/churn_by_contract.png")

    chart_tenure_distribution(df)
    print("Saved images/tenure_distribution.png")

    chart_monthly_charges_by_churn(df)
    print("Saved images/monthly_charges_by_churn.png")

    chart_correlation_heatmap(df)
    print("Saved images/correlation_heatmap.png")

    print("\nAll charts generated.")


if __name__ == "__main__":
    main()