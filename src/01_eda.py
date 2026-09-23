"""
Step 1: Exploratory Data Analysis on the Telco Customer Churn dataset.

Run: python src/01_eda.py
Expects: data/WA_Fn-UseC_-Telco-Customer-Churn.csv
"""
import pandas as pd

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("SHAPE:", df.shape)
    print("=" * 60)

    print("\nDTYPES:")
    print(df.dtypes)

    # TotalCharges is read as object because of blank strings for
    # customers with tenure == 0 (brand-new customers). This is the
    # classic "gotcha" in this dataset.
    print("\nRows where TotalCharges is blank/whitespace:")
    blank_mask = df["TotalCharges"].astype(str).str.strip() == ""
    print(df.loc[blank_mask, ["customerID", "tenure", "TotalCharges"]])

    print("\nMISSING VALUES (raw):")
    print(df.isna().sum()[df.isna().sum() > 0])

    print("\nTARGET BALANCE (Churn):")
    print(df["Churn"].value_counts(normalize=True).round(3))

    print("\nCATEGORICAL CARDINALITY:")
    cat_cols = df.select_dtypes(include="object").columns
    for c in cat_cols:
        print(f"  {c}: {df[c].nunique()} unique -> {df[c].unique()[:6]}")

    print("\nNUMERIC SUMMARY (tenure, MonthlyCharges):")
    print(df[["tenure", "MonthlyCharges"]].describe())

    print("\nDUPLICATE customerIDs:", df["customerID"].duplicated().sum())


if __name__ == "__main__":
    main()