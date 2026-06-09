"""
eda_regression.py
Comprehensive exploratory data analysis for the California Housing dataset.
Prints a full report, saves it to a text file, and saves plots if matplotlib
is available.

Run from ml/regression:
    python eda_regression.py
"""

import os
import pandas as pd
from sklearn.datasets import fetch_california_housing

OUT = "eda_regression_outputs"
os.makedirs(OUT, exist_ok=True)


def main():
    data = fetch_california_housing(as_frame=True)
    df = data.frame                 # 8 features plus the target column
    target = "MedHouseVal"

    lines = []

    def log(s=""):
        print(s)
        lines.append(str(s))

    log("=== California Housing EDA ===")
    log(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    log("")
    log("Column dtypes:")
    log(df.dtypes.to_string())
    log("")
    log("Missing values per column:")
    log(df.isna().sum().to_string())
    log("")
    log("Numeric summary statistics:")
    log(df.describe().T.to_string())
    log("")
    log(f"Target ({target}) distribution:")
    log(df[target].describe().to_string())
    log("Note: target is median house value in units of 100,000 USD, capped at 5.0.")
    log("")
    log("Correlation of each feature with the target (sorted):")
    corr = df.corr(numeric_only=True)[target].sort_values(ascending=False)
    log(corr.to_string())

    with open(os.path.join(OUT, "eda_report.txt"), "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved report to {OUT}/eda_report.txt")

    # Optional plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df[target].hist(bins=50)
        plt.title("Median house value distribution")
        plt.xlabel("Value (100k USD)")
        plt.ylabel("Count")
        plt.savefig(os.path.join(OUT, "target_distribution.png"), bbox_inches="tight")
        plt.clf()

        c = df.corr(numeric_only=True)
        plt.imshow(c, cmap="coolwarm", vmin=-1, vmax=1)
        plt.xticks(range(len(c)), c.columns, rotation=90)
        plt.yticks(range(len(c)), c.columns)
        plt.colorbar()
        plt.title("Feature correlation")
        plt.savefig(os.path.join(OUT, "correlation_heatmap.png"), bbox_inches="tight")
        plt.clf()

        print(f"Saved plots to {OUT}/")
    except ImportError:
        print("matplotlib not installed, skipped plots. Run: pip install matplotlib")


if __name__ == "__main__":
    main()