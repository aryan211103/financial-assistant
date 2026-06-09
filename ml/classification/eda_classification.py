"""
eda_classification.py
Comprehensive exploratory data analysis for the UCI Bank Marketing dataset.
Prints a full report, saves it to a text file, and saves plots if matplotlib
is available.

Run from ml/classification:
    python eda_classification.py
"""

import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

OUT = "eda_classification_outputs"
os.makedirs(OUT, exist_ok=True)


def main():
    ds = fetch_ucirepo(id=222)
    X = ds.data.features.copy()
    y_raw = ds.data.targets.iloc[:, 0]
    y = (y_raw.astype(str).str.lower() == "yes").astype(int)

    categorical = [c for c in X.columns if X[c].dtype == "object"]
    numeric = [c for c in X.columns if c not in categorical]

    lines = []

    def log(s=""):
        print(s)
        lines.append(str(s))

    log("=== Bank Marketing EDA ===")
    log(f"Shape: {X.shape[0]} rows, {X.shape[1]} features")
    log("")
    log("Column dtypes:")
    log(X.dtypes.to_string())
    log("")
    log("Missing values per column:")
    log(X.isna().sum().to_string())
    log("")
    log("Class balance (target = subscribed):")
    counts = y.value_counts().rename({0: "no", 1: "yes"})
    log(counts.to_string())
    log(f"Subscribe rate: {y.mean():.3f}  (this is imbalanced)")
    log("")
    log(f"Numeric features ({len(numeric)}): {numeric}")
    log("Numeric summary statistics:")
    log(X[numeric].describe().T.to_string())
    log("")
    log(f"Categorical features ({len(categorical)}): {categorical}")
    for c in categorical:
        log(f"\nValue counts for '{c}':")
        log(X[c].value_counts(dropna=False).to_string())

    with open(os.path.join(OUT, "eda_report.txt"), "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved report to {OUT}/eda_report.txt")

    # Optional plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        counts.plot(kind="bar")
        plt.title("Class balance (subscribed)")
        plt.ylabel("Count")
        plt.savefig(os.path.join(OUT, "class_balance.png"), bbox_inches="tight")
        plt.clf()

        X["duration"].hist(bins=50)
        plt.title("Call duration distribution")
        plt.xlabel("seconds")
        plt.ylabel("count")
        plt.savefig(os.path.join(OUT, "duration_distribution.png"), bbox_inches="tight")
        plt.clf()

        print(f"Saved plots to {OUT}/")
    except ImportError:
        print("matplotlib not installed, skipped plots. Run: pip install matplotlib")


if __name__ == "__main__":
    main()