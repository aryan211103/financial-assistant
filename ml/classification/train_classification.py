"""
train_classification.py
Trains a Logistic Regression classifier on the UCI Bank Marketing dataset
to predict whether a customer subscribes (yes / no). Encodes categoricals,
scales numerics, evaluates, saves the model, and packages it for SageMaker.

First:  pip install ucimlrepo
Run:    python train_classification.py
"""

import os
import json
import tarfile
import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

OUT_DIR = "model_classification"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    # 1. Load data from UCI (Bank Marketing is dataset id 222)
    ds = fetch_ucirepo(id=222)
    X = ds.data.features.copy()
    y_raw = ds.data.targets.iloc[:, 0]          # the 'y' column: yes / no
    y = (y_raw.astype(str).str.lower() == "yes").astype(int)
    print("Rows:", X.shape[0], "| Features:", X.shape[1])
    print("Subscribe rate:", round(float(y.mean()), 3))

    # 2. Split columns by type (auto, so it works regardless of UCI variant)
    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    print("Categorical:", categorical_cols)
    print("Numeric:", numeric_cols)

    # 3. Preprocess + model in one pipeline
    pre = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ])
    pipe = Pipeline(steps=[
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    # 4. Train / test split (stratified to keep the class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Train
    pipe.fit(X_train, y_train)

    # 6. Evaluate
    preds = pipe.predict(X_test)
    print("\n=== Test metrics ===")
    print("Accuracy :", round(accuracy_score(y_test, preds), 4))
    print("Precision:", round(precision_score(y_test, preds), 4))
    print("Recall   :", round(recall_score(y_test, preds), 4))
    print("F1       :", round(f1_score(y_test, preds), 4))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, preds))

    # 7. Save the pipeline plus the exact feature column order
    joblib.dump(
        {
            "model": pipe,
            "feature_names": list(X.columns),
            "categorical_cols": categorical_cols,
            "numeric_cols": numeric_cols,
        },
        os.path.join(OUT_DIR, "model.joblib"),
    )
    print(f"\nSaved model to {OUT_DIR}/model.joblib")

    # 8. Save one real test row so the deploy smoke test uses valid columns
    sample_row = X_test.iloc[[0]].to_dict(orient="records")
    with open("sample_classification.json", "w") as f:
        json.dump({"instances": sample_row}, f)
    print("Saved sample_classification.json")

    # 9. Package for SageMaker
    with tarfile.open("model_classification.tar.gz", "w:gz") as tar:
        tar.add(os.path.join(OUT_DIR, "model.joblib"), arcname="model.joblib")
    print("Packaged model_classification.tar.gz")


if __name__ == "__main__":
    main()
