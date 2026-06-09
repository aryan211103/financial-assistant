"""
train_regression.py
Trains a Random Forest Regressor on the California Housing dataset,
evaluates it, saves the model, and packages it into model_regression.tar.gz
ready for SageMaker deployment.

Run:  python train_regression.py
"""

import os
import tarfile
import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

OUT_DIR = "model_regression"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    # 1. Load data (8 numeric features, target is median house value in 100k USD)
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target
    feature_names = list(X.columns)
    print("Features:", feature_names)
    print("Rows:", X.shape[0])

    # 2. Light EDA (enough to show in your notebook / demo)
    print("\nFeature summary:")
    print(X.describe().T[["mean", "std", "min", "max"]])
    print("\nTarget summary:")
    print(y.describe())

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # NOTE: Random Forest is a tree model. It splits on thresholds, so feature
    # scaling does NOT change results and is not needed. The assignment says
    # standardize "where appropriate" - for trees it is not appropriate, so we
    # intentionally skip it. (Say exactly this if asked in the interview.)

    # 4. Train
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 5. Evaluate
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    print("\n=== Test metrics ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R2  : {r2:.4f}")

    # 6. Save model + feature names together
    joblib.dump(
        {"model": model, "feature_names": feature_names},
        os.path.join(OUT_DIR, "model.joblib"),
    )
    print(f"\nSaved model to {OUT_DIR}/model.joblib")

    # 7. Package into the tarball SageMaker expects
    with tarfile.open("model_regression.tar.gz", "w:gz") as tar:
        tar.add(os.path.join(OUT_DIR, "model.joblib"), arcname="model.joblib")
    print("Packaged model_regression.tar.gz")


if __name__ == "__main__":
    main()
