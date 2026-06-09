"""
inference_classification.py
SageMaker inference handler for the Bank Marketing classifier.

Request:  {"instances": [{"age": 35, "job": "admin.", ...}, ...]}
Response: {"predictions": [{"label": 1, "probability": 0.83}, ...]}
The probability is the chance of subscribing (class 1).
"""

import os
import json
import joblib
import pandas as pd


def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(request_body, content_type="application/json"):
    if content_type == "application/json":
        data = json.loads(request_body)
        return pd.DataFrame(data["instances"])
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_df, bundle):
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    cat_cols = bundle.get("categorical_cols", [])
    num_cols = bundle.get("numeric_cols", [])
    input_df = input_df.reindex(columns=feature_names)
    # Force dtypes to match training, or the encoder misreads a categorical
    # column as numeric and crashes on np.isnan.
    for c in cat_cols:
        input_df[c] = input_df[c].astype(str)
    for c in num_cols:
        input_df[c] = pd.to_numeric(input_df[c], errors="coerce")
    labels = model.predict(input_df)
    probs = model.predict_proba(input_df)[:, 1]
    return labels, probs


def output_fn(prediction, accept="application/json"):
    labels, probs = prediction
    out = [
        {"label": int(l), "probability": round(float(p), 4)}
        for l, p in zip(labels, probs)
    ]
    return json.dumps({"predictions": out})
