"""
inference.py
SageMaker inference script for the prebuilt sklearn container.
Defines the four functions the SageMaker model server calls.

Request contract (what your Streamlit app will send):
    JSON body: {"instances": [[f1, f2, f3, f4, f5, f6, f7, f8]]}
Response:
    JSON body: {"predictions": [value, ...]}
"""

import os
import json
import joblib
import numpy as np


def model_fn(model_dir):
    """Load the saved artifact from the model directory."""
    bundle = joblib.load(os.path.join(model_dir, "model.joblib"))
    return bundle  # dict: {"model": ..., "feature_names": [...]}


def input_fn(request_body, content_type="application/json"):
    """Parse the incoming request into a numpy array."""
    if content_type == "application/json":
        data = json.loads(request_body)
        print(f"---------Input----------{input}")
        instances = data["instances"]
        return np.array(instances, dtype=float)
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, bundle):
    """Run the prediction."""
    model = bundle["model"]
    return model.predict(input_data)


def output_fn(prediction, accept="application/json"):
    """Serialize the prediction as JSON."""
    print(f"------------prediction--------------{prediction}")
    return json.dumps({"predictions": prediction.tolist()})
