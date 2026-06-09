"""
deploy_classification.py
Deploys the Bank Marketing classifier as a serverless SageMaker endpoint.

Run train_classification.py first, then:
    python deploy_classification.py
"""

import json
import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.serverless import ServerlessInferenceConfig


ROLE_ARN = "arn:aws:iam::542129211116:role/SageMakerExecutionRole"
ENDPOINT_NAME = "bank-subscription"



def main():
    session = sagemaker.Session()
    region = session.boto_region_name
    bucket = session.default_bucket()
    print(f"Region: {region} | Bucket: {bucket}")

    model_s3 = session.upload_data(
        path="model_classification.tar.gz",
        bucket=bucket,
        key_prefix="models/classification",
    )
    print("Model uploaded to:", model_s3)

    sklearn_model = SKLearnModel(
        model_data=model_s3,
        role=ROLE_ARN,
        entry_point="inference_classification.py",
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=session,
    )

    serverless_config = ServerlessInferenceConfig(
        memory_size_in_mb=2048,
        max_concurrency=5,
    )

    print("Deploying serverless endpoint.")
    sklearn_model.deploy(
        serverless_inference_config=serverless_config,
        endpoint_name=ENDPOINT_NAME,
    )
    print("Endpoint is live:", ENDPOINT_NAME)

    # Smoke test using the real sample row saved during training
    with open("sample_classification.json") as f:
        sample = json.load(f)
    runtime = boto3.client("sagemaker-runtime", region_name=region)
    resp = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(sample),
    )
    print("Test prediction:", resp["Body"].read().decode())


if __name__ == "__main__":
    main()
