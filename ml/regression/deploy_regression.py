"""
deploy_regression.py
Uploads the trained model to S3 and deploys it as a real-time SageMaker endpoint.
Run:  python deploy_regression.py
"""

import json
import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.serverless import ServerlessInferenceConfig

# Sagemaker connection
ROLE_ARN = "arn:aws:iam::542129211116:role/SageMakerExecutionRole"
ENDPOINT_NAME = "housing-regression"

def main():
    session = sagemaker.Session()
    region = session.boto_region_name
    bucket = session.default_bucket()
    print(f"Region: {region} | Bucket: {bucket}")

    # 1. Upload the local model tarball to S3
    model_s3 = session.upload_data(
        path="model_regression.tar.gz",
        bucket=bucket,
        key_prefix="models/regression",
    )
    print("Model uploaded to:", model_s3)

    # 2. Wrap it in the prebuilt sklearn container
    sklearn_model = SKLearnModel(
        model_data=model_s3,
        role=ROLE_ARN,
        entry_point="inference.py",
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=session,
    )

    # 3. Deploying a real-time endpoint
    serverless_config = ServerlessInferenceConfig(
        memory_size_in_mb=2048,
        max_concurrency=5,
    )
    print("Deploying serverless endpoint. This takes a few minutes...")
    sklearn_model.deploy(
        serverless_inference_config=serverless_config,
        endpoint_name=ENDPOINT_NAME,
    )
    print("Endpoint is live:", ENDPOINT_NAME)

    # 4. Quick smoke test
    runtime = boto3.client("sagemaker-runtime", region_name=region)
    sample = {"instances": [[8.3252, 41.0, 6.984, 1.023, 322.0, 2.555, 37.88, -122.23]]}
    resp = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(sample),
    )
    print("Test prediction:", resp["Body"].read().decode())


if __name__ == "__main__":
    main()
