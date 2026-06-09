"""Delete SageMaker endpoints. Defaults to both. Run: python delete_endpoint.py"""
import boto3, sys

ENDPOINTS = ["housing-regression", "bank-subscription"]
sm = boto3.client("sagemaker", region_name="us-east-1")

for name in (sys.argv[1:] or ENDPOINTS):
    try:
        sm.delete_endpoint(EndpointName=name)
        sm.delete_endpoint_config(EndpointConfigName=name)
        print("Deleted:", name)
    except Exception as e:
        print("Skipped", name, ":", e)