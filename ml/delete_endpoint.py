"""
delete_endpoint.py
Run this the moment you finish recording the demo. Stops the hourly billing.

Run:  python delete_endpoint.py
"""

import boto3

ENDPOINT_NAME = "housing-regression"  # change per endpoint

sm = boto3.client("sagemaker", region_name="us-east-1")
sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
sm.delete_endpoint_config(EndpointConfigName=ENDPOINT_NAME)
print("Deleted endpoint and config:", ENDPOINT_NAME)
