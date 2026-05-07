"""Deploy an approved registered model to a real-time SageMaker Endpoint.

Production equivalent of `local_serve.py`. Where local_serve loads a
bundle directly off disk and calls model_fn, this:
  1. References a registered model package in the Model Package Group
  2. Tells SageMaker to spin up an inference container (the SKLearn DLC
     for our sklearn bundles) that hosts model_fn behind HTTPS
  3. Returns a Predictor object you can `.predict(...)` against

This is what "model.deploy driver" means in the README — a small wrapper
that calls the SageMaker SDK's `model.deploy()`. Same shape as
`local_train_dlc.py` is for training.

Usage:
    python deploy_endpoint.py \\
        --model-package-arn arn:aws:sagemaker:us-east-1:ACCT:model-package/sage-baker-sklearn/3 \\
        --endpoint-name sage-baker-sklearn-prod \\
        --role-arn arn:aws:iam::ACCT:role/SageMakerExecutionRole

Then test from any AWS-authed shell:
    aws sagemaker-runtime invoke-endpoint \\
        --endpoint-name sage-baker-sklearn-prod \\
        --content-type application/json \\
        --body '[[5.1,3.5,...]]' /tmp/out.json

Or from Python:
    from sagemaker.predictor import Predictor
    Predictor(endpoint_name="sage-baker-sklearn-prod").predict([[5.1, 3.5, ...]])

To tear down (stops billing): `aws sagemaker delete-endpoint --endpoint-name <name>`
"""
import argparse

import boto3
from sagemaker import ModelPackage, Session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-package-arn", required=True,
                        help="ARN of the registered model package version (from pipeline.py)")
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--role-arn", required=True,
                        help="SageMaker execution role with model-deploy + S3 read perms")
    parser.add_argument("--instance-type", default="ml.t2.medium",
                        help="Smallest viable: ml.t2.medium (~$0.07/hr). Scale up for throughput.")
    parser.add_argument("--instance-count", type=int, default=1)
    args = parser.parse_args()

    session = Session(boto3.Session())

    model = ModelPackage(
        model_package_arn=args.model_package_arn,
        role=args.role_arn,
        sagemaker_session=session,
    )

    print(f"deploying {args.model_package_arn}")
    print(f"  endpoint: {args.endpoint_name}")
    print(f"  instance: {args.instance_count} × {args.instance_type}")

    predictor = model.deploy(
        initial_instance_count=args.instance_count,
        instance_type=args.instance_type,
        endpoint_name=args.endpoint_name,
    )

    print(f"\nendpoint live: {args.endpoint_name}")
    print("invoke locally with:")
    print(f"  Predictor(endpoint_name='{args.endpoint_name}').predict([[...features...]])")
    print(f"\nbilling stops only when you delete it:")
    print(f"  aws sagemaker delete-endpoint --endpoint-name {args.endpoint_name}")


if __name__ == "__main__":
    main()
