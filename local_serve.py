"""Deploy the trained model to a local SageMaker endpoint and run a prediction.

Run after local_train.py — it picks up the model artifact from the last run.
"""
import os
import glob
from sagemaker.local import LocalSession
from sagemaker.sklearn.model import SKLearnModel

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "dummy")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "dummy")

session = LocalSession()
session.config = {"local": {"local_code": True}}

# Local Mode writes model.tar.gz under /tmp/tmp*/model.tar.gz; grab the newest.
candidates = sorted(glob.glob("/tmp/tmp*/model.tar.gz"), key=os.path.getmtime)
if not candidates:
    raise SystemExit("No local model artifact found — run local_train.py first.")
model_data = "file://" + candidates[-1]
print("using model:", model_data)

model = SKLearnModel(
    model_data=model_data,
    role="arn:aws:iam::000000000000:role/SageMakerRole",
    entry_point="train.py",
    source_dir=".",
    framework_version="1.2-1",
    py_version="py3",
    sagemaker_session=session,
)

predictor = model.deploy(initial_instance_count=1, instance_type="local")
try:
    sample = [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]
    print("prediction:", predictor.predict(sample))
finally:
    predictor.delete_endpoint()
