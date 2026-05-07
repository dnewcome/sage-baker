"""Run training in SageMaker Local Mode (DLC) against features pulled from Feast.

Pattern A: Feast retrieval happens *outside* the container. We do a
point-in-time historical join on the host, materialize the joined frame
to a parquet, and pass that parquet to the SKLearn DLC as the training
channel. The trainer inside the container is plain "read parquet, train,
save bundle" — no Feast install needed in the container, no risk of
numpy/pyarrow upgrades shattering the DLC's pre-built wheels.

This is the pattern that translates cleanly to a real SageMaker Pipeline:
a ProcessingStep does Feast retrieval and writes parquet to S3, then a
TrainingStep consumes that parquet. Same shape, just with S3 instead of
local files.

Prereqs:
  * AWS creds set (`aws configure` or sso) so the DLC pull works
  * `feast apply` has been run in feature_repo/
  * data prepared with `python prepare_sonar.py`
"""
import os
import shutil

import pandas as pd
from feast import FeatureStore
from sagemaker.local import LocalSession
from sagemaker.sklearn.estimator import SKLearn

import boto3

if boto3.Session().get_credentials() is None:
    raise SystemExit(
        "Real AWS credentials required for DLC pulls — see local_train_dlc.py."
    )

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

# Snap-confined Docker can't bind-mount /tmp paths.
SCRATCH = os.path.abspath(".sm-scratch")
os.makedirs(SCRATCH, exist_ok=True)
os.environ["TMPDIR"] = SCRATCH

# --- 1. Feast retrieval (host-side) -------------------------------------
FEATURE_REPO = "./feature_repo"
FEATURE_REFS = [f"sonar_bands:f{i}" for i in range(60)]
MATERIALIZED_DIR = os.path.abspath("./materialized")

entity_df = pd.read_parquet(os.path.join(FEATURE_REPO, "data", "sonar_labels.parquet"))
print(f"loaded entity_df: {len(entity_df)} rows")

store = FeatureStore(repo_path=FEATURE_REPO)
joined = store.get_historical_features(
    entity_df=entity_df, features=FEATURE_REFS
).to_df()
print(f"feast join: {len(joined)} rows × {len(joined.columns)} cols")

# Wipe + write the materialized parquet. Single file in a directory keeps
# the SageMaker channel tidy — the trainer's glob picks it up.
if os.path.isdir(MATERIALIZED_DIR):
    shutil.rmtree(MATERIALIZED_DIR)
os.makedirs(MATERIALIZED_DIR)
joined.to_parquet(os.path.join(MATERIALIZED_DIR, "training.parquet"), index=False)
print(f"wrote {MATERIALIZED_DIR}/training.parquet")

# --- 2. DLC training (inside the container) -----------------------------
session = LocalSession()
session.config = {"local": {"local_code": True, "container_root": SCRATCH}}

def _container_env():
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not uri:
        return {}
    uri = uri.replace("127.0.0.1", "host.docker.internal").replace("localhost", "host.docker.internal")
    return {"MLFLOW_TRACKING_URI": uri}

estimator = SKLearn(
    entry_point="train.py",
    source_dir="src",
    role="arn:aws:iam::000000000000:role/SageMakerRole",
    instance_type="local",
    instance_count=1,
    framework_version="1.2-1",
    py_version="py3",
    hyperparameters={"n-estimators": 200, "max-depth": 4},
    environment=_container_env(),
    sagemaker_session=session,
)

estimator.fit({"train": f"file://{MATERIALIZED_DIR}/"})
print("\nmodel artifact:", estimator.model_data)
