"""Inference via MLflow's Model Registry — exercises the full pyfunc path.

Different from local_serve.py: that script reads a bundle from a local
directory. This one:

  1. Talks to the MLflow tracking server (MLFLOW_TRACKING_URI) to look
     up the registered model name/version.
  2. Fetches the artifacts from MLflow's artifact store (./mlartifacts/
     locally; S3 in production) into a temp dir.
  3. Loads the BundleWrapper we registered, which calls model_fn on
     the unpacked bundle to reconstruct the model.

So the *trained weights* never live in MLflow itself — they live in
the artifact store. MLflow holds the metadata pointer and the
versioning workflow.

Prereqs:
  * MLflow server running (mlflow server --backend-store-uri ... etc)
  * MLFLOW_TRACKING_URI set
  * A model registered (run any trainer with MLFLOW_TRACKING_URI set —
    register_bundle_as_pyfunc fires automatically)

Usage:
    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
    python mlflow_serve.py
    python mlflow_serve.py --name sagebaker-lightgbm --version 1
"""
import argparse
import os
import sys

import mlflow
import mlflow.pyfunc
import pandas as pd

# BundleWrapper.load_context will call train.model_fn (or
# train_lightgbm.model_fn / train_torch.model_fn) — those modules need
# to be importable from sys.path.
sys.path.insert(0, "src")

if not os.environ.get("MLFLOW_TRACKING_URI"):
    raise SystemExit("Set MLFLOW_TRACKING_URI first (e.g. http://127.0.0.1:5000)")

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="sagebaker-sklearn",
                    help="registered model name in MLflow")
parser.add_argument("--version", default="latest",
                    help='registered version, or "latest"')
args = parser.parse_args()

uri = f"models:/{args.name}/{args.version}"
print(f"tracking server: {os.environ['MLFLOW_TRACKING_URI']}")
print(f"loading: {uri}")

model = mlflow.pyfunc.load_model(uri)
print(f"loaded: {type(model).__name__}")

# A few real rows as a sanity check. Pyfunc accepts DataFrames natively.
df = pd.read_csv("data/sonar.csv")
X = df.drop(columns=["target"]).head(5)
y = df["target"].head(5).tolist()
preds = model.predict(X)

# pyfunc may return ndarray, list, or DataFrame depending on flavor.
if hasattr(preds, "tolist"):
    preds = preds.tolist()
elif hasattr(preds, "values"):
    preds = preds.values.flatten().tolist()

print(f"\n  actual:    {y}")
print(f"  predicted: {preds}")
print(f"  matches:   {sum(int(a) == int(p) for a, p in zip(y, preds))} / {len(y)}")
