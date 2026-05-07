"""Test inference against a trained model bundle.

Exercises the contract every inference path uses — `model_fn(model_dir)`
returns a model object, you call .predict on it. SageMaker's inference
container does exactly this internally; so does our MLflow PyFunc
wrapper. Testing model_fn here = testing all of them.

By default, looks at ./model_sklearn/ (the host-side trainer's output).
Pass `--artifact path/to/model.tar.gz` to test a SageMaker artifact
instead — but watch out for sklearn version skew between the trainer's
container and your host venv (this is exactly the pickle-coupling
problem the bundle architecture warns about; the framework_version
field in config.json is the signal).
"""
import argparse
import glob
import os
import sys
import tarfile
import tempfile

import pandas as pd
import sklearn

sys.path.insert(0, "src")
import bundle  # type: ignore  # noqa: E402
from train import model_fn  # type: ignore  # noqa: E402


def latest_dlc_artifact():
    """Find the most recent SageMaker model.tar.gz under .sm-scratch/."""
    SCRATCH = os.path.abspath(".sm-scratch")
    candidates = sorted(
        glob.glob(os.path.join(SCRATCH, "tmp*/compressed_artifacts/model.tar.gz")),
        key=os.path.getmtime,
    )
    return candidates[-1] if candidates else None


def warn_version_skew(model_dir):
    cfg = bundle.load_config(model_dir)
    trained = cfg.get("framework_version", "?")
    current = sklearn.__version__ if cfg.get("framework") == "sklearn" else "?"
    if trained != current:
        print(f"⚠ framework_version skew: trained with {trained}, "
              f"loading with {current} — pickle may fail")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="./model_sklearn",
                        help="path to a bundle directory (default: ./model_sklearn)")
    parser.add_argument("--artifact", help="path to model.tar.gz; if set, extracted to a temp dir")
    args = parser.parse_args()

    if args.artifact:
        tmp = tempfile.mkdtemp(prefix="bundle_")
        with tarfile.open(args.artifact) as tf:
            tf.extractall(tmp)
        model_dir = tmp
        print(f"extracted {args.artifact} → {model_dir}")
    else:
        if not os.path.isdir(args.model_dir):
            sys.exit(f"no bundle at {args.model_dir} — run "
                     f"`python src/train.py --train ./data --model-dir ./model_sklearn` first")
        model_dir = args.model_dir

    print(f"bundle contents: {sorted(os.listdir(model_dir))}")
    warn_version_skew(model_dir)

    model = model_fn(model_dir)
    print(f"loaded: {type(model).__name__}")

    df = pd.read_csv("data/sonar.csv")
    X = df.drop(columns=["target"]).head(5).values.tolist()
    y = df["target"].head(5).tolist()
    predictions = model.predict(X).tolist()

    print(f"\n  actual:    {y}")
    print(f"  predicted: {predictions}")
    print(f"  matches:   {sum(a == p for a, p in zip(y, predictions))} / {len(y)}")


if __name__ == "__main__":
    main()
