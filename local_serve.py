"""Test inference against a trained model bundle.

Exercises the contract every inference path uses — `model_fn(model_dir)`
returns a model object, you call .predict on it. SageMaker's inference
container does exactly this internally; so does our MLflow PyFunc
wrapper. Testing model_fn here = testing all of them.

By default, looks at ./models/sklearn/ (the host-side trainer's output).
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

import importlib

import pandas as pd
import sklearn

sys.path.insert(0, "src")
import bundle  # type: ignore  # noqa: E402

# Each framework's trainer exposes its own model_fn. Dispatch by the
# `framework` field in config.json so this script handles any bundle
# layout regardless of which trainer produced it.
LOADER_MODULE = {
    "sklearn": "train",
    "torch": "train_torch",
    "lightgbm": "train_lightgbm",
}


def latest_dlc_artifact():
    """Find the most recent SageMaker model.tar.gz under .sm-scratch/."""
    SCRATCH = os.path.abspath(".sm-scratch")
    candidates = sorted(
        glob.glob(os.path.join(SCRATCH, "tmp*/compressed_artifacts/model.tar.gz")),
        key=os.path.getmtime,
    )
    return candidates[-1] if candidates else None


def warn_version_skew(model_dir):
    """Compare the bundle's framework_version against the *importing*
    framework's actual version. Only sklearn really has the load-failure
    risk (joblib + pickle); torch/lightgbm have stable file formats."""
    cfg = bundle.load_config(model_dir)
    trained = cfg.get("framework_version")
    framework = cfg.get("framework")
    if framework == "sklearn":
        current = sklearn.__version__
    else:
        return  # only sklearn pickles, others have stable file formats
    if trained and trained != current:
        print(f"⚠ sklearn version skew: trained with {trained}, "
              f"loading with {current} — pickle load may fail")


def fetch_features_via_feast(cfg, signal_ids):
    """Look up online features from Feast by entity ID.

    This is the realistic serving shape for a feature-store-backed model:
    callers pass an entity ID, the feature store provides the values.
    The model never sees raw user input — it sees what Feast returns,
    which is the same view the trainer used historically.

    Critically, Feast can return None for missing values (TTL expired,
    feature not yet materialized, etc.). The model has to handle that
    gracefully — that's a *training-time* concern, not a serving-time
    one. The right fix is to train with realistic missingness in the
    historical join, not to paper over it here.
    """
    from feast import FeatureStore  # opt-in import
    store = FeatureStore(repo_path=cfg["feature_repo"])
    response = store.get_online_features(
        features=cfg["feature_refs"],
        entity_rows=[{"signal_id": sid} for sid in signal_ids],
    ).to_dict()
    feature_cols = [ref.split(":")[1] for ref in cfg["feature_refs"]]
    return pd.DataFrame({c: response[c] for c in feature_cols})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="./models/sklearn",
                        help="path to a bundle directory (default: ./models/sklearn)")
    parser.add_argument("--artifact", help="path to model.tar.gz; if set, extracted to a temp dir")
    parser.add_argument("--signal-ids", default="0,50,100,200",
                        help="comma-separated signal_ids for Feast-backed bundles")
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
                     f"`python src/train.py --train ./data --model-dir ./models/sklearn` first")
        model_dir = args.model_dir

    print(f"bundle contents: {sorted(os.listdir(model_dir))}")
    warn_version_skew(model_dir)

    cfg = bundle.load_config(model_dir)
    fw = cfg.get("framework", "sklearn")
    mod_name = LOADER_MODULE.get(fw)
    if not mod_name:
        sys.exit(f"unknown framework {fw!r}; expected one of {sorted(LOADER_MODULE)}")
    print(f"framework: {fw} -> using {mod_name}.model_fn")
    model_fn = importlib.import_module(mod_name).model_fn
    model = model_fn(model_dir)
    print(f"loaded: {type(model).__name__}")

    task = cfg.get("task", "classification")
    print(f"task: {task}")

    # Dispatch on whether the bundle is Feast-backed. Feast bundles
    # record `feature_refs` in config.json — that's the trigger to
    # fetch features by entity ID instead of expecting raw values.
    if cfg.get("feature_refs"):
        signal_ids = [int(s) for s in args.signal_ids.split(",")]
        print(f"feast-backed bundle — looking up features for signal_ids={signal_ids}")
        X = fetch_features_via_feast(cfg, signal_ids)
        print(f"got {X.shape[0]} rows × {X.shape[1]} cols from feast online store")
        labels = pd.read_parquet("feature_repo/data/sonar_labels.parquet")
        y = labels.set_index("signal_id").loc[signal_ids, "target"].tolist()
        predictions = model.predict(X).tolist()
    else:
        # find a sample CSV in data/ — works for sonar, california, iris alike
        import glob
        sample_csv = next(iter(sorted(glob.glob("data/*.csv"))), None)
        if not sample_csv:
            sys.exit("no data/*.csv to score against — run a `make data-*` target first")
        print(f"direct-features bundle — using first 5 rows from {sample_csv}")
        df = pd.read_csv(sample_csv)
        X = df.drop(columns=["target"]).head(5)
        y = df["target"].head(5).tolist()
        predictions = model.predict(X.values.tolist()).tolist()

    if task == "regression":
        print(f"\n  actual:     {[round(v, 4) for v in y]}")
        print(f"  predicted:  {[round(v, 4) for v in predictions]}")
        residuals = [round(p - a, 4) for a, p in zip(y, predictions)]
        print(f"  residuals:  {residuals}")
        mae = sum(abs(r) for r in residuals) / len(residuals)
        print(f"  mean |residual|: {mae:.4f}")
    else:
        print(f"\n  actual:    {y}")
        print(f"  predicted: {predictions}")
        print(f"  matches:   {sum(int(a) == int(p) for a, p in zip(y, predictions))} / {len(y)}")


if __name__ == "__main__":
    main()
