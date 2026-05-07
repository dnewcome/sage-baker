"""Trainer that pulls features via Feast's point-in-time historical join.

Run after `feast apply` (see README). Same model bundle output as
src/train.py — the only difference is *how* the features arrive: instead
of `pd.read_csv(...)`, we ask Feast for a historical-features dataframe
keyed by entity + timestamp. The training/serving skew problem goes
away because the same FeatureView is used at inference time too.

Usage:
    python src/train_feast.py --feature-repo ./feature_repo \\
                              --model-dir ./model_feast
"""
import argparse
import json
import os
import sys

import joblib
import pandas as pd
import sklearn
from feast import FeatureStore
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import bundle
import tracking

WEIGHTS_FILE = "model.joblib"
FEATURE_REFS = [f"sonar_bands:f{i}" for i in range(60)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-repo", default="./feature_repo")
    parser.add_argument("--labels-file", default="data/sonar_labels.parquet",
                        help="path relative to --feature-repo")
    parser.add_argument("--model-dir", default="./model_feast")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    # The entity_df carries the entity keys, event timestamps, and labels
    # — Feast uses (signal_id, event_timestamp) to do point-in-time joins
    # with the FeatureView so no future data leaks into past examples.
    labels_path = os.path.join(args.feature_repo, args.labels_file)
    entity_df = pd.read_parquet(labels_path)
    print(f"loaded {labels_path}: {len(entity_df)} rows")

    store = FeatureStore(repo_path=args.feature_repo)
    print(f"feast store: {store.project} (registry={store.config.registry.path})")

    # Historical retrieval: features as of each entity_df row's
    # event_timestamp. This is the bit that fixes training/serving skew —
    # the same feature definitions are used here and at inference time.
    feature_df = store.get_historical_features(
        entity_df=entity_df, features=FEATURE_REFS
    ).to_df()
    print(f"retrieved {len(feature_df)} rows × {len(feature_df.columns)} cols from feast")

    feature_cols = [f"f{i}" for i in range(60)]
    X = feature_df[feature_cols]
    y = feature_df["target"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    run_params = {
        "n-estimators": args.n_estimators,
        "max-depth": args.max_depth,
        "feature_refs": ",".join(FEATURE_REFS),
        "feast_project": store.project,
    }
    with tracking.mlflow_run(run_name="sklearn-rf-feast", params=run_params,
                             tags={"framework": "sklearn", "feature_store": "feast"}):
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42
        )
        clf.fit(X_train, y_train)

        acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"validation_accuracy={acc:.4f}")
        tracking.log_metrics({"validation_accuracy": acc})

        # config.json now records the feature view contract — model_fn
        # uses the same refs to pull online features at inference time.
        bundle.save_config(args.model_dir, {
            "framework": "sklearn",
            "framework_version": sklearn.__version__,
            "estimator": type(clf).__name__,
            "estimator_module": type(clf).__module__,
            "params": clf.get_params(),
            "weights_file": WEIGHTS_FILE,
            "feature_refs": FEATURE_REFS,
            "feature_repo": os.path.abspath(args.feature_repo),
            "classes": [int(c) for c in clf.classes_.tolist()],
        })

        joblib.dump(clf, os.path.join(args.model_dir, WEIGHTS_FILE))

        bundle.save_metadata(args.model_dir, extras={
            "validation_accuracy": acc,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "feast_project": store.project,
        })

        tracking.log_bundle(args.model_dir)


def model_fn(model_dir):
    """Load the model. The caller is responsible for fetching features
    via the feature_refs recorded in config.json — keeps this loader
    framework-/framework-store-agnostic."""
    config = bundle.load_config(model_dir)
    return joblib.load(os.path.join(model_dir, config["weights_file"]))


def predict_one(model_dir, signal_id):
    """End-to-end serving demo: read config, look up online features for
    a single entity, run the model. This is the path SageMaker / your
    inference container would take."""
    config = bundle.load_config(model_dir)
    store = FeatureStore(repo_path=config["feature_repo"])
    features = store.get_online_features(
        features=config["feature_refs"],
        entity_rows=[{"signal_id": signal_id}],
    ).to_dict()
    feature_cols = [f"f{i}" for i in range(60)]
    X = pd.DataFrame([{c: features[c][0] for c in feature_cols}])
    clf = model_fn(model_dir)
    return int(clf.predict(X)[0])


if __name__ == "__main__":
    main()
