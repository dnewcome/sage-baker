"""Training script SageMaker runs inside the container.

SageMaker conventions:
  /opt/ml/input/data/<channel>/  -- training data
  /opt/ml/input/config/hyperparameters.json  -- hyperparameters
  /opt/ml/model/                 -- saved model artifacts (tarred to model.tar.gz)
"""
import argparse
import json
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

HP_PATH = "/opt/ml/input/config/hyperparameters.json"


def load_hyperparameters():
    """SageMaker writes hyperparameters to a JSON file; values arrive as strings."""
    if not os.path.exists(HP_PATH):
        return {}
    with open(HP_PATH) as f:
        return json.load(f)


def main():
    hp = load_hyperparameters()
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=int(hp.get("n-estimators", 100)))
    parser.add_argument("--max-depth", type=int, default=int(hp.get("max-depth", 5)))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    args, _ = parser.parse_known_args()

    df = pd.read_csv(os.path.join(args.train, "iris.csv"))
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"validation_accuracy={acc:.4f}")

    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    """Used by the SKLearn inference container to load the model."""
    return joblib.load(os.path.join(model_dir, "model.joblib"))


if __name__ == "__main__":
    main()
