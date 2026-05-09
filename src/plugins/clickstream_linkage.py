"""Clickstream linkage plugin: predict same-user from a pair of events.

Trains on the pair-level dataset emitted by `prepare_linkage.py`, where
each row is a (event_a, event_b) pair with binary target = 1 if both
events came from the same true user, else 0. The features are
symmetric — same-fingerprint, same-ip-bucket, abs time delta, etc. —
so the model never learns artifacts of pair ordering.

Why this is non-trivial in the real world
-----------------------------------------
With perfect features the linkage problem is easy. The interesting
case is the realistic one: fingerprints aren't unique-per-user
(devices are shared; fingerprints drift), IP buckets shift with NAT,
user_ids are mostly null. The model has to combine weak signals.

The current synthetic dataset's `device_fingerprint` is
near-unique-per-user, so AUC will be artificially high (~1.0).
Track the breakdown by `same_fingerprint` to see whether the model
is learning anything beyond fingerprint matching. To make this
problem harder, add fingerprint-collision noise to the simulator.

Bundle config notes
-------------------
The plugin records `feature_importances_` (where available) in the
extra config so a downstream tool / notebook can audit which signals
drove the prediction.
"""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .base import TrainingPlugin


class ClickstreamLinkagePlugin(TrainingPlugin):
    name = "clickstream_linkage"
    task = "classification"

    def prepare(self, df: pd.DataFrame):
        y = df["target"].astype(int)
        X = df.drop(columns=["target"])
        return X, y

    def evaluate(self, model, X_test, y_true):
        proba = model.predict_proba(X_test)[:, 1]
        return "validation_auc", float(roc_auc_score(y_true, proba))

    def build_model(self, params: dict):
        return HistGradientBoostingClassifier(
            max_iter=int(params.get("max_iter", 200)),
            max_depth=int(params.get("max_depth", 4)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            min_samples_leaf=int(params.get("min_samples_leaf", 20)),
            random_state=42,
        )
