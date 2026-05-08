"""Default plugin: generic CSV/parquet with a ``target`` column, RandomForest.

This is the original train.py behaviour expressed as a plugin. It works
with any dataset that has a numeric ``target`` column and no special feature
engineering — iris, sonar, and similar toy datasets all fall here.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .base import TrainingPlugin

# Columns that are bookkeeping (Feast metadata, split keys) rather than features.
_SKIP = {"target", "signal_id", "event_timestamp"}


class DefaultPlugin(TrainingPlugin):
    name = "default"
    task = "classification"

    def prepare(self, df: pd.DataFrame):
        feature_cols = [c for c in df.columns if c not in _SKIP]
        X = df[feature_cols]
        y = df["target"].astype(int)
        return X, y

    def build_model(self, params: dict):
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=int(params.get("max_depth", 5)),
            random_state=42,
        )
