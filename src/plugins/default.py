"""Default plugin: generic CSV/parquet with a ``target`` column, RandomForest.

This is the original train.py behaviour expressed as a plugin. It works
with any dataset that has a numeric ``target`` column and no special feature
engineering — iris, sonar, and similar toy datasets all fall here.

For the validation metric: when the target is binary, this plugin uses
ROC-AUC instead of accuracy. AUC gives the agent.py loop a continuous
signal — even tiny model changes nudge probabilities, so the agent can
hill-climb. Accuracy on a 42-row test set quantizes to 43 buckets, which
flattens the optimization surface for hyperparameter tweaks.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

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

    def evaluate(self, model, X_test, y_true) -> tuple:
        # Binary + has predict_proba → ROC-AUC. Continuous, sensitive to
        # small model changes — what the agent loop needs to climb.
        if y_true.nunique() == 2 and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
            return "validation_auc", float(roc_auc_score(y_true, proba))
        # Multiclass / non-probabilistic models: fall back to accuracy.
        return "validation_accuracy", float(accuracy_score(y_true, model.predict(X_test)))

    def build_model(self, params: dict):
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=int(params.get("max_depth", 5)),
            random_state=42,
        )
