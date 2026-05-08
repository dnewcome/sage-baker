"""Housing plugin: regression on California housing dataset.

This is the regression counterpart to DefaultPlugin — same plugin
contract, but `task = "regression"` and the metric is R² (higher is
better, bounded above by 1).

Input: any CSV/parquet with a continuous `target` column. Defaults
match data/california.csv (8 numeric features, target = median house
value in $100K units).
"""
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

from .base import TrainingPlugin

# Same skip list as DefaultPlugin so Feast bookkeeping cols are dropped.
_SKIP = {"target", "signal_id", "event_timestamp"}


class HousingPlugin(TrainingPlugin):
    name = "housing"
    task = "regression"

    def prepare(self, df: pd.DataFrame):
        feature_cols = [c for c in df.columns if c not in _SKIP]
        X = df[feature_cols]
        y = df["target"].astype(float)  # continuous, NOT cast to int
        return X, y

    def evaluate(self, y_true, y_pred):
        # Explicit override (the base class default would also pick R²
        # via task = "regression", but stating it here keeps the plugin
        # self-documenting).
        return "validation_r2", float(r2_score(y_true, y_pred))

    def build_model(self, params: dict):
        return GradientBoostingRegressor(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=int(params.get("max_depth", 3)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            random_state=42,
        )

    def extra_config(self, model, X: pd.DataFrame) -> dict:
        # Regression has no `classes_`; record nothing classification-specific.
        return {}

    def prepare_data(self, output_dir: str, seed: int = 42, extra_args=None):
        """Fetch California housing from sklearn and write data/california.csv.

        Called by the root-level prepare.py dispatcher
        (``python prepare.py --plugin housing``). sklearn-bundled, no
        download required.
        """
        import hashlib
        import json
        import os
        import shutil
        from datetime import datetime, timezone

        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        bunch = fetch_california_housing(as_frame=True)
        df = bunch.frame.rename(columns={"MedHouseVal": "target"})
        out_csv = os.path.join(output_dir, "california.csv")
        df.to_csv(out_csv, index=False)

        data_hash = hashlib.sha256(open(out_csv, "rb").read()).hexdigest()
        lineage = {
            "source": "sklearn.datasets.fetch_california_housing",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "dataset_sha256": data_hash,
            "dataset_n_rows": len(df),
            "feature_names": list(df.columns.drop("target")),
            "target_stats": {
                "min": float(df["target"].min()),
                "max": float(df["target"].max()),
                "mean": float(df["target"].mean()),
            },
        }
        with open(os.path.join(output_dir, "lineage.json"), "w") as f:
            json.dump(lineage, f, indent=2)

        print(f"wrote {out_csv} ({len(df)} rows × {len(df.columns)} cols)")
        print(f"wrote {output_dir}/lineage.json (sha256: {data_hash[:16]}...)")
