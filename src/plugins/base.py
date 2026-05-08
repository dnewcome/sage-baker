"""Base class for metric-specific training plugins.

Adding a new metric
-------------------
1. Create ``src/plugins/<metric>.py`` with a subclass of TrainingPlugin.
2. Register it in ``src/plugins/__init__.py``.

The generic harness (``src/train.py``) handles everything outside the plugin
contract: data loading, train/test split, bundle serialization, MLflow
tracking. The plugin owns feature engineering, the model class, and its
hyperparameter defaults.
"""
import pandas as pd


class TrainingPlugin:
    # Override in every subclass.
    name: str = "base"

    def prepare(self, df: pd.DataFrame) -> tuple:
        """Feature engineering + target extraction.

        Receives the raw DataFrame loaded from the train channel
        (CSV or parquet). Returns ``(X, y)`` where X is a DataFrame
        of model input features and y is a Series of integer labels.

        Column pruning, type casting, and derived feature creation all
        belong here. Anything that must match at inference time should
        be mirrored in the plugin's corresponding inference helper.
        """
        raise NotImplementedError

    def build_model(self, params: dict):
        """Instantiate an unfitted sklearn-compatible estimator.

        ``params`` is a flat dict of strings — the same format SageMaker
        uses for hyperparameters.json. Parse what you need; ignore the
        rest. Provide defaults for anything that might not be present.

        Example::

            def build_model(self, params):
                return LGBMClassifier(
                    n_estimators=int(params.get("n_estimators", 100)),
                    num_leaves=int(params.get("num_leaves", 31)),
                )
        """
        raise NotImplementedError

    def extra_config(self, model, X: pd.DataFrame) -> dict:
        """Extra fields to merge into config.json. Optional.

        Use this to record plugin-specific metadata that the inference
        side needs — e.g. which columns are categorical, the prediction
        threshold, or feature importance rankings.
        """
        return {}

    def prepare_data(self, output_dir: str, seed: int = 42, extra_args: list = None) -> None:
        """Generate synthetic training data into ``output_dir``.

        Override in plugins that can generate their own synthetic data for
        local development and CI. The root-level ``prepare.py`` dispatches
        here so all prepare logic lives alongside the plugin it belongs to.

        Raise ``NotImplementedError`` (the default) if the plugin has no
        synthetic data generator — the caller will show a helpful message.
        """
        raise NotImplementedError(
            f"Plugin '{self.name}' has no prepare_data() implementation. "
            "Provide a real dataset via the --train argument instead."
        )
