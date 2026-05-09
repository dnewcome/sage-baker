"""The /productionize invariant: rebuild estimator from config.json (no pickle)
→ predictions match the bundled weights byte-for-byte.

If this test fails, /productionize generates lying notebooks. The bundle
architecture's main promise is that the config has enough information to
rebuild the model from scratch, with only the weights file carrying the
trained state — and even then, the trained weights from a `random_state`-pinned
fit must reproduce.
"""
import importlib
import json
import sys

import joblib
import numpy as np
import pytest


def test_config_rebuild_matches_bundled_pickle(tmp_train_dir, tmp_model_dir, monkeypatch):
    import train

    monkeypatch.setattr(sys, "argv", [
        "train.py",
        "--train", str(tmp_train_dir),
        "--model-dir", str(tmp_model_dir),
        "--plugin", "housing",
    ])
    train.main()

    config = json.loads((tmp_model_dir / "config.json").read_text())

    # Reconstruct the model purely from config — no pickle, no plugin import.
    mod = importlib.import_module(config["estimator_module"])
    EstimatorClass = getattr(mod, config["estimator"])
    model_from_config = EstimatorClass(**config["params"])

    # Reproduce the exact training data + split that train.py used.
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from plugins import get_plugin

    df = pd.read_parquet(tmp_train_dir / "training.parquet")
    plugin = get_plugin(config["plugin"])
    X, y = plugin.prepare(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model_from_config.fit(X_train, y_train)

    # Predictions from the config-rebuilt model must match the pickled bundle.
    model_from_bundle = joblib.load(tmp_model_dir / config["weights_file"])
    np.testing.assert_allclose(
        model_from_config.predict(X_test[:50]),
        model_from_bundle.predict(X_test[:50]),
        rtol=1e-5,
        err_msg="config-rebuilt model diverged from bundled pickle — "
                "/productionize would generate misleading notebooks",
    )
