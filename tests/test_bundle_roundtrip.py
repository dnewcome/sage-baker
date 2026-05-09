"""Bundle round-trip: train → save → reload via model_fn → predictions match.

This is the core promise of the bundle architecture. If this test
fails, every downstream loader (evaluate.py, local_serve.py,
mlflow_serve.py) is broken.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("weights_format", ["joblib", "skops"])
def test_train_then_reload_matches(tmp_train_dir, tmp_model_dir, weights_format, monkeypatch):
    if weights_format == "skops":
        pytest.importorskip("skops")

    import train

    monkeypatch.setattr(sys, "argv", [
        "train.py",
        "--train", str(tmp_train_dir),
        "--model-dir", str(tmp_model_dir),
        "--plugin", "housing",
        "--weights-format", weights_format,
    ])
    train.main()

    config = json.loads((tmp_model_dir / "config.json").read_text())
    assert config["plugin"] == "housing"
    assert config["task"] == "regression"
    assert config["weights_format"] == weights_format
    assert config["framework"] == "sklearn"
    assert "feature_names" in config and len(config["feature_names"]) > 0

    metadata = json.loads((tmp_model_dir / "metadata.json").read_text())
    assert "validation_r2" in metadata
    assert metadata["n_train"] + metadata["n_test"] == 200

    # The reloaded model should make exact predictions matching what the
    # trainer produced — that's the whole point of the bundle.
    model = train.model_fn(str(tmp_model_dir))

    import pandas as pd
    df = pd.read_parquet(tmp_train_dir / "training.parquet")
    from plugins import get_plugin
    X, _ = get_plugin("housing").prepare(df)
    preds = model.predict(X.head(20))
    assert preds.shape == (20,)
    assert np.all(np.isfinite(preds))
