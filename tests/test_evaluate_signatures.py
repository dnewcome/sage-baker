"""train.py supports both evaluate() signatures.

The harness uses `inspect.signature` to dispatch:
  - new: evaluate(model, X_test, y_true)   ← can use predict_proba
  - old: evaluate(y_true, y_pred)           ← legacy plugin shape

This test exercises both shapes via temporary plugin classes dropped
into src/plugins/private/ at runtime (and cleaned up afterwards).
"""
import json
import sys
from pathlib import Path

import pytest


PRIVATE_DIR = Path(__file__).resolve().parent.parent / "src" / "plugins" / "private"


_LEGACY_PLUGIN = """
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from plugins.base import TrainingPlugin


class LegacyEvalPlugin(TrainingPlugin):
    name = "test_legacy_eval"
    task = "regression"

    def prepare(self, df):
        return df.drop(columns=["target"]), df["target"].astype(float)

    def evaluate(self, y_true, y_pred):  # 2-arg signature
        return "validation_r2", float(r2_score(y_true, y_pred))

    def build_model(self, params):
        return LinearRegression()
"""


_NEW_SIG_PLUGIN = """
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from plugins.base import TrainingPlugin


class NewSigEvalPlugin(TrainingPlugin):
    name = "test_new_sig_eval"
    task = "regression"

    def prepare(self, df):
        return df.drop(columns=["target"]), df["target"].astype(float)

    def evaluate(self, model, X_test, y_true):  # 3-arg signature
        return "validation_r2", float(r2_score(y_true, model.predict(X_test)))

    def build_model(self, params):
        return LinearRegression()
"""


@pytest.fixture
def temp_private_plugin():
    """Yields a function that drops a .py file into private/, cleans up after."""
    PRIVATE_DIR.mkdir(exist_ok=True)
    (PRIVATE_DIR / "__init__.py").touch(exist_ok=True)
    written = []

    def _drop(filename: str, source: str) -> None:
        f = PRIVATE_DIR / filename
        f.write_text(source)
        written.append(f)
        # Force re-discovery so the new file shows up in the registry.
        for mod in list(sys.modules):
            if mod.startswith("plugins") or mod == "train":
                del sys.modules[mod]

    yield _drop

    for f in written:
        f.unlink(missing_ok=True)
    for mod in list(sys.modules):
        if mod.startswith("plugins") or mod == "train":
            del sys.modules[mod]


@pytest.mark.parametrize("filename, source, plugin_name", [
    ("eval_legacy_test.py", _LEGACY_PLUGIN, "test_legacy_eval"),
    ("eval_newsig_test.py", _NEW_SIG_PLUGIN, "test_new_sig_eval"),
])
def test_evaluate_signature_dispatch(
    tmp_train_dir, tmp_model_dir, monkeypatch, temp_private_plugin,
    filename, source, plugin_name,
):
    """Both 2-arg and 3-arg evaluate() signatures must round-trip through train.py."""
    temp_private_plugin(filename, source)

    import train

    monkeypatch.setattr(sys, "argv", [
        "train.py",
        "--train", str(tmp_train_dir),
        "--model-dir", str(tmp_model_dir),
        "--plugin", plugin_name,
    ])
    train.main()

    metadata = json.loads((tmp_model_dir / "metadata.json").read_text())
    assert "validation_r2" in metadata, \
        f"plugin {plugin_name} did not record validation_r2 — dispatch failed"
