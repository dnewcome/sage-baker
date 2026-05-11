"""Flask test-client smoke for the serving harness.

Stubs out the plugin + model so the test is hermetic — no real training
pipeline, no joblib version coupling on a real bundle.

`plugins` and `serve` are imported per-test, not at module level: a
sibling test (test_evaluate_signatures) purges sys.modules['plugins*']
at teardown, which would invalidate module-level refs here.
"""
import importlib
import json
import sys

import joblib
import numpy as np
import pandas as pd
import pytest


class _StubServeModel:
    """Module-level so joblib can pickle/unpickle it across the bundle round-trip."""
    def predict_proba(self, X):
        # Always class-1 with prob 0.8 → above threshold 0.5 → prediction 1.
        return np.tile([[0.2, 0.8]], (len(X), 1))


def _fresh_plugins():
    """Return a fresh `plugins` module (re-imported if a teardown wiped it)."""
    if "plugins" in sys.modules:
        return importlib.reload(sys.modules["plugins"])
    import plugins
    return plugins


def _import_serve_fresh():
    """Drop the cached `serve` module so module-level _load() runs with current env."""
    sys.modules.pop("serve", None)
    import serve  # noqa: F401
    return sys.modules["serve"]


@pytest.fixture
def stub_bundle(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    joblib.dump(_StubServeModel(), bundle_dir / "model.joblib")
    (bundle_dir / "config.json").write_text(json.dumps({"prediction_threshold": 0.5}))
    return bundle_dir


@pytest.fixture
def client(monkeypatch, stub_bundle):
    plugins = _fresh_plugins()
    # Define the stub inside the fixture so it inherits the same TrainingPlugin
    # identity that serve.py will use after re-import.
    TrainingPlugin = plugins.TrainingPlugin

    class _StubServePlugin(TrainingPlugin):
        name = "_serve_stub"
        task = "classification"

        def prepare_inference(self, raw_input):
            return pd.DataFrame(raw_input)

    monkeypatch.setitem(plugins._SUPERVISED_REGISTRY, "_serve_stub", _StubServePlugin)
    monkeypatch.setenv("PLUGIN_NAME", "_serve_stub")
    monkeypatch.setenv("MODEL_DIR", str(stub_bundle))
    serve = _import_serve_fresh()
    return serve.app.test_client()


def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.data == b"ok"


def test_predict_returns_predictions(client):
    r = client.post("/predict", json=[{"x": 1}, {"x": 2}])
    assert r.status_code == 200
    body = r.get_json()
    assert body == {"predictions": [1, 1]}


def test_predict_accepts_single_dict_payload(client):
    """The route auto-wraps a non-list body in a list."""
    r = client.post("/predict", json={"x": 1})
    assert r.status_code == 200
    assert r.get_json() == {"predictions": [1]}


def test_predict_501_when_prepare_inference_missing(tmp_path, monkeypatch):
    """A plugin that doesn't override prepare_inference yields 501, not 500."""
    _fresh_plugins()  # ensure sys.modules has a usable plugins package
    bundle_dir = tmp_path / "no-prepare-bundle"
    bundle_dir.mkdir()
    joblib.dump(_StubServeModel(), bundle_dir / "model.joblib")
    (bundle_dir / "config.json").write_text("{}")

    # "default" is built-in and inherits the NotImplementedError default.
    monkeypatch.setenv("PLUGIN_NAME", "default")
    monkeypatch.setenv("MODEL_DIR", str(bundle_dir))
    serve = _import_serve_fresh()
    client = serve.app.test_client()

    r = client.post("/predict", json=[{"x": 1}])
    assert r.status_code == 501
    assert "prepare_inference" in r.get_json()["error"]
