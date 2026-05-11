"""External-plugin discovery — covers _discover_plugins_in_dir + PLUGIN_DIR.

`plugins` is imported per-test, not at module level, because
test_evaluate_signatures purges sys.modules['plugins*'] at teardown.
A module-level import here would go stale after that point.
"""
import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture
def plugins_mod():
    """Return a fresh `plugins` module + restore registries on teardown."""
    if "plugins" in sys.modules:
        mod = importlib.reload(sys.modules["plugins"])
    else:
        import plugins as mod  # noqa: F401
    saved_sup = dict(mod._SUPERVISED_REGISTRY)
    saved_rec = dict(mod._RECOMMENDER_REGISTRY)
    yield mod
    mod._SUPERVISED_REGISTRY.clear()
    mod._SUPERVISED_REGISTRY.update(saved_sup)
    mod._RECOMMENDER_REGISTRY.clear()
    mod._RECOMMENDER_REGISTRY.update(saved_rec)


EXTERNAL_PLUGIN_SRC = """
from plugins.base import TrainingPlugin

class ExternalDemo(TrainingPlugin):
    name = "_external_demo"
    task = "classification"
"""


def test_discover_picks_up_a_dropped_in_plugin(tmp_path, plugins_mod):
    (tmp_path / "demo.py").write_text(EXTERNAL_PLUGIN_SRC)

    plugins_mod._discover_plugins_in_dir(str(tmp_path), "plugins.external")

    inst = plugins_mod.get_plugin("_external_demo")
    assert inst.name == "_external_demo"
    assert inst.task == "classification"


def test_underscore_and_non_py_files_are_skipped(tmp_path, plugins_mod):
    (tmp_path / "_private.py").write_text(EXTERNAL_PLUGIN_SRC)
    (tmp_path / "README.md").write_text("not a plugin")

    plugins_mod._discover_plugins_in_dir(str(tmp_path), "plugins.external")

    assert "_external_demo" not in plugins_mod._SUPERVISED_REGISTRY


def test_missing_dir_is_a_noop(tmp_path, plugins_mod):
    """Pointing at a non-existent dir is fine — used to be a startup crash risk."""
    plugins_mod._discover_plugins_in_dir(str(tmp_path / "does-not-exist"), "plugins.external")
    # Nothing to assert — just shouldn't raise.


def test_plugin_dir_env_var_triggers_discovery(tmp_path, monkeypatch, plugins_mod):
    """PLUGIN_DIR is the user-facing env contract for the docker harness."""
    (tmp_path / "envdemo.py").write_text(
        EXTERNAL_PLUGIN_SRC.replace("_external_demo", "_env_demo")
                          .replace("ExternalDemo", "EnvDemo")
    )
    monkeypatch.setenv("PLUGIN_DIR", str(tmp_path))

    plugins_mod._discover_private_plugins()

    assert "_env_demo" in plugins_mod._SUPERVISED_REGISTRY
