"""Every supervised plugin honors the TrainingPlugin protocol.

Catches the kind of regression where someone adds a plugin that
returns a numpy array from prepare() (breaks bundle's
`feature_names = list(X.columns)`) or forgets to set `task`.
"""
import pandas as pd
import pytest

from plugins import _SUPERVISED_REGISTRY


@pytest.mark.parametrize("name", sorted(_SUPERVISED_REGISTRY))
def test_plugin_has_required_attributes(name):
    plugin = _SUPERVISED_REGISTRY[name]()
    assert plugin.name == name, f"plugin.name {plugin.name!r} != registry key {name!r}"
    assert plugin.task in {"classification", "regression"}, \
        f"plugin {name} declares task={plugin.task!r}"


def test_default_plugin_prepare_shape(housing_df):
    """DefaultPlugin: classification on a target-bearing dataframe."""
    from plugins import get_plugin

    df = housing_df.copy()
    df["target"] = (df["target"] > df["target"].median()).astype(int)
    plugin = get_plugin("default")
    X, y = plugin.prepare(df)
    assert isinstance(X, pd.DataFrame), "prepare() must return a DataFrame for X"
    assert isinstance(y, pd.Series), "prepare() must return a Series for y"
    assert "target" not in X.columns, "target column leaked into X"
    assert len(X) == len(y) == len(df)


def test_housing_plugin_end_to_end(housing_df):
    """HousingPlugin: regression, fit + predict on the slice."""
    from plugins import get_plugin

    plugin = get_plugin("housing")
    X, y = plugin.prepare(housing_df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert y.dtype == float

    model = plugin.build_model({})
    assert hasattr(model, "fit") and hasattr(model, "predict")
    model.fit(X, y)

    # Plugins are allowed either evaluate signature; mirror train.py's dispatch.
    import inspect
    if len(inspect.signature(plugin.evaluate).parameters) >= 3:
        metric_name, value = plugin.evaluate(model, X, y)
    else:
        metric_name, value = plugin.evaluate(y, model.predict(X))
    assert metric_name.startswith("validation_")
    assert -1.0 <= value <= 1.0  # R² range, allowing for negative on a tiny fit


def test_registry_lookup_unknown_raises():
    from plugins import get_plugin
    with pytest.raises(ValueError, match="Unknown supervised plugin"):
        get_plugin("does_not_exist")
