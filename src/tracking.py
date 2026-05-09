"""MLflow tracking — opt-in, no-op when MLFLOW_TRACKING_URI is unset.

The trainer code calls these functions unconditionally; this module decides
whether to actually talk to an MLflow server. That keeps the trainer free
of `if mlflow_enabled:` branches and makes it safe to run offline.

Usage:
    with tracking.mlflow_run(run_name="...", params={...}) as run:
        ... train ...
        tracking.log_metrics({"validation_accuracy": acc})
        tracking.log_bundle(model_dir)

Tracking URIs:
    file:./mlruns         local filesystem (default if you ever set this)
    http://localhost:5000 local mlflow server (run via `mlflow server`)
    https://...           any remote mlflow server (e.g. company's)

When training inside a docker container (BYOC / DLC), pass the env var
through and use `http://host.docker.internal:5000` instead of localhost.
"""
import contextlib
import os


def _enabled():
    return bool(os.environ.get("MLFLOW_TRACKING_URI"))


@contextlib.contextmanager
def mlflow_run(run_name=None, params=None, tags=None):
    """Context manager wrapping mlflow.start_run; yields None when disabled."""
    if not _enabled():
        yield None
        return
    import mlflow
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        if params:
            mlflow.log_params(params)
        yield run


def log_metrics(metrics, step=None):
    if not _enabled():
        return
    import mlflow
    mlflow.log_metrics(metrics, step=step)


def log_bundle(model_dir, artifact_path="model"):
    """Log the entire bundle dir (config.json + weights + metadata.json)
    as opaque artifacts. Loading happens via your own load(dir), not via
    mlflow.X.load_model — so MLflow never pickles your class."""
    if not _enabled():
        return
    import mlflow
    mlflow.log_artifacts(model_dir, artifact_path=artifact_path)


def register_bundle_as_pyfunc(model_dir, model_fn, registered_name=None,
                              artifact_path="pyfunc_model", pip_requirements=None):
    """Wrap our model_fn in a custom mlflow.pyfunc.PythonModel and log it.

    What this gets you that log_bundle alone doesn't: the model shows up in
    MLflow's "Models" tab (and the Model Registry if `registered_name` is
    set) — which other systems, including SageMaker's MLflow integration,
    look at to deploy. The wrapper calls our model_fn at load time, so
    MLflow never pickles the user's class.

    `model_fn` is the same function trainers expose for SageMaker
    inference; we just bridge it to MLflow's pyfunc interface.
    """
    if not _enabled():
        return None
    import mlflow
    import mlflow.pyfunc

    # The class lives here, but only its *string identifier* is what gets
    # serialized — the registry knows it as "BundleWrapper". The user's
    # model class is reconstructed via their model_fn at load time.
    class BundleWrapper(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            import json
            import os
            bundle_dir = context.artifacts["bundle"]
            self._model = model_fn(bundle_dir)
            with open(os.path.join(bundle_dir, "config.json")) as f:
                config = json.load(f)
            # Threshold lives in config.json so it travels with the
            # bundle through every serving path (MLflow, SageMaker,
            # local_serve), not just MLflow's wrapper. Default 0.5 keeps
            # back-compat with bundles that don't set it. Tune for
            # imbalanced classes (e.g. 0.15 for a 6%-positive problem).
            self._threshold = float(config.get("prediction_threshold", 0.5))
            self._task = config.get("task", "classification")

        def predict(self, context, model_input, params=None):
            # Regression / non-probabilistic models: pass through.
            if self._task != "classification" or not hasattr(self._model, "predict_proba"):
                return self._model.predict(model_input)
            proba = self._model.predict_proba(model_input)
            # Multiclass: predict_proba threshold doesn't apply; argmax via predict.
            if proba.shape[1] != 2:
                return self._model.predict(model_input)
            # Binary: apply the configured threshold for class output.
            return (proba[:, 1] >= self._threshold).astype(int)

    return mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=BundleWrapper(),
        artifacts={"bundle": model_dir},
        registered_model_name=registered_name,
        pip_requirements=pip_requirements,
    )
