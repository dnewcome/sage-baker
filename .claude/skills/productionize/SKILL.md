---
name: productionize
description: Generate a starter Jupyter notebook from the current best agent run — loads the same data via lineage, rebuilds the model from config.json (not the pickle), lays out evaluation/EDA/production-checklist cells. Use this after `agent.py` converges on a winning plugin and the researcher is ready to begin productionizing.
---

# Productionize: agent run → starter notebook

This skill produces `notebooks/<plugin>_productionize.ipynb`, a hand-editable
starting point for taking an agent-loop result toward production. The key
property: the notebook reconstructs the estimator from `config.json` (class
+ params + feature names), not by unpickling the bundle. That proves the
bundle's main promise — that weights and code are decoupled — and gives the
researcher a clean code-only path to lift into work code.

## When to invoke

After `agent.py` (or `make train`) has produced a `model_<plugin>/` bundle
the researcher likes. Typical flow: agent converges → researcher wants to
poke at the result, plot residuals, sanity-check it against fresh data,
think about deployment. This skill creates that scratchpad.

## Arguments

- No arg → use the most recently modified `model_*/` directory.
- `<plugin>` → use `model_<plugin>/` (e.g. `productionize housing`).

## Steps

1. **Identify the target bundle.** If an arg was passed, resolve to
   `model_<arg>/`. Otherwise list `model_*/` and pick the one with the
   newest mtime. Fail gracefully if no bundle exists ("run `make train`
   first") or if `config.json` is missing.

2. **Read the inputs:**
   - `model_<plugin>/config.json` — `task`, `framework`,
     `framework_version`, `estimator`, `estimator_module`, `params`,
     `feature_names`, `weights_file`, `metric_name`, optionally `classes`.
   - `model_<plugin>/metadata.json` — metric value, `n_train`/`n_test`,
     `dataset_file`, optionally `data_lineage`.
   - `src/plugins/<plugin>.py` (or `src/plugins/private/<plugin>.py`) —
     so the notebook can reference `prepare()` / `build_model()` directly.

3. **Generate the notebook** at `notebooks/<plugin>_productionize.ipynb`
   using `nbformat`. Use the cell structure below. Branch on `task`
   (classification vs regression) and `framework` (sklearn vs torch vs
   lightgbm) and on whether `data_lineage.source == "bigquery"`.

4. **Tell the user what was written.** Path + a one-line summary
   ("framework, task, metric=value, N rows"). Don't auto-launch Jupyter.

## Notebook cell structure

Cells in order. Markdown cells give the researcher narrative they can
edit; code cells should be runnable end-to-end without changes.

### Header (markdown)
```
# Productionize <plugin>
Generated <YYYY-MM-DD> from `model_<plugin>/`.
- task: <classification|regression>
- estimator: <ClassName> from <module>
- metric: <metric_name>=<value> on <n_test> test rows
- bundle saved: <metadata.saved_at>
- data: <dataset_file>, <n_train + n_test> rows total

This notebook is a *starting point* — edit freely.
```

### Setup (code)
```python
import os, sys, json
if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('..')
sys.path.insert(0, 'src')
%load_ext autoreload
%autoreload 2
```

### Inspect the bundle (code)
```python
config = json.loads(open('model_<plugin>/config.json').read())
metadata = json.loads(open('model_<plugin>/metadata.json').read())
print(json.dumps(config, indent=2)[:500])
print('---')
print(json.dumps(metadata, indent=2)[:500])
```

### Reload the data (code)

**If `data_lineage.source == "bigquery"`:** emit a `%%bigquery` cell using
the `query` from lineage. Add a follow-up cell that loads the same
parquet from `data/` so reruns are reproducible offline.

**Otherwise:** load `data/<dataset_file>` (parquet or csv based on
extension).

In both cases follow with a `df.shape` / `df.describe().T.head()` cell.

### Reconstruct the model from config (code)

This is the production-ready cell — no pickle, no plugin import:

```python
import importlib
mod = importlib.import_module(config['estimator_module'])
EstimatorClass = getattr(mod, config['estimator'])
model_from_config = EstimatorClass(**config['params'])
```

### Reproduce the train/test split (code)

Use the plugin's `prepare()` and the same `random_state=42` /
`test_size=0.2` that `src/train.py` uses, so metrics are comparable:

```python
from plugins import get_plugin
from sklearn.model_selection import train_test_split

plugin = get_plugin(config['plugin'])
X, y = plugin.prepare(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model_from_config.fit(X_train, y_train)
```

### Sanity-check vs the bundled weights (code)

Load the bundle's pickle and confirm predictions match. This is what
proves the config-only rebuild is faithful.

```python
import joblib
model_from_bundle = joblib.load('model_<plugin>/model.joblib')

# For classification, compare probabilities; for regression, predictions.
if config['task'] == 'classification' and hasattr(model_from_bundle, 'predict_proba'):
    import numpy as np
    np.testing.assert_allclose(
        model_from_config.predict_proba(X_test[:50]),
        model_from_bundle.predict_proba(X_test[:50]),
        rtol=1e-5,
    )
else:
    import numpy as np
    np.testing.assert_allclose(
        model_from_config.predict(X_test[:50]),
        model_from_bundle.predict(X_test[:50]),
        rtol=1e-5,
    )
print('OK — config-rebuilt model matches bundled weights')
```

(If `weights_format` is `skops`, swap `joblib.load` for `skops.io.load`.)

### Re-evaluate (code)

Use the plugin's evaluate signature. The harness handles both shapes:

```python
import inspect
n_params = len(inspect.signature(plugin.evaluate).parameters)
if n_params >= 3:
    name, value = plugin.evaluate(model_from_config, X_test, y_test)
else:
    name, value = plugin.evaluate(y_test, model_from_config.predict(X_test))
print(f'{name}={value:.4f}  (bundle reported {config["metric_name"]}={metadata[config["metric_name"]]:.4f})')
```

### EDA — task-aware

**Classification:** target distribution bar chart; confusion matrix on
the test set (`sklearn.metrics.ConfusionMatrixDisplay.from_estimator`);
ROC curve if binary (`RocCurveDisplay.from_estimator`).

**Regression:** target histogram; predicted-vs-actual scatter with y=x
reference line; residual plot (`y_pred - y_true` vs `y_pred`).

**Feature importance** (any task, if `hasattr(model, 'feature_importances_')`):
top-10 horizontal bar chart with `config['feature_names']` as labels.

### Production checklist (markdown)

```
## Production checklist

- [ ] Pin dependencies — this bundle was trained with
  <framework>=<framework_version>, python <metadata.python>.
  Cross-version loads of joblib pickles are brittle; consider skops
  or rebuilding from config.
- [ ] Lock the data lineage — confirm `metadata.data_lineage.dataset_sha256`
  is reproducible. <if BQ: parameterize the query for fresh data.>
- [ ] Decide retraining cadence (event-driven? scheduled? drift-monitored?).
- [ ] Wire metric monitoring — alert if <metric_name> drops below
  <value - 0.05> on production traffic.
- [ ] If deploying to SageMaker: see `pipeline.py` (training pipeline)
  and `deploy_endpoint.py` (endpoint deploy). For MLflow Registry, see
  `mlflow_serve.py`.
- [ ] Sanity-check inference container — same sklearn version host vs
  container, otherwise expect silent prediction skew.
- [ ] Add a smoke test against `local_serve.py` before deploy.
```

## Implementation tips

- Build the notebook with `nbformat` (`pip` ships it with jupyter):
  ```python
  import nbformat as nbf
  nb = nbf.v4.new_notebook()
  nb.cells = [
      nbf.v4.new_markdown_cell("..."),
      nbf.v4.new_code_cell("..."),
  ]
  with open(out_path, 'w') as f:
      nbf.write(nb, f)
  ```
  Don't hand-write the .ipynb JSON — easy to break.

- The skill is allowed to run a small Python snippet via Bash to
  inspect the bundle before generating the notebook. This is useful
  for filling in things like the metric value in markdown cells.

- Don't add cells that depend on packages outside `requirements.txt`
  / `requirements-*.txt`. If the plugin uses LightGBM, surface that
  via the existing `requirements-lightgbm.txt`, don't introduce new
  deps.

- Keep the notebook short — ~15 cells. The researcher will add their
  own; don't pre-fill speculative analysis.

## Failure modes

- **No `model_*/` bundle:** ask the user to run `make train` (or
  `python agent.py ...`) first.
- **Plugin source missing** (e.g. someone deleted
  `src/plugins/private/X.py` after the bundle was built): generate
  the notebook anyway, but mark the `prepare()` cell as TODO and
  point at `config['feature_names']` as the schema source of truth.
- **`data_lineage` absent:** generate a placeholder data-loading
  cell with a TODO and the `dataset_file` value.
