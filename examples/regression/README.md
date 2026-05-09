# Regression (continuous target)

Continuous-target prediction (housing prices, propensity scores,
forecasted values). Demonstrates: how the plugin contract handles
non-classification tasks, regression-specific evaluation (R² / RMSE /
MAE), and the harness's `task=regression` dispatch in `evaluate.py`
and `local_serve.py`.

## What it is

The same plugin / bundle / agent / serving infrastructure, but with
`task = "regression"` and an R² metric instead of accuracy or AUC.
Uses sklearn-bundled California housing — 20K rows, 8 numeric
features, continuous target (median house value in $100K units). No
download required.

The `HousingPlugin` shows substantial feature engineering on top of
the raw features (per-household ratios, log transforms, geographic
distances to major California cities, KMeans cluster IDs, MedInc
polynomial features). It's the most feature-rich plugin in the repo
— useful as a reference for "how far you can push prepare()."

## Quickstart

```bash
make data-housing             # ./data/california.csv  (sklearn-bundled, no network)
make train-housing            # bundle in ./models/housing/
                              # → validation_r2≈0.85
```

## Files

| Path | What it is |
| --- | --- |
| [`src/plugins/housing.py`](../../src/plugins/housing.py) | The plugin: heavy feature engineering + HistGradientBoostingRegressor |
| [`src/plugins/base.py`](../../src/plugins/base.py) | `TrainingPlugin` base — note `task: str = "classification"` defaults; housing overrides |
| [`agent_regression.md`](../../agent_regression.md) | Agent-loop program for regression problems |
| `data/california.csv` | sklearn-bundled California housing (gitignored) |
| `models/housing/` | Bundle output (gitignored) |

## Try it different ways

### Run the agent loop

```bash
.venv/bin/python agent.py \
  --plugin src/plugins/housing.py \
  --program agent_regression.md
```

The agent program for regression uses the same shape as
classification but with R²-aware strategy hints (no class-imbalance
concerns, regression-specific loss functions, scaling matters more
for linear models).

### Compare against a simple baseline

The current plugin has dozens of engineered features. Try a baseline
with just the raw 8 features — instructive to see how much lift the
feature engineering buys.

### Evaluate beyond R²

[`evaluate.py`](../../evaluate.py) reads `task` from `config.json`
and dispatches accordingly. For housing, the output JSON has R² +
RMSE + MAE; for clickstream, precision/recall/F1. Run
`python evaluate.py --model ./models/housing --test ./data --output
./eval/housing` and inspect `metrics.json`.

## Scale to production

Regression problems show up as:

- **Forecasting** (revenue, demand, price). Time-series shape; the
  plugin contract works but you'd want a rolling-window evaluation
  rather than random split. See the housing plugin's `extra_config`
  for how to record additional fields per regression task.
- **Propensity scores** (P(convert)) — borderline classification vs
  regression. If you want a probability-shaped output, train a
  classifier with `predict_proba`; if you want a continuous expected-
  value output, regression on the value directly.
- **Calibration models** — wrap an existing classifier's
  probabilities into a calibration regressor. Plugin pattern works.

Same local-iterate / cloud-train story as the other examples: commit
the plugin, cloud CI re-runs against a real warehouse extract.
`evaluate.py`'s task-aware metric output makes regression results
comparable across models without per-call configuration.
