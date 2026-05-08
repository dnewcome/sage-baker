# Plan

## Phase 1 — Regression task + plugin (now)

Goal: extend sage-baker beyond binary/multiclass classification so the
plugin abstraction covers the full supervised-learning shape. Forces
out hidden classification-specific assumptions in the harness.

### Dataset

`sklearn.datasets.fetch_california_housing` — 8 numeric features
(median income, house age, etc.), continuous `target` (median house
value), ~20K rows, sklearn-bundled (no download). Public, no auth,
runs on any clone.

### Refactor scope (what's classification-only today)

| File | Classification assumption |
| ---- | ------------------------- |
| `src/train.py` | imports `accuracy_score` directly, prints `validation_accuracy=…`, records `clf.classes_` in config.json |
| `evaluate.py` | precision/recall/f1, casts target to int, dispatches binary vs macro by class count |
| `local_serve.py` | counts predicted-vs-actual "matches" |
| `agent.py` | parses `validation_accuracy=` from stdout |
| `src/plugins/base.py` | docstring says "integer labels" |
| `src/plugins/default.py` | `RandomForestClassifier`, `target.astype(int)` |
| `program.md` | strategy hints all assume classification |

### Approach

**Plugins declare their task and own their metric.** Add to the base:

```python
class TrainingPlugin:
    task: str = "classification"   # or "regression"
    def evaluate(self, y_true, y_pred) -> tuple[str, float]:
        # default = accuracy; regression plugins override
        ...
```

Higher-is-better convention everywhere (R² for regression — bounded
above by 1, comparable across datasets, agent's `>` comparison still
makes sense).

Trainer prints `validation_<metric>=…` (the plugin chose the name)
and the agent's grep widens to match any `validation_\w+=` line.

### Files to add

- `prepare_housing.py` — writes `data/california.csv` + `lineage.json`
- `src/plugins/housing.py` — `HousingPlugin(task="regression")` using
  `GradientBoostingRegressor`. Overrides `evaluate` to return R².
- `program_regression.md` — agent constraints + strategy hints for
  the regression case (different model classes, no class imbalance
  concern, RMSE/R² intuition)

### Files to modify

- `src/plugins/base.py` — add `task` attr + `evaluate()` default
- `src/plugins/default.py` — explicit `task = "classification"`,
  override `evaluate()` to record the explicit accuracy default
- `src/train.py` — call `plugin.evaluate(y_test, clf.predict(X_test))`
  instead of hardcoded `accuracy_score`. Skip `clf.classes_` for
  regression. Use the metric name in the log/metadata.
- `evaluate.py` — read `task` from config.json, dispatch metrics
  accordingly (classification: precision/recall/f1; regression:
  R²/RMSE/MAE)
- `local_serve.py` — dispatch on task (regression: print residuals
  not "matches")
- `agent.py` — broaden the metric regex
- `Makefile` — `data-housing`, `train-housing` targets

### Smoke test

```bash
make data-housing
make train MODEL_DIR=./model_housing --plugin housing
make serve MODEL_DIR=./model_housing
.venv/bin/python evaluate.py --model ./model_housing --test ./data --output ./eval_housing
cat ./eval_housing/metrics.json
```

Expectation: bundle has `task: "regression"` and `validation_r2` in
metadata; `local_serve` shows residuals; `evaluate.py` produces R² +
RMSE + MAE.

---

## Phase 2 — Public-data recommender (later)

Goal: make `make train-als` work without synthetic data, on a real
public dataset, so the recommender path is exercisable on a clone.

### Dataset

**MovieLens-100K** (100K ratings, 943 users × 1682 movies, 1.7 MB).
Public, no auth, canonical recommender-system benchmark. The `ml-100k`
zip is downloadable from grouplens.org.

### Approach

Add `prepare_movielens.py` that downloads + extracts + writes
parquets matching the existing `als` plugin's expected schema (user,
item, rating, timestamp). The plugin doesn't change.

Add a `train-movielens` make target, and document the recommender
demo as a first-class path.

### Files to add

- `prepare_movielens.py`
- Update README to reference it

---

## Out of scope for this round

- **Smoke test suite + CI** — locks in current behaviour; would be
  the natural next step after Phase 2. Phase 1 is more valuable
  because it forces robustness; tests come after the API is stable.
- **Hyperparameter tuning beyond agent.py** — Optuna / sklearn
  HalvingGridSearch could plug into the same harness later.
- **More public datasets** (adult-income for categoricals,
  fashion-mnist for torch) — same pattern as housing once the
  abstraction is in place.
- **Drift detection** (Evidently/TFDV) — orthogonal; do once we have
  enough deployed-model usage to make it relevant.
