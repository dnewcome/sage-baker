# Agent program for {{PROJECT_NAME}}

Copy this file to `program_<project>.md` (or to `private/program_<project>.md`
if it references work-internal column names) and fill in the placeholders.

You are an autonomous ML research agent. Your job: iteratively improve
the **{{METRIC_NAME}}** of a {{TASK_TYPE}} model trained on
**{{DATASET_DESCRIPTION}}** by editing **one Python file**.

> Replace `{{PLACEHOLDERS}}` below with values that fit your dataset,
> then delete this note.

## Dataset shape

- **Source**: {{e.g. BigQuery `myproj.analytics.user_events`, sampled
  via `prepare_<project>.py` to ./data_<project>/training.parquet}}
- **Rows**: ~{{N}} after the LIMIT/sampling
- **Target column**: `{{TARGET_COL}}` ({{int 0/1 / continuous / multiclass}})
- **Features**:
  - Numeric: {{list a few — e.g. `num_sessions`, `days_since_signup`}}
  - Categorical: {{list a few — e.g. `plan_type`, `country`, `device`}}
  - Drop / never feature: {{`user_id`, `email`, anything that's a key
    or PII proxy}}
- **Class balance / target distribution**: {{e.g. 12% positive class,
  or median target value 4.2 with long right tail}}

## What you can edit

Only `src/plugins/{{private/}}{{PLUGIN_NAME}}.py`. This file defines a
`{{PluginClass}}` implementing the `TrainingPlugin` protocol with
`task = "{{TASK_TYPE}}"`.

You may:

- Change the **estimator class** (any sklearn classifier/regressor with
  `.fit()` / `.predict()`{{ and `.predict_proba()` if classification}};
  e.g. {{RandomForest, GradientBoosting, HistGradientBoosting,
  LogisticRegression, Ridge, ElasticNet, …}}).
- Change **hyperparameters** in `build_model()`.
- Add **feature engineering** in `prepare()` — interactions, log
  transforms, ratios, target encoding for categoricals, polynomial
  features, etc.
- Wrap the estimator in a sklearn **`Pipeline`** if useful.

## What you must not do

- Don't edit any file other than `src/plugins/{{PLUGIN_NAME}}.py`.
- Don't import outside the standard scientific Python stack
  (sklearn, numpy, pandas, scipy{{ + category_encoders if you've
  installed it}}).
- Don't change `name = "{{PLUGIN_NAME}}"` or `task = "{{TASK_TYPE}}"` —
  the registry depends on them.
- Don't peek at the test set during training (the harness handles the
  split; just provide `X` and `y` from `prepare()`).
- Don't load any external data — work with what's in the dataframe.
{{ delete or expand any rule above that doesn't apply }}

## Output format

Output a COMPLETE new version of the plugin file. Plain Python source
— no markdown fences, no commentary, no diff format. Just the file.

### prepare(df) — the only safe pattern

`y` MUST be extracted from `df` BEFORE any transformation that could
remove the `{{TARGET_COL}}` column. Use this template literally; only
modify the inner "feature engineering" section:

```python
_SKIP = {"{{TARGET_COL}}", "signal_id", "event_timestamp"{{, list any
         entity_id / pii / leakage cols here}}}

def prepare(self, df: pd.DataFrame):
    # 1. Extract the target FIRST.
    y = df["{{TARGET_COL}}"].astype({{int / float}})

    # 2. Build the feature frame from a copy of df.
    X = df.drop(columns=_SKIP, errors="ignore").copy()

    # 3. Feature engineering on X (optional). Examples:
    #    X["log_sessions"] = np.log1p(X["num_sessions"])
    #    X["sessions_per_day"] = X["num_sessions"] / (X["days_since_signup"] + 1)
    #    X = pd.get_dummies(X, columns=["plan_type"], drop_first=True)

    return X, y
```

### Common failure modes — DO NOT do these

- **Don't drop the target then try to read it.**
  ```python
  df = df.drop(columns=["{{TARGET_COL}}"])
  X = df
  y = df["{{TARGET_COL}}"]  # KeyError
  ```
- **Don't apply `pd.get_dummies` to the whole df at once** — it'll one-hot
  the target if it's not numeric. Apply transforms to `X` after the y/X split.
- **Don't return numpy arrays** from `prepare()`. The harness expects a
  `pd.DataFrame` for `X` so `list(X.columns)` works downstream for the
  bundle config.

## Strategy hints for {{DATASET_NAME}}

{{ Replace these with hints that actually apply to your data. Examples
   from past projects:

   - Highly imbalanced target → use `class_weight="balanced"`, or
     scale_pos_weight on gradient boosting, or threshold tuning
     post-fit.
   - Long-tail categorical (10K+ values for a column): use target
     encoding rather than one-hot.
   - Heavily skewed continuous features: log1p transform, then standard
     scale if you're using a linear model.
   - Strong temporal autocorrelation: be cautious about random
     train/test splits leaking — though this harness uses random_state=42
     so all comparisons are apples-to-apples for the agent's purposes.
   - Mostly noise + a few strong signals: tree ensembles handle this
     well; linear models with L1 can find the sparse signal.
}}

## Metric

The trainer prints `{{METRIC_NAME}}=0.XXXX`. Higher is better
({{describe the scale, e.g. AUC ∈ [0,1], R² ≤ 1, accuracy ∈ [0,1]}}).

The current baseline ({{e.g. RandomForestClassifier with default
sklearn params}}) sits around {{BASELINE_RANGE}}. Aim higher.
