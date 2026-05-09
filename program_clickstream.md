# Agent program for clickstream conversion classification

You are an autonomous ML research agent. Your job: iteratively improve
the **validation_auc** of a binary classifier predicting
**session-level conversion** on synthetic clickstream data, by editing
**one Python file**: `src/plugins/clickstream.py`.

## Dataset shape

- **Source**: `simulate/scenarios/fuzzy_clickstream` produces
  `./data_fuzzy/training.parquet` (default 500 users × ~4 sessions ×
  ~4 events ≈ 8K rows). Run `make data-fuzzy` to regenerate.
- **Ground truth** (you must NOT use it as features): the simulator
  writes `./data_fuzzy/ground_truth.parquet` with `true_user_id`,
  `true_cohort`, `is_converted_session`. The harness never feeds this
  to your plugin — `train.py` only reads `training.parquet`.
- **Per-event columns in training.parquet**:
  - `event_id`, `timestamp`, `session_id`
  - `user_id` (NULL on logged-out events; 100% NULL by default since
    `identified_user_fraction=0.0`)
  - `device_fingerprint` (stable per true user, near-unique by default —
    treat as high cardinality)
  - `ip_bucket` (small integer; multiple users share)
  - `event_type` ∈ {page_view, click, add_to_cart, checkout, conversion}
  - `referrer` ∈ {google, direct, facebook, email, affiliate, organic}
  - `value` (0 except on conversion events)
  - `session_converted` — the binary label, broadcast to every event
    in the session
- **Class balance**: ~5–7% positive sessions (cohort-driven; the
  `intender` cohort converts ~2.5× the base rate).

## What you can edit

Only `src/plugins/clickstream.py`. The plugin must remain a
`ClickstreamPlugin(TrainingPlugin)` with `name = "clickstream"` and
`task = "classification"`. You may:

- Change the **estimator class** in `build_model()` (any sklearn
  classifier with `.fit()` / `.predict_proba()`; HistGradientBoosting,
  RandomForest, GradientBoosting, LogisticRegression with scaling,
  etc.).
- Tune **hyperparameters**.
- Add **feature engineering** in `prepare()`. Aggregate event-level
  data however you like — additional event-type counts, distinct
  page counts, time-window stats, ratios, interaction features, etc.
- Wrap the estimator in a sklearn `Pipeline` (e.g. with
  `StandardScaler` for linear models).

## What you must not do

- **Don't introduce any feature that encodes the post-decision event
  count, directly or indirectly.** The simulator deterministically
  appends `add_to_cart`, `checkout`, and `conversion` events to
  converting sessions, and *zero* such events to non-converting ones.
  Any feature that recovers this count = a trivial 1.0-AUC model
  that learned the simulator's funnel shape, not generalizable signal.

  Specifically forbidden — these all leak:
  - Counting `add_to_cart`, `checkout`, or `conversion` events as features.
  - `total_events_per_session` (raw count from `df`, not filtered by
    `_PRE_DECISION_EVENT_TYPES`). This *includes* post-decision events.
  - `total_events − n_pre_events` or any equivalent subtraction. This
    IS the post-decision count by construction.
  - `value` summed/maxed per session — `value > 0` only on conversion
    events.
  - `df["event_type"].nunique()` per session — converting sessions have
    more event types.
  - Any aggregation over the *full* `df` rather than the pre-decision
    subset. Compute features only from `pre = df[df["event_type"].isin(_PRE_DECISION_EVENT_TYPES)]`.

  Sanity check: after `prepare(df)`, run the model on a held-out split
  and look at `feature_importances_`. If any single feature alone
  gives >0.95 AUC, you have a leak — stop and remove it. The honest
  baseline on this dataset sits around 0.74; anything above ~0.85 is
  almost certainly leakage.
- Don't peek at `ground_truth.parquet` — it's not loaded by the
  harness, but you also can't try to.
- Don't drop the session aggregation; the target is per-session, so
  rows must be unique per `session_id` before train/test split.
- Don't change `name = "clickstream"` or `task = "classification"`.
- Don't import anything outside the standard scientific Python stack
  (sklearn, numpy, pandas, scipy).

## Output format

Output a COMPLETE new version of `src/plugins/clickstream.py`. Plain
Python source — no markdown fences, no commentary, no diff. Just the
file.

### prepare(df) — the safe pattern

The aggregation must produce one row per `session_id`. The trainer
will then split by row, which becomes a session-level split. Do
**not** return event-level rows.

```python
def prepare(self, df: pd.DataFrame):
    # 1. Filter to pre-decision events for feature computation.
    pre = df[df["event_type"].isin(_PRE_DECISION_EVENT_TYPES)].copy()

    # 2. Aggregate to session level.
    features = pre.groupby("session_id").agg(...).reset_index(drop=True)

    # 3. Per-session target — broadcast value, take first.
    targets = df.groupby("session_id")["session_converted"].first().astype(int)

    # 4. Align indices and return.
    return features, targets.loc[features.index]
```

### Common failure modes — DO NOT do these

- **Don't pivot on the full `event_type` set without filtering** —
  that re-introduces add_to_cart/checkout/conversion as features.
- **Don't use `value` directly** — it's nonzero only on conversion
  events, so summing it across the session is essentially the label.
- **Don't return a DataFrame indexed by session_id** without
  resetting the index — `train.py` does `train_test_split(X, y)`
  which expects integer-indexed rows.
- **Don't pass `max_features` to `HistGradientBoostingClassifier`** —
  it doesn't accept that argument and will raise `TypeError` at
  init. `max_features` exists on `RandomForestClassifier` and
  `GradientBoostingClassifier` (the non-Hist variants); the Hist
  version uses `max_leaf_nodes` / `max_bins` / `l2_regularization`
  for capacity control instead. If you want to switch estimators
  to one that supports `max_features`, change the import to
  `RandomForestClassifier` or `GradientBoostingClassifier`.

## Strategy hints

- **Cohort signal** is buried in pre-decision behavior counts and
  referrer mix. Different cohorts (browser, intender, bargain_hunter,
  loyalist) have different click/page-view ratios — try ratio
  features like `clicks / pre_events`.
- **`ip_bucket`** is a small-integer categorical with ~150 unique
  values for 500 users. Tree models split it like a categorical;
  linear models would need target encoding or hashing.
- **`device_fingerprint`** is near-unique per user with default
  scenario settings, so by itself it's a noise feature for a
  session-level model. Consider `device_fingerprint_hash % K` for
  modest K (e.g., 32, 64) to encode loose neighborhoods.
- **Time features**: hour of day, day of week from `timestamp` —
  cohort behavior likely varies by time.
- **Time-between-events**: median gap, max gap, percentiles. Engaged
  users have rhythmic gaps; bouncers have one big gap.
- **Class imbalance**: `class_weight="balanced"` on
  RandomForestClassifier or `scale_pos_weight` on gradient boosting
  can help on the ~6% positive rate. Also consider
  `HistGradientBoostingClassifier(class_weight="balanced")`.

## Metric

The trainer prints `validation_auc=0.XXXX`. Higher is better
(AUC ∈ [0, 1], 0.5 = random).

The current baseline (HistGradientBoostingClassifier with default
plugin settings) sits around **0.74**. Aim higher.
