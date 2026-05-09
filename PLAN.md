# Plan

## Phase 1 — Regression task + plugin (DONE 2026-05-08)

Goal: extend sagebaker beyond binary/multiclass classification so the
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
make train MODEL_DIR=./models/housing --plugin housing
make serve MODEL_DIR=./models/housing
.venv/bin/python evaluate.py --model ./models/housing --test ./data --output ./eval/housing
cat ./eval/housing/metrics.json
```

Expectation: bundle has `task: "regression"` and `validation_r2` in
metadata; `local_serve` shows residuals; `evaluate.py` produces R² +
RMSE + MAE.

---

## Phase 2 — Public-data recommender (DONE 2026-05-08)

Goal: make `make train-als` work without synthetic data, on a real
public dataset, so the recommender path is exercisable on a clone.

### Result

`prep/prepare_movielens.py` fetches MovieLens-100K (1.7 MB, no auth) from
grouplens.org, maps the schema to what the ALS plugin already
expects (`user_id` / `item_id` / `weight` + bonus `timestamp`),
writes `data/movielens.csv` + `lineage.json`. ALS plugin needed zero
changes.

End-to-end:

```
make data-movielens && make install-recommender && make train-als
```

Metrics on the real data: hit_rate@10 = 0.82, recall@10 = 0.16,
ndcg@10 = 0.21 — within the typical ALS-on-MovieLens-100K range.

Single make target added: `data-movielens`.

---

## Phase 3 — Smoke test suite (now)

Goal: lock in the contracts that this session has stabilized — bundle
round-trip, plugin protocol, evaluate-signature dispatch, config-only
model rebuild — so future edits surface regressions instead of
silently breaking the `/productionize` skill or `agent.py` loop.

### Tests to write

- **`test_bundle_roundtrip.py`** — train HousingPlugin on a 200-row
  fixture, save bundle, reload via `train.model_fn`, assert
  predictions match. Covers the joblib path; one parametrized
  variant covers the skops path.
- **`test_plugin_contract.py`** — every registered supervised plugin
  has `name`, `task` ∈ {classification, regression}; `prepare(df)`
  returns `(DataFrame, Series)`; `build_model({})` returns a fittable
  estimator with `.fit` / `.predict`.
- **`test_evaluate_signatures.py`** — `train.py` accepts both
  `evaluate(y_true, y_pred)` (legacy) and
  `evaluate(model, X_test, y_true)` (new). Exercises the
  `inspect.signature` dispatch.
- **`test_config_rebuild.py`** — the `/productionize` invariant: rebuild
  the estimator from `config.json` (class + params, no pickle) → fit
  → predictions are byte-identical to the bundled weights. If this
  fails, `/productionize` is lying.
- **`test_lineage_capture.py`** — prepare-script writes `lineage.json`,
  `train.py` reads it, lineage shows up in `metadata.json`.

### Constraints

- Run in <30s on a clean clone (HousingPlugin's `prepare_data()` is
  sklearn-bundled, no network).
- No fixtures larger than ~200 rows.
- Use `pytest`. Already installed in the venv.

### Files to add

- `tests/conftest.py` — shared fixtures: tmp model dir, the housing
  dataframe, a mock plugin for the signature-dispatch test.
- `tests/test_*.py` — one file per area above.
- `requirements-dev.txt` — `pytest` (and that's it for now).
- `Makefile` target `test` — runs `pytest -q tests/`.

### Out of scope this phase

- GitHub Actions wiring. Easy follow-on once the suite is green
  locally; just adds `pip install -r requirements.txt
  -r requirements-dev.txt && make test` to a workflow.
- Recommender plugin tests (different harness, separate file shape).
- BigQuery / Feast / DLC tests — these need credentials or extra
  dependencies; they'd be integration tests, not smoke.

---

## Phase 4+ candidates (not committed to)

Stuff that came up in the planning conversation as natural follow-ons.
Listed in rough value × ease order; pick whatever the next push needs.

- **Per-dataset agent Makefile targets** — `make agent-housing`,
  `make agent-sonar`, `make agent-bq`. One-line shortcuts for the
  multi-dataset workflow.
- **Run history persistence** — `runs/<plugin>/<timestamp>.json` per
  iteration of `agent.py`. Today the loop is amnesic between
  invocations; persisting it lets you compare strategies across runs
  and is the data layer the eventual REST/MCP store would query.
- **Dataset fingerprinting helper** — small util that hashes
  `(shape, target_dtype, target_distribution, feature_dtype_counts)`.
  Prerequisite for similarity-based recall.
- **Knowledge backing store (REST first, MCP later)** — once run
  history + fingerprints exist there's something real to serve. REST
  is the simpler protocol to iterate schema on; MCP shim layers on
  top once the surface stabilizes.
- **GitHub Actions CI** — wraps Phase 3 once the test suite is green.

---

## Phase 5 — LLM fine-tune sandbox (planned, not committed)

Goal: prove sagebaker's bundle/productionize patterns extend to a
LoRA-fine-tuned open-weights LLM, end-to-end, deployed via SageMaker.
Frame the work as "what's the smallest demo that lets us evaluate
whether this is worth productizing for the company?" Keep the agent
loop out — LLM training iterations are too slow for that scope.

### Phase 5a — Classification fine-tune (start here)

**Use case (TBD, pick one):**
- Ticket / email routing — multi-class over internal categories
- Field extraction — structured JSON from semi-structured docs
- Entity resolution — pair-level "same canonical?" (extends the
  existing `product_matcher` plugin with title-LLM features)

The classification angle maps cleanly to the existing
`TrainingPlugin` contract — output is a class label, evaluation is
accuracy/F1, predict-proba semantics survive. Lowest-risk on-ramp.

**Model + training stack:**
- Base: 3B–8B open weights (Llama 3.1 8B Instruct, Qwen 2.5 7B,
  or Mistral 7B). Pin one once the use case is locked.
- Adapter: LoRA via `peft` — only adapter weights trained,
  ~10–50 MB instead of the 14 GB base.
- Trainer: HuggingFace `Trainer` + `accelerate`, single-GPU
  (A10G or L4 fits a 7B-with-LoRA at batch 1–4).
- Container: HuggingFace SageMaker DLC (skip BYOC for this phase
  unless DLC version pins block us).

**New bundle contract (the interesting part):**

```
models/<plugin>/
├── config.json           ← base_model_id, peft_config (rank/alpha/targets),
│                           label_map, tokenizer_name, max_seq_len,
│                           weights_format: "lora-adapter"
├── metadata.json         ← training_tokens, training_seconds,
│                           hardware, eval metric, lineage
├── adapter_model.safetensors  ← LoRA weights only
└── tokenizer/            ← tokenizer.json + special_tokens_map
```

`model_fn(model_dir)` dispatches on `weights_format == "lora-adapter"`
→ load base from HF cache, apply adapter, return a wrapper exposing
sklearn-compatible `predict` / `predict_proba`. Base model isn't
bundled (multi-GB) — the inference container caches it once and
reuses across requests.

**Files to add:**
- `src/plugins/llm_classifier.py` — `LlmClassifierPlugin`
- `src/train_llm.py` — HF Trainer + PEFT driver
- `prep/prepare_<usecase>.py` — dataset prep → parquet with
  `text` + `label`
- `pyproject.toml` `[dependency-groups] llm` —
  `transformers`, `peft`, `accelerate`, `datasets`, `bitsandbytes`
  (4-bit base loading), `sentencepiece`
- `Makefile` targets: `data-<usecase>`, `train-llm`, `install-llm`

**Files to modify:**
- `src/bundle.py` — handle `lora-adapter` weights format
- `local_serve.py` — dispatch on `lora-adapter` (route to the new
  plugin's `model_fn`)
- `src/plugins/base.py` — no change anticipated; the classification
  contract should still fit

**Deploy path:**
1. Train via `src/train_llm.py` on a SageMaker Training Job (HF DLC,
   single `g5.2xlarge` or `g6.2xlarge`).
2. Bundle uploaded to S3.
3. Deploy via the existing `deploy_endpoint.py` flow but with the
   HF inference DLC and a custom `inference.py` that runs the
   plugin's `model_fn` + `predict`.
4. Smoke-test against `local_serve.py` first (loads adapter into a
   local base model) — same script that handles the sklearn case
   today.

**Smoke test (the experiment's success criterion):**
```bash
make data-<usecase>          # 1000-row dataset, 5–10 classes
make install-llm
make train-llm               # ~30 min on a single A10G; LoRA only
make serve MODEL_DIR=./models/llm-<usecase>
# query with a few held-out examples; eyeball accuracy
.venv/bin/python evaluate.py --model ./models/llm-<usecase> --test ./data
cat ./eval/llm-<usecase>/metrics.json
```

Bundle shape proves out, contract changes stay minimal, deploy path
matches the sklearn case.

### Phase 5b — Generative assistant (deferred)

Out of scope for the first experiment. If 5a goes well, the natural
follow-on is a generative use case (internal Q&A, code helper, draft
generation). What would need to change:

- **Plugin contract** — `predict` doesn't fit; need `generate(prompt,
  max_tokens, ...)`. Possibly a new base class `GenerativePlugin`.
- **Bundle weights** — adapter alone may not suffice; could need a
  merged-and-quantized full model for inference (4–8 GB instead of
  10s of MB). Re-think `weights_format`.
- **Serving** — vLLM or TGI on a dedicated GPU instance, not the HF
  inference DLC. Streaming responses need a different `model_fn`
  contract (yield tokens, don't return a single prediction).
- **Evaluation** — accuracy doesn't apply. Use `lm-evaluation-harness`
  for benchmarks, or human eval on a held-out task set. Not a
  single-metric agent loop.
- **Cost** — generative inference is meaningfully more expensive than
  classification; budget the experiment differently.

Don't start 5b until 5a's bundle/deploy contract is locked in. The
risk: if 5a doesn't ship cleanly, 5b's added complexity hides the
underlying question (does the bundle pattern extend at all?).

### Out of scope this phase (5a + 5b both)

- **Pretraining from scratch** — wrong tool entirely; use Megatron-LM
  / NeMo / Composer.
- **Multi-GPU / distributed training** — 7B LoRA fits on a single
  A10G; if it doesn't, re-scope.
- **RAG / vector store** — orthogonal to fine-tuning. The retrieval
  plugin already exists for that path.
- **Prompt engineering vs. fine-tune comparison** — interesting but
  scope creep. Pick one for the experiment.
- **Model merging / model souping** — research-y, not productizable
  yet, and wouldn't match the bundle shape cleanly.

---

## Out of scope for the foreseeable future

- **Hyperparameter tuning beyond agent.py** — Optuna / sklearn
  HalvingGridSearch could plug into the same harness later.
- **More public datasets** (adult-income for categoricals,
  fashion-mnist for torch) — same pattern as housing once the
  abstraction is in place.
- **Drift detection** (Evidently/TFDV) — orthogonal; do once we have
  enough deployed-model usage to make it relevant.
