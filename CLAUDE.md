# sagebaker — Claude Code working notes

A personal sandbox for ML pipeline patterns: training plugins, the
bundle architecture (config + weights split), MLflow / Feast /
BigQuery integration, an autoresearch-style agent loop, and a path
toward SageMaker.

`README.md` is the canonical reference. This file is the short
"what to know before doing anything in this repo" for Claude.

## Repo shape

- `src/plugins/` — plugin classes implementing `TrainingPlugin`. New
  models go here. `src/plugins/private/` is gitignored for work code.
- `src/train.py`, `src/train_torch.py`, `src/train_lightgbm.py`,
  `src/train_recommender.py` — training drivers per framework.
- `agent.py` — autoresearch loop driving Anthropic API over a plugin.
- `prepare_*.py` — dataset prep scripts; each writes
  `data/<name>.{csv,parquet}` + `data/lineage.json`.
- `models/<plugin>/` — bundle output (config.json + metadata.json +
  weights). **Gitignored.**
- `data/`, `mlflow.db`, `mlartifacts/`, `mlruns/`, `materialized/` —
  generated. **Gitignored.**

## Skills available

- `/productionize [<plugin>]` — generate a starter notebook from the
  best agent run. Reads `models/<plugin>/{config,metadata}.json` and
  writes `notebooks/<plugin>_productionize.ipynb`. See
  `.claude/skills/productionize/SKILL.md` for the full spec.

## Two ways to run the agent loop

1. **`agent.py` (autonomous, headless)** — needs `ANTHROPIC_API_KEY`
   in `.env`. Good for overnight runs. Edits a plugin file in place,
   trains, keeps if better, reverts otherwise. Five safety features
   (baseline run, failure feedback, byte-identity guard, stuck signal,
   compact diff) make it converge instead of churn.

2. **Chat in this Claude Code session (interactive)** — *this* is
   itself an agent harness. The user can say "iterate on
   `src/plugins/X.py` toward higher AUC, retrain via `make train`,
   keep best, show me the diff each time." No API key needed because
   the LLM is already the assistant in chat. Slower per iteration but
   pausable, redirectable, and free under existing CC usage.

The skill suite + `make` targets are designed for path 2; `agent.py`
is path 1.

## Conventions

- **Bundle architecture**: every trainer writes
  `config.json` + `metadata.json` + a weights file. Loaders dispatch
  on `weights_format` (joblib / skops / safetensors / lightgbm-text),
  never on pickled class identity. `model_fn(model_dir)` is the
  contract.
- **Private code**: anything specific to the user's day job lives
  under `src/plugins/private/` (gitignored) or `Makefile.private`
  (also gitignored). Never commit work-specific column names,
  business logic, or company names.
- **Treat additions as exploratory** — sagebaker is a playground,
  not a product. Demo scripts and runnable patterns beat abstraction.
- **`make help`** lists every target.

## Common workflows

```bash
# Train + evaluate the default plugin
make data-sonar
make train

# Run the autonomous agent loop
make agent

# Open Jupyter (with .env exported, BQ credentials wired up)
make jupyter

# Bring up the local MLflow UI
make mlflow-server
```

In a Claude Code session:
```
/productionize default
```

## When responding

- Frame ML choices in terms of the production pain they fix
  (pickle coupling, train/serve skew, unseen categoricals) — those
  are the concrete bugs that motivated this sandbox.
- Prefer pragmatic shortcuts over canonical paths: the project has
  chosen access keys over SSO, BYOC over DLC, SQLite over DynamoDB.
- For exploratory questions, give 2-3 sentence recommendations with
  tradeoffs, not full implementation plans.
- For optimization loops, prefer continuous metrics (AUC, R²) over
  accuracy on small test sets — accuracy quantizes too coarsely and
  the loop deadlocks.
