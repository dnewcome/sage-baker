# `feature_store/` — Feast feature definitions + state

This is sagebaker's [Feast](https://feast.dev/) feature store. Two
purposes live here:

1. **Source** (tracked in git) — Python files defining what entities
   and feature views Feast knows about, plus the `feature_store.yaml`
   that picks the storage backends.
2. **State** (gitignored, regenerated) — the parquet files containing
   actual feature values, the registry `.db`, and the online store
   `.db`. Produced by `prep/prepare_sonar.py` + `feast apply` +
   `feast materialize`.

## What's in here

| File | Tracked? | Purpose |
|---|---|---|
| `feature_store.yaml` | ✓ | Feast config: project name, registry path, online/offline store types |
| `entities.py` | ✓ | `Entity` definitions — what features are *about* (e.g. `sonar_signal`) |
| `features.py` | ✓ | `FeatureView` definitions — feature schemas keyed by entity |
| `data/*.parquet` | gitignored | Offline store source data (the actual feature values) |
| `data/*.db` | gitignored | Registry + online store SQLite files (regenerable from `feast apply`) |

The split is intentional: feature definitions are code (review, diff,
version), feature *values* are data (regenerable, not committed).

## How it's used

### At training time — `src/train_feast.py`

Pulls features by point-in-time join against the offline store, so
training rows have features as they existed at each row's
`event_timestamp` (no future leakage):

```bash
make data-sonar          # writes feature_store/data/sonar_features.parquet
make feast-apply         # `feast apply` + `feast materialize-incremental`
make train-feast         # train_feast.py joins features by signal_id, fits sklearn
```

### At serving time — `local_serve.py --feature-store`

Looks up features online by entity ID (`signal_id`) from the SQLite
online store, then calls `model_fn` to predict:

```bash
.venv/bin/python local_serve.py \
    --model-dir ./models/feast \
    --feature-store ./feature_store
# → fetches f0..f59 for each signal_id from data/online_store.db,
#   then runs the bundled sklearn model.
```

Same `model_fn(model_dir)` runs in both contexts; only the feature
source changes.

## Quickstart (cold start)

```bash
make install-feast       # pip install --group feast
make data-sonar          # prep script writes the parquets here
make feast-apply         # registers definitions, populates online store
make train-feast         # historical-join training
.venv/bin/python local_serve.py --model-dir ./models/feast --feature-store ./feature_store
```

To wipe and rebuild from scratch:

```bash
rm -rf data/registry.db data/online_store.db
make feast-apply
```

## Translating to production

Two config swaps in `feature_store.yaml` move this from local to
SageMaker-deployable, with no Python changes:

| Local | Production |
|---|---|
| `online_store.type: sqlite` | `postgres` (RDS) or `redis` (ElastiCache) |
| `offline_store.type: file` + `FileSource(path=...)` (in `features.py`) | swap path to `s3://bucket/key` — same `FileSource` class works |

The Feast registry file (`registry.db`) can be moved to S3 by setting
`registry: s3://bucket/registry.db`. After that, training and serving
talk to managed backends instead of local files. See the README's
"Feast on the DLC path" section for the SageMaker integration story.

## See also

- [README.md](../README.md) — sections "Feature store: Feast prototype"
  and "Feast on the DLC path" for the deeper architecture story
- [`src/train_feast.py`](../src/train_feast.py) — host-side trainer
  with point-in-time historical join
- [`drivers/local_train_feast_dlc.py`](../drivers/local_train_feast_dlc.py)
  — host fetches features, container trains
- [`local_serve.py`](../local_serve.py) — `--feature-store` flag
  switches feature source from request payload to Feast online lookup
