# Examples

Sage-baker by **problem** rather than by file type. Each subdirectory
is a focused README pointing at the real code (plugins, prep scripts,
programs) — examples don't duplicate code, they just orient you.

## Pick the one closest to your day-job problem

| Scenario | Problem | Plugin | Synthetic data |
| --- | --- | --- | --- |
| [conversion-prediction/](conversion-prediction/) | Predict session conversion from anonymous clickstream | `clickstream` | `fuzzy_clickstream` |
| [product-matching/](product-matching/) | Cross-retailer dedup: same canonical product? | `product_matcher` | `product_catalog` |
| [semantic-search/](semantic-search/) | Find similar products by title (FAISS retrieval) | `product_search` | `product_catalog` |
| [record-linkage/](record-linkage/) | Link anonymous events to the same true user | `clickstream_linkage` | `fuzzy_clickstream` |
| [recommender/](recommender/) | Collaborative-filtering top-K (ALS) | `als` | MovieLens-100K (real public data) |
| [regression/](regression/) | Continuous target prediction | `housing` | sklearn's California housing |

## How each example is structured

Same five sections every time, so once you've read one you know the
shape:

1. **What it is** — the ML problem in one paragraph
2. **Quickstart** — one or two `make` commands to reproduce
3. **Files** — paths to the plugin, prep script, program (if the
   scenario supports the agent loop), data path, model dir
4. **Try it different ways** — agent loop, hyperparameter changes,
   threshold tuning, harder data settings
5. **Scale to production** — what changes when this moves from
   local laptop to staging to SageMaker. The local-iterate /
   cloud-train pattern from the main README, made concrete per
   scenario

## Suggested onboarding path

1. Pick whichever scenario is closest to your real ML problem.
2. Run its Quickstart end-to-end. Confirm `validation_<metric>=…`
   matches the README.
3. Read its plugin file and prep script.
4. Open the [main README](../README.md) to understand the
   architecture pieces you just used (plugin contract, bundle
   layout, MLflow tracking, etc.).
5. Try the agent loop on the scenario. Watch what proposals get
   kept vs reverted.
6. When you've got something working, follow the "Scale to
   production" section to see how the same code runs in cloud CI
   against the full data.
