# Semantic search (FAISS over product titles)

Given a free-text query (or another product's title), return the
top-K most similar products from the catalog. Powers
recall-first product discovery, "find me something like this,"
and the recall stage of a two-stage matcher/recommender cascade.

## What it is

Each product title is encoded once as a 384-dim vector using
`sentence-transformers/all-MiniLM-L6-v2` (80 MB, fast on CPU). The
vectors are stored in a FAISS index. At query time, the query text
is embedded once and FAISS returns the top-K nearest by L2 distance.

The bundle layout extends sage-baker's standard one with
`index.faiss` + `corpus.parquet` — the FAISS index plus the metadata
to return at query time. Embedder loaded **once** at container
startup, query is one transformer forward pass + sub-ms FAISS lookup.
This avoids the "embed-per-request" anti-pattern that makes naive
implementations slow.

## Quickstart

```bash
make install-retrieval        # one-time: sentence-transformers + faiss-cpu
make data-products            # ./data/products/  (360 retailer offerings)
make train-search             # bundle in ./models/search/  (corpus.parquet + index.faiss)
```

Then test queries in-process:

```python
import sys; sys.path.insert(0, "src")
import train_retrieval
model = train_retrieval.model_fn("./models/search")

results = model.predict(["iphone 15 pro 256"], k=5)[0]
for r in results:
    print(r["retailer"], r["title"], r["_distance"])
```

Demonstrated query behavior (see commit history for the reproducible
output):
- `"iphone 15 pro 256"` → all 3 retailer offerings of iPhone 15 Pro
  256GB at distance ≈ 0.21
- `"noise cancelling headphones"` → Sony WH-1000XM5 (no literal
  keyword match)
- `"stand mixer kitchenaid"` → KitchenAid Artisan listings across
  retailers despite title noise (lowercased, brand reordered)

## Files

| Path | What it is |
| --- | --- |
| [`src/plugins/product_search.py`](../../src/plugins/product_search.py) | Retrieval plugin: title → vector, generic over catalog shape |
| [`src/plugins/base_retrieval.py`](../../src/plugins/base_retrieval.py) | `RetrievalPlugin` base contract (prepare_corpus, build_embedder, build_index, query) |
| [`src/train_retrieval.py`](../../src/train_retrieval.py) | Indexing harness — embed corpus in one batch, build FAISS, persist |
| [`requirements-retrieval.txt`](../../requirements-retrieval.txt) | sentence-transformers + faiss-cpu |
| `models/search/` | Bundle output: `config.json`, `index.faiss`, `corpus.parquet`, `metadata.json` (gitignored) |

## Try it different ways

### Swap the embedder

`product_search.embedder_model` defaults to
`sentence-transformers/all-MiniLM-L6-v2`. Try
`mpnet-base-v2` (higher quality, ~500ms vs 100ms encode) or a tighter
distilled MiniLM-3 layer (faster, lower quality). Edit the plugin
attribute, retrain, reload.

### Different index types

`base_retrieval.RetrievalPlugin.build_index` defaults to `IndexFlatL2`
(exact, fine for ~100K items). Override in a subclass for IVF /
HNSW / product-quantized variants when the corpus is larger. The
bundle still serializes via `faiss.write_index` regardless.

### Use it as the recall stage of a cascade

For "similar products that will convert," combine retrieval (this
example) with a ranker:

1. **Recall** (~ms): top-K candidates from FAISS by embedding
   distance.
2. **Rank** (~10ms): a binary classifier scores `(user_features,
   candidate_product_features) → P(convert)` for each candidate.

The user-side ranker would be a new plugin similar to
[product-matching](../product-matching/) but with user features
joined in.

## Scale to production

The two real considerations going from local to prod:

- **Embedder weights vs. HF download.** Today the bundle records
  the HF model name and the inference container downloads on first
  load. Production (especially air-gapped) wants the weights
  vendored into the bundle (safetensors) for offline portability.
  Tracked as a follow-on; see the docstring in
  [`product_search.py`](../../src/plugins/product_search.py).
- **Index size + serving cost.** FAISS in-process is fine up to
  ~10M items. Beyond that, you're looking at a managed vector DB
  (Pinecone, Weaviate, pgvector). The bundle abstraction still
  applies — `build_index` would write metadata pointing at the
  vector-DB collection rather than a local file.

The local-iterate / cloud-train pattern works the same here: commit
the plugin code, cloud CI rebuilds the index over the full
warehouse catalog, registers via prod MLflow.
