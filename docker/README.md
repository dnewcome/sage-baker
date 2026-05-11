# sagebaker serving images

Two base images covering the common model dependency families.
Plugin code and model weights are injected at runtime — nothing
model-specific is baked in.

## Images

| Image | Base | Added deps | Use for |
|-------|------|-----------|---------|
| `tabular` | AWS sklearn DLC | LightGBM, imbalanced-learn | Classification/regression with LightGBM or sklearn |
| `embedding` | AWS PyTorch DLC | sentence-transformers, FAISS, spaCy | Recommenders, semantic search, dense embeddings |

## Runtime configuration

Both images are configured entirely via environment variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `PLUGIN_NAME` | yes | Plugin to serve (e.g. `fill`) |
| `MODEL_S3_URI` | yes | S3 URI of the bundle directory |
| `PLUGIN_DIR` | no | Path to plugin `.py` files (if not in image) |
| `PORT` | no | Serving port (default: 8080) |

## Building

```bash
# tabular
docker build -t sagebaker-tabular:latest docker/tabular/

# embedding
docker build -t sagebaker-embedding:latest docker/embedding/

# pin versions
docker build \
  --build-arg SAGEBAKER_VERSION=0.2.0 \
  --build-arg LIGHTGBM_VERSION=4.3.0 \
  -t sagebaker-tabular:0.2.0 docker/tabular/
```

## Deploying a plugin without rebuilding

Bake plugin `.py` files into a thin layer on top of the base:

```dockerfile
FROM sagebaker-tabular:1.0
COPY plugins/ /opt/ml/plugins/
ENV PLUGIN_DIR=/opt/ml/plugins
```

Then configure the model via ECS env vars:

```json
{ "name": "PLUGIN_NAME", "value": "fill" },
{ "name": "MODEL_S3_URI", "value": "s3://your-bucket/models/fill/v42/" }
```

Updating the model = new S3 path + ECS force-new-deployment. No image rebuild.
