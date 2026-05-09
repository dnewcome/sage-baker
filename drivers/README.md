# `drivers/` — SageMaker training drivers

Each file here is a small Python script that hands a training job to
SageMaker's Python SDK. They're not training code themselves — the
real training lives in `src/train.py` (and friends). Drivers are the
*invocation* layer: they pick a container, point it at data, set
hyperparameters, and call `.fit()`.

Think of them as the "submit a SageMaker job" recipes, with all the
fiddly boilerplate (container roots, Local Mode quirks, framework
versions) pre-tuned for this repo.

## How a driver works

Every driver follows the same shape:

1. **Set up a SageMaker session** — `LocalSession()` for local mode,
   regular `Session()` for cloud.
2. **Construct an estimator** — either a generic `Estimator(image_uri=...)`
   for BYOC, or a framework-specific one (`SKLearn`, `PyTorch`,
   `HuggingFace`) for DLC.
3. **Point at training data** — a `file://` URL for local mode, an
   `s3://` URL for cloud.
4. **Call `estimator.fit({channel: data_uri})`** — SageMaker mounts
   the data inside the container at `/opt/ml/input/data/<channel>/`
   and runs the entry point.

Inside the container, training reads from `/opt/ml/input/data/<channel>/`,
writes the bundle to `/opt/ml/model/`, and SageMaker tarballs it on
exit. This part is universal across drivers — it's `src/train.py`'s
job, and it doesn't know whether it's running on a laptop or in the
cloud.

## What's in this directory

| Driver | Path | When to use |
|---|---|---|
| `local_train.py` | BYOC + Local Mode | No AWS account needed. Builds and runs your own Docker image. Fastest iteration; full control. |
| `local_train_dlc.py` | DLC + Local Mode | Uses the AWS-pre-built scikit-learn Deep Learning Container. Needs AWS creds (any account) for the ECR pull. Matches AWS-canonical pipelines. |
| `local_train_feast_dlc.py` | DLC + Local Mode + Feast | Pattern: feature retrieval happens *outside* the container (host-side Feast point-in-time join), the joined parquet is the training channel. Translates cleanly to a real SageMaker Pipeline (Processing → Training). |

The matching Makefile targets:

```bash
make train-byoc        # → drivers/local_train.py     (after `make image`)
make train-dlc         # → drivers/local_train_dlc.py
make train-feast-dlc   # → drivers/local_train_feast_dlc.py
```

## BYOC vs. DLC — which to use

**BYOC** (Bring Your Own Container) — `local_train.py`:
- You build the image (`make image` → `Dockerfile` at repo root).
- You control every dep version. No ECR pull required for local runs
  (uses your local Docker tag).
- Best for: rapid iteration, custom system deps, or air-gapped work.

**DLC** (Deep Learning Container) — `local_train_dlc.py` /
`local_train_feast_dlc.py`:
- AWS-maintained image with pinned framework versions and security
  patches.
- Requires AWS creds for the initial ECR pull (image is publicly
  readable, ECR still wants an auth token).
- Best for: matching what a real cloud SageMaker job would run with.

Sagebaker leans BYOC for sandbox work (faster, no AWS account needed)
and falls back to DLC when the experiment needs to mirror a
production pipeline.

## Local Mode — what's actually happening

All three drivers use SageMaker Local Mode. That means:

- `instance_type="local"` instead of an EC2 type.
- Training runs in a Docker container *on your laptop*.
- The SageMaker SDK still mounts data, runs the entry point, and
  tarballs the output exactly as it would in the cloud.
- Nothing actually goes to AWS — except (for DLC drivers) the
  ECR image pull.

This is what makes the BYOC/DLC paths interchangeable with cloud
SageMaker: the entry point doesn't know the difference. To run in the
cloud, swap `LocalSession()` for `Session()` and the `file://` paths
for `s3://`.

## Train/serve skew gotcha

`local_train_dlc.py` sets `source_dir="src"`, which uploads the
contents of `src/` to the container as the source bundle. SageMaker
then auto-`pip install`s anything inside `source_dir` named
`requirements.txt`. **Don't put a `requirements.txt` in `src/`** —
it'll upgrade numpy and binary-incompatibilize the DLC's pinned
sklearn/pandas. See the README's "Beyond pickle" / "Training/serving
skew" sections for the full story.

## Adding a new driver

If you need a torch DLC or a HuggingFace DLC, copy `local_train_dlc.py`
and swap the framework class:

```python
from sagemaker.pytorch import PyTorch                 # torch DLC
from sagemaker.huggingface import HuggingFace         # HF DLC

estimator = HuggingFace(
    entry_point="train_llm.py",
    source_dir="src",
    transformers_version="4.x",
    pytorch_version="2.x",
    py_version="py310",
    instance_type="local_gpu",   # or "ml.g5.2xlarge" for cloud
    ...
)
```

The `entry_point` script lives in `src/`. Same `model_fn(model_dir)`
loader contract, same bundle layout — only the container image and
the framework class change.

## See also

- [README.md](../README.md) — sections "BYOC (offline)", "DLC (with
  AWS credentials)", and "Productionizing on SageMaker" for the
  deeper walkthrough
- [`src/train.py`](../src/train.py) — what runs *inside* the
  container; the universal part
- [`Dockerfile`](../Dockerfile) — the BYOC image
- [`pipeline.py`](../pipeline.py) — production SageMaker Pipeline
  sketch (cloud, untested) — the natural follow-on to these drivers
