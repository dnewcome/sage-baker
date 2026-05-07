# sage-baker

Local SageMaker training sandbox using **SageMaker Local Mode**, with two
interchangeable paths: a **bring-your-own-container (BYOC)** image for fully
offline use, and the **AWS Deep Learning Container (DLC)** image for
production-parity workflows.

## What this is

SageMaker has a "Local Mode" where the SDK runs training/inference jobs in
Docker containers on your machine using the same `/opt/ml/...` directory
contracts as real SageMaker. By default it pulls Deep Learning Container
(DLC) images from a regional ECR registry, which requires real AWS credentials
(any account works — the images are public-read, but ECR still demands a real
auth token to issue the pull).

This project supports both:

- **BYOC** (`local_train.py`) — small local image, follows the algorithm
  container contract (a `train` command on `PATH` reading `/opt/ml/input/`
  and writing `/opt/ml/model/`). Fully offline. No AWS account needed.
- **DLC** (`local_train_dlc.py`) — official AWS scikit-learn DLC image.
  Requires real AWS creds for the initial ECR pull; runs locally after that.
  Uses the `entry_point` + `source_dir` flow, so edits to `train.py` don't
  require any rebuild.

## Repo layout

```
Dockerfile          minimal Python + scikit-learn image with a `train` command (BYOC)
src/bundle.py       generic helpers for the standard model-bundle layout
src/train.py        sklearn training script — example of using bundle.py
prepare_data.py     writes data/iris.csv (toy multiclass dataset)
prepare_sonar.py    writes data/sonar.csv (Sonar Rocks vs Mines, binary)
local_train.py      BYOC driver — uses the local image, no AWS account
local_train_dlc.py  DLC driver  — uses the AWS scikit-learn DLC image
local_serve.py      placeholder — does not work yet (see "Serving", below)
requirements.txt    sagemaker<3, boto3, scikit-learn, pandas, docker
```

The training script lives in `src/` so the DLC's `source_dir` can point at a
clean directory containing only training code. SageMaker auto-`pip install`s
any `requirements.txt` inside `source_dir`; if the project's outer
`requirements.txt` (sagemaker, boto3, etc.) leaks into the container it
upgrades numpy and binary-incompatibilizes the pre-installed sklearn/pandas.
Don't put a `requirements.txt` inside `src/` unless you genuinely need extra
deps in the training container.

The training script follows SageMaker conventions:

| Path                                          | Purpose                              |
| --------------------------------------------- | ------------------------------------ |
| `/opt/ml/input/data/<channel>/`               | Input data (mounted per channel)     |
| `/opt/ml/input/config/hyperparameters.json`   | Hyperparameters (string-typed)       |
| `/opt/ml/model/`                              | Where the model is written           |
| `/opt/ml/output/`                             | Where failure outputs go             |

## Setup

Requirements: Docker, Python 3.10+, ~1 GB free disk.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Running training

Generate a dataset (pick one — each script wipes `data/` and writes one CSV):

```bash
.venv/bin/python prepare_data.py    # iris (3-class, 150 rows)
# or
.venv/bin/python prepare_sonar.py   # Rocks vs Mines (binary, 208 rows)
```

`train.py` reads whichever CSV is in the train channel — no code changes
needed to swap datasets, as long as the CSV has a `target` column.

### BYOC (offline)

```bash
docker build -t sage-baker-sklearn:latest .
.venv/bin/python local_train.py
```

### DLC (with AWS credentials)

AWS now recommends IAM Identity Center (SSO) with short-lived credentials
over long-lived access keys. One-time setup:

```bash
aws configure sso                 # creates a profile entry in ~/.aws/config
```

Then before each session:

```bash
aws sso login --profile your-profile
export AWS_PROFILE=your-profile
.venv/bin/python local_train_dlc.py
```

(Long-lived `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` env vars still work
if you need them.)

The DLC image (~3 GB) is pulled once and cached in your local Docker daemon;
subsequent runs are offline.

Both paths produce the same shape of output:

```
algo-1-XXXX  | validation_accuracy=1.0000
algo-1-XXXX exited with code 0
model artifact: file:///.../sage-baker/.sm-scratch/.../compressed_artifacts/model.tar.gz
```

The model artifact (`model.tar.gz` containing `model.joblib`) lives under
`.sm-scratch/`.

## When to use which

| Use BYOC when …                              | Use DLC when …                              |
| -------------------------------------------- | ------------------------------------------- |
| no AWS credentials available                 | you have any AWS account                    |
| you want a small (~200 MB) image             | image size doesn't matter                   |
| deps are simple (sklearn / pandas / etc.)    | you want AWS-tested framework + GPU stack   |
| training script is stable                    | you want to iterate on `train.py` fast      |
| serving doesn't matter                       | you want `/ping` + `/invocations` for free  |

The big practical wins of DLC are the `entry_point` flow (no rebuilds on
script edits) and the inference toolkit (working serving in one `.deploy()`
call). BYOC's wins are zero AWS dependency and a small, fully-controlled
image.

## Architecture: separating code from weights

The single most important design decision in a training system is **what
gets persisted with the model and what stays in code**. Pickle (and
`torch.save(model, ...)`, and the default `mlflow.<flavor>.log_model`)
freezes the running Python object — including a reference to the class
that defined it. When the class moves, gets renamed, or has a method
added, unpickling either explodes or silently loads the wrong thing. This
is the trap where every code change forces a retrain.

The fix is a layered model: **code in git, weights as data, config as JSON.**

### Bundle layout

The training script writes a directory with this shape, regardless of
framework:

```
model/
├── config.json           how to build the model (arch, hyperparams,
│                         weights_file pointer, feature schema)
├── <weights_file>        the actual numbers (model.joblib, model.safetensors, …)
├── preprocessor.json     [optional] preprocessor state (scaler stats,
│                         label maps, vocab refs)
└── metadata.json         provenance: timestamp, python version, git sha,
                          training metrics
```

`config.json` knows the name of the weights file. The loader reads the
config, instantiates the model class with those args, and loads weights
from the file the config points at. The class definition lives in your
repo — versioned, debuggable, hot-editable. Want to fix a bug in
`forward()`? Edit, reload weights, done. No retrain.

### Format choices

| Thing                          | Format                                       |
| ------------------------------ | -------------------------------------------- |
| Torch / JAX / TF tensors       | **`safetensors`** (mmap, no pickle, no RCE)  |
| Embeddings (raw arrays)        | safetensors or `.npz`                        |
| Tokenizers, HF models          | `tokenizer.save_pretrained(dir)`             |
| HF model end-to-end            | `model.save_pretrained(dir)` — already does the layout above |
| Hyperparameters / model config | JSON                                         |
| sklearn pipelines              | `joblib` (pickle) is least-bad — pin `scikit-learn==X.Y` and accept it. For *important* models, extract coefficients / tree structures into JSON manually. |

`safetensors` is the boring-good default for tensor data: zero-copy mmap,
no arbitrary code execution on load, supported by torch / JAX / TF / HF.

### How this maps to other systems

- **SageMaker.** Whatever you put in `/opt/ml/model/` gets tarred to
  `model.tar.gz`. Write the bundle there during training; the inference
  container's `model_fn(model_dir)` calls your `load(model_dir)` to
  reassemble. Same function, two consumers.
- **MLflow.** Two clean paths. Either treat MLflow as tracking only and
  use `mlflow.log_artifacts("model/")` to log the bundle as opaque files —
  load via your own `load(dir)`, never `mlflow.X.load_model`. Or wrap
  `load(dir)` in a custom `mlflow.pyfunc.PythonModel` so
  `mlflow.pyfunc.load_model(uri)` does the right thing. The point is to
  never let MLflow pickle your class.
- **Plain `pickle` / `torch.save(model, ...)` / `mlflow.X.log_model`.**
  These are exactly what we're avoiding. They can't survive code changes
  and they're a remote-code-execution hazard on load.

### What's in this repo

- `src/bundle.py` — generic JSON helpers (`save_config`, `load_config`,
  `save_metadata`, `load_metadata`). Framework-agnostic.
- `src/train.py` — concrete sklearn example. Trains a `RandomForest`,
  writes the bundle layout via `bundle.py`, and exposes `model_fn(model_dir)`
  that reads the bundle back. The same `model_fn` is what SageMaker's
  inference container calls.

To add a new framework, drop a parallel trainer (e.g.
`src/train_torch.py`) that calls the same `bundle.save_config(...)` /
`bundle.save_metadata(...)` and writes its weights with `safetensors`
instead of `joblib`. The bundle layout is the contract; what each
framework writes inside it is its own business.

## Hyperparameters

`local_train.py` passes `n-estimators` and `max-depth` to the estimator;
SageMaker writes these to `/opt/ml/input/config/hyperparameters.json` inside
the container. `train.py` reads that file and applies them as `argparse`
defaults. SageMaker stringifies all hyperparameters, so cast to the type you
want when reading.

## Gotchas

A few things that bit us; worth knowing if you adapt this to a different
framework or environment.

- **SageMaker SDK v3 removed `sagemaker.local`.** Pin to `sagemaker<3` until
  Local Mode lands in v3 (or it stays gone — TBD).
- **Snap-installed Docker is confined.** It can't bind-mount paths under
  `/tmp`, which is where the SDK normally drops its `docker-compose.yaml` and
  per-job scratch dirs. `local_train.py` works around this by setting
  `TMPDIR` and `local.container_root` to `.sm-scratch/` under the project.
  If your Docker is from `docker.io` / `docker-ce` (apt) you can drop that.
- **Real AWS credentials are NOT required for BYOC**, but boto3 still needs
  *something* in its credential chain plus a region — `local_train.py` sets
  dummy values via env vars before constructing `LocalSession`.
- **`role=` is ignored in Local Mode** but the SDK still validates it as a
  string. Any ARN-shaped string works.
- **`image_uri=` bypasses the framework's ECR lookup.** As long as the
  reference has no registry prefix and the image is present locally, Docker
  will use it without trying to pull.

## Customizing for your model

To swap in your own training:

1. Replace the body of `train.py` with your training code, keeping:
   - reads from `args.train` (a directory of input files)
   - writes a model file to `args.model_dir`
2. Update `Dockerfile` deps if you need PyTorch / TF / etc.
3. Update `local_train.py` hyperparameters and the `inputs={...}` dict passed
   to `fit()` (one key per channel).

For larger projects you probably want `entry_point` + `source_dir` so you can
edit the script without rebuilding the image. That requires installing the
`sagemaker-training` toolkit in the image (the package that interprets the
`sagemaker_program` / `sagemaker_submit_directory` hyperparameters and runs
your script). It's heavier and its install can be finicky on slim Python
images, which is why this scaffold bakes the script into the image instead.

## Serving (not implemented)

`local_serve.py` is left over from an earlier attempt that used the AWS
scikit-learn DLC. It will not work against this BYOC image because the image
has no `serve` command.

To add serving, the image needs a `serve` command on `PATH` that starts an
HTTP server on port 8080 with two routes:

- `GET /ping` → `200` when ready
- `POST /invocations` → consumes the request body, returns predictions

Flask + gunicorn is the usual minimal setup. Then `local_serve.py` can use
`sagemaker.model.Model(image_uri="sage-baker-sklearn:latest", ...)` and
`.deploy(instance_type="local")` to spin up a local endpoint.

## Cleaning up

`.sm-scratch/` accumulates artifacts and per-job dirs. Safe to delete between
runs:

```bash
rm -rf .sm-scratch
```

Containers and the `sagemaker-local` Docker network are torn down by the SDK
at the end of each `fit()`.
