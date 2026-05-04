# sage-baker

Local SageMaker training sandbox using **SageMaker Local Mode** with a
**bring-your-own-container (BYOC)** image. Runs entirely on the local Docker
daemon — no AWS account, no ECR pulls, no cloud resources.

## What this is

SageMaker has a "Local Mode" where the SDK runs training/inference jobs in
Docker containers on your machine using the same `/opt/ml/...` directory
contracts as real SageMaker. By default it pulls AWS Deep Learning Container
(DLC) images from a regional ECR registry, which requires real AWS credentials.

This project skips ECR entirely by building a small local image and pointing
the SDK at it via `image_uri=`. The container follows the
"algorithm container" contract: a `train` command on `PATH` that reads its
inputs from `/opt/ml/input/` and writes the model to `/opt/ml/model/`.

## Repo layout

```
Dockerfile          minimal Python + scikit-learn image with a `train` command
train.py            training script — runs inside the container
prepare_data.py     writes data/iris.csv (toy dataset)
local_train.py      driver: builds an Estimator in Local Mode and calls fit()
local_serve.py      placeholder — does not work yet (see "Serving", below)
requirements.txt    sagemaker<3, boto3, scikit-learn, pandas, docker
```

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

```bash
docker build -t sage-baker-sklearn:latest .
.venv/bin/python prepare_data.py
.venv/bin/python local_train.py
```

Expected tail of output:

```
algo-1-XXXX  | validation_accuracy=1.0000
algo-1-XXXX exited with code 0
model artifact: file:///.../sage-baker/.sm-scratch/.../compressed_artifacts/model.tar.gz
```

The model artifact (`model.tar.gz` containing `model.joblib`) lives under
`.sm-scratch/`.

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
