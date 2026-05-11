"""Generic helpers for the standard model-bundle layout.

A trained model is written to a directory with this shape:

    model/
    ├── config.json           how to build the model (arch, hyperparams,
    │                         weights_file pointer, feature names, etc.)
    ├── <weights file>        the actual numbers (e.g. model.joblib for
    │                         sklearn, model.safetensors for torch)
    ├── preprocessor.json     [optional] preprocessor state (scaler stats,
    │                         label maps, vocab refs)
    └── metadata.json         training-run provenance (timestamp, python
                              version, git sha, validation metrics)

This module knows about the JSON files only. The framework-specific trainer
writes whatever weights file it wants and records its name in `config.json`
under `weights_file`. The corresponding loader reads `config.json` to learn
both *how* to instantiate the model and *which* file to load weights from.

Keeping these helpers framework-agnostic means a sklearn trainer, a torch
trainer, and a HF trainer can all share the same on-disk contract — and the
same `load(model_dir) -> model` shape on the inference side.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone


CONFIG_FILE = "config.json"
METADATA_FILE = "metadata.json"
PREPROCESSOR_FILE = "preprocessor.json"


def save_config(model_dir, config):
    """Write the model config — required for every bundle."""
    with open(os.path.join(model_dir, CONFIG_FILE), "w") as f:
        json.dump(config, f, indent=2, default=str)


def load_config(model_dir):
    with open(os.path.join(model_dir, CONFIG_FILE)) as f:
        return json.load(f)


def save_preprocessor(model_dir, state):
    """Write preprocessor state if the trainer has any (scaler params, etc.)."""
    with open(os.path.join(model_dir, PREPROCESSOR_FILE), "w") as f:
        json.dump(state, f, indent=2, default=str)


def load_preprocessor(model_dir):
    path = os.path.join(model_dir, PREPROCESSOR_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_metadata(model_dir, extras=None):
    """Write reproducibility info. `extras` merges in (e.g. metrics)."""
    meta = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "git_sha": _git_sha(),
    }
    if extras:
        meta.update(extras)
    with open(os.path.join(model_dir, METADATA_FILE), "w") as f:
        json.dump(meta, f, indent=2, default=str)


def load_metadata(model_dir):
    path = os.path.join(model_dir, METADATA_FILE)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_lineage(data_dir):
    """Read a lineage.json sidecar written by a prepare-* script.

    Trainers call this to pick up dataset provenance (source, query,
    snapshot timestamp, sha256) and merge it into the model bundle's
    metadata. None if the prepare script didn't write one.
    """
    path = os.path.join(data_dir, "lineage.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _stream_to_file(s3_response, local_path: str, chunk_size: int = 8 * 1024 * 1024) -> None:
    """Stream an S3 GetObject response to disk with a stderr progress line."""
    total = int(s3_response.get("ContentLength", 0))
    filename = os.path.basename(local_path)
    done = 0
    with open(local_path, "wb") as f:
        for chunk in s3_response["Body"].iter_chunks(chunk_size=chunk_size):
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = done / total * 100
                print(
                    f"\r  {filename}: {done / 1e6:.0f} / {total / 1e6:.0f} MB ({pct:.0f}%)",
                    end="", flush=True, file=sys.stderr,
                )
    if total:
        print(f"\r  {filename}: {total / 1e6:.0f} MB — done.           ", file=sys.stderr)


def resolve_model_dir(model_dir: str) -> str:
    """Resolve ``model_dir`` to a local filesystem path.

    If ``model_dir`` is a local path it is returned unchanged.

    If it is an S3 URI (``s3://bucket/key``), the model is downloaded and
    its local directory path is returned.  Two layouts are supported:

    Single-file model (production pkl artifact) — only ``s3:GetObject`` needed
        s3://bucket/models/cc_product_recommender/v1/model_run-123
        s3://bucket/models/cc_product_recommender/v1/model_run-123.pkl
        → tries the key as-is, then with .pkl/.joblib appended;
          writes a minimal config.json so load_bundle() finds the file.

    Sage-baker bundle (multiple files at a prefix) — requires ``s3:ListBucket``
        s3://bucket/models/fillrate/run-20260510/
        → lists the prefix and downloads config.json, model.joblib, etc.

    Caching
    -------
    Downloaded files are cached under ``MODEL_CACHE_DIR`` (env var, default
    ``~/.cache/sage-baker/models``).  The cache key is the S3 URI path so
    a model is only downloaded once per machine.  Since production artifact
    filenames include a run ID or timestamp, a new model deployment
    automatically gets a fresh cache entry — no manual invalidation needed.
    Set ``MODEL_CACHE_DIR=`` (empty) to disable caching and always use a
    temporary directory.
    """
    if not model_dir.startswith("s3://"):
        return model_dir

    import tempfile
    import boto3
    from botocore.exceptions import ClientError

    without_scheme = model_dir[5:]
    bucket, _, key = without_scheme.partition("/")
    s3 = boto3.client("s3")

    # --- Resolve cache directory ------------------------------------------
    cache_root = os.environ.get(
        "MODEL_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "sage-baker", "models"),
    )
    if cache_root:
        bundle_dir = os.path.join(cache_root, bucket, key.rstrip("/"))
        os.makedirs(bundle_dir, exist_ok=True)
    else:
        bundle_dir = tempfile.mkdtemp(prefix="sage-baker-bundle-")

    # --- Try direct GetObject first (only s3:GetObject required) ----------
    # download_file() calls HeadObject internally; get_object() does not.
    for candidate in [key, key + ".pkl", key + ".joblib"]:
        filename = os.path.basename(candidate)
        local_path = os.path.join(bundle_dir, filename)
        if os.path.exists(local_path):
            print(f"  {filename}: cached, skipping download.", file=sys.stderr)
            if not os.path.exists(os.path.join(bundle_dir, CONFIG_FILE)):
                with open(os.path.join(bundle_dir, CONFIG_FILE), "w") as f:
                    json.dump({"weights_file": filename}, f)
            return bundle_dir
        try:
            resp = s3.get_object(Bucket=bucket, Key=candidate)
            _stream_to_file(resp, local_path)
            with open(os.path.join(bundle_dir, CONFIG_FILE), "w") as f:
                json.dump({"weights_file": filename}, f)
            return bundle_dir
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code not in ("NoSuchKey", "404"):
                raise  # real permission error — surface immediately
            continue

    # --- Fall back to prefix listing for multi-file bundles ---------------
    paginator = s3.get_paginator("list_objects_v2")
    objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=key.rstrip("/") + "/"):
        objects.extend(page.get("Contents", []))

    if not objects:
        raise FileNotFoundError(
            f"No S3 objects found at {model_dir} "
            "(tried direct GetObject and prefix listing)"
        )

    prefix = key.rstrip("/") + "/"
    for obj in objects:
        rel = obj["Key"][len(prefix):] or os.path.basename(obj["Key"])
        local_path = os.path.join(bundle_dir, rel)
        if os.path.exists(local_path):
            print(f"  {rel}: cached, skipping download.", file=sys.stderr)
            continue
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        resp = s3.get_object(Bucket=bucket, Key=obj["Key"])
        _stream_to_file(resp, local_path)

    return bundle_dir


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None
