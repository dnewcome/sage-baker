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


def resolve_model_dir(model_dir: str) -> str:
    """Resolve ``model_dir`` to a local filesystem path.

    If ``model_dir`` is a local path it is returned unchanged.

    If it is an S3 URI (``s3://bucket/prefix``), all objects under that
    prefix are downloaded to a temporary directory and its path is returned.
    Two layouts are supported:

    Sage-baker bundle (multiple files at prefix)
        s3://bucket/models/fillrate/run-20260510/
        → downloads config.json, model.joblib, metadata.json, …

    Legacy single-file model (production pkl artifact)
        s3://bucket/models/cc_product_recommender/v1/model_run-123
        s3://bucket/models/cc_product_recommender/v1/model_run-123.pkl
        → downloads the single pkl; writes a minimal config.json so
          load_bundle() can locate the weights file by name.

    The temporary directory is created with ``tempfile.mkdtemp`` and
    lives for the lifetime of the process — no cleanup is performed.
    """
    if not model_dir.startswith("s3://"):
        return model_dir

    import tempfile
    import boto3

    without_scheme = model_dir[5:]
    bucket, _, prefix = without_scheme.partition("/")
    s3 = boto3.client("s3")

    # List all objects at the prefix (handles bundle case and exact-key case).
    paginator = s3.get_paginator("list_objects_v2")
    objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects.extend(page.get("Contents", []))

    # Production models may be referenced without the .pkl extension.
    if not objects:
        for ext in (".pkl", ".joblib"):
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix + ext)
            if resp.get("Contents"):
                objects = resp["Contents"]
                break

    if not objects:
        raise FileNotFoundError(f"No S3 objects found at {model_dir}")

    tmp_dir = tempfile.mkdtemp(prefix="sage-baker-bundle-")

    for obj in objects:
        key = obj["Key"]
        # Preserve the relative path within the prefix so bundle/ layouts
        # land in the right places.  A bare-key object (prefix IS the key)
        # is saved by its basename.
        rel = key[len(prefix):].lstrip("/") or os.path.basename(key)
        local_path = os.path.join(tmp_dir, rel)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket, key, local_path)

    # Single-file bundle: synthesize a config.json so load_bundle() finds
    # the weights without knowing the timestamped filename in advance.
    local_files = os.listdir(tmp_dir)
    if len(local_files) == 1 and CONFIG_FILE not in local_files:
        with open(os.path.join(tmp_dir, CONFIG_FILE), "w") as f:
            json.dump({"weights_file": local_files[0]}, f)

    return tmp_dir


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None
