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


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None
