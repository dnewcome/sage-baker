"""Shared pytest fixtures.

Tests run from the repo root. The `src/` directory is added to sys.path
so `import train`, `import bundle`, `import plugins` all work the same
way they do in the SageMaker container (where `source_dir="src"` puts
that directory on the path).
"""
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import fetch_california_housing


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(scope="session")
def housing_df() -> pd.DataFrame:
    """Small slice of California housing — enough to fit a tiny model fast."""
    bunch = fetch_california_housing(as_frame=True)
    df = bunch.frame.rename(columns={"MedHouseVal": "target"})
    return df.sample(n=200, random_state=42).reset_index(drop=True)


@pytest.fixture
def tmp_train_dir(tmp_path: Path, housing_df: pd.DataFrame) -> Path:
    """A directory laid out the way `src/train.py` expects: holds one .parquet."""
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    housing_df.to_parquet(train_dir / "training.parquet", index=False)
    return train_dir


@pytest.fixture
def tmp_model_dir(tmp_path: Path) -> Path:
    out = tmp_path / "model"
    out.mkdir()
    return out


@pytest.fixture(autouse=True)
def silence_mlflow(monkeypatch):
    """Make sure tests never write to a real MLflow tracking server."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_REGISTERED_MODEL", raising=False)
