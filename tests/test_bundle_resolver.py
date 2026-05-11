"""bundle.resolve_model_dir() — the dicey bit of the serve harness.

S3 paths are mocked at the boto3 client boundary so tests stay
hermetic (no network, no moto dep).
"""
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import bundle


def test_local_path_passthrough(tmp_path):
    """A local path should be returned unchanged — no S3 work."""
    assert bundle.resolve_model_dir(str(tmp_path)) == str(tmp_path)
    # Trailing slash shouldn't matter.
    assert bundle.resolve_model_dir(str(tmp_path) + "/") == str(tmp_path) + "/"


def _stream(body: bytes):
    """Build a fake S3 GetObject response whose Body.iter_chunks yields bytes."""
    resp = {"ContentLength": len(body)}
    resp["Body"] = MagicMock()
    resp["Body"].iter_chunks = lambda chunk_size: iter([body])
    return resp


def test_s3_single_file_via_get_object(tmp_path, monkeypatch):
    """Single-file artifact: GetObject succeeds for the bare key, no listing needed."""
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "cache"))

    fake = MagicMock()
    fake.get_object.return_value = _stream(b"fake-weights")
    with patch("boto3.client", return_value=fake):
        out = bundle.resolve_model_dir("s3://my-bucket/models/single/model_run-123")

    assert Path(out, "model_run-123").read_bytes() == b"fake-weights"
    config = json.loads(Path(out, "config.json").read_text())
    assert config["weights_file"] == "model_run-123"

    # First get_object call should be the bare key, before any .pkl/.joblib fallback.
    first_call = fake.get_object.call_args_list[0]
    assert first_call.kwargs["Key"] == "models/single/model_run-123"


def test_s3_single_file_falls_back_to_pkl_extension(tmp_path, monkeypatch):
    """Production artifacts often live at <key>.pkl; resolver should retry once."""
    from botocore.exceptions import ClientError

    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "cache"))

    no_such_key = ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
    fake = MagicMock()
    fake.get_object.side_effect = [no_such_key, _stream(b"pkl-weights")]
    with patch("boto3.client", return_value=fake):
        out = bundle.resolve_model_dir("s3://b/models/x/run-1")

    config = json.loads(Path(out, "config.json").read_text())
    assert config["weights_file"] == "run-1.pkl"
    assert Path(out, "run-1.pkl").read_bytes() == b"pkl-weights"


def test_s3_prefix_listing_fallback(tmp_path, monkeypatch):
    """When no single-file key works, fall back to ListObjects + multi-file download."""
    from botocore.exceptions import ClientError

    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "cache"))

    no_such_key = ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
    fake = MagicMock()
    # All three direct GetObject attempts (bare, .pkl, .joblib) fail.
    fake.get_object.side_effect = [
        no_such_key, no_such_key, no_such_key,
        _stream(b"config-bytes"),
        _stream(b"weights-bytes"),
    ]
    paginator = MagicMock()
    paginator.paginate.return_value = [{
        "Contents": [
            {"Key": "models/bundle/config.json"},
            {"Key": "models/bundle/model.joblib"},
        ]
    }]
    fake.get_paginator.return_value = paginator

    with patch("boto3.client", return_value=fake):
        out = bundle.resolve_model_dir("s3://b/models/bundle")

    assert Path(out, "config.json").read_bytes() == b"config-bytes"
    assert Path(out, "model.joblib").read_bytes() == b"weights-bytes"


def test_s3_cache_hit_skips_download(tmp_path, monkeypatch):
    """Second call against the same URI hits the cache — no GetObject."""
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "cache"))

    fake = MagicMock()
    fake.get_object.return_value = _stream(b"weights")
    with patch("boto3.client", return_value=fake):
        out1 = bundle.resolve_model_dir("s3://b/cached/run-1")
        out2 = bundle.resolve_model_dir("s3://b/cached/run-1")

    assert out1 == out2
    # Only the first call downloaded; the second was a cache hit.
    assert fake.get_object.call_count == 1


def test_s3_atomic_rename_leaves_no_tmp(tmp_path, monkeypatch):
    """Successful download should not leave a partial .tmp file behind."""
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "cache"))

    fake = MagicMock()
    fake.get_object.return_value = _stream(b"weights")
    with patch("boto3.client", return_value=fake):
        out = bundle.resolve_model_dir("s3://b/single/run-1")

    leftover = list(Path(out).glob("*.tmp"))
    assert leftover == [], f"unexpected .tmp leftovers: {leftover}"
