"""Lineage propagates from prepare → train → bundle.

If a prepare-script writes data/lineage.json alongside the dataset,
src/train.py picks it up and embeds it under metadata.data_lineage.
This is the audit trail that lets you trace a deployed model back to
the source data + query.
"""
import json
import sys


def test_lineage_json_round_trips_into_metadata(tmp_train_dir, tmp_model_dir, monkeypatch):
    lineage = {
        "source": "test-fixture",
        "fetched_at": "2026-05-08T00:00:00Z",
        "dataset_sha256": "deadbeef" * 8,
        "dataset_n_rows": 200,
    }
    (tmp_train_dir / "lineage.json").write_text(json.dumps(lineage))

    import train
    monkeypatch.setattr(sys, "argv", [
        "train.py",
        "--train", str(tmp_train_dir),
        "--model-dir", str(tmp_model_dir),
        "--plugin", "housing",
    ])
    train.main()

    metadata = json.loads((tmp_model_dir / "metadata.json").read_text())
    assert "data_lineage" in metadata, "lineage.json was not embedded in metadata.json"
    assert metadata["data_lineage"] == lineage
