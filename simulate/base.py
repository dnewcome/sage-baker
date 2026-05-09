"""Base contracts for synthetic-data scenarios.

Each scenario subclasses Scenario and emits a SimulationResult with three
distinct artifacts:

  training.parquet      what the model sees: features + labels
  ground_truth.parquet  the simulator's hidden state — true user IDs,
                        true attribution edges, true cohort membership.
                        NEVER joined into training; saved separately so
                        evaluation can use it but features can't leak it.
  lineage.json          scenario name, seed, parameters, sha256 of the
                        training frame — same format prepare_*.py writes.

The training frame goes through the existing trainer/plugin/agent loop
unchanged. Ground truth is loaded only by evaluation code that knows it
exists (e.g. an attribution-comparison harness).
"""
import dataclasses
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd


@dataclasses.dataclass
class SimulationResult:
    training: pd.DataFrame
    ground_truth: pd.DataFrame
    lineage: dict[str, Any]

    def write(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.training.to_parquet(os.path.join(output_dir, "training.parquet"), index=False)
        self.ground_truth.to_parquet(os.path.join(output_dir, "ground_truth.parquet"), index=False)
        with open(os.path.join(output_dir, "lineage.json"), "w") as f:
            json.dump(self.lineage, f, indent=2, default=str)


class Scenario:
    """Subclass and register in simulate.__init__ (or drop in
    simulate/scenarios/private/ for auto-discovery)."""

    # Override in every subclass.
    name: str = "base"
    description: str = ""

    # Defaults the CLI exposes; subclasses override and the driver merges
    # CLI flags on top.
    default_params: dict[str, Any] = {}

    def generate(self, seed: int = 42, **params) -> SimulationResult:
        raise NotImplementedError

    # ---- helpers shared across scenarios ---------------------------

    @staticmethod
    def make_lineage(scenario_name: str, seed: int, params: dict, training: pd.DataFrame) -> dict:
        """Build the lineage block in the same shape prepare_*.py uses."""
        # pandas' object-hash isn't cryptographic but is deterministic and
        # cheap; good enough for a "did the data change between runs" check.
        sha = hashlib.sha256(
            pd.util.hash_pandas_object(training, index=False).values.tobytes()
        ).hexdigest()
        return {
            "source": "simulate",
            "scenario": scenario_name,
            "seed": seed,
            "params": params,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "dataset_sha256": sha,
            "dataset_n_rows": len(training),
            "feature_names": list(training.columns),
        }
