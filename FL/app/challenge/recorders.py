"""Run manifests and append-only round parquet recording."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def results_root() -> Path:
    root = Path(os.environ.get("RESULTS_DIR", "/results"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def new_run_id() -> str:
    import datetime as dt

    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def write_manifest(run_id: str, node_roles: dict[str, str]) -> Path:
    path = results_root() / run_id / "manifest.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "node_roles": node_roles,
        "ground_truth": "controlled deployment mapping; do not use as a detector feature",
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def append_rounds(run_id: str, rows: list[dict[str, Any]]) -> Path:
    path = results_root() / run_id / "rounds.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if path.exists():
        frame = pd.concat([pd.read_parquet(path), frame], ignore_index=True)
    frame.to_parquet(path, index=False)
    return path


def dump_private_delta(run_id: str, server_round: int, delta: list[np.ndarray]) -> Path:
    directory = results_root() / run_id / "private"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"delta_round{server_round:04d}.npz"
    np.savez_compressed(path, **{f"layer_{i}": value for i, value in enumerate(delta)})
    return path


def log_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)

