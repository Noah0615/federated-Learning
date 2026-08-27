"""FedAvg with server-only challenge deltas and post-hoc telemetry rows."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from challenge.recorders import append_rounds, dump_private_delta, log_json


def _flatten(arrays: Iterable[np.ndarray]) -> np.ndarray:
    values = [np.asarray(array, dtype=np.float64).reshape(-1) for array in arrays]
    return np.concatenate(values) if values else np.empty(0, dtype=np.float64)


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return None
    return float(np.dot(left, right) / denominator)


class DeltaFedAvg(FedAvg):
    """FedAvg that records update/challenge similarity without exposing delta."""

    def __init__(
        self,
        *,
        sigma: float,
        seed: int,
        run_id: str,
        node_roles: dict[str, str],
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.sigma = sigma
        self.seed = seed
        self.run_id = run_id
        self.node_roles = node_roles
        self._global_before: list[np.ndarray] | None = None

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        self._global_before = [np.array(value, copy=True) for value in arrays.to_numpy_ndarrays()]
        config = ConfigRecord(dict(config))
        config["server-round"] = server_round
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        replies = list(replies)
        delta: list[np.ndarray] = []
        if self._global_before is not None:
            rng = np.random.default_rng(self.seed ^ server_round)
            delta = [
                rng.normal(0.0, self.sigma, size=value.shape).astype(value.dtype)
                for value in self._global_before
            ]
            if self.sigma > 0.0:
                dump_private_delta(self.run_id, server_round, delta)

        rows: list[dict[str, object]] = []
        reference = _flatten(delta)
        before = _flatten(self._global_before or [])
        for reply in replies:
            arrays = reply.content.get("arrays")
            if arrays is None or self._global_before is None:
                continue
            update = _flatten(arrays.to_numpy_ndarrays()) - before
            metadata = reply.content.get("metadata")
            identity = str(metadata.get("node-identity", "unknown")) if metadata else "unknown"
            metrics = reply.content.get("metrics")
            fit_wallclock = float(metrics.get("fit_wallclock_s", 0.0)) if metrics else 0.0
            num_examples = int(metrics.get("num-examples", 0)) if metrics else 0
            rows.append(
                {
                    "run_id": self.run_id,
                    "round": server_round,
                    "node_identity": identity,
                    "role": self.node_roles.get(identity, "unknown"),
                    "fit_wallclock_s": fit_wallclock,
                    "cos_u_delta": _cosine(update, reference) if self.sigma > 0.0 else None,
                    "num_examples": num_examples,
                }
            )
        if rows:
            append_rounds(self.run_id, rows)
            log_json(
                {
                    "round_recorded": True,
                    "run_id": self.run_id,
                    "round": server_round,
                    "rows": len(rows),
                }
            )

        aggregated, metrics = super().aggregate_train(server_round, replies)
        if aggregated is not None and self.sigma > 0.0:
            aggregated = ArrayRecord(
                [value + noise for value, noise in zip(aggregated.to_numpy_ndarrays(), delta)]
            )
        return aggregated, metrics

