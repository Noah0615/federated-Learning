"""Flower ServerApp entry point for M0 and the server-only challenge."""

from __future__ import annotations

import os

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from challenge.recorders import new_run_id, write_manifest
from challenge.strategies import DeltaFedAvg
from challenge.task import Net

app = ServerApp()


def _node_roles() -> dict[str, str]:
    raw = os.environ.get("NODE_ROLE_MAP", "")
    result: dict[str, str] = {}
    for item in raw.split(","):
        if ":" in item:
            identity, role = item.split(":", 1)
            result[identity.strip()] = role.strip()
    return result


@app.main()
def main(grid: Grid, context: Context) -> None:
    run_id = new_run_id()
    node_roles = _node_roles()
    write_manifest(run_id, node_roles)
    model = Net()
    strategy = DeltaFedAvg(
        sigma=float(context.run_config["delta-sigma"]),
        seed=int(context.run_config["delta-seed"]),
        run_id=run_id,
        node_roles=node_roles,
        fraction_train=float(context.run_config["fraction-train"]),
        fraction_evaluate=float(context.run_config["fraction-evaluate"]),
        min_train_nodes=int(context.run_config["min-available-nodes"]),
        min_evaluate_nodes=int(context.run_config["min-available-nodes"]),
        min_available_nodes=int(context.run_config["min-available-nodes"]),
    )
    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(model.state_dict()),
        train_config=ConfigRecord({"lr": float(context.run_config["learning-rate"])}),
        num_rounds=int(context.run_config["num-server-rounds"]),
    )
    output = os.path.join(os.environ.get("RESULTS_DIR", "/results"), run_id, "final_model.pt")
    import torch

    torch.save(result.arrays.to_torch_state_dict(), output)

