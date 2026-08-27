
from __future__ import annotations

import json
import logging
import os
import time

import torch
from flwr.clientapp import ClientApp
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)

from attackers.precompute import make_update
from challenge.task import Net, evaluate_model, load_data, train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = ClientApp()


def _identity(context: Context) -> str:
    return os.environ.get(
        "NODE_IDENTITY", str(context.node_config.get("node-identity", context.node_id))
    )


def _partition(context: Context) -> tuple[int, int]:
    return (
        int(context.node_config.get("partition-id", 0)),
        int(context.node_config.get("num-partitions", 5)),
    )


@app.train()
def train(msg: Message, context: Context) -> Message:
    started = time.perf_counter()
    identity = _identity(context)
    round_number = int(msg.content["config"].get("server-round", 0))
    partition_id, num_partitions = _partition(context)
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mode = os.environ.get("ATTACK_MODE", "honest")

    if mode == "precompute":
        model = make_update(model, round_number, seed=91011)
        train_loss = 0.0
        num_examples = 0
    else:
        loader, _ = load_data(
            partition_id,
            num_partitions,
            int(context.run_config["batch-size"]),
            int(context.run_config["mnist-max-samples"]),
        )
        train_loss = train_model(
            model,
            loader,
            int(context.run_config["local-epochs"]),
            float(msg.content["config"].get("lr", context.run_config["learning-rate"])),
            device,
        )
        num_examples = len(loader.dataset)

    elapsed = time.perf_counter() - started
    event = {
        "node_assign_event": True,
        "env_identity": identity,
        "flower_node_id": context.node_id,
        "role": "evil_precompute" if mode == "precompute" else "honest",
        "server_round": round_number,
    }
    logger.info(json.dumps(event, sort_keys=True))
    metrics = MetricRecord(
        {
            "train_loss": float(train_loss),
            "num-examples": int(num_examples),
            "fit_wallclock_s": float(elapsed),
        }
    )
    metadata = ConfigRecord({"node-identity": identity, "role": event["role"]})
    return Message(
        content=RecordDict(
            {"arrays": ArrayRecord(model.state_dict()), "metrics": metrics, "metadata": metadata}
        ),
        reply_to=msg,
    )


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    partition_id, num_partitions = _partition(context)
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, loader = load_data(
        partition_id,
        num_partitions,
        int(context.run_config["batch-size"]),
        int(context.run_config["mnist-max-samples"]),
    )
    loss, accuracy = evaluate_model(model, loader, device)
    return Message(
        content=RecordDict(
            {
                "metrics": MetricRecord(
                    {
                        "eval_loss": float(loss),
                        "eval_acc": float(accuracy),
                        "num-examples": len(loader.dataset),
                    }
                )
            }
        ),
        reply_to=msg,
    )

