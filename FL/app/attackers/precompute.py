"""Minimal deterministic precompute attacker.

The attacker shares the exact ClientApp wire interface with honest clients. It
does not receive the server challenge vector; its update is generated from a
fixed seed and the current round so the experiment has a reproducible control.
"""

from __future__ import annotations

import torch

from challenge.task import Net


def make_update(model: Net, round_number: int, seed: int) -> Net:
    """Return a reproducible update independent of the server's challenge."""

    generator = torch.Generator(device="cpu").manual_seed(seed + round_number)
    with torch.no_grad():
        for parameter in model.parameters():
            noise = torch.randn(
                parameter.shape,
                generator=generator,
                device=parameter.device,
                dtype=parameter.dtype,
            )
            parameter.add_(0.002 * noise / max(1.0, noise.norm().item()))
    return model

