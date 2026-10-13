"""Shared fixtures/helpers for minispeed tests."""

from __future__ import annotations

import os
import socket

import torch
import torch.nn as nn
import torch.nn.functional as F


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE computed in fp32 — the standard way to keep the loss well scaled
    when the model itself runs in fp16/bf16."""
    return F.mse_loss(pred.float(), target.float())


class MLP(nn.Module):
    def __init__(self, in_dim: int = 8, hidden: int = 32, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(seed: int = 1234, device: str = "cpu") -> MLP:
    torch.manual_seed(seed)
    return MLP().to(device)


def make_batch(
    steps: int,
    gas: int,
    world: int,
    micro: int,
    seed: int = 7,
    device: str = "cpu",
) -> torch.Tensor:
    """Returns (x, y) shaped [steps, gas, world * micro, dim].

    Layout matters for the distributed tests: rank ``r`` trains on
    ``[s, m, r*micro:(r+1)*micro]`` while the single-process reference trains
    on ``[s, m, :]`` — the union of all ranks' shards. With equal shard sizes
    the averaged DP gradient equals the full-batch gradient, so losses must
    match.
    """
    g = torch.Generator().manual_seed(seed)
    n = steps * gas * world * micro
    x = torch.randn(n, 8, generator=g)
    y = torch.randn(n, 1, generator=g)
    x = x.view(steps, gas, world * micro, 8)
    y = y.view(steps, gas, world * micro, 1)
    return x.to(device), y.to(device)


def base_config(
    micro: int = 4,
    gas: int = 2,
    lr: float = 1e-3,
    wd: float = 0.01,
    clip: float = 1.0,
    dtype: str = "fp32",
    zero_stage: int = 0,
    warmup_steps: int = 0,
    total_steps: int | None = None,
    scheduler: str = "WarmupLR",
    loss_scale_power: int = 8,
    scale_window: int = 100,
) -> dict:
    sched_params = {
        "warmup_min_lr": 0.0,
        "warmup_max_lr": lr,
        "warmup_num_steps": warmup_steps,
    }
    if total_steps is not None:
        sched_params["total_num_steps"] = total_steps
    cfg = {
        "train_micro_batch_size_per_gpu": micro,
        "gradient_accumulation_steps": gas,
        "gradient_clipping": clip,
        "dtype": dtype,
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": lr, "betas": [0.9, 0.999], "eps": 1e-8,
                       "weight_decay": wd},
        },
        "scheduler": {"type": scheduler, "params": sched_params},
        "zero_optimization": {"stage": zero_stage},
        "steps_per_print": 1000,
    }
    if dtype == "fp16":
        cfg["fp16"] = {
            "enabled": True,
            "loss_scale": 0,
            "initial_scale_power": loss_scale_power,
            "loss_scale_window": scale_window,
        }
    elif dtype == "bf16":
        cfg["bf16"] = {"enabled": True}
    return cfg


def train_engine(engine, x, y, steps: int, gas: int, micro: int, device):
    """Standard loop: forward/backward per micro-batch, step() per batch.

    ``x``/``y`` are shaped [steps, gas, N, dim]; returns per-micro-batch losses.
    """
    losses = []
    engine.train()
    for s in range(steps):
        for m in range(gas):
            xb = x[s, m].to(device)
            yb = y[s, m].to(device)
            loss = mse_loss(engine(xb), yb)
            losses.append(float(loss.detach()))
            engine.backward(loss)
            engine.step()
    return losses


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def env_without(*keys: str) -> dict:
    """Snapshot of os.environ without the given keys (for spawn workers)."""
    return {k: v for k, v in os.environ.items() if k not in keys}
