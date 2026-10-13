"""Sharded checkpoint save/load (DeepSpeed-flavored layout).

save_dir/
  latest                          # text file containing the newest tag
  <tag>/                          # e.g. global_step120
    mp_rank_00_model_states.pt    # module state_dict (written by rank 0)
    zero_pp_rank<r>_mp_rank_00_optim_states.pt   # per-rank optimizer shard +
                                                 # fp32 master weights +
                                                 # scheduler / scaler state
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

from . import comm

MODEL_STATES = "mp_rank_00_model_states.pt"
LATEST = "latest"


def _optim_states_name(rank: int) -> str:
    return f"zero_pp_rank{rank}_mp_rank_00_optim_states.pt"


def _read_latest(save_dir: str) -> str:
    path = os.path.join(save_dir, LATEST)
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


@torch.no_grad()
def save_checkpoint(engine, save_dir: str, tag: Optional[str] = None) -> str:
    tag = tag or f"global_step{engine.global_steps}"
    ckpt_dir = os.path.join(save_dir, tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    # stage 3: params live as 1/N slices between steps — all-gather them so
    # rank 0 can write a full model state (no-op for stages 0-2).
    was_gathered = engine.optimizer.params_gathered
    engine.optimizer.gather_parameters()

    # Model architecture + weights: rank 0 only (identical on all ranks).
    if engine.rank == 0:
        torch.save(
            {"module": engine.module.state_dict(), "tag": tag},
            os.path.join(ckpt_dir, MODEL_STATES),
        )
        with open(os.path.join(save_dir, LATEST), "w", encoding="utf-8") as f:
            f.write(tag)

    # Per-rank optimizer shard + fp32 master buffers.
    state = {
        "optimizer": engine.optimizer.state_dict(),
        "fp32_groups": [g.data.clone() for g in engine.flat_groups],
        "lr_scheduler": engine.lr_scheduler.state_dict(),
        "loss_scaler": engine.loss_scaler.state_dict(),
        "global_steps": engine.global_steps,
        "micro_steps": engine.micro_steps,
        "grad_norm": engine.grad_norm,
        "rank": engine.rank,
        "world_size": engine.world_size,
    }
    torch.save(state, os.path.join(ckpt_dir, _optim_states_name(engine.rank)))
    comm.barrier()
    if not was_gathered:
        engine.optimizer.free_parameters()  # stage 3: back to shards
    return ckpt_dir


@torch.no_grad()
def load_checkpoint(
    engine, load_dir: str, tag: Optional[str] = None
) -> Tuple[str, dict]:
    tag = tag or _read_latest(load_dir)
    ckpt_dir = os.path.join(load_dir, tag)

    # stage 3: load_state_dict needs full-shaped params (no-op for stages 0-2)
    was_gathered = engine.optimizer.params_gathered
    engine.optimizer.gather_parameters()

    # 1) model weights (all ranks read the same file)
    model_path = os.path.join(ckpt_dir, MODEL_STATES)
    model_state = torch.load(model_path, map_location="cpu", weights_only=True)
    engine.module.load_state_dict(model_state["module"])

    # 2) this rank's optimizer shard + master weights
    optim_path = os.path.join(ckpt_dir, _optim_states_name(engine.rank))
    if not os.path.exists(optim_path):
        raise FileNotFoundError(
            f"Missing optimizer shard for rank {engine.rank}: {optim_path}"
        )
    state = torch.load(optim_path, map_location="cpu", weights_only=True)

    saved_world = int(state.get("world_size", engine.world_size))
    if saved_world != engine.world_size:
        raise RuntimeError(
            f"Checkpoint was saved with world_size={saved_world} but current "
            f"world_size={engine.world_size}; ZeRO shards do not transfer."
        )

    # 3) restore fp32 master buffers, then push them into the model params
    for g, buf in zip(engine.flat_groups, state["fp32_groups"]):
        g.data.copy_(buf)
        g.copy_flat_to_model()

    # 4) stage 3: rebuild the params from the restored masters and return the
    #    parameters to their pre-load state (freed, or gathered if they were)
    if engine.config.zero_stage >= 3:
        engine.optimizer.free_parameters()
        if was_gathered:
            engine.optimizer.gather_parameters()

    # 5) optimizer / scheduler / scaler / counters
    engine.optimizer.load_state_dict(state["optimizer"])
    engine.lr_scheduler.load_state_dict(state["lr_scheduler"])
    engine.loss_scaler.load_state_dict(state["loss_scaler"])
    engine.global_steps = int(state.get("global_steps", 0))
    engine.micro_steps = int(state.get("micro_steps", 0))
    engine.grad_norm = state.get("grad_norm")
    comm.barrier()
    return tag, state
