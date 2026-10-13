"""Distributed communication helpers for minispeed.

All wrappers are no-ops when running in a single process (no process group),
so the exact same engine code path works for single-GPU and multi-GPU runs.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def _world_size_from_env() -> int:
    try:
        return int(os.environ.get("WORLD_SIZE", 1))
    except ValueError:
        return 1


def resolve_backend() -> str:
    """Pick a communication backend: env override > nccl > gloo.

    NCCL requires one distinct GPU per rank; when there are more ranks than
    GPUs (e.g. ``torchrun --nproc_per_node=2`` on a single-GPU laptop) we fall
    back to gloo, which happily moves CUDA tensors between local ranks.
    """
    backend = os.environ.get("MINISPEED_BACKEND")
    if backend:
        return backend
    if torch.cuda.is_available():
        if _world_size_from_env() <= torch.cuda.device_count():
            return "nccl"
        return "gloo"
    return "gloo"


def resolve_device() -> torch.device:
    """Pick the compute device: env override > cuda(local_rank) if available > cpu."""
    device_str = os.environ.get("MINISPEED_DEVICE")
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        # clamp: several ranks may share one physical GPU
        local_rank = min(get_local_rank(), torch.cuda.device_count() - 1)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_distributed() -> bool:
    """True only when collective ops are actually needed."""
    return (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )


def init_distributed() -> None:
    """Initialize the process group from torchrun/env:// variables.

    Does nothing when WORLD_SIZE/RANK are not present (plain single process),
    or when the process group already exists.
    """
    if dist.is_initialized():
        return
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return

    backend = resolve_backend()
    if backend == "nccl":
        dev = resolve_device()
        torch.cuda.set_device(dev if dev.type == "cuda" else 0)    # Single-rank groups still init so that collectives are valid no-ops.
    dist.init_process_group(backend=backend, init_method="env://")


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def all_reduce(tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
    """In-place sum/max/min all-reduce across ranks."""
    if is_distributed():
        reduce_op = {
            "sum": dist.ReduceOp.SUM,
            "max": dist.ReduceOp.MAX,
            "min": dist.ReduceOp.MIN,
        }[op]
        dist.all_reduce(tensor, op=reduce_op)
    return tensor


def all_reduce_flag(flag: bool) -> bool:
    """Global logical OR of a python bool (used for inf/nan detection)."""
    if not is_distributed():
        return flag
    t = torch.tensor([0 if flag else 1], dtype=torch.uint8, device=_flag_device())
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item() == 0)


def reduce_scatter(output: torch.Tensor, input_: torch.Tensor) -> torch.Tensor:
    """Reduce-scatter `input_` (world_size * output.numel()) into `output`."""
    if is_distributed():
        dist.reduce_scatter_tensor(output, input_)
    else:  # world_size == 1: plain copy
        output.copy_(input_)
    return output


def all_gather_flat(output: torch.Tensor, local_shard: torch.Tensor) -> torch.Tensor:
    """All-gather per-rank shards into one flat buffer ordered by rank."""
    if is_distributed():
        dist.all_gather_into_tensor(output, local_shard)
    else:
        output.copy_(local_shard)
    return output


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    if is_distributed():
        dist.broadcast(tensor, src=src)
    return tensor


def _flag_device() -> torch.device:
    # NCCL collectives must run on CUDA tensors.
    if resolve_backend() == "nccl" and torch.cuda.is_available():
        return torch.device("cuda", get_local_rank())
    return torch.device("cpu")