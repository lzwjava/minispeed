"""ZeRO stages 1 & 2 (plus stage 0 = plain data parallel) in one class.

Mirrors DeepSpeed's ``stage_1_and_2.py``: the difference between the stages is
only *how gradients are reduced* and *what is partitioned*:

============= =============================================================
stage 0       full grads all-reduced, full optimizer state on every rank
stage 1       full grads all-reduced, optimizer state partitioned, params
              all-gathered after every step
stage 2       gradients reduce-scattered (each rank owns 1/N of the grad
              vector), optimizer state partitioned, params all-gathered
============= =============================================================
"""

from __future__ import annotations

from typing import List

import torch

from .. import comm
from ..flat import FlatParamGroup
from ..optim.adam import FlatAdam


class ZeROOptimizer:
    def __init__(
        self,
        flat_groups: List[FlatParamGroup],
        option_groups: List[dict],
        zero_stage: int = 0,
    ):
        if zero_stage not in (0, 1, 2):
            raise ValueError(f"Unsupported ZeRO stage: {zero_stage}")
        self.flat_groups = flat_groups
        self.zero_stage = zero_stage
        self.world_size = comm.get_world_size()
        self.rank = comm.get_rank()
        self.adam = FlatAdam(flat_groups, option_groups, zero_stage=zero_stage)
        self.grad_norm: float | None = None
        # cached all-gather destination buffers (one per group)
        self._gather_bufs = [
            torch.empty_like(g.data) if (zero_stage >= 1 and self.world_size > 1)
            else None
            for g in flat_groups
        ]

    # ------------------------------------------------------- torch-ish surface
    @property
    def param_groups(self) -> List[dict]:
        return self.adam.param_groups

    @property
    def state(self) -> List[dict]:
        return self.adam.state

    def zero_grad(self) -> None:
        for g in self.flat_groups:
            g.zero_grad()

    # ------------------------------------------------------------ reduction
    @torch.no_grad()
    def reduce_gradients(self) -> None:
        """Turn local gradients into globally reduced gradients in-place.

        stage 0/1: all-reduce(SUM) + divide by world_size  -> full buffer
        stage 2:   reduce-scatter(SUM) + divide by world_size -> local shard
        """
        ws = self.world_size
        for g in self.flat_groups:
            if self.zero_stage >= 2:
                comm.reduce_scatter(g.grad_shard, g.grad)
                if ws > 1:
                    g.grad_shard.div_(ws)
            else:
                comm.all_reduce(g.grad)
                if ws > 1:
                    g.grad.div_(ws)

    # ----------------------------------------------------------- clipping
    @torch.no_grad()
    def clip_gradients(self, max_norm: float | None) -> float:
        """Global L2 gradient norm (across all ranks) + optional clipping.

        For stage 2 each rank only holds its gradient shard, so the local
        sum-of-squares is a partition of the global norm — one SUM reduce
        gives the exact global norm. For stage 0/1 gradients are already
        identical on every rank after the all-reduce.
        """
        local_sumsq = torch.zeros((), dtype=torch.float32, device=self._grad_device())
        for g in self.flat_groups:
            local_sumsq += g.local_grad_sumsq()

        if self.zero_stage >= 2 and self.world_size > 1:
            comm.all_reduce(local_sumsq, op="sum")

        norm = float(local_sumsq.sqrt().item())
        self.grad_norm = norm

        if max_norm is not None and norm > max_norm:
            clip_coef = max_norm / (norm + 1e-6)
            for g in self.flat_groups:
                g.scale_grads(clip_coef)
        return norm

    # --------------------------------------------------------------- step
    @torch.no_grad()
    def step(self) -> bool:
        """Update this rank's parameter shard, then sync the full buffers."""
        updated = self.adam.step()
        if self.zero_stage >= 1:
            self._gather_parameters()
        return updated

    @torch.no_grad()
    def _gather_parameters(self) -> None:
        if self.world_size <= 1:
            return
        for g, buf in zip(self.flat_groups, self._gather_bufs):
            comm.all_gather_flat(buf, g.shard)
            g.data.copy_(buf)

    # ---------------------------------------------------------- persistence
    def state_dict(self) -> dict:
        return self.adam.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.adam.load_state_dict(state)

    def _grad_device(self) -> torch.device:
        return self.flat_groups[0].grad.device
