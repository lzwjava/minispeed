"""Flat (flattened) parameter groups.

DeepSpeed flattens each parameter group into one contiguous fp32 buffer so that
an entire group can be reduced/all-gathered with a single collective and
updated with vectorized tensor ops. minispeed does the same.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch
from torch import nn


def _round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


class FlatParamGroup:
    """One parameter group flattened into contiguous buffers.

    Buffers (all fp32, the master copy of the weights):
      - ``data``       full flat parameter buffer, size ``numel_padded``
      - ``grad``       full flat gradient buffer (stages 0/1), size ``numel_padded``
      - ``grad_shard`` reduce-scattered gradient shard (stage 2), size ``shard_size``

    Sharding: ``numel_padded`` is divisible by world_size, rank ``r`` owns
    ``[r * shard_size, (r + 1) * shard_size)``.
    """

    def __init__(
        self,
        params: Sequence[nn.Parameter],
        *,
        rank: int,
        world_size: int,
        zero_stage: int,
        device: torch.device,
    ):
        if len(params) == 0:
            raise ValueError("FlatParamGroup requires at least one parameter")
        self.params: List[nn.Parameter] = list(params)
        self.zero_stage = zero_stage

        self.numel = sum(p.numel() for p in self.params)
        self.numel_padded = _round_up(self.numel, world_size)
        self.shard_size = self.numel_padded // world_size

        if zero_stage >= 1:
            self.shard_start = rank * self.shard_size
            self.shard_end = self.shard_start + self.shard_size
        else:  # stage 0: every rank owns / updates the whole buffer
            self.shard_start = 0
            self.shard_end = self.numel_padded

        self.data = torch.zeros(self.numel_padded, dtype=torch.float32, device=device)
        self.grad = torch.zeros(self.numel_padded, dtype=torch.float32, device=device)
        self.grad_shard: torch.Tensor | None = (
            torch.zeros(self.shard_size, dtype=torch.float32, device=device)
            if zero_stage >= 2
            else None
        )

        # (flat_offset, parameter) lookup table
        self._offsets: List[Tuple[int, nn.Parameter]] = []
        offset = 0
        for p in self.params:
            self._offsets.append((offset, p))
            offset += p.numel()

    # ------------------------------------------------------------- conversions
    @torch.no_grad()
    def copy_model_to_flat(self) -> None:
        """Snapshot model parameters into the fp32 master buffer."""
        for offset, p in self._offsets:
            n = p.numel()
            self.data[offset : offset + n].copy_(p.data.reshape(-1))

    @torch.no_grad()
    def copy_flat_to_model(self) -> None:
        """Write the fp32 master buffer back into (possibly fp16) params."""
        for offset, p in self._offsets:
            n = p.numel()
            p.data.copy_(self.data[offset : offset + n].view(p.data.shape))

    # ------------------------------------------------------------------ grads
    @torch.no_grad()
    def flatten_grads(self) -> None:
        """Copy ``param.grad`` (fp16/fp32) into the fp32 flat grad buffer.

        Parameters without gradients contribute zeros (e.g. frozen params).
        """
        self.grad.zero_()
        for offset, p in self._offsets:
            if p.grad is None:
                continue
            n = p.numel()
            self.grad[offset : offset + n].copy_(p.grad.data.reshape(-1))
            p.grad = None  # free the model-dtype gradient memory

    @torch.no_grad()
    def zero_grad(self) -> None:
        self.grad.zero_()
        if self.grad_shard is not None:
            self.grad_shard.zero_()

    def grad_for_shard(self) -> torch.Tensor:
        """Gradient tensor covering this rank's owned slice (post-reduction)."""
        if self.grad_shard is not None:
            return self.grad_shard
        return self.grad[self.shard_start : self.shard_end]

    # ------------------------------------------------------------- statistics
    def local_grad_sumsq(self) -> torch.Tensor:
        g = self.grad_for_shard()
        return g.pow(2).sum()

    @property
    def shard(self) -> torch.Tensor:
        """This rank's slice of the fp32 parameter buffer."""
        return self.data[self.shard_start : self.shard_end]

    @torch.no_grad()
    def unscale_grads(self, scale: float) -> None:
        if scale == 1.0:
            return
        self.grad.div_(scale)

    @torch.no_grad()
    def scale_grads(self, factor: float) -> None:
        """Apply a gradient-clipping coefficient to this rank's owned grads."""
        if factor == 1.0:
            return
        if self.grad_shard is not None:
            self.grad_shard.mul_(factor)
        else:
            self.grad.mul_(factor)

    def numel_shard(self) -> int:
        return self.shard_end - self.shard_start


def build_flat_groups(
    param_groups: Iterable[dict],
    *,
    rank: int,
    world_size: int,
    zero_stage: int,
    device: torch.device,
) -> Tuple[List[FlatParamGroup], List[dict]]:
    """Turn torch-style param group dicts into FlatParamGroups + option dicts."""
    flat_groups: List[FlatParamGroup] = []
    option_groups: List[dict] = []
    for group in param_groups:
        params = [p for p in group["params"] if p.requires_grad]
        if not params:
            continue
        opts = {k: v for k, v in group.items() if k != "params"}
        flat_groups.append(
            FlatParamGroup(
                params,
                rank=rank,
                world_size=world_size,
                zero_stage=zero_stage,
                device=device,
            )
        )
        option_groups.append(opts)
    if not flat_groups:
        raise ValueError("No trainable parameters found")
    return flat_groups, option_groups
