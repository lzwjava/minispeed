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
                       (stage 3 after :meth:`partition_params`: this rank's
                       shard only, size ``shard_size``)
      - ``grad``       full flat gradient buffer (stages 0/1/2 while reducing;
                       ``None`` in stage 3 outside an accumulation boundary —
                       rebuilt for flatten + reduce-scatter, then dropped)
      - ``grad_shard`` reduce-scattered gradient shard (stages 2/3), size ``shard_size``

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
        # stage 3 rebuilds the full grad buffer at every accumulation boundary
        # and drops it again right after the reduce-scatter.
        self.grad: torch.Tensor | None = (
            None
            if zero_stage >= 3
            else torch.zeros(self.numel_padded, dtype=torch.float32, device=device)
        )
        self.grad_shard: torch.Tensor | None = (
            torch.zeros(self.shard_size, dtype=torch.float32, device=device)
            if zero_stage >= 2
            else None
        )

        # (flat_offset, parameter) lookup table + original sizes. Sizes are
        # cached because stage 3 temporarily replaces p.data with a smaller
        # partition, after which p.numel() no longer describes the parameter.
        self._offsets: List[Tuple[int, nn.Parameter]] = []
        self._numels: List[int] = []
        self._shapes: List[Tuple[int, ...]] = []
        offset = 0
        for p in self.params:
            self._offsets.append((offset, p))
            self._numels.append(p.numel())
            self._shapes.append(tuple(p.shape))
            offset += p.numel()

        # stage 3 flag: data has been reduced to this rank's shard
        self._partitioned = False

    @property
    def master_range(self) -> Tuple[int, int]:
        """Flat range covered by ``self.data`` (the fp32 master buffer).

        stages 0-2 (and stage 3 before partitioning): the full buffer;
        stage 3 after :meth:`partition_params`: this rank's shard.
        """
        if self._partitioned:
            return self.shard_start, self.shard_end
        return 0, self.numel_padded

    # ------------------------------------------------------------- conversions
    def _pdata_range(self, offset: int, n: int, p: nn.Parameter) -> Tuple[int, int]:
        """Global flat range covered by ``p.data``.

        Full params (stages 0-2, stage 3 while gathered) cover the whole
        parameter ``[offset, offset + n)``; a stage-3 placeholder covers only
        the locally-owned slice ``[max(offset, shard_start), ...)``.
        """
        if p.data.numel() == n:
            return offset, offset + n
        start = max(offset, self.shard_start)
        return start, start + p.data.numel()

    @torch.no_grad()
    def copy_model_to_flat(self) -> None:
        """Snapshot model parameters into the fp32 master buffer.

        Only the elements this buffer covers are written (the parameter ∩
        ``master_range`` intersection), so for a stage-3 group this fills
        exactly this rank's shard — from the full params while gathered, or
        from the local placeholders while free.
        """
        ms, me = self.master_range
        for (offset, p), n in zip(self._offsets, self._numels):
            ps, pe = self._pdata_range(offset, n, p)
            lo, hi = max(ps, ms), min(pe, me)
            if lo >= hi:
                continue
            src = p.data.reshape(-1)[lo - ps : hi - ps]
            self.data[lo - ms : hi - ms].copy_(src)

    @torch.no_grad()
    def copy_flat_to_model(self, full_flat: torch.Tensor | None = None) -> None:
        """Write fp32 masters back into the (possibly fp16) params.

        ``full_flat`` is a full buffer covering ``[0, numel_padded)`` (the
        stage-3 gather result); by default ``self.data`` covering
        ``master_range`` is used. Only the covered parameter elements are
        touched, so stage 3 can refresh either the full params (while
        gathered) or just the local placeholder slices (while free).
        """
        if full_flat is not None:
            buf, base = full_flat, 0
        else:
            buf, base = self.data, self.master_range[0]
        end = base + buf.numel()
        for (offset, p), n in zip(self._offsets, self._numels):
            ps, pe = self._pdata_range(offset, n, p)
            lo, hi = max(ps, base), min(pe, end)
            if lo >= hi:
                continue
            dst = p.data.view(-1)[lo - ps : hi - ps]
            dst.copy_(buf[lo - base : hi - base])

    # --------------------------------------------------------- stage 3 params
    @torch.no_grad()
    def partition_params(self) -> None:
        """ZeRO-3: keep only this rank's master shard and replace each
        parameter's data with its locally-owned slice.

        Afterwards ``self.data`` has size ``shard_size`` and the model params
        hold flat model-dtype placeholders; the full parameter tensors only
        exist between :meth:`materialize_params` and
        :meth:`dematerialize_params`.
        """
        if self._partitioned:
            return
        self.data = self.data[self.shard_start : self.shard_end].clone()
        self._partitioned = True
        self.dematerialize_params()
        self.copy_flat_to_model()  # fill placeholders from the shard masters

    @torch.no_grad()
    def materialize_params(self) -> None:
        """ZeRO-3 gather (step 1): allocate full-shaped parameter tensors
        for :meth:`copy_flat_to_model` to fill."""
        if not self._partitioned:
            return
        for (offset, p), shape in zip(self._offsets, self._shapes):
            p.data = torch.empty(shape, dtype=p.data.dtype, device=p.data.device)

    @torch.no_grad()
    def dematerialize_params(self) -> None:
        """ZeRO-3 free: params drop back to flat locally-owned placeholders."""
        if not self._partitioned:
            return
        ms, me = self.shard_start, self.shard_end
        for (offset, p), n in zip(self._offsets, self._numels):
            lo, hi = max(offset, ms), min(offset + n, me)
            p.data = torch.empty(
                max(0, hi - lo), dtype=p.data.dtype, device=p.data.device
            )

    # ------------------------------------------------------------------ grads
    @torch.no_grad()
    def flatten_grads(self) -> None:
        """Copy ``param.grad`` (fp16/fp32) into the fp32 flat grad buffer.

        Parameters without gradients contribute zeros (e.g. frozen params).
        The buffer is (re)allocated on demand — stage 3 drops it after every
        reduce-scatter to keep the resident gradient at shard size.
        """
        if self.grad is None:
            self.grad = torch.zeros(self.numel_padded, dtype=torch.float32,
                                    device=self.data.device)
        else:
            self.grad.zero_()
        for (offset, p), n in zip(self._offsets, self._numels):
            if p.grad is None:
                continue
            self.grad[offset : offset + n].copy_(p.grad.data.reshape(-1))
            p.grad = None  # free the model-dtype gradient memory

    @torch.no_grad()
    def zero_grad(self) -> None:
        if self.grad is not None:
            if self.zero_stage >= 3:
                self.grad = None  # free; rebuilt by flatten_grads()
            else:
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
        if self._partitioned:
            return self.data  # stage 3: data *is* the shard
        return self.data[self.shard_start : self.shard_end]

    @torch.no_grad()
    def unscale_grads(self, scale: float) -> None:
        if scale == 1.0 or self.grad is None:
            return
        self.grad.div_(scale)

    @torch.no_grad()
    def scale_grads(self, factor: float) -> None:
        """Apply a gradient-clipping coefficient to this rank's owned grads."""
        if factor == 1.0:
            return
        if self.grad_shard is not None:
            self.grad_shard.mul_(factor)
        elif self.grad is not None:
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
