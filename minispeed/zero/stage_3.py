"""ZeRO stage 3: parameters, gradients *and* optimizer state are partitioned.

Stages 1/2 shard the optimizer state (+ gradients at stage 2) but every rank
still keeps the **full** fp32 master buffer and the full model parameters
resident. Stage 3 shards everything: each rank only ever stores

  * its ``1/world_size`` slice of the fp32 master buffer
    (``FlatParamGroup.data`` after ``partition_params``),
  * the matching Adam moments,
  * its reduce-scattered gradient shard, and
  * a ``1/world_size`` placeholder slice of each model parameter
    (a flat model-dtype tensor standing in for ``p.data`` between steps).

The full parameter tensors only exist between *gather* and *free*::

    engine(x)  ──►  gather_parameters()   one all-gather + cast per group
                   (the transient full fp32 master buffer is released
                    as soon as the params are materialized)
       ...          forward/backward run on the full fp16/bf16 params
    engine.step ──►  Adam on the local master shard
                   ──►  free_parameters()  params drop back to their slices,
                        refreshed from the updated masters

Gradient handling is identical to stage 2 (reduce-scatter, shard-sized
resident gradients, exact global norm from shard sums-of-squares) — the full
flat gradient buffer only lives from flatten to reduce-scatter.

Compared with DeepSpeed, minispeed gathers the *whole model at once* instead
of per-layer buckets: one collective per parameter group, no prefetch or
re-use-distance machinery. That is a deliberate simplification — the peak
transient cost is one full fp32 buffer per group at gather time.
"""

from __future__ import annotations

from typing import List

import torch

from .. import comm
from ..flat import FlatParamGroup
from .stage_1_and_2 import ZeROOptimizer


class ZeROStage3Optimizer(ZeROOptimizer):
    """Optimizer-side ZeRO-3 plus parameter gather/free.

    The gradient reduction, clipping and Adam update are exactly stage 2's
    (the base class methods); what this class adds is that the *parameters*
    are partitioned too and materialized on demand.
    """

    def __init__(
        self,
        flat_groups: List[FlatParamGroup],
        option_groups: List[dict],
    ):
        # stage 3 reuses the stage-2 gradient/optimizer-state semantics
        super().__init__(flat_groups, option_groups, zero_stage=3)
        self._gathered = False

    # ------------------------------------------------- parameter lifecycle
    @property
    def params_gathered(self) -> bool:
        return self._gathered

    @torch.no_grad()
    def gather_parameters(self) -> None:
        """All-gather the fp32 master shards and materialize full params.

        Collective: every rank must call this together (the engine does so
        lazily on the first :meth:`DeepSpeedEngine.forward` of a window, and
        checkpointing gathers around save/load).
        """
        if self._gathered:
            return
        for g in self.flat_groups:
            g.materialize_params()  # p.data = empty tensors of real shape
            # transient full fp32 masters; released right after the copy
            full = torch.empty(
                g.numel_padded, dtype=torch.float32, device=g.data.device
            )
            comm.all_gather_flat(full, g.shard)
            g.copy_flat_to_model(full)  # masters -> params (casts to model dtype)
            del full
        self._gathered = True

    @torch.no_grad()
    def free_parameters(self) -> None:
        """Replace the full params by their locally-owned slices.

        The placeholders are refreshed from this rank's master shard, so the
        sharded model-dtype weights always mirror the (up-to-date) masters.
        """
        if not self._gathered:
            return
        for g in self.flat_groups:
            g.dematerialize_params()  # p.data = flat local slice (model dtype)
            g.copy_flat_to_model()    # refresh slice from this rank's shard
        self._gathered = False

    @torch.no_grad()
    def sync_parameters(self) -> None:
        """After the optimizer step: drop the full params back to their slices.

        Unlike stages 0-2 there is no master all-gather to propagate — every
        rank already holds the full (gathered) params, and the updated
        masters are local shards that get re-gathered at the next forward.
        """
        self.free_parameters()

    # --------------------------------------------------------------- step
    @torch.no_grad()
    def step(self) -> bool:
        """Update only the local master shard (params are synced lazily)."""
        self.adam.step()
        return True

    # ------------------------------------------------------------ reduction
    @torch.no_grad()
    def reduce_gradients(self) -> None:
        super().reduce_gradients()
        for g in self.flat_groups:
            # the full flat grad buffer has been reduce-scattered into
            # g.grad_shard; dropping it keeps resident gradients at shard size
            g.grad = None
