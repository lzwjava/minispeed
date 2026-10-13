"""The minispeed training engine (mini version of DeepSpeedEngine).

Responsibilities:
  * data-parallel gradient reduction (all-reduce) or ZeRO reduce-scatter
  * gradient accumulation with loss scaling by 1/gas
  * fp16/bf16 mixed precision with a fp32 master copy of the weights
  * dynamic loss scaling + inf/nan skip
  * gradient clipping (global norm)
  * LR scheduling and sharded checkpointing

Typical usage::

    import minispeed as ms

    engine, optimizer, _, lr_scheduler = ms.initialize(
        model=model, model_parameters=params, config="ds_config.json"
    )
    for step, (x, y) in enumerate(loader):
        loss = engine(x, y)
        engine.backward(loss)
        engine.step()          # no-op until the accumulation boundary
"""

from __future__ import annotations

import logging
from typing import Iterable, Iterator, List, Optional, Union

import torch
from torch import nn

from . import checkpoint as ckpt
from . import comm
from .config import DeepSpeedConfig, load_config
from .flat import FlatParamGroup, build_flat_groups
from .optim import build_scheduler
from .precision import DynamicLossScaler, global_overflow, has_overflow
from .utils import Timers
from .zero import build_zero_optimizer

logger = logging.getLogger("minispeed")


def _normalize_param_groups(
    model: nn.Module,
    model_parameters,
    defaults: dict,
) -> List[dict]:
    """Accept torch.optim-style param groups, a flat param iterable, or None."""
    if model_parameters is None:
        raw_groups = [list(model.parameters())]
    elif isinstance(model_parameters, dict):
        raw_groups = [model_parameters]
    else:
        items = list(model_parameters)
        if len(items) > 0 and isinstance(items[0], dict):
            raw_groups = items  # list of torch.optim-style group dicts
        else:
            raw_groups = [items]  # single flat group

    normalized = []
    for group in raw_groups:
        if isinstance(group, dict):
            params = group.get("params", [])
            opts = {k: v for k, v in group.items() if k != "params"}
        else:
            params, opts = group, {}
        merged = dict(defaults)
        merged.update(opts)
        merged["params"] = [
            p for p in params
            if isinstance(p, nn.Parameter) and p.requires_grad
        ]
        if merged["params"]:
            normalized.append(merged)
    if not normalized:
        raise ValueError("No trainable parameters found")
    return normalized


class DeepSpeedEngine:
    """Wraps a ``torch.nn.Module`` with distributed + mixed-precision training."""

    def __init__(
        self,
        model: nn.Module,
        config: Union[DeepSpeedConfig, dict, str, None],
        model_parameters=None,
    ):
        self.config = load_config(config, world_size=comm.get_world_size())
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()
        self.device = comm.resolve_device()
        self.timers = Timers()

        self.module = model.to(self.device)

        # --- identical starting point on every rank -------------------------
        self._broadcast_model()

        # --- flatten into fp32 master buffers --------------------------------
        groups = _normalize_param_groups(
            model, model_parameters, self.config.optimizer_params
        )
        self.flat_groups: List[FlatParamGroup]
        option_groups: List[dict]
        self.flat_groups, option_groups = build_flat_groups(
            groups,
            rank=self.rank,
            world_size=self.world_size,
            zero_stage=self.config.zero_stage,
            device=self.device,
        )
        for g in self.flat_groups:
            g.copy_model_to_flat()  # fp32 master snapshot (params are fp32 here)

        # --- cast model params to the training dtype -------------------------
        self._param_dtype = None
        if self.config.dtype != "fp32":
            target = self.config.torch_dtype
            self._param_dtype = target
            for g in self.flat_groups:
                for p in g.params:
                    p.data = p.data.to(target)

        # --- ZeRO-3: masters + params shrink to this rank's shard ------------
        if self.config.zero_stage >= 3:
            for g in self.flat_groups:
                g.partition_params()

        # --- optimizer / scheduler / loss scaler ------------------------------
        self.optimizer = build_zero_optimizer(
            self.flat_groups,
            option_groups,
            zero_stage=self.config.zero_stage,
        )
        self.lr_scheduler = build_scheduler(
            self.optimizer, self.config.scheduler_type, self.config.scheduler_params
        )

        scaler_enabled = self.config.fp16_enabled
        self.loss_scaler = DynamicLossScaler(
            enabled=scaler_enabled,
            init_scale=self.config.loss_scale,
            scale_window=self.config.loss_scale_window,
        )
        if scaler_enabled and self.config.loss_scale == 0:
            self.loss_scaler.set_scale_from_power(self.config.initial_scale_power)

        # --- counters ---------------------------------------------------------
        self.micro_steps = 0       # forward/backward micro-batches performed
        self.global_steps = 0      # completed (non-skipped) optimizer steps
        self._consumed_boundary = -1
        self.skipped_steps = 0     # overflow-skipped optimizer steps
        self.grad_norm: Optional[float] = None

    # ------------------------------------------------------------- properties
    @property
    def gas(self) -> int:
        return self.config.gas

    @property
    def is_gradient_accumulation_boundary(self) -> bool:
        return self.micro_steps > 0 and self.micro_steps % self.gas == 0

    @property
    def _boundary_pending(self) -> bool:
        return (
            self.is_gradient_accumulation_boundary
            and self._consumed_boundary != self.micro_steps
        )

    @property
    def loss_scale(self) -> float:
        return self.loss_scaler.loss_scale_factor

    @property
    def parameters(self) -> Iterable[nn.Parameter]:
        return self.module.parameters()

    # ------------------------------------------------------------ nn surface
    def _cast_inputs(self, obj):
        """Recursively cast floating-point inputs to the model dtype so that
        fp32 batches work directly with fp16/bf16 weights (a mini version of
        DeepSpeed's automatic input casting)."""
        if self._param_dtype is None:
            return obj
        if isinstance(obj, torch.Tensor):
            if obj.is_floating_point() and obj.dtype != self._param_dtype:
                return obj.to(self._param_dtype)
            return obj
        if isinstance(obj, dict):
            return {k: self._cast_inputs(v) for k, v in obj.items()}
        if isinstance(obj, tuple):
            return tuple(self._cast_inputs(v) for v in obj)
        if isinstance(obj, list):
            return [self._cast_inputs(v) for v in obj]
        return obj

    def forward(self, *args, **kwargs):
        # ZeRO-3: parameters are partitioned between steps — all-gather them
        # on demand (no-op for stages 0-2 / while already gathered).
        if not self.optimizer.params_gathered:
            with self.timers.timer("gather"):
                self.optimizer.gather_parameters()
        return self.module(
            *self._cast_inputs(args), **self._cast_inputs(kwargs)
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self, mode: bool = True):
        self.module.train(mode)
        return self

    def eval(self):
        self.module.eval()
        return self

    def __iter__(self) -> Iterator:
        """DeepSpeed-compatible unpacking:
        ``engine, optimizer, _, lr_scheduler = ms.initialize(...)``."""
        return iter((self, self.optimizer, None, self.lr_scheduler))

    # ---------------------------------------------------------------- backward
    def _scale_loss(self, loss: torch.Tensor, scale_wrt_gas: bool) -> torch.Tensor:
        factor = self.loss_scaler.loss_scale_factor
        if scale_wrt_gas and self.gas > 1:
            factor /= self.gas
        if factor == 1.0:
            return loss
        return loss * factor

    def backward(
        self,
        loss: torch.Tensor,
        scale_wrt_gas: bool = True,
        retain_graph: bool = False,
        create_graph: bool = False,
    ) -> None:
        """Run ``loss.backward()`` with loss scaling applied.

        Gradients accumulate on the model parameters across micro-batches
        (standard PyTorch behavior); they are flattened, reduced and applied
        in :meth:`step` at the accumulation boundary.
        """
        with self.timers.timer("backward"):
            scaled = self._scale_loss(loss, scale_wrt_gas=scale_wrt_gas)
            scaled.backward(retain_graph=retain_graph, create_graph=create_graph)
        self.micro_steps += 1

    # ------------------------------------------------------------------- step
    def step(self) -> bool:
        """Advance the gradient-accumulation window.

        Returns True only when an optimizer update was actually applied.
        At the accumulation boundary this performs:
        flatten -> overflow check -> unscale -> reduce -> clip -> update ->
        gather -> scheduler step.
        """
        if not self._boundary_pending:
            return False
        self._consumed_boundary = self.micro_steps
        with self.timers.timer("step"):
            return self._take_optimizer_step()

    @torch.no_grad()
    def _take_optimizer_step(self) -> bool:
        # 1) fp16/fp32 model grads -> contiguous fp32 flat buffer
        for g in self.flat_groups:
            g.flatten_grads()

        # 2) global inf/nan check (cheap: one flag reduce)
        local_bad = any(has_overflow(g.grad) for g in self.flat_groups)
        overflow = global_overflow(local_bad)

        if overflow:
            # skip this update entirely: no reduce, no scheduler step
            self.loss_scaler.update(overflow=True)
            self.skipped_steps += 1
            self.optimizer.zero_grad()
            self.optimizer.free_parameters()  # ZeRO-3: params back to shards
            if self.skipped_steps % max(1, self.config.steps_per_print) == 0:
                logger.warning(
                    "rank %d: detected inf/nan gradients at micro_step=%d, "
                    "skipping step (loss_scale=%.1f, skipped=%d)",
                    self.rank,
                    self.micro_steps,
                    self.loss_scaler.scale,
                    self.skipped_steps,
                )
            return False

        # 3) remove the loss scale so the optimizer sees true gradients
        scale = self.loss_scaler.loss_scale_factor
        if scale != 1.0:
            for g in self.flat_groups:
                g.unscale_grads(scale)

        # 4) data-parallel / ZeRO reduction
        with self.timers.timer("reduce"):
            self.optimizer.reduce_gradients()

        # 5) global gradient norm + clipping
        self.grad_norm = self.optimizer.clip_gradients(
            self.config.gradient_clipping
        )

        # 6) update this rank's shard and sync parameters across ranks
        #    (stages 0-2: all-gather masters → model; stage 3: free the full
        #     params — they are re-gathered from the updated masters at the
        #     next forward)
        self.optimizer.step()
        self.optimizer.sync_parameters()

        # 7) bookkeeping
        self.loss_scaler.update(overflow=False)
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        self.global_steps += 1
        return True

    def zero_grad(self, set_to_none: bool = True) -> None:
        for g in self.flat_groups:
            g.zero_grad()
        for p in self.module.parameters():
            p.grad = None

    # ----------------------------------------------------------- checkpointing
    def save_checkpoint(self, save_dir: str, tag: Optional[str] = None, **kwargs):
        return ckpt.save_checkpoint(self, save_dir, tag=tag, **kwargs)

    def load_checkpoint(self, load_dir: str, tag: Optional[str] = None, **kwargs):
        return ckpt.load_checkpoint(self, load_dir, tag=tag, **kwargs)

    def state_dict(self):
        # stage 3: parameters live as 1/N slices between steps — gather them
        # so the returned tensors carry their real shapes (collective).
        self.optimizer.gather_parameters()
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.gather_parameters()  # stage 3 needs full-shaped params
        self.module.load_state_dict(state_dict)
        for g in self.flat_groups:
            g.copy_model_to_flat()

    # ----------------------------------------------------------------- helpers
    @torch.no_grad()
    def _broadcast_model(self) -> None:
        if not comm.is_distributed():
            return
        for p in self.module.parameters():
            comm.broadcast(p.data, src=0)
        for b in self.module.buffers():
            comm.broadcast(b.data, src=0)

    def get_loss_scale(self) -> float:
        return self.loss_scaler.scale

    def get_gradient_norm(self) -> Optional[float]:
        return self.grad_norm

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DeepSpeedEngine({self.config!r}, "
            f"global_steps={self.global_steps}, micro_steps={self.micro_steps})"
        )


def initialize(
    model: Optional[nn.Module] = None,
    model_parameters=None,
    config: Union[DeepSpeedConfig, dict, str, None] = None,
    model_args=None,  # accepted for API compatibility, unused
):
    """Initialize distributed + build the engine (mini ``deepspeed.initialize``).

    Returns the engine; it unpacks like DeepSpeed:
    ``engine, optimizer, _, lr_scheduler = initialize(...)``.
    """
    if model is None:
        raise ValueError("model is required")
    comm.init_distributed()
    engine = DeepSpeedEngine(model, config, model_parameters=model_parameters)
    if comm.get_rank() == 0:
        logger.info("minispeed initialized: %s", engine.config)
    return engine
