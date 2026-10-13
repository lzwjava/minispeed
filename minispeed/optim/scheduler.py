"""LR schedulers compatible with the DeepSpeed scheduler config schema.

All schedulers are stepped once per *optimizer* step (i.e. once per gradient
accumulation boundary, after a non-skipped update) by the engine.
"""

from __future__ import annotations

from typing import Optional


class _BaseLRScheduler:
    """Holds a reference to the flat optimizer and writes ``lr`` into its
    ``param_groups`` (the same contract as ``torch.optim.lr_scheduler``)."""

    def __init__(
        self,
        optimizer,
        warmup_min_lr: float = 0.0,
        warmup_max_lr: float = 1e-3,
        warmup_num_steps: int = 0,
        total_num_steps: Optional[int] = None,
    ):
        self.optimizer = optimizer
        self.warmup_min_lr = float(warmup_min_lr)
        self.warmup_max_lr = float(warmup_max_lr)
        self.warmup_num_steps = max(0, int(warmup_num_steps))
        self.total_num_steps = (
            int(total_num_steps) if total_num_steps is not None else None
        )
        self.last_epoch = -1  # becomes 0 on the first step()
        self.base_lrs = [
            float(g.get("lr", self.warmup_max_lr)) for g in optimizer.param_groups
        ]

    # ------------------------------------------------------------------ math
    def _warmup_lr(self, epoch: int) -> float:
        if self.warmup_num_steps <= 0:
            return self.warmup_max_lr
        if epoch < self.warmup_num_steps:
            return self.warmup_min_lr + (self.warmup_max_lr - self.warmup_min_lr) * (
                epoch / self.warmup_num_steps
            )
        return self.warmup_max_lr

    def get_lr(self) -> float:  # pragma: no cover - overridden
        raise NotImplementedError

    # ----------------------------------------------------------------- hooks
    def step(self, epoch: Optional[int] = None) -> None:
        self.last_epoch = self.last_epoch + 1 if epoch is None else epoch
        lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def get_last_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def state_dict(self) -> dict:
        return {
            "type": type(self).__name__,
            "last_epoch": self.last_epoch,
            "warmup_min_lr": self.warmup_min_lr,
            "warmup_max_lr": self.warmup_max_lr,
            "warmup_num_steps": self.warmup_num_steps,
            "total_num_steps": self.total_num_steps,
        }

    def load_state_dict(self, state: dict) -> None:
        self.last_epoch = int(state["last_epoch"])


class WarmupLR(_BaseLRScheduler):
    """Linear warmup then constant at ``warmup_max_lr``."""

    def get_lr(self) -> float:
        return self._warmup_lr(self.last_epoch)


class WarmupDecayLR(_BaseLRScheduler):
    """Linear warmup then linear decay to ``warmup_min_lr`` at
    ``total_num_steps`` (DeepSpeed's WarmupDecayLR behavior)."""

    def get_lr(self) -> float:
        epoch = self.last_epoch
        if epoch < self.warmup_num_steps:
            return self._warmup_lr(epoch)
        if self.total_num_steps is None:
            return self.warmup_max_lr
        total = self.total_num_steps
        warm = self.warmup_num_steps
        if epoch >= total:
            return self.warmup_min_lr
        frac = (total - epoch) / max(1, total - warm)
        return self.warmup_min_lr + (self.warmup_max_lr - self.warmup_min_lr) * frac


class ConstantLR(WarmupLR):
    """Alias kept for config compatibility."""


def build_scheduler(optimizer, scheduler_type: str, params: dict):
    total = params.get("total_num_steps")
    kwargs = dict(
        warmup_min_lr=params.get("warmup_min_lr", 0.0),
        warmup_max_lr=params.get("warmup_max_lr"),
        warmup_num_steps=params.get("warmup_num_steps", 0),
        total_num_steps=total,
    )
    if scheduler_type in ("WarmupLR", "WarmupLRConst", "ConstantLR"):
        return WarmupLR(optimizer, **kwargs)
    if scheduler_type == "WarmupDecayLR":
        return WarmupDecayLR(optimizer, **kwargs)
    raise ValueError(f"Unknown scheduler type: {scheduler_type}")
