"""Mixed-precision helpers: dynamic loss scaling (DeepSpeed-style)."""

from __future__ import annotations

import torch

from . import comm


class DynamicLossScaler:
    """Adaptive loss scaler.

    Mirrors DeepSpeed's ``DynamicLossScaler``:
      - scale starts at ``2 ** initial_scale_power`` and grows by 2x after
        ``loss_scale_window`` consecutive successful steps,
      - on inf/nan gradients the step is skipped and the scale is halved.

    ``enabled=False`` (fp32 / bf16) makes the scaler a no-op with scale 1.
    """

    def __init__(
        self,
        enabled: bool = True,
        init_scale: float = 0.0,
        scale_window: int = 1000,
        min_scale: float = 1.0,
        max_scale: float = 2.0**24,
    ):
        self.enabled = enabled
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_window = max(1, scale_window)
        if not enabled:
            self.scale = 1.0
        elif init_scale and init_scale > 0:
            self.scale = float(init_scale)
        else:
            self.scale = 1.0  # caller normally overrides via initial_scale_power
        self.good_steps = 0
        self.skipped_steps = 0

    def set_scale_from_power(self, power: int) -> None:
        if self.enabled:
            self.scale = float(2.0 ** int(power))

    @property
    def loss_scale_factor(self) -> float:
        return self.scale if self.enabled else 1.0

    def update(self, overflow: bool) -> None:
        """Adjust the scale after an optimizer step attempt."""
        if not self.enabled:
            return
        if overflow:
            self.scale = max(self.scale / 2.0, self.min_scale)
            self.good_steps = 0
            self.skipped_steps += 1
        else:
            self.good_steps += 1
            if self.good_steps >= self.scale_window and self.scale < self.max_scale:
                self.scale = min(self.scale * 2.0, self.max_scale)
                self.good_steps = 0

    def state_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "scale": self.scale,
            "good_steps": self.good_steps,
            "skipped_steps": self.skipped_steps,
        }

    def load_state_dict(self, state: dict) -> None:
        self.enabled = bool(state.get("enabled", self.enabled))
        self.scale = float(state.get("scale", self.scale))
        self.good_steps = int(state.get("good_steps", 0))
        self.skipped_steps = int(state.get("skipped_steps", 0))


def has_overflow(*grad_buffers: torch.Tensor) -> bool:
    """True if any element of any buffer is inf/nan (local check)."""
    for buf in grad_buffers:
        if buf is None or buf.numel() == 0:
            continue
        # isfinite().all() is cheaper than two separate reductions.
        if not torch.isfinite(buf).all():
            return True
    return False


def global_overflow(local_overflow: bool) -> bool:
    """OR the local overflow flag across all data-parallel ranks."""
    return comm.all_reduce_flag(local_overflow)
