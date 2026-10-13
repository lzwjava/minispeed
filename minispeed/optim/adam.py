"""Optimizers operating on flat fp32 buffers with optional ZeRO sharding."""

from __future__ import annotations

from typing import List

import torch

from ..flat import FlatParamGroup


class FlatAdam:
    """Adam/AdamW over :class:`FlatParamGroup` buffers.

    Differences from ``torch.optim.AdamW`` are intentional and minimal:

    * parameters live in a contiguous fp32 master buffer (enables fp16/fp16
      master weights and single-collective all-gather),
    * the optimizer state (exp_avg / exp_avg_sq) only covers this rank's
      ZeRO shard ``[shard_start, shard_end)`` — that is exactly what
      ZeRO stage 1/2 partitions,
    * gradients handed to :meth:`step` are already reduced (all-reduced or
      reduce-scattered) and unscaled.

    With ``zero_stage == 0`` the shard covers the whole buffer, i.e. every
    rank keeps the full optimizer state (classic data parallel).
    """

    def __init__(
        self,
        flat_groups: List[FlatParamGroup],
        option_groups: List[dict],
        zero_stage: int = 0,
    ):
        self.flat_groups = flat_groups
        self.zero_stage = zero_stage

        # torch-style param_groups so schedulers can poke ``lr`` directly.
        self.param_groups: List[dict] = []
        self._state: List[dict] = []
        for flat, opts in zip(flat_groups, option_groups):
            group = dict(opts)
            group.setdefault("lr", 1e-3)
            group.setdefault("betas", (0.9, 0.999))
            group.setdefault("eps", 1e-8)
            group.setdefault("weight_decay", 0.0)
            self.param_groups.append(group)

            n = flat.numel_shard()
            self._state.append(
                {
                    "step": 0,
                    "exp_avg": torch.zeros(n, dtype=torch.float32, device=flat.data.device),
                    "exp_avg_sq": torch.zeros(n, dtype=torch.float32, device=flat.data.device),
                }
            )

    # ------------------------------------------------------------------- step
    @torch.no_grad()
    def step(self) -> bool:
        """Apply one optimizer update to this rank's owned parameter shard."""
        for group, flat, state in zip(self.param_groups, self.flat_groups, self._state):
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            wd = float(group.get("weight_decay", 0.0))

            shard = flat.shard  # fp32 slice owned by this rank
            grad = flat.grad_for_shard()
            if grad.shape != shard.shape:
                raise RuntimeError(
                    f"grad/shard shape mismatch: {grad.shape} vs {shard.shape}"
                )

            m = state["exp_avg"]
            v = state["exp_avg_sq"]
            state["step"] += 1
            step_t = state["step"]

            # Decoupled weight decay (AdamW).
            if wd != 0.0:
                shard.add_(shard, alpha=-lr * wd)

            # biased moment estimates
            m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            bias_c1 = 1.0 - beta1 ** step_t
            bias_c2 = 1.0 - beta2 ** step_t
            denom = (v / bias_c2).sqrt().add_(eps)
            shard.addcdiv_(m / bias_c1, denom, value=-lr)

        return True

    # ------------------------------------------------------------------ state
    def zero_grad(self) -> None:
        for flat in self.flat_groups:
            flat.zero_grad()

    def state_dict(self) -> dict:
        return {
            "param_groups": [
                {k: v for k, v in g.items()} for g in self.param_groups
            ],
            "shard_state": [
                {
                    "step": st["step"],
                    "exp_avg": st["exp_avg"].clone(),
                    "exp_avg_sq": st["exp_avg_sq"].clone(),
                }
                for st in self._state
            ],
        }

    def load_state_dict(self, state_dict: dict) -> None:
        for saved_group, group in zip(state_dict["param_groups"], self.param_groups):
            for k, v in saved_group.items():
                if k != "params":
                    group[k] = v
        for saved, st in zip(state_dict["shard_state"], self._state):
            st["step"] = int(saved["step"])
            st["exp_avg"].copy_(saved["exp_avg"])
            st["exp_avg_sq"].copy_(saved["exp_avg_sq"])

    @property
    def state(self) -> List[dict]:
        return self._state
