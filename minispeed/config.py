"""DeepSpeed-style JSON config parsing for minispeed.

Supports the subset of the DeepSpeed config schema that matters for a minimal
implementation:

    {
      "train_batch_size": 32,
      "train_micro_batch_size_per_gpu": 8,
      "gradient_accumulation_steps": 4,
      "gradient_clipping": 1.0,
      "dtype": "fp16",                     # fp32 | bf16 | fp16
      "optimizer": {
        "type": "AdamW",
        "params": {"lr": 1e-3, "betas": [0.9, 0.999], "eps": 1e-8,
                   "weight_decay": 0.01}
      },
      "scheduler": {
        "type": "WarmupDecayLR",
        "params": {"warmup_min_lr": 0.0, "warmup_max_lr": 1e-3,
                   "warmup_num_steps": 100, "total_num_steps": 1000}
      },
      "fp16": {"enabled": true, "loss_scale": 0, "initial_scale_power": 16},
      "bf16": {"enabled": true},
      "zero_optimization": {"stage": 2}
    }
"""

from __future__ import annotations

import json
from typing import Optional, Union

SUPPORTED_DTYPES = ("fp32", "fp16", "bf16")
SUPPORTED_OPTIMIZERS = ("AdamW", "Adam")
SUPPORTED_SCHEDULERS = ("WarmupDecayLR", "WarmupLR", "WarmupLRConst", "ConstantLR")


class ConfigError(ValueError):
    pass


def _as_dict(config: Union[dict, str, None]) -> dict:
    if config is None:
        return {}
    if isinstance(config, str):
        with open(config, "r", encoding="utf-8") as f:
            return json.load(f)
    if isinstance(config, dict):
        return config
    raise ConfigError(f"Unsupported config type: {type(config)}")


class DeepSpeedConfig:
    """Parsed + validated DeepSpeed config."""

    def __init__(self, config: Union[dict, str, None], world_size: int = 1):
        raw = _as_dict(config)
        self.world_size = world_size
        self.raw = raw

        # --- batch / accumulation -----------------------------------------
        self.micro_batch = int(raw.get("train_micro_batch_size_per_gpu", 1))
        train_batch = raw.get("train_batch_size")
        gas = raw.get("gradient_accumulation_steps")

        if gas is None and train_batch is not None:
            per_step = self.micro_batch * world_size
            if train_batch % per_step != 0:
                raise ConfigError(
                    f"train_batch_size={train_batch} is not divisible by "
                    f"micro_batch*world_size={per_step}"
                )
            gas = train_batch // per_step
        self.gas = int(gas) if gas is not None else 1

        expected = self.micro_batch * self.gas * world_size
        if train_batch is not None and int(train_batch) != expected:
            raise ConfigError(
                f"train_batch_size={train_batch} != "
                f"micro_batch({self.micro_batch}) * gas({self.gas}) * "
                f"world_size({world_size}) = {expected}"
            )
        self.train_batch_size = expected

        # --- precision ------------------------------------------------------
        self.dtype = self._resolve_dtype(raw)

        # --- fp16 loss scaling ---------------------------------------------
        fp16 = raw.get("fp16", {}) or {}
        bf16 = raw.get("bf16", {}) or {}
        self.fp16_enabled = self.dtype == "fp16"
        self.bf16_enabled = self.dtype == "bf16"
        if fp16.get("enabled") and bf16.get("enabled"):
            raise ConfigError("fp16 and bf16 cannot both be enabled")
        self.loss_scale = float(fp16.get("loss_scale", 0.0))  # 0 => dynamic
        self.initial_scale_power = int(fp16.get("initial_scale_power", 16))
        self.loss_scale_window = int(fp16.get("loss_scale_window", 1000))
        self.scale_wrt_gas = bool(raw.get("scale_wrt_gas", True))

        # --- gradient clipping ---------------------------------------------
        clip = raw.get("gradient_clipping")
        self.gradient_clipping: Optional[float] = float(clip) if clip else None

        # --- zero ------------------------------------------------------------
        zero = raw.get("zero_optimization", {}) or {}
        if zero is False:
            zero = {}
        self.zero_stage = int(zero.get("stage", 0))
        if self.zero_stage not in (0, 1, 2):
            raise ConfigError(
                f"Unsupported zero stage {self.zero_stage} "
                "(minispeed implements stages 0, 1 and 2)"
            )
        if zero.get("offload_optimizer") or zero.get("offload_param"):
            raise ConfigError(
                "CPU/NVMe offload is not implemented in minispeed"
            )
        if zero.get("stage3_gather_16bit_weights_on_model_save"):
            raise ConfigError("stage 3 options are not supported")

        # --- optimizer --------------------------------------------------------
        opt = raw.get("optimizer", {}) or {}
        self.optimizer_type = str(opt.get("type", "AdamW"))
        if self.optimizer_type not in SUPPORTED_OPTIMIZERS:
            raise ConfigError(
                f"Unsupported optimizer '{self.optimizer_type}', "
                f"expected one of {SUPPORTED_OPTIMIZERS}"
            )
        opt_params = dict(opt.get("params", {}) or {})
        opt_params.setdefault("lr", 1e-3)
        opt_params.setdefault("betas", (0.9, 0.999))
        opt_params.setdefault("eps", 1e-8)
        opt_params.setdefault("weight_decay", 0.0)
        opt_params["betas"] = tuple(float(b) for b in opt_params["betas"])
        self.optimizer_params = opt_params

        # --- scheduler ---------------------------------------------------------
        sched = raw.get("scheduler", {}) or {}
        self.scheduler_type = str(sched.get("type", "WarmupLR"))
        if self.scheduler_type not in SUPPORTED_SCHEDULERS:
            raise ConfigError(
                f"Unsupported scheduler '{self.scheduler_type}', "
                f"expected one of {SUPPORTED_SCHEDULERS}"
            )
        sched_params = dict(sched.get("params", {}) or {})
        sched_params.setdefault("warmup_num_steps", 0)
        sched_params.setdefault(
            "warmup_max_lr", float(self.optimizer_params["lr"])
        )
        sched_params.setdefault("warmup_min_lr", 0.0)
        self.scheduler_params = sched_params

        # --- misc ---------------------------------------------------------------
        self.steps_per_print = int(raw.get("steps_per_print", 10))
        self.seed = int(raw.get("seed", 1234))
        self.drop_overflow = bool(raw.get("drop_overflow_on_loss", True))

    # ------------------------------------------------------------------ helpers
    def _resolve_dtype(self, raw: dict) -> str:
        dtype = raw.get("dtype")
        if dtype is not None:
            dtype = str(dtype).lower()
            if dtype not in SUPPORTED_DTYPES:
                raise ConfigError(f"dtype must be one of {SUPPORTED_DTYPES}")
            return dtype
        fp16 = raw.get("fp16", {}) or {}
        bf16 = raw.get("bf16", {}) or {}
        if fp16.get("enabled"):
            return "fp16"
        if bf16.get("enabled"):
            return "bf16"
        return "fp32"

    @property
    def torch_dtype(self):
        import torch

        return {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[self.dtype]

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"DeepSpeedConfig(dtype={self.dtype}, zero_stage={self.zero_stage}, "
            f"micro_batch={self.micro_batch}, gas={self.gas}, "
            f"world_size={self.world_size}, "
            f"clip={self.gradient_clipping})"
        )


def load_config(
    config: Union[DeepSpeedConfig, dict, str, None], world_size: int = 1
) -> DeepSpeedConfig:
    if isinstance(config, DeepSpeedConfig):
        return config
    return DeepSpeedConfig(config, world_size=world_size)