"""minispeed — a mini DeepSpeed: GPU AI framework for distributed training.

Implements the core of DeepSpeed in a few thousand lines of readable PyTorch:

  * :class:`~minispeed.engine.DeepSpeedEngine`  — the training engine
  * ZeRO stages 1 & 2 (optimizer-state / gradient sharding + param all-gather)
  * fp16/bf16 mixed precision with fp32 master weights + dynamic loss scaling
  * gradient accumulation, global gradient clipping, inf/nan skip
  * DeepSpeed-style WarmupDecayLR schedulers and sharded checkpoints

Usage::

    import minispeed as ms

    engine, optim, _, lr_sched = ms.initialize(
        model=model, model_parameters=model.parameters(), config=ds_config
    )
    loss = engine(x, y)
    engine.backward(loss)
    engine.step()
"""

from . import comm
from .config import ConfigError, DeepSpeedConfig
from .engine import DeepSpeedEngine, initialize
from .optim import WarmupDecayLR, WarmupLR
from .precision import DynamicLossScaler
from .utils import Timers, set_seed
from .zero import ZeROOptimizer

__version__ = "0.1.0"

__all__ = [
    "initialize",
    "DeepSpeedEngine",
    "DeepSpeedConfig",
    "ConfigError",
    "ZeROOptimizer",
    "DynamicLossScaler",
    "WarmupLR",
    "WarmupDecayLR",
    "Timers",
    "set_seed",
    "comm",
    "__version__",
]
