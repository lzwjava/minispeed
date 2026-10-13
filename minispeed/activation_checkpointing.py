"""Gradient checkpointing (activation recomputation) — a thin, DeepSpeed-style
wrapper around ``torch.utils.checkpoint``.

Saves activation memory at the cost of a recompute in backward::

    from minispeed.activation_checkpointing import checkpoint

    hidden = checkpoint(self.block, hidden, use_reentrant=False)
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint as _torch_checkpoint


def checkpoint(function, *args, **kwargs):
    """Run ``function`` under activation checkpointing."""
    use_reentrant = kwargs.pop("use_reentrant", False)
    return _torch_checkpoint(function, *args, use_reentrant=use_reentrant, **kwargs)


def checkpoint_sequential(functions, *args, **kwargs):
    use_reentrant = kwargs.pop("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint_sequential(
        functions, *args, use_reentrant=use_reentrant, **kwargs
    )


__all__ = ["checkpoint", "checkpoint_sequential"]
