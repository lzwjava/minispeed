"""Small utilities: timers and reproducibility helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator, List

import torch

from . import comm


class Timer:
    def __init__(self, name: str):
        self.name = name
        self.elapsed = 0.0
        self.started_at: float | None = None

    @contextmanager
    def section(self) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.elapsed += time.perf_counter() - start

    def reset(self) -> None:
        self.elapsed = 0.0


class Timers:
    """Named wall-clock timers (DeepSpeed exposes a similar ``engine.timers``)."""

    def __init__(self) -> None:
        self._timers: Dict[str, Timer] = {}

    def __call__(self, name: str) -> Timer:
        if name not in self._timers:
            self._timers[name] = Timer(name)
        return self._timers[name]

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        with self(name).section():
            yield

    def names(self) -> List[str]:
        return list(self._timers)

    def reset(self) -> None:
        for t in self._timers.values():
            t.reset()

    def log(self, rank0_only: bool = True) -> str:
        if rank0_only and comm.get_rank() != 0:
            return ""
        return " | ".join(
            f"{name}: {t.elapsed:.3f}s" for name, t in self._timers.items()
        )


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
