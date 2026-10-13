"""Unit tests for the dynamic loss scaler and scheduler math."""

import torch

from minispeed.optim.scheduler import WarmupDecayLR, WarmupLR
from minispeed.precision import DynamicLossScaler, has_overflow


class _DummyOptimizer:
    def __init__(self, lr=1.0):
        self.param_groups = [{"lr": lr}]


def test_scaler_grows_after_window():
    scaler = DynamicLossScaler(enabled=True, init_scale=8.0, scale_window=3)
    for _ in range(3):
        scaler.update(overflow=False)
    assert scaler.scale == 16.0


def test_scaler_halves_on_overflow_and_resets_window():
    scaler = DynamicLossScaler(enabled=True, init_scale=16.0, scale_window=2)
    scaler.update(overflow=False)
    scaler.update(overflow=True)
    assert scaler.scale == 8.0
    assert scaler.skipped_steps == 1
    assert scaler.good_steps == 0
    # two good steps needed again before growing
    scaler.update(overflow=False)
    assert scaler.scale == 8.0
    scaler.update(overflow=False)
    assert scaler.scale == 16.0


def test_scaler_disabled_is_noop():
    scaler = DynamicLossScaler(enabled=False)
    assert scaler.loss_scale_factor == 1.0
    scaler.update(overflow=True)
    assert scaler.scale == 1.0


def test_has_overflow_detects_inf_and_nan():
    assert not has_overflow(torch.zeros(4))
    assert has_overflow(torch.tensor([1.0, float("inf")]))
    assert has_overflow(torch.tensor([float("nan")]))
    assert has_overflow(torch.zeros(2), torch.tensor([float("nan")]))


def test_warmup_lr_schedule():
    opt = _DummyOptimizer(lr=0.0)
    sched = WarmupLR(opt, warmup_min_lr=0.0, warmup_max_lr=1.0,
                     warmup_num_steps=4)
    lrs = []
    for _ in range(6):
        sched.step()
        lrs.append(sched.get_last_lr())
    # linear warmup: 0, .25, .5, .75, then constant 1.0
    assert lrs == [0.0, 0.25, 0.5, 0.75, 1.0, 1.0]


def test_warmup_decay_lr_schedule():
    opt = _DummyOptimizer(lr=0.0)
    sched = WarmupDecayLR(opt, warmup_min_lr=0.0, warmup_max_lr=1.0,
                          warmup_num_steps=2, total_num_steps=6)
    lrs = []
    for _ in range(7):
        sched.step()
        lrs.append(round(sched.get_last_lr(), 6))
    # warmup 0 -> 1 over 2 steps, then linear decay to 0 at step 6
    assert lrs == [0.0, 0.5, 1.0, 0.75, 0.5, 0.25, 0.0]
