"""GPU tests: fp16 mixed precision, inf/nan skip, and 2-rank NCCL training.

Skipped automatically when CUDA is not available.
"""

import os

import pytest
import torch

import minispeed as ms
from tests.common import base_config, build_model, free_port, make_batch, train_engine
from tests.common import mse_loss

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required"
)

STEPS = 6
GAS = 2
MICRO = 4


@pytest.fixture(autouse=True)
def gpu_env(monkeypatch):
    monkeypatch.setenv("MINISPEED_DEVICE", "cuda")
    monkeypatch.setenv("MINISPEED_BACKEND", "nccl")


def test_fp16_trains_on_gpu():
    cfg = base_config(micro=MICRO, gas=GAS, dtype="fp16",
                      loss_scale_power=4, scale_window=3, lr=1e-2, clip=1.0)
    x, y = make_batch(STEPS, GAS, world=1, micro=MICRO, device="cuda")
    model = build_model(device="cuda")
    engine = ms.initialize(model=model, model_parameters=list(model.parameters()),
                           config=cfg)
    losses = train_engine(engine, x, y, STEPS, GAS, MICRO, device="cuda")

    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0]
    # weights are fp16 in the model, fp32 masters in the flat buffer
    assert next(model.parameters()).dtype == torch.float16
    assert engine.flat_groups[0].data.dtype == torch.float32
    # dynamic loss scaling: grew after `scale_window` clean steps
    assert engine.get_loss_scale() > 2.0 ** 4
    assert engine.loss_scaler.skipped_steps == 0
    # gradients are clipped to the global norm
    assert engine.grad_norm is not None and engine.grad_norm > 0


def test_fp16_overflow_skips_step():
    cfg = base_config(micro=MICRO, gas=GAS, dtype="fp16", loss_scale_power=6,
                      scale_window=100, lr=1e-2, clip=1.0, warmup_steps=5,
                      scheduler="WarmupLR")
    x, y = make_batch(STEPS + 2, GAS, world=1, micro=MICRO, device="cuda")
    model = build_model(device="cuda")
    engine = ms.initialize(model=model, model_parameters=list(model.parameters()),
                           config=cfg)
    train_engine(engine, x, y, 1, GAS, MICRO, device="cuda")

    # poison a gradient with inf between backward() and step()
    loss = mse_loss(engine(x[1, 0]), y[1, 0])
    engine.backward(loss)
    loss = mse_loss(engine(x[1, 1]), y[1, 1])
    engine.backward(loss)
    for p in model.parameters():
        if p.grad is not None and p.grad.numel():
            p.grad.reshape(-1)[0] = float("inf")
            break

    steps_before = engine.global_steps
    scale_before = engine.get_loss_scale()
    lr_before = engine.lr_scheduler.get_last_lr()
    params_before = {k: v.clone() for v in [0] for k, v in []} or {
        n: p.clone() for n, p in model.named_parameters()
    }

    assert engine.step() is False  # overflow -> step skipped
    assert engine.global_steps == steps_before
    assert engine.skipped_steps == 1
    assert engine.get_loss_scale() == scale_before / 2  # scale halved
    assert engine.lr_scheduler.get_last_lr() == lr_before  # no sched step
    for n, p in model.named_parameters():
        torch.testing.assert_close(p, params_before[n])  # weights untouched

    # training must recover on the next clean accumulation window
    losses = train_engine(engine, x, y, 2, GAS, MICRO, device="cuda")
    assert all(torch.isfinite(torch.tensor(losses)))
    assert engine.global_steps == steps_before + 2


def _nccl_worker(rank, world_size, port, tmpdir):
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
            "MINISPEED_BACKEND": "nccl",
            # ranks may share a physical GPU (device_count < world_size)
            "MINISPEED_DEVICE": f"cuda:{rank % torch.cuda.device_count()}",
        }
    )
    cfg = base_config(micro=MICRO, gas=GAS, dtype="fp16", zero_stage=2,
                      loss_scale_power=4, scale_window=8, lr=1e-2, clip=1.0)
    model = build_model(seed=1234)
    engine = ms.initialize(model=model, model_parameters=list(model.parameters()),
                           config=cfg)
    # both ranks intentionally train on the SAME batches: with identical data
    # their recorded losses must stay bit-for-bit close after every gather.
    x, y = make_batch(STEPS, GAS, world=1, micro=MICRO, device=str(engine.device))
    losses = train_engine(engine, x, y, STEPS, GAS, MICRO, device=engine.device)

    torch.save(
        {
            "losses": torch.tensor(losses),
            "params": {k: v.clone() for k, v in model.state_dict().items()},
            "global_steps": engine.global_steps,
            "scale": engine.get_loss_scale(),
            "skipped": engine.loss_scaler.skipped_steps,
        },
        os.path.join(tmpdir, f"nccl_rank{rank}.pt"),
    )
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.parametrize("stage", [1, 2])
def test_two_rank_nccl_fp16(stage, tmp_path):
    # NCCL requires a distinct GPU per rank; the single-GPU, two-rank case is
    # covered by test_two_ranks_one_gpu (gloo over CUDA tensors) instead.
    if torch.cuda.device_count() < 2:
        pytest.skip("NCCL multi-rank needs >= 2 visible CUDA devices")
    world = 2
    port = free_port()
    torch.multiprocessing.spawn(
        _nccl_worker, args=(world, port, str(tmp_path)), nprocs=world, join=True
    )
    r0 = torch.load(tmp_path / "nccl_rank0.pt", weights_only=False)
    r1 = torch.load(tmp_path / "nccl_rank1.pt", weights_only=False)

    assert r0["global_steps"] == STEPS
    assert r0["skipped"] == 0  # clean run: no overflow skips
    assert r0["scale"] >= 2.0 ** 4
    assert all(torch.isfinite(r0["losses"]))
    assert r0["losses"][-1] < r0["losses"][0]

    # identical data on both ranks -> losses must track each other
    torch.testing.assert_close(r0["losses"], r1["losses"], rtol=1e-3, atol=1e-3)
    # params synced across ranks by the ZeRO all-gather
    for k in r0["params"]:
        torch.testing.assert_close(r0["params"][k], r1["params"][k],
                                   rtol=1e-4, atol=1e-5)


def _gloo_cuda_worker(rank, world_size, stage, port, tmpdir, dtype):
    """2 ranks sharing ONE GPU: NCCL forbids this, gloo allows it."""
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
            "MINISPEED_BACKEND": "gloo",
            "MINISPEED_DEVICE": "cuda",
        }
    )
    cfg = base_config(micro=MICRO, gas=GAS, dtype=dtype, zero_stage=stage,
                      clip=1.0, lr=1e-3, wd=0.01)
    model = build_model(seed=1234)
    engine = ms.initialize(model=model, model_parameters=list(model.parameters()),
                           config=cfg)
    x, y = make_batch(STEPS, GAS, WORLD, MICRO, device="cuda")
    x_r = x[:, :, rank * MICRO : (rank + 1) * MICRO, :]
    y_r = y[:, :, rank * MICRO : (rank + 1) * MICRO, :]
    losses = train_engine(engine, x_r, y_r, STEPS, GAS, MICRO, device=engine.device)
    torch.save(
        {"losses": torch.tensor(losses),
         "params": {k: v.clone() for k, v in model.state_dict().items()},
         "global_steps": engine.global_steps},
        os.path.join(tmpdir, f"gloo_{dtype}_stage{stage}_rank{rank}.pt"),
    )
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def _gpu_reference(dtype: str):
    cfg = base_config(micro=MICRO * WORLD, gas=GAS, dtype=dtype, zero_stage=0,
                      clip=1.0, lr=1e-3, wd=0.01)
    model = build_model(seed=1234)
    engine = ms.initialize(model=model, model_parameters=list(model.parameters()),
                           config=cfg)
    x, y = make_batch(STEPS, GAS, WORLD, MICRO, device="cuda")
    losses = train_engine(engine, x, y, STEPS, GAS, MICRO, device=engine.device)
    return torch.tensor(losses)


WORLD = 2


@pytest.mark.parametrize("stage", [0, 1, 2])
@pytest.mark.parametrize("dtype", ["fp32", "fp16"])
def test_two_ranks_one_gpu(stage, dtype, tmp_path):
    port = free_port()
    torch.multiprocessing.spawn(
        _gloo_cuda_worker,
        args=(WORLD, stage, port, str(tmp_path), dtype),
        nprocs=WORLD,
        join=True,
    )
    r0 = torch.load(tmp_path / f"gloo_{dtype}_stage{stage}_rank0.pt",
                    weights_only=False)
    r1 = torch.load(tmp_path / f"gloo_{dtype}_stage{stage}_rank1.pt",
                    weights_only=False)
    ref = _gpu_reference(dtype)

    assert r0["global_steps"] == STEPS
    mean_losses = (r0["losses"] + r1["losses"]) / 2
    torch.testing.assert_close(mean_losses, ref, rtol=5e-3, atol=1e-4)
    for k in r0["params"]:
        torch.testing.assert_close(r0["params"][k], r1["params"][k],
                                   rtol=1e-4, atol=1e-5)
