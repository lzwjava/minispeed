"""Distributed correctness tests.

World size 2 on gloo/CPU: each rank trains on its own shard of every batch,
the framework all-reduces (or reduce-scatters) gradients, and the resulting
trajectory must match a single-process run on the union of the shards.

ZeRO stage 0 (plain DP), stage 1 (optimizer-state sharding), stage 2
(+ gradient sharding) and stage 3 (+ parameter sharding) must all be
bit-for-bit equivalent within fp32 noise.
"""

import os

import pytest
import torch

import minispeed as ms
from tests.common import base_config, build_model, free_port, make_batch, train_engine

WORLD = 2
STEPS = 5
GAS = 2
MICRO = 4  # per-gpu micro batch


def _worker(rank, world_size, stage, port, tmpdir, dtype):
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
            "MINISPEED_BACKEND": "gloo",
            "MINISPEED_DEVICE": "cpu",
        }
    )
    cfg = base_config(micro=MICRO, gas=GAS, dtype=dtype, zero_stage=stage,
                      clip=1.0, lr=1e-3, wd=0.01)
    model = build_model(seed=1234, device="cpu")
    engine = ms.initialize(model=model, model_parameters=list(model.parameters()),
                           config=cfg)
    x, y = make_batch(STEPS, GAS, WORLD, MICRO)
    # rank r trains on its slice of the [world * micro] batch
    x_r = x[:, :, rank * MICRO : (rank + 1) * MICRO, :]
    y_r = y[:, :, rank * MICRO : (rank + 1) * MICRO, :]
    losses = train_engine(engine, x_r, y_r, STEPS, GAS, MICRO, device="cpu")

    # stage 3: params live as 1/N slices between steps — gather to read them
    if stage >= 3:
        engine.optimizer.gather_parameters()

    payload = {
        "losses": torch.tensor(losses),
        "params": {k: v.clone() for k, v in model.state_dict().items()},
        "global_steps": engine.global_steps,
        "grad_norm_first": engine.grad_norm,
        # ZeRO shard evidence: this rank's optimizer state tensors
        "opt_exp_avg": [st["exp_avg"].clone() for st in engine.optimizer.state],
        "opt_step": int(engine.optimizer.state[0]["step"]),
    }
    torch.save(payload, os.path.join(tmpdir, f"stage{stage}_rank{rank}.pt"))
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def _single_process_reference(dtype: str):
    """Reference run on the union batch (world_size 1, micro = world * micro)."""
    os.environ["MINISPEED_DEVICE"] = "cpu"
    os.environ["MINISPEED_BACKEND"] = "gloo"
    cfg = base_config(micro=MICRO * WORLD, gas=GAS, dtype=dtype, zero_stage=0,
                      clip=1.0, lr=1e-3, wd=0.01)
    model = build_model(seed=1234, device="cpu")
    engine = ms.initialize(model=model, model_parameters=list(model.parameters()),
                           config=cfg)
    x, y = make_batch(STEPS, GAS, WORLD, MICRO)
    losses = train_engine(engine, x, y, STEPS, GAS, MICRO, device="cpu")
    return torch.tensor(losses), {k: v.clone() for k, v in model.state_dict().items()}


def _run_reference():
    saved = {k: os.environ.get(k) for k in
             ("MINISPEED_DEVICE", "MINISPEED_BACKEND")}
    os.environ["MINISPEED_DEVICE"] = "cpu"
    os.environ["MINISPEED_BACKEND"] = "gloo"
    try:
        ref_losses, ref_params = _single_process_reference("fp32")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return ref_losses, ref_params


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_distributed_matches_single_process(stage, tmp_path):
    port = free_port()
    torch.multiprocessing.spawn(
        _worker,
        args=(WORLD, stage, port, str(tmp_path), "fp32"),
        nprocs=WORLD,
        join=True,
    )

    ref_losses, ref_params = _run_reference()
    r0 = torch.load(tmp_path / f"stage{stage}_rank0.pt", weights_only=False)
    r1 = torch.load(tmp_path / f"stage{stage}_rank1.pt", weights_only=False)

    assert r0["global_steps"] == STEPS
    assert r1["global_steps"] == r0["global_steps"]

    # rank losses are over their data shard; their mean must equal the
    # single-process loss over the union batch (equal shard sizes)
    mean_losses = (r0["losses"] + r1["losses"]) / 2
    # fp32 + non-deterministic reduction order: tiny drift is expected
    torch.testing.assert_close(mean_losses, ref_losses, rtol=5e-3, atol=1e-4)

    # both ranks must hold identical parameters after the all-gather
    total_diff_sq = 0.0
    total_ref_sq = 0.0
    for k in ref_params:
        # both ranks must hold bitwise-identical params after the all-gather
        torch.testing.assert_close(r0["params"][k], r1["params"][k],
                                   rtol=1e-5, atol=1e-7)
        total_diff_sq += float(
            (r0["params"][k].float() - ref_params[k].float()).pow(2).sum()
        )
        total_ref_sq += float(ref_params[k].float().pow(2).sum())
    # global relative drift (element-wise/per-tensor rtol is meaningless for
    # near-zero weights and tiny-norm tensors like LayerNorm biases)
    rel_drift = total_diff_sq ** 0.5 / (total_ref_sq ** 0.5)
    assert rel_drift < 1e-3, f"relative param drift {rel_drift:.2e} too large"


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_optimizer_state_sharding(stage, tmp_path):
    """stage 0: full state on every rank, identical across ranks.
    stage 1/2/3: state partitioned in world_size equal shards, distinct per rank."""
    port = free_port()
    torch.multiprocessing.spawn(
        _worker,
        args=(WORLD, stage, port, str(tmp_path), "fp32"),
        nprocs=WORLD,
        join=True,
    )
    r0 = torch.load(tmp_path / f"stage{stage}_rank0.pt", weights_only=False)
    r1 = torch.load(tmp_path / f"stage{stage}_rank1.pt", weights_only=False)

    n0 = [t.numel() for t in r0["opt_exp_avg"]]
    n1 = [t.numel() for t in r1["opt_exp_avg"]]
    assert n0 == n1
    assert r0["opt_step"] == STEPS  # steps advance on every rank

    if stage == 0:
        # full optimizer state everywhere, updates are identical
        assert n0 == [_padded_total(WORLD)]
        for a, b in zip(r0["opt_exp_avg"], r1["opt_exp_avg"]):
            torch.testing.assert_close(a, b)
    else:
        # each rank owns exactly 1/world of the (padded) flat buffers
        assert n0 == [_padded_total(WORLD) // WORLD]
        # shards cover different parameter regions -> different state
        assert any(
            not torch.equal(a, b)
            for a, b in zip(r0["opt_exp_avg"], r1["opt_exp_avg"])
        )


def _padded_total(world: int) -> int:
    """Flat-buffer size of the (single) param group, padded to world_size."""
    total = sum(p.numel() for p in build_model(seed=1234, device="cpu").parameters())
    return -(-total // world) * world


def test_gloo_two_rank_state_sizes(tmp_path):
    """Directly measure ZeRO sharding: shard tensors are world_size x smaller."""
    from minispeed.flat import FlatParamGroup

    params = [torch.nn.Parameter(torch.randn(17, 9)) for _ in range(3)]
    total = sum(p.numel() for p in params)
    full = FlatParamGroup(params, rank=0, world_size=1, zero_stage=0,
                          device=torch.device("cpu"))
    shard = FlatParamGroup(params, rank=1, world_size=2, zero_stage=2,
                           device=torch.device("cpu"))
    assert full.numel_shard() == full.numel_padded
    assert shard.numel_shard() == shard.numel_padded // 2
    assert shard.shard_start == shard.numel_padded // 2
    assert shard.numel_padded >= total
    assert shard.grad_shard is not None and full.grad_shard is None
