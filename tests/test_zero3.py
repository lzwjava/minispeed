"""ZeRO stage 3: partitioned parameters + equivalence with plain DP.

Stage 3 must be numerically identical to stage 0 while the resident state
(fp32 masters, gradients, parameters) shrinks to ``1/world_size`` between
compute phases: params are flat locally-owned slices between steps and full
tensors only while a forward/backward window is open.
"""

import copy

import pytest
import torch

import minispeed as ms
from tests.common import base_config, build_model, make_batch, mse_loss, train_engine

STEPS = 4
GAS = 2
MICRO = 4


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("MINISPEED_DEVICE", "cpu")
    monkeypatch.setenv("MINISPEED_BACKEND", "gloo")


def _run(stage: int):
    cfg = base_config(micro=MICRO, gas=GAS, zero_stage=stage, clip=1.0, lr=1e-3)
    x, y = make_batch(STEPS, GAS, world=1, micro=MICRO)
    model = build_model(seed=1234, device="cpu")
    engine = ms.initialize(model=model, model_parameters=list(model.parameters()),
                           config=cfg)
    losses = train_engine(engine, x, y, STEPS, GAS, MICRO, device="cpu")
    # engine.state_dict() gathers stage-3 params back to their real shapes
    return engine, torch.tensor(losses), engine.state_dict()


def test_stage3_matches_stage0():
    """Same data + same reduction ops => identical trajectory and weights."""
    eng0, losses0, params0 = _run(0)
    eng3, losses3, params3 = _run(3)

    torch.testing.assert_close(losses3, losses0, rtol=1e-5, atol=1e-7)
    assert eng3.global_steps == eng0.global_steps == STEPS
    for k in params0:
        torch.testing.assert_close(params3[k], params0[k],
                                   rtol=1e-5, atol=1e-7)


def test_stage3_params_gather_and_free_lifecycle():
    cfg = base_config(micro=MICRO, gas=GAS, zero_stage=3, clip=1.0)
    x, y = make_batch(1, GAS, world=1, micro=MICRO)
    model = build_model(device="cpu")
    engine = ms.initialize(model=model, model_parameters=list(model.parameters()),
                           config=cfg)
    g = engine.flat_groups[0]

    # between steps the params are flat locally-owned slices (1/N per rank)
    assert not engine.optimizer.params_gathered
    assert all(p.data.dim() == 1 for p in model.parameters())
    assert g.grad is None  # full flat grad buffer not resident

    # opening a forward window materializes the real shapes
    loss = mse_loss(engine(x[0, 0]), y[0, 0])
    assert engine.optimizer.params_gathered
    for p, orig in zip(model.parameters(), build_model(device="cpu").parameters()):
        assert p.shape == orig.shape

    engine.backward(loss)
    assert engine.step() is False  # gas boundary not reached: params stay open
    assert engine.optimizer.params_gathered

    # at the boundary the params drop back to their slices and the full grad
    # buffer is gone (masters + Adam moments + grad shard stay at shard size)
    for m in range(1, GAS):
        loss = mse_loss(engine(x[0, m]), y[0, m])
        engine.backward(loss)
    assert engine.step() is True
    assert not engine.optimizer.params_gathered
    assert all(p.data.dim() == 1 for p in model.parameters())
    assert g.grad is None
    assert g.data.numel() == g.numel_shard()
    assert engine.optimizer.state[0]["exp_avg"].numel() == g.numel_shard()

    # raw model.state_dict() shows the partitioned view; engine.state_dict()
    # (which gathers) shows the real one
    raw = model.state_dict()
    full = engine.state_dict()
    for k in full:
        assert raw[k].dim() == 1 and full[k].shape.numel() == raw[k].numel()


def test_stage3_partition_roundtrip():
    """Partition/materialize/dematerialize preserve values exactly."""
    from minispeed.flat import FlatParamGroup

    torch.manual_seed(0)
    params = [torch.nn.Parameter(torch.randn(7, 5)), torch.nn.Parameter(torch.randn(3))]
    originals = [p.data.clone() for p in params]
    total = sum(p.numel() for p in params)

    g = FlatParamGroup(params, rank=1, world_size=2, zero_stage=3,
                       device=torch.device("cpu"))
    g.copy_model_to_flat()
    g.partition_params()

    # master buffer sharded; each param reduced to its locally-owned slice
    assert g.data.numel() == g.numel_padded // 2
    assert g.shard_start == g.numel_padded // 2
    assert sum(p.numel() for p in params) == g.numel_shard()
    assert all(p.data.dim() == 1 for p in params)

    # gather: materialize + fill from a full master buffer
    full = torch.zeros(g.numel_padded)
    offset = 0
    for orig in originals:
        n = orig.numel()
        full[offset : offset + n].copy_(orig.reshape(-1))
        offset += n
    g.materialize_params()
    for p, orig in zip(params, originals):
        assert p.shape == orig.shape
    g.copy_flat_to_model(full)
    for p, orig in zip(params, originals):
        torch.testing.assert_close(p.data, orig)

    # free: placeholders equal the locally-owned elements of the masters
    # (masters here are the original values — no optimizer step happened)
    g.dematerialize_params()
    g.copy_flat_to_model()
    ms_, me_ = g.shard_start, g.shard_end
    offset = 0
    for p, orig in zip(params, originals):
        n = orig.numel()
        lo, hi = max(offset, ms_), min(offset + n, me_)
        assert p.data.numel() == hi - lo
        torch.testing.assert_close(
            p.data, orig.reshape(-1)[lo - offset : hi - offset]
        )
        offset += n

    # and gathering again restores the exact original tensors
    g.materialize_params()
    g.copy_flat_to_model(full)
    for p, orig in zip(params, originals):
        torch.testing.assert_close(p.data, orig)


def test_stage3_checkpoint_roundtrip_restores_training():
    cfg = base_config(micro=MICRO, gas=GAS, warmup_steps=4,
                      scheduler="WarmupDecayLR", total_steps=40,
                      zero_stage=3)
    x, y = make_batch(STEPS + 2, GAS, world=1, micro=MICRO)
    model = build_model(device="cpu")
    engine = ms.initialize(model=model, model_parameters=list(model.parameters()),
                           config=cfg)
    train_engine(engine, x, y, 2, GAS, MICRO, device="cpu")

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        engine.save_checkpoint(d, tag="mid")
        # save must have returned params to their freed state
        assert not engine.optimizer.params_gathered

        expected_losses = train_engine(engine, x, y, 2, GAS, MICRO, device="cpu")
        expected_steps = engine.global_steps
        expected_params = copy.deepcopy(engine.state_dict())

        engine.load_checkpoint(d, tag="mid")
        assert engine.global_steps == expected_steps - 2

        replay_losses = train_engine(engine, x, y, 2, GAS, MICRO, device="cpu")
        torch.testing.assert_close(
            torch.tensor(replay_losses), torch.tensor(expected_losses),
            rtol=1e-5, atol=1e-7,
        )
        for k, v in engine.state_dict().items():
            torch.testing.assert_close(v, expected_params[k],
                                       rtol=1e-5, atol=1e-7)


def test_stage3_overflow_skip_frees_params():
    """inf/nan gradients skip the step and still return params to shards."""
    cfg = base_config(micro=MICRO, gas=1, zero_stage=3, clip=1.0,
                      dtype="fp16", loss_scale_power=6, scale_window=100)
    x, y = make_batch(1, 1, world=1, micro=MICRO)
    model = build_model(device="cpu")
    engine = ms.initialize(model=model, model_parameters=list(model.parameters()),
                           config=cfg)
    loss = mse_loss(engine(x[0, 0]), y[0, 0])
    engine.backward(loss)
    for p in model.parameters():
        if p.grad is not None and p.grad.numel():
            p.grad.reshape(-1)[0] = float("inf")
            break

    scale_before = engine.get_loss_scale()
    assert engine.step() is False          # overflow -> skipped
    assert engine.skipped_steps == 1
    assert engine.global_steps == 0
    assert engine.get_loss_scale() == scale_before / 2
    assert not engine.optimizer.params_gathered   # freed despite the skip
    assert engine.flat_groups[0].grad is None

    # training recovers on the next clean window
    losses = train_engine(engine, x, y, 1, 1, MICRO, device="cpu")
    assert all(torch.isfinite(torch.tensor(losses)))
    assert engine.global_steps == 1


def test_stage3_eval_forward_works_after_training():
    """engine(x) after a completed step re-gathers transparently."""
    cfg = base_config(micro=MICRO, gas=GAS, zero_stage=3)
    x, y = make_batch(1, GAS, world=1, micro=MICRO)
    model = build_model(device="cpu")
    engine = ms.initialize(model=model, model_parameters=list(model.parameters()),
                           config=cfg)
    train_engine(engine, x, y, 1, GAS, MICRO, device="cpu")
    engine.eval()
    with torch.no_grad():
        out = engine(x[0, 0])
    assert out.shape == (MICRO, 1)
    assert engine.optimizer.params_gathered