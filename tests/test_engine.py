"""Single-process correctness tests: minispeed engine vs plain PyTorch."""

import copy

import pytest
import torch
import torch.nn.functional as F

import minispeed as ms
from tests.common import base_config, build_model, make_batch, train_engine

STEPS = 6
GAS = 2
MICRO = 4


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("MINISPEED_DEVICE", "cpu")
    monkeypatch.setenv("MINISPEED_BACKEND", "gloo")


def torch_reference(x, y, steps, gas, micro, lr, wd, clip, seed=1234):
    """Plain PyTorch loop mirroring what the engine should compute."""
    model = build_model(seed=seed, device="cpu")
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd
    )
    losses = []
    for s in range(steps):
        for m in range(gas):
            xb = x[s, m]
            yb = y[s, m]
            loss = F.mse_loss(model(xb), yb) / gas  # scale_wrt_gas
            losses.append(float(loss.detach()) * gas)
            loss.backward()
            if (m + 1) % gas == 0:
                if clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
    return losses, model


def test_engine_matches_plain_torch_adamw():
    lr, wd, clip = 1e-3, 0.01, 1.0
    cfg = base_config(micro=MICRO, gas=GAS, lr=lr, wd=wd, clip=clip,
                      scheduler="WarmupLR", warmup_steps=0)
    x, y = make_batch(STEPS, GAS, world=1, micro=MICRO)

    ref_losses, ref_model = torch_reference(x, y, STEPS, GAS, MICRO, lr, wd, clip)

    torch.manual_seed(1234)
    model = build_model(seed=1234, device="cpu")
    engine = ms.initialize(model=model, model_parameters=list(model.parameters()),
                           config=cfg)
    losses = train_engine(engine, x, y, STEPS, GAS, MICRO, device="cpu")

    assert engine.global_steps == STEPS
    torch.testing.assert_close(
        torch.tensor(losses), torch.tensor(ref_losses), rtol=1e-4, atol=1e-6
    )
    for p_eng, p_ref in zip(model.parameters(), ref_model.parameters()):
        torch.testing.assert_close(p_eng, p_ref, rtol=1e-4, atol=1e-6)


def test_gradient_accumulation_equals_large_batch():
    """gas=2 over 2 micro-batches must equal one optimizer step on the
    concatenated batch (loss averaging done by 1/gas scaling)."""
    lr, wd = 5e-3, 0.0
    cfg = base_config(micro=MICRO, gas=GAS, lr=lr, wd=wd, clip=None)
    x, y = make_batch(STEPS, GAS, world=1, micro=MICRO)

    # reference: full batch per optimizer step, no accumulation
    ref_cfg = base_config(micro=MICRO * GAS, gas=1, lr=lr, wd=wd, clip=None)
    ref_model = build_model(seed=1234, device="cpu")
    ref_engine = ms.initialize(model=ref_model,
                               model_parameters=list(ref_model.parameters()),
                               config=ref_cfg)
    x_big = x.reshape(STEPS, 1, GAS * MICRO, 8)
    y_big = y.reshape(STEPS, 1, GAS * MICRO, 1)
    train_engine(ref_engine, x_big, y_big, STEPS, 1, GAS * MICRO, device="cpu")

    model = build_model(seed=1234, device="cpu")
    engine = ms.initialize(model=model, model_parameters=list(model.parameters()),
                           config=cfg)
    train_engine(engine, x, y, STEPS, GAS, MICRO, device="cpu")

    for p_acc, p_ref in zip(model.parameters(), ref_model.parameters()):
        torch.testing.assert_close(p_acc, p_ref, rtol=1e-4, atol=1e-6)


def test_step_applies_once_per_boundary():
    cfg = base_config(micro=MICRO, gas=GAS)
    model = build_model(device="cpu")
    engine = ms.initialize(model=model, model_parameters=list(model.parameters()),
                           config=cfg)
    x, y = make_batch(1, GAS, world=1, micro=MICRO)

    applied = []
    for m in range(GAS):
        loss = F.mse_loss(engine(x[0, m]), y[0, m])
        engine.backward(loss)
        applied.append(engine.step())
        assert engine.is_gradient_accumulation_boundary == (m == GAS - 1)
    assert applied == [False, True]
    # a repeated step() at the same boundary must be a no-op
    assert engine.step() is False
    assert engine.global_steps == 1


def test_bf16_training_runs_and_learns():
    cfg = base_config(micro=MICRO, gas=GAS, dtype="bf16", lr=1e-2, clip=None)
    x, y = make_batch(STEPS, GAS, world=1, micro=MICRO)
    model = build_model(device="cpu")
    engine = ms.initialize(model=model, model_parameters=list(model.parameters()),
                           config=cfg)
    losses = train_engine(engine, x, y, STEPS, GAS, MICRO, device="cpu")

    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0]
    # model params were cast to bf16, masters stay fp32
    assert next(model.parameters()).dtype == torch.bfloat16
    assert engine.flat_groups[0].data.dtype == torch.float32
    # unscale must be a no-op for bf16 (no loss scaling)
    assert engine.get_loss_scale() == 1.0


def test_checkpoint_roundtrip_restores_training():
    cfg = base_config(micro=MICRO, gas=GAS, warmup_steps=4,
                      scheduler="WarmupDecayLR", total_steps=40)
    x, y = make_batch(STEPS + 2, GAS, world=1, micro=MICRO)
    model = build_model(device="cpu")
    engine = ms.initialize(model=model, model_parameters=list(model.parameters()),
                           config=cfg)
    train_engine(engine, x, y, 2, GAS, MICRO, device="cpu")

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        engine.save_checkpoint(d, tag="mid")
        expected_losses = train_engine(engine, x, y, 2, GAS, MICRO, device="cpu")
        expected_steps = engine.global_steps
        expected_params = copy.deepcopy(model.state_dict())

        engine.load_checkpoint(d, tag="mid")
        assert engine.global_steps == expected_steps - 2

        # replay: must reproduce the exact same trajectory
        replay_losses = train_engine(engine, x, y, 2, GAS, MICRO, device="cpu")
        torch.testing.assert_close(
            torch.tensor(replay_losses), torch.tensor(expected_losses),
            rtol=1e-5, atol=1e-7,
        )
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, expected_params[k], rtol=1e-5, atol=1e-7)


def test_initialize_tuple_unpacking():
    model = build_model(device="cpu")
    engine, optimizer, _, lr_scheduler = ms.initialize(
        model=model, model_parameters=list(model.parameters()),
        config=base_config(),
    )
    assert isinstance(engine, ms.DeepSpeedEngine)
    assert optimizer is engine.optimizer
    assert lr_scheduler is engine.lr_scheduler
    assert engine.gas == GAS


def test_activation_checkpointing_matches():
    from minispeed.activation_checkpointing import checkpoint

    model = build_model(device="cpu")
    x, _ = make_batch(1, 1, world=1, micro=MICRO)

    # grads without checkpointing
    out = model(x[0, 0])
    out.sum().backward()
    ref_grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)

    # grads with checkpointing
    out = checkpoint(lambda h: model(h), x[0, 0])
    out.sum().backward()
    for p, g in zip(model.parameters(), ref_grads):
        torch.testing.assert_close(p.grad, g, rtol=1e-5, atol=1e-7)
