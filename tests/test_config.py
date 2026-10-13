import pytest

from minispeed.config import ConfigError, DeepSpeedConfig


def test_gas_derived_from_train_batch_size():
    cfg = DeepSpeedConfig(
        {
            "train_batch_size": 32,
            "train_micro_batch_size_per_gpu": 8,
        },
        world_size=2,
    )
    assert cfg.gas == 2
    assert cfg.train_batch_size == 32


def test_inconsistent_batch_sizes_raise():
    with pytest.raises(ConfigError):
        DeepSpeedConfig(
            {"train_batch_size": 33, "train_micro_batch_size_per_gpu": 8},
            world_size=2,
        )


def test_dtype_resolution():
    assert DeepSpeedConfig({}).dtype == "fp32"
    assert DeepSpeedConfig({"fp16": {"enabled": True}}).dtype == "fp16"
    assert DeepSpeedConfig({"bf16": {"enabled": True}}).dtype == "bf16"
    assert DeepSpeedConfig({"dtype": "bf16"}).dtype == "bf16"
    with pytest.raises(ConfigError):
        DeepSpeedConfig({"dtype": "fp8"})


def test_fp16_and_bf16_conflict():
    with pytest.raises(ConfigError):
        DeepSpeedConfig({"fp16": {"enabled": True}, "bf16": {"enabled": True}})


def test_zero_stage_validation():
    assert DeepSpeedConfig({"zero_optimization": {"stage": 1}}).zero_stage == 1
    with pytest.raises(ConfigError):
        DeepSpeedConfig({"zero_optimization": {"stage": 3}})
    with pytest.raises(ConfigError):
        DeepSpeedConfig(
            {"zero_optimization": {"stage": 1, "offload_optimizer": {"device": "cpu"}}}
        )


def test_defaults_from_optimizer_lr():
    cfg = DeepSpeedConfig(
        {"optimizer": {"type": "AdamW", "params": {"lr": 3e-4}}}
    )
    assert cfg.scheduler_params["warmup_max_lr"] == 3e-4
    assert cfg.optimizer_params["betas"] == (0.9, 0.999)
