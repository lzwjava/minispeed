# minispeed

**mini DeepSpeed** — a minimal GPU AI framework for distributed training, written
in a few thousand lines of readable PyTorch.

It implements the *core* of [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
from first principles: flat fp32 parameter buffers, a handful of collectives,
ZeRO sharding, dynamic loss scaling and a DeepSpeed-compatible JSON config.

```python
import minispeed as ms

engine, optimizer, _, lr_scheduler = ms.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json",          # same schema as DeepSpeed
)

for x, y in loader:
    loss = model_loss(engine(x), y)
    engine.backward(loss)             # loss scaling + grad accumulation
    engine.step()                     # no-op until the accumulation boundary
```

## Features

| Feature | Status |
| --- | --- |
| Data-parallel training (`torchrun`, NCCL / gloo) | ✅ |
| ZeRO stage 1 (optimizer-state sharding) | ✅ |
| ZeRO stage 2 (+ gradient sharding via reduce-scatter) | ✅ |
| fp16 + fp32 master weights + **dynamic loss scaling** | ✅ |
| inf/nan gradient detection → skip step, halve scale | ✅ |
| bf16 (no loss scaling) | ✅ |
| Gradient accumulation (`train_micro_batch_size_per_gpu` × `gas`) | ✅ |
| Global gradient clipping (exact global norm, even under ZeRO-2) | ✅ |
| `WarmupLR` / `WarmupDecayLR` (DeepSpeed scheduler schema) | ✅ |
| DeepSpeed-style JSON config | ✅ |
| Sharded checkpoints (`global_stepN/`, `latest` pointer, resume) | ✅ |
| Activation checkpointing helper | ✅ |
| Timers (`engine.timers`) | ✅ |
| ZeRO stage 3 (parameter sharding) | ❌ |
| CPU / NVMe offload, pipeline parallel, compression | ❌ |

## Quickstart

```bash
pip install -e .            # or: pip install -e . --break-system-packages
# running straight from the checkout also works: examples/ bootstraps sys.path

# single GPU, fp16 + ZeRO stage 2
python examples/train_lm.py --dtype fp16 --zero 2

# multi-process — minispeed picks NCCL when there is one GPU per rank,
# and falls back to gloo (CUDA tensors) when ranks share a GPU
torchrun --nproc_per_node=2 examples/train_lm.py --dtype fp16 --zero 2

# resume where you left off (all ranks must use the same world size)
torchrun --nproc_per_node=2 examples/train_lm.py --steps 300 --resume out

# DeepSpeed-style JSON config
python examples/train_lm.py --config examples/ds_config.json --steps 50
```

Tests (CPU + GPU, single- and multi-process):

```bash
python -m pytest tests/ -q        # 36 passed, 2 skipped on a 1-GPU machine
```

## Configuration

The config schema is a subset of DeepSpeed's `ds_config.json`:

```jsonc
{
  "train_micro_batch_size_per_gpu": 8,
  "train_batch_size": 32,            // optional: validated against gas × world
  "gradient_accumulation_steps": 4,
  "gradient_clipping": 1.0,
  "dtype": "fp16",                   // fp32 | fp16 | bf16
  "fp16": { "enabled": true, "loss_scale": 0,
            "initial_scale_power": 16, "loss_scale_window": 1000 },
  "optimizer":   { "type": "AdamW",
                   "params": { "lr": 1e-3, "betas": [0.9, 0.999],
                               "eps": 1e-8, "weight_decay": 0.01 } },
  "scheduler":   { "type": "WarmupDecayLR",
                   "params": { "warmup_min_lr": 0.0, "warmup_max_lr": 1e-3,
                               "warmup_num_steps": 100,
                               "total_num_steps": 1000 } },
  "zero_optimization": { "stage": 2 }   // 0, 1 or 2
}
```

Invalid combinations (`stage: 3`, offload, fp16+bf16, inconsistent batch sizes)
fail fast with a `ConfigError`.

## How it works

```
minispeed/
├── engine.py                  DeepSpeedEngine: backward / step / boundaries
├── config.py                  DeepSpeed JSON parsing + validation
├── comm.py                    backend & device choice, collective wrappers
├── flat.py                    FlatParamGroup: params → contiguous fp32 buffers
├── precision.py               DynamicLossScaler, inf/nan detection
├── checkpoint.py              sharded save/load (global_stepN + latest)
├── zero/stage_1_and_2.py      reduction, sharding, all-gather
├── optim/adam.py              AdamW over flat buffers + ZeRO shards
├── optim/scheduler.py         WarmupLR / WarmupDecayLR
└── activation_checkpointing.py
```

One training step (at a gradient-accumulation boundary):

```
micro-batches        loss × (1/gas) × loss_scale → backward()   [engine.backward]
   ↓                 param.grad accumulates normally (PyTorch)
boundary reached     1. flatten param.grad → fp32 flat buffer    (flat.py)
                     2. inf/nan? → one flag all-reduce            (precision.py)
                        └─ yes: halve scale, skip step, zero grads
                     3. grads /= loss_scale                       (unscale)
                     4. stage 0/1: all-reduce + ÷ world   │ stage 2: reduce-scatter
                     5. global grad norm → clip           │ (exact via SUM of
                        (stage 2: one SUM over shards)    │  shard sum-of-squares)
                     6. AdamW on this rank's shard  → stage ≥1: all-gather params
                     7. fp32 master → model dtype (fp16/bf16)
                     8. loss-scale window ↑, lr_scheduler.step(), zero grads
```

### ZeRO in three lines

* **stage 0** — all-reduce gradients; every rank keeps the full optimizer state.
* **stage 1** — all-reduce gradients; `exp_avg`/`exp_avg_sq` cover only
  `[rank·shard, (rank+1)·shard)` of the flat buffer; updated shard is
  all-gathered so every rank ends the step with identical fp32 masters.
* **stage 2** — same, but gradients are `reduce_scatter_tensor`ed, so each rank
  never materializes the full gradient. The global norm stays exact because the
  per-rank shard sums-of-squares are a partition of the whole gradient.

`numel` is padded to a multiple of `world_size` so shards are equal; all
collectives operate **once per parameter group** (the flat buffer *is* the
communication bucket).

### Checkpoint layout

```
out/
├── latest                       # text file: newest tag
└── global_step120/
    ├── mp_rank_00_model_states.pt                      # module state_dict
    ├── zero_pp_rank0_mp_rank_00_optim_states.pt        # rank 0 shard + masters
    └── zero_pp_rank1_mp_rank_00_optim_states.pt        # rank 1 shard + masters
```

The optimizer files carry the fp32 master weights, scheduler and loss-scaler
state, so resume restores an *identical* trajectory (verified by
`test_checkpoint_roundtrip_restores_training`). ZeRO shards are tied to the
world size that wrote them — resuming with a different `world_size` raises.

## Testing

| Test file | What it proves |
| --- | --- |
| `test_engine.py` | engine trajectory ≡ plain `torch.optim.AdamW` loop; accumulation ≡ big batch; bf16; checkpoint replay; boundary semantics |
| `test_distributed.py` | 2 ranks (gloo/CPU), stages 0/1/2 ≡ single-process run on the union batch; optimizer state really partitioned |
| `test_gpu.py` | fp16 on GPU, loss-scale growth, forced-∞ skip (weights untouched), 2 ranks × 1 GPU (gloo/CUDA), NCCL ≥ 2 GPUs |
| `test_precision.py`, `test_config.py` | loss-scaler state machine, scheduler math, config validation |

## Environment variables

| Variable | Effect |
| --- | --- |
| `MINISPEED_BACKEND` | force `nccl` or `gloo` |
| `MINISPEED_DEVICE` | force device, e.g. `cpu`, `cuda:1` |

Defaults: **NCCL** when `world_size ≤ device_count`, otherwise **gloo**;
device defaults to `cuda:LOCAL_RANK` (clamped) when CUDA is visible, else CPU.

## Limitations

* ZeRO stages 3+, optimizer/parameter offload, pipeline/tensor parallel,
  compression and FP1 optimizer are intentionally out of scope.
* Sharded checkpoints require the same `world_size` to resume.
* The optimizer is a single flat Adam/AdamW (one hyperparameter set per
  parameter group, as in `torch.optim`).

MIT licensed.
