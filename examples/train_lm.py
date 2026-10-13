"""minispeed example: train a tiny character-level GPT on an embedded corpus.

Demonstrates the whole DeepSpeed-style surface in one file:
DeepSpeed JSON config, fp16 loss scaling, ZeRO sharding, gradient
accumulation, WarmupDecayLR, checkpoint save/resume and rank-0 logging.

Single GPU::

    python examples/train_lm.py --dtype fp16 --zero 2
    python examples/train_lm.py --dtype fp16 --zero 3   # params sharded too

Multi-process (2 ranks share one GPU -> minispeed picks gloo automatically,
2 GPUs -> NCCL)::

    torchrun --nproc_per_node=2 examples/train_lm.py --dtype fp16 --zero 2

Resume::

    python examples/train_lm.py --resume out/latest
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# allow running the example straight from a git checkout (no pip install)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import minispeed as ms  # noqa: E402
from minispeed import comm  # noqa: E402

# ---------------------------------------------------------------------------
# A tiny embedded corpus (written for this example, public domain / CC0).
# ---------------------------------------------------------------------------
CORPUS = """
the data flows in micro batches and the gradients meet in the middle.
each rank owns a shard of the optimizer state and the weights are gathered
again after every step. the loss scale rises until an infinity appears, then
it halves and the step is skipped. warm up slowly, decay linearly, clip the
global norm and the small model learns to predict the next character.
the data flows in micro batches and the gradients meet in the middle.
distributed training is simple when the buffers are flat and the collectives
are few. all reduce the gradients, all gather the parameters, keep a master
copy in full precision and let every rank remember only its own slice.
the data flows in micro batches and the gradients meet in the middle.
zero stage one shards the optimizer, stage two shards the gradients as well,
stage three shards the weights and gathers them again for every forward pass,
and the model keeps training as if there were no partitions at all.
measure the timers, print the norm, save the checkpoint and resume tomorrow.
the data flows in micro batches and the gradients meet in the middle.
"""


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        att = att.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(att)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab: int, block_size: int = 64,
                 n_embd: int = 128, n_layer: int = 2, n_head: int = 4):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(Block(n_embd, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab, bias=False)
        self.head.weight = self.wte.weight  # weight tying

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


def make_dataset():
    """Encode the corpus; returns (ids, vocab_size, stoi, itos)."""
    chars = sorted(set(CORPUS))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    ids = torch.tensor([stoi[ch] for ch in CORPUS], dtype=torch.long)
    return ids, len(chars), stoi, itos


def get_batch(ids: torch.Tensor, block_size: int, batch: int,
              start: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic contiguous windows starting at ``start`` (different
    ranks get different windows, so batches are truly data parallel)."""
    xs, ys = [], []
    for i in range(batch):
        off = (start + i * 7) % max(1, ids.numel() - block_size)
        window = ids[off : off + block_size + 1]
        if window.numel() < block_size + 1:
            window = torch.cat([window, ids[: block_size + 1 - window.numel()]])
        xs.append(window[:-1])
        ys.append(window[1:])
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def build_config(args, world: int) -> dict:
    if args.config:
        return args.config  # DeepSpeed JSON path handed straight to ms.initialize
    cfg = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.gas,
        "gradient_clipping": 1.0,
        "dtype": args.dtype,
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": args.lr, "betas": [0.9, 0.95], "eps": 1e-8,
                       "weight_decay": 0.1},
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": args.lr * 0.1,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": args.warmup,
                "total_num_steps": args.steps,
            },
        },
        "zero_optimization": {"stage": args.zero},
        "steps_per_print": args.log_every,
    }
    if args.dtype == "fp16":
        cfg["fp16"] = {"enabled": True, "loss_scale": 0,
                       "initial_scale_power": 16, "loss_scale_window": 200}
    elif args.dtype == "bf16":
        cfg["bf16"] = {"enabled": True}
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="minispeed tiny LM example")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="micro batch size per gpu")
    parser.add_argument("--gas", type=int, default=2,
                        help="gradient accumulation steps")
    parser.add_argument("--dtype", default="fp16",
                        choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--zero", type=int, default=2, choices=[0, 1, 2, 3],
                        help="ZeRO stage: 1 optimizer state, 2 + gradients, "
                             "3 + parameters")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--ckpt-dir", default="out")
    parser.add_argument("--save-every", type=int, default=150)
    parser.add_argument("--resume", default=None,
                        help="checkpoint dir (resumes latest) or dir/tag, "
                             "e.g. out or out/global_step120")
    parser.add_argument("--config", default=None,
                        help="path to a DeepSpeed-style JSON config")
    args = parser.parse_args()

    comm.init_distributed()
    rank, world = comm.get_rank(), comm.get_world_size()
    device = comm.resolve_device()
    ms.set_seed(1337)  # identical init on every rank (engine also broadcasts)

    ids, vocab, stoi, itos = make_dataset()
    model = TinyGPT(vocab, args.block_size, args.n_embd, args.n_layer, args.n_head)

    engine, optimizer, _, lr_scheduler = ms.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=build_config(args, world),
    )
    if args.resume:
        if os.path.isdir(args.resume):  # resume dir -> follow the `latest` file
            engine.load_checkpoint(args.resume)
        else:  # explicit dir/tag
            engine.load_checkpoint(os.path.dirname(args.resume),
                                   tag=os.path.basename(args.resume))
        if rank == 0:
            print(f"resumed from {args.resume} at step {engine.global_steps}")

    n_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"minispeed {ms.__version__} | world={world} device={device} "
              f"| dtype={engine.config.dtype} zero={engine.config.zero_stage} "
              f"| params={n_params/1e3:.1f}K batch="
              f"{engine.config.train_batch_size}")

    tokens_per_step = engine.config.train_batch_size * args.block_size
    t0 = time.time()
    step = engine.global_steps

    while step < args.steps:
        # each rank reads its own slice of the corpus -> true data parallelism
        win = (step * engine.gas * args.batch_size + rank * args.batch_size)
        win = win % max(1, ids.numel() - args.block_size)
        for m in range(engine.gas):
            start = (win + m * args.batch_size * world) % max(
                1, ids.numel() - args.block_size
            )
            x, y = get_batch(ids, args.block_size, args.batch_size, start, device)
            logits = engine(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1)
            )
            engine.backward(loss)
            applied = engine.step()

        if applied:
            step = engine.global_steps
            if rank == 0 and (step % args.log_every == 0 or step == 1):
                dt = time.time() - t0
                tps = args.log_every * tokens_per_step / max(dt, 1e-9)
                print(
                    f"step {step:4d}/{args.steps} | loss {loss.item():.4f} | "
                    f"ppl {math.exp(min(loss.item(), 20)):8.2f} | "
                    f"lr {lr_scheduler.get_last_lr():.2e} | "
                    f"norm {engine.grad_norm:.3f} | "
                    f"scale {engine.get_loss_scale():.0f} | "
                    f"tok/s {tps:,.0f} | {engine.timers.log()}"
                )
                engine.timers.reset()
                t0 = time.time()
            if args.save_every and step % args.save_every == 0:
                engine.save_checkpoint(args.ckpt_dir)
        else:
            step = engine.global_steps  # overflow skip: keep looping

    # checkpointing is collective: every rank must participate
    engine.save_checkpoint(args.ckpt_dir)

    if rank == 0:
        print(f"done. final loss {loss.item():.4f}, "
              f"skipped steps {engine.skipped_steps}, "
              f"checkpoint -> {args.ckpt_dir}")

        # quick sampling demo
        engine.eval()
        prompt = "the data "
        context = torch.tensor([[stoi[c] for c in prompt]], device=device)
        with torch.no_grad():
            for _ in range(64):
                logits = engine(context[:, -args.block_size:])
                probs = F.softmax(logits[:, -1].float() / 0.8, dim=-1)
                nxt = torch.multinomial(probs, 1)
                context = torch.cat([context, nxt], dim=1)
        sample = "".join(itos[int(i)] for i in context[0].tolist())
        print(f"sample: {sample!r}")

    comm.barrier()
    if comm.is_distributed():
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
