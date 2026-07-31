#!/usr/bin/env python3
"""Measure bf16 decoder-LM training throughput for Phase M capacity planning.

The default 24-layer, width-1280, SwiGLU model has approximately 0.52B
trainable parameters (including a tied 32k-token embedding).  It is a
synthetic decoder-only workload, not a quality or convergence experiment:
random token IDs, sequence packing, dataloader cost, checkpoint I/O, and
distributed overhead are deliberately excluded.  Activation checkpointing is
enabled by default so the model fits a 16 GB GPU; it increases the measured
forward/backward compute to reflect a practical memory-constrained run.

Run from the Linux-resident WSL clone, not /mnt/c:
  python scripts/52_probe_lm_budget.py --out out/probe_lm_budget.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


class Block(nn.Module):
    """Pre-norm causal attention plus SwiGLU MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int, n_heads: int) -> None:
        super().__init__()
        if hidden_size % n_heads:
            raise ValueError("hidden_size must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.attn_norm = nn.RMSNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_norm = nn.RMSNorm(hidden_size)
        self.gate_up = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        q, k, v = self.qkv(self.attn_norm(x)).chunk(3, dim=-1)
        batch, seq_len, _ = q.shape
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch, seq_len, -1)
        x = residual + self.proj(attended)
        residual = x
        gate, up = self.gate_up(self.mlp_norm(x)).chunk(2, dim=-1)
        return residual + self.down(F.silu(gate) * up)


class ProbeLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        n_layers: int,
        n_heads: int,
        activation_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList(
            Block(hidden_size, intermediate_size, n_heads) for _ in range(n_layers)
        )
        self.norm = nn.RMSNorm(hidden_size)
        self.activation_checkpointing = activation_checkpointing

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            if self.activation_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        # Tied output weights avoid counting a second 32k x hidden matrix.
        return F.linear(self.norm(x), self.embed.weight)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("out/probe_lm_budget.json"))
    ap.add_argument("--vocab-size", type=int, default=32_000)
    ap.add_argument("--hidden-size", type=int, default=1280)
    ap.add_argument("--intermediate-size", type=int, default=3456)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--heads", type=int, default=20)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--micro-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-steps", type=int, default=2)
    ap.add_argument("--measure-steps", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-activation-checkpointing", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; run this probe in the WSL CUDA environment.")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("The selected CUDA device does not support bf16.")

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    model = ProbeLM(
        args.vocab_size, args.hidden_size, args.intermediate_size, args.layers, args.heads,
        not args.no_activation_checkpointing,
    ).to(device=device, dtype=torch.bfloat16)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
    param_count = sum(parameter.numel() for parameter in model.parameters())
    tokens_per_optimizer_step = args.micro_batch_size * args.seq_len * args.grad_accum

    def run_optimizer_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            input_ids = torch.randint(
                args.vocab_size, (args.micro_batch_size, args.seq_len), device=device
            )
            labels = torch.roll(input_ids, shifts=-1, dims=1)
            logits = model(input_ids)
            F.cross_entropy(logits[:, :-1].flatten(0, 1), labels[:, :-1].flatten()).div_(
                args.grad_accum
            ).backward()
        optimizer.step()

    for _ in range(args.warmup_steps):
        run_optimizer_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(args.measure_steps):
        run_optimizer_step()
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start

    result = {
        "status": "measured",
        "measurement_scope": "synthetic single-GPU bf16 forward+backward+AdamW step",
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "parameters": param_count,
        "parameters_billions": param_count / 1e9,
        "precision": "bf16",
        "model_assumptions": {
            "architecture": "decoder-only, MHA, SwiGLU, tied embedding/output",
            "activation_checkpointing": not args.no_activation_checkpointing,
            "synthetic_data": True,
            "excluded_overheads": [
                "dataloader",
                "tokenization",
                "checkpoint I/O",
                "distributed communication",
            ],
        },
        "sequence_length": args.seq_len,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "tokens_per_optimizer_step": tokens_per_optimizer_step,
        "warmup_steps": args.warmup_steps,
        "measured_optimizer_steps": args.measure_steps,
        "measured_elapsed_seconds": elapsed_s,
        "measured_tokens_per_second": (tokens_per_optimizer_step * args.measure_steps) / elapsed_s,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
