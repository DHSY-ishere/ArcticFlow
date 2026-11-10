"""Roofline latency model for one continuous-batching engine step.

A step processes some prefill chunks (compute-bound) and some decode tokens (memory-bound).

  FLOPs  = 2*P*n_tok                              (dense matmuls, every token)
         + 4*L*H*sum_i n_i*(ctx_i + n_i/2)        (attention scores+values; causal => n/2)
  Bytes  = weights (read once per step) + KV read (ctx_i * kv_bytes_per_token for every sequence
           attending) + KV write (n_tok * kv_bytes_per_token)
  t_step = max(FLOPs / (peak*MFU), Bytes / (BW*MBU)) + launch_overhead

Prefill of N new tokens is huge-FLOPs/small-bytes => compute bound; a decode step of B sequences is
tiny-FLOPs/weights-bytes => bandwidth bound. Prefix-cache hits delete prefill FLOPs but not the KV
read bytes, which is exactly the effect this model must preserve. The Go twin
(benchmark/sim.go) uses the same constants.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    name: str = "qwen2.5-coder-1.5b"
    params: float = 1.54e9
    layers: int = 28
    hidden: int = 1536
    kv_heads: int = 2
    head_dim: int = 128
    dtype_bytes: int = 2

    @property
    def kv_bytes_per_token(self) -> int:
        return 2 * self.layers * self.kv_heads * self.head_dim * self.dtype_bytes   # K and V

    @property
    def weight_bytes(self) -> float:
        return self.params * self.dtype_bytes


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    peak_tflops: float     # dense bf16
    mem_bw_gbs: float
    mfu: float = 0.45
    mbu: float = 0.70
    overhead_ms: float = 0.6   # kernel launch + scheduler per step


HARDWARE = {
    "h100": HardwareProfile("h100", 989.0, 3350.0),
    "a100": HardwareProfile("a100", 312.0, 2039.0),
    "a10g": HardwareProfile("a10g", 125.0, 600.0, overhead_ms=0.8),
    "l4": HardwareProfile("l4", 121.0, 300.0, overhead_ms=0.8),
}


class LatencyModel:
    def __init__(self, hw: HardwareProfile | str = "a100", model: ModelProfile | None = None):
        self.hw = HARDWARE[hw] if isinstance(hw, str) else hw
        self.m = model or ModelProfile()
        self._flops_rate = self.hw.peak_tflops * 1e12 * self.hw.mfu
        self._bytes_rate = self.hw.mem_bw_gbs * 1e9 * self.hw.mbu

    def prefill_flops(self, n_new: int, n_ctx: int) -> float:
        """FLOPs to prefill n_new tokens on top of n_ctx already-cached tokens."""
        return 2 * self.m.params * n_new + 4 * self.m.layers * self.m.hidden * n_new * (n_ctx + n_new / 2)

    def step_time(self, prefill: list[tuple[int, int]], decode_ctx: list[int], decode_tokens: int | None = None) -> float:
        """Seconds for one step.
        prefill: [(n_new_tokens, ctx_before)], decode_ctx: context length of each decoding seq,
        decode_tokens: total tokens verified this step (>= len(decode_ctx); speculation raises it)."""
        d_tok = len(decode_ctx) if decode_tokens is None else decode_tokens
        n_tok = d_tok + sum(n for n, _ in prefill)
        if n_tok == 0:
            return 0.0
        m = self.m
        flops = 2 * m.params * n_tok
        for n, ctx in prefill:
            flops += 4 * m.layers * m.hidden * n * (ctx + n / 2)
        per = (d_tok / len(decode_ctx)) if decode_ctx else 0
        for ctx in decode_ctx:
            flops += 4 * m.layers * m.hidden * per * (ctx + per / 2)
        kv_read = sum(ctx for _, ctx in prefill) + sum(decode_ctx)
        bytes_ = m.weight_bytes + (kv_read + n_tok) * m.kv_bytes_per_token
        return max(flops / self._flops_rate, bytes_ / self._bytes_rate) + self.hw.overhead_ms / 1000.0
