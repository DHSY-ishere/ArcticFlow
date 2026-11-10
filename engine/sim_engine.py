"""Deterministic continuous-batching inference simulator (CPU only).

One asyncio task plays the role of the GPU. Every iteration it
  1. admits waiting requests while the KV block manager has room and the batch has a free slot
     (prefix-cache lookup happens here: cached prompt tokens skip prefill),
  2. forms a step: one decode (or speculative-verify) slot per decoding sequence + chunked-prefill
     tokens for admitted sequences up to `token_budget`,
  3. sleeps for the roofline step time (latency_model.LatencyModel) x `time_scale`,
  4. emits tokens: a sequence whose prefill finished emits its first token at the end of that step;
     decoding sequences emit 1 + accepted-draft tokens (replayed from the speculation plan).

Because the loop runs once per *step* (not per token or per request) a single Python process can
serve >1k requests/s of simulated traffic. Wall-clock latencies observed by clients are the modelled
GPU latencies divided by `time_scale` (default 1.0 = real time).
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field

from kv_blocks import Allocation, KVBlockManager
from latency_model import LatencyModel


@dataclass
class SimRequest:
    request_id: str
    prompt_len: int
    hashes: list[int]
    pieces: list[str]                   # surface text of each output token
    steps: list[tuple[int, int]]        # speculation plan for tokens[1:]: (emitted, draft_len)
    spec_proposed: int = 0
    spec_accepted: int = 0
    events: asyncio.Queue = field(default_factory=asyncio.Queue)
    # runtime
    t_submit: float = 0.0
    t_first: float = 0.0
    alloc: Allocation | None = None
    cached_tokens: int = 0
    prefill_left: int = 0
    ctx: int = 0
    si: int = 0          # next step index
    pos: int = 1         # next output token index (token 0 comes from prefill)
    dead: bool = False   # cancelled by the client; resources already released

    @property
    def n_out(self) -> int:
        return len(self.pieces)


class SimEngine:
    def __init__(self, latency: LatencyModel, kv: KVBlockManager, max_batch: int = 128,
                 token_budget: int = 8192, time_scale: float = 1.0):
        self.lat, self.kv = latency, kv
        self.max_batch, self.token_budget, self.time_scale = max_batch, token_budget, time_scale
        self.waiting: deque[SimRequest] = deque()
        self.running: list[SimRequest] = []
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        # counters
        self.prefill_tokens = 0
        self.cached_tokens = 0
        self.decode_steps = 0
        self.completed = 0
        self.busy_s = 0.0
        self.flops_saved = 0.0
        self.flops_spent = 0.0

    # -- lifecycle ---------------------------------------------------------------------------------
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._run(), name="sim-engine")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    @property
    def queue_depth(self) -> int:
        return len(self.waiting) + len(self.running)

    def submit(self, r: SimRequest) -> None:
        r.t_submit = time.monotonic()
        if (r.prompt_len + r.n_out) > self.kv.total * self.kv.bs:
            r.events.put_nowait(("error", "request exceeds total KV capacity"))
            return
        self.waiting.append(r)
        self._wake.set()

    def cancel(self, r: SimRequest) -> None:
        """Client went away: free its resources immediately."""
        if r.dead:
            return
        r.dead = True
        if r in self.waiting:
            self.waiting.remove(r)
        if r in self.running:
            self.running.remove(r)
            if r.alloc:
                self.kv.release(r.alloc)

    # -- main loop ---------------------------------------------------------------------------------
    def _admit(self) -> None:
        while self.waiting and len(self.running) < self.max_batch:
            r = self.waiting[0]
            alloc = self.kv.admit(r.hashes, r.prompt_len, r.n_out)
            if alloc is None:
                break
            self.waiting.popleft()
            r.alloc = alloc
            r.cached_tokens = min(len(alloc.cached_blocks) * self.kv.bs, r.prompt_len - 1)  # last token always recomputed
            r.prefill_left = r.prompt_len - r.cached_tokens
            r.ctx = r.cached_tokens
            self.cached_tokens += r.cached_tokens
            self.flops_saved += self.lat.prefill_flops(r.cached_tokens, 0)
            self.running.append(r)
            r.events.put_nowait(("start", {"prompt_tokens": r.prompt_len, "cached_tokens": r.cached_tokens,
                                           "queue_ms": round((time.monotonic() - r.t_submit) * 1000, 3)}))

    async def _run(self) -> None:
        lat = self.lat
        while True:
            if not self.running and not self.waiting:
                self._wake.clear()
                await self._wake.wait()
                continue
            self._admit()
            if not self.running:
                await asyncio.sleep(0.001)      # waiting on KV room; blocked until something frees
                continue

            decoding = [r for r in self.running if r.prefill_left == 0]
            dec_tokens = sum(1 + r.steps[r.si][1] if r.si < len(r.steps) else 1 for r in decoding)
            budget = max(0, self.token_budget - dec_tokens)
            chunks: list[tuple[SimRequest, int]] = []
            for r in self.running:
                if r.prefill_left > 0 and budget > 0:
                    n = min(r.prefill_left, budget)
                    chunks.append((r, n))
                    budget -= n
            dt = lat.step_time([(n, r.ctx) for r, n in chunks], [r.ctx for r in decoding], dec_tokens)
            self.busy_s += dt
            self.flops_spent += sum(lat.prefill_flops(n, r.ctx) for r, n in chunks)
            if dt * self.time_scale > 0.0002:
                await asyncio.sleep(dt * self.time_scale)
            else:
                await asyncio.sleep(0)
            self.decode_steps += 1
            now = time.monotonic()

            for r, n in chunks:
                if r.dead:
                    continue
                r.prefill_left -= n
                r.ctx += n
                self.prefill_tokens += n
                if r.prefill_left == 0:
                    self.kv.commit_prefill(r.alloc)       # type: ignore[arg-type]
                    r.t_first = now
                    r.events.put_nowait(("tok", r.pieces[0], 1))
                    if r.n_out == 1:
                        self._finish(r)
            for r in decoding:
                if r.dead:
                    continue
                if r.si >= len(r.steps):
                    self._finish(r)
                    continue
                emitted, _ = r.steps[r.si]
                r.si += 1
                txt = "".join(r.pieces[r.pos:r.pos + emitted])
                r.pos += emitted
                r.ctx += emitted
                r.events.put_nowait(("tok", txt, emitted))
                if r.pos >= r.n_out:
                    self._finish(r)

    def _finish(self, r: SimRequest) -> None:
        if r in self.running:
            self.running.remove(r)
        if r.alloc:
            self.kv.release(r.alloc)
        self.completed += 1
        r.events.put_nowait(("end", {"ttft_ms": round((r.t_first - r.t_submit) * 1000, 3),
                                     "e2e_ms": round((time.monotonic() - r.t_submit) * 1000, 3)}))

    def stats(self) -> dict:
        tot = self.prefill_tokens + self.cached_tokens
        return {"queue_depth": self.queue_depth, "running": len(self.running), "waiting": len(self.waiting),
                "completed": self.completed, "prefill_tokens_computed": self.prefill_tokens,
                "prompt_tokens_cached": self.cached_tokens,
                "cache_hit_ratio": round(self.cached_tokens / tot, 4) if tot else 0.0,
                "prefill_tflops_saved": round(self.flops_saved / 1e12, 3),
                "prefill_tflops_spent": round(self.flops_spent / 1e12, 3),
                "gpu_busy_s": round(self.busy_s, 3), "kv": self.kv.stats()}
