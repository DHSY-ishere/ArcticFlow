"""Inference backends. Every backend exposes

    async generate(req: GenRequest) -> AsyncIterator[dict]   # start / token* / done | error
    stats() -> dict
    queue_depth -> int

so the FastAPI worker is backend-agnostic. `SimBackend` is the CPU latency simulator; the real
HuggingFace backend lives in hf_backend.py (needs torch + transformers).
"""
from __future__ import annotations

import asyncio
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import AsyncIterator

import tokenizer as tk
from kv_blocks import KVBlockManager
from latency_model import LatencyModel, ModelProfile
from semantic_compiler import CompiledModel, SemanticModelError, compile_model
from sim_engine import SimEngine, SimRequest
from speculative_sql import NGramSQLDrafter, speculative_plan
from sql_planner import PlanError, plan_sql


@dataclass
class GenRequest:
    request_id: str
    semantic_model: str
    messages: list[dict]
    max_tokens: int = 256
    prompt: str | None = None
    prompt_tokens: list[int] | None = None
    prompt_len: int | None = None            # router-supplied (skips tokenizing in the worker)
    block_hashes: list[int] | None = None    # router-supplied chained FNV-1a block hashes
    execute: bool = False
    session_id: str | None = None
    extra: dict = field(default_factory=dict)

    def user_questions(self) -> list[str]:
        return [m["content"] for m in self.messages if m.get("role", "user") != "assistant"]

    def tokens(self) -> list[int]:
        if self.prompt_tokens:
            return self.prompt_tokens
        return tk.encode(self.prompt or tk.build_prompt(self.semantic_model, self.messages))


class _PlanArtifacts:
    __slots__ = ("sql", "pieces", "steps", "proposed", "accepted", "jump")

    def __init__(self, sql, pieces, steps, proposed, accepted, jump):
        self.sql, self.pieces, self.steps = sql, pieces, steps
        self.proposed, self.accepted, self.jump = proposed, accepted, jump


class SimBackend:
    name = "sim"

    def __init__(self, hw: str | None = None, kv_blocks: int | None = None, max_batch: int | None = None,
                 time_scale: float | None = None, spec_k: int = 8):
        env = os.environ.get
        self.lat = LatencyModel(hw or env("ARCTIC_SIM_HW", "a100"), ModelProfile())
        self.kv = KVBlockManager(kv_blocks or int(env("ARCTIC_KV_BLOCKS", "4096")), tk.BLOCK_SIZE,
                                 self.lat.m.kv_bytes_per_token)
        self.engine = SimEngine(self.lat, self.kv, max_batch or int(env("ARCTIC_MAX_BATCH", "128")),
                                int(env("ARCTIC_TOKEN_BUDGET", "8192")),
                                time_scale if time_scale is not None else float(env("ARCTIC_SIM_TIME_SCALE", "1.0")))
        self.spec_k = spec_k
        self._plans: OrderedDict[tuple, _PlanArtifacts] = OrderedDict()
        self._bases: dict[str, NGramSQLDrafter] = {}
        self.spec_proposed = 0
        self.spec_accepted = 0

    async def start(self) -> None:
        self.engine.start()

    async def stop(self) -> None:
        await self.engine.stop()

    @property
    def queue_depth(self) -> int:
        return self.engine.queue_depth

    def _base_drafter(self, model: CompiledModel) -> NGramSQLDrafter:
        b = self._bases.get(model.digest)
        if b is None:
            if len(self._bases) > 64:
                self._bases.pop(next(iter(self._bases)))
            b = NGramSQLDrafter(k=self.spec_k)
            b.prime(tk.encode(model.raw), [v.sql for v in model.verified_queries])
            self._bases[model.digest] = b
        return b

    def _artifacts(self, model: CompiledModel, questions: list[str], max_tokens: int) -> _PlanArtifacts:
        key = (model.digest, tuple(questions), max_tokens)
        hit = self._plans.get(key)
        if hit is not None:
            self._plans.move_to_end(key)
            return hit
        plan = plan_sql(model, questions)
        if not model.accepts(plan.sql):       # the grammar is the contract; never emit outside it
            raise PlanError("planned SQL violates the compiled grammar")
        pieces = tk.pieces(plan.sql)[:max_tokens]
        ids = [tk.piece_id(p) for p in pieces]
        drafter = NGramSQLDrafter(k=self.spec_k, base=self._base_drafter(model))
        steps, st = speculative_plan(ids, pieces, 1, drafter, model.dfa)
        art = _PlanArtifacts(plan.sql, pieces, steps, st.proposed, st.accepted, st.jump_tokens)
        self._plans[key] = art
        if len(self._plans) > 20000:
            self._plans.popitem(last=False)
        return art

    async def generate(self, req: GenRequest) -> AsyncIterator[dict]:
        try:
            model = await asyncio.to_thread(compile_model, req.semantic_model)
            art = self._artifacts(model, req.user_questions(), req.max_tokens)
        except (SemanticModelError, PlanError) as e:
            yield {"type": "error", "message": str(e)}
            return
        if req.block_hashes is not None and req.prompt_len:
            plen, hashes = req.prompt_len, req.block_hashes
        else:
            tokens = req.tokens()
            plen, hashes = len(tokens), tk.hash_blocks(tokens)
        sr = SimRequest(req.request_id, plen, hashes, art.pieces, art.steps,
                        art.proposed, art.accepted)
        self.engine.submit(sr)
        info: dict = {}
        try:
            while True:
                kind, *rest = await sr.events.get()
                if kind == "start":
                    info = rest[0]
                    yield {"type": "start", "request_id": req.request_id, **info}
                elif kind == "tok":
                    yield {"type": "token", "text": rest[0], "n": rest[1]}
                elif kind == "end":
                    self.spec_proposed += art.proposed
                    self.spec_accepted += art.accepted
                    yield {"type": "done", "sql": art.sql, "finish_reason": "stop",
                           "usage": {"prompt_tokens": sr.prompt_len, "cached_tokens": sr.cached_tokens,
                                     "completion_tokens": sr.n_out, "spec_proposed": art.proposed,
                                     "spec_accepted": art.accepted, "spec_steps": len(art.steps) + 1,
                                     "grammar_jump_tokens": art.jump, **rest[0]}}
                    return
                elif kind == "error":
                    yield {"type": "error", "message": rest[0]}
                    return
        finally:
            self.engine.cancel(sr)     # no-op if finished; frees KV if the client disconnected

    def stats(self) -> dict:
        s = self.engine.stats()
        s["backend"] = "sim"
        s["hardware"] = self.lat.hw.name
        s["spec_acceptance_rate"] = round(self.spec_accepted / self.spec_proposed, 4) if self.spec_proposed else 0.0
        return s
