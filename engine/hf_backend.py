"""Real-model backend: HuggingFace transformers (e.g. Qwen/Qwen2.5-Coder-1.5B-Instruct).

What is real here (nothing is simulated):
  * prefix KV reuse  - the system block (instructions + semantic-model YAML) is prefilled once, its
                       DynamicCache is kept in an LRU (byte-budgeted) and deep-copied per request;
  * grammar-constrained decoding - logits are restricted to the TokenFSM of the compiled semantic
                       model at every position, so output is *guaranteed* to be in the grammar;
  * N-gram speculative decoding - drafts (grammar jump-forward + SQL n-gram) are verified in one
                       forward pass with exact greedy matching, KV is cropped to the accepted length.

Requires `pip install -r requirements-real.txt` (torch, transformers). One request is decoded at a
time per process (HF eager generation has no continuous batching); scale out by running more
workers - that is precisely what the router is for. For batched serving use vLLM.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import os
import threading
import time
from collections import OrderedDict
from typing import AsyncIterator

import tokenizer as tk
from backends import GenRequest
from semantic_compiler import CompiledModel, SemanticModelError, compile_model
from speculative_sql import BOILERPLATE, NGramSQLDrafter

SYSTEM_HEAD = ("You are ArcticFlow Analyst, a text-to-SQL engine. Generate exactly one read-only DuckDB "
               "SELECT statement grounded strictly in the semantic model below. Output SQL only.\n\n"
               "### SEMANTIC MODEL\n")


class HFBackend:
    name = "hf"

    def __init__(self, model_id: str | None = None, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model_id or os.environ.get("ARCTIC_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=dtype).to(self.device).eval()
        cfg = self.model.config
        heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        self.kv_bpt = 2 * cfg.num_hidden_layers * heads * (cfg.hidden_size // cfg.num_attention_heads) * (2 if dtype == torch.bfloat16 else 4)
        self.spec_k = int(os.environ.get("ARCTIC_SPEC_K", "8"))
        self._budget = int(float(os.environ.get("ARCTIC_PREFIX_CACHE_GB", "2")) * (1 << 30))
        self._prefix: OrderedDict[str, tuple[list[int], object]] = OrderedDict()
        self._prefix_bytes = 0
        self._gpu = threading.Lock()
        self._depth = 0
        self._vocab: list[str | None] | None = None
        self._fsm_cache: dict[str, object] = {}
        self._base: dict[str, NGramSQLDrafter] = {}
        self._idx: dict[tuple[str, int], object] = {}
        self.hit_tokens = self.lookup_tokens = 0
        self.spec_proposed = self.spec_accepted = 0
        eos = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.eos_id = eos if isinstance(eos, int) and eos >= 0 else self.tok.eos_token_id

    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    @property
    def queue_depth(self) -> int:
        return self._depth

    # -- grammar / vocab ---------------------------------------------------------------------------
    def _vocab_strings(self) -> list[str | None]:
        if self._vocab is None:
            special = set(self.tok.all_special_ids)
            v: list[str | None] = []
            for i in range(len(self.tok)):
                s = None if i in special else self.tok.decode([i], clean_up_tokenization_spaces=False)
                v.append(s if s and "�" not in s and all(32 <= ord(c) < 127 for c in s) else None)
            self._vocab = v
        return self._vocab

    def _fsm(self, model: CompiledModel):
        f = self._fsm_cache.get(model.digest)
        if f is None:
            f = self._fsm_cache[model.digest] = model.token_fsm(self._vocab_strings(), self.eos_id)
        return f

    def _drafter(self, model: CompiledModel) -> NGramSQLDrafter:
        base = self._base.get(model.digest)
        if base is None:
            base = NGramSQLDrafter(k=self.spec_k)
            for text in (*BOILERPLATE, model.raw, *[v.sql for v in model.verified_queries]):
                base.index(self.tok.encode(text, add_special_tokens=False))
            self._base[model.digest] = base
        return NGramSQLDrafter(k=self.spec_k, base=base)

    # -- prefix KV cache ---------------------------------------------------------------------------
    def _render(self, req: GenRequest) -> tuple[str, str]:
        system = SYSTEM_HEAD + req.semantic_model
        msgs = [{"role": "system", "content": system}] + [
            {"role": m.get("role", "user"), "content": m["content"]} for m in req.messages]
        full = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        head = self.tok.apply_chat_template(msgs[:1], tokenize=False)
        if not full.startswith(head):
            return "", full
        return head, full[len(head):]

    def _get_prefix(self, head: str):
        """Returns (ids, cache_copy, hit). Caller holds the GPU lock."""
        torch = self.torch
        key = hashlib.sha256(head.encode()).hexdigest()
        hit = self._prefix.get(key)
        if hit is not None:
            self._prefix.move_to_end(key)
            return hit[0], copy.deepcopy(hit[1]), True
        ids = self.tok(head, add_special_tokens=False).input_ids
        from transformers import DynamicCache
        cache = DynamicCache()
        with torch.no_grad():
            self.model(input_ids=torch.tensor([ids], device=self.device), past_key_values=cache, use_cache=True)
        size = len(ids) * self.kv_bpt
        while self._prefix and self._prefix_bytes + size > self._budget:
            _, (old_ids, _) = self._prefix.popitem(last=False)
            self._prefix_bytes -= len(old_ids) * self.kv_bpt
        self._prefix[key] = (ids, cache)
        self._prefix_bytes += size
        return ids, copy.deepcopy(cache), False

    # -- decoding ----------------------------------------------------------------------------------
    def _pick(self, logits, fsm, state: int, model_digest: str) -> int:
        torch = self.torch
        allowed = fsm.allowed(state)
        if not allowed:
            raise RuntimeError("grammar dead end")
        k = (model_digest, state)
        idx = self._idx.get(k)
        if idx is None:
            idx = self._idx[k] = torch.tensor(allowed, device=logits.device)
        return allowed[int(torch.argmax(logits[idx]))]

    def _run(self, req: GenRequest, model: CompiledModel, emit) -> None:
        torch = self.torch
        fsm = self._fsm(model)
        vocab = self._vocab_strings()
        head, tail = self._render(req)
        t0 = time.monotonic()
        with self._gpu, torch.no_grad():
            if head:
                prefix_ids, cache, hit = self._get_prefix(head)
            else:
                from transformers import DynamicCache
                prefix_ids, cache, hit = [], DynamicCache(), False
            suffix = self.tok(tail, add_special_tokens=False).input_ids
            total = len(prefix_ids) + len(suffix)
            cached = len(prefix_ids) if hit else 0
            self.hit_tokens += cached
            self.lookup_tokens += total
            emit({"type": "start", "request_id": req.request_id, "prompt_tokens": total, "cached_tokens": cached,
                  "queue_ms": round((time.monotonic() - t0) * 1000, 3)})
            out = self.model(input_ids=torch.tensor([suffix], device=self.device), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            cache_len = total
            state = fsm.start
            gen: list[int] = []
            drafter = self._drafter(model)
            proposed = accepted = steps = jump = 0

            def push(tids: list[int]) -> bool:
                nonlocal state
                done = False
                text = ""
                for t in tids:
                    if t == self.eos_id:
                        done = True
                        break
                    gen.append(t)
                    text += vocab[t] or ""
                    state = fsm.advance(state, t)
                if text:
                    emit({"type": "token", "text": text, "n": len(tids)})
                return done

            first = self._pick(out.logits[0, -1], fsm, state, model.digest)
            t_first = time.monotonic()
            finished = push([first])
            pending = first
            while not finished and len(gen) < req.max_tokens:
                s = state
                draft: list[int] = []
                n_forced = 0
                while len(draft) < self.spec_k:                      # 1) grammar jump-forward
                    ft = fsm.forced_token(s)
                    if ft is None:
                        break
                    n_forced += 1
                    draft.append(ft)
                    s = fsm.advance(s, ft)
                if len(draft) < self.spec_k:                         # 2) SQL n-gram
                    for t in drafter.propose(prefix_ids[-4:] + suffix[-4:] + gen + draft, self.spec_k - len(draft)):
                        if t == self.eos_id or t not in fsm.allowed_set(s):
                            break
                        draft.append(t)
                        s = fsm.advance(s, t)
                inp = torch.tensor([[pending] + draft], device=self.device)
                out = self.model(input_ids=inp, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                logits = out.logits[0]
                emitted: list[int] = []
                s, acc = state, 0
                for i in range(len(draft) + 1):
                    tid = self._pick(logits[i], fsm, s, model.digest)
                    emitted.append(tid)
                    if tid == self.eos_id:
                        break
                    s = fsm.advance(s, tid)
                    if i < len(draft) and tid == draft[i]:
                        acc += 1
                        continue
                    break
                cache_len += 1 + acc
                cache.crop(cache_len)
                jump += min(n_forced, acc)
                proposed += len(draft)
                accepted += acc
                steps += 1
                drafter.observe(gen[-4:], emitted)
                finished = push(emitted)
                pending = emitted[-1]
            self.spec_proposed += proposed
            self.spec_accepted += accepted
            sql = "".join(vocab[t] or "" for t in gen)
            n_all = len(gen)
            emit({"type": "done", "sql": sql, "finish_reason": "stop" if finished else "length",
                  "usage": {"prompt_tokens": total, "cached_tokens": cached, "completion_tokens": n_all,
                            "spec_proposed": proposed, "spec_accepted": accepted, "spec_steps": steps + 1, "grammar_jump_tokens": jump,
                            "ttft_ms": round((t_first - t0) * 1000, 3), "e2e_ms": round((time.monotonic() - t0) * 1000, 3)}})

    async def generate(self, req: GenRequest) -> AsyncIterator[dict]:
        try:
            model = await asyncio.to_thread(compile_model, req.semantic_model)
        except SemanticModelError as e:
            yield {"type": "error", "message": str(e)}
            return
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def emit(ev: dict) -> None:
            loop.call_soon_threadsafe(q.put_nowait, ev)

        def work() -> None:
            try:
                self._run(req, model, emit)
            except Exception as e:  # noqa: BLE001 - surfaced to the client as an error event
                emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
            finally:
                emit({"type": "_eof"})

        self._depth += 1
        fut = loop.run_in_executor(None, work)
        try:
            while True:
                ev = await q.get()
                if ev["type"] == "_eof":
                    break
                yield ev
        finally:
            self._depth -= 1
            await fut

    def stats(self) -> dict:
        return {"backend": "hf", "model": self.model_id, "device": self.device, "queue_depth": self._depth,
                "prefix_cache_entries": len(self._prefix), "prefix_cache_bytes": self._prefix_bytes,
                "cache_hit_ratio": round(self.hit_tokens / self.lookup_tokens, 4) if self.lookup_tokens else 0.0,
                "spec_acceptance_rate": round(self.spec_accepted / self.spec_proposed, 4) if self.spec_proposed else 0.0}
