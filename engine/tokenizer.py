"""Deterministic pseudo-tokenizer + block hashing, byte-compatible with router/tokenizer.go
and router/radix_cache.go. Router and worker therefore agree on token counts and on the
chained FNV-1a block hashes that identify KV pages."""
from __future__ import annotations

import re

VOCAB_SIZE = 151643
BLOCK_SIZE = 16

_PIECE = re.compile(r"[ \t\n\f\r]?[A-Za-z0-9_]+|[ \t\n\f\r]?[^ \t\n\f\r A-Za-z0-9_]|[ \t\n\f\r]+")

_FNV32_OFFSET, _FNV32_PRIME = 0x811C9DC5, 0x01000193
_FNV64_OFFSET, _FNV64_PRIME = 14695981039346656037, 1099511628211
_M64 = (1 << 64) - 1


def pieces(text: str) -> list[str]:
    return _PIECE.findall(text)


def _fnv32(s: str) -> int:
    h = _FNV32_OFFSET
    for b in s.encode("utf-8"):
        h = ((h ^ b) * _FNV32_PRIME) & 0xFFFFFFFF
    return h


_ID_CACHE: dict[str, int] = {}


def piece_id(p: str) -> int:
    v = _ID_CACHE.get(p)
    if v is None:
        v = _fnv32(p) % VOCAB_SIZE
        if len(_ID_CACHE) < 500_000:
            _ID_CACHE[p] = v
    return v


def encode(text: str) -> list[int]:
    return [piece_id(p) for p in _PIECE.findall(text)]


def hash_blocks(tokens: list[int], block_size: int = BLOCK_SIZE) -> list[int]:
    """Chained FNV-1a 64 over full blocks (identical to router.HashBlocks)."""
    out: list[int] = []
    h = _FNV64_OFFSET
    for i in range(len(tokens) // block_size):
        for t in tokens[i * block_size:(i + 1) * block_size]:
            for shift in (0, 8, 16, 24):
                h = ((h ^ ((t >> shift) & 0xFF)) * _FNV64_PRIME) & _M64
        out.append(h)
    return out


PREAMBLE = ("You are ArcticFlow Analyst, a text-to-SQL engine. Generate exactly one read-only DuckDB "
            "SELECT statement grounded strictly in the semantic model below.\n\n### SEMANTIC MODEL\n")
CONVO = "\n### CONVERSATION\n"


def build_prompt(semantic_model: str, messages: list[dict]) -> str:
    """Mirror of router.BuildPrompt (used only when a client hits a worker directly)."""
    parts = [PREAMBLE, semantic_model]
    if not semantic_model.endswith("\n"):
        parts.append("\n")
    parts.append(CONVO)
    for m in messages:
        parts.append(("ASSISTANT: " if m.get("role") == "assistant" else "USER: ") + m["content"] + "\n")
    parts.append("ASSISTANT:")
    return "".join(parts)
