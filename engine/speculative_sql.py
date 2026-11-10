"""N-gram speculative decoding tailored to SQL.

SQL generated from a semantic model is extremely repetitive: keywords (SELECT/FROM/GROUP BY/
ORDER BY/JOIN ... ON), qualified identifiers (`orders.amount`) and join predicates all already
appear in the prompt (the YAML) and in the output so far. A *prompt-lookup* style n-gram drafter
therefore proposes long, high-acceptance drafts at zero model cost; the target model verifies the
whole draft in ONE forward pass (cost ~ one decode step, since decode is bandwidth-bound).

Draft sources, in priority order:
  1. grammar jump-forward  - tokens forced by the DFA/FSM (certain; acceptance == 1.0)
  2. n-gram table over [SQL boilerplate corpus + verified queries + prompt + generated-so-far],
     longest-suffix-first (n = n_max..1), most-frequent continuation, chained for up to k tokens.
Verification is exact greedy-match, so output is identical to non-speculative greedy decoding.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import tokenizer as tk
from regex_fsm import DFA

# Structural skeletons of the SQL shapes the grammar admits. They prime the table so the very
# first SELECT already drafts `FROM` / `GROUP BY` / `ORDER BY ... DESC LIMIT` boilerplate.
BOILERPLATE = [
    "SELECT a.b, SUM(a.c) AS total FROM a JOIN b ON a.x = b.x GROUP BY a.b ORDER BY total DESC LIMIT 10",
    "SELECT a.b, COUNT(a.c) AS n FROM a JOIN b ON a.x = b.x JOIN c ON a.y = c.y WHERE a.d = 'v' AND YEAR(a.e) = 2024 GROUP BY a.b ORDER BY n DESC LIMIT 100",
    "SELECT DATE_TRUNC('month', a.d) AS d_month, AVG(a.c) AS avg_c FROM a WHERE a.e = 'v' GROUP BY d_month ORDER BY d_month ASC",
    "SELECT COUNT(*) AS row_count FROM a WHERE a.b > 100 ORDER BY row_count DESC LIMIT 1000",
]


@dataclass
class SpecStats:
    proposed: int = 0
    accepted: int = 0
    steps: int = 0
    emitted: int = 0
    jump_tokens: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_step(self) -> float:
        return self.emitted / self.steps if self.steps else 0.0


class NGramSQLDrafter:
    def __init__(self, n_max: int = 4, k: int = 8, min_count: int = 1, base: "NGramSQLDrafter | None" = None):
        # `base` is a shared read-only table (per semantic model: YAML + verified queries + boilerplate);
        # this instance only stores the per-request overlay, so creating one is O(1).
        self.n_max, self.k, self.min_count, self.base = n_max, k, min_count, base
        self.table: list[dict[tuple, Counter]] = [defaultdict(Counter) for _ in range(n_max + 1)]
        self._seq: list[int] = []

    def index(self, tokens: list[int]) -> None:
        """Add a token sequence to the n-gram table."""
        for n in range(1, self.n_max + 1):
            t = self.table[n]
            for i in range(len(tokens) - n):
                t[tuple(tokens[i:i + n])][tokens[i + n]] += 1

    def prime(self, prompt_tokens: list[int], corpus: list[str] = ()) -> None:
        for text in (*BOILERPLATE, *corpus):
            self.index(tk.encode(text))
        self.index(prompt_tokens)

    def observe(self, history: list[int], new_tokens: list[int]) -> None:
        """Learn from accepted output (so repeated sub-expressions draft well)."""
        tail = history[-self.n_max:]
        seq = tail + new_tokens
        for n in range(1, self.n_max + 1):
            for j in range(len(tail), len(seq)):
                if j - n >= 0:
                    self.table[n][tuple(seq[j - n:j])][seq[j]] += 1

    def propose(self, history: list[int], limit: int | None = None) -> list[int]:
        k = self.k if limit is None else min(self.k, limit)
        draft: list[int] = []
        ctx = list(history[-self.n_max:])
        while len(draft) < k:
            nxt = None
            for n in range(min(self.n_max, len(ctx)), 0, -1):
                key = tuple(ctx[-n:])
                c = self.table[n].get(key)
                if self.base is not None:
                    b = self.base.table[n].get(key)
                    if b:
                        c = (c + b) if c else b
                if c:
                    tok, cnt = c.most_common(1)[0]
                    if cnt >= self.min_count:
                        nxt = tok
                        break
            if nxt is None:
                break
            draft.append(nxt)
            ctx.append(nxt)
        return draft


def jump_forward(dfa: DFA, state: int) -> list[int]:
    """Grammar-forced continuation as complete pieces. The trailing piece is dropped if it touches the
    end of the forced text, because the next (unforced) character could still extend it."""
    text = dfa.forced_text(state)
    if not text:
        return []
    ps = tk.pieces(text)
    if ps and "".join(ps) == text and len(ps) >= 1:
        ps = ps[:-1]
    return [tk.piece_id(p) for p in ps]


def speculative_plan(tokens: list[int], pieces: list[str], start: int, drafter: NGramSQLDrafter,
                     dfa: DFA | None = None) -> tuple[list[tuple[int, int]], SpecStats]:
    """Replay greedy decoding of the target sequence `tokens` (surface strings `pieces`) with
    speculation, starting after `tokens[:start]` were already emitted (e.g. by prefill).

    Returns (steps, stats); each step is (tokens_emitted, draft_len) and costs one target forward
    pass. With a DFA the drafter first uses grammar jump-forward (certain tokens), then n-grams.
    """
    stats, steps = SpecStats(), []
    pos = start
    state = dfa.run("".join(pieces[:start])) if dfa is not None else None
    while pos < len(tokens):
        draft: list[int] = []
        if state is not None and state >= 0:
            draft = jump_forward(dfa, state)  # type: ignore[arg-type]
            stats.jump_tokens += len(draft)
        if len(draft) < drafter.k:
            draft = draft + drafter.propose(tokens[:pos] + draft, drafter.k - len(draft))
        draft = draft[:drafter.k]
        acc = 0
        while acc < len(draft) and pos + acc < len(tokens) and draft[acc] == tokens[pos + acc]:
            acc += 1
        emitted = min(acc + 1, len(tokens) - pos)      # accepted drafts + 1 verifier token
        stats.proposed += len(draft)
        stats.accepted += min(acc, emitted)
        stats.steps += 1
        stats.emitted += emitted
        steps.append((emitted, len(draft)))
        drafter.observe(tokens[max(0, pos - drafter.n_max):pos], tokens[pos:pos + emitted])
        if state is not None and state >= 0:
            state = dfa.run("".join(pieces[pos:pos + emitted]), state)  # type: ignore[union-attr]
        pos += emitted
    return steps, stats
