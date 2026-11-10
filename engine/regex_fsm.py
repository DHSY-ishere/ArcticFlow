"""Regex -> NFA (Thompson) -> DFA (subset construction) compiler and a token-level FSM.

Supported syntax (enough for SQL grammars): literals, escapes (\\. \\( ...), \\d \\s \\w,
classes [a-z0-9_] / [^...], groups (...), alternation |, quantifiers * + ? {m} {m,n}.
The alphabet is printable ASCII (32..126); that is all the SQL dialect we emit needs.

TokenFSM lifts the character DFA to a vocabulary: allowed(state) is the set of token ids
whose *entire string* can be consumed from `state` without hitting a dead state - exactly the
logits mask used by constrained decoding (Outlines / XGrammar style).
"""
from __future__ import annotations

from dataclasses import dataclass, field

ALPHABET = frozenset(chr(c) for c in range(32, 127))
_DIGITS = frozenset("0123456789")
_WORD = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
_SPACE = frozenset(" ")


class RegexError(ValueError):
    pass


# ----- AST ---------------------------------------------------------------------------------
# ('set', frozenset) | ('cat', [..]) | ('alt', [..]) | ('star', n) | ('plus', n) | ('opt', n) | ('eps',)

class _Parser:
    def __init__(self, s: str):
        self.s, self.i = s, 0

    def peek(self):
        return self.s[self.i] if self.i < len(self.s) else None

    def eat(self):
        c = self.s[self.i]
        self.i += 1
        return c

    def parse(self):
        node = self.alt()
        if self.i != len(self.s):
            raise RegexError(f"unexpected {self.peek()!r} at {self.i}")
        return node

    def alt(self):
        branches = [self.cat()]
        while self.peek() == "|":
            self.eat()
            branches.append(self.cat())
        return branches[0] if len(branches) == 1 else ("alt", branches)

    def cat(self):
        items = []
        while self.peek() not in (None, "|", ")"):
            items.append(self.quant())
        if not items:
            return ("eps",)
        return items[0] if len(items) == 1 else ("cat", items)

    def quant(self):
        atom = self.atom()
        while True:
            c = self.peek()
            if c == "*":
                self.eat(); atom = ("star", atom)
            elif c == "+":
                self.eat(); atom = ("plus", atom)
            elif c == "?":
                self.eat(); atom = ("opt", atom)
            elif c == "{":
                self.eat()
                j = self.s.index("}", self.i)
                body = self.s[self.i:j]
                self.i = j + 1
                lo, sep, hi = body.partition(",")
                lo_n = int(lo)
                hi_n = lo_n if not sep else (int(hi) if hi else None)
                atom = _repeat(atom, lo_n, hi_n)
            else:
                return atom

    def atom(self):
        c = self.eat()
        if c == "(":
            node = self.alt()
            if self.peek() != ")":
                raise RegexError("unclosed group")
            self.eat()
            return node
        if c == "[":
            return ("set", self.cls())
        if c == ".":
            return ("set", ALPHABET)
        if c == "\\":
            return ("set", self.escape())
        return ("set", frozenset(c))

    def escape(self):
        e = self.eat()
        return {"d": _DIGITS, "w": _WORD, "s": _SPACE}.get(e, frozenset(e))

    def cls(self):
        neg = self.peek() == "^"
        if neg:
            self.eat()
        chars: set[str] = set()
        first = True
        while True:
            if self.peek() is None:
                raise RegexError("unclosed class")
            c = self.eat()
            if c == "]" and not first:
                break
            first = False
            if c == "\\":
                chars |= self.escape()
                continue
            if self.peek() == "-" and self.i + 1 < len(self.s) and self.s[self.i + 1] != "]":
                self.eat()
                hi = self.eat()
                chars |= {chr(x) for x in range(ord(c), ord(hi) + 1)}
            else:
                chars.add(c)
        return frozenset(ALPHABET - chars) if neg else frozenset(chars & ALPHABET)


def _repeat(atom, lo: int, hi: int | None):
    parts = [atom] * lo
    if hi is None:
        parts.append(("star", atom))
    else:
        opt: tuple = ("eps",)
        for _ in range(hi - lo):  # nested optionals: a{0,3} = (a(a(a)?)?)?
            opt = ("opt", ("cat", [atom, opt]) if opt != ("eps",) else atom)
        if hi > lo:
            parts.append(opt)
    return ("cat", parts) if parts else ("eps",)


# ----- NFA ---------------------------------------------------------------------------------

class _NFA:
    def __init__(self):
        self.eps: list[list[int]] = []
        self.edges: list[list[tuple[frozenset, int]]] = []

    def new(self) -> int:
        self.eps.append([])
        self.edges.append([])
        return len(self.eps) - 1

    def build(self, node) -> tuple[int, int]:
        kind = node[0]
        s, e = self.new(), self.new()
        if kind == "eps":
            self.eps[s].append(e)
        elif kind == "set":
            self.edges[s].append((node[1], e))
        elif kind == "cat":
            cur = s
            for child in node[1]:
                cs, ce = self.build(child)
                self.eps[cur].append(cs)
                cur = ce
            self.eps[cur].append(e)
        elif kind == "alt":
            for child in node[1]:
                cs, ce = self.build(child)
                self.eps[s].append(cs)
                self.eps[ce].append(e)
        elif kind in ("star", "plus", "opt"):
            cs, ce = self.build(node[1])
            self.eps[s].append(cs)
            self.eps[ce].append(e)
            if kind != "plus":
                self.eps[s].append(e)
            if kind != "opt":
                self.eps[ce].append(cs)
        else:
            raise RegexError(kind)
        return s, e

    def closure(self, states) -> frozenset:
        stack, seen = list(states), set(states)
        while stack:
            for t in self.eps[stack.pop()]:
                if t not in seen:
                    seen.add(t)
                    stack.append(t)
        return frozenset(seen)


class DFA:
    """Lazily-determinised DFA. States are NFA-state sets materialised on first visit, so only the
    parts of the automaton a query actually walks are ever built (a grammar for a 150-column model
    has ~16k DFA states; one SQL statement visits ~250 of them)."""

    def __init__(self, nfa: "_NFA", start_set: frozenset, end: int):
        self._nfa, self._end = nfa, end
        self._sets: list[frozenset] = [start_set]
        self._ids: dict[frozenset, int] = {start_set: 0}
        self._rows: dict[int, dict[str, int]] = {}
        self.start = 0

    def row(self, state: int) -> dict[str, int]:
        """char -> next state for every legal char from `state`."""
        r = self._rows.get(state)
        if r is not None:
            return r
        nfa = self._nfa
        moves: dict[str, set[int]] = {}
        for st in self._sets[state]:
            for chars, tgt in nfa.edges[st]:
                for ch in chars:
                    moves.setdefault(ch, set()).add(tgt)
        r = {}
        for ch, tgts in moves.items():
            nxt = nfa.closure(tgts)
            nid = self._ids.get(nxt)
            if nid is None:
                nid = self._ids[nxt] = len(self._sets)
                self._sets.append(nxt)
            r[ch] = nid
        self._rows[state] = r
        return r

    def is_accepting(self, state: int) -> bool:
        return state >= 0 and self._end in self._sets[state]

    def step(self, state: int, ch: str) -> int:
        return self.row(state).get(ch, -1)

    def run(self, text: str, state: int | None = None) -> int:
        s = self.start if state is None else state
        for ch in text:
            s = self.row(s).get(ch, -1)
            if s < 0:
                return -1
        return s

    def accepts(self, text: str) -> bool:
        return self.is_accepting(self.run(text))

    def forced_text(self, state: int, limit: int = 256) -> str:
        """Longest string forced from `state` (exactly one outgoing char each step, not accepting).
        This is the 'jump-forward' text: free, certain tokens for speculation."""
        out = []
        while len(out) < limit and not self.is_accepting(state):
            row = self.row(state)
            if len(row) != 1:
                break
            (ch, nxt), = row.items()
            out.append(ch)
            state = nxt
        return "".join(out)

    @property
    def num_states(self) -> int:
        """States materialised so far (the full automaton is built lazily)."""
        return len(self._sets)


def compile_regex(pattern: str) -> DFA:
    ast = _Parser(pattern).parse()
    nfa = _NFA()
    start, end = nfa.build(ast)
    return DFA(nfa, nfa.closure([start]), end)


# ----- Token-level FSM -----------------------------------------------------------------------

@dataclass
class TokenFSM:
    """Vocabulary-level view of a DFA. `vocab[i]` is the surface string of token i (None to ban)."""
    dfa: DFA
    vocab: list[str | None]
    eos_id: int | None = None
    _by_first: dict[str, list[int]] = field(default_factory=dict, repr=False)
    _allowed: dict[int, list[int]] = field(default_factory=dict, repr=False)
    _sets: dict[int, frozenset] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        for i, s in enumerate(self.vocab):
            if s:
                self._by_first.setdefault(s[0], []).append(i)

    @property
    def start(self) -> int:
        return self.dfa.start

    def allowed(self, state: int) -> list[int]:
        """Token ids legal from `state` (EOS included iff state is accepting). Memoised."""
        hit = self._allowed.get(state)
        if hit is not None:
            return hit
        out: list[int] = []
        for ch in self.dfa.row(state):
            for tid in self._by_first.get(ch, ()):
                if self.dfa.run(self.vocab[tid], state) >= 0:  # type: ignore[arg-type]
                    out.append(tid)
        if self.eos_id is not None and self.dfa.is_accepting(state):
            out.append(self.eos_id)
        self._allowed[state] = out
        return out

    def allowed_set(self, state: int) -> frozenset[int]:
        s = self._sets.get(state)
        if s is None:
            s = self._sets[state] = frozenset(self.allowed(state))
        return s

    def advance(self, state: int, tid: int) -> int:
        if tid == self.eos_id:
            return state
        return self.dfa.run(self.vocab[tid], state)  # type: ignore[arg-type]

    def is_accepting(self, state: int) -> bool:
        return self.dfa.is_accepting(state)

    def forced_token(self, state: int) -> int | None:
        """If exactly one token is legal (and the state is not accepting), return it."""
        a = self.allowed(state)
        return a[0] if len(a) == 1 and a[0] != self.eos_id else None
