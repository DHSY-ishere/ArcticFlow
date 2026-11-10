"""Deterministic semantic-model-grounded NL -> SQL planner.

This is the *brain* of the simulator backend (the stand-in for an LLM in latency-simulation
mode): it resolves a question against the model's names/synonyms/sample values and renders SQL in
the exact canonical shape the compiled grammar accepts. It is intentionally rule-based so the
load tests are bit-for-bit reproducible. The HF backend does not use it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from semantic_compiler import GRANULARITIES, Column, CompiledModel, Metric, compile_model

TOP_WORDS = {"top", "best", "highest", "most", "largest", "biggest", "greatest"}
BOTTOM_WORDS = {"bottom", "lowest", "least", "worst", "smallest", "fewest"}
GRAN_WORDS = {"daily": "day", "day": "day", "weekly": "week", "week": "week", "monthly": "month",
              "month": "month", "quarterly": "quarter", "quarter": "quarter", "yearly": "year",
              "annual": "year", "annually": "year"}
NUM_OPS = (("greater than", ">"), ("more than", ">"), ("over", ">"), ("above", ">"), ("at least", ">="),
           ("less than", "<"), ("under", "<"), ("below", "<"), ("at most", "<="))


class PlanError(ValueError):
    pass


def _stem(w: str) -> str:
    w = w.lower().replace("'s", "")
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+(?:'s)?", text)


@dataclass
class Plan:
    metrics: list[Metric] = field(default_factory=list)
    dims: list[Column] = field(default_factory=list)
    time: tuple[Column, str] | None = None
    filters: list[str] = field(default_factory=list)
    filter_cols: list[Column] = field(default_factory=list)
    order_desc: bool = True
    limit: int | None = None
    sql: str = ""


class _Lexicon:
    def __init__(self, model: CompiledModel):
        self.entries: dict[tuple[str, ...], tuple[str, object]] = {}
        self.max_n = 1
        prio = {"metric": 0, "dim": 1, "time": 2, "fact": 3}

        def add(phrase: str, kind: str, obj: object):
            key = tuple(_stem(w) for w in _words(phrase.replace("_", " ")))
            if not key:
                return
            cur = self.entries.get(key)
            if cur is None or prio[kind] < prio[cur[0]]:
                self.entries[key] = (kind, obj)
                self.max_n = max(self.max_n, len(key))

        for m in model.metrics:
            for p in (m.name, *m.synonyms):
                add(p, "metric", m)
        for c in model.columns():
            kind = {"dimension": "dim", "time_dimension": "time", "fact": "fact"}.get(c.kind)
            if kind:
                for p in (c.name, *c.synonyms):
                    add(p, kind, c)

    def scan(self, words: list[str]) -> list[tuple[int, int, str, object]]:
        stems = [_stem(w) for w in words]
        out, i = [], 0
        while i < len(stems):
            for n in range(min(self.max_n, len(stems) - i), 0, -1):
                hit = self.entries.get(tuple(stems[i:i + n]))
                if hit:
                    out.append((i, n, hit[0], hit[1]))
                    i += n
                    break
            else:
                i += 1
        return out


_LEX: dict[str, _Lexicon] = {}


def _lex(model: CompiledModel) -> _Lexicon:
    lx = _LEX.get(model.digest)
    if lx is None:
        if len(_LEX) > 128:
            _LEX.pop(next(iter(_LEX)))
        lx = _LEX[model.digest] = _Lexicon(model)
    return lx


def _plan_one(model: CompiledModel, question: str, prev: Plan | None) -> Plan:
    words = _words(question)
    low = [w.lower() for w in words]
    lx = _lex(model)
    hits = lx.scan(words)

    p = Plan()
    metric_seen: set[str] = set()
    dim_seen: set[str] = set()
    fact_cols: list[Column] = []
    time_col: Column | None = None
    for _, _, kind, obj in hits:
        if kind == "metric" and obj.name not in metric_seen:  # type: ignore[attr-defined]
            metric_seen.add(obj.name); p.metrics.append(obj)  # type: ignore[arg-type]
        elif kind == "dim" and obj.qualified not in dim_seen:  # type: ignore[attr-defined]
            dim_seen.add(obj.qualified); p.dims.append(obj)  # type: ignore[arg-type]
        elif kind == "time":
            time_col = obj  # type: ignore[assignment]
        elif kind == "fact":
            fact_cols.append(obj)  # type: ignore[arg-type]

    # --- filters from sample values (e.g. "in EMEA", "completed orders") -----------------------
    text_l = " " + " ".join(low) + " "
    for c in model.columns("dimension"):
        for v in c.sample_values:
            if f" {v.lower()} " in text_l:
                p.filters.append(f"{c.qualified} = '{v}'")
                p.filter_cols.append(c)
    ym = re.search(r"\b((?:19|20)\d\d)\b", question)
    year = ym.group(1) if ym else None
    for phrase, op in NUM_OPS:
        nm = re.search(rf"\b{phrase}\s+(\d+(?:\.\d+)?)\b", question.lower())
        if nm and fact_cols:
            c = fact_cols[0]
            p.filters.append(f"{c.qualified} {op} {nm.group(1)}")
            p.filter_cols.append(c)
            break

    # --- inheritance for follow-up turns --------------------------------------------------------
    if prev is not None:
        if not p.metrics:
            p.metrics = list(prev.metrics)
        if not p.dims and not metric_seen and time_col is None and not any(w in GRAN_WORDS for w in low):
            p.dims = list(prev.dims)
            p.time = prev.time
        for f, c in zip(prev.filters, prev.filter_cols):
            if f not in p.filters and all(c.qualified != x.qualified for x in p.filter_cols):
                p.filters.append(f); p.filter_cols.append(c)

    # --- granularity / time -------------------------------------------------------------------
    gran = next((GRAN_WORDS[w] for w in low if w in GRAN_WORDS), None)
    if "trend" in low or "over" in low and "time" in low:
        gran = gran or "month"
    if p.time is None and (gran or time_col is not None):
        tc = time_col
        if tc is None:
            tabs = {t for m in p.metrics for t in m.tables} or {d.table for d in p.dims}
            cands = [c for c in model.columns("time_dimension") if c.table in tabs] or model.columns("time_dimension")
            tc = cands[0] if cands else None
        if tc is not None:
            p.time = (tc, gran or "month")
    if year:
        tabs = {t for m in p.metrics for t in m.tables} or {d.table for d in p.dims}
        cands = [c for c in model.columns("time_dimension") if c.table in tabs] or model.columns("time_dimension")
        if cands:
            p.filters.append(f"YEAR({cands[0].qualified}) = {year}")
            p.filter_cols.append(cands[0])

    # --- ordering ------------------------------------------------------------------------------
    if any(w in BOTTOM_WORDS for w in low):
        p.order_desc = False
    tm = re.search(r"\b(?:top|bottom|first)\s+(\d{1,4})\b", question.lower())
    if tm:
        p.limit = max(1, int(tm.group(1)))
    elif any(w in TOP_WORDS | BOTTOM_WORDS for w in low):
        p.limit = 10

    if not p.metrics:
        # dimension-only question: use the dim table's first metric, else a row count
        tabs = {d.table for d in p.dims}
        p.metrics = [next((m for m in model.metrics if m.tables[0] in tabs), None)] if p.dims else []
        p.metrics = [m for m in p.metrics if m]
    if not p.metrics and not p.dims and p.time is None:
        raise PlanError("could not map the question onto the semantic model")
    p.sql = _render(model, p)
    return p


def _render(model: CompiledModel, p: Plan) -> str:
    items = [d.qualified for d in p.dims]
    group = [d.qualified for d in p.dims]
    order_alias = None
    if p.time:
        tc, g = p.time
        alias = f"{tc.name}_{g}"
        items.insert(0, f"DATE_TRUNC('{g}', {tc.qualified}) AS {alias}")
        group.insert(0, alias)
        order_alias = alias
    items += [f"{m.expr} AS {m.name}" for m in p.metrics] or ["COUNT(*) AS row_count"]
    metric_names = [m.name for m in p.metrics] or ["row_count"]

    tables = {t for m in p.metrics for t in m.tables} | {d.table for d in p.dims} | {c.table for c in p.filter_cols}
    if p.time:
        tables.add(p.time[0].table)
    base = p.metrics[0].tables[0] if p.metrics else (p.dims[0].table if p.dims else p.time[0].table)  # type: ignore[index]
    sql = f"SELECT {', '.join(items)} FROM {base}"
    joined = {base}
    for t in sorted(tables - {base}):
        path = model.join_path(base, t)
        if path is None:
            raise PlanError(f"no join path from {base} to {t}")
        for rel, nxt in path:
            if nxt not in joined:
                joined.add(nxt)
                sql += f" JOIN {nxt} ON {rel.condition()}"
    if p.filters:
        sql += " WHERE " + " AND ".join(dict.fromkeys(p.filters))
    if group and (p.metrics or True):
        sql += " GROUP BY " + ", ".join(group)
    if order_alias:
        sql += f" ORDER BY {order_alias} ASC"
    elif p.dims:
        sql += f" ORDER BY {metric_names[0]} {'DESC' if p.order_desc else 'ASC'}"
    if p.limit:
        sql += f" LIMIT {p.limit}"
    elif p.dims and not p.time:
        sql += " LIMIT 100"
    return sql


@lru_cache(maxsize=65536)
def _plan_cached(digest: str, questions: tuple[str, ...]) -> Plan:
    model = _MODELS[digest]
    prev: Plan | None = None
    for q in questions:
        try:
            prev = _plan_one(model, q, prev)
        except PlanError:
            if q is questions[-1]:
                raise
    assert prev is not None
    return prev


_MODELS: dict[str, CompiledModel] = {}


def plan_sql(model: CompiledModel, user_questions: list[str]) -> Plan:
    """Plan the SQL for the latest user question, carrying context from earlier user turns."""
    if not user_questions:
        raise PlanError("no user question")
    _MODELS[model.digest] = model
    # verified-query short-circuit (single-turn, near-exact match)
    if len(user_questions) == 1:
        qa = {_stem(w) for w in _words(user_questions[0])}
        for vq in model.verified_queries:
            qb = {_stem(w) for w in _words(vq.question)}
            if qa and len(qa & qb) / len(qa | qb) >= 0.85 and model.accepts(vq.sql):
                return Plan(sql=vq.sql.strip())
    return _plan_cached(model.digest, tuple(user_questions))


def plan_for_yaml(raw_yaml: str, user_questions: list[str]) -> Plan:
    return plan_sql(compile_model(raw_yaml), user_questions)
