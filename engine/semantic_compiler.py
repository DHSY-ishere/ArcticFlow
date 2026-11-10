"""Cortex-Analyst-style YAML semantic model -> typed schema + SQL grammar constraint.

Parsed sections: tables (dimensions, time_dimensions, facts, metrics/measures, primary_key,
base_table), relationships (relationship_columns, join_type), top-level metrics and
verified_queries. The compiler then emits

  * a *regex* that describes every SELECT the model is allowed to produce (only declared
    tables/columns/metrics, only declared join conditions, typed literals), and
  * a character DFA for that regex (regex_fsm.compile_regex), liftable to a token-level FSM
    (TokenFSM) for logits masking / jump-forward / speculative-draft pruning.

The grammar is regular, so it cannot express semantic cross-clause rules (e.g. "GROUP BY must
list every non-aggregated select item"). DuckDB's binder enforces those at validation time.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from functools import cached_property

import yaml

from regex_fsm import DFA, TokenFSM, compile_regex

GRANULARITIES = ("day", "week", "month", "quarter", "year")
_TYPE_MAP = (("BOOL", "boolean"), ("DATE", "date"), ("TIME", "date"), ("INT", "number"), ("NUM", "number"),
             ("FLOAT", "number"), ("DOUBLE", "number"), ("DEC", "number"), ("REAL", "number"), ("LONG", "number"))
_SQL_WORDS = {"AS", "DISTINCT", "CASE", "WHEN", "THEN", "ELSE", "END", "AND", "OR", "NOT", "NULL", "IN", "IS",
              "BETWEEN", "TRUE", "FALSE", "INTERVAL", "INTEGER", "DOUBLE", "FLOAT", "VARCHAR", "DATE", "BIGINT"}
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")
_TOKEN = re.compile(r"'[^']*'|[A-Za-z_][A-Za-z0-9_]*(?!\w)")


class SemanticModelError(ValueError):
    pass


def _norm_type(raw: str | None) -> str:
    up = (raw or "TEXT").upper()
    for key, val in _TYPE_MAP:
        if key in up:
            return val
    return "string"


@dataclass(frozen=True)
class Column:
    table: str
    name: str            # logical name
    physical: str        # physical column in the table
    kind: str            # dimension | time_dimension | fact | key
    data_type: str       # string | number | date | boolean
    description: str = ""
    synonyms: tuple[str, ...] = ()
    sample_values: tuple[str, ...] = ()

    @property
    def qualified(self) -> str:
        return f"{self.table}.{self.physical}"


@dataclass(frozen=True)
class Metric:
    name: str
    expr: str                       # fully-qualified SQL aggregate expression
    tables: tuple[str, ...]
    description: str = ""
    synonyms: tuple[str, ...] = ()


@dataclass(frozen=True)
class Relationship:
    name: str
    left: str
    right: str
    pairs: tuple[tuple[str, str], ...]
    join_type: str = "inner"

    def condition(self) -> str:
        return " AND ".join(f"{self.left}.{l} = {self.right}.{r}" for l, r in self.pairs)


@dataclass
class Table:
    name: str
    description: str
    columns: dict[str, Column] = field(default_factory=dict)   # by logical name


@dataclass
class VerifiedQuery:
    name: str
    question: str
    sql: str


# ----------------------------------------------------------------------------------------------

def _lit(s: str) -> str:
    """Escape for regex_fsm."""
    return re.sub(r"([\\.()\[\]{}*+?|])", r"\\\1", s)


def _alt(items: list[str], literal: bool = True) -> str:
    items = sorted(set(items))
    body = "|".join(_lit(i) if literal else i for i in items)
    return f"({body})"


def _qualify(expr: str, table: str, physical_cols: set[str]) -> str:
    def sub(m: re.Match) -> str:
        tok = m.group(0)
        if tok.startswith("'"):
            return tok
        tail = expr[m.end():m.end() + 1]
        head = expr[max(0, m.start() - 1):m.start()]
        if head == "." or tail == "(" or tok.upper() in _SQL_WORDS or tok not in physical_cols:
            return tok
        return f"{table}.{tok}"
    return _TOKEN.sub(sub, expr)


class CompiledModel:
    """Parsed + compiled semantic model."""

    def __init__(self, raw_yaml: str):
        self.raw = raw_yaml
        self.digest = hashlib.sha256(raw_yaml.encode()).hexdigest()[:16]
        try:
            doc = yaml.safe_load(raw_yaml)
        except yaml.YAMLError as e:
            raise SemanticModelError(f"invalid YAML: {e}") from e
        if not isinstance(doc, dict) or not doc.get("tables"):
            raise SemanticModelError("semantic model needs a non-empty 'tables' list")
        self.name: str = str(doc.get("name", "semantic_model"))
        self.description: str = str(doc.get("description", ""))
        self.tables: dict[str, Table] = {}
        self.metrics: list[Metric] = []
        self.relationships: list[Relationship] = []
        self.verified_queries: list[VerifiedQuery] = []
        self._parse(doc)

    # -- parsing ---------------------------------------------------------------------------
    def _parse(self, doc: dict) -> None:
        pending_metrics: list[tuple[str, dict, set[str]]] = []
        for t in doc["tables"]:
            tname = t.get("name")
            if not tname or not _IDENT.match(tname):
                raise SemanticModelError(f"table needs an identifier name, got {tname!r}")
            tbl = Table(tname, str(t.get("description", "")))
            for section, kind in (("dimensions", "dimension"), ("time_dimensions", "time_dimension"),
                                  ("facts", "fact")):
                for c in t.get(section) or []:
                    name = c["name"]
                    expr = str(c.get("expr", name)).strip()
                    phys = expr if _IDENT.match(expr) else name
                    dtype = "date" if kind == "time_dimension" else _norm_type(c.get("data_type"))
                    tbl.columns[name] = Column(
                        tname, name, phys, kind, dtype, str(c.get("description", "")),
                        tuple(map(str, c.get("synonyms") or ())), tuple(map(str, c.get("sample_values") or ())))
            for pk in (t.get("primary_key") or {}).get("columns", []):
                if not any(c.physical == pk for c in tbl.columns.values()):
                    tbl.columns[pk] = Column(tname, pk, pk, "key", "number")
            self.tables[tname] = tbl
            for section in ("metrics", "measures"):
                for m in t.get(section) or []:
                    pending_metrics.append((tname, m, {c.physical for c in tbl.columns.values()}))

        for r in doc.get("relationships") or []:
            l, rt = r["left_table"], r["right_table"]
            if l not in self.tables or rt not in self.tables:
                raise SemanticModelError(f"relationship {r.get('name')} references unknown table")
            pairs = tuple((p["left_column"], p["right_column"]) for p in r["relationship_columns"])
            for lc, rc in pairs:  # make join keys addressable/typed even when undeclared
                for tn, col in ((l, lc), (rt, rc)):
                    if not any(c.physical == col for c in self.tables[tn].columns.values()):
                        self.tables[tn].columns[col] = Column(tn, col, col, "key", "number")
            self.relationships.append(Relationship(
                r.get("name", f"{l}_to_{rt}"), l, rt, pairs, str(r.get("join_type", "inner")).lower()))

        # table-level metrics need physical columns incl. implicit keys -> resolve after tables parsed
        for tname, m, _ in pending_metrics:
            phys = {c.physical for c in self.tables[tname].columns.values()}
            expr = str(m["expr"]).strip()
            agg = m.get("default_aggregation")
            if agg and _IDENT.match(expr):
                expr = f"{str(agg).upper()}({expr})"
            self._add_metric(m, _qualify(expr, tname, phys))
        for m in doc.get("metrics") or []:
            self._add_metric(m, str(m["expr"]).strip())
        for v in doc.get("verified_queries") or []:
            self.verified_queries.append(VerifiedQuery(v.get("name", ""), str(v["question"]), str(v["sql"]).strip()))

    def _add_metric(self, m: dict, qexpr: str) -> None:
        used = tuple(t for t in self.tables if re.search(rf"\b{re.escape(t)}\.", qexpr))
        if not used:
            raise SemanticModelError(f"metric {m.get('name')!r}: expression must reference a table: {qexpr}")
        self.metrics.append(Metric(m["name"], qexpr, used, str(m.get("description", "")),
                                   tuple(map(str, m.get("synonyms") or ()))))

    # -- schema helpers ----------------------------------------------------------------------
    def columns(self, kind: str | None = None) -> list[Column]:
        return [c for t in self.tables.values() for c in t.columns.values() if kind is None or c.kind == kind]

    def join_path(self, src: str, dst: str) -> list[tuple[Relationship, str]] | None:
        """BFS over the relationship graph. Returns [(relationship, table_to_join_next), ...]."""
        if src == dst:
            return []
        prev: dict[str, tuple[str, Relationship]] = {}
        frontier, seen = [src], {src}
        while frontier:
            nxt = []
            for cur in frontier:
                for r in self.relationships:
                    for a, b in ((r.left, r.right), (r.right, r.left)):
                        if a == cur and b not in seen:
                            seen.add(b)
                            prev[b] = (cur, r)
                            nxt.append(b)
            frontier = nxt
        if dst not in prev:
            return None
        path, cur = [], dst
        while cur != src:
            p, r = prev[cur]
            path.append((r, cur))
            cur = p
        return path[::-1]

    # -- grammar -----------------------------------------------------------------------------
    @cached_property
    def regex(self) -> str:
        dims = [c.qualified for c in self.columns("dimension")]
        times = self.columns("time_dimension")
        time_items = [f"DATE_TRUNC('{g}', {c.qualified}) AS {c.name}_{g}" for c in times for g in GRANULARITIES]
        time_alias = [f"{c.name}_{g}" for c in times for g in GRANULARITIES]
        metric_items = [f"{m.expr} AS {m.name}" for m in self.metrics] + ["COUNT(*) AS row_count"]
        metric_alias = [m.name for m in self.metrics] + ["row_count"]
        all_cols = [c.qualified for c in self.columns()]

        item = _alt(dims + time_items + metric_items)
        group_key = _alt(dims + time_alias + [c.qualified for c in times])
        order_key = _alt(dims + time_alias + metric_alias)

        joins = []
        for r in self.relationships:
            cond = r.condition()
            joins.append(f"JOIN {r.right} ON {cond}")
            joins.append(f"JOIN {r.left} ON {cond}")
        from_clause = f"FROM {_alt(list(self.tables))}"
        if joins:
            from_clause += f"( (LEFT )?{_alt(joins)})*"

        conds = []
        for c in self.columns():
            q = _lit(c.qualified)
            if c.data_type == "string":
                conds.append(f"{q} (=|!=) '[A-Za-z0-9 _-]{{1,32}}'")
            elif c.data_type == "number":
                conds.append(f"{q} (=|!=|<|>|<=|>=) -?[0-9]{{1,12}}(\\.[0-9]{{1,6}})?")
            elif c.data_type == "date":
                conds.append(f"{q} (=|<|>|<=|>=) '[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}'")
                conds.append(f"YEAR\\({q}\\) = [0-9]{{4}}")
            else:
                conds.append(f"{q} = (TRUE|FALSE)")
        assert all_cols
        cond = "(" + "|".join(conds) + ")"
        return (f"SELECT {item}(, {item})* {from_clause}"
                f"( WHERE {cond}( AND {cond})*)?"
                f"( GROUP BY {group_key}(, {group_key})*)?"
                f"( ORDER BY {order_key}( ASC| DESC)?(, {order_key}( ASC| DESC)?)*)?"
                f"( LIMIT [1-9][0-9]{{0,3}})?")

    @cached_property
    def dfa(self) -> DFA:
        return compile_regex(self.regex)

    def accepts(self, sql: str) -> bool:
        return self.dfa.accepts(sql.strip())

    def token_fsm(self, vocab: list[str | None], eos_id: int | None = None) -> TokenFSM:
        return TokenFSM(self.dfa, vocab, eos_id)

    def summary(self) -> dict:
        return {"name": self.name, "digest": self.digest, "tables": list(self.tables),
                "metrics": [m.name for m in self.metrics], "relationships": [r.name for r in self.relationships],
                "regex_chars": len(self.regex), "dfa_states": self.dfa.num_states}


_CACHE: dict[str, CompiledModel] = {}


def compile_model(raw_yaml: str) -> CompiledModel:
    """Memoised by content digest: the semantic model is the shared prefix, so it repeats."""
    key = hashlib.sha256(raw_yaml.encode()).hexdigest()[:16]
    m = _CACHE.get(key)
    if m is None:
        if len(_CACHE) >= 256:
            _CACHE.pop(next(iter(_CACHE)))
        m = _CACHE[key] = CompiledModel(raw_yaml)
    return m
