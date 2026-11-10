"""Read-only DuckDB execution + Arrow result handoff (HTTP Arrow IPC stream and Arrow Flight).

Safety layers for model-generated SQL:
  1. exactly one statement, parsed by DuckDB's own parser (json_serialize_sql) and required to be a
     plain SELECT node - DDL/DML/PRAGMA/COPY/ATTACH never parse into one;
  2. `enable_external_access=false` + `lock_configuration=true`: no file/URL reads, no extension
     loading, settings cannot be re-enabled from SQL;
  3. per-query cursor with a wall-clock timeout (con.interrupt) and a row cap.
"""
from __future__ import annotations

import json
import secrets
import threading
import time
import zlib
from collections import OrderedDict

import duckdb
import numpy as np
import pyarrow as pa

from semantic_compiler import CompiledModel


class SQLValidationError(ValueError):
    pass


class SQLExecutor:
    def __init__(self, seed_rows: int = 5000, max_rows: int = 100_000, timeout_s: float = 10.0):
        self.con = duckdb.connect(":memory:")
        self.seed_rows, self.max_rows, self.timeout_s = seed_rows, max_rows, timeout_s
        self._seeded: set[str] = set()
        self._lock = threading.Lock()
        self._locked = False

    # -- synthetic data ----------------------------------------------------------------------------
    def ensure_tables(self, model: CompiledModel) -> None:
        """Materialise deterministic synthetic data for any table of the model that doesn't exist yet
        (demo / load-test convenience; real deployments attach their own warehouse tables)."""
        with self._lock:
            missing = [t for t in model.tables if t not in self._seeded]
            if not missing:
                return
            existing = {r[0] for r in self.con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
            dim_tables = {r.right for r in model.relationships}
            built: dict[tuple[str, str], np.ndarray] = {}
            order = sorted(missing, key=lambda t: (t not in dim_tables, t))   # parents first
            for tname in order:
                if tname in existing:
                    self._seeded.add(tname)
                    continue
                n = max(50, self.seed_rows // 20) if tname in dim_tables else self.seed_rows
                rng = np.random.default_rng(zlib.crc32(f"{model.name}/{tname}".encode()))
                cols: dict[str, pa.Array] = {}
                fks: dict[str, tuple[str, str]] = {}
                for r in model.relationships:
                    if r.left == tname:
                        for lc, rc in r.pairs:
                            fks[lc] = (r.right, rc)
                for c in model.tables[tname].columns.values():
                    ph = c.physical
                    if ph in cols:
                        continue
                    if ph in fks and fks[ph] in built:                       # foreign key -> sample parent keys
                        parent = built[fks[ph]]
                        cols[ph] = pa.array(parent[rng.integers(0, len(parent), n)])
                    elif c.data_type == "number" and (c.kind == "key" or ph.endswith("_id")):   # surrogate key
                        arr = np.arange(1, n + 1, dtype=np.int64)
                        cols[ph] = pa.array(arr)
                        built[(tname, ph)] = arr
                    elif c.data_type == "number":
                        if "quantity" in c.name or "count" in c.name:
                            cols[ph] = pa.array(rng.integers(1, 20, n))
                        else:
                            cols[ph] = pa.array(np.round(rng.gamma(2.0, 60.0, n), 2))
                    elif c.data_type == "date":
                        days = rng.integers(0, 3 * 365, n).astype("timedelta64[D]")
                        cols[ph] = pa.array((np.datetime64("2022-01-01") + days).astype("datetime64[D]"))
                    elif c.data_type == "boolean":
                        cols[ph] = pa.array(rng.random(n) < 0.5)
                    else:
                        vals = list(c.sample_values) or [f"{c.name}_{i}" for i in range(1, 21)]
                        p = rng.dirichlet(np.ones(len(vals)) * 2.0)
                        cols[ph] = pa.array(np.array(vals, dtype=object)[rng.choice(len(vals), n, p=p)].tolist())
                tbl = pa.table(cols)
                self.con.register("_seed_tmp", tbl)
                self.con.execute(f'CREATE TABLE "{tname}" AS SELECT * FROM _seed_tmp')
                self.con.unregister("_seed_tmp")
                self._seeded.add(tname)
            self._lock_down()

    def _lock_down(self) -> None:
        if self._locked:
            return
        self.con.execute("SET enable_external_access=false")
        self.con.execute("SET lock_configuration=true")
        self._locked = True

    # -- validation + execution -------------------------------------------------------------------
    def validate(self, sql: str) -> str:
        s = sql.strip().rstrip(";").strip()
        if not s or ";" in s:
            raise SQLValidationError("exactly one statement is allowed")
        cur = self.con.cursor()
        try:
            raw = cur.execute("SELECT json_serialize_sql(?)", [s]).fetchone()[0]
        finally:
            cur.close()
        doc = json.loads(raw)
        if doc.get("error"):
            raise SQLValidationError(f"not a valid single SELECT: {doc.get('error_message', 'parse error')}")
        stmts = doc.get("statements", [])
        if len(stmts) != 1 or stmts[0].get("node", {}).get("type") != "SELECT_NODE":
            raise SQLValidationError("only plain SELECT statements are allowed")
        return s

    def execute(self, sql: str) -> pa.Table:
        s = self.validate(sql)
        cur = self.con.cursor()
        timer = threading.Timer(self.timeout_s, cur.interrupt)
        timer.start()
        try:
            cur.execute(f"SELECT * FROM ({s}) AS q LIMIT {self.max_rows}")
            # DuckDB >= 2.0 renamed fetch_arrow_table -> to_arrow_table
            fetch = getattr(cur, "to_arrow_table", None) or cur.fetch_arrow_table
            return fetch()
        except duckdb.InterruptException as e:
            raise SQLValidationError("query timed out") from e
        except duckdb.Error as e:
            raise SQLValidationError(f"execution error: {e}") from e
        finally:
            timer.cancel()
            cur.close()


class ResultStore:
    """Bounded, TTL'd store of Arrow tables addressed by opaque ids (the Flight ticket)."""

    def __init__(self, max_bytes: int = 256 << 20, ttl_s: float = 300.0):
        self.max_bytes, self.ttl = max_bytes, ttl_s
        self._d: OrderedDict[str, tuple[float, pa.Table]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def put(self, table: pa.Table) -> str:
        rid = secrets.token_hex(8)
        with self._lock:
            self._d[rid] = (time.monotonic(), table)
            self._bytes += table.nbytes
            self._gc()
        return rid

    def get(self, rid: str) -> pa.Table | None:
        with self._lock:
            self._gc()
            item = self._d.get(rid)
            return item[1] if item else None

    def _gc(self) -> None:
        now = time.monotonic()
        while self._d:
            rid, (ts, t) = next(iter(self._d.items()))
            if now - ts > self.ttl or self._bytes > self.max_bytes:
                self._d.popitem(last=False)
                self._bytes -= t.nbytes
            else:
                break


def start_flight_server(store: ResultStore, host: str, port: int):
    """Arrow Flight endpoint: DoGet(ticket=result id) streams the stored table's record batches
    without row-wise re-encoding (columnar buffers go straight onto the wire)."""
    import pyarrow.flight as fl

    class _Server(fl.FlightServerBase):
        def do_get(self, context, ticket):
            t = store.get(ticket.ticket.decode())
            if t is None:
                raise fl.FlightUnavailableError("unknown or expired result id")
            return fl.RecordBatchStream(t)

        def get_flight_info(self, context, descriptor):
            rid = descriptor.path[0].decode() if descriptor.path else ""
            t = store.get(rid)
            if t is None:
                raise fl.FlightUnavailableError("unknown or expired result id")
            ep = fl.FlightEndpoint(rid.encode(), [])
            return fl.FlightInfo(t.schema, descriptor, [ep], t.num_rows, t.nbytes)

    server = _Server(f"grpc://{host}:{port}")
    threading.Thread(target=server.serve, name="arrow-flight", daemon=True).start()
    return server
