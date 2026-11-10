"""ArcticFlow inference worker (FastAPI).

  POST /v1/generate        SSE: start -> token* -> done(+ optional DuckDB result handle)
  GET  /v1/results/{id}    result as an Apache Arrow IPC stream
  :8815 (Arrow Flight)     DoGet(ticket=id) -> record batches
  POST /v1/compile         compile a semantic model, report grammar size
  POST /v1/sql             validate + execute read-only SQL directly
  GET  /healthz, /stats

Backend selection: ARCTIC_BACKEND=sim (default, CPU latency simulator) | hf (real HuggingFace model).
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager

import pyarrow as pa
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from backends import GenRequest
from semantic_compiler import SemanticModelError, compile_model
from sql_executor import ResultStore, SQLExecutor, SQLValidationError, start_flight_server

WORKER_NAME = os.environ.get("ARCTIC_WORKER_NAME", "worker")
FLIGHT_PORT = int(os.environ.get("ARCTIC_FLIGHT_PORT", "8815"))
PUBLIC_HOST = os.environ.get("ARCTIC_PUBLIC_HOST", WORKER_NAME)


def _sse(ev: dict) -> bytes:
    return b"data: " + json.dumps(ev, separators=(",", ":"), default=str).encode() + b"\n\n"


def make_backend():
    kind = os.environ.get("ARCTIC_BACKEND", "sim").lower()
    if kind == "hf":
        from hf_backend import HFBackend
        return HFBackend()
    from backends import SimBackend
    return SimBackend()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.backend = make_backend()
    app.state.executor = SQLExecutor(seed_rows=int(os.environ.get("ARCTIC_SEED_ROWS", "5000")))
    app.state.results = ResultStore()
    app.state.flight = None
    if os.environ.get("ARCTIC_FLIGHT", "1") == "1":
        app.state.flight = start_flight_server(app.state.results, "0.0.0.0", FLIGHT_PORT)
    await app.state.backend.start()
    yield
    await app.state.backend.stop()
    if app.state.flight is not None:
        app.state.flight.shutdown()


app = FastAPI(title="ArcticFlow worker", lifespan=lifespan)


def _run_sql(app: FastAPI, raw_yaml: str | None, sql: str) -> dict:
    """Blocking: seed (if needed), validate, execute, stash Arrow table. Runs in a thread."""
    ex: SQLExecutor = app.state.executor
    if raw_yaml:
        ex.ensure_tables(compile_model(raw_yaml))
    t0 = time.perf_counter()
    table = ex.execute(sql)
    rid = app.state.results.put(table)
    preview = table.slice(0, 5).to_pylist()
    return {"id": rid, "rows": table.num_rows, "bytes": table.nbytes,
            "columns": [{"name": f.name, "type": str(f.type)} for f in table.schema],
            "exec_ms": round((time.perf_counter() - t0) * 1000, 3),
            "flight": {"host": PUBLIC_HOST, "port": FLIGHT_PORT, "ticket": rid},
            "arrow_ipc": f"/v1/results/{WORKER_NAME}/{rid}", "preview": preview}


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "worker": WORKER_NAME, "backend": app.state.backend.name}


@app.get("/stats")
async def stats():
    return {"worker": WORKER_NAME, **app.state.backend.stats()}


@app.post("/v1/compile")
async def compile_endpoint(request: Request):
    body = await request.json()
    try:
        m = await asyncio.to_thread(compile_model, body["semantic_model"])
        return m.summary()
    except (SemanticModelError, KeyError) as e:
        raise HTTPException(400, str(e)) from e


@app.post("/v1/generate")
async def generate(request: Request):
    body = await request.json()
    try:
        req = GenRequest(
            request_id=body.get("request_id") or uuid.uuid4().hex[:16],
            semantic_model=body["semantic_model"], messages=body["messages"],
            max_tokens=int(body.get("max_tokens") or 256), prompt=body.get("prompt"),
            prompt_tokens=body.get("prompt_tokens"), prompt_len=body.get("prompt_len"),
            block_hashes=body.get("block_hashes"), execute=bool(body.get("execute")),
            session_id=body.get("session_id"))
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(400, f"bad request: {e}") from e
    if not req.messages:
        raise HTTPException(400, "messages must not be empty")

    async def stream():
        async for ev in app.state.backend.generate(req):
            if ev["type"] == "start":
                ev["worker"] = WORKER_NAME
            elif ev["type"] == "done" and req.execute:
                try:
                    ev["result"] = await asyncio.to_thread(_run_sql, app, req.semantic_model, ev["sql"])
                except (SQLValidationError, SemanticModelError) as e:
                    ev["result"] = {"error": str(e)}
            yield _sse(ev)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/v1/sql")
async def sql_endpoint(request: Request):
    body = await request.json()
    try:
        return await asyncio.to_thread(_run_sql, app, body.get("semantic_model"), body["sql"])
    except SQLValidationError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/v1/results/{rid}")
async def result_arrow(rid: str):
    table: pa.Table | None = app.state.results.get(rid)
    if table is None:
        raise HTTPException(404, "unknown or expired result id")
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as w:
        for b in table.to_batches(max_chunksize=65536):
            w.write_batch(b)
    buf = sink.getvalue()
    view = memoryview(buf)                      # slices below are zero-copy views of the Arrow buffer

    def chunks():
        for i in range(0, len(view), 1 << 20):
            yield view[i:i + (1 << 20)]

    return StreamingResponse(chunks(), media_type="application/vnd.apache.arrow.stream",
                             headers={"X-Arrow-Rows": str(table.num_rows)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), log_level="warning",
                timeout_keep_alive=75, backlog=4096)
