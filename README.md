# ArcticFlow

**A distributed KV-cache-aware LLM inference router and a grammar-constrained semantic SQL engine**, modeled on Snowflake Cortex Analyst (YAML semantic models → SQL) and vLLM/SGLang RadixAttention (prefix-reuse scheduling).

The workload that motivates it: every analytical question to a text-to-SQL service is prefixed by the *same* multi-thousand-token YAML semantic model. Prefilling that prefix is >95% of the FLOPs of a request, and it is identical across requests. If the router sends a request to a worker that already holds the prefix's KV pages, prefill collapses to the few dozen tokens of the question.

```
                    ┌────────────────────────────── arctic-router (Go) ───────────────────────────────┐
 HTTP/SSE  ─────────►  tokenize ─► block-hash ─► Radix index ─► score(w) = α·overlap − β·depth ─► pick │
 gRPC      ─────────►                  │            (RWMutex,        │            ▲                     │
 (server-stream)    │                  │             LRU/worker)     │            │ in-flight counters  │
                    │                  ▼                             ▼            │                     │
                    │        optimistic insert ◄─────────── SSE reverse proxy ────┘   /metrics ──► Prometheus ─► Grafana
                    └──────────────────────────────────────────────┬───────────────────────────────────┘
                              cached_tokens / spec stats (truth) ▲ │  POST /v1/generate  (SSE)
                    ┌──────────────────────┬─────────────────────┴─┴────────────────────┐
                    ▼                      ▼                                            ▼
            ┌─ worker w1 ─┐        ┌─ worker w2 ─┐                              ┌─ worker w3 ─┐
            │ FastAPI     │        │     ...     │                              │     ...     │
            │ ┌─────────┐ │        └─────────────┘                              └─────────────┘
            │ │backend  │ │   sim: continuous-batching engine + roofline latency + paged KV (LRU, refcount)
            │ │sim | hf │ │   hf : transformers + real prefix-KV reuse + FSM-masked logits + n-gram speculation
            │ └────┬────┘ │
            │ semantic_compiler: YAML ─► regex ─► lazy DFA ─► TokenFSM        ┌── DuckDB (read-only, locked)
            │ speculative_sql : grammar jump-forward + SQL n-gram drafter ───►│   json_serialize_sql SELECT-only gate
            │ executor ───────────────────────────────────────────────────────┘   └─► Arrow table ─► Flight :8815 / IPC
            └─────────────┘
```

## Repository layout

| Path | What |
|---|---|
| `router/radix_cache.go` | Thread-safe radix index of chained FNV-1a block hashes; per-worker LRU leaf-first eviction |
| `router/balancer.go` | `KVAwareBalancer` (α/β scoring), `RoundRobin`, health checker |
| `router/proxy.go`, `grpc.go` | HTTP/SSE reverse proxy with failover; gRPC `arcticflow.v1.Router/Generate` (server-streaming) |
| `router/metrics.go` | Prometheus series (below) |
| `engine/semantic_compiler.py`, `regex_fsm.py` | Cortex-style YAML → typed model → SQL regex → lazily-determinised DFA → token-level FSM |
| `engine/worker.py` | FastAPI worker: SSE generation, DuckDB execution, Arrow IPC + Flight results |
| `engine/sim_engine.py`, `latency_model.py`, `kv_blocks.py` | CPU latency simulator: continuous batching, roofline step time, paged KV with prefix sharing |
| `engine/hf_backend.py` | Real HuggingFace backend (Qwen2.5-Coder-1.5B by default) |
| `engine/speculative_sql.py` | N-gram + grammar-jump-forward speculative decoding for SQL |
| `engine/sql_planner.py` | Deterministic NL→SQL planner used as the simulator's "model" |
| `benchmark/` | Discrete-event fleet simulator driving the *real* router code, workload generator, `bench_test.go`, `cmd/loadgen` |
| `docker-compose.yml`, `deploy/` | Router + 3 workers + Prometheus + provisioned dark Grafana |

## Quickstart

```bash
docker compose up --build                      # radix KV-aware routing; Grafana http://localhost:3000, router :8080
ARCTIC_POLICY=roundrobin docker compose up --build    # baseline for A/B

curl -N localhost:8080/v1/generate -H 'content-type: application/json' -d @request.json
# request.json: {"semantic_model": "<yaml text>", "messages":[{"role":"user","content":"top 5 customers by revenue"}], "execute": true}

go test ./...                                  # router + benchmark tests (use -race for the router)
cd engine && pip install -r requirements-dev.txt && pytest
go run ./benchmark/cmd/loadgen -sweep          # the 5,000-session simulation (≈1 min)
go run ./benchmark/cmd/loadgen -mode http -target http://localhost:8080 -sessions 800   # live run
```

Real model instead of the simulator: `ARCTIC_REAL=1 ARCTIC_BACKEND=hf docker compose up --build` (installs torch/transformers; downloads `Qwen/Qwen2.5-Coder-1.5B-Instruct`). **This path is implemented but was not executed in development** (no torch/GPU/model download available) — treat it as untested.

Metrics (`/metrics` on the router): `arctic_kv_cache_hit_ratio`, `arctic_ttft_latency_ms` (histogram), `arctic_active_in_flight_tokens{worker}`, `arctic_speculative_acceptance_rate{worker}`, plus `arctic_queue_depth`, `arctic_prompt_tokens_total`, `arctic_cached_tokens_total`, `arctic_radix_nodes`, `arctic_radix_evictions_total`, `arctic_route_retries_total`, `arctic_requests_total`.

## Radix-tree routing

**Block hashing.** A prompt of tokens `t₁…t_n` is cut into full pages of `B=16` tokens (a partial tail page is never shareable). Page `i` is identified by a *chained* hash

```
h_0 = FNV-1a offset basis
h_i = FNV-1a-64( h_{i-1} ‖ t_{(i-1)B+1} … t_{iB} )        (each 32-bit token fed little-endian, byte-wise)
```

so `h_i` commits to the whole prefix: `h_i(P) = h_i(Q)` ⇔ the first `iB` tokens of `P` and `Q` agree (up to 2⁻⁶⁴ collisions). A prompt is therefore a *path* `h_1 → h_2 → …` in a trie whose alphabet is page hashes — a radix tree at page granularity. Each node carries a 64-bit residency bitmask (which workers hold that page) and, per worker, an element in that worker's LRU list.

**Overlap.** For worker `w`, `o_w = max k : ∀ i≤k, w ∈ holders(node_i)` — the contiguous resident prefix. One root-to-leaf walk computes `o_w` for *all* workers at once: keep an `alive` bitmask, `alive &= node.holders` at each depth, and record depth−1 for every worker that drops out. Cost `O(depth)` under an `RLock`, independent of worker count (≤64). Measured: see `BenchmarkRoute`.

**Score.**

```
ratio_w = o_w · B / |prompt|                       ∈ [0,1]  (KV_Overlap_Ratio)
score_w = α · ratio_w  −  β · depth_w              depth_w = requests in w's batch queue
route   = argmax_w score_w      (ties: rotating start index, so cold prompts spread)
```

Between the cache owner *o* and a cold worker *c*, the router prefers *o* iff `α(ratio_o − ratio_c) > β(depth_o − depth_c)`. With full overlap the owner keeps traffic until it is `α/β` requests deeper than the alternative — the **spill threshold**. Choose it so queueing delay on the owner ≈ prefill time saved: `β/α ≈ τ / T_prefill`, where `τ` is the marginal queueing delay of one more queued request and `T_prefill = (2·P·n)/(F·MFU)` is the cold prefill time (e.g. 8k tokens, P=1.54B, A100 at 45% MFU ≈ 0.19 s). Benchmarks use α=1, β=0.004 (spill past ~250 queued requests ≈ one full batch). When the owner spills, the new worker prefills once and becomes a second replica, which is exactly hot-model replication.

**Eviction.** Each worker has an LRU list capped at its page budget (mirroring the worker's real KV pool). A path is touched *deepest page first*, so a parent is always more recent than its children and the LRU tail is always a leaf — the same leaf-first property as vLLM/SGLang. Nodes held by nobody and having no children are pruned.

**Optimistic commit.** The router inserts the prompt for the chosen worker at routing time (before prefill finishes) so a burst of same-prefix requests chases one worker instead of all missing together. Ground truth comes back in the worker's `start` event (`cached_tokens`), which drives `arctic_kv_cache_hit_ratio`; the router's index is a *prediction*, not the source of truth.

## Semantic compiler and constrained decoding

`semantic_compiler.py` parses Cortex Analyst-style YAML (`tables` → `dimensions` / `time_dimensions` / `facts` / `metrics`, `relationships`, top-level `metrics`, `verified_queries`), qualifies metric expressions, and emits one regex describing every admissible statement:

```
SELECT item(, item)* FROM tbl( (LEFT )?JOIN tbl ON a.x = b.x)* ( WHERE cond( AND cond)*)?
       ( GROUP BY key(, key)*)? ( ORDER BY key( ASC| DESC)?(, …)*)? ( LIMIT [1-9][0-9]{0,3})?
```

Only declared columns/metrics/join predicates are reachable; literals are typed per column (string/number/date/bool). `regex_fsm.py` is a from-scratch Thompson-NFA → **lazily determinised** DFA (a 150-column model has ~16k DFA states, but one statement visits ~250; eager construction cost 4 s of CPU per model). `TokenFSM` lifts the character DFA to a vocabulary: `allowed(state)` = tokens whose whole string can be consumed from `state` — the logits mask. The grammar is *regular*, so cross-clause rules (every non-aggregate select item must appear in `GROUP BY`) are enforced by DuckDB's binder, not the grammar.

## Speculative decoding for SQL

Drafts come from (1) **grammar jump-forward** — if the DFA has exactly one outgoing character, those characters are certain; complete pieces become free draft tokens — then (2) an **n-gram table** (n=4…1, most-frequent continuation, chained to `k=8`) over `SQL boilerplate + verified queries + the YAML + output so far`. Verification is exact greedy matching in one forward pass, so output is identical to non-speculative decoding; with iid per-token acceptance `a`, expected tokens/step is `(1−a^{k+1})/(1−a)`. Per-worker `spec_proposed/accepted` flow to `arctic_speculative_acceptance_rate`.

Measured on the example model: **~2.4 tokens/forward step** (e.g. 44 tokens in 18 steps). The *acceptance rate* metric reads low (~13–22%) because the drafter always proposes up to `k=8` tokens and the tail of most drafts is rejected; rate and speedup are different quantities. The benchmark's `SpecTokPerStep=2.4` is taken from this measurement.

## Latency & KV-memory model (simulator)

One engine step = decode slots + chunked-prefill tokens (budget 8192). Roofline:

```
FLOPs = 2·P·n_tok + 4·L·H·Σ n_i·(ctx_i + n_i/2)
Bytes = P·dtype + (Σ ctx_i + n_tok)·kv_bytes_per_token        kv_bytes/token = 2·L·kv_heads·d_head·dtype = 28,672 B (Qwen2.5-1.5B, bf16)
t_step = max(FLOPs/(peak·MFU), Bytes/(BW·MBU)) + launch_overhead
```

Prefill is compute-bound, decode bandwidth-bound; a prefix hit deletes prefill FLOPs but not KV-read bytes. The KV pool is a paged allocator with ref-counted shared prefix pages, leaf-first LRU, and admission back-pressure (private pages are reserved at admission, no preemption). `go` and `python` implementations share the same constants.

## Benchmark: 5,000 concurrent multi-turn sessions

`benchmark/` is a discrete-event simulation (virtual time) that drives the **real** `router.RadixCache` and `router.Policy` implementations against modeled GPU workers. 15,000 queries; each session = 3 turns whose prompts share the model prefix and grow with history; 24 semantic models of 4–12k tokens with Zipf popularity (s=1.15). 12 H100-class workers, 57k-token KV prefix budget each. Deterministic (seeded); regenerate with `ARCTIC_WRITE_RESULTS=1 go test ./benchmark -run Headline -v`.

| Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p99 | Throughput (out tok/s) | KV hit | GPU compute saved | GPU-s saved | GPU util | Imbalance (max/mean) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Round-Robin | 689.4 ms | 3.95 s | 4.69 s | 7.56 s | 12,228 | 54.6% | 53.9% | 653 | 59% | 1.02 |
| Least-Loaded | 571.4 ms | 2.52 s | 2.94 s | 5.81 s | 15,427 | 54.9% | 54.2% | 657 | 73% | 1.01 |
| **Radix KV-Aware** | **3.7 ms** | **6.7 ms** | **8.7 ms** | **181.8 ms** | **17,095** | **98.8%** | **98.6%** | **1,195** | 49% | 1.56 |

"GPU compute saved" = prefill FLOPs avoided vs. a no-cache baseline. Throughput is output tokens ÷ makespan; because arrivals are think-time-paced, it is only partly capacity-bound — the clearest capacity signal is the tail latency and GPU-seconds.

**Read this before quoting the ratios.** The gap is a function of *working set ÷ per-worker KV capacity*, and the scenario sits in the regime where round-robin thrashes. Sweeping capacity (same workload):

| KV capacity / worker | Round-Robin hit | Radix hit | RR TTFT p50 / p99 | Radix TTFT p50 / p99 |
|---:|---:|---:|---:|---:|
| 1500 blocks (24k tok) | 30.6% | 97.7% | 6.45 s / 13.79 s | 3.9 ms / 34.4 ms |
| 3600 blocks (57k tok) | 54.6% | 98.8% | 689.4 ms / 4.69 s | 3.7 ms / 8.7 ms |
| 8000 blocks (128k tok) | 80.5% | 98.9% | 5.7 ms / 249.8 ms | 3.6 ms / 8.0 ms |
| 16000 blocks (256k tok) | 96.7% | 98.9% | 3.5 ms / 94.9 ms | 3.6 ms / 7.8 ms |

When every worker can hold every model (≥~256k tokens here) round-robin catches up on hit ratio and the advantage shrinks to tail latency. Radix routing also concentrates load (imbalance 1.56 vs 1.02), the price of affinity that the β term bounds. Simplifications: the router's shadow index and each simulated worker's cache have identical capacity (a real worker also pins pages of running requests); step times are modeled, not measured on silicon.

## Live-stack measurements (Python simulator workers)

Honest numbers from running the real stack (3 Python workers + Go router, Docker Desktop on a 14-core Windows laptop, 12 models × ~6.4k-token YAML):

* Routing works end to end: repeated prompt → same worker, `cached_tokens` 880/884; 0 errors on warm runs.
* Closed-loop 300 sessions (no think time), warm: **~180 req/s**, TTFT p50 ≈ 350 ms, hit ratio ≈ 96%.
* This is **well below** 1,000 req/s. Each Python worker is a single-core asyncio process doing JSON + SSE framing of ~25 KB prompts; the router and the simulated GPU are nowhere near saturated (router CPU <1%). The backend alone does ~690 req/s per process with HTTP removed. Scaling out = more worker containers. The 5,000-session result above comes from the discrete-event simulator, not this live path.
* Cold start is slow: the first burst triggers per-model grammar/plan warm-up on every worker.

## Safety of generated SQL

1. Output is grammar-constrained (token FSM in HF mode; asserted against the compiled DFA in sim mode).
2. DuckDB's own parser (`json_serialize_sql`) must yield exactly one `SELECT_NODE`; DDL/DML/`COPY`/`ATTACH`/`PRAGMA` never parse as one.
3. `enable_external_access=false` + `lock_configuration=true` (no file/URL access, settings immutable), query timeout via `interrupt`, row cap.

Tables absent from DuckDB are seeded with deterministic synthetic data derived from the YAML (demo/load-test convenience); real deployments attach their own tables.

Results are Arrow tables: the worker returns `flight://host:8815` + ticket (`DoGet` streams record batches) and an Arrow IPC endpoint (`/v1/results/{worker}/{id}`, proxied byte-for-byte by the router). DuckDB hands over columnar buffers without row-wise re-encoding; the HTTP IPC path serializes once into a contiguous buffer and streams slices of it. It is "no per-row copies", not literally zero memcpy end to end.

## Configuration

| Env | Default | Meaning |
|---|---|---|
| `ARCTIC_POLICY` | `radix` | `radix` or `roundrobin` |
| `ARCTIC_ALPHA` / `ARCTIC_BETA` | `1.0` / `0.01` | overlap weight / per-queued-request penalty |
| `ARCTIC_CAPACITY_BLOCKS` (router) = `ARCTIC_KV_BLOCKS` (worker) | `4096` | KV pages per worker; keep equal |
| `ARCTIC_BACKEND` | `sim` | `sim` or `hf` |
| `ARCTIC_SIM_HW` | `h100` in compose | `h100`/`a100`/`a10g`/`l4` |
| `ARCTIC_SIM_TIME_SCALE` | `1.0` | divide simulated latencies (faster tests) |
| `ARCTIC_MODEL` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | HF backend model |

## Known limitations

* The simulator's "model" is a rule-based planner (`sql_planner.py`), not an LLM; it makes load tests reproducible but says nothing about SQL quality.
* The HF backend (real prefix-KV reuse, FSM-masked logits, speculative verify with KV crop) is written but **unexecuted**; no vLLM backend exists.
* gRPC is real HTTP/2 server-streaming but uses a JSON codec (`application/grpc+json`) instead of protobuf, to avoid a protoc toolchain.
* Prometheus/Grafana containers could not be pulled in the build environment (registry network failures), so the dashboard JSON, provisioning files and scrape config are validated only structurally (`docker compose config`, JSON parse) — the router/worker containers *were* built and run. Regenerate the dashboard with `python deploy/grafana/make_dashboard.py`.
* Router and worker tokenizers are a deterministic pseudo-tokenizer (verified byte-identical across Go and Python by `test_go_python_parity`), not the model's BPE; the HF backend prefix-caches at the YAML-system-block boundary rather than per page.

## Tests

`go test -race ./router` (radix semantics, leaf-first LRU, concurrency, balancer hot-spot shedding, SSE proxy, failover, gRPC streaming, metrics) · `go test ./benchmark` (determinism, radix-vs-RR, advantage-vanishes-when-cache-fits, imbalance bound) · `pytest engine/tests` (regex/FSM, grammar acceptance & rejection, planner→grammar→DuckDB for 8 questions incl. follow-ups, SQL-injection rejection, speculation losslessness, KV manager, roofline, worker SSE + Arrow end to end, Go↔Python tokenizer/hash parity).
