# Benchmark results (discrete-event simulation)

**Setup:** 12 workers (h100, Qwen2.5-Coder-1.5B), 5000 concurrent sessions x 3 turns = 15000 queries, 24 semantic models (4000-12000 tokens, Zipf s=1.15), KV prefix capacity 3600 blocks (57k tokens) per worker, speculation 2.4 tok/step.

| Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p99 | Throughput (out tok/s) | KV hit ratio | GPU compute saved | GPU-s saved | GPU util | Load imbalance (max/mean) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Round-Robin | 689.4 ms | 3.95 s | 4.69 s | 7.56 s | 12228 | 54.6% | 53.9% | 653 | 59% | 1.02 |
| Least-Loaded | 571.4 ms | 2.52 s | 2.94 s | 5.81 s | 15427 | 54.9% | 54.2% | 657 | 73% | 1.01 |
| Radix KV-Aware | 3.7 ms | 6.7 ms | 8.7 ms | 181.8 ms | 17095 | 98.8% | 98.6% | 1195 | 49% | 1.56 |

**Radix KV-Aware vs Round-Robin:** TTFT p50 188.3x lower, p99 539.4x lower, throughput 1.40x, prefill FLOPs avoided +44.7 pts (542 extra GPU-seconds).

## Sensitivity to KV capacity

| KV capacity / worker | Round-Robin hit | Radix hit | RR TTFT p50 / p99 | Radix TTFT p50 / p99 |
|---:|---:|---:|---:|---:|
| 1500 blocks (24k tok) | 30.6% | 97.7% | 6.45 s / 13.79 s | 3.9 ms / 34.4 ms |
| 3600 blocks (57k tok) | 54.6% | 98.8% | 689.4 ms / 4.69 s | 3.7 ms / 8.7 ms |
| 8000 blocks (128k tok) | 80.5% | 98.9% | 5.7 ms / 249.8 ms | 3.6 ms / 8.0 ms |
| 16000 blocks (256k tok) | 96.7% | 98.9% | 3.5 ms / 94.9 ms | 3.6 ms / 7.8 ms |
