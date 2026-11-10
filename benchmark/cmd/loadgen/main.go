// Command loadgen compares routing policies.
//
//	-mode sim   discrete-event simulation: Round-Robin vs Least-Loaded vs Radix KV-Aware on 5,000
//	            concurrent multi-turn sessions (uses the real router.RadixCache / Policy code).
//	-mode http  drive a LIVE router over HTTP/SSE (docker compose up) with the same session model and
//	            report the measured TTFT / throughput / worker-reported KV reuse.
package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/arcticflow/arcticflow/benchmark"
)

func main() {
	cfg := benchmark.DefaultConfig()
	mode := flag.String("mode", "sim", "sim | http")
	out := flag.String("out", "", "also write the Markdown report to this file")
	hw := flag.String("hw", "h100", "sim: h100 | a100 | a10g")
	flag.IntVar(&cfg.Workers, "workers", cfg.Workers, "sim: number of workers")
	flag.IntVar(&cfg.Sessions, "sessions", cfg.Sessions, "concurrent multi-turn sessions")
	flag.IntVar(&cfg.Turns, "turns", cfg.Turns, "turns per session")
	flag.IntVar(&cfg.Models, "models", cfg.Models, "distinct semantic models")
	flag.Float64Var(&cfg.RampSec, "ramp", cfg.RampSec, "seconds over which sessions start")
	flag.Float64Var(&cfg.ThinkSec, "think", cfg.ThinkSec, "mean think time between turns (s)")
	flag.IntVar(&cfg.CapacityBlocks, "capacity", cfg.CapacityBlocks, "sim: KV blocks per worker")
	flag.Float64Var(&cfg.ZipfS, "zipf", cfg.ZipfS, "model popularity skew (>1)")
	flag.Float64Var(&cfg.Alpha, "alpha", cfg.Alpha, "sim: KV overlap weight")
	flag.Float64Var(&cfg.Beta, "beta", cfg.Beta, "sim: queue-depth penalty")
	sweep := flag.Bool("sweep", false, "sim: also run the KV-capacity sensitivity sweep")
	target := flag.String("target", "http://localhost:8080", "http: router base URL")
	filler := flag.Int("filler-cols", 150, "http: filler columns per semantic model (controls YAML size)")
	execute := flag.Bool("execute", false, "http: also execute the SQL on the worker's DuckDB")
	maxConc := flag.Int("max-concurrency", 0, "http: cap on simultaneously open requests (0 = unlimited)")
	flag.Parse()
	cfg.HW = benchmark.HardwareByName(*hw)

	var md string
	t0 := time.Now()
	switch *mode {
	case "sim":
		md = benchmark.Markdown(cfg, benchmark.RunComparison(cfg))
		if *sweep {
			md += "\n### Sensitivity to KV capacity\n\n" + benchmark.SweepCapacity(cfg, []int{1500, 3600, 8000, 16000})
		}
	case "http":
		r, err := runHTTP(cfg, *target, *filler, *execute, *maxConc)
		if err != nil {
			fmt.Fprintln(os.Stderr, "loadgen:", err)
			os.Exit(1)
		}
		md = r
	default:
		fmt.Fprintln(os.Stderr, "unknown -mode")
		os.Exit(2)
	}
	fmt.Println(md)
	fmt.Fprintf(os.Stderr, "(wall %s)\n", time.Since(t0).Round(time.Millisecond))
	if *out != "" {
		if err := os.WriteFile(*out, []byte(md), 0o644); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
}
