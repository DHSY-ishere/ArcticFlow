package benchmark

import (
	"fmt"
	"strings"
)

// Markdown renders setup + comparison table + delta line. results[0] is the baseline.
func Markdown(cfg Config, results []*Result) string {
	var b strings.Builder
	fmt.Fprintf(&b, "**Setup:** %d workers (%s, %s), %d concurrent sessions x %d turns = %d queries, %d semantic models "+
		"(%d-%d tokens, Zipf s=%.2f), KV prefix capacity %d blocks (%dk tokens) per worker, speculation %.1f tok/step.\n\n",
		cfg.Workers, cfg.HW.Name, "Qwen2.5-Coder-1.5B", cfg.Sessions, cfg.Turns, cfg.Sessions*cfg.Turns, cfg.Models,
		cfg.ModelTokMin, cfg.ModelTokMax, cfg.ZipfS, cfg.CapacityBlocks, cfg.CapacityBlocks*16/1000, cfg.SpecTokPerStep)
	b.WriteString(Table(results))
	if len(results) >= 2 {
		base, last := results[0], results[len(results)-1]
		fmt.Fprintf(&b, "\n**%s vs %s:** TTFT p50 %.1fx lower, p99 %.1fx lower, throughput %.2fx, "+
			"prefill FLOPs avoided +%.1f pts (%.0f extra GPU-seconds).\n",
			last.Policy, base.Policy,
			Percentile(base.TTFTms, 50)/Percentile(last.TTFTms, 50), Percentile(base.TTFTms, 99)/Percentile(last.TTFTms, 99),
			last.Throughput()/base.Throughput(), 100*(last.SavedFrac()-base.SavedFrac()),
			last.GPUSecondsSaved()-base.GPUSecondsSaved())
	}
	return b.String()
}

// Table renders the metrics table: TTFT p50/p95/p99, throughput, KV hit ratio, GPU compute saved.
func Table(results []*Result) string {
	var b strings.Builder
	b.WriteString("| Policy | TTFT p50 | TTFT p95 | TTFT p99 | E2E p99 | Throughput (out tok/s) | KV hit ratio | GPU compute saved | GPU-s saved | GPU util | Load imbalance (max/mean) |\n")
	b.WriteString("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
	for _, r := range results {
		fmt.Fprintf(&b, "| %s | %s | %s | %s | %s | %.0f | %.1f%% | %.1f%% | %.0f | %.0f%% | %.2f |\n", r.Policy,
			ms(Percentile(r.TTFTms, 50)), ms(Percentile(r.TTFTms, 95)), ms(Percentile(r.TTFTms, 99)),
			ms(Percentile(r.E2Ems, 99)), r.Throughput(), 100*r.HitRatio(), 100*r.SavedFrac(), r.GPUSecondsSaved(),
			100*r.Utilization, r.BusyImbalance)
	}
	return b.String()
}

func ms(v float64) string {
	if v >= 1000 {
		return fmt.Sprintf("%.2f s", v/1000)
	}
	return fmt.Sprintf("%.1f ms", v)
}

// RunComparison runs round-robin, least-loaded and radix policies on identical workloads.
func RunComparison(cfg Config) []*Result {
	return []*Result{
		Run(cfg, "Round-Robin", RoundRobinFactory),
		Run(cfg, "Least-Loaded", LeastLoadedFactory(cfg)),
		Run(cfg, "Radix KV-Aware", RadixFactory(cfg)),
	}
}

// SweepCapacity compares round-robin vs radix as per-worker KV capacity grows relative to the
// working set (sum of distinct semantic-model prompt sizes). It shows where prefix-aware routing
// stops mattering: when every worker can hold every model, round-robin catches up.
func SweepCapacity(cfg Config, capacities []int) string {
	var b strings.Builder
	b.WriteString("| KV capacity / worker | Round-Robin hit | Radix hit | RR TTFT p50 / p99 | Radix TTFT p50 / p99 |\n|---:|---:|---:|---:|---:|\n")
	for _, c := range capacities {
		cfg.CapacityBlocks = c
		rr := Run(cfg, "rr", RoundRobinFactory)
		rx := Run(cfg, "radix", RadixFactory(cfg))
		fmt.Fprintf(&b, "| %d blocks (%dk tok) | %.1f%% | %.1f%% | %s / %s | %s / %s |\n", c, c*16/1000,
			100*rr.HitRatio(), 100*rx.HitRatio(),
			ms(Percentile(rr.TTFTms, 50)), ms(Percentile(rr.TTFTms, 99)),
			ms(Percentile(rx.TTFTms, 50)), ms(Percentile(rx.TTFTms, 99)))
	}
	return b.String()
}
