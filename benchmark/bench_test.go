package benchmark

import (
	"os"
	"testing"

	"github.com/arcticflow/arcticflow/router"
)

func smallConfig() Config {
	c := DefaultConfig()
	c.Workers, c.Sessions, c.Turns, c.Models = 4, 400, 3, 12
	c.RampSec = 10
	c.CapacityBlocks = 2500
	return c
}

func TestSimulationDeterministic(t *testing.T) {
	c := smallConfig()
	a, b := Run(c, "x", RadixFactory(c)), Run(c, "x", RadixFactory(c))
	if a.Requests != b.Requests || a.CachedTokens != b.CachedTokens || Percentile(a.TTFTms, 99) != Percentile(b.TTFTms, 99) {
		t.Fatal("simulation is not deterministic for a fixed seed")
	}
	if a.Requests != c.Sessions*c.Turns {
		t.Fatalf("expected %d completed requests, got %d", c.Sessions*c.Turns, a.Requests)
	}
}

func TestRadixBeatsRoundRobin(t *testing.T) {
	c := smallConfig()
	rr := Run(c, "rr", RoundRobinFactory)
	rx := Run(c, "radix", RadixFactory(c))
	t.Logf("RR hit %.3f p50 %.1fms p99 %.1fms | radix hit %.3f p50 %.1fms p99 %.1fms", rr.HitRatio(), Percentile(rr.TTFTms, 50),
		Percentile(rr.TTFTms, 99), rx.HitRatio(), Percentile(rx.TTFTms, 50), Percentile(rx.TTFTms, 99))
	if rx.HitRatio() <= rr.HitRatio()+0.15 {
		t.Fatalf("radix hit ratio %.3f should clearly exceed round-robin %.3f", rx.HitRatio(), rr.HitRatio())
	}
	if Percentile(rx.TTFTms, 50) >= Percentile(rr.TTFTms, 50) || rx.SavedFrac() <= rr.SavedFrac() {
		t.Fatal("radix must lower median TTFT and avoid more prefill FLOPs")
	}
}

// With a cache big enough to hold every model on every worker, affinity cannot help much:
// the benchmark must reflect that honestly instead of being tuned to always win.
func TestAdvantageVanishesWhenEverythingFits(t *testing.T) {
	c := smallConfig()
	c.CapacityBlocks = 200000
	rr, rx := Run(c, "rr", RoundRobinFactory), Run(c, "radix", RadixFactory(c))
	if d := rx.HitRatio() - rr.HitRatio(); d > 0.10 {
		t.Fatalf("with unlimited cache the hit-ratio gap should be small, got %.3f", d)
	}
}

func TestHotSpotPenaltyBoundsImbalance(t *testing.T) {
	c := smallConfig()
	c.ZipfS = 3.0 // extreme skew: one model dominates
	noPenalty := c
	noPenalty.Beta = 0
	pen, raw := Run(c, "pen", RadixFactory(c)), Run(noPenalty, "raw", RadixFactory(noPenalty))
	t.Logf("imbalance with beta=%.3f: %.2f ; beta=0: %.2f", c.Beta, pen.BusyImbalance, raw.BusyImbalance)
	if pen.BusyImbalance >= raw.BusyImbalance {
		t.Fatal("queue-depth penalty must reduce load imbalance under skewed popularity")
	}
}

// TestHeadline5000 is the published benchmark: 5,000 concurrent multi-turn sessions.
// Run: go test ./benchmark -run Headline -v   (ARCTIC_WRITE_RESULTS=1 writes benchmark/RESULTS.md)
func TestHeadline5000(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode")
	}
	c := DefaultConfig()
	md := Markdown(c, RunComparison(c))
	t.Log("\n" + md)
	if os.Getenv("ARCTIC_WRITE_RESULTS") == "1" {
		sweep := SweepCapacity(c, []int{1500, 3600, 8000, 16000})
		_ = os.WriteFile("RESULTS.md", []byte("# Benchmark results (discrete-event simulation)\n\n"+md+"\n## Sensitivity to KV capacity\n\n"+sweep), 0o644)
	}
}

func BenchmarkSim5000Radix(b *testing.B) {
	c := DefaultConfig()
	for i := 0; i < b.N; i++ {
		r := Run(c, "radix", RadixFactory(c))
		b.ReportMetric(Percentile(r.TTFTms, 50), "ttft_p50_ms")
		b.ReportMetric(Percentile(r.TTFTms, 99), "ttft_p99_ms")
		b.ReportMetric(100*r.HitRatio(), "hit_%")
		b.ReportMetric(r.Throughput(), "out_tok/s")
	}
}

func BenchmarkSim5000RoundRobin(b *testing.B) {
	c := DefaultConfig()
	for i := 0; i < b.N; i++ {
		r := Run(c, "rr", RoundRobinFactory)
		b.ReportMetric(Percentile(r.TTFTms, 50), "ttft_p50_ms")
		b.ReportMetric(Percentile(r.TTFTms, 99), "ttft_p99_ms")
		b.ReportMetric(100*r.HitRatio(), "hit_%")
		b.ReportMetric(r.Throughput(), "out_tok/s")
	}
}

// Routing decision cost at production scale (8k-token prompts, 12 workers).
func BenchmarkRouteDecision(b *testing.B) {
	ws := make([]*router.Worker, 12)
	for i := range ws {
		ws[i] = router.NewWorker(i, "w", "")
	}
	cache := router.NewRadixCache(12, 16, 1<<20)
	bal := router.NewKVAwareBalancer(cache, ws, 1, 0.004)
	toks := SynthTokens(9, 1, 8000)
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			d, _ := bal.Route(toks, 100, 0)
			bal.Release(d)
		}
	})
}
