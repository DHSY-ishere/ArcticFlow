package router

import (
	"net/http"
	"sync/atomic"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Metrics holds every Prometheus series the router exports.
type Metrics struct {
	Registry *prometheus.Registry

	promptTokens atomic.Int64
	cachedTokens atomic.Int64

	ttft         *prometheus.HistogramVec
	requests     *prometheus.CounterVec
	promptTotal  prometheus.Counter
	cachedTotal  prometheus.Counter
	routeOverlap prometheus.Histogram
	retries      prometheus.Counter
}

// NewMetrics registers all collectors for the given worker set and cache.
func NewMetrics(workers []*Worker, cache *RadixCache) *Metrics {
	reg := prometheus.NewRegistry()
	m := &Metrics{Registry: reg}

	// arctic_kv_cache_hit_ratio: worker-reported (ground truth) cached prompt
	// tokens divided by prompt tokens, cumulative since router start.
	reg.MustRegister(prometheus.NewGaugeFunc(prometheus.GaugeOpts{
		Name: "arctic_kv_cache_hit_ratio",
		Help: "Fraction of prompt tokens served from worker KV cache (worker-reported, cumulative).",
	}, func() float64 {
		p := m.promptTokens.Load()
		if p == 0 {
			return 0
		}
		return float64(m.cachedTokens.Load()) / float64(p)
	}))

	m.ttft = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "arctic_ttft_latency_ms",
		Help:    "Time to first token, measured at the router (ms).",
		Buckets: prometheus.ExponentialBuckets(2, 1.6, 18), // 2ms .. ~9s
	}, []string{"worker"})
	m.requests = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "arctic_requests_total", Help: "Requests by worker and outcome.",
	}, []string{"worker", "outcome"})
	m.promptTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "arctic_prompt_tokens_total", Help: "Prompt tokens processed (cached + computed)."})
	m.cachedTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "arctic_cached_tokens_total", Help: "Prompt tokens served from KV cache."})
	m.routeOverlap = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name: "arctic_route_predicted_overlap_ratio", Help: "Radix-predicted KV overlap ratio at routing time.",
		Buckets: prometheus.LinearBuckets(0, 0.1, 11)})
	m.retries = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "arctic_route_retries_total", Help: "Upstream attempts retried on another worker."})
	reg.MustRegister(m.ttft, m.requests, m.promptTotal, m.cachedTotal, m.routeOverlap, m.retries)

	for _, w := range workers {
		w := w
		lbl := prometheus.Labels{"worker": w.Name}
		reg.MustRegister(
			prometheus.NewGaugeFunc(prometheus.GaugeOpts{
				Name: "arctic_active_in_flight_tokens", ConstLabels: lbl,
				Help: "Prompt+completion tokens of requests currently in flight on the worker.",
			}, func() float64 { return float64(w.InFlightTokens()) }),
			prometheus.NewGaugeFunc(prometheus.GaugeOpts{
				Name: "arctic_queue_depth", ConstLabels: lbl,
				Help: "Requests currently in the worker's batch queue (router view).",
			}, func() float64 { return float64(w.QueueDepth()) }),
			prometheus.NewGaugeFunc(prometheus.GaugeOpts{
				Name: "arctic_speculative_acceptance_rate", ConstLabels: lbl,
				Help: "Accepted / proposed draft tokens of N-gram speculative decoding (cumulative).",
			}, func() float64 {
				p, a := w.SpecCounts()
				if p == 0 {
					return 0
				}
				return float64(a) / float64(p)
			}),
			prometheus.NewGaugeFunc(prometheus.GaugeOpts{
				Name: "arctic_worker_healthy", ConstLabels: lbl, Help: "1 if the worker passes health checks.",
			}, func() float64 {
				if w.Healthy() {
					return 1
				}
				return 0
			}),
		)
	}
	if cache != nil {
		reg.MustRegister(
			prometheus.NewGaugeFunc(prometheus.GaugeOpts{
				Name: "arctic_radix_nodes", Help: "Nodes in the router's radix index."},
				func() float64 { return float64(cache.Stats().Nodes) }),
			prometheus.NewCounterFunc(prometheus.CounterOpts{
				Name: "arctic_radix_evictions_total", Help: "Blocks evicted from the radix index by LRU."},
				func() float64 { return float64(cache.Stats().Evictions) }),
		)
	}
	return m
}

// Handler serves /metrics.
func (m *Metrics) Handler() http.Handler {
	return promhttp.HandlerFor(m.Registry, promhttp.HandlerOpts{})
}

// ObserveRoute records the routing-time prediction.
func (m *Metrics) ObserveRoute(d Decision) {
	if m == nil {
		return
	}
	m.routeOverlap.Observe(d.Ratio)
}

// ObserveStart records worker-reported prefix reuse.
func (m *Metrics) ObserveStart(promptTokens, cachedTokens int) {
	if m == nil {
		return
	}
	m.promptTokens.Add(int64(promptTokens))
	m.cachedTokens.Add(int64(cachedTokens))
	m.promptTotal.Add(float64(promptTokens))
	m.cachedTotal.Add(float64(cachedTokens))
}

// ObserveTTFT records time-to-first-token in milliseconds.
func (m *Metrics) ObserveTTFT(worker string, ms float64) {
	if m == nil {
		return
	}
	m.ttft.WithLabelValues(worker).Observe(ms)
}

// ObserveOutcome counts a finished request.
func (m *Metrics) ObserveOutcome(worker, outcome string) {
	if m == nil {
		return
	}
	m.requests.WithLabelValues(worker, outcome).Inc()
}

// ObserveRetry counts a failover.
func (m *Metrics) ObserveRetry() {
	if m != nil {
		m.retries.Inc()
	}
}
