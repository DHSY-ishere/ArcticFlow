package router

import (
	"context"
	"errors"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

// Worker is one inference backend as seen by the router.
type Worker struct {
	ID   int
	Name string
	URL  string

	inflight       atomic.Int64 // active requests in the worker's batch queue
	inflightTokens atomic.Int64 // prompt+max-completion tokens of active requests
	healthy        atomic.Bool

	specProposed atomic.Int64
	specAccepted atomic.Int64
}

// NewWorker creates a healthy worker.
func NewWorker(id int, name, url string) *Worker {
	w := &Worker{ID: id, Name: name, URL: url}
	w.healthy.Store(true)
	return w
}

func (w *Worker) QueueDepth() int64        { return w.inflight.Load() }
func (w *Worker) InFlightTokens() int64    { return w.inflightTokens.Load() }
func (w *Worker) Healthy() bool            { return w.healthy.Load() }
func (w *Worker) SetHealthy(ok bool)       { w.healthy.Store(ok) }
func (w *Worker) SpecCounts() (p, a int64) { return w.specProposed.Load(), w.specAccepted.Load() }

// Decision is the outcome of routing one request.
type Decision struct {
	Worker        *Worker
	PromptTokens  int
	OverlapTokens int     // tokens of the prompt predicted to be KV-resident
	Ratio         float64 // OverlapTokens / PromptTokens
	Score         float64
	hashes        []uint64
	reservedToks  int64
}

// ErrNoWorkers is returned when every worker is unhealthy or excluded.
var ErrNoWorkers = errors.New("router: no healthy workers available")

// Policy chooses a worker for a tokenized prompt. Implementations must be
// safe for concurrent use. Route reserves capacity on the chosen worker;
// callers MUST call Release exactly once per successful Route.
type Policy interface {
	Name() string
	// Route picks a worker. `exclude` is a bitmask of workers to skip (retries).
	// `reserveTokens` is added to the worker's in-flight token gauge.
	Route(tokens []uint32, reserveTokens int, exclude uint64) (Decision, error)
	Release(d Decision)
}

// ---------------------------------------------------------------------------
// Prefix-aware balancer
// ---------------------------------------------------------------------------

// KVAwareBalancer scores every healthy worker i as
//
//	score_i = alpha * overlap_i/|prompt| - beta * queueDepth_i
//
// and routes to argmax. The first term rewards workers that already hold a
// long prefix of the prompt in their KV cache (skipping prefill FLOPs); the
// second penalizes deep batch queues, which bounds the hot-spotting that pure
// affinity routing causes when one semantic model is much more popular.
type KVAwareBalancer struct {
	Cache   *RadixCache
	Workers []*Worker
	Alpha   float64
	Beta    float64

	mu sync.Mutex // serializes score+reserve so concurrent routes see fresh depth
	rr uint64     // tie-break rotation
}

// NewKVAwareBalancer builds the balancer.
func NewKVAwareBalancer(cache *RadixCache, workers []*Worker, alpha, beta float64) *KVAwareBalancer {
	return &KVAwareBalancer{Cache: cache, Workers: workers, Alpha: alpha, Beta: beta}
}

func (b *KVAwareBalancer) Name() string { return "radix" }

func (b *KVAwareBalancer) Route(tokens []uint32, reserveTokens int, exclude uint64) (Decision, error) {
	hashes := HashBlocks(tokens, b.Cache.BlockSize())
	var mask uint64
	for _, w := range b.Workers {
		if w.Healthy() && exclude&(1<<uint(w.ID)) == 0 {
			mask |= 1 << uint(w.ID)
		}
	}
	if mask == 0 {
		return Decision{}, ErrNoWorkers
	}
	matched := b.Cache.Match(hashes, mask)

	b.mu.Lock()
	defer b.mu.Unlock()
	var best *Worker
	var bestScore, bestRatio float64
	var bestOverlap int
	b.rr++
	n := len(b.Workers)
	for k := 0; k < n; k++ {
		w := b.Workers[(int(b.rr)+k)%n] // rotating start => fair tie-breaks
		if mask&(1<<uint(w.ID)) == 0 {
			continue
		}
		overlap := matched[w.ID] * b.Cache.BlockSize()
		ratio := 0.0
		if len(tokens) > 0 {
			ratio = float64(overlap) / float64(len(tokens))
		}
		score := b.Alpha*ratio - b.Beta*float64(w.QueueDepth())
		if best == nil || score > bestScore {
			best, bestScore, bestRatio, bestOverlap = w, score, ratio, overlap
		}
	}
	d := Decision{Worker: best, PromptTokens: len(tokens), OverlapTokens: bestOverlap,
		Ratio: bestRatio, Score: bestScore, hashes: hashes, reservedToks: int64(reserveTokens)}
	best.inflight.Add(1)
	best.inflightTokens.Add(d.reservedToks)
	// Optimistic commit: the worker is about to prefill this prompt, so later
	// requests with the same prefix should chase it immediately instead of
	// waiting for this request to finish.
	b.Cache.Insert(best.ID, hashes)
	return d, nil
}

// WorkerScore is one worker's line in a routing explanation.
type WorkerScore struct {
	Worker        string  `json:"worker"`
	Healthy       bool    `json:"healthy"`
	OverlapTokens int     `json:"overlap_tokens"`
	OverlapRatio  float64 `json:"overlap_ratio"`
	QueueDepth    int64   `json:"queue_depth"`
	Score         float64 `json:"score"`
	Chosen        bool    `json:"chosen"`
}

// Explainer is implemented by policies that can show their scoring without side effects.
type Explainer interface {
	Explain(tokens []uint32) []WorkerScore
}

// Explain is a dry run of Route: it scores every worker but reserves nothing and does not touch
// the cache index. Ties resolve to the lowest worker ID (Route rotates its start, so the real
// pick among exact ties may differ).
func (b *KVAwareBalancer) Explain(tokens []uint32) []WorkerScore {
	hashes := HashBlocks(tokens, b.Cache.BlockSize())
	var mask uint64
	for _, w := range b.Workers {
		if w.Healthy() {
			mask |= 1 << uint(w.ID)
		}
	}
	matched := b.Cache.Match(hashes, mask)
	out := make([]WorkerScore, 0, len(b.Workers))
	best := -1
	for _, w := range b.Workers {
		s := WorkerScore{Worker: w.Name, Healthy: w.Healthy(), QueueDepth: w.QueueDepth()}
		if s.Healthy {
			s.OverlapTokens = matched[w.ID] * b.Cache.BlockSize()
			if len(tokens) > 0 {
				s.OverlapRatio = float64(s.OverlapTokens) / float64(len(tokens))
			}
			s.Score = b.Alpha*s.OverlapRatio - b.Beta*float64(s.QueueDepth)
			if best < 0 || s.Score > out[best].Score {
				best = len(out)
			}
		}
		out = append(out, s)
	}
	if best >= 0 {
		out[best].Chosen = true
	}
	return out
}

func (b *KVAwareBalancer) Release(d Decision) {
	d.Worker.inflight.Add(-1)
	d.Worker.inflightTokens.Add(-d.reservedToks)
}

// ---------------------------------------------------------------------------
// Baselines
// ---------------------------------------------------------------------------

// RoundRobin ignores cache state entirely; it is the baseline policy.
type RoundRobin struct {
	Workers []*Worker
	next    atomic.Uint64
}

func NewRoundRobin(workers []*Worker) *RoundRobin { return &RoundRobin{Workers: workers} }

func (r *RoundRobin) Name() string { return "roundrobin" }

func (r *RoundRobin) Route(tokens []uint32, reserveTokens int, exclude uint64) (Decision, error) {
	n := len(r.Workers)
	for k := 0; k < n; k++ {
		w := r.Workers[int(r.next.Add(1)-1)%n]
		if !w.Healthy() || exclude&(1<<uint(w.ID)) != 0 {
			continue
		}
		w.inflight.Add(1)
		w.inflightTokens.Add(int64(reserveTokens))
		return Decision{Worker: w, PromptTokens: len(tokens), reservedToks: int64(reserveTokens)}, nil
	}
	return Decision{}, ErrNoWorkers
}

func (r *RoundRobin) Release(d Decision) {
	d.Worker.inflight.Add(-1)
	d.Worker.inflightTokens.Add(-d.reservedToks)
}

// ---------------------------------------------------------------------------
// Health checking
// ---------------------------------------------------------------------------

// HealthChecker polls GET /healthz on each worker and flips the healthy flag.
// A worker that becomes unhealthy has its shadow cache entries dropped, since
// a restarted worker comes back with an empty KV cache.
type HealthChecker struct {
	Workers  []*Worker
	Cache    *RadixCache // may be nil (round-robin)
	Interval time.Duration
	Client   *http.Client
	// FailThreshold is the number of consecutive failed probes before a worker is marked
	// unhealthy (default 3). One slow /healthz on a busy worker must not evict its cache index.
	FailThreshold int

	mu    sync.Mutex
	fails map[int]int
}

// Run blocks until ctx is cancelled.
func (h *HealthChecker) Run(ctx context.Context) {
	t := time.NewTicker(h.Interval)
	defer t.Stop()
	h.probeAll(ctx)
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			h.probeAll(ctx)
		}
	}
}

func (h *HealthChecker) probeAll(ctx context.Context) {
	var wg sync.WaitGroup
	for _, w := range h.Workers {
		wg.Add(1)
		go func(w *Worker) {
			defer wg.Done()
			timeout := 3 * h.Interval
			if timeout < 3*time.Second {
				timeout = 3 * time.Second
			}
			c, cancel := context.WithTimeout(ctx, timeout)
			defer cancel()
			req, _ := http.NewRequestWithContext(c, http.MethodGet, w.URL+"/healthz", nil)
			resp, err := h.Client.Do(req)
			ok := err == nil && resp.StatusCode == http.StatusOK
			if resp != nil {
				resp.Body.Close()
			}
			thr := h.FailThreshold
			if thr <= 0 {
				thr = 3
			}
			h.mu.Lock()
			if h.fails == nil {
				h.fails = map[int]int{}
			}
			if ok {
				h.fails[w.ID] = 0
			} else {
				h.fails[w.ID]++
			}
			down := !ok && h.fails[w.ID] >= thr
			h.mu.Unlock()
			if ok {
				w.SetHealthy(true)
			} else if down {
				was := w.Healthy()
				w.SetHealthy(false)
				if was && h.Cache != nil {
					h.Cache.RemoveWorker(w.ID)
				}
			}
		}(w)
	}
	wg.Wait()
}
