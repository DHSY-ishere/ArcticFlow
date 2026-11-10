package benchmark

import (
	"container/heap"
	"math"
	"math/rand"
	"sort"

	"github.com/arcticflow/arcticflow/router"
)

// Config describes one fleet + workload.
type Config struct {
	Workers        int
	Sessions       int // concurrent multi-turn sessions
	Turns          int // turns per session
	Models         int // distinct semantic models (tenants)
	ModelTokMin    int // semantic-model prompt size range (tokens)
	ModelTokMax    int
	ZipfS          float64 // popularity skew across models (>1)
	RampSec        float64 // sessions start uniformly over [0, RampSec]
	ThinkSec       float64 // mean think time between turns (exponential)
	QuestionTok    int
	AnswerTokMin   int // output (SQL) tokens
	AnswerTokMax   int
	SpecTokPerStep float64 // mean tokens emitted per decode step with N-gram speculation (1 = off)
	SpecDraftLen   int     // verify width per decode step when speculating
	CapacityBlocks int     // KV cache blocks per worker (16 tokens each) available for prefix caching
	MaxBatch       int
	TokenBudget    int
	HW             Hardware
	Model          ModelProfile
	Alpha, Beta    float64
	Seed           int64
}

// DefaultConfig is the headline benchmark: 5,000 concurrent multi-turn sessions on 12 H100-class workers.
func DefaultConfig() Config {
	return Config{
		Workers: 12, Sessions: 5000, Turns: 3, Models: 24,
		ModelTokMin: 4000, ModelTokMax: 12000, ZipfS: 1.15,
		RampSec: 40, ThinkSec: 4, QuestionTok: 24, AnswerTokMin: 50, AnswerTokMax: 110,
		SpecTokPerStep: 2.4, SpecDraftLen: 6,
		CapacityBlocks: 3600, MaxBatch: 256, TokenBudget: 8192,
		HW: H100, Model: Qwen15B, Alpha: 1.0, Beta: 0.004, Seed: 42,
	}
}

// Result aggregates one policy run.
type Result struct {
	Policy        string
	Requests      int
	TTFTms        []float64
	E2Ems         []float64
	OutputTokens  int64
	PromptTokens  int64
	CachedTokens  int64
	MakespanS     float64
	FLOPsFull     float64 // prefill FLOPs if nothing were cached
	FLOPsSpent    float64
	ComputeRate   float64
	BusyImbalance float64 // max/mean worker busy time
	MeanQueue     float64
	Utilization   float64 // mean worker busy fraction over the makespan
}

func (r *Result) HitRatio() float64 {
	if r.PromptTokens == 0 {
		return 0
	}
	return float64(r.CachedTokens) / float64(r.PromptTokens)
}
func (r *Result) FLOPsSaved() float64 { return r.FLOPsFull - r.FLOPsSpent }
func (r *Result) SavedFrac() float64 {
	if r.FLOPsFull == 0 {
		return 0
	}
	return r.FLOPsSaved() / r.FLOPsFull
}
func (r *Result) GPUSecondsSaved() float64 { return r.FLOPsSaved() / r.ComputeRate }
func (r *Result) Throughput() float64      { return float64(r.OutputTokens) / r.MakespanS }
func (r *Result) Percentile(xs []float64, p float64) float64 {
	return Percentile(xs, p)
}

// Percentile of an unsorted slice (nearest-rank on a sorted copy).
func Percentile(xs []float64, p float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	i := int(math.Ceil(p/100*float64(len(s)))) - 1
	if i < 0 {
		i = 0
	}
	return s[i]
}

// ---- event machinery -------------------------------------------------------------------------

type evKind uint8

const (
	evArrive evKind = iota
	evStep
)

type event struct {
	t    float64
	seq  uint64
	kind evKind
	req  *simReq // evArrive
	w    int     // evStep
}
type evHeap []*event

func (h evHeap) Len() int { return len(h) }
func (h evHeap) Less(i, j int) bool {
	if h[i].t != h[j].t {
		return h[i].t < h[j].t
	}
	return h[i].seq < h[j].seq
}
func (h evHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *evHeap) Push(x any)   { *h = append(*h, x.(*event)) }
func (h *evHeap) Pop() any {
	o := *h
	n := len(o)
	x := o[n-1]
	*h = o[:n-1]
	return x
}

type simReq struct {
	session, turn int
	model         int
	arrive        float64
	tokens        []uint32
	hashes        []uint64
	outTok        int
	decision      router.Decision
	cached        int
	prefillLeft   int
	ctx           int
	produced      int
	firstAt       float64
	started       bool
}

type simWorker struct {
	id          int
	truth       *router.RadixCache // ground-truth cache of this worker (single-worker radix)
	waiting     []*simReq
	running     []*simReq
	stepPending bool
	busy        float64
}

// Run simulates the configured workload against the policy built by mk.
// mk receives the shared router-side shadow cache and the worker handles; for non-cache policies it
// can ignore the cache.
func Run(cfg Config, name string, mk func(cache *router.RadixCache, ws []*router.Worker) router.Policy) *Result {
	rng := rand.New(rand.NewSource(cfg.Seed))
	lat := NewLatencyModel(cfg.HW, cfg.Model)
	bs := router.DefaultBlockSize

	ws := make([]*router.Worker, cfg.Workers)
	sims := make([]*simWorker, cfg.Workers)
	for i := range ws {
		ws[i] = router.NewWorker(i, "w", "")
		sims[i] = &simWorker{id: i, truth: router.NewRadixCache(1, bs, cfg.CapacityBlocks)}
	}
	shadow := router.NewRadixCache(cfg.Workers, bs, cfg.CapacityBlocks)
	policy := mk(shadow, ws)

	// Per-model prompt size (stable per model) and popularity.
	modelTok := make([]int, cfg.Models)
	for m := range modelTok {
		modelTok[m] = cfg.ModelTokMin + rng.Intn(cfg.ModelTokMax-cfg.ModelTokMin+1)
	}
	zipf := rand.NewZipf(rng, cfg.ZipfS, 1, uint64(cfg.Models-1))
	sessModel := make([]int, cfg.Sessions)
	sessStart := make([]float64, cfg.Sessions)
	for s := range sessModel {
		sessModel[s] = int(zipf.Uint64())
		sessStart[s] = rng.Float64() * cfg.RampSec
	}
	modelPrefix := make([][]uint32, cfg.Models)
	for m := range modelPrefix {
		modelPrefix[m] = SynthTokens(1, uint64(m), modelTok[m])
	}

	h := &evHeap{}
	var seq uint64
	push := func(e *event) { seq++; e.seq = seq; heap.Push(h, e) }

	newReq := func(s, turn int, t float64) *simReq {
		m := sessModel[s]
		toks := make([]uint32, 0, len(modelPrefix[m])+(turn+1)*(cfg.QuestionTok+cfg.AnswerTokMax))
		toks = append(toks, modelPrefix[m]...)
		for j := 0; j < turn; j++ { // history: previous user turn + assistant SQL
			toks = append(toks, SynthTokens(2, uint64(s)<<8|uint64(j), cfg.QuestionTok)...)
			toks = append(toks, SynthTokens(3, uint64(s)<<8|uint64(j), cfg.AnswerTokMin+(s+j)%(cfg.AnswerTokMax-cfg.AnswerTokMin+1))...)
		}
		toks = append(toks, SynthTokens(2, uint64(s)<<8|uint64(turn), cfg.QuestionTok)...)
		return &simReq{session: s, turn: turn, model: m, arrive: t, tokens: toks,
			hashes: router.HashBlocks(toks, bs),
			outTok: cfg.AnswerTokMin + (s+turn)%(cfg.AnswerTokMax-cfg.AnswerTokMin+1)}
	}
	for s := 0; s < cfg.Sessions; s++ {
		push(&event{t: sessStart[s], kind: evArrive, req: &simReq{session: s, turn: 0}})
	}

	res := &Result{Policy: name, ComputeRate: lat.ComputeRate()}
	var lastDone float64
	inflight := 0
	var queueArea, lastT float64

	kick := func(w *simWorker, now float64) {
		if !w.stepPending && (len(w.running) > 0 || len(w.waiting) > 0) {
			w.stepPending = true
			push(&event{t: now, kind: evStep, w: w.id})
		}
	}

	finish := func(w *simWorker, r *simReq, now float64) {
		policy.Release(r.decision)
		inflight--
		res.TTFTms = append(res.TTFTms, (r.firstAt-r.arrive)*1000)
		res.E2Ems = append(res.E2Ems, (now-r.arrive)*1000)
		res.OutputTokens += int64(r.outTok)
		res.Requests++
		if now > lastDone {
			lastDone = now
		}
		if r.turn+1 < cfg.Turns {
			think := rng.ExpFloat64() * cfg.ThinkSec
			push(&event{t: now + think, kind: evArrive, req: &simReq{session: r.session, turn: r.turn + 1}})
		}
	}

	for h.Len() > 0 {
		e := heap.Pop(h).(*event)
		now := e.t
		queueArea += float64(inflight) * (now - lastT)
		lastT = now
		switch e.kind {
		case evArrive:
			r := newReq(e.req.session, e.req.turn, now)
			d, err := policy.Route(r.tokens, len(r.tokens)+r.outTok, 0)
			if err != nil {
				panic(err)
			}
			r.decision = d
			r.tokens = nil // hashes are all the simulator needs from here on
			inflight++
			w := sims[d.Worker.ID]
			w.waiting = append(w.waiting, r)
			kick(w, now)

		case evStep:
			w := sims[e.w]
			w.stepPending = false
			// admit
			for len(w.waiting) > 0 && len(w.running) < cfg.MaxBatch {
				r := w.waiting[0]
				w.waiting = w.waiting[1:]
				c := w.truth.Match(r.hashes, 1)[0] * bs
				plen := r.decision.PromptTokens
				if c >= plen {
					c = plen - 1 // the last token is always recomputed to produce logits
				}
				r.cached, r.ctx, r.prefillLeft = c, c, plen-c
				res.PromptTokens += int64(plen)
				res.CachedTokens += int64(c)
				res.FLOPsFull += lat.PrefillFLOPs(plen, 0)
				res.FLOPsSpent += lat.PrefillFLOPs(plen-c, c)
				w.running = append(w.running, r)
			}
			// compose the step: decode slots first, then chunked prefill within the token budget
			var decoders []*simReq
			decTok := 0
			spec := cfg.SpecTokPerStep > 1
			for _, r := range w.running {
				if r.prefillLeft == 0 && r.started {
					decoders = append(decoders, r)
					if spec {
						decTok += 1 + cfg.SpecDraftLen
					} else {
						decTok++
					}
				}
			}
			budget := cfg.TokenBudget - decTok
			if budget < 0 {
				budget = 0
			}
			var chunks []Chunk
			var chunkReqs []*simReq
			for _, r := range w.running {
				if r.prefillLeft > 0 && budget > 0 {
					n := r.prefillLeft
					if n > budget {
						n = budget
					}
					chunks = append(chunks, Chunk{n, r.ctx})
					chunkReqs = append(chunkReqs, r)
					budget -= n
				}
			}
			ctxs := make([]int, len(decoders))
			for i, r := range decoders {
				ctxs[i] = r.ctx
			}
			dt := lat.StepTime(chunks, ctxs, decTok)
			end := now + dt
			w.busy += dt

			for i, r := range chunkReqs {
				n := chunks[i].N
				r.prefillLeft -= n
				r.ctx += n
				if r.prefillLeft == 0 {
					w.truth.Insert(0, r.hashes) // prompt pages become shareable
					r.started, r.firstAt, r.produced = true, end, 1
				}
			}
			for _, r := range decoders {
				em := 1
				if spec {
					em = int(cfg.SpecTokPerStep)
					if rng.Float64() < cfg.SpecTokPerStep-float64(em) {
						em++
					}
				}
				r.produced += em
				r.ctx += em
			}
			// requests whose last tokens were produced this step complete at `end`
			kept := w.running[:0]
			for _, r := range w.running {
				if r.started && r.produced >= r.outTok {
					finish(w, r, end)
					continue
				}
				kept = append(kept, r)
			}
			w.running = kept
			if len(w.running) > 0 || len(w.waiting) > 0 {
				w.stepPending = true
				push(&event{t: end, kind: evStep, w: w.id})
			}
		}
	}

	res.MakespanS = lastDone
	var sum, mx float64
	for _, w := range sims {
		sum += w.busy
		if w.busy > mx {
			mx = w.busy
		}
	}
	if sum > 0 {
		res.BusyImbalance = mx / (sum / float64(len(sims)))
		res.Utilization = sum / float64(len(sims)) / lastDone
	}
	if lastT > 0 {
		res.MeanQueue = queueArea / lastT
	}
	return res
}

// Policies used by the comparison.
func RoundRobinFactory(_ *router.RadixCache, ws []*router.Worker) router.Policy {
	return router.NewRoundRobin(ws)
}

// LeastLoadedFactory is KV-oblivious join-shortest-queue (alpha = 0).
func LeastLoadedFactory(cfg Config) func(*router.RadixCache, []*router.Worker) router.Policy {
	return func(c *router.RadixCache, ws []*router.Worker) router.Policy {
		return router.NewKVAwareBalancer(c, ws, 0, 1)
	}
}

// RadixFactory is the prefix-aware policy under test.
func RadixFactory(cfg Config) func(*router.RadixCache, []*router.Worker) router.Policy {
	return func(c *router.RadixCache, ws []*router.Worker) router.Policy {
		return router.NewKVAwareBalancer(c, ws, cfg.Alpha, cfg.Beta)
	}
}
