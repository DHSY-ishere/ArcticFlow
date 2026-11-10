package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/arcticflow/arcticflow/benchmark"
)

type generateReq struct {
	SessionID     string       `json:"session_id"`
	SemanticModel string       `json:"semantic_model"`
	Messages      []messageDTO `json:"messages"`
	MaxTokens     int          `json:"max_tokens"`
	Execute       bool         `json:"execute"`
}
type messageDTO struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}
type event struct {
	Type         string `json:"type"`
	PromptTokens int    `json:"prompt_tokens"`
	CachedTokens int    `json:"cached_tokens"`
	SQL          string `json:"sql"`
	Message      string `json:"message"`
	Usage        struct {
		CompletionTokens int `json:"completion_tokens"`
		SpecProposed     int `json:"spec_proposed"`
		SpecAccepted     int `json:"spec_accepted"`
	} `json:"usage"`
}

// runHTTP drives a live router. Every session keeps its own conversation (assistant turns carry the
// SQL the worker returned), so later turns share a growing prefix exactly like a real analyst chat.
func runHTTP(cfg benchmark.Config, target string, fillerCols int, execute bool, maxConc int) (string, error) {
	client := &http.Client{Transport: &http.Transport{
		DialContext:         (&net.Dialer{Timeout: 5 * time.Second}).DialContext,
		MaxIdleConns:        8192,
		MaxIdleConnsPerHost: 8192,
	}}
	policy := fetchPolicy(client, target)
	if policy == "" {
		return "", fmt.Errorf("router at %s is not reachable (GET /v1/workers failed)", target)
	}

	models := make([]string, cfg.Models)
	for i := range models {
		models[i] = benchmark.GenSemanticYAML(i, fillerCols)
	}
	rng := rand.New(rand.NewSource(cfg.Seed))
	zipf := rand.NewZipf(rng, cfg.ZipfS, 1, uint64(cfg.Models-1))

	var (
		mu           sync.Mutex
		res          = &benchmark.Result{Policy: "live router (" + policy + ")", ComputeRate: benchmark.NewLatencyModel(benchmark.H100, benchmark.Qwen15B).ComputeRate()}
		lat          = benchmark.NewLatencyModel(benchmark.H100, benchmark.Qwen15B)
		errs         int
		errKinds     = map[string]int{}
		specP, specA int64
		sem          chan struct{}
	)
	if maxConc > 0 {
		sem = make(chan struct{}, maxConc)
	}
	start := time.Now()
	var wg sync.WaitGroup
	for s := 0; s < cfg.Sessions; s++ {
		model := models[zipf.Uint64()]
		delay := time.Duration(rng.Float64() * cfg.RampSec * float64(time.Second))
		seed := rng.Int63()
		wg.Add(1)
		go func(s int, model string, delay time.Duration, seed int64) {
			defer wg.Done()
			time.Sleep(delay)
			r := rand.New(rand.NewSource(seed))
			var msgs []messageDTO
			for turn := 0; turn < cfg.Turns; turn++ {
				q := benchmark.Questions[r.Intn(len(benchmark.Questions))]
				if turn > 0 && r.Intn(3) == 0 {
					q = benchmark.FollowUps[r.Intn(len(benchmark.FollowUps))]
				}
				msgs = append(msgs, messageDTO{"user", q})
				if sem != nil {
					sem <- struct{}{}
				}
				ttft, e2e, ev, err := oneRequest(client, target, generateReq{
					SessionID: fmt.Sprint("s", s), SemanticModel: model, Messages: msgs, MaxTokens: 256, Execute: execute})
				if sem != nil {
					<-sem
				}
				mu.Lock()
				if err != nil {
					errs++
					errKinds[err.Error()]++
					mu.Unlock()
					return
				}
				res.Requests++
				res.TTFTms = append(res.TTFTms, ttft)
				res.E2Ems = append(res.E2Ems, e2e)
				res.PromptTokens += int64(ev.start.PromptTokens)
				res.CachedTokens += int64(ev.start.CachedTokens)
				res.OutputTokens += int64(ev.done.Usage.CompletionTokens)
				res.FLOPsFull += lat.PrefillFLOPs(ev.start.PromptTokens, 0)
				res.FLOPsSpent += lat.PrefillFLOPs(ev.start.PromptTokens-ev.start.CachedTokens, ev.start.CachedTokens)
				specP += int64(ev.done.Usage.SpecProposed)
				specA += int64(ev.done.Usage.SpecAccepted)
				mu.Unlock()
				msgs = append(msgs, messageDTO{"assistant", ev.done.SQL})
				time.Sleep(time.Duration(r.ExpFloat64() * cfg.ThinkSec * float64(time.Second)))
			}
		}(s, model, delay, seed)
	}
	wg.Wait()
	res.MakespanS = time.Since(start).Seconds()
	res.BusyImbalance, res.Utilization = 0, 0
	out := fmt.Sprintf("**Live run:** %d sessions x %d turns against `%s` (policy `%s`), %d semantic models, %d errors, spec acceptance %.1f%%, %d requests in %.1fs = %.0f req/s.\n\n",
		cfg.Sessions, cfg.Turns, target, policy, cfg.Models, errs, pct(specA, specP), res.Requests, res.MakespanS,
		float64(res.Requests)/res.MakespanS)
	for k, n := range errKinds {
		out += fmt.Sprintf("- %d x error: %s\n", n, k)
	}
	out += benchmark.Table([]*benchmark.Result{res})
	out += "\n(GPU util / load-imbalance columns are not observable from the client; see Grafana.)\n"
	return out, nil
}

func pct(a, b int64) float64 {
	if b == 0 {
		return 0
	}
	return 100 * float64(a) / float64(b)
}

type pair struct{ start, done event }

func oneRequest(c *http.Client, target string, body generateReq) (ttftMs, e2eMs float64, out pair, err error) {
	b, _ := json.Marshal(body)
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, target+"/v1/generate", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	t0 := time.Now()
	resp, err := c.Do(req)
	if err != nil {
		return 0, 0, out, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, 0, out, fmt.Errorf("status %d", resp.StatusCode)
	}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<20), 8<<20)
	first := true
	for sc.Scan() {
		p, ok := bytes.CutPrefix(sc.Bytes(), []byte("data: "))
		if !ok {
			continue
		}
		var ev event
		if err := json.Unmarshal(p, &ev); err != nil {
			continue
		}
		switch ev.Type {
		case "start":
			out.start = ev
		case "token":
			if first {
				first = false
				ttftMs = float64(time.Since(t0).Microseconds()) / 1000
			}
		case "done":
			out.done = ev
			return ttftMs, float64(time.Since(t0).Microseconds()) / 1000, out, nil
		case "error":
			return 0, 0, out, fmt.Errorf("worker error: %s", ev.Message)
		}
	}
	return 0, 0, out, fmt.Errorf("stream ended without done: %v", sc.Err())
}

func fetchPolicy(c *http.Client, target string) string {
	resp, err := c.Get(target + "/v1/workers")
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	var v struct {
		Policy string `json:"policy"`
	}
	_ = json.NewDecoder(resp.Body).Decode(&v)
	return v.Policy
}
