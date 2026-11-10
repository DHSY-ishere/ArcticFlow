package router

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strings"
	"time"
)

// GenerateRequest is the client-facing request (HTTP JSON body / gRPC message).
type GenerateRequest struct {
	SessionID     string    `json:"session_id,omitempty"`
	SemanticModel string    `json:"semantic_model"` // Cortex-style YAML text
	Messages      []Message `json:"messages"`
	MaxTokens     int       `json:"max_tokens,omitempty"`
	Execute       bool      `json:"execute,omitempty"` // run SQL on the worker's DuckDB
	Stream        *bool     `json:"stream,omitempty"`  // default true
}

type upstreamRequest struct {
	GenerateRequest
	RequestID string `json:"request_id"`
	Prompt    string `json:"prompt"`
	// The router already tokenized and block-hashed the prompt; shipping the hashes (not the token
	// array) saves the worker re-tokenizing/re-hashing ~10^4 tokens per request.
	PromptLen   int      `json:"prompt_len"`
	BlockHashes []uint64 `json:"block_hashes"`
}

type eventHead struct {
	Type         string `json:"type"`
	PromptTokens int    `json:"prompt_tokens"`
	CachedTokens int    `json:"cached_tokens"`
	Usage        *struct {
		SpecProposed int `json:"spec_proposed"`
		SpecAccepted int `json:"spec_accepted"`
	} `json:"usage"`
}

// Server is the reverse proxy core shared by the HTTP/SSE and gRPC front-ends.
type Server struct {
	Policy     Policy
	Workers    []*Worker
	Cache      *RadixCache // nil for cache-oblivious policies
	Metrics    *Metrics
	Client     *http.Client
	MaxRetries int
	Log        *slog.Logger
}

// NewUpstreamClient returns an HTTP client tuned for thousands of concurrent
// long-lived SSE streams to a handful of workers.
func NewUpstreamClient() *http.Client {
	return &http.Client{Transport: &http.Transport{
		DialContext:           (&net.Dialer{Timeout: 3 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
		MaxIdleConns:          8192,
		MaxIdleConnsPerHost:   4096,
		IdleConnTimeout:       60 * time.Second, // must stay below the worker keep-alive (75s)
		ResponseHeaderTimeout: 30 * time.Second,
		ForceAttemptHTTP2:     false,
	}}
}

func newID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// Generate routes the request, proxies the worker's SSE stream, and calls emit
// once per event payload (raw JSON). onRoute fires before any emit.
func (s *Server) Generate(ctx context.Context, req *GenerateRequest, emit func([]byte) error, onRoute func(Decision)) error {
	if strings.TrimSpace(req.SemanticModel) == "" {
		return errors.New("semantic_model is required")
	}
	if len(req.Messages) == 0 {
		return errors.New("messages must not be empty")
	}
	if req.MaxTokens <= 0 {
		req.MaxTokens = 256
	}
	t0 := time.Now()
	prompt := BuildPrompt(req.SemanticModel, req.Messages)
	tokens := Tokenize(prompt)
	up := upstreamRequest{GenerateRequest: *req, RequestID: newID(), Prompt: prompt, PromptLen: len(tokens),
		BlockHashes: HashBlocks(tokens, DefaultBlockSize)}
	body, err := json.Marshal(up)
	if err != nil {
		return err
	}

	var exclude uint64
	var lastErr error
	for attempt := 0; attempt <= s.MaxRetries; attempt++ {
		d, err := s.Policy.Route(tokens, len(tokens)+req.MaxTokens, exclude)
		if err != nil {
			return err
		}
		s.Metrics.ObserveRoute(d)
		if onRoute != nil {
			onRoute(d)
		}
		forwarded, err := s.stream(ctx, d, body, t0, emit)
		s.Policy.Release(d)
		if err == nil {
			s.Metrics.ObserveOutcome(d.Worker.Name, "ok")
			return nil
		}
		s.Metrics.ObserveOutcome(d.Worker.Name, "error")
		if forwarded || ctx.Err() != nil {
			return err
		}
		// Nothing reached the client yet: fail over to another worker.
		lastErr = err
		exclude |= 1 << uint(d.Worker.ID)
		// Quarantine until the next successful health probe. The cache index is NOT dropped here:
		// a transient error says nothing about the worker's KV contents (the health checker evicts
		// it after FailThreshold consecutive failed probes).
		d.Worker.SetHealthy(false)
		s.Metrics.ObserveRetry()
		s.Log.Warn("upstream failed, retrying", "worker", d.Worker.Name, "err", err)
	}
	return fmt.Errorf("all attempts failed: %w", lastErr)
}

// stream proxies one upstream attempt. `forwarded` reports whether any bytes
// were already emitted to the client (after which failover is unsafe).
func (s *Server) stream(ctx context.Context, d Decision, body []byte, t0 time.Time, emit func([]byte) error) (forwarded bool, err error) {
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, d.Worker.URL+"/v1/generate", bytes.NewReader(body))
	if err != nil {
		return false, err
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Accept", "text/event-stream")
	resp, err := s.Client.Do(hreq)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return false, fmt.Errorf("worker %s: status %d: %s", d.Worker.Name, resp.StatusCode, bytes.TrimSpace(b))
	}

	rd := bufio.NewReaderSize(resp.Body, 64<<10)
	gotFirstToken, gotDone := false, false
	for {
		line, rerr := rd.ReadBytes('\n')
		if len(line) > 0 {
			line = bytes.TrimRight(line, "\r\n")
			if payload, ok := bytes.CutPrefix(line, []byte("data:")); ok {
				payload = bytes.TrimSpace(payload)
				var h eventHead
				_ = json.Unmarshal(payload, &h)
				switch h.Type {
				case "start":
					s.Metrics.ObserveStart(h.PromptTokens, h.CachedTokens)
				case "token":
					if !gotFirstToken {
						gotFirstToken = true
						s.Metrics.ObserveTTFT(d.Worker.Name, float64(time.Since(t0).Microseconds())/1000)
					}
				case "done":
					gotDone = true
					if h.Usage != nil {
						d.Worker.specProposed.Add(int64(h.Usage.SpecProposed))
						d.Worker.specAccepted.Add(int64(h.Usage.SpecAccepted))
					}
				}
				if eerr := emit(payload); eerr != nil {
					return true, eerr
				}
				forwarded = true
			}
		}
		if rerr != nil {
			if rerr == io.EOF && gotDone {
				return forwarded, nil
			}
			if rerr == io.EOF {
				rerr = io.ErrUnexpectedEOF
			}
			return forwarded, rerr
		}
	}
}

// ---------------------------------------------------------------------------
// HTTP / SSE front-end
// ---------------------------------------------------------------------------

// Handler returns the HTTP mux (generate, results proxy, admin, metrics).
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/generate", s.handleGenerate)
	mux.HandleFunc("GET /v1/results/{worker}/{id}", s.handleResults)
	mux.HandleFunc("GET /v1/workers", s.handleWorkers)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write([]byte("ok")) })
	if s.Metrics != nil {
		mux.Handle("GET /metrics", s.Metrics.Handler())
	}
	return mux
}

func (s *Server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 8<<20)
	var req GenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
		return
	}
	stream := req.Stream == nil || *req.Stream
	rc := http.NewResponseController(w)
	onRoute := func(d Decision) {
		w.Header().Set("X-Arctic-Worker", d.Worker.Name)
		w.Header().Set("X-Arctic-Predicted-Overlap", fmt.Sprintf("%.4f", d.Ratio))
	}

	if !stream {
		var last []byte
		err := s.Generate(r.Context(), &req, func(p []byte) error {
			var h eventHead
			if json.Unmarshal(p, &h) == nil && (h.Type == "done" || h.Type == "error") {
				last = append(last[:0], p...)
			}
			return nil
		}, onRoute)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(last)
		return
	}

	started := false
	err := s.Generate(r.Context(), &req, func(p []byte) error {
		if !started {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("X-Accel-Buffering", "no")
			w.WriteHeader(http.StatusOK)
			started = true
		}
		if _, err := w.Write([]byte("data: ")); err != nil {
			return err
		}
		if _, err := w.Write(p); err != nil {
			return err
		}
		if _, err := w.Write([]byte("\n\n")); err != nil {
			return err
		}
		return rc.Flush()
	}, onRoute)
	if err != nil {
		if !started {
			code := http.StatusBadGateway
			if errors.Is(err, ErrNoWorkers) {
				code = http.StatusServiceUnavailable
			}
			http.Error(w, err.Error(), code)
			return
		}
		b, _ := json.Marshal(map[string]string{"type": "error", "message": err.Error()})
		_, _ = fmt.Fprintf(w, "data: %s\n\n", b)
		_ = rc.Flush()
	}
}

// handleResults proxies Arrow IPC result streams from a worker without
// decoding them: bytes flow worker -> client through io.Copy.
func (s *Server) handleResults(w http.ResponseWriter, r *http.Request) {
	name, id := r.PathValue("worker"), r.PathValue("id")
	for _, wk := range s.Workers {
		if wk.Name != name {
			continue
		}
		req, _ := http.NewRequestWithContext(r.Context(), http.MethodGet, wk.URL+"/v1/results/"+id, nil)
		resp, err := s.Client.Do(req)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		defer resp.Body.Close()
		w.Header().Set("Content-Type", resp.Header.Get("Content-Type"))
		w.WriteHeader(resp.StatusCode)
		rc := http.NewResponseController(w)
		buf := make([]byte, 128<<10)
		for {
			n, rerr := resp.Body.Read(buf)
			if n > 0 {
				if _, werr := w.Write(buf[:n]); werr != nil {
					return
				}
				_ = rc.Flush()
			}
			if rerr != nil {
				return
			}
		}
	}
	http.Error(w, "unknown worker", http.StatusNotFound)
}

func (s *Server) handleWorkers(w http.ResponseWriter, r *http.Request) {
	type wv struct {
		Name          string `json:"name"`
		URL           string `json:"url"`
		Healthy       bool   `json:"healthy"`
		QueueDepth    int64  `json:"queue_depth"`
		InFlightToks  int64  `json:"in_flight_tokens"`
		ResidentBlock int    `json:"resident_blocks"`
	}
	var blocks []int
	out := struct {
		Policy  string `json:"policy"`
		Workers []wv   `json:"workers"`
		Nodes   int    `json:"radix_nodes"`
	}{Policy: s.Policy.Name()}
	if s.Cache != nil {
		st := s.Cache.Stats()
		blocks, out.Nodes = st.BlocksByWorker, st.Nodes
	}
	for _, wk := range s.Workers {
		v := wv{Name: wk.Name, URL: wk.URL, Healthy: wk.Healthy(), QueueDepth: wk.QueueDepth(), InFlightToks: wk.InFlightTokens()}
		if blocks != nil {
			v.ResidentBlock = blocks[wk.ID]
		}
		out.Workers = append(out.Workers, v)
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(out)
}
