package router

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// fakeWorker speaks the worker SSE protocol and models a per-prefix KV cache so cached_tokens is real.
func fakeWorker(t *testing.T, name string, hits *atomic.Int64) *httptest.Server {
	seen := map[uint64]bool{}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) { w.Write([]byte("ok")) })
	mux.HandleFunc("POST /v1/generate", func(w http.ResponseWriter, r *http.Request) {
		var req upstreamRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		cached := 0
		for _, h := range req.BlockHashes {
			if !seen[h] {
				seen[h] = true
				continue
			}
			cached += DefaultBlockSize
		}
		hits.Add(int64(cached))
		w.Header().Set("Content-Type", "text/event-stream")
		f := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"type\":\"start\",\"worker\":%q,\"prompt_tokens\":%d,\"cached_tokens\":%d}\n\n", name, req.PromptLen, cached)
		f.Flush()
		for i := 0; i < 3; i++ {
			fmt.Fprintf(w, "data: {\"type\":\"token\",\"text\":\"t%d \",\"n\":1}\n\n", i)
			f.Flush()
		}
		fmt.Fprint(w, "data: {\"type\":\"done\",\"sql\":\"SELECT 1\",\"usage\":{\"spec_proposed\":10,\"spec_accepted\":7}}\n\n")
		f.Flush()
	})
	return httptest.NewServer(mux)
}

func newTestServer(t *testing.T, urls []string, policy string) (*Server, []*Worker) {
	ws := make([]*Worker, len(urls))
	for i, u := range urls {
		ws[i] = NewWorker(i, fmt.Sprintf("w%d", i), u)
	}
	var pol Policy
	var cache *RadixCache
	if policy == "radix" {
		cache = NewRadixCache(len(ws), 16, 100000)
		pol = NewKVAwareBalancer(cache, ws, 1, 0.01)
	} else {
		pol = NewRoundRobin(ws)
	}
	return &Server{Policy: pol, Workers: ws, Cache: cache, Metrics: NewMetrics(ws, cache), Client: NewUpstreamClient(),
		MaxRetries: 2, Log: slog.New(slog.NewTextHandler(io.Discard, nil))}, ws
}

func bigModel() string {
	return strings.Repeat("tables:\n  - name: orders\n    dimensions: [region, segment, channel]\n", 120)
}

func post(t *testing.T, url, body string) (*http.Response, []map[string]any) {
	resp, err := http.Post(url+"/v1/generate", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	var evs []map[string]any
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		if p, ok := strings.CutPrefix(sc.Text(), "data: "); ok {
			var m map[string]any
			if err := json.Unmarshal([]byte(p), &m); err != nil {
				t.Fatalf("bad event %q", p)
			}
			evs = append(evs, m)
		}
	}
	resp.Body.Close()
	return resp, evs
}

func TestProxyAffinityMetricsAndSSE(t *testing.T) {
	var h0, h1 atomic.Int64
	w0, w1 := fakeWorker(t, "w0", &h0), fakeWorker(t, "w1", &h1)
	defer w0.Close()
	defer w1.Close()
	srv, _ := newTestServer(t, []string{w0.URL, w1.URL}, "radix")
	front := httptest.NewServer(srv.Handler())
	defer front.Close()

	body, _ := json.Marshal(GenerateRequest{SemanticModel: bigModel(), Messages: []Message{{Role: "user", Content: "revenue by region"}}})
	var first string
	for i := 0; i < 6; i++ {
		resp, evs := post(t, front.URL, string(body))
		if len(evs) != 5 || evs[0]["type"] != "start" || evs[4]["type"] != "done" {
			t.Fatalf("bad stream: %v", evs)
		}
		wk := resp.Header.Get("X-Arctic-Worker")
		if first == "" {
			first = wk
		} else if wk != first {
			t.Fatalf("request %d went to %s, expected affinity to %s", i, wk, first)
		}
		if i > 0 && evs[0]["cached_tokens"].(float64) < 1000 {
			t.Fatalf("worker-reported cache hit too small: %v", evs[0]["cached_tokens"])
		}
	}
	m, _ := http.Get(front.URL + "/metrics")
	b, _ := io.ReadAll(m.Body)
	text := string(b)
	for _, want := range []string{"arctic_kv_cache_hit_ratio", "arctic_ttft_latency_ms_bucket", "arctic_active_in_flight_tokens{worker=",
		"arctic_speculative_acceptance_rate{worker=\"" + first + "\"} 0.7"} {
		if !strings.Contains(text, want) {
			t.Fatalf("metrics missing %q", want)
		}
	}
}

func TestProxyFailover(t *testing.T) {
	var h0, h1 atomic.Int64
	dead := fakeWorker(t, "w0", &h0)
	live := fakeWorker(t, "w1", &h1)
	defer live.Close()
	deadURL := dead.URL
	dead.Close() // connection refused
	srv, ws := newTestServer(t, []string{deadURL, live.URL}, "roundrobin")
	front := httptest.NewServer(srv.Handler())
	defer front.Close()
	body, _ := json.Marshal(GenerateRequest{SemanticModel: bigModel(), Messages: []Message{{Role: "user", Content: "q"}}})
	for i := 0; i < 4; i++ {
		resp, evs := post(t, front.URL, string(body))
		if resp.StatusCode != 200 || len(evs) != 5 || resp.Header.Get("X-Arctic-Worker") != "w1" {
			t.Fatalf("failover failed: status %d worker %q evs %d", resp.StatusCode, resp.Header.Get("X-Arctic-Worker"), len(evs))
		}
	}
	if ws[0].Healthy() {
		t.Fatal("dead worker should have been marked unhealthy")
	}
	for _, w := range ws {
		if w.QueueDepth() != 0 {
			t.Fatalf("leaked in-flight count on %s: %d", w.Name, w.QueueDepth())
		}
	}
}

func TestProxyBadRequestAndNonStream(t *testing.T) {
	var h atomic.Int64
	w := fakeWorker(t, "w0", &h)
	defer w.Close()
	srv, _ := newTestServer(t, []string{w.URL}, "roundrobin")
	front := httptest.NewServer(srv.Handler())
	defer front.Close()
	resp, _ := http.Post(front.URL+"/v1/generate", "application/json", strings.NewReader(`{"messages":[]}`))
	if resp.StatusCode != http.StatusBadGateway {
		t.Fatalf("want 502 for invalid request, got %d", resp.StatusCode)
	}
	resp, err := http.Post(front.URL+"/v1/generate", "application/json",
		strings.NewReader(`{"semantic_model":"x: 1","messages":[{"role":"user","content":"hi"}],"stream":false}`))
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&out)
	if out["type"] != "done" || out["sql"] != "SELECT 1" {
		t.Fatalf("non-stream response wrong: %v", out)
	}
}

func TestGRPCStream(t *testing.T) {
	var h atomic.Int64
	w := fakeWorker(t, "w0", &h)
	defer w.Close()
	srv, _ := newTestServer(t, []string{w.URL}, "radix")
	gs := grpc.NewServer()
	RegisterGRPC(gs, &GRPCService{S: srv})
	lis := newLocalListener(t)
	go gs.Serve(lis)
	defer gs.Stop()

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.CallContentSubtype("json")))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	st, err := conn.NewStream(ctx, &grpc.StreamDesc{ServerStreams: true}, "/arcticflow.v1.Router/Generate")
	if err != nil {
		t.Fatal(err)
	}
	if err := st.SendMsg(&GenerateRequest{SemanticModel: bigModel(), Messages: []Message{{Role: "user", Content: "q"}}}); err != nil {
		t.Fatal(err)
	}
	_ = st.CloseSend()
	var types []string
	for {
		var ev GenerateEvent
		if err := st.RecvMsg(&ev); err != nil {
			if err != io.EOF {
				t.Fatal(err)
			}
			break
		}
		var h eventHead
		_ = json.Unmarshal(ev.Data, &h)
		types = append(types, h.Type)
	}
	if strings.Join(types, ",") != "start,token,token,token,done" {
		t.Fatalf("grpc stream: %v", types)
	}
	var hr HealthResponse
	if err := conn.Invoke(ctx, "/arcticflow.v1.Router/Health", &HealthRequest{}, &hr); err != nil || hr.HealthyWorkers != 1 {
		t.Fatalf("health: %v %+v", err, hr)
	}
}

func newLocalListener(t *testing.T) net.Listener {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	return l
}
