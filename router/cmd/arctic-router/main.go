// Command arctic-router is the KV-cache-aware LLM inference router.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"google.golang.org/grpc"

	"github.com/arcticflow/arcticflow/router"
)

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envF(key string, def float64) float64 {
	if v, err := strconv.ParseFloat(os.Getenv(key), 64); err == nil {
		return v
	}
	return def
}

func envI(key string, def int) int {
	if v, err := strconv.Atoi(os.Getenv(key)); err == nil {
		return v
	}
	return def
}

func main() {
	var (
		httpAddr = flag.String("http", env("ARCTIC_HTTP_ADDR", ":8080"), "HTTP/SSE listen address")
		grpcAddr = flag.String("grpc", env("ARCTIC_GRPC_ADDR", ":9090"), "gRPC listen address")
		workers  = flag.String("workers", env("ARCTIC_WORKERS", "http://localhost:8000"), "comma-separated worker base URLs (optionally name=url)")
		policy   = flag.String("policy", env("ARCTIC_POLICY", "radix"), "routing policy: radix | roundrobin")
		alpha    = flag.Float64("alpha", envF("ARCTIC_ALPHA", 1.0), "weight of KV overlap ratio")
		beta     = flag.Float64("beta", envF("ARCTIC_BETA", 0.01), "penalty per queued request")
		block    = flag.Int("block-size", envI("ARCTIC_BLOCK_SIZE", router.DefaultBlockSize), "KV page size in tokens")
		capacity = flag.Int("capacity-blocks", envI("ARCTIC_CAPACITY_BLOCKS", 4096), "per-worker KV blocks tracked by the radix index (must match worker ARCTIC_KV_BLOCKS)")
		health   = flag.Duration("health-interval", 2*time.Second, "worker health probe interval")
	)
	flag.Parse()
	log := slog.New(slog.NewTextHandler(os.Stderr, nil))

	var ws []*router.Worker
	for i, spec := range strings.Split(*workers, ",") {
		spec = strings.TrimSpace(spec)
		if spec == "" {
			continue
		}
		name, url := fmt.Sprintf("w%d", i), spec
		if k, v, ok := strings.Cut(spec, "="); ok {
			name, url = k, v
		}
		ws = append(ws, router.NewWorker(i, name, strings.TrimRight(url, "/")))
	}
	if len(ws) == 0 {
		log.Error("no workers configured")
		os.Exit(2)
	}

	var (
		pol   router.Policy
		cache *router.RadixCache
	)
	switch *policy {
	case "radix":
		cache = router.NewRadixCache(len(ws), *block, *capacity)
		pol = router.NewKVAwareBalancer(cache, ws, *alpha, *beta)
	case "roundrobin":
		pol = router.NewRoundRobin(ws)
	default:
		log.Error("unknown policy", "policy", *policy)
		os.Exit(2)
	}

	client := router.NewUpstreamClient()
	srv := &router.Server{Policy: pol, Workers: ws, Cache: cache, Metrics: router.NewMetrics(ws, cache),
		Client: client, MaxRetries: 2, Log: log}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go (&router.HealthChecker{Workers: ws, Cache: cache, Interval: *health, Client: client}).Run(ctx)

	hs := &http.Server{Addr: *httpAddr, Handler: srv.Handler(), ReadHeaderTimeout: 10 * time.Second}
	gs := grpc.NewServer()
	router.RegisterGRPC(gs, &router.GRPCService{S: srv})

	lis, err := net.Listen("tcp", *grpcAddr)
	if err != nil {
		log.Error("grpc listen", "err", err)
		os.Exit(1)
	}
	go func() {
		if err := gs.Serve(lis); err != nil {
			log.Error("grpc serve", "err", err)
		}
	}()
	go func() {
		log.Info("arctic-router up", "http", *httpAddr, "grpc", *grpcAddr, "policy", pol.Name(), "workers", len(ws),
			"alpha", *alpha, "beta", *beta)
		if err := hs.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Error("http serve", "err", err)
			stop()
		}
	}()

	<-ctx.Done()
	sctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	gs.GracefulStop()
	_ = hs.Shutdown(sctx)
}
