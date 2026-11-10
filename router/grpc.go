package router

import (
	"context"
	"encoding/json"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/encoding"
	"google.golang.org/grpc/status"
)

// JSONCodec lets us serve a real gRPC (HTTP/2, framed, server-streaming) API
// without a protoc toolchain: messages are JSON (content-type
// application/grpc+json). Clients select it with grpc.CallContentSubtype("json").
type JSONCodec struct{}

func (JSONCodec) Marshal(v any) ([]byte, error)   { return json.Marshal(v) }
func (JSONCodec) Unmarshal(b []byte, v any) error { return json.Unmarshal(b, v) }
func (JSONCodec) Name() string                    { return "json" }

func init() { encoding.RegisterCodec(JSONCodec{}) }

// GenerateEvent is one streamed event; Data is the worker's raw JSON event
// (types: start, token, done, error).
type GenerateEvent struct {
	Data json.RawMessage `json:"data"`
}

// GRPCService implements arcticflow.v1.Router.
type GRPCService struct{ S *Server }

type routerServer interface {
	generate(*GenerateRequest, grpc.ServerStream) error
}

func (g *GRPCService) generate(req *GenerateRequest, st grpc.ServerStream) error {
	err := g.S.Generate(st.Context(), req, func(p []byte) error {
		return st.SendMsg(&GenerateEvent{Data: append(json.RawMessage(nil), p...)})
	}, func(d Decision) {
		_ = st.SetHeader(map[string][]string{"x-arctic-worker": {d.Worker.Name}})
	})
	if err != nil {
		if err == ErrNoWorkers {
			return status.Error(codes.Unavailable, err.Error())
		}
		return status.Error(codes.Internal, err.Error())
	}
	return nil
}

// HealthRequest / HealthResponse for the unary Health RPC.
type HealthRequest struct{}
type HealthResponse struct {
	Status         string `json:"status"`
	HealthyWorkers int    `json:"healthy_workers"`
}

func (g *GRPCService) health(ctx context.Context, _ *HealthRequest) (*HealthResponse, error) {
	n := 0
	for _, w := range g.S.Workers {
		if w.Healthy() {
			n++
		}
	}
	return &HealthResponse{Status: "SERVING", HealthyWorkers: n}, nil
}

// RegisterGRPC attaches arcticflow.v1.Router{Generate (server-stream), Health}.
func RegisterGRPC(gs *grpc.Server, svc *GRPCService) {
	gs.RegisterService(&grpc.ServiceDesc{
		ServiceName: "arcticflow.v1.Router",
		HandlerType: (*routerServer)(nil),
		Methods: []grpc.MethodDesc{{
			MethodName: "Health",
			Handler: func(srv any, ctx context.Context, dec func(any) error, _ grpc.UnaryServerInterceptor) (any, error) {
				in := new(HealthRequest)
				if err := dec(in); err != nil {
					return nil, err
				}
				return srv.(*GRPCService).health(ctx, in)
			},
		}},
		Streams: []grpc.StreamDesc{{
			StreamName:    "Generate",
			ServerStreams: true,
			Handler: func(srv any, stream grpc.ServerStream) error {
				in := new(GenerateRequest)
				if err := stream.RecvMsg(in); err != nil {
					return err
				}
				return srv.(*GRPCService).generate(in, stream)
			},
		}},
		Metadata: "arcticflow/router.json",
	}, svc)
}
