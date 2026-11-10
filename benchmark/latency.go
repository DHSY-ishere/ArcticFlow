// Package benchmark contains a discrete-event GPU-fleet simulator that drives the REAL router
// code (router.RadixCache + router.Policy implementations) with virtual time, a workload
// generator for multi-turn analytical sessions, and report formatting.
package benchmark

// Hardware / model constants mirror engine/latency_model.py exactly (same roofline).

// Hardware describes one accelerator.
type Hardware struct {
	Name       string
	PeakTFLOPs float64
	MemBWGBs   float64
	MFU, MBU   float64
	OverheadS  float64
}

// Hardware profiles (dense bf16 peak, HBM bandwidth).
var (
	H100 = Hardware{"h100", 989, 3350, 0.45, 0.70, 0.0006}
	A100 = Hardware{"a100", 312, 2039, 0.45, 0.70, 0.0006}
	A10G = Hardware{"a10g", 125, 600, 0.45, 0.70, 0.0008}
)

// HardwareByName resolves a profile (default A100).
func HardwareByName(n string) Hardware {
	switch n {
	case "h100":
		return H100
	case "a10g":
		return A10G
	}
	return A100
}

// ModelProfile is Qwen2.5-Coder-1.5B by default.
type ModelProfile struct {
	Params                                  float64
	Layers, Hidden, KVHeads, HeadDim, DType int
}

// Qwen15B is the reference model.
var Qwen15B = ModelProfile{1.54e9, 28, 1536, 2, 128, 2}

func (m ModelProfile) KVBytesPerToken() float64 {
	return float64(2 * m.Layers * m.KVHeads * m.HeadDim * m.DType)
}

// LatencyModel computes roofline step times.
type LatencyModel struct {
	HW        Hardware
	M         ModelProfile
	flopsRate float64
	bytesRate float64
}

// NewLatencyModel builds the model.
func NewLatencyModel(hw Hardware, m ModelProfile) *LatencyModel {
	return &LatencyModel{HW: hw, M: m, flopsRate: hw.PeakTFLOPs * 1e12 * hw.MFU, bytesRate: hw.MemBWGBs * 1e9 * hw.MBU}
}

// PrefillFLOPs: FLOPs to prefill nNew tokens on top of nCtx cached tokens.
func (l *LatencyModel) PrefillFLOPs(nNew, nCtx int) float64 {
	n, c := float64(nNew), float64(nCtx)
	return 2*l.M.Params*n + 4*float64(l.M.Layers*l.M.Hidden)*n*(c+n/2)
}

// Chunk is a prefill chunk: nNew tokens after ctx tokens of context.
type Chunk struct{ N, Ctx int }

// StepTime returns the seconds for one engine step.
func (l *LatencyModel) StepTime(prefill []Chunk, decodeCtx []int, decodeTokens int) float64 {
	nTok := decodeTokens
	for _, c := range prefill {
		nTok += c.N
	}
	if nTok == 0 {
		return 0
	}
	flops := 2 * l.M.Params * float64(nTok)
	kvRead := 0.0
	for _, c := range prefill {
		flops += 4 * float64(l.M.Layers*l.M.Hidden) * float64(c.N) * (float64(c.Ctx) + float64(c.N)/2)
		kvRead += float64(c.Ctx)
	}
	if len(decodeCtx) > 0 {
		per := float64(decodeTokens) / float64(len(decodeCtx))
		for _, c := range decodeCtx {
			flops += 4 * float64(l.M.Layers*l.M.Hidden) * per * (float64(c) + per/2)
			kvRead += float64(c)
		}
	}
	bytes := l.M.Params*float64(l.M.DType) + (kvRead+float64(nTok))*l.M.KVBytesPerToken()
	t := flops / l.flopsRate
	if b := bytes / l.bytesRate; b > t {
		t = b
	}
	return t + l.HW.OverheadS
}

// ComputeRate is effective FLOP/s (peak x MFU), used to convert FLOPs saved to GPU-seconds.
func (l *LatencyModel) ComputeRate() float64 { return l.flopsRate }
