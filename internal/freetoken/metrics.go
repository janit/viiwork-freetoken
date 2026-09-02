package freetoken

import (
	"encoding/json"
	"fmt"
)

// This file is the whole of what FreeToken's control plane looks like to a
// mesh node, and it is where the engine differs from vLLM most concretely.
//
// vLLM publishes Prometheus text at /metrics, and the vLLM node parses it by
// probing a list of metric-name aliases because the names move between engine
// versions. FreeToken publishes two JSON documents instead — /health for
// lifecycle and /v1/stats for runtime numbers — which are structured, so
// absence is expressible (a null pool, a missing key) rather than something to
// be inferred from a name that failed to match. Both are read on the health
// tick and are therefore up to one interval stale: they are a view for the
// dashboard, never an admission signal. Routing reads the in-flight counter.

// Lifecycle states as FreeToken's /health reports them.
const (
	healthOK      = "ok"
	healthLoading = "loading"
	healthError   = "error"

	// maintServing is the only maintenance state that accepts work. The engine
	// answers /health with status "ok" while a runtime cache rebuild is in
	// progress and 503s every generation request for its duration, so status
	// alone is not readiness — see Health.Ready.
	maintServing = "serving"
)

// Health is FreeToken's /health document.
//
// Three shapes share one struct because the engine returns all three from one
// endpoint with HTTP 200 in every case. That last part is the trap in porting
// the vLLM node's probe: there, a 200 from /health means ready. Here a model
// that is still loading its weights — minutes, on the models this engine
// exists to run — also answers 200, and a node that treated that as ready
// would advertise the backend to the mesh and have every routed request come
// back 503.
type Health struct {
	Status string `json:"status"`
	// Maintenance is "serving" once loading finished, and "rebuilding" or
	// "failed" while a live cache resize is in flight. Absent on the loading
	// and error shapes.
	Maintenance string `json:"maintenance"`
	Model       string `json:"model"`
	Version     string `json:"version"`
	// Message carries the fatal error on the error shape.
	Message string `json:"message"`
	// Phase and Progress describe what loading is doing. Reported to the
	// activity log so an operator watching a large model load sees movement
	// rather than a silent starting backend.
	Phase    string `json:"phase"`
	Progress struct {
		DoneBytes  int64 `json:"done_bytes"`
		TotalBytes int64 `json:"total_bytes"`
	} `json:"progress"`
}

// Ready reports whether the backend will actually serve a request.
func (h Health) Ready() bool {
	// An older engine build may omit maintenance from the ready shape; status
	// "ok" without it is the serving case.
	return h.Status == healthOK && (h.Maintenance == "" || h.Maintenance == maintServing)
}

// Describe is a short human-readable reason a backend is not ready, for the
// activity log. Empty when it is.
func (h Health) Describe() string {
	switch {
	case h.Ready():
		return ""
	case h.Status == healthError:
		if h.Message != "" {
			return "failed: " + h.Message
		}
		return "failed"
	case h.Status == healthLoading:
		if h.Progress.TotalBytes > 0 {
			return fmt.Sprintf("loading (%s, %d%%)", h.Phase,
				100*h.Progress.DoneBytes/h.Progress.TotalBytes)
		}
		if h.Phase != "" {
			return "loading (" + h.Phase + ")"
		}
		return "loading"
	case h.Maintenance != "":
		return "maintenance: " + h.Maintenance
	}
	return "not ready: " + h.Status
}

// ParseHealth decodes a /health body. A body that is not the document we
// expect yields ok=false, which the caller treats as a failed probe — the same
// outcome as no answer at all, and the right one: something is listening on
// that port that is not the engine we spawned.
func ParseHealth(body []byte) (Health, bool) {
	var h Health
	if err := json.Unmarshal(body, &h); err != nil {
		return Health{}, false
	}
	if h.Status == "" {
		return Health{}, false
	}
	return h, true
}

// pool is one of FreeToken's cache pools as /v1/stats reports it. The engine
// sends null for a pool a given model does not use, so a pointer field is the
// difference between "empty" and "not applicable".
type pool struct {
	UsedPages  int64 `json:"used_pages"`
	TotalPages int64 `json:"total_pages"`
	UsedSlots  int64 `json:"used_slots"`
	TotalSlots int64 `json:"total_slots"`
}

func (p *pool) occupancy() (float64, bool) {
	if p == nil {
		return 0, false
	}
	switch {
	case p.TotalPages > 0:
		return float64(p.UsedPages) / float64(p.TotalPages), true
	case p.TotalSlots > 0:
		return float64(p.UsedSlots) / float64(p.TotalSlots), true
	}
	return 0, false
}

type statsDoc struct {
	KV    *pool `json:"kv"`
	Mamba *pool `json:"mamba"`
	SWA   *pool `json:"swa"`

	VRAMBytes int64 `json:"vram_bytes"`
	// GPUs is the engine's own account of the card it bound. Added upstream
	// after 0.1.2 and therefore absent on the released engine, which is why a
	// missing entry has to read as "unknown" rather than as a mismatch.
	//
	// Index is deliberately not read. It is the engine's *visible* CUDA
	// ordinal, and under this node's one-card-per-process pinning exactly one
	// device is visible, so it is always 0 and carries no information. UUID is
	// the only field here that names a physical card.
	GPUs []struct {
		Name string `json:"name"`
		UUID string `json:"uuid"`
	} `json:"gpus"`
	Throughput struct {
		DecodeTPS  float64 `json:"decode_tps"`
		PrefillTPS float64 `json:"prefill_tps"`
	} `json:"throughput"`
	Requests struct {
		Active    int64 `json:"active"`
		Completed int64 `json:"completed"`
		P95ms     int64 `json:"p95_ms"`
	} `json:"requests"`
}

// Stats is the subset of FreeToken's /v1/stats this node acts on.
type Stats struct {
	NumRunning int64
	// CachePct is fractional occupancy of the fullest cache pool, 0..1. It is
	// the pressure signal: a backend near 1 is about to start evicting or
	// preempting, which shows up as latency long before it shows up as an
	// error.
	//
	// Collected but not yet published. meshapi.BackendInfo carries no
	// occupancy field, so this reaches balancer.BackendState and stops there,
	// exactly as on the vLLM node. It is kept accurate rather than dropped
	// because the field is the obvious next addition to the protocol, and
	// because a number that is wrong while nobody looks stays wrong when
	// somebody does.
	//
	// It is a maximum across pools rather than the KV pool alone because which
	// pools exist depends on the model. A dense-attention model reports kv; a
	// sliding-window model reports swa and a null kv; a linear-attention or
	// hybrid model reports mamba slots. Reading only kv would report a
	// permanent zero for two of the three.
	CachePct float64
	// DecodeTPS and VRAMBytes are reported for the dashboard, not acted on.
	DecodeTPS float64
	VRAMBytes int64
	// GPUUUID is the card the engine says it actually bound, and GPUName its
	// marketing name. Empty on an engine that predates the field.
	//
	// This is the only thing in either document that can be checked against
	// something outside the engine, and it is worth checking: pinning happens
	// through CUDA_VISIBLE_DEVICES, whose index CUDA reads in whatever order
	// CUDA_DEVICE_ORDER selects, and getting that wrong produces a backend
	// that serves perfectly while every dashboard attributes its load and its
	// wattage to a neighbouring card. See Manager.verifyPinning.
	GPUUUID string
	GPUName string
	// Found records which of the above actually appeared, so a consumer can
	// tell "no requests running" from "this build does not report that".
	Found map[string]bool
}

// ParseStats reads a /v1/stats body.
//
// Note what is absent: FreeToken publishes no queue-depth gauge. Its scheduler
// runs --max-running-requests concurrently and holds the rest, but never says
// how many it is holding, so there is no waiting count to read here. The node
// derives one on its own side of the socket instead — see Backend.refreshStats.
func ParseStats(body []byte) (Stats, bool) {
	var doc statsDoc
	if err := json.Unmarshal(body, &doc); err != nil {
		return Stats{}, false
	}
	st := Stats{
		NumRunning: doc.Requests.Active,
		DecodeTPS:  doc.Throughput.DecodeTPS,
		VRAMBytes:  doc.VRAMBytes,
		Found:      map[string]bool{"running": true},
	}
	for name, p := range map[string]*pool{"kv": doc.KV, "swa": doc.SWA, "mamba": doc.Mamba} {
		occ, ok := p.occupancy()
		if !ok {
			continue
		}
		st.Found[name] = true
		st.Found["cache"] = true
		if occ > st.CachePct {
			st.CachePct = occ
		}
	}
	if doc.Throughput.DecodeTPS > 0 {
		st.Found["decode_tps"] = true
	}
	if doc.VRAMBytes > 0 {
		st.Found["vram"] = true
	}
	// The first entry is the primary TP rank. There is only ever one: FreeToken
	// has --tensor-parallel-size but does not yet place more than one card, and
	// the list exists upstream so that it can later.
	if len(doc.GPUs) > 0 {
		st.GPUName = doc.GPUs[0].Name
		if u := doc.GPUs[0].UUID; u != "" {
			st.GPUUUID = u
			st.Found["gpu_uuid"] = true
		}
	}
	return st, true
}

// modelsDoc is FreeToken's /v1/models. Only the context length is read: the
// model id is already known (the node told the engine what to call it), and
// everything else is presentation.
type modelsDoc struct {
	Data []struct {
		MaxModelLen   int64 `json:"max_model_len"`
		ContextLength int64 `json:"context_length"`
	} `json:"data"`
}

// ParseContextLen reads the CHECKPOINT's context length from a /v1/models body.
//
// This is an upper bound, not what the backend will actually accept — see
// ParseKVCapacity, and prefer the smaller of the two. On the deployment this
// was written against the two differ by a factor of sixteen.
//
// Both spellings are read because the engine emits both, and a build that drops
// either should not silently produce a blank.
func ParseContextLen(body []byte) (int64, bool) {
	var doc modelsDoc
	if err := json.Unmarshal(body, &doc); err != nil {
		return 0, false
	}
	for _, m := range doc.Data {
		if m.ContextLength > 0 {
			return m.ContextLength, true
		}
		if m.MaxModelLen > 0 {
			return m.MaxModelLen, true
		}
	}
	return 0, false
}

// cacheStatusDoc is FreeToken's /v1/cache/status.
type cacheStatusDoc struct {
	Geometry struct {
		NumPages int64 `json:"num_pages"`
		PageSize int64 `json:"page_size"`
	} `json:"geometry"`
}

// ParseKVCapacity reads how many tokens of KV the engine actually allocated,
// which is the real ceiling on a request's prompt plus generation.
//
// This is the number that matters and it is not the one /v1/models reports. The
// engine sizes its KV pool from the VRAM left after weights and the MoE expert
// cache, so on a card serving a model far larger than itself the pool is a
// small fraction of what the checkpoint allows. Measured on an RTX 5090 serving
// DeepSeek-V4-Flash: /v1/models advertises 1,048,576 tokens, the KV pool holds
// 64,128, and a 70,175-token prompt comes back
//
//	400 context_length_exceeded
//	"prompt is too long: 70175 tokens > 64128 maximum (prompt + generation)"
//
// Publishing the advertised figure would therefore put a number on the
// dashboard that the backend rejects — worse than a blank, because a client
// would size a request by it.
//
// Deliberately NOT read from /v1/stats, which also reports kv.total_pages: that
// document carries the page size from the unresolved config rather than the
// resolved one (1 rather than 128 for a DSV4 checkpoint), so multiplying them
// there understates the pool by two orders of magnitude. /v1/cache/status
// reports the geometry the engine actually built.
func ParseKVCapacity(body []byte) (int64, bool) {
	var doc cacheStatusDoc
	if err := json.Unmarshal(body, &doc); err != nil {
		return 0, false
	}
	g := doc.Geometry
	if g.NumPages <= 0 || g.PageSize <= 0 {
		return 0, false
	}
	return g.NumPages * g.PageSize, true
}
