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

	VRAMBytes  int64 `json:"vram_bytes"`
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
	return st, true
}
