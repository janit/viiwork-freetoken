// Package proxy is the node's HTTP surface: the OpenAI-compatible inference
// API, the mesh endpoints peers poll, and the dashboards.
//
// The dashboard pages are served straight from viiwork's `web` package rather
// than being copied here. That is deliberate — the fleet view has to look and
// behave identically whichever node happens to serve it, and a fork of the
// page would drift the moment either side gained a column.
package proxy

import (
	"encoding/json"
	"io"
	"log"
	"net/http"
	"runtime"
	"strconv"
	"time"

	"github.com/janit/viiwork-freetoken/internal/activity"
	"github.com/janit/viiwork-freetoken/internal/balancer"
	"github.com/janit/viiwork-freetoken/internal/gpu"
	"github.com/janit/viiwork-freetoken/internal/model"
	"github.com/janit/viiwork-freetoken/internal/peer"
	"github.com/janit/viiwork/meshapi"
	"github.com/janit/viiwork/web"
)

// Version is stamped at build time and published on /v1/cluster.
var Version = "dev"

// maxRequestBody bounds what a client may send. The body is buffered whole
// because it is read twice — once to find the model and prompt, once to
// forward — and an unbounded buffer is a trivial way to exhaust a node.
const maxRequestBody = 64 << 20

type Handler struct {
	registry *peer.Registry
	balancer *balancer.Balancer
	activity *activity.Log
	prompts  *activity.PromptStore
	gpuHist  *gpu.History
	gpuCast  *gpu.Broadcaster
	gpuAvail func() bool
	cors     *CORS

	maxInFlight int
	hostname    string
}

func NewHandler(reg *peer.Registry, bal *balancer.Balancer, log *activity.Log, prompts *activity.PromptStore, maxInFlight int) *Handler {
	return &Handler{
		registry: reg, balancer: bal, activity: log, prompts: prompts,
		maxInFlight: maxInFlight, hostname: reg.Hostname(),
	}
}

func (h *Handler) SetCORS(c *CORS) { h.cors = c }

func (h *Handler) SetGPUSource(hist *gpu.History, cast *gpu.Broadcaster, avail func() bool) {
	h.gpuHist, h.gpuCast, h.gpuAvail = hist, cast, avail
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	defer func() {
		if rv := recover(); rv != nil {
			buf := make([]byte, 4096)
			n := runtime.Stack(buf, false)
			log.Printf("[PANIC] %s %s: %v\n%s", r.Method, r.URL.Path, rv, buf[:n])
			http.Error(w, `{"error":{"message":"internal server error","type":"internal"}}`, http.StatusInternalServerError)
		}
	}()

	// CORS runs before routing: the allow header has to land on every response
	// including the SSE streams and the error paths, and a preflight has to be
	// answered here or not at all, since the switch below matches only GET and
	// POST.
	if h.cors != nil && h.cors.apply(w, r) {
		return
	}

	switch {
	case r.URL.Path == meshapi.PathHealth && r.Method == "GET":
		h.handleHealth(w, r)
	case r.URL.Path == meshapi.PathModels && r.Method == "GET":
		h.handleModels(w, r)
	case r.URL.Path == meshapi.PathStatus && r.Method == "GET":
		h.handleStatus(w, r)
	case r.URL.Path == meshapi.PathCluster && r.Method == "GET":
		h.handleCluster(w, r)

	case r.URL.Path == "/" && r.Method == "GET":
		w.Header().Set("Content-Type", "text/html")
		w.Write(web.DashboardHTML)
	case r.URL.Path == "/mesh" && r.Method == "GET":
		w.Header().Set("Content-Type", "text/html")
		w.Write(web.MeshHTML)
	case r.URL.Path == "/prompt" && r.Method == "GET":
		// A full page rather than a modal: each dashboard row is a real link,
		// so a middle- or cmd-click fans a batch of requests out into
		// background tabs to be read side by side.
		w.Header().Set("Content-Type", "text/html")
		w.Write(web.PromptHTML)

	case r.URL.Path == meshapi.PathChatCompletions || r.URL.Path == meshapi.PathCompletions || r.URL.Path == meshapi.PathEmbeddings:
		if r.Method != "POST" {
			http.Error(w, `{"error":{"message":"method not allowed","type":"invalid_request"}}`, http.StatusMethodNotAllowed)
			return
		}
		h.handleProxy(w, r)

	case r.URL.Path == "/v1/metrics" && r.Method == "GET":
		h.handleMetrics(w, r)
	case r.URL.Path == "/v1/metrics/stream" && r.Method == "GET":
		h.handleMetricsStream(w, r)
	case r.URL.Path == meshapi.PathActivity && r.Method == "GET":
		h.handleActivity(w, r)
	case r.URL.Path == meshapi.PathActivityStream && r.Method == "GET":
		h.handleActivityStream(w, r)
	case r.URL.Path == meshapi.PathMeshStream && r.Method == "GET":
		h.handleMeshStream(w, r)
	case r.URL.Path == meshapi.PathPrompts && r.Method == "GET":
		h.handlePromptLookup(w, r)
	case r.URL.Path == meshapi.PathMeshPrompt && r.Method == "GET":
		h.handleMeshPrompt(w, r)
	default:
		http.NotFound(w, r)
	}
}

func (h *Handler) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	healthy := h.balancer.HealthyCount()
	if healthy == 0 {
		// A node with no healthy backend is not ready to serve, and saying so
		// is what lets a load balancer or a compose healthcheck act on it.
		w.WriteHeader(http.StatusServiceUnavailable)
	}
	json.NewEncoder(w).Encode(map[string]any{
		"status":           map[bool]string{true: "ok", false: "degraded"}[healthy > 0],
		"healthy_backends": healthy,
		"total_backends":   len(h.balancer.Backends()),
		"version":          Version,
	})
}

func (h *Handler) handleModels(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(model.ModelsResponse{Object: "list", Data: h.registry.AllModels()})
}

// handleProxy is the inference path: pick a route, forward, and record what
// happened.
func (h *Handler) handleProxy(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, maxRequestBody))
	if err != nil {
		http.Error(w, `{"error":{"message":"could not read request body","type":"invalid_request"}}`, http.StatusBadRequest)
		return
	}
	var req struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, `{"error":{"message":"request body is not valid JSON","type":"invalid_request"}}`, http.StatusBadRequest)
		return
	}
	modelName := req.Model
	if modelName == "" {
		// A client that named no model gets this node's own, which is what
		// makes a single-model node usable from a bare curl.
		modelName = h.registry.LocalModel()
	}

	routes := h.registry.Resolve(modelName)
	// A request already forwarded once is served locally or not at all. Two
	// nodes that each believe the other owns a model would otherwise bounce it
	// between them forever.
	if r.Header.Get(originHeader) != "" {
		routes = localOnly(routes)
	}

	route, err := peer.PickRoute(routes, h.maxInFlight)
	if err != nil {
		h.writeRouteError(w, modelName, err)
		return
	}

	rid := activity.NewRequestID()
	start := time.Now()
	if h.prompts != nil {
		if prompt := extractPromptText(body); prompt != "" {
			h.prompts.Store(rid, start.Unix(), modelName, prompt)
		}
	}
	// Capture wraps the outer writer, so what is recorded is what the client
	// actually received. One wrap covers both the local and peer-routed
	// branches.
	var capw *captureWriter
	if h.prompts != nil {
		w, capw = newCaptureWriter(w)
		defer func() {
			h.prompts.StoreOutput(rid, time.Now().Unix(), modelName, capw.Output(), time.Since(start).Milliseconds())
		}()
	}

	if route.Type == peer.RouteLocal {
		dest := route.Backend.Label()
		h.emitRequest(rid, route.Backend.GPUID(), meshapi.RequestStarted(modelName, dest))
		aborted := proxyRequest(w, r, body, route.Backend)
		elapsed := time.Since(start).Round(time.Millisecond)
		if aborted {
			h.emitRequest(rid, route.Backend.GPUID(), meshapi.RequestAborted(modelName, dest, elapsed))
		} else {
			h.emitRequest(rid, route.Backend.GPUID(), meshapi.RequestDone(modelName, dest, elapsed))
		}
		return
	}

	dest := meshapi.PeerLabel(route.Addr)
	h.emitRequest(rid, -1, meshapi.RequestStarted(modelName, dest))
	// Write-through in-flight: later picks on this node see the dispatch
	// immediately, rather than waiting for the next poll of the peer's status.
	if route.Peer != nil {
		route.Peer.IncLocalInFlight()
	}
	proxyToPeer(w, r, body, route.Addr, h.registry.NodeID())
	if route.Peer != nil {
		route.Peer.DecLocalInFlight()
	}
	h.emitRequest(rid, -1, meshapi.RequestDone(modelName, dest, time.Since(start).Round(time.Millisecond)))
}

func (h *Handler) writeRouteError(w http.ResponseWriter, modelName string, err error) {
	w.Header().Set("Content-Type", "application/json")
	if err == balancer.ErrBackpressure {
		// The node is working and at capacity: the client should slow down,
		// not fail over. Distinct from 503, which means this node is broken.
		w.Header().Set("Retry-After", "1")
		w.WriteHeader(http.StatusTooManyRequests)
		json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{
			"message": "all backends at capacity", "type": "rate_limit",
		}})
		return
	}
	w.Header().Set("Retry-After", "5")
	w.WriteHeader(http.StatusServiceUnavailable)
	json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{
		"message": "no healthy backend for model " + strconv.Quote(modelName), "type": "unavailable",
	}})
}

func (h *Handler) emitRequest(rid int64, gpuID int, msg string) {
	if h.activity == nil {
		return
	}
	// "%s" rather than the message as a format string: a model name or peer
	// address containing a percent sign would otherwise be mangled into
	// %!d(MISSING) on the dashboard.
	h.activity.EmitRequest(rid, gpuID, "%s", msg)
}

func localOnly(routes []peer.Route) []peer.Route {
	var out []peer.Route
	for _, r := range routes {
		if r.Type == peer.RouteLocal {
			out = append(out, r)
		}
	}
	return out
}
