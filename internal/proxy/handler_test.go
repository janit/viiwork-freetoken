package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/janit/viiwork-freetoken/internal/activity"
	"github.com/janit/viiwork-freetoken/internal/balancer"
	"github.com/janit/viiwork-freetoken/internal/peer"
	"github.com/janit/viiwork/meshapi"
)

// fakeVLLM stands in for a vLLM server on the inference path.
func fakeVLLM(t *testing.T, h http.HandlerFunc) *balancer.BackendState {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	st := balancer.NewBackendState([]int{0}, strings.TrimPrefix(srv.URL, "http://"), time.Second)
	st.SetStatus(balancer.StatusHealthy)
	st.SetMaxModelLen(32768)
	st.SetMaxNumSeqs(64)
	return st
}

type harness struct {
	h       *Handler
	events  *activity.Log
	prompts *activity.PromptStore
	reg     *peer.Registry
}

func newHarness(t *testing.T, modelName string, states []*balancer.BackendState, peers []*peer.PeerState) *harness {
	t.Helper()
	events := activity.NewLog(100)
	prompts := activity.NewPromptStore(100)
	reg := peer.NewRegistry("node-1", modelName, states, peers, time.Second)
	reg.SetLocation("testhost", "testhost:9601")
	reg.SetPromptHistory(prompts.Max())
	bal := balancer.New(states, 7, 8)
	return &harness{h: NewHandler(reg, bal, events, prompts, 8), events: events, prompts: prompts, reg: reg}
}

func post(t *testing.T, h http.Handler, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	r := httptest.NewRequest("POST", path, strings.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	return w
}

func get(t *testing.T, h http.Handler, path string) *httptest.ResponseRecorder {
	t.Helper()
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("GET", path, nil))
	return w
}

func TestChatCompletionRoutesToLocalBackend(t *testing.T) {
	backend := fakeVLLM(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != meshapi.PathChatCompletions {
			t.Errorf("backend got path %q", r.URL.Path)
		}
		w.Write([]byte(`{"choices":[{"message":{"content":"hello there"}}]}`))
	})
	hn := newHarness(t, "my-model", []*balancer.BackendState{backend}, nil)

	w := post(t, hn.h, meshapi.PathChatCompletions,
		`{"model":"my-model","messages":[{"role":"user","content":"hi"}]}`)
	if w.Code != http.StatusOK {
		t.Fatalf("got %d: %s", w.Code, w.Body)
	}
	// X-GPU-Backend is how a client attributes a response to a card.
	if got := w.Header().Get("X-GPU-Backend"); got != "gpu-0" {
		t.Errorf("X-GPU-Backend: got %q, want gpu-0", got)
	}
}

// A client that names no model gets this node's own, which is what makes a
// single-model node usable from a bare curl.
func TestEmptyModelDefaultsToLocal(t *testing.T) {
	backend := fakeVLLM(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	})
	hn := newHarness(t, "my-model", []*balancer.BackendState{backend}, nil)
	if w := post(t, hn.h, meshapi.PathChatCompletions, `{"messages":[]}`); w.Code != http.StatusOK {
		t.Fatalf("got %d: %s", w.Code, w.Body)
	}
}

// The activity messages are protocol: the dashboard splits on the arrow and
// matches the terminal word to clear an in-flight row.
func TestRequestEmitsProtocolActivityMessages(t *testing.T) {
	backend := fakeVLLM(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"choices":[{"message":{"content":"x"}}]}`))
	})
	hn := newHarness(t, "my-model", []*balancer.BackendState{backend}, nil)
	post(t, hn.h, meshapi.PathChatCompletions, `{"model":"my-model","messages":[]}`)

	events := hn.events.Recent()
	var start, done *meshapi.Event
	for i := range events {
		if events[i].Type != meshapi.EventRequest {
			continue
		}
		if meshapi.IsRequestTerminal(events[i].Message) {
			done = &events[i]
		} else {
			start = &events[i]
		}
	}
	if start == nil || done == nil {
		t.Fatalf("want a start and a terminal event, got %+v", events)
	}
	model, dest, ok := meshapi.SplitRequestMessage(start.Message)
	if !ok || model != "my-model" || dest != "gpu-0" {
		t.Errorf("start message %q parsed as (%q, %q, %v)", start.Message, model, dest, ok)
	}
	if start.RequestID != done.RequestID {
		t.Errorf("start and done must share a request id: %d vs %d", start.RequestID, done.RequestID)
	}
}

// A model name containing a percent sign must not be mangled into a format
// verb on the dashboard.
func TestModelNameWithPercentSurvives(t *testing.T) {
	backend := fakeVLLM(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"choices":[{"message":{"content":"x"}}]}`))
	})
	hn := newHarness(t, "odd%dmodel", []*balancer.BackendState{backend}, nil)
	post(t, hn.h, meshapi.PathChatCompletions, `{"model":"odd%dmodel","messages":[]}`)
	for _, e := range hn.events.Recent() {
		if strings.Contains(e.Message, "MISSING") {
			t.Errorf("message mangled by format expansion: %q", e.Message)
		}
	}
}

func TestPromptAndOutputCaptured(t *testing.T) {
	backend := fakeVLLM(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"choices":[{"message":{"content":"the answer"}}]}`))
	})
	hn := newHarness(t, "m", []*balancer.BackendState{backend}, nil)
	post(t, hn.h, meshapi.PathChatCompletions,
		`{"model":"m","messages":[{"role":"user","content":"the question"}]}`)

	var rid int64
	for _, e := range hn.events.Recent() {
		if e.RequestID != 0 {
			rid = e.RequestID
		}
	}
	entry, ok := hn.prompts.Get(rid)
	if !ok {
		t.Fatalf("nothing stored for rid %d", rid)
	}
	if !strings.Contains(entry.Prompt, "the question") {
		t.Errorf("prompt: got %q", entry.Prompt)
	}
	if entry.Output != "the answer" {
		t.Errorf("output: got %q", entry.Output)
	}
}

// A streamed response must reach the client token by token, and be recorded as
// the assembled text rather than as raw SSE frames.
func TestStreamingResponseCaptured(t *testing.T) {
	backend := fakeVLLM(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, tok := range []string{"Hel", "lo", " world"} {
			io.WriteString(w, `data: {"choices":[{"delta":{"content":"`+tok+`"}}]}`+"\n\n")
			w.(http.Flusher).Flush()
		}
		io.WriteString(w, "data: [DONE]\n\n")
	})
	hn := newHarness(t, "m", []*balancer.BackendState{backend}, nil)
	w := post(t, hn.h, meshapi.PathChatCompletions, `{"model":"m","stream":true,"messages":[{"role":"user","content":"q"}]}`)
	if !strings.Contains(w.Body.String(), "Hel") {
		t.Fatalf("stream not relayed: %s", w.Body)
	}
	var rid int64
	for _, e := range hn.events.Recent() {
		if e.RequestID != 0 {
			rid = e.RequestID
		}
	}
	entry, _ := hn.prompts.Get(rid)
	if entry.Output != "Hello world" {
		t.Errorf("assembled output: got %q, want %q", entry.Output, "Hello world")
	}
}

// No healthy backend is a 503 that will clear on respawn; saturation is a 429
// against a working node. Collapsing them loses the distinction a client
// retrying against a mesh needs.
func TestNoBackendIs503(t *testing.T) {
	dead := balancer.NewBackendState([]int{0}, "127.0.0.1:1", time.Second)
	dead.SetStatus(balancer.StatusDead)
	hn := newHarness(t, "m", []*balancer.BackendState{dead}, nil)
	w := post(t, hn.h, meshapi.PathChatCompletions, `{"model":"m","messages":[]}`)
	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("got %d, want 503", w.Code)
	}
	if w.Header().Get("Retry-After") == "" {
		t.Error("503 should carry Retry-After")
	}
}

func TestUnknownModelIs503(t *testing.T) {
	backend := fakeVLLM(t, func(w http.ResponseWriter, r *http.Request) {})
	hn := newHarness(t, "m", []*balancer.BackendState{backend}, nil)
	w := post(t, hn.h, meshapi.PathChatCompletions, `{"model":"nobody-serves-this","messages":[]}`)
	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("got %d, want 503", w.Code)
	}
}

func TestSaturationIs429(t *testing.T) {
	backend := fakeVLLM(t, func(w http.ResponseWriter, r *http.Request) {})
	for i := 0; i < 8; i++ {
		backend.IncrInFlight()
	}
	hn := newHarness(t, "m", []*balancer.BackendState{backend}, nil)
	w := post(t, hn.h, meshapi.PathChatCompletions, `{"model":"m","messages":[]}`)
	if w.Code != http.StatusTooManyRequests {
		t.Errorf("got %d, want 429", w.Code)
	}
}

func TestMalformedBodyIs400(t *testing.T) {
	backend := fakeVLLM(t, func(w http.ResponseWriter, r *http.Request) {})
	hn := newHarness(t, "m", []*balancer.BackendState{backend}, nil)
	if w := post(t, hn.h, meshapi.PathChatCompletions, `not json`); w.Code != http.StatusBadRequest {
		t.Errorf("got %d, want 400", w.Code)
	}
}

func TestMethodNotAllowed(t *testing.T) {
	hn := newHarness(t, "m", nil, nil)
	if w := get(t, hn.h, meshapi.PathChatCompletions); w.Code != http.StatusMethodNotAllowed {
		t.Errorf("got %d, want 405", w.Code)
	}
}

// Two nodes that each believe the other owns a model would bounce a request
// between them forever without this.
func TestForwardedRequestIsNotForwardedAgain(t *testing.T) {
	peerSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("an already-forwarded request must not be forwarded again")
	}))
	defer peerSrv.Close()
	p := peer.NewPeerState(strings.TrimPrefix(peerSrv.URL, "http://"))
	p.Update(meshapi.StatusResponse{NodeID: "node-2", Models: []string{"m"}})

	// No healthy local backend, and the only peer is off limits.
	hn := newHarness(t, "m", nil, []*peer.PeerState{p})
	r := httptest.NewRequest("POST", meshapi.PathChatCompletions, strings.NewReader(`{"model":"m","messages":[]}`))
	r.Header.Set(originHeader, "node-9")
	w := httptest.NewRecorder()
	hn.h.ServeHTTP(w, r)
	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("got %d, want 503 rather than a second forward", w.Code)
	}
}

func TestStatusIsValidProtocolPayload(t *testing.T) {
	backend := fakeVLLM(t, func(w http.ResponseWriter, r *http.Request) {})
	hn := newHarness(t, "my-model", []*balancer.BackendState{backend}, nil)
	w := get(t, hn.h, meshapi.PathStatus)
	if w.Code != http.StatusOK {
		t.Fatalf("got %d", w.Code)
	}
	var got meshapi.StatusResponse
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatalf("status is not a valid protocol payload: %v", err)
	}
	if got.NodeID != "node-1" || got.Hostname != "testhost" {
		t.Errorf("identity missing: %+v", got)
	}
	if len(got.Models) != 1 || got.Models[0] != "my-model" {
		t.Errorf("models: %v", got.Models)
	}
	if len(got.Backends) != 1 || got.Backends[0].Status != meshapi.StatusHealthy {
		t.Errorf("backends: %+v", got.Backends)
	}
	// The dashboard sizes its own prompt list from this rather than keeping a
	// second copy of the number.
	if got.PromptHistory != hn.prompts.Max() {
		t.Errorf("prompt_history: got %d, want %d", got.PromptHistory, hn.prompts.Max())
	}
	// vLLM's scheduler shape maps onto the slot fields.
	if got.Backends[0].SlotCtx != 32768 || got.Backends[0].SlotCount != 64 {
		t.Errorf("scheduler shape not published: %+v", got.Backends[0])
	}
}

func TestClusterIsValidProtocolPayload(t *testing.T) {
	backend := fakeVLLM(t, func(w http.ResponseWriter, r *http.Request) {})
	hn := newHarness(t, "my-model", []*balancer.BackendState{backend}, nil)
	w := get(t, hn.h, meshapi.PathCluster)
	var got meshapi.ClusterResponse
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatalf("cluster is not a valid protocol payload: %v", err)
	}
	if got.Local.Model != "my-model" || len(got.Local.Backends) != 1 {
		t.Errorf("local info: %+v", got.Local)
	}
	// A node with no chassis control must publish nothing rather than an empty
	// list, which would read as "enabled, no hosts".
	if got.PowerControl != nil {
		t.Errorf("power_control must be absent on a node that controls nothing: %+v", got.PowerControl)
	}
}

func TestModelsAggregatesPeers(t *testing.T) {
	p := peer.NewPeerState("gb1:9302")
	p.Update(meshapi.StatusResponse{NodeID: "node-2", Models: []string{"amd-model"}})
	hn := newHarness(t, "nvidia-model", nil, []*peer.PeerState{p})
	w := get(t, hn.h, meshapi.PathModels)
	var got struct {
		Data []struct{ ID string } `json:"data"`
	}
	json.Unmarshal(w.Body.Bytes(), &got)
	ids := map[string]bool{}
	for _, d := range got.Data {
		ids[d.ID] = true
	}
	if !ids["nvidia-model"] || !ids["amd-model"] {
		t.Errorf("a client on any node must see the whole fleet: %v", ids)
	}
}

func TestHealthReportsDegradedWithNoBackends(t *testing.T) {
	dead := balancer.NewBackendState([]int{0}, "127.0.0.1:1", time.Second)
	dead.SetStatus(balancer.StatusDead)
	hn := newHarness(t, "m", []*balancer.BackendState{dead}, nil)
	w := get(t, hn.h, meshapi.PathHealth)
	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("got %d, want 503 so a compose healthcheck can act on it", w.Code)
	}
}

func TestDashboardPagesServed(t *testing.T) {
	hn := newHarness(t, "m", nil, nil)
	for _, path := range []string{"/", "/mesh", "/prompt"} {
		w := get(t, hn.h, path)
		if w.Code != http.StatusOK {
			t.Errorf("%s: got %d", path, w.Code)
		}
		if !strings.Contains(w.Header().Get("Content-Type"), "text/html") {
			t.Errorf("%s: content type %q", path, w.Header().Get("Content-Type"))
		}
	}
}

// The mesh page is viiwork's, unmodified, so the fleet view is identical
// whichever node serves it.
func TestMeshPageIsTheSharedDashboard(t *testing.T) {
	hn := newHarness(t, "m", nil, nil)
	body := get(t, hn.h, "/mesh").Body.String()
	for _, want := range []string{meshapi.PathMeshStream, meshapi.PathCluster} {
		if !strings.Contains(body, want) {
			t.Errorf("mesh page does not reference %s", want)
		}
	}
}

func TestPanicRecovered(t *testing.T) {
	hn := newHarness(t, "m", nil, nil)
	// gpuInfo panics if the history is a non-nil interface holding nil; force
	// a panic through a handler that will dereference it.
	hn.h.gpuHist = nil
	w := get(t, hn.h, "/v1/metrics")
	if w.Code != http.StatusOK {
		t.Errorf("metrics on a node without GPUs should degrade, got %d", w.Code)
	}
}
