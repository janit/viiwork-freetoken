package freetoken

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/janit/viiwork-freetoken/internal/balancer"
	"github.com/janit/viiwork-freetoken/internal/config"
)

func testBackend(t *testing.T, gpuIDs []int, port int, vc config.FreeTokenConfig) *Backend {
	t.Helper()
	state := balancer.NewBackendState(gpuIDs, "127.0.0.1:"+strconv.Itoa(port), time.Second)
	return NewBackend(gpuIDs, port, config.ModelConfig{Name: "my-model", Path: "/models/m"}, vc, state, time.Second)
}

func argValue(args []string, flag string) (string, bool) {
	for i, a := range args {
		if a == flag && i+1 < len(args) {
			return args[i+1], true
		}
	}
	return "", false
}

func TestArgs(t *testing.T) {
	b := testBackend(t, []int{2}, 9901, config.FreeTokenConfig{
		MemoryRatio: 0.9, MaxSeqLen: 32768, MaxRunningRequests: 8, MoEBackend: "offload",
	})
	args := b.Args()
	if args[0] != "serve" {
		t.Fatalf("subcommand: %v", args[:1])
	}
	// --model, not a positional: FreeToken's serve takes the checkpoint as a
	// flag, and the value may be a local dir, an HF repo id or an FTW dir.
	if v, _ := argValue(args, "--model"); v != "/models/m" {
		t.Errorf("model: got %q", v)
	}
	// Without --served-model-name FreeToken advertises the model by the
	// basename of --model, and no client could ask for it by the name the
	// fleet uses.
	if v, _ := argValue(args, "--served-model-name"); v != "my-model" {
		t.Errorf("served-model-name: got %q", v)
	}
	// The card is NOT chosen here. See TestEnvPinsTheCard.
	if _, ok := argValue(args, "--gpu"); ok {
		t.Error("--gpu must not be generated: the released engine does not have the flag")
	}
	if v, _ := argValue(args, "--port"); v != "9901" {
		t.Errorf("port: got %q", v)
	}
	if v, _ := argValue(args, "--max-seq-len-override"); v != "32768" {
		t.Errorf("max-seq-len-override: got %q", v)
	}
	if v, _ := argValue(args, "--max-running-requests"); v != "8" {
		t.Errorf("max-running-requests: got %q", v)
	}
	if v, _ := argValue(args, "--moe-backend"); v != "offload" {
		t.Errorf("moe-backend: got %q", v)
	}
	// Loopback only: the backend is an implementation detail, and binding it
	// wider would put an unauthenticated inference server on the LAN.
	if v, _ := argValue(args, "--host"); v != "127.0.0.1" {
		t.Errorf("host: got %q", v)
	}
}

func TestUnsetOptionalArgsOmitted(t *testing.T) {
	b := testBackend(t, []int{0}, 9901, config.FreeTokenConfig{MemoryRatio: 0.9, MaxRunningRequests: 4})
	args := strings.Join(b.Args(), " ")
	// Passing --max-seq-len-override 0 would cap the context at nothing, and
	// --moe-backend "" is not a backend name. Leaving both off is what lets
	// the engine resolve them from the checkpoint and the GPU, which is the
	// whole reason they are optional.
	if strings.Contains(args, "--max-seq-len-override") || strings.Contains(args, "--moe-backend") {
		t.Errorf("unset options must be omitted, not passed empty: %s", args)
	}
}

// Operator args go last so they can always win: argparse takes the last
// occurrence of a flag.
func TestExtraArgsComeLast(t *testing.T) {
	b := testBackend(t, []int{0}, 9901, config.FreeTokenConfig{
		MemoryRatio: 0.9, MaxRunningRequests: 4,
		ExtraArgs: []string{"--max-running-requests", "16", "--attention-backend", "trtllm"},
	})
	args := b.Args()
	last := -1
	for i, a := range args {
		if a == "--max-running-requests" {
			last = i
		}
	}
	if v := args[last+1]; v != "16" {
		t.Errorf("operator override lost: last --max-running-requests is %q, want 16", v)
	}
}

// GPU selection is where the engine's two generations disagree. FreeToken
// 0.1.2, the current PyPI release, has no --gpu flag: a server process takes
// CUDA device 0. Later builds add one, but also honour a preset
// CUDA_VISIBLE_DEVICES as a quota and bind the single visible device when
// --gpu is omitted. The environment is therefore the only mechanism correct on
// both, which is why Args generates no --gpu.
func TestEnvPinsTheCard(t *testing.T) {
	b := testBackend(t, []int{4}, 9901, config.FreeTokenConfig{MemoryRatio: 0.9})
	env := b.Env()
	if !slices.Contains(env, "CUDA_VISIBLE_DEVICES=4") {
		t.Error("the backend must be pinned to its card through the environment")
	}
	// Without this, CUDA enumerates fastest-first rather than in nvidia-smi's
	// order, and on a host with unlike cards CUDA_VISIBLE_DEVICES=4 selects a
	// different GPU than gpus.devices: [4] meant — so the node would report
	// this backend's load against another card's utilisation and power.
	if !slices.Contains(env, "CUDA_DEVICE_ORDER=PCI_BUS_ID") {
		t.Error("device order must be pinned to nvidia-smi's, or the index means something else")
	}
	// Appended, not replacing the inherited environment: the engine needs
	// PATH, HOME and HF_TOKEN like any other process.
	if len(env) <= 2 {
		t.Error("the environment must be inherited, not replaced")
	}
	for _, e := range os.Environ() {
		if strings.HasPrefix(e, "PATH=") && !slices.Contains(env, e) {
			t.Error("inherited PATH was dropped")
		}
	}
}

// serverOn starts a test server and returns a backend pointed at its port, so
// the probe path is exercised end to end without a GPU or a real vLLM.
func serverOn(t *testing.T, h http.Handler) *Backend {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	port, err := strconv.Atoi(srv.URL[strings.LastIndexByte(srv.URL, ':')+1:])
	if err != nil {
		t.Fatal(err)
	}
	return testBackend(t, []int{0}, port, config.FreeTokenConfig{MemoryRatio: 0.9, MaxRunningRequests: 4})
}

// health is a /health body for a serving engine.
const healthServing = `{"status":"ok","model":"my-model","maintenance":"serving","uptime_s":30}`

func TestProbeHealthyRefreshesStats(t *testing.T) {
	b := serverOn(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			w.Write([]byte(healthServing))
		case "/v1/stats":
			w.Write([]byte(`{"requests":{"active":3},"kv":{"used_pages":1,"total_pages":2}}`))
		default:
			http.NotFound(w, r)
		}
	}))
	b.State.IncrInFlight()
	b.State.IncrInFlight()
	b.State.IncrInFlight()
	b.State.IncrInFlight()
	if !b.Probe(context.Background()) {
		t.Fatal("probe should succeed against a serving /health")
	}
	if b.State.NumRunning() != 3 || b.State.KVCachePct() != 0.5 {
		t.Errorf("stats not refreshed: running=%d kv=%v", b.State.NumRunning(), b.State.KVCachePct())
	}
	// FreeToken publishes no queue-depth gauge, so waiting is derived: the
	// backend binds loopback and this node is its only client, so the requests
	// it has handed over that the engine has not started running are exactly
	// the queue.
	if b.State.NumWaiting() != 1 {
		t.Errorf("waiting: got %d, want 4 in flight minus 3 running", b.State.NumWaiting())
	}
}

// /v1/stats is a snapshot taken a moment before this node reads its own
// in-flight counter, so running can legitimately exceed it. That skew is not a
// negative queue.
func TestDerivedWaitingNeverGoesNegative(t *testing.T) {
	b := serverOn(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.Write([]byte(healthServing))
			return
		}
		w.Write([]byte(`{"requests":{"active":9}}`))
	}))
	if !b.Probe(context.Background()) {
		t.Fatal("probe should succeed")
	}
	if got := b.State.NumWaiting(); got != 0 {
		t.Errorf("waiting: got %d, want 0", got)
	}
}

// The trap in porting the vLLM node's probe. FreeToken answers /health with
// 200 while the model is still loading — minutes, on the models this engine
// exists to run — and 503s every generation request for that whole time. A
// node that read only the status code would advertise the backend to the mesh
// and have every routed request fail.
func TestProbeRejectsLoadingEngine(t *testing.T) {
	b := serverOn(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"loading","phase":"weights","progress":{"done_bytes":1,"total_bytes":4}}`))
	}))
	if b.Probe(context.Background()) {
		t.Error("a loading engine answers 200 and must still not count as healthy")
	}
	if got := b.Health().Describe(); got != "loading (weights, 25%)" {
		t.Errorf("the probe must retain why: got %q", got)
	}
}

// A live cache rebuild takes the engine out of service without changing
// status, and 503s requests for its duration.
func TestProbeRejectsMaintenance(t *testing.T) {
	b := serverOn(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"status":"ok","maintenance":"rebuilding"}`))
	}))
	if b.Probe(context.Background()) {
		t.Error("an engine mid-rebuild must not count as healthy")
	}
}

func TestProbeFailsOnNonOK(t *testing.T) {
	b := serverOn(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	if b.Probe(context.Background()) {
		t.Error("a 503 /health must not count as healthy")
	}
}

// A broken or absent /v1/stats must not make an otherwise serving backend look
// dead: the stats are a view for the dashboard, and health is the contract.
func TestProbeSurvivesBrokenStats(t *testing.T) {
	b := serverOn(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.Write([]byte(healthServing))
			return
		}
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	if !b.Probe(context.Background()) {
		t.Error("health must not depend on stats")
	}
}

// Something is listening on the port, but it is not the engine we spawned.
func TestProbeFailsOnForeignListener(t *testing.T) {
	b := serverOn(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("<html>nginx</html>"))
	}))
	if b.Probe(context.Background()) {
		t.Error("a 200 that is not a health document must not count as healthy")
	}
}

func TestProbeFailsWhenNothingListening(t *testing.T) {
	// Port 1 is reserved and never has a listener in a test environment.
	b := testBackend(t, []int{0}, 1, config.FreeTokenConfig{MemoryRatio: 0.9})
	if b.Probe(context.Background()) {
		t.Error("probe against a closed port must fail")
	}
}

// A subprocess can emit a line in several writes; a prefix must not land
// mid-sentence.
func TestPrefixWriterBuffersPartialLines(t *testing.T) {
	var buf bytes.Buffer
	w := newPrefixWriter(&buf, "[gpu-0] ")
	w.Write([]byte("hello "))
	w.Write([]byte("world\nsecond\n"))
	got := buf.String()
	want := "[gpu-0] hello world\n[gpu-0] second\n"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestPrefixWriterHoldsUnterminatedTail(t *testing.T) {
	var buf bytes.Buffer
	w := newPrefixWriter(&buf, "[x] ")
	w.Write([]byte("no newline yet"))
	if buf.Len() != 0 {
		t.Errorf("partial line emitted early: %q", buf.String())
	}
	w.Write([]byte("\n"))
	if got := buf.String(); got != "[x] no newline yet\n" {
		t.Errorf("got %q", got)
	}
}

func TestReadRSSMBHandlesMissingPid(t *testing.T) {
	if got := readRSSMB(0); got != 0 {
		t.Errorf("got %d, want 0 for an unknown pid", got)
	}
	// Our own pid must produce something plausible.
	if got := readRSSMB(1); got < 0 {
		t.Errorf("got negative RSS %d", got)
	}
}
