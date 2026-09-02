// Package freetoken spawns and supervises FreeToken server processes, one per
// GPU, and keeps their routable state up to date.
//
// This is the genuinely engine-specific half of a mesh node. Everything else —
// the balancer, the peer registry, the dashboards — is shared with the other
// implementations of this mesh. Three things about FreeToken shape what is
// here and nowhere else:
//
//   - One GPU per process. The engine has a --tensor-parallel-size flag but
//     rejects more than one --gpu entry, so the unit of deployment is the card.
//     That makes the topology the same as viiwork's llama.cpp nodes and simpler
//     than the vLLM node's tensor-parallel groups.
//   - GPU selection has to work across two engine generations that disagree
//     about how it is spelled. See Env — this is the one place where being
//     wrong costs you two backends fighting over one card.
//   - Readiness is a field in a JSON document, not an HTTP status. /health
//     answers 200 while the model is still loading. See Probe.
package freetoken

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/janit/viiwork-freetoken/internal/balancer"
	"github.com/janit/viiwork-freetoken/internal/config"
)

// Backend is one supervised FreeToken process.
type Backend struct {
	GPUIDs []int
	Port   int
	Model  config.ModelConfig
	Engine config.FreeTokenConfig

	State *balancer.BackendState

	mu      sync.Mutex
	cmd     *exec.Cmd
	started time.Time
	// respawns counts restarts, so a model that will never load stops being
	// retried rather than keeping the machine permanently busy failing.
	respawns int
	// failures is the consecutive failed-probe count feeding the threshold
	// ladder.
	failures int
	// health is the last /health document read, kept so the manager can say
	// what a not-yet-ready backend is actually doing.
	health Health
	// gpuUUID is the card the engine reported binding, from /v1/stats. Empty
	// until the first refresh, and on any engine at or before 0.1.2.
	gpuUUID string
	// ctxLearned records that the served context length has been read from the
	// engine, so it is asked once per process rather than on every tick. It is
	// a property of the loaded checkpoint and cannot change while the process
	// lives.
	ctxLearned bool

	client *http.Client
	logOut io.Writer
}

func NewBackend(gpuIDs []int, port int, model config.ModelConfig, fc config.FreeTokenConfig, state *balancer.BackendState, timeout time.Duration) *Backend {
	return &Backend{
		GPUIDs: gpuIDs, Port: port, Model: model, Engine: fc, State: state,
		client: &http.Client{Timeout: timeout},
		logOut: os.Stdout,
	}
}

// Addr is where this backend listens. Always loopback: the process is a local
// implementation detail, and binding it to a routable interface would expose
// an unauthenticated inference server on the LAN alongside the node's own API.
func (b *Backend) Addr() string { return "127.0.0.1:" + strconv.Itoa(b.Port) }

func (b *Backend) Label() string { return b.State.Label() }

// gpuSpec is this backend's card as nvidia-smi numbers it.
//
// One entry, always. The config layer is what guarantees that (see
// config.Backends), and the engine would reject anything else.
func (b *Backend) gpuSpec() string {
	ids := make([]string, len(b.GPUIDs))
	for i, id := range b.GPUIDs {
		ids[i] = strconv.Itoa(id)
	}
	return strings.Join(ids, ",")
}

// Args builds the ft serve command line.
//
// Deliberately short. FreeToken resolves dtype, attention backend, MoE cache
// size, KV capacity, page size, CUDA-graph sizes and the tool-call and
// reasoning parsers from the checkpoint and the GPU, and does it better than a
// config file can — the right value for --page-size depends on the model
// family, and for --attention-backend on the card. What is generated here is
// the set a *node* must own because it is deciding it, not the engine:
// where to listen, what to call the model on the mesh, and the three knobs
// whose value is published as part of the mesh payload. Everything else belongs
// in extra_args. Which card to take is not here either — that is Env's job, for
// the reason given there.
func (b *Backend) Args() []string {
	args := []string{
		"serve",
		"--model", b.Model.Path,
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(b.Port),
		// Without this FreeToken advertises the model by the basename of
		// --model, so a config pointing at /models/DSV4-Flash-NVFP4 would
		// publish "DSV4-Flash-NVFP4" to the mesh and no client could ask for
		// it by the name the fleet uses.
		"--served-model-name", b.Model.Name,
		"--memory-ratio", strconv.FormatFloat(b.Engine.MemoryRatio, 'f', -1, 64),
		"--max-running-requests", strconv.Itoa(b.Engine.MaxRunningRequests),
	}
	if b.Engine.MaxSeqLen > 0 {
		args = append(args, "--max-seq-len-override", strconv.Itoa(b.Engine.MaxSeqLen))
	}
	if b.Engine.MoEBackend != "" {
		args = append(args, "--moe-backend", b.Engine.MoEBackend)
	}
	// Operator args go last so they can always override a generated one:
	// argparse takes the last occurrence.
	return append(args, b.Engine.ExtraArgs...)
}

// Env is the process environment: inherited, plus the two variables that pin
// this backend to its card.
//
// GPU selection is the one place where the engine's two generations disagree.
// FreeToken 0.1.2, the current PyPI release, has no --gpu flag at all: a server
// process takes CUDA device 0 and the only way to say which card that is comes
// from the environment. Later builds add --gpu, resolved through NVML to a
// UUID — but they also read a preset CUDA_VISIBLE_DEVICES as a quota the
// selection must stay inside, and bind the single visible device when --gpu is
// omitted.
//
// So CUDA_VISIBLE_DEVICES alone is correct on both, and --gpu is correct on
// only one. That is why Args does not generate --gpu: a node is supervising
// whichever engine the operator installed, and there is no version to probe
// for that this does not already handle.
//
// CUDA_DEVICE_ORDER is the subtle half. CUDA enumerates devices fastest-first
// by default, which is NOT nvidia-smi's order, so on a host with unlike cards
// CUDA_VISIBLE_DEVICES=1 can select a different card than gpus.devices: [1]
// meant — and the node would then report that backend's load against the wrong
// GPU's utilisation and power. PCI_BUS_ID is the order nvidia-smi uses, and
// setting it makes the config file, nvidia-smi, this node's GPU collector and
// the dashboard all name the same card the same way.
//
// Both are prepended, so an operator who exports either one deliberately still
// wins — os.Environ comes first only in the sense that a later duplicate is
// what exec honours.
//
// One consequence for extra_args: on an engine new enough to have --gpu, an
// index passed there counts *within* CUDA_VISIBLE_DEVICES, which is a
// one-element list. --gpu 0 is the only value that means anything, and it is
// what the engine already does.
func (b *Backend) Env() []string {
	return append(os.Environ(),
		"CUDA_DEVICE_ORDER=PCI_BUS_ID",
		"CUDA_VISIBLE_DEVICES="+b.gpuSpec(),
	)
}

// Start spawns the process. It does not wait for readiness; the manager's
// health loop promotes the backend to healthy when it starts serving.
func (b *Backend) Start(ctx context.Context) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.cmd != nil && b.cmd.Process != nil && b.cmd.ProcessState == nil {
		return fmt.Errorf("backend %s already running", b.Label())
	}
	cmd := exec.Command(b.Engine.Binary, b.Args()...)
	cmd.Env = b.Env()
	// Its own process group, so a shutdown can signal the whole tree.
	// FreeToken runs its scheduler, tokenizer and detokenizer as separate
	// processes, and signalling only the parent leaves those holding VRAM.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	prefix := "[" + b.Label() + "] "
	cmd.Stdout = newPrefixWriter(b.logOut, prefix)
	cmd.Stderr = newPrefixWriter(b.logOut, prefix)
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start %s: %w", b.Label(), err)
	}
	b.cmd = cmd
	b.started = time.Now()
	b.failures = 0
	b.health = Health{}
	b.ctxLearned = false
	b.State.SetStatus(balancer.StatusStarting)
	// Publish the configured shape now rather than waiting for a scrape: these
	// are what we told FreeToken to do, so they are known from the moment it is
	// spawned, and the mesh view is otherwise blank for the minutes a large
	// model takes to load.
	b.State.SetMaxModelLen(int64(b.Engine.MaxSeqLen))
	b.State.SetMaxNumSeqs(int64(b.Engine.MaxRunningRequests))
	// Reap the process so a crashed backend does not linger as a zombie. The
	// health loop is what notices it stopped answering.
	go func() { _ = cmd.Wait() }()
	return nil
}

// Stop asks the process group to exit, escalating to SIGKILL after grace.
// Signalling the group rather than the pid is what actually frees the VRAM:
// FreeToken's scheduler and tokenizer workers are children, and a SIGTERM to
// the parent alone can leave them resident.
func (b *Backend) Stop(grace time.Duration) {
	b.mu.Lock()
	cmd := b.cmd
	b.cmd = nil
	b.mu.Unlock()
	if cmd == nil || cmd.Process == nil {
		return
	}
	pgid, err := syscall.Getpgid(cmd.Process.Pid)
	if err != nil {
		pgid = cmd.Process.Pid
	}
	_ = syscall.Kill(-pgid, syscall.SIGTERM)

	done := make(chan struct{})
	go func() {
		for {
			if err := syscall.Kill(-pgid, 0); err != nil {
				close(done)
				return
			}
			time.Sleep(100 * time.Millisecond)
		}
	}()
	select {
	case <-done:
	case <-time.After(grace):
		_ = syscall.Kill(-pgid, syscall.SIGKILL)
	}
	b.State.SetStatus(balancer.StatusDead)
}

// Running reports whether a process is currently supervised.
func (b *Backend) Running() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.cmd != nil && b.cmd.Process != nil
}

func (b *Backend) pid() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.cmd == nil || b.cmd.Process == nil {
		return 0
	}
	return b.cmd.Process.Pid
}

// Uptime is how long the current process has been up.
func (b *Backend) Uptime() time.Duration {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.started.IsZero() {
		return 0
	}
	return time.Since(b.started)
}

// Health is the last /health document read. Zero before the first probe.
func (b *Backend) Health() Health {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.health
}

// GPUUUID is the card the engine reported binding on the last stats refresh.
// Empty when it reported none, which an engine at or before 0.1.2 always does.
func (b *Backend) GPUUUID() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.gpuUUID
}

// Probe reads /health and, when the engine is serving, refreshes the scheduler
// stats and RSS that ride on the mesh payload. It returns whether the backend
// will actually take a request.
//
// The vLLM node's equivalent stops at the HTTP status. Here it cannot:
// FreeToken answers /health with 200 in every lifecycle state, including the
// many minutes a frontier MoE model spends loading and any window in which a
// live cache rebuild has taken the engine out of service. Both of those 503
// every generation request. Readiness is Health.Ready, and a 200 that is not
// ready is a failed probe.
func (b *Backend) Probe(ctx context.Context) bool {
	h, ok := b.readHealth(ctx)

	b.mu.Lock()
	prev := b.health
	b.health = h
	// A live cache rebuild resizes the KV pool, which moves the context
	// ceiling. Coming out of one invalidates what we learned.
	if prev.Maintenance != "" && prev.Maintenance != maintServing && h.Ready() {
		b.ctxLearned = false
	}
	b.mu.Unlock()

	if !ok || !h.Ready() {
		return false
	}
	b.refreshStats(ctx)
	b.learnContextLen(ctx)
	b.State.SetRSSMB(readRSSMB(b.pid()))
	return true
}

// learnContextLen publishes the context this backend will actually accept, when
// the config did not pin one. Once per engine generation.
//
// Start already published freetoken.max_seq_len, which is the operator's answer
// and always wins. This covers the case where they gave none — the right
// default, since the checkpoint's own limit is usually what you want and
// restating it in YAML is a second place to get it wrong.
//
// The number published is the SMALLER of what the checkpoint allows and what
// the engine's KV pool holds, because that is the one a request is measured
// against. Those are not close on the hardware this node exists for: the engine
// sizes KV from the VRAM left after weights and the expert cache, so serving a
// model several times the size of the card leaves a pool far below the
// checkpoint's limit. Measured on an RTX 5090 serving DeepSeek-V4-Flash,
// /v1/models advertises 1,048,576 tokens and the backend rejects anything over
// 64,128. Publishing the advertised figure would put a number on the dashboard
// that the backend refuses, which is worse than a blank — a client would size a
// request by it and get a 400.
//
// Both reads are best-effort. A missing one narrows the answer rather than
// voiding it; only when neither is available is nothing published.
func (b *Backend) learnContextLen(ctx context.Context) {
	if b.Engine.MaxSeqLen > 0 {
		return
	}
	b.mu.Lock()
	done := b.ctxLearned
	b.mu.Unlock()
	if done {
		return
	}

	var limit int64
	if body, ok := b.get(ctx, "/v1/models"); ok {
		if n, ok := ParseContextLen(body); ok {
			limit = n
		}
	}
	if body, ok := b.get(ctx, "/v1/cache/status"); ok {
		if n, ok := ParseKVCapacity(body); ok && (limit == 0 || n < limit) {
			limit = n
		}
	}

	b.mu.Lock()
	// Latched either way: an engine that did not answer will not start
	// answering mid-generation, and retrying every tick would poll a
	// control-plane endpoint forever for a number that is not coming.
	b.ctxLearned = true
	b.mu.Unlock()
	if limit > 0 {
		b.State.SetMaxModelLen(limit)
	}
}

// get is a bounded GET against this backend. Used for the control-plane reads
// that are not on the health path.
func (b *Backend) get(ctx context.Context, path string) ([]byte, bool) {
	req, err := http.NewRequestWithContext(ctx, "GET", "http://"+b.Addr()+path, nil)
	if err != nil {
		return nil, false
	}
	resp, err := b.client.Do(req)
	if err != nil {
		return nil, false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, false
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 256<<10))
	if err != nil {
		return nil, false
	}
	return body, true
}

func (b *Backend) readHealth(ctx context.Context) (Health, bool) {
	req, err := http.NewRequestWithContext(ctx, "GET", "http://"+b.Addr()+"/health", nil)
	if err != nil {
		return Health{}, false
	}
	resp, err := b.client.Do(req)
	if err != nil {
		return Health{}, false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		io.Copy(io.Discard, resp.Body)
		return Health{}, false
	}
	// Bounded read: the document is a few hundred bytes, and this runs on
	// every health tick against a process we do not control.
	body, err := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	if err != nil {
		return Health{}, false
	}
	return ParseHealth(body)
}

func (b *Backend) refreshStats(ctx context.Context) {
	req, err := http.NewRequestWithContext(ctx, "GET", "http://"+b.Addr()+"/v1/stats", nil)
	if err != nil {
		return
	}
	resp, err := b.client.Do(req)
	if err != nil {
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return
	}
	st, ok := ParseStats(body)
	if !ok {
		return
	}
	b.mu.Lock()
	b.gpuUUID = st.GPUUUID
	b.mu.Unlock()
	// FreeToken publishes no queue-depth gauge, so the waiting count is
	// derived here rather than read. It is exact, not an estimate: the backend
	// binds loopback and this node is its only client, so every request the
	// engine has not started running is one of ours. Anything else would have
	// to be reported as a zero indistinguishable from an idle queue.
	waiting := b.State.InFlight() - st.NumRunning
	if waiting < 0 {
		// The two numbers are read a moment apart, and /v1/stats is a snapshot
		// taken before this node decremented its own counter. A negative is
		// that skew, not a queue.
		waiting = 0
	}
	b.State.SetSchedulerStats(st.NumRunning, waiting, st.CachePct)
}

// readRSSMB reads resident set size for a pid. Returns zero when unavailable,
// which is reported as absent rather than as a real zero.
//
// This covers the parent only. FreeToken runs its scheduler, tokenizer and
// detokenizer as separate processes holding their own memory, so this
// understates the total — and understates it by more than on the vLLM node,
// because the whole point of an offload MoE backend is that the expert weights
// live in host RAM. It is a useful trend line for one process, not an
// accounting of the model's footprint.
func readRSSMB(pid int) int64 {
	if pid <= 0 {
		return 0
	}
	data, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/statm")
	if err != nil {
		return 0
	}
	fields := strings.Fields(string(data))
	if len(fields) < 2 {
		return 0
	}
	pages, err := strconv.ParseInt(fields[1], 10, 64)
	if err != nil {
		return 0
	}
	return pages * int64(os.Getpagesize()) / (1024 * 1024)
}
