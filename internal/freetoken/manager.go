package freetoken

import (
	"context"
	"log"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/janit/viiwork-freetoken/internal/balancer"
	"github.com/janit/viiwork-freetoken/internal/config"
	"github.com/janit/viiwork/meshapi"
)

// EventSink is the activity log, as the manager needs it. Kept this narrow so
// the process supervisor does not depend on the whole activity package.
type EventSink interface {
	Emit(typ string, gpuID int, format string, args ...any)
}

// Manager spawns the configured backends and keeps them alive.
type Manager struct {
	cfg      *config.Config
	backends []*Backend
	logger   *log.Logger
	events   EventSink
	// gpuUUIDs is the host's nvidia-smi index to card-UUID map, supplied once
	// at startup and read-only afterwards. Nil when nvidia-smi is unavailable,
	// which disables the pinning check rather than failing it.
	gpuUUIDs map[int]string

	stopOnce sync.Once

	mu sync.Mutex
	// lastPhase is the load phase each backend was last seen in, so progress is
	// logged on change rather than on every tick.
	lastPhase map[*Backend]string
}

func NewManager(cfg *config.Config, events EventSink) (*Manager, error) {
	specs := cfg.Backends()
	m := &Manager{
		cfg:    cfg,
		logger: log.New(os.Stdout, "[manager] ", log.LstdFlags),
		events: events,
	}
	for _, s := range specs {
		state := balancer.NewBackendState(s.GPUIDs, "127.0.0.1:"+strconv.Itoa(s.Port), cfg.Balancer.LatencyWindow.Duration)
		m.backends = append(m.backends, NewBackend(s.GPUIDs, s.Port, cfg.Model, cfg.FreeToken, state, cfg.Health.Timeout.Duration))
	}
	return m, nil
}

// SetGPUUUIDs supplies the host's index-to-UUID map, against which each
// backend's self-reported card is checked once it becomes healthy. Call it
// before Start; it is not a runtime knob, and a UUID does not change for the
// life of a card.
//
// Optional. Passing nil, or never calling it, skips the check.
func (m *Manager) SetGPUUUIDs(u map[int]string) { m.gpuUUIDs = u }

func (m *Manager) Backends() []*Backend { return m.backends }

func (m *Manager) States() []*balancer.BackendState {
	out := make([]*balancer.BackendState, len(m.backends))
	for i, b := range m.backends {
		out[i] = b.State
	}
	return out
}

// Start spawns every backend. A backend that fails to spawn is marked dead and
// the rest still start: a node serving three of four GPUs is far better than a
// node serving none, and the dead one is visible on the dashboard.
func (m *Manager) Start(ctx context.Context) {
	for _, b := range m.backends {
		if err := b.Start(ctx); err != nil {
			m.logger.Printf("%s failed to spawn: %v", b.Label(), err)
			b.State.SetStatus(balancer.StatusDead)
			m.emit(meshapi.EventBackend, b, "%s failed to spawn: %v", b.Label(), err)
			continue
		}
		m.logger.Printf("%s spawned (pid %d, GPU %v, port %d)", b.Label(), b.pid(), b.GPUIDs, b.Port)
		m.emit(meshapi.EventBackend, b, "%s starting on GPU %v", b.Label(), b.GPUIDs)
	}
}

// Run drives the health loop until the context is cancelled.
func (m *Manager) Run(ctx context.Context) {
	t := time.NewTicker(m.cfg.Health.Interval.Duration)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			m.tick(ctx)
		}
	}
}

func (m *Manager) tick(ctx context.Context) {
	var wg sync.WaitGroup
	for _, b := range m.backends {
		if b.State.Status() == balancer.StatusDead {
			continue
		}
		wg.Add(1)
		go func(b *Backend) {
			defer wg.Done()
			m.checkOne(ctx, b)
		}(b)
	}
	wg.Wait()
}

func (m *Manager) checkOne(ctx context.Context, b *Backend) {
	ok := b.Probe(ctx)

	b.mu.Lock()
	prev := b.State.Status()
	if ok {
		b.failures = 0
		b.mu.Unlock()
		if prev != balancer.StatusHealthy {
			b.State.SetStatus(balancer.StatusHealthy)
			m.logger.Printf("%s healthy", b.Label())
			m.emit(meshapi.EventBackend, b, "%s healthy", b.Label())
			m.verifyPinning(b)
		}
		return
	}

	// A hard failure observed on the inference path (EOF, connection refused)
	// means the process is definitively gone, so the three-strike ladder is
	// pointless delay — respawn after this one probe.
	hard := b.State.TakeHardFailure()
	b.failures++
	failures := b.failures
	b.mu.Unlock()

	// While a backend is still starting, a failed probe is the expected case:
	// FreeToken loads weights, sizes its cache pools and captures CUDA graphs
	// before it serves, and on an offload MoE backend it then fills a GPU
	// expert cache from host memory. On the models this engine exists to run
	// that is minutes. Only the startup timeout ends that patience.
	//
	// Unlike the vLLM node, the wait is not silent: /health reports a phase and
	// a byte count while loading, so the log says which of those minutes is
	// which, and a backend that is stuck is distinguishable from one that is
	// merely slow.
	if prev == balancer.StatusStarting && !hard {
		if b.Uptime() > b.Engine.StartupTimeout.Duration {
			m.logger.Printf("%s failed to become ready within %s (%s)", b.Label(), b.Engine.StartupTimeout.Duration, b.Health().Describe())
			m.emit(meshapi.EventBackend, b, "%s startup timed out after %s", b.Label(), b.Engine.StartupTimeout.Duration)
			m.respawn(ctx, b)
			return
		}
		m.noteProgress(b)
		return
	}

	if prev == balancer.StatusHealthy {
		b.State.SetStatus(balancer.StatusUnhealthy)
		// Say why when the engine told us. A backend taken out of service by a
		// live cache rebuild reports that in /health, and is a different
		// problem from one that stopped answering.
		if why := b.Health().Describe(); why != "" {
			m.emit(meshapi.EventBackend, b, "%s unhealthy: %s", b.Label(), why)
		} else {
			m.emit(meshapi.EventBackend, b, "%s unhealthy", b.Label())
		}
	}
	if hard || failures >= m.cfg.Health.FailureThreshold {
		m.respawn(ctx, b)
	}
}

// verifyPinning compares the card the engine says it bound against the card
// this node told it to take. Once per engine generation, on the transition to
// healthy — which also re-checks after a respawn, for free.
//
// Worth doing because the failure it catches is otherwise silent. This node
// pins with CUDA_VISIBLE_DEVICES, whose index CUDA resolves in whatever order
// CUDA_DEVICE_ORDER selects; get that wrong and the backend loads, serves and
// answers every probe while the GPU panel attributes its load, its VRAM and its
// wattage to a neighbouring card. Nothing else in the node would notice, and on
// a host recording energy the misattribution is written to disk.
//
// A mismatch is reported, not acted on. The backend is serving correctly and
// taking it out of the mesh would trade a wrong label for a lost GPU.
//
// Three cases are deliberately not a mismatch, all of them "unknown":
// FreeToken only began reporting its GPU after 0.1.2, so the released engine
// reports none; nvidia-smi may be unavailable, leaving nothing to compare
// against; and a hypothetical multi-card backend has no single expected card.
// Absent is not zero here as much as anywhere else on the wire — a check that
// guessed would cry wolf on every node running the released engine.
func (m *Manager) verifyPinning(b *Backend) {
	got := b.GPUUUID()
	if got == "" || len(m.gpuUUIDs) == 0 || len(b.GPUIDs) != 1 {
		return
	}
	want, ok := m.gpuUUIDs[b.GPUIDs[0]]
	if !ok || want == got {
		return
	}
	m.logger.Printf("%s: pinned to GPU %d (%s) but the engine bound %s — check gpus.devices and CUDA_DEVICE_ORDER",
		b.Label(), b.GPUIDs[0], want, got)
	m.emit(meshapi.EventBackend, b, "%s is on the wrong GPU: expected %s, engine bound %s",
		b.Label(), want, got)
}

func (m *Manager) respawn(ctx context.Context, b *Backend) {
	b.mu.Lock()
	if b.respawns >= m.cfg.Health.MaxRespawns {
		b.mu.Unlock()
		b.State.SetStatus(balancer.StatusDead)
		m.logger.Printf("%s dead after %d respawns; leaving it down", b.Label(), b.respawns)
		m.emit(meshapi.EventBackend, b, "%s dead after %d respawns", b.Label(), b.respawns)
		return
	}
	b.respawns++
	n := b.respawns
	b.mu.Unlock()

	m.logger.Printf("%s respawning (attempt %d/%d)", b.Label(), n, m.cfg.Health.MaxRespawns)
	m.emit(meshapi.EventBackend, b, "%s respawning (%d/%d)", b.Label(), n, m.cfg.Health.MaxRespawns)
	b.Stop(m.cfg.FreeToken.ShutdownGrace.Duration)
	if err := b.Start(ctx); err != nil {
		m.logger.Printf("%s respawn failed: %v", b.Label(), err)
		b.State.SetStatus(balancer.StatusDead)
	}
}

// Shutdown stops every backend. Safe to call more than once.
func (m *Manager) Shutdown() {
	m.stopOnce.Do(func() {
		var wg sync.WaitGroup
		for _, b := range m.backends {
			wg.Add(1)
			go func(b *Backend) {
				defer wg.Done()
				b.Stop(m.cfg.FreeToken.ShutdownGrace.Duration)
			}(b)
		}
		wg.Wait()
		m.logger.Println("all backends stopped")
	})
}

// noteProgress logs what a loading backend is doing, once per distinct phase.
// Once per phase rather than once per tick because the health loop runs every
// few seconds and a large model loads for minutes: a line per tick would be
// hundreds of them, and the reader would stop looking.
func (m *Manager) noteProgress(b *Backend) {
	desc := b.Health().Describe()
	if desc == "" {
		return
	}
	phase := b.Health().Phase
	m.mu.Lock()
	if m.lastPhase == nil {
		m.lastPhase = map[*Backend]string{}
	}
	same := m.lastPhase[b] == phase
	m.lastPhase[b] = phase
	m.mu.Unlock()
	if same {
		return
	}
	m.logger.Printf("%s %s", b.Label(), desc)
	m.emit(meshapi.EventBackend, b, "%s %s", b.Label(), desc)
}

func (m *Manager) emit(typ string, b *Backend, format string, args ...any) {
	if m.events != nil {
		m.events.Emit(typ, b.State.GPUID(), format, args...)
	}
}
