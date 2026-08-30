package freetoken

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/janit/viiwork-freetoken/internal/balancer"
	"github.com/janit/viiwork-freetoken/internal/config"
)

type recorder struct {
	mu     sync.Mutex
	events []string
}

func (r *recorder) Emit(typ string, gpuID int, format string, args ...any) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, typ+": "+fmt.Sprintf(format, args...))
}

func (r *recorder) contains(sub string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, e := range r.events {
		if len(sub) <= len(e) && contains(e, sub) {
			return true
		}
	}
	return false
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

// testManager builds a manager whose backends point at a dead port, so every
// probe fails. /bin/true stands in for the engine: it spawns successfully,
// which is what the respawn path needs, and exits immediately, which is what a
// crashed backend looks like.
func testManager(t *testing.T, tune func(*config.Config)) (*Manager, *recorder) {
	t.Helper()
	cfg := &config.Config{}
	cfg.Model = config.ModelConfig{Name: "m", Path: "/models/m"}
	cfg.GPUs = config.GPUsConfig{Devices: []int{0}, BasePort: 1}
	cfg.ApplyDefaults()
	cfg.FreeToken.Binary = "/bin/true"
	cfg.Health.FailureThreshold = 3
	cfg.Health.MaxRespawns = 2
	cfg.FreeToken.ShutdownGrace = config.Duration{Duration: 50 * time.Millisecond}
	if tune != nil {
		tune(cfg)
	}
	rec := &recorder{}
	m, err := NewManager(cfg, rec)
	if err != nil {
		t.Fatal(err)
	}
	return m, rec
}

// While a backend is starting, a failed probe is the expected case: FreeToken
// spends minutes loading weights, sizing cache pools and filling an expert
// cache before it serves. Only the startup timeout ends that patience.
func TestStartingBackendToleratesFailedProbes(t *testing.T) {
	m, _ := testManager(t, nil)
	b := m.backends[0]
	b.started = time.Now() // inside the startup window
	for i := 0; i < 10; i++ {
		m.checkOne(context.Background(), b)
	}
	if got := b.State.Status(); got != balancer.StatusStarting {
		t.Errorf("got %v, want still starting — a loading model must not be respawned", got)
	}
	if b.respawns != 0 {
		t.Errorf("respawned %d times during startup", b.respawns)
	}
}

func TestStartupTimeoutTriggersRespawn(t *testing.T) {
	m, rec := testManager(t, func(c *config.Config) {
		c.FreeToken.StartupTimeout = config.Duration{Duration: time.Millisecond}
	})
	b := m.backends[0]
	b.started = time.Now().Add(-time.Second) // well past the timeout
	m.checkOne(context.Background(), b)
	if b.respawns != 1 {
		t.Errorf("respawns = %d, want 1", b.respawns)
	}
	if !rec.contains("startup timed out") {
		t.Error("timeout should be visible on the activity feed")
	}
}

// A healthy backend must survive transient probe failures and respawn only
// once the threshold is crossed.
func TestHealthyBackendRespawnsAtThreshold(t *testing.T) {
	m, _ := testManager(t, nil)
	b := m.backends[0]
	b.State.SetStatus(balancer.StatusHealthy)

	m.checkOne(context.Background(), b)
	if b.respawns != 0 {
		t.Fatal("one failed probe must not respawn")
	}
	if b.State.Status() != balancer.StatusUnhealthy {
		t.Error("a failed probe should drop the backend out of the picker immediately")
	}
	m.checkOne(context.Background(), b)
	if b.respawns != 0 {
		t.Fatal("two failed probes must not respawn with a threshold of 3")
	}
	m.checkOne(context.Background(), b)
	if b.respawns != 1 {
		t.Errorf("respawns = %d, want 1 at the threshold", b.respawns)
	}
}

// EOF or connection-refused on the inference path means the process is
// definitively gone, so waiting out the ladder is pointless delay.
func TestHardFailureSkipsTheLadder(t *testing.T) {
	m, _ := testManager(t, nil)
	b := m.backends[0]
	b.State.SetStatus(balancer.StatusHealthy)
	b.State.NoteHardFailure()
	m.checkOne(context.Background(), b)
	if b.respawns != 1 {
		t.Errorf("respawns = %d, want 1 after a hard failure", b.respawns)
	}
}

// A model that cannot load will never load. An unbounded respawn loop turns
// that into a machine permanently busy failing.
func TestRespawnsAreBounded(t *testing.T) {
	m, rec := testManager(t, nil)
	b := m.backends[0]
	b.State.SetStatus(balancer.StatusHealthy)
	for i := 0; i < 20; i++ {
		b.State.NoteHardFailure()
		m.checkOne(context.Background(), b)
	}
	if b.respawns > m.cfg.Health.MaxRespawns {
		t.Errorf("respawned %d times, cap is %d", b.respawns, m.cfg.Health.MaxRespawns)
	}
	if b.State.Status() != balancer.StatusDead {
		t.Errorf("got %v, want dead once respawns are exhausted", b.State.Status())
	}
	if !rec.contains("dead after") {
		t.Error("giving up should be visible on the activity feed")
	}
}

// A dead backend is skipped entirely; the tick must not keep probing it.
func TestTickSkipsDeadBackends(t *testing.T) {
	m, _ := testManager(t, nil)
	b := m.backends[0]
	b.State.SetStatus(balancer.StatusDead)
	before := b.failures
	m.tick(context.Background())
	if b.failures != before {
		t.Error("a dead backend should not be probed")
	}
}

// One backend failing to spawn must not stop the others: three of four GPUs
// serving beats none.
func TestStartContinuesAfterSpawnFailure(t *testing.T) {
	m, _ := testManager(t, func(c *config.Config) {
		c.GPUs.Devices = []int{0, 1}
		c.FreeToken.Binary = "/nonexistent/ft"
	})
	m.Start(context.Background())
	if len(m.backends) != 2 {
		t.Fatalf("got %d backends", len(m.backends))
	}
	for _, b := range m.backends {
		if b.State.Status() != balancer.StatusDead {
			t.Errorf("%s: got %v, want dead", b.Label(), b.State.Status())
		}
	}
}

func TestStatesMirrorBackends(t *testing.T) {
	m, _ := testManager(t, func(c *config.Config) {
		c.GPUs.Devices = []int{0, 2, 3}
	})
	states := m.States()
	if len(states) != 3 {
		t.Fatalf("got %d states, want one per device", len(states))
	}
	// One card per backend, so every label is the mesh's single-GPU form rather
	// than the tensor-parallel-group form the vLLM node produces. The index in
	// it is the physical one the process was pinned with, so the config file,
	// nvidia-smi and the dashboard all name the same card the same way.
	if states[0].Label() != "gpu-0" || states[1].Label() != "gpu-2" || states[2].Label() != "gpu-3" {
		t.Errorf("labels: %q, %q, %q", states[0].Label(), states[1].Label(), states[2].Label())
	}
}

func TestShutdownIsIdempotent(t *testing.T) {
	m, _ := testManager(t, nil)
	m.Shutdown()
	m.Shutdown() // must not panic or double-signal
}
