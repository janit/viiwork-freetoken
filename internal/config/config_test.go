package config

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/janit/viiwork/energy"
)

func writeCfg(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "cfg.yaml")
	if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

const minimal = `
model:
  name: test-model
  path: /models/test
gpus:
  devices: [0, 1]
`

func TestLoadMinimal(t *testing.T) {
	cfg, err := Load(writeCfg(t, minimal))
	if err != nil {
		t.Fatalf("a config naming only the model and its GPUs must be legal: %v", err)
	}
	if cfg.Server.Port != DefaultPort || cfg.Server.MeshPort != DefaultMeshPort {
		t.Errorf("ports not defaulted: %+v", cfg.Server)
	}
	if cfg.FreeToken.MaxRunningRequests != 4 || cfg.FreeToken.MemoryRatio != 0.90 {
		t.Errorf("freetoken defaults missing: %+v", cfg.FreeToken)
	}
	if cfg.FreeToken.Binary != "ft" {
		t.Errorf("binary = %q, want the engine's CLI name", cfg.FreeToken.Binary)
	}
}

// The backpressure ceiling defaults to what FreeToken was told to run
// concurrently. A lower number leaves the GPU idle while requests queue in this
// process instead.
func TestMaxInFlightDefaultsToMaxRunningRequests(t *testing.T) {
	cfg, err := Load(writeCfg(t, minimal+"\nfreetoken:\n  max_running_requests: 32\n"))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Balancer.MaxInFlightPerBackend != 32 {
		t.Errorf("got %d, want 32", cfg.Balancer.MaxInFlightPerBackend)
	}
}

// FreeToken serves one GPU per process, so the device list maps one-to-one
// onto backends and ports. This is the structural difference from the vLLM
// node, where the unit of deployment is a tensor-parallel group.
func TestOneBackendPerDevice(t *testing.T) {
	cfg, err := Load(writeCfg(t, `
model: {name: m, path: /p}
gpus:
  devices: [0, 2, 3]
  base_port: 9901
`))
	if err != nil {
		t.Fatal(err)
	}
	want := []Backend{
		{GPUIDs: []int{0}, Port: 9901},
		{GPUIDs: []int{2}, Port: 9902},
		{GPUIDs: []int{3}, Port: 9903},
	}
	if got := cfg.Backends(); !reflect.DeepEqual(got, want) {
		t.Errorf("got %+v, want %+v", got, want)
	}
}

// A mixed fleet shares override lists with the vLLM node, where this key is
// meaningful. Here it is not, and the error has to say why rather than report
// an unknown flag and send the operator hunting for a typo.
func TestTensorParallelOverrideExplainsItself(t *testing.T) {
	cfg, err := Load(writeCfg(t, minimal))
	if err != nil {
		t.Fatal(err)
	}
	err = cfg.ApplyOverrides([]string{"--gpus.tensor_parallel_size", "2"})
	if err == nil || !strings.Contains(err.Error(), "one GPU per process") {
		t.Fatalf("want an error explaining the engine's limit, got %v", err)
	}
	// Size 1 is what this node does anyway, so it is accepted rather than
	// failing a compose file that spells out the default.
	if err := cfg.ApplyOverrides([]string{"--gpus.tensor_parallel_size", "1"}); err != nil {
		t.Errorf("size 1 must be accepted: %v", err)
	}
}

// moe_backend is validated here because a typo is otherwise discovered minutes
// into a model load, as an argparse error from a subprocess.
func TestMoEBackendValidated(t *testing.T) {
	if _, err := Load(writeCfg(t, minimal+"\nfreetoken:\n  moe_backend: hybrid\n")); err != nil {
		t.Fatalf("hybrid is a real backend: %v", err)
	}
	_, err := Load(writeCfg(t, minimal+"\nfreetoken:\n  moe_backend: offloaded\n"))
	if err == nil || !strings.Contains(err.Error(), "moe_backend") {
		t.Fatalf("want a moe_backend error, got %v", err)
	}
}

func TestMemoryRatioValidated(t *testing.T) {
	_, err := Load(writeCfg(t, minimal+"\nfreetoken:\n  memory_ratio: 1.5\n"))
	if err == nil || !strings.Contains(err.Error(), "memory_ratio") {
		t.Fatalf("want a memory_ratio error, got %v", err)
	}
}

func TestDuplicateDeviceRejected(t *testing.T) {
	_, err := Load(writeCfg(t, "model: {name: m, path: /p}\ngpus: {devices: [0, 0]}\n"))
	if err == nil || !strings.Contains(err.Error(), "twice") {
		t.Fatalf("want a duplicate-device error, got %v", err)
	}
}

func TestRequiredFields(t *testing.T) {
	for _, tc := range []struct{ name, body, want string }{
		{"no model name", "model: {path: /p}\ngpus: {devices: [0]}", "model.name"},
		{"no model path", "model: {name: m}\ngpus: {devices: [0]}", "model.path"},
		{"no devices", "model: {name: m, path: /p}", "gpus.devices"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Load(writeCfg(t, tc.body))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("want an error naming %s, got %v", tc.want, err)
			}
		})
	}
}

// A backend port landing on the node's own listener is a startup failure with
// a confusing message, so it is caught at load.
func TestBackendPortCollisionRejected(t *testing.T) {
	_, err := Load(writeCfg(t, `
server: {port: 9701}
model: {name: m, path: /p}
gpus: {devices: [0], base_port: 9701}
`))
	if err == nil || !strings.Contains(err.Error(), "collides") {
		t.Fatalf("want a collision error, got %v", err)
	}
}

// yaml.v3 decodes a bare integer into time.Duration as nanoseconds, which
// turns "timeout: 3" into 3ns and a node that fails every request.
func TestBareIntegerDurationRejected(t *testing.T) {
	_, err := Load(writeCfg(t, minimal+"\npeers:\n  timeout: 3\n"))
	if err == nil || !strings.Contains(err.Error(), "duration") {
		t.Fatalf("want a duration error, got %v", err)
	}
}

func TestDurationParses(t *testing.T) {
	cfg, err := Load(writeCfg(t, minimal+"\npeers:\n  timeout: \"7s\"\nfreetoken:\n  startup_timeout: \"20m\"\n"))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Peers.Timeout.Duration != 7*time.Second {
		t.Errorf("timeout: got %v", cfg.Peers.Timeout)
	}
	if cfg.FreeToken.StartupTimeout.Duration != 20*time.Minute {
		t.Errorf("startup_timeout: got %v", cfg.FreeToken.StartupTimeout)
	}
}

// An explicit empty list means "send no CORS header" and must survive
// defaulting, which is a different thing from the key being absent.
func TestExplicitEmptyCORSSurvivesDefaults(t *testing.T) {
	cfg, err := Load(writeCfg(t, minimal+"\nserver:\n  cors:\n    allow_origins: []\n"))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Server.CORS.AllowOrigins == nil || len(cfg.Server.CORS.AllowOrigins) != 0 {
		t.Errorf("explicit empty list was replaced by defaults: %v", cfg.Server.CORS.AllowOrigins)
	}
}

func TestApplyOverrides(t *testing.T) {
	cfg, err := Load(writeCfg(t, minimal))
	if err != nil {
		t.Fatal(err)
	}
	if err := cfg.ApplyOverrides([]string{"--server.port", "9999", "--gpus.devices", "2,3,4", "--model.name", "other"}); err != nil {
		t.Fatal(err)
	}
	if cfg.Server.Port != 9999 {
		t.Errorf("port: got %d", cfg.Server.Port)
	}
	if !reflect.DeepEqual(cfg.GPUs.Devices, []int{2, 3, 4}) {
		t.Errorf("devices: got %v", cfg.GPUs.Devices)
	}
	if cfg.Model.Name != "other" {
		t.Errorf("model: got %q", cfg.Model.Name)
	}
}

func TestUnknownOverrideRejected(t *testing.T) {
	cfg := &Config{}
	if err := cfg.ApplyOverrides([]string{"--nope", "1"}); err == nil {
		t.Fatal("want an error for an unknown override")
	}
}

func TestApplyDefaultsIdempotent(t *testing.T) {
	cfg, err := Load(writeCfg(t, minimal))
	if err != nil {
		t.Fatal(err)
	}
	before := *cfg
	cfg.ApplyDefaults()
	if !reflect.DeepEqual(before, *cfg) {
		t.Error("ApplyDefaults is not idempotent")
	}
}

// --- energy ---

func TestEnergyDefaultsOffWithSharedGeometry(t *testing.T) {
	var c Config
	c.ApplyDefaults()
	if c.Energy.Enabled {
		t.Error("energy must default off: it needs a writable volume, and node wattage double-counts if two instances on a host both record")
	}
	if c.Energy.Dir != DefaultEnergyDir {
		t.Errorf("dir = %q, want %q", c.Energy.Dir, DefaultEnergyDir)
	}
	if c.Energy.SampleInterval.Duration != 30*time.Second {
		t.Errorf("sample_interval = %v, want 30s", c.Energy.SampleInterval.Duration)
	}
	// Geometry must come from the shared package, or the two implementations
	// can recreate each other's rings and discard the history.
	if c.Energy.MinuteSlots != energy.DefaultMinuteSlots ||
		c.Energy.HourSlots != energy.DefaultHourSlots ||
		c.Energy.DaySlots != energy.DefaultDaySlots {
		t.Errorf("geometry = %d/%d/%d, want viiwork's %d/%d/%d",
			c.Energy.MinuteSlots, c.Energy.HourSlots, c.Energy.DaySlots,
			energy.DefaultMinuteSlots, energy.DefaultHourSlots, energy.DefaultDaySlots)
	}
}

// Enabling energy without naming a directory is legal and gets the shared
// default. Whether that path is actually writable is not knowable here — it is
// a property of the deployment's volumes — so energy.Open reports it at startup
// and the node degrades to publishing no kWh rather than refusing to serve.
func TestEnergyEnabledWithoutDirTakesTheDefault(t *testing.T) {
	cfg, err := Load(writeCfg(t, minimal+"\nenergy:\n  enabled: true\n"))
	if err != nil {
		t.Fatalf("energy.enabled with no dir must be legal: %v", err)
	}
	if cfg.Energy.Dir != DefaultEnergyDir {
		t.Errorf("dir = %q, want the default %q", cfg.Energy.Dir, DefaultEnergyDir)
	}
}

func TestEnergyBlockParses(t *testing.T) {
	cfg, err := Load(writeCfg(t, minimal+"\nenergy:\n  enabled: true\n  dir: /var/lib/viiwork/energy\n  sample_interval: \"15s\"\n"))
	if err != nil {
		t.Fatalf("energy block should parse: %v", err)
	}
	if !cfg.Energy.Enabled {
		t.Error("enabled not read")
	}
	if cfg.Energy.SampleInterval.Duration != 15*time.Second {
		t.Errorf("sample_interval = %v, want 15s", cfg.Energy.SampleInterval.Duration)
	}
}

func TestEnergyDefaultDirMatchesViiwork(t *testing.T) {
	// Shared path, not a vendor-specific one: the store format is shared, so a
	// host that changes implementation keeps its history where it already is.
	if DefaultEnergyDir != "/var/lib/viiwork/energy" {
		t.Errorf("DefaultEnergyDir = %q, want viiwork's path", DefaultEnergyDir)
	}
}
