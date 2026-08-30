// Package config loads viiwork-freetoken.yaml and turns it into the shape the
// rest of the node needs.
//
// The layout deliberately mirrors viiwork's config where the concept is the
// same (server, peers, balancer, activity), because an operator running a
// mixed fleet edits both files and gratuitous divergence is a tax on them.
// Where FreeToken genuinely differs the config says so plainly rather than
// pretending the engines are the same. Two differences drive most of this
// file:
//
//   - FreeToken serves one GPU per process. It has a --tensor-parallel-size
//     flag, but the released engine has no way to place more than one card and
//     the unreleased one rejects more than one --gpu entry, so there is no
//     tensor-parallel group to configure: the unit of deployment is the card,
//     exactly as in viiwork. See GPUsConfig.
//   - Its reason for existing is MoE offload — running a model far larger than
//     VRAM by keeping experts in host memory. freetoken.moe_backend is
//     therefore a first-class key here rather than an extra_arg, because on
//     this engine it is the difference between a model loading and not.
package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// Duration is a time.Duration that parses from a YAML string ("30s", "15m").
// yaml.v3 will happily decode a bare integer into a time.Duration as a
// nanosecond count, which turns "timeout: 3" into 3ns and a node that fails
// every request — so the string form is required.
type Duration struct {
	time.Duration
}

func (d *Duration) UnmarshalYAML(value *yaml.Node) error {
	var s string
	if err := value.Decode(&s); err != nil {
		return fmt.Errorf("duration must be a quoted string like \"30s\": %w", err)
	}
	parsed, err := time.ParseDuration(s)
	if err != nil {
		return err
	}
	d.Duration = parsed
	return nil
}

func (d Duration) MarshalYAML() (any, error) { return d.String(), nil }

type Config struct {
	Server    ServerConfig    `yaml:"server"`
	Node      NodeConfig      `yaml:"node"`
	Model     ModelConfig     `yaml:"model"`
	GPUs      GPUsConfig      `yaml:"gpus"`
	FreeToken FreeTokenConfig `yaml:"freetoken"`
	Health    HealthConfig    `yaml:"health"`
	Balancer  BalancerConfig  `yaml:"balancer"`
	Peers     PeersConfig     `yaml:"peers"`
	Activity  ActivityConfig  `yaml:"activity"`
	Energy    EnergyConfig    `yaml:"energy"`
}

type ServerConfig struct {
	Host string `yaml:"host"`
	Port int    `yaml:"port"`
	// MeshPort is a second listener whose "/" is the mesh dashboard. It is
	// contended, not assigned: every instance on the host asks for it and the
	// OS gives it to one, with the losers retrying. That is what makes the
	// fleet view reachable at a fixed address without a designated instance or
	// a reverse proxy. Zero disables the listener.
	MeshPort int        `yaml:"mesh_port"`
	CORS     CORSConfig `yaml:"cors"`
}

// CORSConfig is the browser-origin allowlist. This node authenticates nothing
// and reachability is the authorization model, so the list is not protecting
// the API from anyone who can already reach it with curl — it stops a page in
// a tailnet member's browser from driving the fleet through that member's
// network position. Keep it narrow. An empty list sends no CORS header at all.
type CORSConfig struct {
	AllowOrigins []string `yaml:"allow_origins"`
	// AllowTailnetIPs additionally allows origins addressed by a literal
	// Tailscale IP. MagicDNS names are the common case, but a tailnet is just
	// as often browsed by its 100.x address, and an allowlist that quietly
	// failed for half the ways you reach the same host would be worse than no
	// allowlist. A pointer so an explicit `false` is distinguishable from
	// absent, which defaults to true.
	AllowTailnetIPs *bool `yaml:"allow_tailnet_ips"`
}

type NodeConfig struct {
	// Hostname overrides os.Hostname(). It has to be a name other hosts can
	// resolve, because peers key per-host deduplication on it: co-located
	// instances are configured by IP, and deriving a host identity from the
	// dial address splits one machine into several on the dashboard.
	Hostname string `yaml:"hostname"`
	// ID pins the node id across restarts. Left empty a random one is minted,
	// which is fine — nothing durable is keyed on it.
	ID string `yaml:"id"`
}

type ModelConfig struct {
	// Name is the model id this node advertises to the mesh and answers
	// requests for. It is what a client asks for and what the dashboard groups
	// by, so it must be unique per model across the fleet — two nodes claiming
	// one name are treated as replicas of it and will be load-balanced against
	// each other.
	Name string `yaml:"name"`
	// Path is what FreeToken loads, passed as --model: a local directory, a
	// HuggingFace repo id, or a directory in FreeToken's own FTW fast-load
	// format, which the engine auto-detects.
	Path string `yaml:"path"`
}

type GPUsConfig struct {
	// Devices lists the physical GPU indices this instance owns, as nvidia-smi
	// numbers them. Several instances can share a host by taking disjoint
	// slices, and each device gets its own backend process.
	//
	// These are the same numbers internal/gpu reads from nvidia-smi, and the
	// node holds them to that: each backend is pinned with CUDA_VISIBLE_DEVICES
	// under CUDA_DEVICE_ORDER=PCI_BUS_ID, which is the order nvidia-smi itself
	// enumerates in. Without that second variable CUDA orders devices
	// fastest-first, and on a host with unlike cards an index here would select
	// a different GPU than the one the dashboard shows it against. See
	// freetoken.Backend.Env.
	Devices []int `yaml:"devices"`
	// BasePort is the first port handed to a backend; each subsequent backend
	// takes the next one.
	BasePort int `yaml:"base_port"`
}

type FreeTokenConfig struct {
	// Binary is the FreeToken CLI. Not a path to a Python entry point: `ft` is
	// a console script inside the engine's virtualenv, so this is usually
	// either "ft" with that venv on PATH, or an absolute path into its bin/.
	Binary string `yaml:"binary"`
	// ExtraArgs are appended verbatim to the ft serve command line, after every
	// argument this config generates, so an operator can always win an
	// argument with the defaults.
	//
	// This is where the engine's long tail lives — --attention-backend,
	// --page-size, --moe-cache-rate, --cache-type and the rest. They are not
	// mirrored as YAML keys because FreeToken resolves them from the checkpoint
	// and the GPU on its own, and a mirrored key with a zero value cannot
	// express "let the engine decide".
	ExtraArgs []string `yaml:"extra_args"`
	// MaxSeqLen is FreeToken's --max-seq-len-override, published to the mesh as
	// the backend's context size. Zero leaves the checkpoint's own limit in
	// place, and the mesh then reports the context size as unknown rather than
	// as zero.
	MaxSeqLen int `yaml:"max_seq_len"`
	// MaxRunningRequests is FreeToken's --max-running-requests: how many
	// requests its scheduler will run concurrently. Published as the backend's
	// slot count, and the honest ceiling for balancer.MaxInFlightPerBackend.
	//
	// The engine's default is 4, two orders of magnitude below vLLM's, and that
	// is not a conservative guess to be tuned away: it is what an edge-native
	// engine can hold in a consumer card's VRAM alongside an expert cache.
	// Raising it costs KV capacity.
	MaxRunningRequests int `yaml:"max_running_requests"`
	// MemoryRatio is FreeToken's --memory-ratio: the fraction of FREE VRAM the
	// engine may use for weights, expert cache and KV together. Free, not
	// total — a card already holding another process's allocation is measured
	// as it is, so this is not the same quantity as vLLM's
	// --gpu-memory-utilization even though both are a 0..1 fraction.
	MemoryRatio float64 `yaml:"memory_ratio"`
	// MoEBackend is FreeToken's --moe-backend: "fused", "offload", "cpu",
	// "hybrid", or empty for the engine's own choice.
	//
	// A first-class key because it is the one that decides whether a frontier
	// MoE model runs at all on the card in front of you. "fused" keeps every
	// expert resident and is the fast path for a model that fits; the
	// offload family streams experts from host memory over PCIe or computes
	// them on the CPU, which is how a 290B model runs on a 32GB card. Left
	// empty the engine picks, using a `ft bench bw` profile when one exists.
	MoEBackend string `yaml:"moe_backend"`
	// StartupTimeout is how long a backend may take to report a healthy
	// /health before it is declared failed. Generous by default: FreeToken
	// loads weights, sizes its cache pools and captures CUDA graphs before it
	// serves, and on an offload backend it also fills an expert cache from
	// host memory.
	StartupTimeout Duration `yaml:"startup_timeout"`
	// ShutdownGrace is how long a backend gets to exit on SIGTERM before
	// SIGKILL.
	ShutdownGrace Duration `yaml:"shutdown_grace"`
}

type HealthConfig struct {
	// Interval is the health-check and metrics-scrape period.
	Interval Duration `yaml:"interval"`
	// Timeout bounds one health probe.
	Timeout Duration `yaml:"timeout"`
	// FailureThreshold is how many consecutive failed probes mark a backend
	// unhealthy and trigger a respawn.
	FailureThreshold int `yaml:"failure_threshold"`
	// MaxRespawns is how many times a backend is restarted before it is left
	// dead. A model that cannot load will never load, and an unbounded respawn
	// loop turns that into a machine that is permanently busy failing.
	MaxRespawns int `yaml:"max_respawns"`
}

type BalancerConfig struct {
	LatencyWindow Duration `yaml:"latency_window"`
	// HighLoadThreshold switches the picker from lowest-latency-idle to
	// least-loaded once this many requests are in flight on the node.
	HighLoadThreshold int `yaml:"high_load_threshold"`
	// MaxInFlightPerBackend is the backpressure ceiling; past it the node
	// returns 429 rather than queueing without bound. Defaults to
	// freetoken.max_running_requests, because the engine's own scheduler queue
	// is the real constraint and a lower number here just leaves the GPU idle.
	MaxInFlightPerBackend int `yaml:"max_in_flight_per_backend"`
}

type PeersConfig struct {
	// Hosts are peer node addresses ("gb1:9302"). Peering is static and
	// symmetric by convention: list this node on the peers it lists, or
	// routing works in only one direction.
	Hosts        []string `yaml:"hosts"`
	PollInterval Duration `yaml:"poll_interval"`
	Timeout      Duration `yaml:"timeout"`
}

type ActivityConfig struct {
	// PromptHistory is how many request prompts and outputs to keep in memory.
	// Published to the mesh so the dashboard sizes its own list from this
	// number rather than keeping a second copy that can drift.
	PromptHistory int `yaml:"prompt_history"`
	// EventHistory is the size of the activity ring the SSE streams replay on
	// connect. It must comfortably exceed a busy minute's events: a consumer
	// rebuilds in-flight state from the replay, and anything that fell off the
	// ring is a request the dashboard cannot learn about.
	EventHistory int `yaml:"event_history"`
}

// EnergyConfig configures the durable kWh store.
//
// Disabled by default, for two reasons. It needs a writable directory that
// outlives the container, which a default cannot invent. And node wattage is a
// whole-host measurement: a host running several instances must enable this on
// exactly ONE of them, or the same draw is recorded several times over. There
// is no way for a node to detect that on its own, so the safe default is off.
//
// What this node records differs from viiwork in one way that matters. With no
// uniform BMC across the NVIDIA hosts, node wattage is the sum of per-GPU board
// power rather than a chassis reading, so there is no aggregate to divide and
// no baseline to recover — see energy.Direct. The store records "nvidia-smi" as
// its source label so a history read outside its mesh context still says what
// it measured.
type EnergyConfig struct {
	Enabled bool   `yaml:"enabled"`
	Dir     string `yaml:"dir"`
	// SampleInterval is how often power is read. Records are always one per
	// minute; sampling faster and averaging is what keeps a card that swings
	// between idle and load from being misrepresented by whichever instant the
	// sample happened to land on.
	SampleInterval Duration `yaml:"sample_interval"`
	// Ring geometry. These are compatibility surface, not tunables: the store
	// format is shared with viiwork, and changing a slot count recreates the
	// rings and discards the history they held.
	MinuteSlots int `yaml:"minute_slots"`
	HourSlots   int `yaml:"hour_slots"`
	DaySlots    int `yaml:"day_slots"`
}

// Load reads a config file, applies defaults for everything absent, and
// validates. Defaults are applied before validation so a minimal file is
// legal — the required fields are the ones no default can invent: which model
// to serve, and where it lives.
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	cfg := &Config{}
	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	cfg.ApplyDefaults()
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return cfg, nil
}

// Backend describes one FreeToken process: the GPU it runs on and the port it
// listens on. Derived from the device list rather than configured, so the
// config and the process table cannot disagree.
type Backend struct {
	GPUIDs []int
	Port   int
}

// Backends assigns one backend per device, in the order the devices are
// listed.
//
// GPUIDs is a slice of one rather than a bare int because the mesh wire format
// and the balancer both carry a backend's cards as a set, and viiwork's AMD
// nodes and the vLLM node do put more than one card behind a backend. Keeping
// the shape means this node's payloads are the same shape as theirs, and means
// tensor parallelism can arrive here as a change to this function alone if the
// engine ever supports it.
func (c *Config) Backends() []Backend {
	out := make([]Backend, 0, len(c.GPUs.Devices))
	for i, dev := range c.GPUs.Devices {
		out = append(out, Backend{GPUIDs: []int{dev}, Port: c.GPUs.BasePort + i})
	}
	return out
}

func (c *Config) Validate() error {
	if c.Model.Name == "" {
		return fmt.Errorf("model.name is required: it is the id this node advertises to the mesh")
	}
	if c.Model.Path == "" {
		return fmt.Errorf("model.path is required: the local directory, HuggingFace repo id or FTW directory FreeToken should load")
	}
	if len(c.GPUs.Devices) == 0 {
		return fmt.Errorf("gpus.devices is required: list the GPU indices this instance owns, as nvidia-smi numbers them")
	}
	seen := map[int]bool{}
	for _, d := range c.GPUs.Devices {
		if d < 0 {
			return fmt.Errorf("gpus.devices contains a negative index (%d)", d)
		}
		if seen[d] {
			return fmt.Errorf("gpus.devices lists GPU %d twice: two backends pinned to one card will fight for its VRAM", d)
		}
		seen[d] = true
	}
	if c.Server.Port == c.Server.MeshPort && c.Server.MeshPort != 0 {
		return fmt.Errorf("server.port and server.mesh_port are both %d", c.Server.Port)
	}
	// A backend port colliding with the node's own listener is a startup
	// failure with a confusing message, so catch it here where the fix is
	// obvious.
	for _, b := range c.Backends() {
		if b.Port == c.Server.Port || (c.Server.MeshPort != 0 && b.Port == c.Server.MeshPort) {
			return fmt.Errorf("backend port %d collides with server.port/mesh_port: move gpus.base_port", b.Port)
		}
	}
	if c.FreeToken.MemoryRatio <= 0 || c.FreeToken.MemoryRatio > 1 {
		return fmt.Errorf("freetoken.memory_ratio must be in (0,1], got %v", c.FreeToken.MemoryRatio)
	}
	switch c.FreeToken.MoEBackend {
	case "", "fused", "offload", "cpu", "hybrid":
	default:
		return fmt.Errorf("freetoken.moe_backend %q is not one of fused, offload, cpu, hybrid (empty lets the engine choose)", c.FreeToken.MoEBackend)
	}
	return nil
}

// ApplyOverrides applies "--dotpath value" CLI arguments over the loaded
// config, matching viiwork's override convention so the two nodes are driven
// the same way from a compose file.
func (c *Config) ApplyOverrides(args []string) error {
	for i := 0; i < len(args); i++ {
		if !strings.HasPrefix(args[i], "--") {
			continue
		}
		key := strings.TrimPrefix(args[i], "--")
		if i+1 >= len(args) {
			return fmt.Errorf("override --%s has no value", key)
		}
		val := args[i+1]
		i++
		if err := c.set(key, val); err != nil {
			return err
		}
	}
	return nil
}

func (c *Config) set(key, val string) error {
	atoi := func() (int, error) { return strconv.Atoi(val) }
	switch key {
	case "server.port":
		v, err := atoi()
		if err != nil {
			return fmt.Errorf("--server.port: %w", err)
		}
		c.Server.Port = v
	case "server.host":
		c.Server.Host = val
	case "server.mesh_port":
		v, err := atoi()
		if err != nil {
			return fmt.Errorf("--server.mesh_port: %w", err)
		}
		c.Server.MeshPort = v
	case "model.name":
		c.Model.Name = val
	case "model.path":
		c.Model.Path = val
	case "gpus.base_port":
		v, err := atoi()
		if err != nil {
			return fmt.Errorf("--gpus.base_port: %w", err)
		}
		c.GPUs.BasePort = v
	case "gpus.tensor_parallel_size":
		// Accepted only to fail with a message that explains itself. A mixed
		// fleet shares compose files and override lists with the vLLM node,
		// where this key is meaningful; here it is not, and "unknown override"
		// would send the operator looking for a typo.
		if val != "1" {
			return fmt.Errorf("--gpus.tensor_parallel_size %s: FreeToken serves one GPU per process, so a backend always spans exactly one card; list every card in gpus.devices instead", val)
		}
	case "gpus.devices":
		var ids []int
		for _, f := range strings.Split(val, ",") {
			v, err := strconv.Atoi(strings.TrimSpace(f))
			if err != nil {
				return fmt.Errorf("--gpus.devices: %q is not a GPU index", f)
			}
			ids = append(ids, v)
		}
		c.GPUs.Devices = ids
	case "node.hostname":
		c.Node.Hostname = val
	default:
		return fmt.Errorf("unknown override --%s", key)
	}
	return nil
}
