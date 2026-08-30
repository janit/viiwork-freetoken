package config

import (
	"time"

	"github.com/janit/viiwork/energy"
)

// Default ports. Each node type in the family takes its own band so that a
// host running more than one of them does not have to renumber anything:
// viiwork uses 9001+ for backends and 91xx-95xx for nodes, the vLLM node 9601
// and 9701+, and this one the band above that.
//
// Note that these are the node's ports, not FreeToken's. The engine's own
// default is 1919; every backend here is given an explicit --port instead,
// because a node runs one process per GPU and they cannot all have it.
const (
	DefaultPort     = 9801
	DefaultBasePort = 9901
	// DefaultMeshPort matches viiwork's, and must: the whole point of the
	// contended mesh port is that the fleet view lives at one address on every
	// host, whichever implementation happens to be holding it.
	DefaultMeshPort = 8086
)

// DefaultEnergyDir mirrors viiwork's path rather than inventing a
// vendor-specific one: the store format is shared, so a host that changes
// implementation keeps its history in place.
const DefaultEnergyDir = "/var/lib/viiwork/energy"

// DefaultCORSOrigins is the tailnet the fleet is expected to sit on, and
// nothing else. A consuming application's own origin is deployment-specific
// and belongs in that deployment's config file, never here: this repo is
// publishable, and a shipped default naming someone's private service both
// leaks it and is wrong for everyone else.
// Tailscale IP ranges are not listed here: they are matched numerically via
// CORSConfig.AllowTailnetIPs, since an origin's host is an address rather than
// a name in that case and a string pattern cannot express a CIDR.
var DefaultCORSOrigins = []string{
	"*.ts.net",
	"localhost",
	"127.0.0.1",
}

// ApplyDefaults fills in everything the file left unset. It is idempotent and
// safe to call on a fully populated config.
func (c *Config) ApplyDefaults() {
	if c.Server.Host == "" {
		c.Server.Host = "0.0.0.0"
	}
	if c.Server.Port == 0 {
		c.Server.Port = DefaultPort
	}
	if c.Server.MeshPort == 0 {
		c.Server.MeshPort = DefaultMeshPort
	}
	if c.Server.CORS.AllowOrigins == nil {
		// nil means "unset, use the default"; an explicit empty list in YAML
		// decodes to a non-nil empty slice and means "send no CORS header",
		// which is a legitimate and different choice.
		c.Server.CORS.AllowOrigins = append([]string(nil), DefaultCORSOrigins...)
	}

	if c.GPUs.BasePort == 0 {
		c.GPUs.BasePort = DefaultBasePort
	}

	if c.FreeToken.Binary == "" {
		c.FreeToken.Binary = "ft"
	}
	if c.FreeToken.MaxRunningRequests == 0 {
		// FreeToken's own default. Named here rather than left to the engine
		// because it is published to the mesh as the backend's slot count and
		// is the default backpressure ceiling, so the node has to know it
		// without asking.
		c.FreeToken.MaxRunningRequests = 4
	}
	if c.FreeToken.MemoryRatio == 0 {
		c.FreeToken.MemoryRatio = 0.90
	}
	if c.FreeToken.StartupTimeout.Duration == 0 {
		// Generous on purpose, and more generous than the vLLM node's. Loading
		// here is not only weights and CUDA graphs: on an offload backend the
		// engine also fills a GPU expert cache from host memory, and the models
		// this engine exists to run are the ones too large to fit in VRAM. A
		// timeout tuned to a model that fits produces a permanent respawn loop
		// that looks like a crash and is really impatience.
		c.FreeToken.StartupTimeout = Duration{20 * time.Minute}
	}
	if c.FreeToken.ShutdownGrace.Duration == 0 {
		c.FreeToken.ShutdownGrace = Duration{30 * time.Second}
	}

	if c.Health.Interval.Duration == 0 {
		c.Health.Interval = Duration{5 * time.Second}
	}
	if c.Health.Timeout.Duration == 0 {
		c.Health.Timeout = Duration{2 * time.Second}
	}
	if c.Health.FailureThreshold == 0 {
		c.Health.FailureThreshold = 3
	}
	if c.Health.MaxRespawns == 0 {
		c.Health.MaxRespawns = 3
	}

	if c.Balancer.LatencyWindow.Duration == 0 {
		c.Balancer.LatencyWindow = Duration{30 * time.Second}
	}
	if c.Balancer.HighLoadThreshold == 0 {
		c.Balancer.HighLoadThreshold = 7
	}
	if c.Balancer.MaxInFlightPerBackend == 0 {
		// FreeToken's scheduler queue is the real constraint, so the honest
		// ceiling is what we told it to run concurrently. Anything lower leaves
		// the GPU idle while requests wait in this process instead.
		//
		// On this engine that number is small — 4 by default — which makes the
		// ceiling bite in normal operation rather than only under abuse. That
		// is the correct behaviour: a node that queues past what the scheduler
		// will run is hiding latency the mesh could have routed around.
		c.Balancer.MaxInFlightPerBackend = c.FreeToken.MaxRunningRequests
	}

	if c.Peers.PollInterval.Duration == 0 {
		c.Peers.PollInterval = Duration{10 * time.Second}
	}
	if c.Peers.Timeout.Duration == 0 {
		c.Peers.Timeout = Duration{3 * time.Second}
	}

	if c.Activity.PromptHistory == 0 {
		c.Activity.PromptHistory = 1000
	}
	if c.Activity.EventHistory == 0 {
		c.Activity.EventHistory = 2000
	}

	if c.Energy.Dir == "" {
		c.Energy.Dir = DefaultEnergyDir
	}
	if c.Energy.SampleInterval.Duration == 0 {
		c.Energy.SampleInterval = Duration{30 * time.Second}
	}
	// Geometry comes from the shared package rather than being restated here,
	// so the two implementations cannot drift into recreating each other's
	// rings.
	if c.Energy.MinuteSlots == 0 {
		c.Energy.MinuteSlots = energy.DefaultMinuteSlots
	}
	if c.Energy.HourSlots == 0 {
		c.Energy.HourSlots = energy.DefaultHourSlots
	}
	if c.Energy.DaySlots == 0 {
		c.Energy.DaySlots = energy.DefaultDaySlots
	}
}
