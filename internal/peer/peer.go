// Package peer is this node's view of the rest of the mesh: which nodes exist,
// what they serve, how loaded they are, and where a request for a given model
// should go.
//
// Peering is static and symmetric by convention. A node polls the addresses in
// its config and is polled by the nodes that list it; there is no discovery
// and no gossip, because the fleet is a handful of machines on a known network
// and a config line is easier to reason about than a membership protocol.
package peer

import (
	"sync"
	"sync/atomic"

	"github.com/janit/viiwork/meshapi"
)

type PeerStatus int

const (
	StatusUnreachable PeerStatus = iota
	StatusReachable
)

func (s PeerStatus) String() string {
	if s == StatusReachable {
		return "reachable"
	}
	return "unreachable"
}

// PeerState is one remote node as this one last saw it.
type PeerState struct {
	Addr string

	// localInFlight tracks requests this node has dispatched to the peer and
	// not yet had a response for. It is updated write-through on every
	// peer-bound call, so the picker is not blind between polls — the poll
	// interval is typically 10s, far longer than a burst. Combined with the
	// polled total via max() in TotalInFlight.
	localInFlight atomic.Int64

	mu              sync.RWMutex
	nodeID          string
	hostname        string
	listenAddr      string
	status          PeerStatus
	models          []string
	backends        []meshapi.BackendInfo
	gpus            []meshapi.GPUInfo
	totalInFlight   int64
	healthyBackends int
	totalBackends   int
	powerWatts      float64
	powerAvailable  bool
	powerSource     string
	hostMemTotalMB  int64
	hostMemUsedMB   int64
	energyKWh24h    float64
	energyKWh30d    float64
	promptHistory   int
}

func NewPeerState(addr string) *PeerState {
	return &PeerState{Addr: addr, status: StatusUnreachable}
}

func (p *PeerState) Update(resp meshapi.StatusResponse) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.nodeID = resp.NodeID
	p.hostname = resp.Hostname
	p.listenAddr = resp.ListenAddr
	p.status = StatusReachable
	p.models = resp.Models
	p.backends = append(p.backends[:0], resp.Backends...)
	p.gpus = append(p.gpus[:0], resp.GPUs...)
	p.totalInFlight = resp.TotalInFlight
	p.healthyBackends = resp.HealthyBackends
	p.totalBackends = resp.TotalBackends
	p.powerWatts = resp.PowerWatts
	p.powerAvailable = resp.PowerAvailable
	p.powerSource = resp.PowerSource
	p.hostMemTotalMB = resp.HostMemTotalMB
	p.hostMemUsedMB = resp.HostMemUsedMB
	p.energyKWh24h = resp.EnergyKWh24h
	p.energyKWh30d = resp.EnergyKWh30d
	p.promptHistory = resp.PromptHistory
}

// MarkUnreachable clears everything derived from a poll.
//
// Clearing rather than keeping the last known values is deliberate: a peer we
// cannot reach is not serving, and leaving its models listed would route
// requests into a hole. The dashboard shows the host as unreachable instead,
// which is the truth.
func (p *PeerState) MarkUnreachable() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.status = StatusUnreachable
	p.models = nil
	p.backends = nil
	p.gpus = nil
	p.totalInFlight = 0
	p.healthyBackends = 0
	p.powerWatts = 0
	p.powerAvailable = false
	p.powerSource = ""
	p.hostMemTotalMB = 0
	p.hostMemUsedMB = 0
	p.energyKWh24h = 0
	p.energyKWh30d = 0
}

func (p *PeerState) NodeID() string {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.nodeID
}

func (p *PeerState) Status() PeerStatus {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.status
}

func (p *PeerState) Hostname() string {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.hostname
}

func (p *PeerState) ListenAddr() string {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.listenAddr
}

func (p *PeerState) Models() []string {
	p.mu.RLock()
	defer p.mu.RUnlock()
	out := make([]string, len(p.models))
	copy(out, p.models)
	return out
}

// HasModel reports whether this peer serves the named model.
//
// Callers on the request path must use this rather than ranging over Models():
// that method defensively copies the whole slice on every call, so route
// resolution would allocate once per peer per request purely to run a
// membership test. Here the check happens under the same read lock with
// nothing escaping.
func (p *PeerState) HasModel(name string) bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	for _, m := range p.models {
		if m == name {
			return true
		}
	}
	return false
}

func (p *PeerState) Backends() []meshapi.BackendInfo {
	p.mu.RLock()
	defer p.mu.RUnlock()
	out := make([]meshapi.BackendInfo, len(p.backends))
	copy(out, p.backends)
	return out
}

func (p *PeerState) GPUs() []meshapi.GPUInfo {
	p.mu.RLock()
	defer p.mu.RUnlock()
	out := make([]meshapi.GPUInfo, len(p.gpus))
	copy(out, p.gpus)
	return out
}

// TotalInFlight returns the larger of the last-polled count and the
// write-through local count. Polled data reflects traffic from every node
// (correct across the mesh but stale between polls); local data tracks
// decisions this node just made (fresh, but only ours). Max keeps the picker
// from underestimating load in either dimension.
func (p *PeerState) TotalInFlight() int64 {
	p.mu.RLock()
	polled := p.totalInFlight
	p.mu.RUnlock()
	if local := p.localInFlight.Load(); local > polled {
		return local
	}
	return polled
}

func (p *PeerState) IncLocalInFlight()    { p.localInFlight.Add(1) }
func (p *PeerState) DecLocalInFlight()    { p.localInFlight.Add(-1) }
func (p *PeerState) LocalInFlight() int64 { return p.localInFlight.Load() }

func (p *PeerState) HealthyBackends() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.healthyBackends
}

func (p *PeerState) PromptHistory() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.promptHistory
}

func (p *PeerState) PowerWatts() float64 {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.powerWatts
}

func (p *PeerState) PowerAvailable() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.powerAvailable
}

func (p *PeerState) PowerSource() string {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.powerSource
}

func (p *PeerState) HostMem() (totalMB, usedMB int64) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.hostMemTotalMB, p.hostMemUsedMB
}

func (p *PeerState) EnergyKWh24h() float64 {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.energyKWh24h
}

func (p *PeerState) EnergyKWh30d() float64 {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.energyKWh30d
}
