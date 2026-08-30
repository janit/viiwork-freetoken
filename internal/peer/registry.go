package peer

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/janit/viiwork-freetoken/internal/balancer"
	"github.com/janit/viiwork-freetoken/internal/model"
	"github.com/janit/viiwork/meshapi"
)

// hostOfAddr strips the port. Only a fallback for a peer too old to report its
// own hostname — see ClusterState for why the reported one is preferred.
func hostOfAddr(addr string) string {
	if i := strings.LastIndex(addr, ":"); i >= 0 {
		return addr[:i]
	}
	return addr
}

// PowerReader is node wattage as the registry needs it. Kept narrow so a node
// without power monitoring simply supplies nothing.
//
// Source names what Watts actually measured, because the two implementations of
// this mesh put physically different quantities in the same wire field: viiwork
// publishes whole-chassis draw over IPMI, and a host with no BMC publishes the
// sum of its GPU board power, which excludes CPU, fans, drives and PSU losses.
// The label is the only thing that lets a fleet total be read honestly.
type PowerReader interface {
	Watts() float64
	Available() bool
	Source() string
}

// EnergyReader is the durable kWh store's rolling totals.
//
// Optional, and absent on most instances by design: node wattage is a per-host
// measurement, so the store runs on exactly one instance per host. The others
// supply no reader and publish nothing, which omitempty turns into a blank
// rather than a claim that the host drew no power.
type EnergyReader interface {
	KWh24h() float64
	KWh30d() float64
}

type Registry struct {
	nodeID        string
	localModel    string
	backends      []*balancer.BackendState
	peers         []*PeerState
	timeout       time.Duration
	logger        *log.Logger
	client        *http.Client
	power         PowerReader
	energy        EnergyReader
	hostname      string
	listenAddr    string
	promptHistory int
	version       string
}

func NewRegistry(nodeID, localModel string, backends []*balancer.BackendState, peers []*PeerState, timeout time.Duration) *Registry {
	return &Registry{
		nodeID: nodeID, localModel: localModel, backends: backends, peers: peers,
		timeout: timeout,
		logger:  log.New(os.Stdout, "[mesh] ", log.LstdFlags),
		client:  &http.Client{Timeout: timeout},
	}
}

func (r *Registry) NodeID() string                     { return r.nodeID }
func (r *Registry) LocalModel() string                 { return r.localModel }
func (r *Registry) Backends() []*balancer.BackendState { return r.backends }
func (r *Registry) Peers() []*PeerState                { return r.peers }
func (r *Registry) Hostname() string                   { return r.hostname }
func (r *Registry) ListenAddr() string                 { return r.listenAddr }
func (r *Registry) PromptHistory() int                 { return r.promptHistory }

func (r *Registry) SetPowerReader(p PowerReader)   { r.power = p }
func (r *Registry) SetEnergyReader(e EnergyReader) { r.energy = e }
func (r *Registry) SetPromptHistory(n int)         { r.promptHistory = n }
func (r *Registry) SetVersion(v string)            { r.version = v }

// backendGPUIDs is the cards one wire backend occupies.
//
// GPUID carries the -1 group sentinel when GPUIDs is populated, so the list
// wins where present and the sentinel is never mistaken for a card.
func backendGPUIDs(b meshapi.BackendInfo) []int {
	if len(b.GPUIDs) > 0 {
		return b.GPUIDs
	}
	if b.GPUID >= 0 {
		return []int{b.GPUID}
	}
	return nil
}

// GPUModels maps GPU id to the model resident on that card, across the whole
// host rather than this instance's slice.
//
// The energy recorder needs the host's full layout, not its own: it records
// every card nvidia-smi reports, and a card whose model it cannot name has its
// draw attributed to nothing. On a multi-instance host the rest of the layout
// is only knowable from the co-located peers, which is why this reads their
// backends too — the same thing viiwork's own recorder does.
//
// Only peers reporting *this* hostname are adopted. GPU ids are per-host, so a
// remote peer's card 0 is not this host's card 0, and adopting its label would
// charge this host's power to a model on another machine. A node that does not
// yet know its own hostname adopts nothing, since it cannot tell the two apart.
//
// Local backends are written last and therefore win: this instance is
// authoritative for the cards it actually supervises.
func (r *Registry) GPUModels() map[int]string {
	out := make(map[int]string)
	if r.hostname != "" {
		for _, p := range r.peers {
			if p.Status() != StatusReachable || p.Hostname() != r.hostname {
				continue
			}
			for _, b := range p.Backends() {
				for _, id := range backendGPUIDs(b) {
					out[id] = b.Model
				}
			}
		}
	}
	for _, b := range r.backends {
		for _, id := range b.GPUIDs {
			out[id] = r.localModel
		}
	}
	return out
}

// SetLocation records the host:port this node listens on.
//
// Hostname must be a name other hosts resolve, not 0.0.0.0: peers key per-host
// deduplication on it, and a host identity derived from the dial address
// splits a multi-instance machine into several hosts on the dashboard — with a
// per-host wattage that then cannot be deduplicated.
func (r *Registry) SetLocation(hostname, listenAddr string) {
	r.hostname = hostname
	r.listenAddr = listenAddr
}

// IsKnownPeer reports whether a node id belongs to a configured peer. It is
// what stops a peer-forwarded request from being bounced onward forever.
func (r *Registry) IsKnownPeer(nodeID string) bool {
	for _, p := range r.peers {
		if p.NodeID() == nodeID {
			return true
		}
	}
	return false
}

// IsKnownAddr reports whether an address is one of this node's configured
// peers. Load bearing for the mesh prompt lookup, which fetches the address it
// is handed and echoes the response — without this check that endpoint is an
// SSRF primitive on a LAN that also carries IPMI.
func (r *Registry) IsKnownAddr(addr string) bool {
	for _, p := range r.peers {
		if p.Addr == addr {
			return true
		}
	}
	return false
}

func (r *Registry) Run(ctx context.Context, interval time.Duration) {
	if len(r.peers) == 0 {
		return
	}
	r.logger.Printf("starting peer poll loop (%d peers, interval %s)", len(r.peers), interval)
	r.PollOnce(ctx)
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			r.PollOnce(ctx)
		}
	}
}

// PollOnce refreshes every peer in parallel. One slow peer must not delay the
// rest: the poll interval is the dashboard's refresh rate for peer backend and
// GPU state, and serialising it would make the whole fleet as slow as its
// worst host.
func (r *Registry) PollOnce(ctx context.Context) {
	var wg sync.WaitGroup
	for _, p := range r.peers {
		wg.Add(1)
		go func(p *PeerState) {
			defer wg.Done()
			r.pollPeer(ctx, p)
		}(p)
	}
	wg.Wait()
}

func (r *Registry) pollPeer(ctx context.Context, p *PeerState) {
	ctx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "GET", "http://"+p.Addr+meshapi.PathStatus, nil)
	if err != nil {
		p.MarkUnreachable()
		return
	}
	resp, err := r.client.Do(req)
	if err != nil {
		p.MarkUnreachable()
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		p.MarkUnreachable()
		return
	}
	var status meshapi.StatusResponse
	// Bounded: a peer's status carries a GPU list and per-backend detail, but
	// nothing that should reach megabytes.
	body, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		p.MarkUnreachable()
		return
	}
	if err := json.Unmarshal(body, &status); err != nil {
		p.MarkUnreachable()
		return
	}
	p.Update(status)
}

// AllModels aggregates local and peer models for /v1/models, so a client
// pointed at any node sees everything the fleet serves.
func (r *Registry) AllModels() []model.ModelEntry {
	var models []model.ModelEntry
	seen := map[string]bool{}
	if r.localModel != "" {
		models = append(models, model.ModelEntry{ID: r.localModel, Object: "model", OwnedBy: "local"})
		seen[r.localModel] = true
	}
	var peerModels []model.ModelEntry
	for _, p := range r.peers {
		for _, m := range p.Models() {
			if seen[m] {
				continue
			}
			seen[m] = true
			peerModels = append(peerModels, model.ModelEntry{ID: m, Object: "model", OwnedBy: "peer"})
		}
	}
	sort.Slice(peerModels, func(i, j int) bool { return peerModels[i].ID < peerModels[j].ID })
	return append(models, peerModels...)
}

// Resolve builds the candidate routes for a model: every healthy local backend
// serving it, plus every reachable peer that reports it.
func (r *Registry) Resolve(modelName string) []Route {
	var routes []Route
	if modelName == r.localModel {
		for _, b := range r.backends {
			if b.Status() != balancer.StatusHealthy {
				continue
			}
			routes = append(routes, Route{Type: RouteLocal, Backend: b, InFlight: b.InFlight()})
		}
	}
	for _, p := range r.peers {
		if p.Status() != StatusReachable || !p.HasModel(modelName) {
			continue
		}
		routes = append(routes, Route{Type: RoutePeer, Addr: p.Addr, Peer: p, InFlight: p.TotalInFlight()})
	}
	return routes
}

// TotalInFlight is this node's own load across its local backends.
func (r *Registry) TotalInFlight() int64 {
	var n int64
	for _, b := range r.backends {
		n += b.InFlight()
	}
	return n
}

// StatusResponse builds this node's /v1/status payload.
func (r *Registry) StatusResponse(gpus []meshapi.GPUInfo, hostMemTotal, hostMemUsed int64) meshapi.StatusResponse {
	resp := meshapi.StatusResponse{
		NodeID:         r.nodeID,
		Hostname:       r.hostname,
		ListenAddr:     r.listenAddr,
		Models:         []string{r.localModel},
		TotalBackends:  len(r.backends),
		PromptHistory:  r.promptHistory,
		GPUs:           gpus,
		HostMemTotalMB: hostMemTotal,
		HostMemUsedMB:  hostMemUsed,
	}
	for _, b := range r.backends {
		resp.Backends = append(resp.Backends, r.backendInfo(b))
		resp.TotalInFlight += b.InFlight()
		if b.Status() == balancer.StatusHealthy {
			resp.HealthyBackends++
		}
	}
	if r.power != nil {
		resp.PowerWatts = r.power.Watts()
		resp.PowerAvailable = r.power.Available()
		resp.PowerSource = r.power.Source()
	}
	if r.energy != nil {
		resp.EnergyKWh24h = r.energy.KWh24h()
		resp.EnergyKWh30d = r.energy.KWh30d()
	}
	return resp
}

// backendInfo maps a local backend onto the wire type.
//
// The Slot* fields describe FreeToken's scheduler rather than llama.cpp's
// slots: SlotCount is the --max-running-requests we gave it, SlotActive is what
// /v1/stats reports actually running, SlotCtx is the context length. TokDecoded
// and TokRemain are deliberately left unset — FreeToken publishes cumulative
// token counters and a sliding-window decode rate, not per-request progress,
// and a cumulative total in a field the dashboard renders as "tokens left in
// this request" would be worse than a blank. See meshapi.BackendInfo on
// omitting what has no honest counterpart.
func (r *Registry) backendInfo(b *balancer.BackendState) meshapi.BackendInfo {
	info := meshapi.BackendInfo{
		GPUID:      b.GPUID(),
		Model:      r.localModel,
		Status:     b.Status().String(),
		InFlight:   b.InFlight(),
		RSSMB:      b.RSSMB(),
		SlotCtx:    b.MaxModelLen(),
		SlotCount:  int(b.MaxNumSeqs()),
		SlotActive: int(b.NumRunning()),
	}
	// GPUIDs is populated only for a genuine group: for a single-card backend
	// GPUID already says everything, and a one-element list would render the
	// backend as a tensor-split group on the dashboard.
	if len(b.GPUIDs) > 1 {
		info.GPUIDs = append(info.GPUIDs, b.GPUIDs...)
	}
	return info
}

// ClusterState assembles the whole-mesh snapshot this node can see.
func (r *Registry) ClusterState() meshapi.ClusterResponse {
	resp := meshapi.ClusterResponse{
		NodeID:   r.nodeID,
		Version:  r.version,
		Hostname: r.hostname,
		Local: meshapi.ClusterLocalInfo{
			Model:         r.localModel,
			ListenAddr:    r.listenAddr,
			PromptHistory: r.promptHistory,
		},
	}
	if r.power != nil {
		resp.Local.PowerWatts = r.power.Watts()
		resp.Local.PowerAvailable = r.power.Available()
		resp.Local.PowerSource = r.power.Source()
	}
	if r.energy != nil {
		resp.Local.EnergyKWh24h = r.energy.KWh24h()
		resp.Local.EnergyKWh30d = r.energy.KWh30d()
	}
	for _, b := range r.backends {
		info := r.backendInfo(b)
		resp.Local.Backends = append(resp.Local.Backends, meshapi.ClusterBackendInfo{
			GPUID: info.GPUID, GPUIDs: info.GPUIDs, Model: info.Model,
			Status: info.Status, InFlight: info.InFlight, RSSMB: info.RSSMB,
			SlotCtx: info.SlotCtx, SlotCount: info.SlotCount, SlotActive: info.SlotActive,
		})
		resp.Local.TotalInFlight += info.InFlight
	}

	models := map[string]bool{}
	if r.localModel != "" {
		models[r.localModel] = true
	}
	hosts := map[string]bool{r.hostname: true}
	for _, p := range r.peers {
		// Prefer the hostname the peer reports over one derived from the
		// address it is dialled on. They disagree exactly where it matters:
		// co-located instances are configured by IP, so deriving from the
		// address splits one machine into two hosts on screen.
		host := p.Hostname()
		if host == "" {
			host = hostOfAddr(p.Addr)
		}
		hosts[host] = true
		info := meshapi.ClusterPeerInfo{
			Addr: p.Addr, Hostname: host, Status: p.Status().String(),
		}
		if p.Status() == StatusReachable {
			info.NodeID = p.NodeID()
			info.Models = p.Models()
			info.GPUs = p.GPUs()
			info.TotalInFlight = p.TotalInFlight()
			info.HealthyBackends = p.HealthyBackends()
			info.PromptHistory = p.PromptHistory()
			info.PowerWatts = p.PowerWatts()
			info.PowerAvailable = p.PowerAvailable()
			info.PowerSource = p.PowerSource()
			info.EnergyKWh24h = p.EnergyKWh24h()
			info.EnergyKWh30d = p.EnergyKWh30d()
			info.HostMemTotalMB, info.HostMemUsedMB = p.HostMem()
			for _, b := range p.Backends() {
				info.Backends = append(info.Backends, meshapi.ClusterBackendInfo{
					GPUID: b.GPUID, GPUIDs: b.GPUIDs, Model: b.Model, Status: b.Status,
					InFlight: b.InFlight, RSSMB: b.RSSMB, SlotCtx: b.SlotCtx,
					SlotCount: b.SlotCount, SlotActive: b.SlotActive,
					TokDecoded: b.TokDecoded, TokRemain: b.TokRemain,
				})
			}
			for _, m := range info.Models {
				models[m] = true
			}
		}
		resp.Peers = append(resp.Peers, info)
	}

	// SingleHost tells the dashboard not to draw per-host comparisons that
	// would all be of one machine.
	resp.SingleHost = len(hosts) <= 1
	for m := range models {
		resp.Models = append(resp.Models, m)
	}
	sort.Strings(resp.Models)
	return resp
}
