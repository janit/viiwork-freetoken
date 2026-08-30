package peer

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/janit/viiwork-freetoken/internal/balancer"
	"github.com/janit/viiwork/meshapi"
)

func healthyState(gpuIDs []int) *balancer.BackendState {
	s := balancer.NewBackendState(gpuIDs, "127.0.0.1:1", time.Second)
	s.SetStatus(balancer.StatusHealthy)
	return s
}

func peerServing(t *testing.T, status meshapi.StatusResponse) (*PeerState, *httptest.Server) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != meshapi.PathStatus {
			http.NotFound(w, r)
			return
		}
		json.NewEncoder(w).Encode(status)
	}))
	t.Cleanup(srv.Close)
	return NewPeerState(strings.TrimPrefix(srv.URL, "http://")), srv
}

func TestPollUpdatesPeer(t *testing.T) {
	p, _ := peerServing(t, meshapi.StatusResponse{
		NodeID: "n2", Hostname: "gb2", Models: []string{"other-model"},
		TotalInFlight: 3, HealthyBackends: 2,
		GPUs: []meshapi.GPUInfo{{GPUID: 0, Util: 50}},
	})
	r := NewRegistry("n1", "my-model", nil, []*PeerState{p}, time.Second)
	r.PollOnce(context.Background())

	if p.Status() != StatusReachable {
		t.Fatal("peer should be reachable")
	}
	if !p.HasModel("other-model") {
		t.Error("model not recorded")
	}
	if p.TotalInFlight() != 3 || p.HealthyBackends() != 2 {
		t.Errorf("load not recorded: %d in flight, %d healthy", p.TotalInFlight(), p.HealthyBackends())
	}
	if len(p.GPUs()) != 1 {
		t.Error("peer GPUs not recorded — the fleet view needs them for every host")
	}
}

// An unreachable peer must stop advertising models, or requests route into a
// hole.
func TestUnreachablePeerClearsModels(t *testing.T) {
	p, srv := peerServing(t, meshapi.StatusResponse{NodeID: "n2", Models: []string{"gone"}})
	r := NewRegistry("n1", "m", nil, []*PeerState{p}, 200*time.Millisecond)
	r.PollOnce(context.Background())
	if !p.HasModel("gone") {
		t.Fatal("setup: model should be present")
	}
	srv.Close()
	r.PollOnce(context.Background())
	if p.Status() != StatusUnreachable {
		t.Error("peer should be unreachable")
	}
	if p.HasModel("gone") {
		t.Error("an unreachable peer must not still advertise models")
	}
	if len(r.Resolve("gone")) != 0 {
		t.Error("no route should resolve to an unreachable peer")
	}
}

func TestAllModelsAggregates(t *testing.T) {
	p, _ := peerServing(t, meshapi.StatusResponse{NodeID: "n2", Models: []string{"zeta", "alpha"}})
	r := NewRegistry("n1", "local-model", nil, []*PeerState{p}, time.Second)
	r.PollOnce(context.Background())
	got := r.AllModels()
	if len(got) != 3 {
		t.Fatalf("got %d models, want 3: %+v", len(got), got)
	}
	// Local first, then peers sorted — a stable order so the list does not
	// reshuffle between polls.
	if got[0].ID != "local-model" || got[0].OwnedBy != "local" {
		t.Errorf("local model should come first: %+v", got[0])
	}
	if got[1].ID != "alpha" || got[2].ID != "zeta" {
		t.Errorf("peer models should be sorted: %+v", got[1:])
	}
}

func TestAllModelsDeduplicates(t *testing.T) {
	p, _ := peerServing(t, meshapi.StatusResponse{NodeID: "n2", Models: []string{"shared"}})
	r := NewRegistry("n1", "shared", nil, []*PeerState{p}, time.Second)
	r.PollOnce(context.Background())
	if got := r.AllModels(); len(got) != 1 {
		t.Errorf("a model served by both must appear once: %+v", got)
	}
}

func TestResolveLocalAndPeer(t *testing.T) {
	p, _ := peerServing(t, meshapi.StatusResponse{NodeID: "n2", Models: []string{"m"}})
	local := healthyState([]int{0})
	r := NewRegistry("n1", "m", []*balancer.BackendState{local}, []*PeerState{p}, time.Second)
	r.PollOnce(context.Background())

	routes := r.Resolve("m")
	if len(routes) != 2 {
		t.Fatalf("got %d routes, want local + peer", len(routes))
	}
	if len(r.Resolve("unknown")) != 0 {
		t.Error("an unknown model must resolve to nothing")
	}
}

func TestResolveSkipsUnhealthyLocal(t *testing.T) {
	bad := balancer.NewBackendState([]int{0}, "a", time.Second)
	bad.SetStatus(balancer.StatusDead)
	r := NewRegistry("n1", "m", []*balancer.BackendState{bad}, nil, time.Second)
	if got := r.Resolve("m"); len(got) != 0 {
		t.Errorf("got %d routes, want none", len(got))
	}
}

// The reported hostname wins over one derived from the dial address: they
// disagree exactly where it matters, on a host whose instances are configured
// by IP.
func TestClusterPrefersReportedHostname(t *testing.T) {
	p, _ := peerServing(t, meshapi.StatusResponse{NodeID: "n2", Hostname: "gb1", Models: []string{"m"}})
	r := NewRegistry("n1", "m", nil, []*PeerState{p}, time.Second)
	r.SetLocation("gb1", "gb1:9601")
	r.PollOnce(context.Background())

	state := r.ClusterState()
	if state.Peers[0].Hostname != "gb1" {
		t.Errorf("got %q, want the reported hostname", state.Peers[0].Hostname)
	}
	// Both instances are on gb1, so this is one host, not two.
	if !state.SingleHost {
		t.Error("co-located instances must not read as separate hosts")
	}
}

func TestClusterFallsBackToAddrHostname(t *testing.T) {
	// A peer too old to report a hostname.
	p, _ := peerServing(t, meshapi.StatusResponse{NodeID: "n2", Models: []string{"m"}})
	r := NewRegistry("n1", "m", nil, []*PeerState{p}, time.Second)
	r.PollOnce(context.Background())
	if got := r.ClusterState().Peers[0].Hostname; got != "127.0.0.1" {
		t.Errorf("got %q, want the address-derived fallback", got)
	}
}

// A single-card backend must not render as a tensor-split group.
func TestBackendInfoOmitsGPUIDsForSingleCard(t *testing.T) {
	r := NewRegistry("n1", "m", []*balancer.BackendState{healthyState([]int{3})}, nil, time.Second)
	info := r.StatusResponse(nil, 0, 0)
	if len(info.Backends) != 1 {
		t.Fatal("no backend reported")
	}
	if info.Backends[0].GPUID != 3 {
		t.Errorf("gpu_id: got %d", info.Backends[0].GPUID)
	}
	if info.Backends[0].GPUIDs != nil {
		t.Errorf("gpu_ids must be absent for one card, got %v", info.Backends[0].GPUIDs)
	}
}

func TestBackendInfoReportsGroups(t *testing.T) {
	r := NewRegistry("n1", "m", []*balancer.BackendState{healthyState([]int{4, 5})}, nil, time.Second)
	info := r.StatusResponse(nil, 0, 0)
	if info.Backends[0].GPUID != -1 {
		t.Errorf("a group has no single gpu id, got %d", info.Backends[0].GPUID)
	}
	if len(info.Backends[0].GPUIDs) != 2 {
		t.Errorf("gpu_ids: got %v", info.Backends[0].GPUIDs)
	}
}

// vLLM publishes cumulative token counters, not per-request progress. A
// cumulative total in a field the dashboard renders as "tokens left" would be
// worse than a blank.
func TestBackendInfoOmitsTokenFields(t *testing.T) {
	r := NewRegistry("n1", "m", []*balancer.BackendState{healthyState([]int{0})}, nil, time.Second)
	b, _ := json.Marshal(r.StatusResponse(nil, 0, 0))
	for _, absent := range []string{"tok_decoded", "tok_remain"} {
		if strings.Contains(string(b), absent) {
			t.Errorf("%s must be omitted: %s", absent, b)
		}
	}
}

func TestIsKnownAddrGuardsMeshPrompt(t *testing.T) {
	p := NewPeerState("gb2:9302")
	r := NewRegistry("n1", "m", nil, []*PeerState{p}, time.Second)
	if !r.IsKnownAddr("gb2:9302") {
		t.Error("a configured peer must be recognised")
	}
	// Without this check the mesh prompt endpoint fetches whatever it is
	// handed, on a LAN that also carries IPMI.
	if r.IsKnownAddr("169.254.169.254:80") {
		t.Error("an unconfigured address must be rejected")
	}
}

// A peer that returns garbage must be marked unreachable, not crash the poll.
func TestMalformedPeerResponseIsUnreachable(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("<html>not json</html>"))
	}))
	defer srv.Close()
	p := NewPeerState(strings.TrimPrefix(srv.URL, "http://"))
	r := NewRegistry("n1", "m", nil, []*PeerState{p}, time.Second)
	r.PollOnce(context.Background())
	if p.Status() != StatusUnreachable {
		t.Error("a peer returning garbage must be unreachable")
	}
}

// --- GPU-to-model mapping, power provenance and energy ---

type fakePower struct {
	watts     float64
	available bool
	source    string
}

func (f fakePower) Watts() float64  { return f.watts }
func (f fakePower) Available() bool { return f.available }
func (f fakePower) Source() string  { return f.source }

type fakeEnergy struct{ kwh24h, kwh30d float64 }

func (f fakeEnergy) KWh24h() float64 { return f.kwh24h }
func (f fakeEnergy) KWh30d() float64 { return f.kwh30d }

func TestGPUModelsMapsLocalBackends(t *testing.T) {
	r := NewRegistry("n1", "granite", []*balancer.BackendState{healthyState([]int{0})}, nil, time.Second)
	got := r.GPUModels()
	if got[0] != "granite" {
		t.Errorf("GPU 0 = %q, want %q", got[0], "granite")
	}
}

// A tensor-parallel group holds several cards, and every one of them is drawing
// power for that model.
func TestGPUModelsCoversEveryCardOfAGroup(t *testing.T) {
	r := NewRegistry("n1", "qwen", []*balancer.BackendState{healthyState([]int{4, 5, 6, 7})}, nil, time.Second)
	got := r.GPUModels()
	for _, id := range []int{4, 5, 6, 7} {
		if got[id] != "qwen" {
			t.Errorf("GPU %d = %q, want %q", id, got[id], "qwen")
		}
	}
}

// The recording instance owns one slice of the host but records every card, so
// it has to learn the rest of the layout from co-located peers or a co-tenant's
// load is charged to nothing.
func TestGPUModelsAdoptsColocatedPeerCards(t *testing.T) {
	p, _ := peerServing(t, meshapi.StatusResponse{
		NodeID:   "n2",
		Hostname: "plexie",
		Models:   []string{"qwen"},
		Backends: []meshapi.BackendInfo{{GPUID: 2, Model: "qwen"}},
	})
	r := NewRegistry("n1", "granite", []*balancer.BackendState{healthyState([]int{0})}, []*PeerState{p}, time.Second)
	r.SetLocation("plexie", "plexie:9103")
	r.PollOnce(context.Background())

	got := r.GPUModels()
	if got[0] != "granite" {
		t.Errorf("local GPU 0 = %q, want %q", got[0], "granite")
	}
	if got[2] != "qwen" {
		t.Errorf("co-located peer GPU 2 = %q, want %q", got[2], "qwen")
	}
}

// A peer on another machine shares GPU ids with this host but not cards.
// Adopting its labels would attribute this host's power to a remote model.
func TestGPUModelsIgnoresRemotePeers(t *testing.T) {
	p, _ := peerServing(t, meshapi.StatusResponse{
		NodeID:   "n2",
		Hostname: "gb1",
		Models:   []string{"translategemma"},
		Backends: []meshapi.BackendInfo{{GPUID: 0, Model: "translategemma"}},
	})
	r := NewRegistry("n1", "granite", []*balancer.BackendState{healthyState([]int{0})}, []*PeerState{p}, time.Second)
	r.SetLocation("plexie", "plexie:9103")
	r.PollOnce(context.Background())

	if got := r.GPUModels()[0]; got != "granite" {
		t.Errorf("GPU 0 = %q, want the local model — a remote peer must not claim this host's cards", got)
	}
}

func TestGPUModelsPrefersLocalOverColocatedPeer(t *testing.T) {
	p, _ := peerServing(t, meshapi.StatusResponse{
		NodeID:   "n2",
		Hostname: "plexie",
		Models:   []string{"qwen"},
		Backends: []meshapi.BackendInfo{{GPUID: 0, Model: "qwen"}},
	})
	r := NewRegistry("n1", "granite", []*balancer.BackendState{healthyState([]int{0})}, []*PeerState{p}, time.Second)
	r.SetLocation("plexie", "plexie:9103")
	r.PollOnce(context.Background())

	if got := r.GPUModels()[0]; got != "granite" {
		t.Errorf("GPU 0 = %q, want the local model to win", got)
	}
}

// A tensor-split peer reports GPUIDs rather than a single GPUID.
func TestGPUModelsReadsPeerGroups(t *testing.T) {
	p, _ := peerServing(t, meshapi.StatusResponse{
		NodeID:   "n2",
		Hostname: "plexie",
		Models:   []string{"qwen"},
		Backends: []meshapi.BackendInfo{{GPUID: -1, GPUIDs: []int{2, 3}, Model: "qwen"}},
	})
	r := NewRegistry("n1", "granite", nil, []*PeerState{p}, time.Second)
	r.SetLocation("plexie", "plexie:9103")
	r.PollOnce(context.Background())

	got := r.GPUModels()
	for _, id := range []int{2, 3} {
		if got[id] != "qwen" {
			t.Errorf("GPU %d = %q, want %q", id, got[id], "qwen")
		}
	}
	if _, ok := got[-1]; ok {
		t.Error("the -1 group sentinel must not become a GPU id")
	}
}

func TestStatusPublishesPowerSource(t *testing.T) {
	r := NewRegistry("n1", "m", nil, nil, time.Second)
	r.SetPowerReader(fakePower{watts: 220.75, available: true, source: "nvidia-smi"})

	resp := r.StatusResponse(nil, 0, 0)
	if resp.PowerWatts != 220.75 || !resp.PowerAvailable {
		t.Errorf("watts=%v available=%v, want 220.75/true", resp.PowerWatts, resp.PowerAvailable)
	}
	if resp.PowerSource != "nvidia-smi" {
		t.Errorf("PowerSource = %q, want %q — a consumer cannot otherwise tell a GPU sum from a chassis reading", resp.PowerSource, "nvidia-smi")
	}
}

// Absent is not zero: an unavailable reading must omit the figure, not publish
// a confident nought that renders as a measurement.
func TestStatusOmitsUnavailablePower(t *testing.T) {
	r := NewRegistry("n1", "m", nil, nil, time.Second)
	r.SetPowerReader(fakePower{watts: 0, available: false, source: "nvidia-smi"})

	resp := r.StatusResponse(nil, 0, 0)
	if resp.PowerAvailable {
		t.Error("PowerAvailable = true, want false")
	}
	if resp.PowerWatts != 0 {
		t.Errorf("PowerWatts = %v, want 0 so omitempty drops it", resp.PowerWatts)
	}
}

func TestStatusPublishesEnergy(t *testing.T) {
	r := NewRegistry("n1", "m", nil, nil, time.Second)
	r.SetEnergyReader(fakeEnergy{kwh24h: 1.25, kwh30d: 30.5})

	resp := r.StatusResponse(nil, 0, 0)
	if resp.EnergyKWh24h != 1.25 || resp.EnergyKWh30d != 30.5 {
		t.Errorf("24h=%v 30d=%v, want 1.25/30.5", resp.EnergyKWh24h, resp.EnergyKWh30d)
	}
}

// The energy store runs on one instance per host, so the others must publish
// nothing rather than a zero that would read as "this host drew no power".
func TestNoEnergyReaderOmitsEnergy(t *testing.T) {
	r := NewRegistry("n1", "m", nil, nil, time.Second)
	resp := r.StatusResponse(nil, 0, 0)
	if resp.EnergyKWh24h != 0 || resp.EnergyKWh30d != 0 {
		t.Errorf("24h=%v 30d=%v, want both zero so omitempty drops them", resp.EnergyKWh24h, resp.EnergyKWh30d)
	}
}

func TestClusterStatePublishesPowerSourceAndEnergy(t *testing.T) {
	r := NewRegistry("n1", "m", nil, nil, time.Second)
	r.SetPowerReader(fakePower{watts: 100, available: true, source: "nvidia-smi"})
	r.SetEnergyReader(fakeEnergy{kwh24h: 2, kwh30d: 60})

	local := r.ClusterState().Local
	if local.PowerSource != "nvidia-smi" {
		t.Errorf("PowerSource = %q, want %q", local.PowerSource, "nvidia-smi")
	}
	if local.EnergyKWh24h != 2 || local.EnergyKWh30d != 60 {
		t.Errorf("24h=%v 30d=%v, want 2/60", local.EnergyKWh24h, local.EnergyKWh30d)
	}
}
