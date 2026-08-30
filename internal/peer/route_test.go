package peer

import (
	"testing"
	"time"

	"github.com/janit/viiwork-freetoken/internal/balancer"
)

func localRoute(inFlight int64) Route {
	b := balancer.NewBackendState([]int{0}, "a", time.Second)
	b.SetStatus(balancer.StatusHealthy)
	return Route{Type: RouteLocal, Backend: b, InFlight: inFlight}
}

func peerRoute(addr string, inFlight int64) Route {
	return Route{Type: RoutePeer, Addr: addr, Peer: NewPeerState(addr), InFlight: inFlight}
}

func TestPickRouteEmpty(t *testing.T) {
	if _, err := PickRoute(nil, 4); err != ErrNoRoute {
		t.Errorf("got %v, want ErrNoRoute", err)
	}
}

// A local backend costs no network hop, so it wins a tie.
func TestPickPrefersLocalOnTie(t *testing.T) {
	routes := []Route{peerRoute("gb2:9302", 0), localRoute(0)}
	got, err := PickRoute(routes, 4)
	if err != nil {
		t.Fatal(err)
	}
	if got.Type != RouteLocal {
		t.Errorf("got %s, want local on a tie", got.Type)
	}
}

// Load steering must beat locality: a busy local backend should yield to an
// idle peer.
func TestPickPrefersLighterPeerOverBusyLocal(t *testing.T) {
	routes := []Route{localRoute(5), peerRoute("gb2:9302", 0)}
	got, err := PickRoute(routes, 16)
	if err != nil {
		t.Fatal(err)
	}
	if got.Type != RoutePeer {
		t.Error("a much lighter peer should win over a loaded local backend")
	}
}

// Local capacity is a hard ceiling; past it the request goes to a peer rather
// than queueing here.
func TestLocalCeilingDivertsToPeer(t *testing.T) {
	routes := []Route{localRoute(4), peerRoute("gb2:9302", 10)}
	got, err := PickRoute(routes, 4)
	if err != nil {
		t.Fatal(err)
	}
	if got.Type != RoutePeer {
		t.Error("a local backend at its ceiling must be skipped even for a busier peer")
	}
}

// Everything local at capacity and no peer is backpressure, not "no route" —
// the node is working, the client should slow down.
func TestAllLocalAtCapacityIsBackpressure(t *testing.T) {
	routes := []Route{localRoute(4), localRoute(4)}
	_, err := PickRoute(routes, 4)
	if err != balancer.ErrBackpressure {
		t.Errorf("got %v, want ErrBackpressure", err)
	}
}

// A deterministic tiebreak pins every racing request to one backend while the
// others idle, because in-flight is a snapshot taken before the increment.
func TestTiedLocalsRoundRobin(t *testing.T) {
	seen := map[*balancer.BackendState]int{}
	routes := []Route{localRoute(0), localRoute(0), localRoute(0)}
	for i := 0; i < 30; i++ {
		got, err := PickRoute(routes, 8)
		if err != nil {
			t.Fatal(err)
		}
		seen[got.Backend]++
	}
	if len(seen) != 3 {
		t.Errorf("tied backends should all be used, got %d of 3", len(seen))
	}
}

func TestTiedPeersRoundRobin(t *testing.T) {
	seen := map[string]int{}
	routes := []Route{peerRoute("a:1", 0), peerRoute("b:1", 0)}
	for i := 0; i < 20; i++ {
		got, err := PickRoute(routes, 8)
		if err != nil {
			t.Fatal(err)
		}
		seen[got.Addr]++
	}
	if len(seen) != 2 {
		t.Errorf("tied peers should both be used, got %v", seen)
	}
}

// A ceiling of zero means unlimited, not "reject everything".
func TestZeroCeilingMeansUnlimited(t *testing.T) {
	routes := []Route{localRoute(100)}
	if _, err := PickRoute(routes, 0); err != nil {
		t.Errorf("got %v, want a route", err)
	}
}
