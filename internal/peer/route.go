package peer

import (
	"errors"
	"sync/atomic"

	"github.com/janit/viiwork-freetoken/internal/balancer"
)

const (
	RouteLocal = "local"
	RoutePeer  = "peer"
)

var ErrNoRoute = errors.New("no route available for model")

type Route struct {
	Type     string
	Backend  *balancer.BackendState // non-nil for local
	Addr     string                 // peer address for remote
	Peer     *PeerState             // non-nil for remote; lets the caller record write-through in-flight
	InFlight int64
}

// peerRRIdx round-robins among peers tied at equal in-flight. Without it the
// picker always returns whichever tied peer comes first in the slice, which
// deterministically pins burst traffic to one peer until its polled count
// finally moves.
var peerRRIdx atomic.Uint64

// localRRIdx round-robins among tied LOCAL backends, for the same reason.
// Route.InFlight is a snapshot taken when routes are built while the increment
// happens later in the proxy, so phase-locked concurrent callers all read the
// same all-zero snapshot. Any deterministic tiebreak then lands every racing
// request on the same backend while the others idle; atomic rotation fans them
// out instead.
var localRRIdx atomic.Uint64

// PickRoute selects among the routes that can serve a model.
//
// Kept deliberately identical to viiwork's, because both implementations route
// into the same mesh: a node that resolved ties differently would produce a
// fleet whose load distribution depended on which node a client happened to
// hit.
//
//   - Drop local routes at their in-flight ceiling.
//   - Among routes tied at the lowest in-flight, prefer local over peer —
//     a local backend costs no extra network hop.
//   - Round-robin within each tied class.
func PickRoute(routes []Route, maxInFlightPerBackend int) (*Route, error) {
	if len(routes) == 0 {
		return nil, ErrNoRoute
	}
	maxLocal := int64(maxInFlightPerBackend)

	var available []*Route
	var minIF int64
	for i := range routes {
		r := &routes[i]
		if r.Type == RouteLocal && maxLocal > 0 && r.InFlight >= maxLocal {
			continue
		}
		if len(available) == 0 || r.InFlight < minIF {
			minIF = r.InFlight
		}
		available = append(available, r)
	}
	if len(available) == 0 {
		return nil, balancer.ErrBackpressure
	}

	var tiedLocals, tiedPeers []*Route
	for _, r := range available {
		if r.InFlight != minIF {
			continue
		}
		if r.Type == RouteLocal {
			tiedLocals = append(tiedLocals, r)
		} else {
			tiedPeers = append(tiedPeers, r)
		}
	}

	if len(tiedLocals) == 1 {
		return tiedLocals[0], nil
	}
	if len(tiedLocals) > 1 {
		idx := localRRIdx.Add(1) - 1
		return tiedLocals[int(idx%uint64(len(tiedLocals)))], nil
	}
	if len(tiedPeers) == 1 {
		return tiedPeers[0], nil
	}
	idx := peerRRIdx.Add(1) - 1
	return tiedPeers[int(idx%uint64(len(tiedPeers)))], nil
}
