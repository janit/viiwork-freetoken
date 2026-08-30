package balancer

import (
	"errors"
	"time"
)

// Balancer picks a local backend for a request.
//
// Two modes, switched automatically on total node load:
//
//   - Under HighLoadThreshold, prefer the **lowest-latency idle** backend.
//     While there is spare capacity, the question is which card will answer
//     fastest, and an idle one always beats a loaded one.
//   - At or above it, prefer the **least-loaded** backend, tie-broken by
//     latency. Once everything is busy, spreading work evenly matters more
//     than picking a historically quick card that is now queued up.
//
// Both modes read in-flight count rather than the engine's own scheduler
// figures. Those arrive on the health tick and are therefore a tick behind, and
// a stale queue depth used for admission sends a burst of requests to the same
// backend before the next poll can reveal what happened.
type Balancer struct {
	backends          []*BackendState
	highLoadThreshold int
	maxInFlight       int64
}

func New(backends []*BackendState, highLoadThreshold, maxInFlightPerBackend int) *Balancer {
	return &Balancer{
		backends:          backends,
		highLoadThreshold: highLoadThreshold,
		maxInFlight:       int64(maxInFlightPerBackend),
	}
}

func (b *Balancer) Backends() []*BackendState { return b.backends }

// TotalInFlight is the node's current load across every backend.
func (b *Balancer) TotalInFlight() int64 {
	var n int64
	for _, s := range b.backends {
		n += s.InFlight()
	}
	return n
}

func (b *Balancer) HealthyCount() int {
	n := 0
	for _, s := range b.backends {
		if s.Status() == StatusHealthy {
			n++
		}
	}
	return n
}

// Pick returns the backend to serve the next request.
//
// The three return cases are distinct and callers must treat them so:
//
//   - a backend: serve it.
//   - (nil, false): no healthy backend exists — a 503 with Retry-After,
//     because the condition is expected to clear when a respawn completes.
//   - (nil, true): every healthy backend is at its in-flight ceiling — a 429,
//     because the node is working and the client should slow down.
//
// Collapsing the last two into one status loses the distinction between "this
// node is broken" and "this node is busy", which is exactly what a client
// retrying against a mesh needs to tell apart.
func (b *Balancer) Pick() (backend *BackendState, saturated bool) {
	var healthy []*BackendState
	for _, s := range b.backends {
		if s.Status() == StatusHealthy {
			healthy = append(healthy, s)
		}
	}
	if len(healthy) == 0 {
		return nil, false
	}

	var eligible []*BackendState
	for _, s := range healthy {
		if b.maxInFlight <= 0 || s.InFlight() < b.maxInFlight {
			eligible = append(eligible, s)
		}
	}
	if len(eligible) == 0 {
		return nil, true
	}

	if b.TotalInFlight() < int64(b.highLoadThreshold) {
		if s := pickIdleLowestLatency(eligible); s != nil {
			return s, false
		}
	}
	return pickLeastLoaded(eligible), false
}

// pickIdleLowestLatency returns the fastest backend with nothing in flight, or
// nil if none is idle — in which case the caller falls through to least-loaded
// rather than forcing work onto a busy backend that merely has good history.
func pickIdleLowestLatency(candidates []*BackendState) *BackendState {
	var best *BackendState
	var bestLat time.Duration
	for _, s := range candidates {
		if s.InFlight() != 0 {
			continue
		}
		lat := s.AvgLatency()
		// A backend with no completed requests in the window reports zero,
		// which sorts first. That is intended: an unmeasured backend is
		// usually one that has been idle, and giving it a request is both the
		// fastest answer and how it acquires a measurement.
		if best == nil || lat < bestLat {
			best, bestLat = s, lat
		}
	}
	return best
}

func pickLeastLoaded(candidates []*BackendState) *BackendState {
	best := candidates[0]
	bestLoad := best.InFlight()
	bestLat := best.AvgLatency()
	for _, s := range candidates[1:] {
		load := s.InFlight()
		if load < bestLoad {
			best, bestLoad, bestLat = s, load, s.AvgLatency()
			continue
		}
		if load == bestLoad {
			if lat := s.AvgLatency(); lat < bestLat {
				best, bestLat = s, lat
			}
		}
	}
	return best
}

// ErrBackpressure means every candidate is at its in-flight ceiling. It is
// distinct from "no route" because the two produce different HTTP statuses:
// this one is a 429 against a working node, not a 503 against a broken one.
var ErrBackpressure = errors.New("all backends at capacity")
