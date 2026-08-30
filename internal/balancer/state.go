// Package balancer tracks per-backend load and picks where a request should
// go.
//
// The routing signal differs from viiwork's in one way that matters. llama.cpp
// exposes a fixed set of slots and viiwork reasons about their occupancy;
// FreeToken instead runs a continuous-batching scheduler that accepts requests
// beyond the --max-running-requests it decodes concurrently and holds the rest.
// In-flight count is therefore the *fresh* signal used for picking, while the
// scheduler's own view (running count, cache pressure) arrives on the health
// tick and is published to the mesh.
//
// One consequence is specific to this engine: --max-running-requests defaults
// to 4, so the gap between "accepted" and "decoding" opens at ordinary
// concurrency rather than only under load. Routing on a health-tick-old
// running count would misroute constantly.
package balancer

import (
	"math"
	"sync"
	"sync/atomic"
	"time"

	"github.com/janit/viiwork/meshapi"
)

type BackendStatus int32

const (
	StatusStarting BackendStatus = iota
	StatusHealthy
	StatusUnhealthy
	StatusDead
)

// String returns the wire vocabulary from meshapi, not a local spelling. The
// dashboard decides what is routable from this word, so it has to be one the
// protocol names.
func (s BackendStatus) String() string {
	switch s {
	case StatusStarting:
		return meshapi.StatusStarting
	case StatusHealthy:
		return meshapi.StatusHealthy
	case StatusUnhealthy:
		return meshapi.StatusUnhealthy
	case StatusDead:
		return meshapi.StatusDead
	default:
		return "unknown"
	}
}

// BackendState is the routable view of one FreeToken process. The process
// itself is owned by the freetoken package; this is what the picker and the
// mesh payload see.
type BackendState struct {
	// GPUIDs are the physical cards this backend spans. Always exactly one on
	// this node — FreeToken serves one GPU per process — but kept as a slice
	// because the mesh wire format carries a backend's cards as a set and the
	// other node types do put several behind one backend.
	GPUIDs []int
	Addr   string

	status   atomic.Int32
	inFlight atomic.Int64

	// Scheduler figures, refreshed from FreeToken's /v1/stats on the health
	// tick. Stale by up to one interval, so they inform the mesh view but never
	// a routing decision on their own.
	numRunning  atomic.Int64
	numWaiting  atomic.Int64
	kvCachePct  atomic.Uint64 // float64 bits
	rssMB       atomic.Int64
	maxModelLen atomic.Int64
	maxNumSeqs  atomic.Int64

	// hardFailure latches a kernel-level signal that the process is gone (EOF
	// or connection refused on the inference path), so the manager can skip
	// the failure ladder and respawn after one failed probe.
	hardFailure atomic.Bool

	mu      sync.Mutex
	samples []latencySample
	window  time.Duration
}

type latencySample struct {
	at  time.Time
	dur time.Duration
}

func NewBackendState(gpuIDs []int, addr string, window time.Duration) *BackendState {
	b := &BackendState{GPUIDs: append([]int(nil), gpuIDs...), Addr: addr, window: window}
	b.status.Store(int32(StatusStarting))
	return b
}

// GPUID is the single-card id, or -1 for a tensor-parallel group. This is the
// shape meshapi.BackendInfo expects: a group has no single id, and reporting
// one of its members would make the group indistinguishable from a plain
// backend on that card.
func (b *BackendState) GPUID() int {
	if len(b.GPUIDs) == 1 {
		return b.GPUIDs[0]
	}
	return -1
}

// Label names this backend the way the whole mesh does: "gpu-3" for one card,
// "ts-4,5" for a group. It is the X-GPU-Backend response header and the
// destination in every activity message, so it comes from meshapi rather than
// being spelled out again here.
func (b *BackendState) Label() string {
	if len(b.GPUIDs) == 1 {
		return meshapi.BackendLabel(b.GPUIDs[0], nil)
	}
	return meshapi.BackendLabel(-1, b.GPUIDs)
}

func (b *BackendState) Status() BackendStatus     { return BackendStatus(b.status.Load()) }
func (b *BackendState) SetStatus(s BackendStatus) { b.status.Store(int32(s)) }
func (b *BackendState) InFlight() int64           { return b.inFlight.Load() }
func (b *BackendState) IncrInFlight()             { b.inFlight.Add(1) }
func (b *BackendState) DecrInFlight()             { b.inFlight.Add(-1) }

func (b *BackendState) SetSchedulerStats(running, waiting int64, kvPct float64) {
	b.numRunning.Store(running)
	b.numWaiting.Store(waiting)
	b.kvCachePct.Store(math.Float64bits(kvPct))
}

func (b *BackendState) NumRunning() int64 { return b.numRunning.Load() }
func (b *BackendState) NumWaiting() int64 { return b.numWaiting.Load() }
func (b *BackendState) KVCachePct() float64 {
	return math.Float64frombits(b.kvCachePct.Load())
}

func (b *BackendState) SetRSSMB(v int64)       { b.rssMB.Store(v) }
func (b *BackendState) RSSMB() int64           { return b.rssMB.Load() }
func (b *BackendState) SetMaxModelLen(v int64) { b.maxModelLen.Store(v) }
func (b *BackendState) MaxModelLen() int64     { return b.maxModelLen.Load() }
func (b *BackendState) SetMaxNumSeqs(v int64)  { b.maxNumSeqs.Store(v) }
func (b *BackendState) MaxNumSeqs() int64      { return b.maxNumSeqs.Load() }

// NoteHardFailure marks the backend unhealthy and latches a flag the manager's
// health tick reads. Called from the proxy when a request sees EOF or
// "connection refused" — kernel-level signals that the process is definitively
// gone. The status flip removes it from the picker immediately; the latch
// tells the manager to skip the failure ladder and respawn after one probe.
func (b *BackendState) NoteHardFailure() {
	b.hardFailure.Store(true)
	b.SetStatus(StatusUnhealthy)
}

// TakeHardFailure reads and clears the latch.
func (b *BackendState) TakeHardFailure() bool { return b.hardFailure.Swap(false) }

// RecordLatency adds a completed request's wall time to the rolling window.
func (b *BackendState) RecordLatency(d time.Duration) {
	now := time.Now()
	b.mu.Lock()
	defer b.mu.Unlock()
	b.samples = append(b.samples, latencySample{at: now, dur: d})
	b.evictLocked(now)
}

// AvgLatency is the mean over the configured window, or zero when nothing has
// completed in it. Zero reads as "no evidence", and the picker treats an
// unmeasured backend as a good candidate rather than a bad one — a backend
// that has served nothing recently is usually idle, not broken.
func (b *BackendState) AvgLatency() time.Duration {
	now := time.Now()
	b.mu.Lock()
	defer b.mu.Unlock()
	b.evictLocked(now)
	if len(b.samples) == 0 {
		return 0
	}
	var total time.Duration
	for _, s := range b.samples {
		total += s.dur
	}
	return total / time.Duration(len(b.samples))
}

func (b *BackendState) evictLocked(now time.Time) {
	if b.window <= 0 {
		return
	}
	cutoff := now.Add(-b.window)
	i := 0
	for i < len(b.samples) && b.samples[i].at.Before(cutoff) {
		i++
	}
	if i > 0 {
		b.samples = append(b.samples[:0], b.samples[i:]...)
	}
}
