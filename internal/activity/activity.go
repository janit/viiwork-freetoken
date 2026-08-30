// Package activity is the node's in-memory event log and prompt store, and
// the SSE fan-out that feeds them to dashboards.
//
// Nothing here is persisted and nothing survives a restart. That is a design
// choice, not a gap: the mesh has no server-side registry of running jobs
// anywhere, and every dashboard reconstructs in-flight state from this stream.
// See meshapi's Event for the wire contract that makes that work.
package activity

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/janit/viiwork/meshapi"
)

// Event is meshapi's, aliased for brevity at the call sites in this package.
type Event = meshapi.Event

var nextRequestID atomic.Int64

// NewRequestID mints an id for one request.
//
// It is a per-process counter, deliberately not a UUID and deliberately not
// cluster-wide: it is short enough to read in a dashboard row, and the mesh
// already knows which node minted it from the event's node tag. The
// consequence is that a prompt lookup only means anything against that node.
func NewRequestID() int64 { return nextRequestID.Add(1) }

// Log is a bounded ring of recent events with live SSE subscribers.
type Log struct {
	mu   sync.RWMutex
	ring []Event
	max  int
	subs map[chan Event]struct{}
}

func NewLog(max int) *Log {
	if max < 1 {
		max = 2000
	}
	return &Log{max: max, subs: make(map[chan Event]struct{})}
}

// Subscribe returns a channel of live events.
//
// Callers must subscribe BEFORE reading Backlog, or an event landing between
// the two calls is delivered by neither — a request whose start is lost that
// way leaves a row that never clears.
func (l *Log) Subscribe() chan Event {
	// Buffered generously: a dashboard that stalls briefly should catch up
	// rather than lose the events that rebuild its state.
	ch := make(chan Event, 256)
	l.mu.Lock()
	l.subs[ch] = struct{}{}
	l.mu.Unlock()
	return ch
}

func (l *Log) Unsubscribe(ch chan Event) {
	l.mu.Lock()
	if _, ok := l.subs[ch]; ok {
		delete(l.subs, ch)
		close(ch)
	}
	l.mu.Unlock()
}

// Backlog returns the retained ring with every event marked as a replay.
//
// The marking matters: a consumer rebuilding keyed state treats these as
// authoritative, while one keeping a visible list has to deduplicate them
// against what it already shows. Without the replay, every event lost in a
// connection gap strands an in-flight row that ages forever — which is exactly
// what a slept laptop or a throttled background tab produces.
func (l *Log) Backlog() []Event {
	l.mu.RLock()
	defer l.mu.RUnlock()
	out := make([]Event, len(l.ring))
	for i, e := range l.ring {
		e.Replay = true
		out[i] = e
	}
	return out
}

// Recent returns the retained ring as history, without replay marking. This
// backs a plain read of the log, where the events are being shown rather than
// used to rebuild state.
func (l *Log) Recent() []Event {
	l.mu.RLock()
	defer l.mu.RUnlock()
	out := make([]Event, len(l.ring))
	copy(out, l.ring)
	return out
}

func (l *Log) SubscriberCount() int {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return len(l.subs)
}

func (l *Log) emit(e Event) {
	l.mu.Lock()
	l.ring = append(l.ring, e)
	if len(l.ring) > l.max {
		l.ring = l.ring[len(l.ring)-l.max:]
	}
	subs := make([]chan Event, 0, len(l.subs))
	for ch := range l.subs {
		subs = append(subs, ch)
	}
	l.mu.Unlock()

	// Sent outside the lock, and dropped rather than blocked on. A wedged
	// subscriber must not stall the request path that emitted this.
	for _, ch := range subs {
		select {
		case ch <- e:
		default:
		}
	}
}

func (l *Log) Emit(typ string, gpuID int, format string, args ...any) {
	l.emit(Event{
		Time:    time.Now().Unix(),
		Type:    typ,
		Message: fmt.Sprintf(format, args...),
		GPUID:   gpuID,
	})
}

func (l *Log) EmitRequest(rid int64, gpuID int, format string, args ...any) {
	l.EmitRequestTask(rid, gpuID, "", format, args...)
}

func (l *Log) EmitRequestTask(rid int64, gpuID int, taskID string, format string, args ...any) {
	l.emit(Event{
		Time:      time.Now().Unix(),
		Type:      meshapi.EventRequest,
		Message:   fmt.Sprintf(format, args...),
		GPUID:     gpuID,
		RequestID: rid,
		TaskID:    taskID,
	})
}
