package proxy

import (
	"context"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/janit/viiwork/meshapi"
)

// readSSE drives a stream handler until the context expires and returns what
// it wrote.
func readSSE(t *testing.T, hn *harness, path string, d time.Duration) string {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), d)
	defer cancel()
	r := httptest.NewRequest("GET", path, nil).WithContext(ctx)
	w := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		hn.h.ServeHTTP(w, r)
		close(done)
	}()
	<-done
	return w.Body.String()
}

// A dashboard reconnecting must be able to rebuild in-flight state from the
// replay, so the ring is resent and every replayed event is marked.
func TestActivityStreamReplaysBacklogMarked(t *testing.T) {
	hn := newHarness(t, "m", nil, nil)
	hn.events.EmitRequest(1, 0, "%s", meshapi.RequestStarted("m", "gpu-0"))
	body := readSSE(t, hn, meshapi.PathActivityStream, 250*time.Millisecond)
	if !strings.Contains(body, "gpu-0") {
		t.Fatalf("backlog not replayed: %s", body)
	}
	if !strings.Contains(body, `"replay":true`) {
		t.Errorf("replayed events must be marked, or a consumer cannot deduplicate them: %s", body)
	}
}

// The mesh stream carries two NAMED events on one connection; an unnamed
// listener would see neither.
func TestMeshStreamUsesNamedEvents(t *testing.T) {
	hn := newHarness(t, "m", nil, nil)
	hn.events.EmitRequest(1, 0, "%s", meshapi.RequestStarted("m", "gpu-0"))
	body := readSSE(t, hn, meshapi.PathMeshStream, 400*time.Millisecond)
	if !strings.Contains(body, "event: "+meshapi.SSEActivity) {
		t.Errorf("no named activity event: %s", body)
	}
	if !strings.Contains(body, "event: "+meshapi.SSECluster) {
		t.Errorf("no named cluster event: %s", body)
	}
}

// Events on the mesh stream are tagged with the node they came from, which is
// how the dashboard attributes a row to a host.
func TestMeshStreamTagsEventsWithNode(t *testing.T) {
	hn := newHarness(t, "m", nil, nil)
	hn.events.EmitRequest(1, 0, "%s", meshapi.RequestStarted("m", "gpu-0"))
	body := readSSE(t, hn, meshapi.PathMeshStream, 400*time.Millisecond)
	if !strings.Contains(body, `"hostname":"testhost"`) {
		t.Errorf("events not tagged with the host: %s", body)
	}
}

// Snapshots are diffed server-side and sent only when they change, which is
// what lets the browser never poll.
func TestClusterSnapshotNotResentUnchanged(t *testing.T) {
	hn := newHarness(t, "m", nil, nil)
	// Long enough for several snapshot ticks with nothing changing.
	body := readSSE(t, hn, meshapi.PathMeshStream, 3*snapshotInterval+300*time.Millisecond)
	if n := strings.Count(body, "event: "+meshapi.SSECluster); n != 1 {
		t.Errorf("cluster snapshot sent %d times with nothing changing; the browser would be polled at %s", n, snapshotInterval)
	}
}

// An exact host-memory figure moves every second and would defeat change
// detection outright, pushing a full snapshot every tick.
func TestHostMemCoarsening(t *testing.T) {
	const total = 64000
	step := int64(total / hostMemBuckets)

	// From nothing, the reading is bucketed.
	first := coarsenHostMem(0, total, 48123)
	if first%step != 0 {
		t.Errorf("first reading not bucketed: %d", first)
	}
	// A small drift holds the published value, so nothing is pushed.
	if got := coarsenHostMem(first, total, 48123+step/4); got != first {
		t.Errorf("small drift moved the value: %d -> %d", first, got)
	}
	// A full step moves it.
	if got := coarsenHostMem(first, total, first+step+1); got == first {
		t.Error("a full step of drift should move the published value")
	}
	// A host sitting astride a boundary must not flip back and forth; that is
	// what the deadband exists for, since plain rounding pushes just as hard.
	astride := first + step
	held := coarsenHostMem(astride, total, astride-1)
	if held != astride {
		t.Errorf("value flipped across a bucket boundary: %d -> %d", astride, held)
	}
	// A host that reports no total at all passes through unchanged.
	if got := coarsenHostMem(0, 0, 1234); got != 1234 {
		t.Errorf("got %d, want the raw value when total is unknown", got)
	}
}

// /v1/cluster keeps the exact figures; only the pushed snapshot is coarsened.
func TestClusterEndpointKeepsExactMemory(t *testing.T) {
	hn := newHarness(t, "m", nil, nil)
	state := hn.h.clusterState()
	total, used := readHostMemory()
	if state.Local.HostMemUsedMB != used || state.Local.HostMemTotalMB != total {
		t.Errorf("cluster endpoint should carry exact memory: got %d/%d, want %d/%d",
			state.Local.HostMemUsedMB, state.Local.HostMemTotalMB, used, total)
	}
}

func TestSSEReaderExtractsDataFrames(t *testing.T) {
	stream := ": keepalive\n\nevent: activity\ndata: {\"type\":\"request\"}\n\ndata: {\"type\":\"system\"}\n\n"
	r := newSSEReader(strings.NewReader(stream))
	first, err := r.next()
	if err != nil || string(first) != `{"type":"request"}` {
		t.Fatalf("got %q, %v", first, err)
	}
	second, err := r.next()
	if err != nil || string(second) != `{"type":"system"}` {
		t.Fatalf("got %q, %v", second, err)
	}
	if _, err := r.next(); err == nil {
		t.Error("want an error at end of stream")
	}
}
