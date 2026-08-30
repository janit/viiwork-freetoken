package balancer

import (
	"testing"
	"time"
)

func newState(t *testing.T, gpuIDs []int, status BackendStatus) *BackendState {
	t.Helper()
	s := NewBackendState(gpuIDs, "127.0.0.1:1", 30*time.Second)
	s.SetStatus(status)
	return s
}

func TestLabelSingleAndTensorParallel(t *testing.T) {
	if got := newState(t, []int{3}, StatusHealthy).Label(); got != "gpu-3" {
		t.Errorf("single card: got %q, want gpu-3", got)
	}
	// A tensor-parallel group has no single GPU id, and labelling it by one of
	// its members would make it indistinguishable from a plain backend there.
	if got := newState(t, []int{4, 5}, StatusHealthy).Label(); got != "ts-4,5" {
		t.Errorf("group: got %q, want ts-4,5", got)
	}
}

func TestGPUIDIsMinusOneForGroups(t *testing.T) {
	if got := newState(t, []int{2}, StatusHealthy).GPUID(); got != 2 {
		t.Errorf("got %d, want 2", got)
	}
	if got := newState(t, []int{2, 3}, StatusHealthy).GPUID(); got != -1 {
		t.Errorf("got %d, want -1 for a group", got)
	}
}

// Status words are the protocol's, not a local spelling: the dashboard decides
// what is routable from this string.
func TestStatusStringsAreProtocolVocabulary(t *testing.T) {
	for s, want := range map[BackendStatus]string{
		StatusStarting: "starting", StatusHealthy: "healthy",
		StatusUnhealthy: "unhealthy", StatusDead: "dead",
	} {
		if got := s.String(); got != want {
			t.Errorf("got %q, want %q", got, want)
		}
	}
}

func TestPickSkipsUnhealthy(t *testing.T) {
	bad := newState(t, []int{0}, StatusDead)
	good := newState(t, []int{1}, StatusHealthy)
	b := New([]*BackendState{bad, good}, 7, 64)
	got, _ := b.Pick()
	if got != good {
		t.Fatalf("picked a non-healthy backend: %v", got)
	}
}

// No healthy backend and saturation are different conditions: one is a 503
// that will clear on respawn, the other a 429 telling a working node's client
// to slow down.
func TestPickDistinguishesNoBackendFromSaturated(t *testing.T) {
	dead := newState(t, []int{0}, StatusDead)
	if got, sat := New([]*BackendState{dead}, 7, 64).Pick(); got != nil || sat {
		t.Errorf("no healthy backend must report (nil, false), got (%v, %v)", got, sat)
	}

	busy := newState(t, []int{0}, StatusHealthy)
	for i := 0; i < 4; i++ {
		busy.IncrInFlight()
	}
	if got, sat := New([]*BackendState{busy}, 7, 4).Pick(); got != nil || !sat {
		t.Errorf("saturated must report (nil, true), got (%v, %v)", got, sat)
	}
}

// Under the threshold the fastest idle backend wins, even though another
// backend has a better average.
func TestPickPrefersIdleUnderThreshold(t *testing.T) {
	fastBusy := newState(t, []int{0}, StatusHealthy)
	fastBusy.RecordLatency(10 * time.Millisecond)
	fastBusy.IncrInFlight()

	slowIdle := newState(t, []int{1}, StatusHealthy)
	slowIdle.RecordLatency(500 * time.Millisecond)

	b := New([]*BackendState{fastBusy, slowIdle}, 7, 64)
	got, _ := b.Pick()
	if got != slowIdle {
		t.Error("under the threshold an idle backend must beat a faster busy one")
	}
}

// Once everything is busy, spreading load beats historical speed.
func TestPickLeastLoadedOverThreshold(t *testing.T) {
	loaded := newState(t, []int{0}, StatusHealthy)
	loaded.RecordLatency(10 * time.Millisecond)
	for i := 0; i < 8; i++ {
		loaded.IncrInFlight()
	}
	lighter := newState(t, []int{1}, StatusHealthy)
	lighter.RecordLatency(900 * time.Millisecond)
	lighter.IncrInFlight()

	b := New([]*BackendState{loaded, lighter}, 7, 64)
	if b.TotalInFlight() < 7 {
		t.Fatalf("test setup: total in flight is %d", b.TotalInFlight())
	}
	got, _ := b.Pick()
	if got != lighter {
		t.Error("over the threshold the least-loaded backend must win")
	}
}

func TestPickTieBreaksOnLatency(t *testing.T) {
	slow := newState(t, []int{0}, StatusHealthy)
	slow.RecordLatency(800 * time.Millisecond)
	fast := newState(t, []int{1}, StatusHealthy)
	fast.RecordLatency(20 * time.Millisecond)
	for _, s := range []*BackendState{slow, fast} {
		for i := 0; i < 9; i++ {
			s.IncrInFlight()
		}
	}
	b := New([]*BackendState{slow, fast}, 7, 64)
	got, _ := b.Pick()
	if got != fast {
		t.Error("equal load must tie-break on latency")
	}
}

func TestLatencyWindowEvicts(t *testing.T) {
	s := NewBackendState([]int{0}, "a", 50*time.Millisecond)
	s.RecordLatency(time.Second)
	if s.AvgLatency() == 0 {
		t.Fatal("sample should be in the window")
	}
	time.Sleep(80 * time.Millisecond)
	if got := s.AvgLatency(); got != 0 {
		t.Errorf("stale sample not evicted: %v", got)
	}
}

func TestHardFailureLatchClearsOnRead(t *testing.T) {
	s := newState(t, []int{0}, StatusHealthy)
	s.NoteHardFailure()
	if s.Status() != StatusUnhealthy {
		t.Error("a hard failure must remove the backend from the picker immediately")
	}
	if !s.TakeHardFailure() {
		t.Error("latch should read true once")
	}
	if s.TakeHardFailure() {
		t.Error("latch must clear on read, or every later tick skips the ladder")
	}
}

func TestSchedulerStatsRoundTrip(t *testing.T) {
	s := newState(t, []int{0}, StatusHealthy)
	s.SetSchedulerStats(12, 3, 0.734)
	if s.NumRunning() != 12 || s.NumWaiting() != 3 {
		t.Errorf("got running=%d waiting=%d", s.NumRunning(), s.NumWaiting())
	}
	if got := s.KVCachePct(); got != 0.734 {
		t.Errorf("kv cache: got %v", got)
	}
}
