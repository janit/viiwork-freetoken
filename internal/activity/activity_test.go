package activity

import (
	"strings"
	"testing"
	"time"

	"github.com/janit/viiwork/meshapi"
)

func TestRingIsBounded(t *testing.T) {
	l := NewLog(3)
	for i := 0; i < 10; i++ {
		l.Emit(meshapi.EventSystem, 0, "e%d", i)
	}
	got := l.Recent()
	if len(got) != 3 {
		t.Fatalf("ring grew to %d", len(got))
	}
	if got[0].Message != "e7" || got[2].Message != "e9" {
		t.Errorf("wrong window retained: %v", []string{got[0].Message, got[2].Message})
	}
}

// Replayed events are the authoritative rebuild of keyed state; a consumer
// keeping a visible list uses the mark to deduplicate. Without it, every event
// lost in a connection gap strands an in-flight row forever.
func TestBacklogMarksReplay(t *testing.T) {
	l := NewLog(10)
	l.EmitRequest(1, 0, "m → gpu-0")
	for _, e := range l.Backlog() {
		if !e.Replay {
			t.Error("backlog event not marked as replay")
		}
	}
	// Recent() is a plain history read and must NOT mark, or a dashboard
	// showing history would treat it as a state rebuild.
	for _, e := range l.Recent() {
		if e.Replay {
			t.Error("Recent() must not mark events as replay")
		}
	}
}

// Marking must not mutate the stored ring, or the second reader sees events
// already flagged.
func TestBacklogDoesNotMutateRing(t *testing.T) {
	l := NewLog(10)
	l.EmitRequest(1, 0, "m → gpu-0")
	_ = l.Backlog()
	for _, e := range l.Recent() {
		if e.Replay {
			t.Error("Backlog mutated the stored event")
		}
	}
}

func TestSubscriberReceivesLiveEvents(t *testing.T) {
	l := NewLog(10)
	ch := l.Subscribe()
	defer l.Unsubscribe(ch)
	l.EmitRequest(7, 1, "m → gpu-1")
	select {
	case e := <-ch:
		if e.RequestID != 7 || e.Type != meshapi.EventRequest {
			t.Errorf("got %+v", e)
		}
	case <-time.After(time.Second):
		t.Fatal("no event delivered")
	}
}

// A wedged subscriber must not stall the request path that emitted the event.
func TestSlowSubscriberDoesNotBlockEmit(t *testing.T) {
	l := NewLog(10)
	ch := l.Subscribe()
	defer l.Unsubscribe(ch)
	done := make(chan struct{})
	go func() {
		for i := 0; i < 5000; i++ {
			l.Emit(meshapi.EventSystem, 0, "flood %d", i)
		}
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("emit blocked on a subscriber that never reads")
	}
}

func TestUnsubscribeIsIdempotent(t *testing.T) {
	l := NewLog(10)
	ch := l.Subscribe()
	l.Unsubscribe(ch)
	l.Unsubscribe(ch) // must not panic on a double close
	if l.SubscriberCount() != 0 {
		t.Errorf("got %d subscribers", l.SubscriberCount())
	}
}

func TestRequestIDsAreUnique(t *testing.T) {
	seen := map[int64]bool{}
	for i := 0; i < 100; i++ {
		id := NewRequestID()
		if seen[id] {
			t.Fatalf("duplicate request id %d", id)
		}
		seen[id] = true
	}
}

func TestPromptStoreRoundTrip(t *testing.T) {
	p := NewPromptStore(10)
	p.Store(1, 100, "m", "the prompt")
	p.StoreOutput(1, 101, "m", "the answer", 1234)
	got, ok := p.Get(1)
	if !ok {
		t.Fatal("entry missing")
	}
	if got.Prompt != "the prompt" || got.Output != "the answer" || got.ElapsedMS != 1234 {
		t.Errorf("got %+v", got)
	}
}

// A request whose prompt could not be extracted still produces output worth
// keeping; dropping it would leave a dashboard row that opens to nothing.
func TestStoreOutputCreatesEntryWithoutPrompt(t *testing.T) {
	p := NewPromptStore(10)
	p.StoreOutput(5, 100, "m", "answer only", 10)
	got, ok := p.Get(5)
	if !ok || got.Output != "answer only" {
		t.Errorf("got %+v ok=%v", got, ok)
	}
}

// A blank entry would still render a link in the dashboard that opens to
// nothing.
func TestEmptyPromptAndOutputDropped(t *testing.T) {
	p := NewPromptStore(10)
	p.Store(1, 0, "m", "")
	p.StoreOutput(2, 0, "m", "", 0)
	if _, ok := p.Get(1); ok {
		t.Error("empty prompt stored")
	}
	if _, ok := p.Get(2); ok {
		t.Error("empty output stored")
	}
}

func TestPromptStoreEvictsOldest(t *testing.T) {
	p := NewPromptStore(2)
	p.Store(1, 0, "m", "one")
	p.Store(2, 0, "m", "two")
	p.Store(3, 0, "m", "three")
	if _, ok := p.Get(1); ok {
		t.Error("oldest entry not evicted")
	}
	if _, ok := p.Get(3); !ok {
		t.Error("newest entry missing")
	}
}

// One pathological prompt must not dominate the store.
func TestOversizedTextTruncated(t *testing.T) {
	p := NewPromptStore(10)
	p.Store(1, 0, "m", strings.Repeat("x", maxPromptChars*2))
	got, _ := p.Get(1)
	if len(got.Prompt) != maxPromptChars {
		t.Errorf("prompt kept at %d chars, cap is %d", len(got.Prompt), maxPromptChars)
	}
}

// A misconfigured cap should degrade to the default, not to a store that drops
// everything written to it.
func TestZeroCapFallsBackToDefault(t *testing.T) {
	p := NewPromptStore(0)
	if p.Max() != DefaultPromptHistory {
		t.Errorf("got %d, want %d", p.Max(), DefaultPromptHistory)
	}
	p.Store(1, 0, "m", "kept")
	if _, ok := p.Get(1); !ok {
		t.Error("a zero cap produced a store that drops everything")
	}
}
