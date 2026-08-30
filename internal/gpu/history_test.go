package gpu

import "testing"

func TestHistoryLatestIsGPUOrdered(t *testing.T) {
	h := NewHistory()
	// Arrive out of order, as a driver listing MIG devices can.
	h.Add([]GPUSample{{GPUID: 3, Utilization: 1}, {GPUID: 0, Utilization: 2}, {GPUID: 1, Utilization: 3}})
	got := h.Latest()
	if len(got) != 3 {
		t.Fatalf("got %d samples", len(got))
	}
	for i, want := range []int{0, 1, 3} {
		if got[i].GPUID != want {
			t.Errorf("position %d: got GPU %d, want %d", i, got[i].GPUID, want)
		}
	}
}

func TestHistoryKeepsMostRecent(t *testing.T) {
	h := NewHistory()
	h.Add([]GPUSample{{GPUID: 0, Utilization: 10}})
	h.Add([]GPUSample{{GPUID: 0, Utilization: 90}})
	if got := h.Latest(); len(got) != 1 || got[0].Utilization != 90 {
		t.Errorf("got %+v, want the later sample", got)
	}
}

// Memory must not grow with uptime; the ring is fixed at construction.
func TestHistoryRingIsBounded(t *testing.T) {
	h := NewHistory()
	for i := 0; i < historySlots+250; i++ {
		h.Add([]GPUSample{{GPUID: 0, Utilization: float64(i)}})
	}
	series := h.Series()
	if len(series[0]) != historySlots {
		t.Errorf("ring grew to %d, want %d", len(series[0]), historySlots)
	}
	// The window must have slid, not stalled at the oldest entries.
	if last := series[0][len(series[0])-1]; last.Utilization != float64(historySlots+249) {
		t.Errorf("newest sample is %v, want the most recent", last.Utilization)
	}
}

func TestBroadcasterDropsRatherThanBlocks(t *testing.T) {
	b := NewBroadcaster()
	ch := b.Subscribe()
	defer b.Unsubscribe(ch)
	// Far more than the channel buffer. A slept browser tab must not apply
	// backpressure to the sampling loop.
	for i := 0; i < 100; i++ {
		b.Publish([]GPUSample{{GPUID: 0, Utilization: float64(i)}})
	}
	if b.SubscriberCount() != 1 {
		t.Errorf("subscriber lost: %d", b.SubscriberCount())
	}
}

func TestUnsubscribeIsIdempotent(t *testing.T) {
	b := NewBroadcaster()
	ch := b.Subscribe()
	b.Unsubscribe(ch)
	b.Unsubscribe(ch) // must not panic on a double close
	if b.SubscriberCount() != 0 {
		t.Errorf("got %d subscribers", b.SubscriberCount())
	}
}
