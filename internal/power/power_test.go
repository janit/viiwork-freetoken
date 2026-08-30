package power

import (
	"testing"

	"github.com/janit/viiwork-freetoken/internal/gpu"
)

// hist builds a History holding one sample per given wattage, GPU ids 0..n-1.
func hist(watts ...float64) *gpu.History {
	h := gpu.NewHistory()
	samples := make([]gpu.GPUSample, 0, len(watts))
	for i, w := range watts {
		samples = append(samples, gpu.GPUSample{GPUID: i, PowerW: w})
	}
	h.Add(samples)
	return h
}

func yes() bool { return true }
func no() bool  { return false }

func TestWattsSumsEveryCard(t *testing.T) {
	s := NewSum(hist(40.5, 160.25, 20), yes)
	if got, want := s.Watts(), 220.75; got != want {
		t.Fatalf("Watts() = %v, want %v", got, want)
	}
}

// The sum has to span the host, not this instance's slice: node wattage is a
// whole-host figure and the energy store's denominator depends on it.
func TestWattsCoversCardsThisInstanceDoesNotOwn(t *testing.T) {
	h := gpu.NewHistory()
	h.Add([]gpu.GPUSample{{GPUID: 0, PowerW: 100}})
	h.Add([]gpu.GPUSample{{GPUID: 7, PowerW: 25}})
	if got, want := NewSum(h, yes).Watts(), 125.0; got != want {
		t.Fatalf("Watts() = %v, want %v", got, want)
	}
}

func TestWattsUsesMostRecentSamplePerCard(t *testing.T) {
	h := gpu.NewHistory()
	h.Add([]gpu.GPUSample{{GPUID: 0, PowerW: 10}})
	h.Add([]gpu.GPUSample{{GPUID: 0, PowerW: 150}})
	if got, want := NewSum(h, yes).Watts(), 150.0; got != want {
		t.Fatalf("Watts() = %v, want %v", got, want)
	}
}

// Absent is not zero. nvidia-smi reports "[N/A]" on cards whose driver does not
// expose wattage, and the collector tracks that separately from whether metrics
// work at all. Publishing 0 W for it would render as a real measurement.
func TestUnavailablePowerReportsNothing(t *testing.T) {
	s := NewSum(hist(40, 160), no)
	if s.Available() {
		t.Fatal("Available() = true, want false")
	}
	if got := s.Watts(); got != 0 {
		t.Fatalf("Watts() = %v with power unavailable, want 0 so omitempty drops it", got)
	}
}

func TestAvailableTracksCollector(t *testing.T) {
	if !NewSum(hist(40), yes).Available() {
		t.Fatal("Available() = false, want true")
	}
}

// An empty history is "not measured yet", not a zero-watt host.
func TestNoSamplesYet(t *testing.T) {
	s := NewSum(gpu.NewHistory(), yes)
	if got := s.Watts(); got != 0 {
		t.Fatalf("Watts() = %v with no samples, want 0", got)
	}
}

// The label is the vocabulary shared with meshapi's PowerSource and the energy
// store's Config.Source, so a history read outside its mesh context still says
// what it measured.
func TestSourceLabel(t *testing.T) {
	if got, want := NewSum(hist(40), yes).Source(), "nvidia-smi"; got != want {
		t.Fatalf("Source() = %q, want %q", got, want)
	}
}

func TestNilAvailabilityFuncIsUnavailable(t *testing.T) {
	s := NewSum(hist(40), nil)
	if s.Available() {
		t.Fatal("Available() = true with nil availability func, want false")
	}
}

func TestReadingsStampModelPerCard(t *testing.T) {
	got := Readings(hist(40, 160, 20), map[int]string{0: "granite", 1: "granite", 2: "qwen"})
	if len(got) != 3 {
		t.Fatalf("got %d readings, want 3", len(got))
	}
	want := []struct {
		id    int
		watts float64
		model string
	}{{0, 40, "granite"}, {1, 160, "granite"}, {2, 20, "qwen"}}
	for i, w := range want {
		if got[i].GPUID != w.id || got[i].Watts != w.watts || got[i].Model != w.model {
			t.Errorf("reading %d = %+v, want id=%d watts=%v model=%q", i, got[i], w.id, w.watts, w.model)
		}
	}
}

// A card nothing claims is unattributed, not misattributed. On a multi-instance
// host the recording instance may not know what a co-tenant runs.
func TestReadingsLeaveUnclaimedCardsUnlabelled(t *testing.T) {
	got := Readings(hist(40, 160), map[int]string{0: "granite"})
	if len(got) != 2 {
		t.Fatalf("got %d readings, want 2", len(got))
	}
	if got[1].Model != "" {
		t.Errorf("unclaimed card model = %q, want empty", got[1].Model)
	}
	if got[1].Watts != 160 {
		t.Errorf("unclaimed card watts = %v, want 160 — still measured", got[1].Watts)
	}
}

// Direct attribution only reconciles if the node figure is the sum of exactly
// the readings reported alongside it. Pinning that here because the two are
// produced by separate calls and nothing else checks they agree.
func TestReadingsSumEqualsNodeWatts(t *testing.T) {
	h := hist(40.5, 160.25, 20)
	var total float64
	for _, r := range Readings(h, nil) {
		total += r.Watts
	}
	if got := NewSum(h, yes).Watts(); got != total {
		t.Fatalf("Watts() = %v but readings sum to %v; Direct attribution would not reconcile", got, total)
	}
}
