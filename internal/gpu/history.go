package gpu

import "sync"

// historySlots is one hour at the default five-second health tick. Fixed at
// construction, so memory cannot grow with uptime.
const historySlots = 720

// History keeps a bounded rolling window of samples per GPU.
//
// It covers every GPU nvidia-smi reports, not only the ones this instance was
// configured with. That is deliberate and load bearing for a multi-instance
// host: the fleet view wants the whole machine, and an instance that published
// only its own slice would leave a co-tenant's cards missing from the host's
// row rather than merely unattributed.
type History struct {
	mu     sync.RWMutex
	series map[int][]GPUSample
	order  []int
}

func NewHistory() *History {
	return &History{series: make(map[int][]GPUSample)}
}

func (h *History) Add(samples []GPUSample) {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, s := range samples {
		if _, ok := h.series[s.GPUID]; !ok {
			h.order = append(h.order, s.GPUID)
			// Sorted insert keeps Latest() in GPU order without sorting on
			// every read, which happens far more often than a new card
			// appears.
			for i := len(h.order) - 1; i > 0 && h.order[i-1] > h.order[i]; i-- {
				h.order[i-1], h.order[i] = h.order[i], h.order[i-1]
			}
		}
		ring := append(h.series[s.GPUID], s)
		if len(ring) > historySlots {
			ring = ring[len(ring)-historySlots:]
		}
		h.series[s.GPUID] = ring
	}
}

// Latest returns the most recent sample for each GPU, in GPU-id order.
func (h *History) Latest() []GPUSample {
	h.mu.RLock()
	defer h.mu.RUnlock()
	out := make([]GPUSample, 0, len(h.order))
	for _, id := range h.order {
		if ring := h.series[id]; len(ring) > 0 {
			out = append(out, ring[len(ring)-1])
		}
	}
	return out
}

// Series returns the full retained window for each GPU, in GPU-id order.
func (h *History) Series() map[int][]GPUSample {
	h.mu.RLock()
	defer h.mu.RUnlock()
	out := make(map[int][]GPUSample, len(h.series))
	for id, ring := range h.series {
		cp := make([]GPUSample, len(ring))
		copy(cp, ring)
		out[id] = cp
	}
	return out
}
