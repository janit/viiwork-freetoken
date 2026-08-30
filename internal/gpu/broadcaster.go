package gpu

import "sync"

// Broadcaster fans live GPU samples out to dashboard SSE subscribers.
//
// Sends are non-blocking onto buffered channels and a full channel drops the
// sample. That is the right trade for a metrics feed: a browser tab that has
// been throttled or slept must not apply backpressure to the sampling loop of
// a node serving inference, and the next tick carries a complete picture
// anyway — these are absolute readings, not deltas, so a dropped one costs a
// frame of animation and nothing else.
type Broadcaster struct {
	mu   sync.RWMutex
	subs map[chan []GPUSample]struct{}
}

func NewBroadcaster() *Broadcaster {
	return &Broadcaster{subs: make(map[chan []GPUSample]struct{})}
}

func (b *Broadcaster) Subscribe() chan []GPUSample {
	ch := make(chan []GPUSample, 8)
	b.mu.Lock()
	b.subs[ch] = struct{}{}
	b.mu.Unlock()
	return ch
}

func (b *Broadcaster) Unsubscribe(ch chan []GPUSample) {
	b.mu.Lock()
	if _, ok := b.subs[ch]; ok {
		delete(b.subs, ch)
		close(ch)
	}
	b.mu.Unlock()
}

func (b *Broadcaster) Publish(samples []GPUSample) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	for ch := range b.subs {
		select {
		case ch <- samples:
		default:
		}
	}
}

func (b *Broadcaster) SubscriberCount() int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return len(b.subs)
}
