package gpu

import (
	"context"
	"log"
	"os"
	"os/exec"
	"sync/atomic"
	"time"
)

type cmdFunc func(ctx context.Context) ([]byte, error)

// argSets are tried in order at startup. Power is requested first because it
// is the most useful thing NVIDIA hardware reports that AMD's gfx906 does not
// report reliably — but a driver that rejects the field must not take
// utilisation and VRAM down with it. The second set is the irreducible one.
var argSets = [][]string{
	{"--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw", "--format=csv,noheader,nounits"},
	{"--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"},
}

func nvidiaSMI(args []string) cmdFunc {
	return func(ctx context.Context) ([]byte, error) {
		ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
		defer cancel()
		return exec.CommandContext(ctx, "nvidia-smi", args...).Output()
	}
}

// StatCollector samples nvidia-smi on the health tick and feeds the history
// and the live broadcaster. It degrades to doing nothing when nvidia-smi is
// unavailable, which is the normal case in a unit test and on a host whose
// driver is being upgraded — GPU metrics are a view, not a dependency of
// serving traffic.
type StatCollector struct {
	history     *History
	broadcaster *Broadcaster
	available   atomic.Bool
	// powerAvailable is tracked separately from available: metrics can be
	// working fine while wattage is absent, and a consumer of per-GPU power
	// needs to know the difference rather than reading zeros.
	powerAvailable atomic.Bool
	logger         *log.Logger
	cmd            cmdFunc
}

func NewStatCollector(history *History, broadcaster *Broadcaster) *StatCollector {
	return newStatCollector(history, broadcaster, nvidiaSMI)
}

// newStatCollector takes the command factory so tests can drive the arg-set
// fallback without a GPU. Probing the real thing is the whole point of the
// fallback, so a test that re-implemented the loop would prove nothing.
func newStatCollector(history *History, broadcaster *Broadcaster, factory func([]string) cmdFunc) *StatCollector {
	c := &StatCollector{
		history:     history,
		broadcaster: broadcaster,
		logger:      log.New(os.Stdout, "[gpu] ", log.LstdFlags),
	}
	ctx, cancel := context.WithTimeout(context.Background(), 6*time.Second)
	defer cancel()

	var lastErr error
	for _, args := range argSets {
		cmd := factory(args)
		out, err := cmd(ctx)
		if err != nil {
			lastErr = err
			continue
		}
		samples := parseCSV(out, time.Now())
		if len(samples) == 0 {
			// Exit 0 with nothing parseable is not a working probe. Accepting
			// it would publish an empty GPU list forever and look like a host
			// with no cards.
			lastErr = errNoSamples
			continue
		}
		c.cmd = cmd
		c.available.Store(true)
		c.powerAvailable.Store(hasPower(samples))
		if c.powerAvailable.Load() {
			c.logger.Printf("nvidia-smi available, %d GPUs, per-GPU power enabled", len(samples))
		} else {
			c.logger.Printf("nvidia-smi available, %d GPUs, without per-GPU power", len(samples))
		}
		return c
	}
	c.logger.Printf("nvidia-smi unavailable: %v (GPU metrics disabled)", lastErr)
	return c
}

type noSamplesError struct{}

func (noSamplesError) Error() string { return "nvidia-smi returned no parseable rows" }

var errNoSamples = noSamplesError{}

func (c *StatCollector) Available() bool      { return c.available.Load() }
func (c *StatCollector) PowerAvailable() bool { return c.powerAvailable.Load() }

// Collect takes one sample set and records it. Safe to call on an unavailable
// collector, where it does nothing.
func (c *StatCollector) Collect(ctx context.Context) []GPUSample {
	if !c.available.Load() || c.cmd == nil {
		return nil
	}
	out, err := c.cmd(ctx)
	if err != nil {
		c.logger.Printf("nvidia-smi failed: %v", err)
		return nil
	}
	samples := parseCSV(out, time.Now())
	if len(samples) == 0 {
		return nil
	}
	c.history.Add(samples)
	if c.broadcaster != nil {
		c.broadcaster.Publish(samples)
	}
	return samples
}

// Run samples on an interval until the context is cancelled.
func (c *StatCollector) Run(ctx context.Context, interval time.Duration) {
	if !c.available.Load() {
		return
	}
	c.Collect(ctx)
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			c.Collect(ctx)
		}
	}
}
