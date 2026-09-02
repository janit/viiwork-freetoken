package gpu

import (
	"context"
	"strconv"
	"strings"
	"time"
)

// uuidArgs asks nvidia-smi for the one property that identifies a card
// independently of any ordering.
//
// Kept out of argSets, and read once rather than sampled, for two reasons. A
// UUID does not change for the life of a card, so re-reading it on every health
// tick would be waste. And GPUSample feeds meshapi.GPUInfo, so a field added
// there is a protocol question for viiwork rather than a local one — this map
// stays on this side of the wire.
var uuidArgs = []string{"--query-gpu=index,uuid", "--format=csv,noheader"}

// UUIDs maps nvidia-smi's GPU index to the card's UUID.
//
// Nil when nvidia-smi is unavailable, which is the normal case in a unit test
// and on a host whose driver is being upgraded. A caller must read an absent
// entry as "unknown" rather than as a mismatch: the point of this map is to
// catch a backend running on the wrong card, and a map that failed to load must
// never manufacture one.
func UUIDs() map[int]string { return uuids(nvidiaSMI) }

// uuids takes the command factory for the same reason newStatCollector does:
// the fallback to nothing is the behaviour worth testing, and a test that
// re-implemented the call would prove nothing.
func uuids(factory func([]string) cmdFunc) map[int]string {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	out, err := factory(uuidArgs)(ctx)
	if err != nil {
		return nil
	}
	return parseUUIDs(out)
}

// parseUUIDs reads `index, uuid` rows. A row that is not both is skipped rather
// than failing the batch, on parseCSV's reasoning: a card the driver cannot
// answer for must not cost us the identity of the other nine.
func parseUUIDs(out []byte) map[int]string {
	m := map[int]string{}
	for _, line := range strings.Split(string(out), "\n") {
		fields := strings.SplitN(strings.TrimSpace(line), ",", 2)
		if len(fields) != 2 {
			continue
		}
		id, err := strconv.Atoi(strings.TrimSpace(fields[0]))
		if err != nil {
			continue
		}
		// The GPU- prefix is what the driver matches on, and requiring it also
		// rejects the [N/A] a virtualised or MIG configuration can print here.
		// An unidentifiable card is absent, not an empty identity that would
		// match nothing and report a correctly pinned backend as wrong.
		uuid := strings.TrimSpace(fields[1])
		if !strings.HasPrefix(uuid, "GPU-") {
			continue
		}
		m[id] = uuid
	}
	if len(m) == 0 {
		return nil
	}
	return m
}
