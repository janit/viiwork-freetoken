package gpu

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"
)

// Verbatim shape of `nvidia-smi --query-gpu=index,utilization.gpu,memory.used,
// memory.total,power.draw --format=csv,noheader,nounits` on a multi-GPU host.
const smiWithPower = `0, 97, 22831, 24564, 341.22
1, 0, 512, 24564, 34.15
2, 45, 18004, 24564, 210.07
`

func TestParseCSVWithPower(t *testing.T) {
	got := parseCSV([]byte(smiWithPower), time.Unix(1000, 0))
	want := []GPUSample{
		{GPUID: 0, Utilization: 97, VRAMUsedMB: 22831, VRAMTotalMB: 24564, PowerW: 341.22, Timestamp: 1000},
		{GPUID: 1, Utilization: 0, VRAMUsedMB: 512, VRAMTotalMB: 24564, PowerW: 34.15, Timestamp: 1000},
		{GPUID: 2, Utilization: 45, VRAMUsedMB: 18004, VRAMTotalMB: 24564, PowerW: 210.07, Timestamp: 1000},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %+v\nwant %+v", got, want)
	}
	if !hasPower(got) {
		t.Error("hasPower should be true when wattage parsed")
	}
}

// Output from the fallback arg set must still parse, leaving power absent.
func TestParseCSVWithoutPower(t *testing.T) {
	got := parseCSV([]byte("0, 12, 100, 24564\n"), time.Unix(1, 0))
	if len(got) != 1 || got[0].PowerW != 0 {
		t.Fatalf("got %+v", got)
	}
	if hasPower(got) {
		t.Error("hasPower should be false with no wattage column")
	}
}

// A card whose driver cannot supply a field prints [N/A] for that field only.
// The rest of the row is real data and must survive.
func TestParseCSVNotAvailableFields(t *testing.T) {
	got := parseCSV([]byte("0, [N/A], 8000, 24564, [N/A]\n"), time.Unix(1, 0))
	if len(got) != 1 {
		t.Fatalf("row dropped: %+v", got)
	}
	if got[0].VRAMUsedMB != 8000 || got[0].VRAMTotalMB != 24564 {
		t.Errorf("VRAM lost to an unrelated [N/A]: %+v", got[0])
	}
	if got[0].PowerW != 0 || got[0].Utilization != 0 {
		t.Errorf("[N/A] should read as absent: %+v", got[0])
	}
	if hasPower(got) {
		t.Error("[N/A] power must not count as available")
	}
}

// One malformed row must not cost the readings for every other card on the
// host.
func TestParseCSVSkipsMalformedRows(t *testing.T) {
	got := parseCSV([]byte("garbage\n1, 50, 100, 200, 30\n\nx, y, z, w\n"), time.Unix(1, 0))
	if len(got) != 1 || got[0].GPUID != 1 {
		t.Fatalf("got %+v, want only GPU 1", got)
	}
}

func fixedCmd(out string, err error) cmdFunc {
	return func(context.Context) ([]byte, error) { return []byte(out), err }
}

// A driver that rejects power.draw must fall back rather than disabling
// metrics: wattage is a bonus, and losing utilisation to gain it is a bad
// trade.
func TestCollectorFallsBackWhenPowerRejected(t *testing.T) {
	var used []string
	c := newStatCollector(NewHistory(), nil, func(args []string) cmdFunc {
		used = args
		if len(args) > 0 && contains(args[0], "power.draw") {
			return fixedCmd("", errors.New("unrecognized field"))
		}
		return fixedCmd("0, 10, 100, 200\n", nil)
	})
	if !c.Available() {
		t.Fatal("collector should be available via the fallback arg set")
	}
	if c.PowerAvailable() {
		t.Error("power must be reported unavailable when the field was rejected")
	}
	if contains(used[0], "power.draw") {
		t.Errorf("settled on the rejected arg set: %v", used)
	}
}

// nvidia-smi can exit 0 and report [N/A] for every card's power. That is not a
// working power probe, and judging it by exit status would publish zeros.
func TestCollectorDetectsPowerlessSuccess(t *testing.T) {
	c := newStatCollector(NewHistory(), nil, func([]string) cmdFunc {
		return fixedCmd("0, 10, 100, 200, [N/A]\n", nil)
	})
	if !c.Available() {
		t.Fatal("metrics should still be available")
	}
	if c.PowerAvailable() {
		t.Error("all-[N/A] power must be reported unavailable")
	}
}

// Exit 0 with nothing parseable is not a working probe; accepting it would
// publish an empty GPU list and look like a host with no cards.
func TestCollectorRejectsEmptyOutput(t *testing.T) {
	c := newStatCollector(NewHistory(), nil, func([]string) cmdFunc {
		return fixedCmd("\n", nil)
	})
	if c.Available() {
		t.Error("empty output must not count as available")
	}
}

func TestCollectorUnavailableWhenBinaryMissing(t *testing.T) {
	c := newStatCollector(NewHistory(), nil, func([]string) cmdFunc {
		return fixedCmd("", errors.New("executable file not found"))
	})
	if c.Available() {
		t.Error("want unavailable")
	}
	if got := c.Collect(context.Background()); got != nil {
		t.Errorf("Collect on an unavailable collector must be a no-op, got %+v", got)
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
