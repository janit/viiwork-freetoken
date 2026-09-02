package gpu

import (
	"errors"
	"testing"
)

// The real output of `nvidia-smi --query-gpu=index,uuid --format=csv,noheader`
// on a two-card host.
const uuidCSV = `0, GPU-2f3a9b1c-8d7e-4a05-b6c1-0e5f9a3d7b42
1, GPU-9e8d7c6b-5a49-4f13-8207-c1b0a4e6d3f5
`

func TestUUIDsMapsIndexToCard(t *testing.T) {
	got := uuids(func([]string) cmdFunc { return fixedCmd(uuidCSV, nil) })
	if len(got) != 2 {
		t.Fatalf("got %d cards, want 2: %v", len(got), got)
	}
	if got[1] != "GPU-9e8d7c6b-5a49-4f13-8207-c1b0a4e6d3f5" {
		t.Errorf("index 1 = %q, want the second card", got[1])
	}
}

// Nil rather than an empty map, and the caller must read that as "unknown".
// Reporting no cards as an empty identity would make every backend look
// mispinned on a host where nvidia-smi is simply absent.
func TestUUIDsAreNilWhenSMIIsUnavailable(t *testing.T) {
	if got := uuids(func([]string) cmdFunc {
		return fixedCmd("", errors.New("executable file not found"))
	}); got != nil {
		t.Errorf("got %v, want nil", got)
	}
}

// A card the driver cannot identify is absent, not an empty-string identity.
// Its neighbours on the same host must survive it.
func TestUUIDsSkipUnidentifiableCards(t *testing.T) {
	got := parseUUIDs([]byte("0, [N/A]\n1, GPU-9e8d7c6b-5a49-4f13-8207-c1b0a4e6d3f5\nnonsense\n"))
	if _, ok := got[0]; ok {
		t.Errorf("an [N/A] card must not appear: %v", got)
	}
	if got[1] == "" {
		t.Errorf("the identifiable card must survive its neighbour: %v", got)
	}
}

func TestUUIDsNilWhenNothingParses(t *testing.T) {
	if got := parseUUIDs([]byte("\n")); got != nil {
		t.Errorf("got %v, want nil", got)
	}
}
