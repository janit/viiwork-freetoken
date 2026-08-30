package proxy

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/janit/viiwork-freetoken/internal/peer"
	"github.com/janit/viiwork/meshapi"
)

func contextWithImmediateCancel() (context.Context, context.CancelFunc) {
	return context.WithCancel(context.Background())
}

func TestPromptLookupReturnsStoredEntry(t *testing.T) {
	hn := newHarness(t, "m", nil, nil)
	hn.prompts.Store(42, 100, "m", "stored prompt")
	w := get(t, hn.h, meshapi.PathPrompts+"?rid=42")
	if w.Code != http.StatusOK {
		t.Fatalf("got %d", w.Code)
	}
	if !strings.Contains(w.Body.String(), "stored prompt") {
		t.Errorf("body: %s", w.Body)
	}
}

func TestPromptLookupUnknownRidIs404(t *testing.T) {
	hn := newHarness(t, "m", nil, nil)
	if w := get(t, hn.h, meshapi.PathPrompts+"?rid=999"); w.Code != http.StatusNotFound {
		t.Errorf("got %d, want 404", w.Code)
	}
	if w := get(t, hn.h, meshapi.PathPrompts+"?rid=notanumber"); w.Code != http.StatusNotFound {
		t.Errorf("bad rid: got %d, want 404", w.Code)
	}
}

// Without the peer-list check this endpoint fetches whatever address it is
// handed and echoes the response, which is an SSRF primitive on a LAN that
// also carries IPMI. This is the guard.
func TestMeshPromptRejectsUnknownAddr(t *testing.T) {
	reached := false
	victim := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reached = true
		w.Write([]byte(`{"secret":"internal service"}`))
	}))
	defer victim.Close()

	hn := newHarness(t, "m", nil, nil)
	w := get(t, hn.h, meshapi.PathMeshPrompt+"?rid=1&addr="+strings.TrimPrefix(victim.URL, "http://"))
	if w.Code != http.StatusForbidden {
		t.Errorf("got %d, want 403", w.Code)
	}
	if reached {
		t.Fatal("an unconfigured address was dialled: this endpoint is an SSRF primitive without the peer check")
	}
}

func TestMeshPromptForwardsToKnownPeer(t *testing.T) {
	peerSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != meshapi.PathPrompts {
			t.Errorf("peer got path %q", r.URL.Path)
		}
		w.Write([]byte(`{"rid":7,"prompt":"from the peer"}`))
	}))
	defer peerSrv.Close()
	addr := strings.TrimPrefix(peerSrv.URL, "http://")

	p := peer.NewPeerState(addr)
	hn := newHarness(t, "m", nil, []*peer.PeerState{p})
	w := get(t, hn.h, meshapi.PathMeshPrompt+"?rid=7&addr="+addr)
	if w.Code != http.StatusOK {
		t.Fatalf("got %d: %s", w.Code, w.Body)
	}
	if !strings.Contains(w.Body.String(), "from the peer") {
		t.Errorf("body: %s", w.Body)
	}
}

// No addr means the id was minted here.
func TestMeshPromptWithoutAddrIsLocal(t *testing.T) {
	hn := newHarness(t, "m", nil, nil)
	hn.prompts.Store(3, 0, "m", "local prompt")
	w := get(t, hn.h, meshapi.PathMeshPrompt+"?rid=3")
	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), "local prompt") {
		t.Errorf("got %d: %s", w.Code, w.Body)
	}
}
