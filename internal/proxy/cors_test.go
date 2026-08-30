package proxy

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/janit/viiwork/meshapi"
)

func testCORS() *CORS {
	return &CORS{Origins: []string{"*.ts.net", "localhost"}, TailnetIPs: true}
}

func TestCORSAllows(t *testing.T) {
	c := testCORS()
	for _, tc := range []struct {
		origin string
		want   bool
	}{
		{"https://laptop.tail1234.ts.net", true},
		{"http://localhost", true},
		{"http://localhost:3000", true},
		{"http://100.101.102.103", true},     // tailnet IPv4
		{"http://[fd7a:115c:a1e0::1]", true}, // tailnet IPv6
		{"https://evil.example.com", false},
		{"http://10.0.0.5", false}, // a LAN IP is not a tailnet IP
		{"", false},
		{"null", false},
		// A "*." rule must not match the bare apex, or "*.ts.net" silently
		// widens into somebody else's domain.
		{"https://ts.net", false},
		// Only schemes a browser sends for a page fetch.
		{"chrome-extension://abcdef", false},
		{"file://", false},
	} {
		if got := c.Allows(tc.origin); got != tc.want {
			t.Errorf("Allows(%q) = %v, want %v", tc.origin, got, tc.want)
		}
	}
}

func TestNilCORSAllowsNothing(t *testing.T) {
	var c *CORS
	if c.Allows("https://anything.ts.net") {
		t.Error("a nil CORS must allow nothing")
	}
}

// EventSource is CORS-bound like any fetch but sends no preflight, so the
// header has to land on the stream's own GET response.
func TestSSEStreamCarriesAllowHeader(t *testing.T) {
	hn := newHarness(t, "m", nil, nil)
	hn.h.SetCORS(testCORS())
	r := httptest.NewRequest("GET", meshapi.PathActivityStream, nil)
	r.Header.Set("Origin", "https://laptop.tail1234.ts.net")
	ctx, cancel := contextWithImmediateCancel()
	r = r.WithContext(ctx)
	cancel()
	w := httptest.NewRecorder()
	hn.h.ServeHTTP(w, r)
	if got := w.Header().Get("Access-Control-Allow-Origin"); got == "" {
		t.Error("SSE response carries no allow header; EventSource would be blocked")
	}
}

// The router matches only GET and POST, so a preflight has to be answered
// ahead of routing or it falls through to 404.
func TestPreflightAnsweredBeforeRouting(t *testing.T) {
	hn := newHarness(t, "m", nil, nil)
	hn.h.SetCORS(testCORS())
	r := httptest.NewRequest("OPTIONS", meshapi.PathChatCompletions, nil)
	r.Header.Set("Origin", "https://laptop.tail1234.ts.net")
	r.Header.Set("Access-Control-Request-Method", "POST")
	w := httptest.NewRecorder()
	hn.h.ServeHTTP(w, r)
	if w.Code != http.StatusNoContent {
		t.Fatalf("got %d, want 204", w.Code)
	}
	if w.Header().Get("Access-Control-Allow-Methods") == "" {
		t.Error("preflight carries no allowed methods")
	}
}

// A refused preflight returns 403 rather than a bare 204: the browser blocks
// either way, but only one of them says why.
func TestRefusedPreflightIs403(t *testing.T) {
	hn := newHarness(t, "m", nil, nil)
	hn.h.SetCORS(testCORS())
	r := httptest.NewRequest("OPTIONS", meshapi.PathChatCompletions, nil)
	r.Header.Set("Origin", "https://evil.example.com")
	r.Header.Set("Access-Control-Request-Method", "POST")
	w := httptest.NewRecorder()
	hn.h.ServeHTTP(w, r)
	if w.Code != http.StatusForbidden {
		t.Errorf("got %d, want 403", w.Code)
	}
}

// The response genuinely differs by origin; a shared cache that missed that
// would hand one origin's allow header to another.
func TestVaryOriginAlwaysSet(t *testing.T) {
	hn := newHarness(t, "m", nil, nil)
	hn.h.SetCORS(testCORS())
	for _, origin := range []string{"https://laptop.tail1234.ts.net", "https://evil.example.com"} {
		r := httptest.NewRequest("GET", meshapi.PathStatus, nil)
		r.Header.Set("Origin", origin)
		w := httptest.NewRecorder()
		hn.h.ServeHTTP(w, r)
		if w.Header().Get("Vary") == "" {
			t.Errorf("%s: Vary not set", origin)
		}
	}
}

// With no CORS configured the node sends no header at all.
func TestNoCORSConfiguredSendsNoHeader(t *testing.T) {
	hn := newHarness(t, "m", nil, nil)
	r := httptest.NewRequest("GET", meshapi.PathStatus, nil)
	r.Header.Set("Origin", "https://laptop.tail1234.ts.net")
	w := httptest.NewRecorder()
	hn.h.ServeHTTP(w, r)
	if w.Header().Get("Access-Control-Allow-Origin") != "" {
		t.Error("an unconfigured node must send no allow header")
	}
}
