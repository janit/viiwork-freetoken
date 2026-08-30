package proxy

import (
	"bytes"
	"net/http"
	"time"

	"github.com/janit/viiwork/meshapi"
)

// originHeader marks a request as already forwarded once.
//
// Loop prevention is the whole reason it exists. Two nodes that each believe
// the other owns a model would otherwise bounce a request between them
// forever; a node seeing its own id here refuses to forward again and serves
// or fails locally instead.
const originHeader = "X-Viiwork-Origin"

// peerClient is separate from backendClient so a slow peer cannot exhaust the
// connection pool the local backends use. Like it, it has no overall timeout:
// the peer is serving a generation that legitimately takes minutes.
var peerClient = &http.Client{
	Transport: &http.Transport{
		Proxy:               nil,
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 16,
		IdleConnTimeout:     90 * time.Second,
		DisableCompression:  true,
	},
}

// proxyToPeer forwards a request to another node and streams the response
// back, so the original client sees one response from one address and never
// learns the mesh exists.
func proxyToPeer(w http.ResponseWriter, r *http.Request, body []byte, addr, nodeID string) (aborted bool) {
	url := "http://" + addr + r.URL.Path
	req, err := http.NewRequestWithContext(r.Context(), r.Method, url, bytes.NewReader(body))
	if err != nil {
		http.Error(w, `{"error":{"message":"bad peer request","type":"internal"}}`, http.StatusInternalServerError)
		return false
	}
	copyRequestHeaders(req, r)
	req.Header.Set(originHeader, nodeID)

	resp, err := peerClient.Do(req)
	if err != nil {
		if isClientAbort(r, err) {
			return true
		}
		http.Error(w, `{"error":{"message":"peer unavailable","type":"unavailable"}}`, http.StatusBadGateway)
		return false
	}
	defer resp.Body.Close()

	copyResponseHeaders(w, resp)
	// Preserve the peer's own backend attribution rather than overwriting it:
	// the request really was served on that host's card, and a client tracing
	// a slow response needs the true origin.
	if resp.Header.Get("X-GPU-Backend") == "" {
		w.Header().Set("X-GPU-Backend", meshapi.PeerLabel(addr))
	}
	w.WriteHeader(resp.StatusCode)
	return streamBody(w, resp.Body, r)
}
