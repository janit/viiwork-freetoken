package proxy

import (
	"bytes"
	"errors"
	"io"
	"log"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/janit/viiwork-freetoken/internal/balancer"
)

// backendClient has no overall timeout on purpose: a long generation is a
// normal request, not a hung one, and a client-side deadline would sever it
// mid-stream. Liveness is the health loop's job; abandonment is detected from
// the caller's context instead.
var backendClient = &http.Client{
	Transport: &http.Transport{
		Proxy: nil, // never route loopback backend calls through a corporate proxy
		DialContext: (&net.Dialer{
			Timeout:   5 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		MaxIdleConns:        200,
		MaxIdleConnsPerHost: 64,
		IdleConnTimeout:     90 * time.Second,
		// Streaming responses must not be buffered by the transport, or
		// token-by-token output arrives in blocks.
		DisableCompression: true,
	},
}

// proxyRequest forwards a request to a local backend and streams the response
// back. It returns whether the client abandoned the request.
func proxyRequest(w http.ResponseWriter, r *http.Request, body []byte, backend *balancer.BackendState) (aborted bool) {
	backend.IncrInFlight()
	defer backend.DecrInFlight()

	start := time.Now()
	url := "http://" + backend.Addr + r.URL.Path
	req, err := http.NewRequestWithContext(r.Context(), r.Method, url, bytes.NewReader(body))
	if err != nil {
		http.Error(w, `{"error":{"message":"bad upstream request","type":"internal"}}`, http.StatusInternalServerError)
		return false
	}
	copyRequestHeaders(req, r)

	resp, err := backendClient.Do(req)
	if err != nil {
		if isClientAbort(r, err) {
			return true
		}
		// EOF or connection refused is a kernel-level signal that the process
		// is gone, not a slow backend. Latching it lets the manager skip the
		// three-strike ladder and respawn after one failed probe.
		if isHardFailure(err) {
			log.Printf("[proxy] %s evicted: hard socket failure on the inference path (%v)", backend.Label(), err)
			backend.NoteHardFailure()
		}
		http.Error(w, `{"error":{"message":"backend unavailable","type":"unavailable"}}`, http.StatusBadGateway)
		return false
	}
	defer resp.Body.Close()

	// Set before WriteHeader or they are lost. X-GPU-Backend is how a client
	// attributes a response to a card, which is the whole point of the label.
	copyResponseHeaders(w, resp)
	w.Header().Set("X-GPU-Backend", backend.Label())
	w.Header().Set("X-Queue-Depth", strconv.FormatInt(backend.InFlight(), 10))
	w.WriteHeader(resp.StatusCode)

	aborted = streamBody(w, resp.Body, r)
	if !aborted && resp.StatusCode < 500 {
		backend.RecordLatency(time.Since(start))
	}
	return aborted
}

// streamBody copies the response, flushing as it goes so a streamed generation
// reaches the client token by token rather than in one block at the end.
func streamBody(w http.ResponseWriter, body io.Reader, r *http.Request) (aborted bool) {
	flusher, _ := w.(http.Flusher)
	buf := make([]byte, 32*1024)
	for {
		n, err := body.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				// The client is gone. Returning rather than continuing to read
				// lets the deferred in-flight decrement run and frees the slot.
				return true
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		if err != nil {
			if errors.Is(err, io.EOF) {
				return false
			}
			return isClientAbort(r, err)
		}
	}
}

// hopByHop headers are connection-scoped and must not be forwarded.
var hopByHop = map[string]bool{
	"Connection":          true,
	"Keep-Alive":          true,
	"Proxy-Authenticate":  true,
	"Proxy-Authorization": true,
	"Te":                  true,
	"Trailer":             true,
	"Transfer-Encoding":   true,
	"Upgrade":             true,
}

func copyRequestHeaders(dst *http.Request, src *http.Request) {
	for k, vs := range src.Header {
		if hopByHop[http.CanonicalHeaderKey(k)] {
			continue
		}
		for _, v := range vs {
			dst.Header.Add(k, v)
		}
	}
	if dst.Header.Get("Content-Type") == "" {
		dst.Header.Set("Content-Type", "application/json")
	}
}

func copyResponseHeaders(w http.ResponseWriter, resp *http.Response) {
	for k, vs := range resp.Header {
		if hopByHop[http.CanonicalHeaderKey(k)] {
			continue
		}
		// Do not clobber CORS headers already set for this response: they were
		// computed from the caller's Origin, and the backend knows nothing
		// about it.
		if strings.HasPrefix(http.CanonicalHeaderKey(k), "Access-Control-") {
			continue
		}
		for _, v := range vs {
			w.Header().Add(k, v)
		}
	}
}

// isClientAbort reports whether an error is the caller having gone away rather
// than the backend failing. The distinction matters: an abort is the client's
// doing and says nothing about backend health, so it must not count toward the
// failure ladder.
func isClientAbort(r *http.Request, err error) bool {
	if r.Context().Err() != nil {
		return true
	}
	return errors.Is(err, http.ErrHandlerTimeout)
}

// isHardFailure reports a kernel-level signal that the backend process is
// gone, as opposed to it being slow or returning an error status.
func isHardFailure(err error) bool {
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return true
	}
	if errors.Is(err, syscallECONNREFUSED) {
		return true
	}
	var opErr *net.OpError
	if errors.As(err, &opErr) {
		return true
	}
	s := err.Error()
	return strings.Contains(s, "connection refused") || strings.Contains(s, "EOF")
}
