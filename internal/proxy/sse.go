package proxy

import (
	"encoding/json"
	"fmt"
	"net/http"
)

// sseWriter is the shared preamble and send path for every event stream.
//
// The headers matter individually: without no-cache an intermediary may buffer
// the stream into a single response, and X-Accel-Buffering is what stops nginx
// doing exactly that when a node sits behind one.
type sseWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
}

func newSSEWriter(w http.ResponseWriter) (*sseWriter, bool) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return nil, false
	}
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()
	return &sseWriter{w: w, flusher: flusher}, true
}

// send writes one event, optionally named. It reports false once the client
// has gone, which is the signal for the caller's loop to unwind and drop its
// subscriptions.
func (s *sseWriter) send(event string, payload any) bool {
	data, err := json.Marshal(payload)
	if err != nil {
		return true // a payload we cannot encode is not a reason to drop the stream
	}
	if event != "" {
		if _, err := fmt.Fprintf(s.w, "event: %s\n", event); err != nil {
			return false
		}
	}
	if _, err := fmt.Fprintf(s.w, "data: %s\n\n", data); err != nil {
		return false
	}
	s.flusher.Flush()
	return true
}

// comment sends a no-op line. Used as a keepalive: an idle stream that writes
// nothing for minutes is indistinguishable from a dead one to an intermediary,
// and gets reaped.
func (s *sseWriter) comment() bool {
	if _, err := fmt.Fprint(s.w, ": keepalive\n\n"); err != nil {
		return false
	}
	s.flusher.Flush()
	return true
}
