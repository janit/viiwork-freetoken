package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"strconv"

	"github.com/janit/viiwork/meshapi"
)

// handlePromptLookup serves this node's own stored prompt and output for a
// request id. It is also what a peer's mesh lookup proxies to: request ids are
// a per-process counter, not cluster-wide, so a lookup only ever makes sense
// against the node that minted it.
func (h *Handler) handlePromptLookup(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	rid, err := strconv.ParseInt(r.URL.Query().Get("rid"), 10, 64)
	if err != nil || h.prompts == nil {
		http.NotFound(w, r)
		return
	}
	entry, ok := h.prompts.Get(rid)
	if !ok {
		http.NotFound(w, r)
		return
	}
	json.NewEncoder(w).Encode(entry)
}

// handleMeshPrompt forwards a lookup to the node that owns the request id.
//
// The addr check is load bearing, not defensive tidiness: without it this
// endpoint fetches whatever address it is handed and echoes the response back
// to the caller, which is an SSRF primitive on a LAN that also carries IPMI
// and other unauthenticated management interfaces. Only addresses already in
// this node's peer list are dialled.
func (h *Handler) handleMeshPrompt(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	q := r.URL.Query()
	rid, err := strconv.ParseInt(q.Get("rid"), 10, 64)
	if err != nil {
		http.Error(w, `{"error":{"message":"bad rid","type":"invalid_request"}}`, http.StatusBadRequest)
		return
	}
	addr := q.Get("addr")
	// No addr means the request was minted here.
	if addr == "" || addr == h.registry.ListenAddr() {
		h.handlePromptLookup(w, r)
		return
	}
	if !h.registry.IsKnownAddr(addr) {
		http.Error(w, `{"error":{"message":"unknown peer address","type":"forbidden"}}`, http.StatusForbidden)
		return
	}

	target := "http://" + addr + meshapi.PathPrompts + "?rid=" + url.QueryEscape(strconv.FormatInt(rid, 10))
	req, err := http.NewRequestWithContext(r.Context(), "GET", target, nil)
	if err != nil {
		http.Error(w, `{"error":{"message":"bad peer request","type":"internal"}}`, http.StatusInternalServerError)
		return
	}
	resp, err := peerClient.Do(req)
	if err != nil {
		http.Error(w, `{"error":{"message":"peer unreachable","type":"unavailable"}}`, http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()
	w.WriteHeader(resp.StatusCode)
	// Bounded: prompt and output are capped at the source, but this node is
	// echoing another's response and should not trust it to be small.
	io.Copy(w, io.LimitReader(resp.Body, 8<<20))
}
