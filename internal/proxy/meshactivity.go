package proxy

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/janit/viiwork-freetoken/internal/peer"
	"github.com/janit/viiwork/meshapi"
)

// snapshotInterval is how often the cluster snapshot is re-evaluated. The
// browser never polls: snapshots are pushed, and only when something in them
// changed.
const snapshotInterval = time.Second

// keepaliveInterval keeps an idle stream from being reaped by an intermediary.
const keepaliveInterval = 20 * time.Second

// hostMemBuckets and the deadband coarsen host memory before it is published
// on the push stream.
//
// An exact figure moves every second on a live host and defeats change
// detection outright, pushing a full snapshot on every tick for a value that
// renders into a strip a few pixels tall. Plain rounding is not enough: a host
// sitting astride a bucket boundary flips on every sample and pushes just as
// hard. The deadband makes the published value hold until the reading has
// drifted a full step away from it. /v1/cluster keeps the exact figures.
const hostMemBuckets = 64

func coarsenHostMem(prev, total, used int64) int64 {
	if total <= 0 {
		return used
	}
	step := total / hostMemBuckets
	if step <= 0 {
		return used
	}
	if prev != 0 && absInt64(used-prev) < step {
		return prev
	}
	return (used / step) * step
}

func absInt64(v int64) int64 {
	if v < 0 {
		return -v
	}
	return v
}

func (h *Handler) handleActivity(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(h.activity.Recent())
}

// handleActivityStream is this node's own event feed. Peers subscribe to it to
// build their merged mesh streams.
func (h *Handler) handleActivityStream(w http.ResponseWriter, r *http.Request) {
	sse, ok := newSSEWriter(w)
	if !ok {
		return
	}
	// Subscribe BEFORE reading the backlog, or an event landing between the
	// two is delivered by neither and strands an in-flight row forever.
	ch := h.activity.Subscribe()
	defer h.activity.Unsubscribe(ch)
	for _, e := range h.activity.Backlog() {
		if !sse.send("", e) {
			return
		}
	}
	keepalive := time.NewTicker(keepaliveInterval)
	defer keepalive.Stop()
	for {
		select {
		case <-r.Context().Done():
			return
		case e, open := <-ch:
			if !open || !sse.send("", e) {
				return
			}
		case <-keepalive.C:
			if !sse.comment() {
				return
			}
		}
	}
}

func (h *Handler) handleMetricsStream(w http.ResponseWriter, r *http.Request) {
	if h.gpuCast == nil {
		http.NotFound(w, r)
		return
	}
	sse, ok := newSSEWriter(w)
	if !ok {
		return
	}
	ch := h.gpuCast.Subscribe()
	defer h.gpuCast.Unsubscribe(ch)
	keepalive := time.NewTicker(keepaliveInterval)
	defer keepalive.Stop()
	for {
		select {
		case <-r.Context().Done():
			return
		case s, open := <-ch:
			if !open || !sse.send("", s) {
				return
			}
		case <-keepalive.C:
			if !sse.comment() {
				return
			}
		}
	}
}

// handleMeshStream is the dashboard's single connection.
//
// Two named events ride on it: "activity" (a MeshEvent, tagged with the node
// it came from) and "cluster" (a full snapshot, pushed only when it changed).
//
// Fan-out happens here, on the server, rather than in the browser. It has to:
// the viewer may not be able to reach peers directly — they are often LAN
// addresses reachable only through the node being viewed — and EventSource is
// CORS-bound in any case. One reachable node is therefore enough to watch the
// whole mesh.
func (h *Handler) handleMeshStream(w http.ResponseWriter, r *http.Request) {
	sse, ok := newSSEWriter(w)
	if !ok {
		return
	}
	ctx := r.Context()

	// One goroutine per source writes into this channel; the loop below is the
	// only writer to the response, so no locking is needed around the SSE
	// writer itself.
	events := make(chan meshapi.MeshEvent, 512)

	localID, localHost := h.registry.NodeID(), h.registry.Hostname()
	local := h.activity.Subscribe()
	defer h.activity.Unsubscribe(local)
	go func() {
		for e := range local {
			select {
			case events <- meshapi.MeshEvent{Event: e, NodeID: localID, Hostname: localHost}:
			case <-ctx.Done():
				return
			}
		}
	}()

	for _, p := range h.registry.Peers() {
		go h.followPeer(ctx, p, events)
	}

	// The local backlog replays immediately. Peer backlogs arrive on their
	// own, since each peer's activity stream replays its ring when we connect.
	for _, e := range h.activity.Backlog() {
		if !sse.send(meshapi.SSEActivity, meshapi.MeshEvent{Event: e, NodeID: localID, Hostname: localHost}) {
			return
		}
	}

	var lastSnapshot []byte
	var lastMem int64
	pushSnapshot := func() bool {
		state := h.clusterState()
		lastMem = coarsenHostMem(lastMem, state.Local.HostMemTotalMB, state.Local.HostMemUsedMB)
		state.Local.HostMemUsedMB = lastMem
		encoded, err := json.Marshal(state)
		if err != nil {
			return true
		}
		if string(encoded) == string(lastSnapshot) {
			return true
		}
		lastSnapshot = encoded
		return sse.send(meshapi.SSECluster, state)
	}
	if !pushSnapshot() {
		return
	}

	snapshots := time.NewTicker(snapshotInterval)
	defer snapshots.Stop()
	keepalive := time.NewTicker(keepaliveInterval)
	defer keepalive.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case e := <-events:
			if !sse.send(meshapi.SSEActivity, e) {
				return
			}
		case <-snapshots.C:
			if !pushSnapshot() {
				return
			}
		case <-keepalive.C:
			if !sse.comment() {
				return
			}
		}
	}
}

// followPeer subscribes to one peer's activity stream and forwards its events,
// tagged with that peer's identity. It reconnects with a fixed delay for as
// long as the viewer's connection lives: a peer that is down now is expected
// back, and the dashboard should pick it up without a page reload.
func (h *Handler) followPeer(ctx context.Context, p *peer.PeerState, out chan<- meshapi.MeshEvent) {
	for ctx.Err() == nil {
		h.streamPeerOnce(ctx, p, out)
		select {
		case <-ctx.Done():
			return
		case <-time.After(2 * time.Second):
		}
	}
}

func (h *Handler) streamPeerOnce(ctx context.Context, p *peer.PeerState, out chan<- meshapi.MeshEvent) {
	req, err := http.NewRequestWithContext(ctx, "GET", "http://"+p.Addr+meshapi.PathActivityStream, nil)
	if err != nil {
		return
	}
	resp, err := peerClient.Do(req)
	if err != nil {
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return
	}

	reader := newSSEReader(resp.Body)
	for {
		data, err := reader.next()
		if err != nil {
			if err != io.EOF && ctx.Err() == nil {
				log.Printf("[mesh] peer %s activity stream ended: %v", p.Addr, err)
			}
			return
		}
		var e meshapi.Event
		if err := json.Unmarshal(data, &e); err != nil {
			continue
		}
		host := p.Hostname()
		if host == "" {
			host = hostOfAddr(p.Addr)
		}
		select {
		case out <- meshapi.MeshEvent{Event: e, NodeID: p.NodeID(), Hostname: host, Addr: p.Addr}:
		case <-ctx.Done():
			return
		}
	}
}

func hostOfAddr(addr string) string {
	if i := strings.LastIndex(addr, ":"); i >= 0 {
		return addr[:i]
	}
	return addr
}

// sseReader pulls `data:` payloads out of an event stream. Deliberately
// minimal: it handles the single-line data frames these streams emit and skips
// comments and event names, which the per-node activity feed does not use.
type sseReader struct {
	br *bufio.Reader
}

func newSSEReader(r io.Reader) *sseReader {
	// A generous buffer: one event is a JSON object with a message field, and
	// a long model name plus a peer address is still far under this.
	return &sseReader{br: bufio.NewReaderSize(r, 64*1024)}
}

func (s *sseReader) next() ([]byte, error) {
	for {
		line, err := s.br.ReadString('\n')
		if err != nil {
			return nil, err
		}
		line = strings.TrimRight(line, "\r\n")
		if strings.HasPrefix(line, "data: ") {
			return []byte(strings.TrimPrefix(line, "data: ")), nil
		}
	}
}
