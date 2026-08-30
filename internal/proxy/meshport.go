package proxy

import (
	"context"
	"log"
	"net"
	"net/http"
	"strconv"
	"time"
)

// meshPortRetry is how long to wait before asking for the port again.
const meshPortRetry = 15 * time.Second

// ServeMeshPort runs a second listener whose "/" is the mesh dashboard.
//
// The port is **contended, not assigned**, and that is the whole design. Every
// node on the host asks for it and the OS gives it to one; the losers retry.
// That is what makes the fleet view reachable at one fixed address per host
// without a designated instance, a per-host config or a reverse proxy: the
// port is up while *any* node on the host is, and hands over on its own when
// the holder restarts. A foreign process holding it is the same case and
// handled the same way — retry quietly, and never fail the node over a
// dashboard.
//
// Which instance answers is irrelevant, since the mesh view is assembled from
// peer state every node has. A viiwork node and a viiwork-freetoken node on the
// same host contend for it equally, and either serves the same page.
func ServeMeshPort(ctx context.Context, port int, h http.Handler) {
	if port <= 0 {
		return
	}
	addr := ":" + strconv.Itoa(port)
	for ctx.Err() == nil {
		ln, err := net.Listen("tcp", addr)
		if err != nil {
			// Almost always "address already in use", meaning another node on
			// this host is serving the page. Nothing to do but wait for it to
			// go away.
			select {
			case <-ctx.Done():
				return
			case <-time.After(meshPortRetry):
			}
			continue
		}
		log.Printf("[mesh] serving the fleet dashboard on %s", addr)
		srv := &http.Server{Handler: meshRoot(h)}
		go func() {
			<-ctx.Done()
			shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			_ = srv.Shutdown(shutdownCtx)
		}()
		if err := srv.Serve(ln); err != nil && ctx.Err() == nil {
			log.Printf("[mesh] dashboard listener stopped: %v", err)
		}
	}
}

// meshRoot rewrites "/" to "/mesh" and delegates, rather than writing the page
// itself.
//
// That is deliberate: the request still passes through the node's own handler,
// so this listener keeps CORS, panic recovery and the whole API. mesh.html
// needs that — it is not a standalone document. It opens /v1/mesh/stream and
// links rows to /prompt, all same-origin. Serving the bytes directly would
// produce a page that loads and then does nothing.
func meshRoot(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" {
			r2 := r.Clone(r.Context())
			r2.URL.Path = "/mesh"
			h.ServeHTTP(w, r2)
			return
		}
		h.ServeHTTP(w, r)
	})
}
