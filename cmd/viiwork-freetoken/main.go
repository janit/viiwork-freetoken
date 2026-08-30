// Command viiwork-freetoken is a viiwork mesh node backed by FreeToken on
// NVIDIA hardware.
//
// It speaks the same protocol as viiwork (see github.com/janit/viiwork/meshapi)
// and serves the same dashboards, so a consumer GPU running a frontier MoE
// model out of host memory appears in the fleet view alongside the Radeon VII
// machines and the vLLM nodes, and takes routed requests from them.
package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/janit/viiwork-freetoken/internal/activity"
	"github.com/janit/viiwork-freetoken/internal/balancer"
	"github.com/janit/viiwork-freetoken/internal/config"
	"github.com/janit/viiwork-freetoken/internal/freetoken"
	"github.com/janit/viiwork-freetoken/internal/gpu"
	"github.com/janit/viiwork-freetoken/internal/peer"
	"github.com/janit/viiwork-freetoken/internal/power"
	"github.com/janit/viiwork-freetoken/internal/proxy"
	"github.com/janit/viiwork/energy"
	"github.com/janit/viiwork/meshapi"
)

// version is injected at build time with -ldflags "-X main.version=...".
var version = "dev"

func main() {
	cfgPath := flag.String("config", "viiwork-freetoken.yaml", "path to the config file")
	showVersion := flag.Bool("version", false, "print the version and exit")
	flag.Parse()

	if *showVersion {
		fmt.Println(version)
		return
	}
	proxy.Version = version
	log.Printf("viiwork-freetoken %s starting", version)

	cfg, err := config.Load(*cfgPath)
	if err != nil {
		log.Fatalf("config: %v", err)
	}
	// Positional args after the flags are dotpath overrides, matching
	// viiwork's convention so both node types are driven the same way from a
	// compose file.
	if err := cfg.ApplyOverrides(flag.Args()); err != nil {
		log.Fatalf("config override: %v", err)
	}
	if err := cfg.Validate(); err != nil {
		log.Fatalf("config: %v", err)
	}

	hostname := cfg.Node.Hostname
	if hostname == "" {
		if hostname, err = os.Hostname(); err != nil {
			log.Fatalf("could not determine hostname (set node.hostname): %v", err)
		}
	}
	nodeID := cfg.Node.ID
	if nodeID == "" {
		nodeID = mintNodeID()
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	// --- observability ---------------------------------------------------
	events := activity.NewLog(cfg.Activity.EventHistory)
	prompts := activity.NewPromptStore(cfg.Activity.PromptHistory)

	gpuHistory := gpu.NewHistory()
	gpuCast := gpu.NewBroadcaster()
	collector := gpu.NewStatCollector(gpuHistory, gpuCast)
	go collector.Run(ctx, cfg.Health.Interval.Duration)

	// --- backends --------------------------------------------------------
	manager, err := freetoken.NewManager(cfg, events)
	if err != nil {
		log.Fatalf("backends: %v", err)
	}
	states := manager.States()
	bal := balancer.New(states, cfg.Balancer.HighLoadThreshold, cfg.Balancer.MaxInFlightPerBackend)

	manager.Start(ctx)
	go manager.Run(ctx)

	// --- mesh ------------------------------------------------------------
	var peers []*peer.PeerState
	for _, addr := range cfg.Peers.Hosts {
		peers = append(peers, peer.NewPeerState(addr))
	}
	registry := peer.NewRegistry(nodeID, cfg.Model.Name, states, peers, cfg.Peers.Timeout.Duration)
	registry.SetLocation(hostname, listenAddrFor(hostname, cfg.Server.Port))
	registry.SetPromptHistory(prompts.Max())
	registry.SetVersion(version)

	// --- power and energy ------------------------------------------------
	//
	// Node wattage here is the sum of per-GPU board power, not a chassis
	// reading: there is no uniform BMC across the NVIDIA hosts. The Source
	// label is what keeps that honest downstream — see internal/power.
	nodeWatts := power.NewSum(gpuHistory, collector.PowerAvailable)
	registry.SetPowerReader(nodeWatts)

	go registry.Run(ctx, cfg.Peers.PollInterval.Duration)

	if cfg.Energy.Enabled {
		if err := startEnergy(ctx, cfg, registry, gpuHistory, nodeWatts); err != nil {
			// Not fatal: a node that cannot record kWh should still serve
			// inference. The mesh reads the absent field as unknown.
			log.Printf("[energy] disabled: %v", err)
		}
	}

	handler := proxy.NewHandler(registry, bal, events, prompts, cfg.Balancer.MaxInFlightPerBackend)
	handler.SetGPUSource(gpuHistory, gpuCast, collector.Available)
	if len(cfg.Server.CORS.AllowOrigins) > 0 {
		tailnet := cfg.Server.CORS.AllowTailnetIPs == nil || *cfg.Server.CORS.AllowTailnetIPs
		handler.SetCORS(&proxy.CORS{Origins: cfg.Server.CORS.AllowOrigins, TailnetIPs: tailnet})
		log.Printf("CORS enabled for origins %v (tailnet IPs: %v)", cfg.Server.CORS.AllowOrigins, tailnet)
	}

	// The fleet dashboard on a fixed per-host port, contended with any other
	// node on this machine. See proxy.ServeMeshPort.
	go proxy.ServeMeshPort(ctx, cfg.Server.MeshPort, handler)

	addr := cfg.Server.Host + ":" + strconv.Itoa(cfg.Server.Port)
	srv := &http.Server{
		Addr:    addr,
		Handler: handler,
		// No WriteTimeout: a streamed generation legitimately runs for
		// minutes, and a write deadline would sever it mid-response. Read
		// timeouts are safe because a request body arrives promptly.
		ReadHeaderTimeout: 10 * time.Second,
	}

	events.Emit(meshapi.EventSystem, -1, "node %s starting on %s (model %s, %d backends)",
		hostname, addr, cfg.Model.Name, len(states))
	log.Printf("serving %s on %s (%d backends, mesh dashboard on :%d)",
		cfg.Model.Name, addr, len(states), cfg.Server.MeshPort)

	serverErr := make(chan error, 1)
	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			serverErr <- err
		}
	}()

	select {
	case err := <-serverErr:
		log.Printf("listener failed: %v", err)
	case <-ctx.Done():
		log.Println("shutting down")
	}

	// Stop accepting first, then stop the backends: a request already in
	// flight should finish against a live backend rather than be cut off by
	// its process disappearing underneath it.
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	_ = srv.Shutdown(shutdownCtx)
	manager.Shutdown()
	log.Println("stopped")
}

// listenAddrFor is the address peers should dial this node on. The configured
// bind host is usually 0.0.0.0, which is meaningless to a peer, so the
// resolvable hostname is published instead.
func listenAddrFor(hostname string, port int) string {
	return hostname + ":" + strconv.Itoa(port)
}

// mintNodeID generates a node id. Nothing durable is keyed on it, so a fresh
// one per start is fine; pin node.id in the config if you would rather it
// survived restarts.
func mintNodeID() string {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "viiwork-freetoken-" + strconv.FormatInt(time.Now().UnixNano(), 16)
	}
	return "viiwork-freetoken-" + strings.ToLower(hex.EncodeToString(b[:]))
}

// startEnergy opens the durable kWh store and starts recording.
//
// The store, the ring format and all the bucketing live in viiwork's energy
// package rather than being reimplemented here: the format is shared across
// both implementations of this mesh, and two independent producers of one
// binary layout would drift. What this node supplies is only the two seams that
// package is built around, plus the one decision that genuinely differs.
//
// That decision is energy.Direct. viiwork measures the chassis and has to
// divide one figure between the models causing it and a baseline drawn anyway.
// Here the node figure IS the sum of the per-GPU readings, so there is no
// aggregate to divide: each card is charged its own measured draw, the shares
// reconcile with the node total exactly, and the baseline is honestly zero.
// Running the chassis model over this producer would attribute 0W to every card
// until a genuinely idle minute is observed — which a busy node may not see for
// weeks. See energy.Direct.
func startEnergy(
	ctx context.Context,
	cfg *config.Config,
	registry *peer.Registry,
	gpuHistory *gpu.History,
	nodeWatts *power.Sum,
) error {
	// Every card the host reports, not just this instance's slice: the store
	// covers the machine, and node wattage is a per-host measurement. This is
	// also why energy.enabled belongs on exactly one instance per host.
	var gpuIDs []int
	for _, s := range gpuHistory.Latest() {
		gpuIDs = append(gpuIDs, s.GPUID)
	}
	// At startup the collector may not have sampled yet, so fall back to the
	// configured devices rather than opening a store with no lanes.
	if len(gpuIDs) == 0 {
		gpuIDs = cfg.GPUs.Devices
	}

	store, err := energy.Open(energy.Config{
		Dir:         cfg.Energy.Dir,
		GPUIDs:      gpuIDs,
		MinuteSlots: cfg.Energy.MinuteSlots,
		HourSlots:   cfg.Energy.HourSlots,
		DaySlots:    cfg.Energy.DaySlots,
		// Recorded beside the data because the bytes are identical whichever
		// was measured, and a directory copied off the host for inspection
		// carries no mesh context to say which.
		Source: power.SourceLabel,
	}, nil)
	if err != nil {
		return err
	}

	// Resolved per sample rather than once: peers may not have been polled at
	// startup, and a host's GPU-to-model layout changes as instances come and
	// go.
	readings := func() []energy.GPUReading {
		return power.Readings(gpuHistory, registry.GPUModels())
	}

	recorder := energy.NewRecorderWithAttribution(
		store, cfg.Energy.SampleInterval.Duration,
		nodeWatts.NodeWatts, readings, energy.Direct, nil,
	)
	registry.SetEnergyReader(store)

	go func() {
		recorder.Run(ctx)
		store.Close()
	}()

	log.Printf("[energy] recording every %s for %d GPUs into %s (source %q; node wattage is per-host — enable this on one instance per host)",
		cfg.Energy.SampleInterval.Duration, len(gpuIDs), cfg.Energy.Dir, power.SourceLabel)
	return nil
}
