package proxy

import (
	"bufio"
	"encoding/json"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/janit/viiwork/meshapi"
)

func (h *Handler) handleStatus(w http.ResponseWriter, r *http.Request) {
	total, used := readHostMemory()
	resp := h.registry.StatusResponse(h.gpuInfo(), total, used)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (h *Handler) handleCluster(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(h.clusterState())
}

// clusterState assembles the snapshot, adding the local-only extras the
// registry has no access to. Shared by the /v1/cluster endpoint and the mesh
// push stream so the two cannot drift.
func (h *Handler) clusterState() meshapi.ClusterResponse {
	state := h.registry.ClusterState()
	state.Version = Version
	state.Local.GPUs = h.gpuInfo()
	state.Local.HostMemTotalMB, state.Local.HostMemUsedMB = readHostMemory()
	return state
}

// gpuInfo maps the latest samples onto the wire type.
//
// This publishes every GPU nvidia-smi reports, not only the cards this
// instance drives. On a host running several instances that means each one
// publishes the same full list, which is why a consumer must key by hostname
// and gpu_id rather than summing as the payloads arrive — otherwise a
// three-instance host reports its cards three times over.
func (h *Handler) gpuInfo() []meshapi.GPUInfo {
	if h.gpuHist == nil {
		return nil
	}
	var out []meshapi.GPUInfo
	for _, s := range h.gpuHist.Latest() {
		out = append(out, meshapi.GPUInfo{
			GPUID: s.GPUID, Util: s.Utilization,
			VRAMUsedMB: s.VRAMUsedMB, VRAMTotalMB: s.VRAMTotalMB,
		})
	}
	return out
}

// readHostMemory reads /proc/meminfo and returns total and used memory in MB.
// Used is MemTotal - MemAvailable, so reclaimable page cache is not counted as
// pressure — on an inference host the cache is usually model weights that the
// kernel will drop on demand, and counting it would show every host at 100%.
func readHostMemory() (totalMB, usedMB int64) {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0, 0
	}
	defer f.Close()

	var memTotal, memAvailable int64
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "MemTotal:") {
			memTotal = parseMemInfoKB(line)
		} else if strings.HasPrefix(line, "MemAvailable:") {
			memAvailable = parseMemInfoKB(line)
		}
		if memTotal > 0 && memAvailable > 0 {
			break
		}
	}
	return memTotal / 1024, (memTotal - memAvailable) / 1024
}

func parseMemInfoKB(line string) int64 {
	fields := strings.Fields(line)
	if len(fields) < 2 {
		return 0
	}
	v, _ := strconv.ParseInt(fields[1], 10, 64)
	return v
}

func (h *Handler) handleMetrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if h.gpuHist == nil || h.gpuAvail == nil || !h.gpuAvail() {
		json.NewEncoder(w).Encode(map[string]any{"available": false})
		return
	}
	json.NewEncoder(w).Encode(map[string]any{
		"available": true,
		"series":    h.gpuHist.Series(),
	})
}
