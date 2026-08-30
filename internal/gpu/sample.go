// Package gpu collects per-GPU utilisation, VRAM and power from nvidia-smi and
// keeps a short rolling history of it for the dashboard and the mesh.
//
// This is one of the two genuinely vendor-specific parts of a mesh node (the
// other being how backends are spawned). Everything downstream — the mesh
// payload, the dashboard, the fleet totals — consumes meshapi.GPUInfo and does
// not care which vendor produced it.
package gpu

// GPUSample is one reading of one GPU. Field-for-field the same shape viiwork
// produces from rocm-smi, because both feed meshapi.GPUInfo and the dashboard
// renders them identically.
type GPUSample struct {
	GPUID       int     `json:"gpu_id"`
	Utilization float64 `json:"util"`
	VRAMUsedMB  float64 `json:"vram_used_mb"`
	VRAMTotalMB float64 `json:"vram_total_mb"`
	// PowerW is the board power draw. omitempty because nvidia-smi reports
	// "[N/A]" for cards whose driver does not expose it, and an absent reading
	// must stay distinguishable from a real zero — a card that is powered on
	// never draws 0 W, so a consumer can read absent as "not measured".
	//
	// Unlike the gfx906 fleet viiwork was built for, this figure is generally
	// trustworthy on NVIDIA hardware: it is a board-level measurement rather
	// than an estimate, which is why per-GPU power is worth reporting here
	// even where whole-node IPMI is unavailable.
	PowerW    float64 `json:"power_w,omitempty"`
	Timestamp int64   `json:"t"`
}
