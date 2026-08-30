package freetoken

import "testing"

// Shape of FreeToken's /v1/stats for a dense-attention model, trimmed. Real
// output also carries the model card, gpus[] and lifetime token counters.
const statsDense = `{
  "instance_id": "3f1a-...",
  "model": {"id": "dsv4-flash"},
  "uptime_s": 942,
  "kv": {"used_pages": 310, "total_pages": 500, "page_size": 128},
  "mamba": null,
  "swa": null,
  "vram_bytes": 29456789504,
  "gpus": [{"index": 0, "name": "NVIDIA GeForce RTX 5090", "total_bytes": 34179383296}],
  "throughput": {"decode_tps": 41.5, "prefill_tps": 1180.2},
  "requests": {"active": 3, "completed": 187, "p95_ms": 8100, "ttft_mean_ms": 620,
               "prompt_tokens_total": 918273, "completion_tokens_total": 55120}
}`

func TestParseStatsDense(t *testing.T) {
	got, ok := ParseStats([]byte(statsDense))
	if !ok {
		t.Fatal("well-formed stats must parse")
	}
	if got.NumRunning != 3 {
		t.Errorf("running: got %d, want 3", got.NumRunning)
	}
	if got.CachePct != 0.62 {
		t.Errorf("cache: got %v, want 310/500", got.CachePct)
	}
	if got.DecodeTPS != 41.5 || got.VRAMBytes != 29456789504 {
		t.Errorf("throughput/vram not read: %+v", got)
	}
	for _, k := range []string{"running", "kv", "cache", "decode_tps", "vram"} {
		if !got.Found[k] {
			t.Errorf("%s not marked found", k)
		}
	}
}

// Which cache pools exist depends on the model family, and the engine sends
// null for the ones a model does not use. A sliding-window model reports swa
// and a null kv; reading only kv would report a permanent zero for it, which
// is the difference between "idle" and "about to start evicting".
func TestCachePressureFollowsWhicheverPoolExists(t *testing.T) {
	for _, tc := range []struct {
		name, body string
		want       float64
		found      string
	}{
		{"sliding window", `{"kv": null, "swa": {"used_pages": 9, "total_pages": 10, "page_size": 1}}`, 0.9, "swa"},
		{"linear attention", `{"kv": null, "mamba": {"used_slots": 1, "total_slots": 4}}`, 0.25, "mamba"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := ParseStats([]byte(tc.body))
			if !ok {
				t.Fatal("must parse")
			}
			if got.CachePct != tc.want {
				t.Errorf("cache: got %v, want %v", got.CachePct, tc.want)
			}
			if !got.Found[tc.found] || !got.Found["cache"] {
				t.Errorf("%s pool not marked found: %+v", tc.found, got.Found)
			}
		})
	}
}

// A hybrid model reports several pools at once. Pressure is the worst of them:
// the pool closest to full is the one that will make the backend slow.
func TestCachePressureTakesTheFullestPool(t *testing.T) {
	got, _ := ParseStats([]byte(`{"kv": {"used_pages": 1, "total_pages": 10},
	                              "mamba": {"used_slots": 7, "total_slots": 8}}`))
	if got.CachePct != 0.875 {
		t.Errorf("got %v, want the mamba pool's 7/8", got.CachePct)
	}
}

// Zero running requests and a pool this model does not have are different
// facts, and the mesh payload must be able to tell them apart.
func TestFoundDistinguishesZeroFromAbsent(t *testing.T) {
	got, ok := ParseStats([]byte(`{"requests": {"active": 0}, "kv": null, "swa": null, "mamba": null}`))
	if !ok {
		t.Fatal("must parse")
	}
	if got.NumRunning != 0 || !got.Found["running"] {
		t.Errorf("explicit zero must be found: %+v", got)
	}
	if got.Found["cache"] || got.Found["kv"] {
		t.Errorf("a model with no pools must not report cache occupancy: %+v", got.Found)
	}
}

// A pool present but sized zero is not a division to attempt.
func TestEmptyPoolIsNotOccupancy(t *testing.T) {
	got, _ := ParseStats([]byte(`{"kv": {"used_pages": 0, "total_pages": 0}}`))
	if got.Found["cache"] || got.CachePct != 0 {
		t.Errorf("got %+v", got)
	}
}

func TestParseStatsRejectsGarbage(t *testing.T) {
	if _, ok := ParseStats([]byte("not json")); ok {
		t.Error("a non-JSON body must not parse as stats")
	}
}

// --- /health -------------------------------------------------------------
//
// The engine answers 200 in every lifecycle state, so readiness is a field
// rather than a status code. These are the states it distinguishes.

func TestHealthReadiness(t *testing.T) {
	for _, tc := range []struct {
		name, body string
		ready      bool
		describe   string
	}{
		{
			"serving",
			`{"status":"ok","model":"m","uptime_s":10,"maintenance":"serving","version":"0.4.1"}`,
			true, "",
		},
		{
			// The trap in porting the vLLM node's probe: 200 OK, and every
			// generation request would come back 503.
			"loading",
			`{"status":"loading","phase":"weights","progress":{"done_bytes":25,"total_bytes":100},"model":"m"}`,
			false, "loading (weights, 25%)",
		},
		{
			// A live cache resize takes the engine out of service without
			// changing status. Also 200, also 503s every request.
			"rebuilding",
			`{"status":"ok","model":"m","maintenance":"rebuilding"}`,
			false, "maintenance: rebuilding",
		},
		{
			"failed",
			`{"status":"error","message":"worker died"}`,
			false, "failed: worker died",
		},
		{
			// An older build may omit maintenance from the ready shape.
			"ok without maintenance",
			`{"status":"ok","model":"m"}`,
			true, "",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h, ok := ParseHealth([]byte(tc.body))
			if !ok {
				t.Fatal("must parse")
			}
			if h.Ready() != tc.ready {
				t.Errorf("ready = %v, want %v", h.Ready(), tc.ready)
			}
			if got := h.Describe(); got != tc.describe {
				t.Errorf("describe = %q, want %q", got, tc.describe)
			}
		})
	}
}

// Something listening on the backend's port that is not the engine we spawned
// is a failed probe, not a healthy backend with an odd payload.
func TestParseHealthRejectsNonEngineBodies(t *testing.T) {
	for _, body := range []string{`{"ok": true}`, `<html>nginx</html>`, ``} {
		if _, ok := ParseHealth([]byte(body)); ok {
			t.Errorf("%q must not parse as a health document", body)
		}
	}
}

// Loading without a byte total still has to say something: the phase alone is
// the difference between a backend that is stuck and one that has moved on.
func TestLoadingWithoutProgressStillDescribesPhase(t *testing.T) {
	h, _ := ParseHealth([]byte(`{"status":"loading","phase":"cuda_graphs"}`))
	if got := h.Describe(); got != "loading (cuda_graphs)" {
		t.Errorf("got %q", got)
	}
}
