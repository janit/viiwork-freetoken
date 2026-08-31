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

// --- /v1/models ----------------------------------------------------------

// Captured from the live engine: FreeToken 0.1.2 serving DeepSeek-V4-Flash.
const liveModels = `{"object":"list","data":[{"id":"DeepSeek-V4-Flash","object":"model","created":1788116321,"owned_by":"FreeToken","root":"/srv/models/DeepSeek-V4-Flash-0731","max_model_len":1048576,"context_length":1048576,"supported_reasoning_efforts":["max","high","low"],"default_reasoning_effort":"low"}]}`

func TestParseContextLen(t *testing.T) {
	got, ok := ParseContextLen([]byte(liveModels))
	if !ok || got != 1048576 {
		t.Errorf("got %d, %v — want the engine's own context length", got, ok)
	}
}

// The engine emits both spellings; a build that drops either must not silently
// produce a blank context on the dashboard.
func TestParseContextLenAcceptsEitherSpelling(t *testing.T) {
	for _, body := range []string{
		`{"data":[{"context_length":32768}]}`,
		`{"data":[{"max_model_len":32768}]}`,
	} {
		if got, ok := ParseContextLen([]byte(body)); !ok || got != 32768 {
			t.Errorf("%s: got %d, %v", body, got, ok)
		}
	}
}

func TestParseContextLenAbsent(t *testing.T) {
	for _, body := range []string{`{"data":[]}`, `{"data":[{"id":"m"}]}`, `nonsense`} {
		if _, ok := ParseContextLen([]byte(body)); ok {
			t.Errorf("%s must not yield a context length", body)
		}
	}
}

// --- recorded from a live engine -----------------------------------------
//
// Captured verbatim from FreeToken 0.1.2 serving DeepSeek-V4-Flash on an RTX
// 5090. Hand-written fixtures encode what you believe the engine emits; these
// encode what it does. Worth keeping precisely because this model contradicted
// an easy assumption: it reports kv AND swa pools at once, so a reader that
// looked only at kv would have shown a permanent zero for the one model this
// node exists to serve.

const liveHealth = `{"status":"ok","model":"DeepSeek-V4-Flash","instance_id":"2d5e3277-a6af-4b89-af99-d2bae5e95375","uptime_s":27261,"maintenance":"serving","version":"0.1.2"}`

const liveStats = `{"instance_id":"2d5e3277-a6af-4b89-af99-d2bae5e95375","model":{"id":"DeepSeek-V4-Flash","ctx":1048576,"attn":"mha","moe":true,"sampling":{"temperature":1.0,"top_p":1.0}},"uptime_s":27261,"kv":{"used_pages":0,"total_pages":501,"page_size":1},"mamba":null,"swa":{"used_pages":0,"total_pages":100,"page_size":128},"vram_bytes":31859933184,"throughput":{"decode_tps":0.0,"prefill_tps":0.0},"requests":{"active":0,"completed":34,"p95_ms":104563,"ttft_mean_ms":5969,"prompt_tokens_total":1134981,"completion_tokens_total":18485}}`

func TestLiveEngineDocuments(t *testing.T) {
	h, ok := ParseHealth([]byte(liveHealth))
	if !ok {
		t.Fatal("the live health document must parse")
	}
	if !h.Ready() {
		t.Errorf("a serving engine must read as ready: %+v", h)
	}

	s, ok := ParseStats([]byte(liveStats))
	if !ok {
		t.Fatal("the live stats document must parse")
	}
	// Both pools at once. This is the case the multi-pool maximum exists for.
	if !s.Found["kv"] || !s.Found["swa"] {
		t.Errorf("DSV4 reports both kv and swa: %v", s.Found)
	}
	if s.Found["mamba"] {
		t.Error("mamba is null here and must not be marked found")
	}
	if s.VRAMBytes != 31859933184 {
		t.Errorf("vram: got %d", s.VRAMBytes)
	}
}

// Recorded from the live engine's /v1/cache/status. The 64,128 this yields is
// the exact figure the backend names when it refuses a longer prompt:
// "prompt is too long: 70175 tokens > 64128 maximum (prompt + generation)".
const liveCacheStatus = `{"state":"serving","last_rebuild":null,"geometry":{"num_pages":501,"page_size":128,"moe_cache_size":1239,"num_mamba_slots":0,"num_experts":256,"num_moe_layers":43,"moe_cache_policy":"lru","unit_bytes":{"kv_per_token":6888,"moe_per_expert":13369344,"mamba_per_slot":0,"swa_per_token":139392},"swa_full_tokens_ratio":0.2,"swa_page_size":128,"num_swa_pages":100,"cache_budget_bytes":19302311526}}`

func TestLiveKVCapacity(t *testing.T) {
	got, ok := ParseKVCapacity([]byte(liveCacheStatus))
	if !ok || got != 64128 {
		t.Errorf("got %d, %v — want the 64128 the engine enforces", got, ok)
	}
	// And it must be the smaller of the two: /v1/models on the same engine
	// advertises 1,048,576, sixteen times what the backend accepts.
	adv, _ := ParseContextLen([]byte(liveModels))
	if adv <= got {
		t.Errorf("this test is pointless unless the advertised %d exceeds the servable %d", adv, got)
	}
}
