#!/usr/bin/env python3
"""One uncontended pass producing every number that gets published.

Earlier runs mixed figures from three scripts with different methodology; this
exists so a published table comes from one run on one machine state.

Usage: bench_final.py <served-model-name> [out.json]

Measures through the NODE (:9801), not the engine, so the numbers include what
the proxy costs. Facts come from the engine's own control plane.
"""
import json, os, random, statistics, sys, threading, time, urllib.request

NODE, ENGINE = "http://127.0.0.1:9801", "http://127.0.0.1:9901"
MODEL = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/bench-{MODEL}.json"
WORDS = open("/usr/share/dict/words").read().split() if os.path.exists("/usr/share/dict/words") else None
R = {"model": MODEL, "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

def get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)

def count_tokens(text):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": text}]}).encode()
    req = urllib.request.Request(ENGINE + "/v1/messages/count_tokens", data=body,
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.load(r).get("input_tokens")
    except Exception:
        return None

def stream(prompt, max_tokens, timeout=2400):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "stream": True, "temperature": 0.0}).encode()
    req = urllib.request.Request(NODE + "/v1/chat/completions", data=body,
                                 headers={"content-type": "application/json"})
    t0 = time.perf_counter(); ttft = None; n = 0; last = t0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                s = raw.decode("utf-8", "replace").strip()
                if not s.startswith("data: "): continue
                p = s[6:]
                if p == "[DONE]": break
                try: d = json.loads(p)
                except Exception: continue
                dl = (d.get("choices") or [{}])[0].get("delta") or {}
                c = dl.get("content") or dl.get("reasoning_content") or ""
                if c:
                    now = time.perf_counter()
                    if ttft is None: ttft = now - t0
                    n += 1; last = now
    except Exception as e:
        return {"ok": False, "err": repr(e)[:200]}
    if ttft is None: return {"ok": False, "err": "no tokens"}
    return {"ok": True, "ttft": ttft, "decode_s": max(last - (t0 + ttft), 1e-9), "n": n}

class Power:
    def __init__(self): self.v = []; self.stop = threading.Event(); self.t = None
    def __enter__(self):
        def run():
            while not self.stop.is_set():
                try:
                    w = get(NODE + "/v1/status", timeout=3).get("power_watts")
                    if w: self.v.append(w)
                except Exception: pass
                time.sleep(1.0)
        self.t = threading.Thread(target=run, daemon=True); self.t.start(); return self
    def __exit__(self, *a):
        self.stop.set(); self.t.join(timeout=3)
    def stats(self):
        return {"mean": round(statistics.mean(self.v),1), "peak": round(max(self.v),1)} if self.v else None

def filler(n_tokens, seed, tpw):
    rnd = random.Random(seed); n = max(1, int(n_tokens / tpw))
    if WORDS: return " ".join(rnd.choice(WORDS) for _ in range(n))
    return " ".join(f"tok{rnd.randrange(10**6)}" for _ in range(n))

# ── facts ────────────────────────────────────────────────────────────────────
try:
    st = get(NODE + "/v1/status"); cs = get(ENGINE + "/v1/cache/status"); mo = get(ENGINE + "/v1/models")
    R["facts"] = {
        "node_slot_ctx": st["backends"][0].get("slot_ctx"),
        "node_slot_count": st["backends"][0].get("slot_count"),
        "advertised_ctx": (mo["data"][0].get("context_length") or mo["data"][0].get("max_model_len")),
        "geometry": cs.get("geometry", {}),
        "vram_total_mb": st["gpus"][0]["vram_total_mb"], "vram_used_mb": st["gpus"][0]["vram_used_mb"],
        "host_mem_total_mb": st.get("host_mem_total_mb"), "host_mem_used_mb": st.get("host_mem_used_mb"),
    }
    g = cs.get("geometry", {})
    R["facts"]["kv_capacity_tokens"] = g.get("num_pages", 0) * g.get("page_size", 0)
except Exception as e:
    R["facts"] = {"err": repr(e)[:200]}
print(json.dumps({"facts": R["facts"]}, indent=2), flush=True)

# ── idle power (quiet first) ────────────────────────────────────────────────
time.sleep(60)
with Power() as p: time.sleep(60)
R["idle_power_w"] = p.stats()
print(json.dumps({"idle_power_w": R["idle_power_w"]}), flush=True)

# ── tokens-per-word calibration ─────────────────────────────────────────────
tpw = (count_tokens(filler(2000, "cal", 1.0)) or 2000) / 2000.0
R["tokens_per_word"] = round(tpw, 2)

# ── TTFT vs prompt length, cold and warm ────────────────────────────────────
R["prefill"] = []
for target in (64, 2000, 8000, 32000):
    prompt = (filler(target, f"pf-{target}-{time.time()}", tpw) + "\n\nReply with: ok.") if target > 200 \
             else "Write one short sentence about snow."
    actual = count_tokens(prompt)
    cold = stream(prompt, 8)
    warm = [stream(prompt, 8) for _ in range(3)]
    warm_ok = [w["ttft"] for w in warm if w["ok"]]
    row = {"target": target, "actual_tokens": actual,
           "ttft_cold_s": round(cold["ttft"], 2) if cold["ok"] else None,
           "ttft_warm_s": round(statistics.median(warm_ok), 2) if warm_ok else None}
    if cold["ok"] and actual: row["prefill_tps"] = round(actual / cold["ttft"])
    R["prefill"].append(row); print(json.dumps(row), flush=True)

# ── decode, single stream, long output ──────────────────────────────────────
ESSAY = ("Write a comprehensive technical essay of at least 900 words on how glaciers form, "
         "flow and retreat. Cover snow accumulation and firn compaction, plastic deformation "
         "and basal sliding, ablation zones, moraine deposition, and ice-albedo feedback. "
         "Write the complete essay in full prose and do not stop early.")
with Power() as p:
    runs = [stream(ESSAY, 512) for _ in range(3)]
ok = [r for r in runs if r["ok"] and r["n"] > 1]
R["decode_single"] = {
    "tokens": [r["n"] for r in ok],
    "tps_median": round(statistics.median([(r["n"]-1)/r["decode_s"] for r in ok]), 1) if ok else None,
    "ttft_s": round(statistics.median([r["ttft"] for r in ok]), 2) if ok else None,
    "power_w": p.stats()}
print(json.dumps({"decode_single": R["decode_single"]}), flush=True)

# ── concurrency ─────────────────────────────────────────────────────────────
R["concurrency"] = []
for c in (1, 2, 4, 8):
    res = []; lock = threading.Lock()
    with Power() as p:
        def w():
            r = stream(ESSAY, 384)
            with lock: res.append(r)
        t0 = time.perf_counter()
        ths = [threading.Thread(target=w) for _ in range(c)]
        [t.start() for t in ths]; [t.join() for t in ths]
        wall = time.perf_counter() - t0
    ok = [r for r in res if r["ok"] and r["n"] > 1]
    rej = [r for r in res if not r["ok"]]
    per = [(r["n"]-1)/r["decode_s"] for r in ok]
    row = {"concurrency": c, "completed": len(ok), "rejected": len(rej),
           "wall_s": round(wall,1), "tokens_total": sum(r["n"] for r in ok),
           "aggregate_decode_tps": round(sum(per),1) if per else None,
           "per_stream_tps": round(statistics.median(per),1) if per else None,
           "power_w": p.stats()}
    if per and p.stats():
        row["joules_per_token"] = round(p.stats()["mean"] / sum(per), 1)
    R["concurrency"].append(row); print(json.dumps(row), flush=True)

# ── exact context ceiling ───────────────────────────────────────────────────
cap = R["facts"].get("kv_capacity_tokens") or 0
R["context"] = {"kv_capacity_tokens": cap, "advertised": R["facts"].get("advertised_ctx")}
if cap:
    for label, target in (("just_under", int(cap * 0.98)), ("just_over", int(cap * 1.1))):
        prompt = filler(target, f"ctx-{label}-{time.time()}", tpw) + "\n\nReply with: ok."
        actual = count_tokens(prompt)
        body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": 4}).encode()
        req = urllib.request.Request(NODE + "/v1/chat/completions", data=body,
                                     headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=2400) as r:
                json.load(r); out = {"actual_tokens": actual, "status": r.status}
        except urllib.error.HTTPError as e:
            out = {"actual_tokens": actual, "status": e.code, "err": e.read().decode("utf-8","replace")[:220]}
        except Exception as e:
            out = {"actual_tokens": actual, "status": None, "err": repr(e)[:200]}
        R["context"][label] = out; print(json.dumps({label: out}), flush=True)

open(OUT, "w").write(json.dumps(R, indent=2))
print("WROTE " + OUT, flush=True)
