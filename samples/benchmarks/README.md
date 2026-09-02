# Benchmarks: frontier MoE models on one RTX 5090

What a 32 GB consumer card does when the model does not fit in it, measured on
one machine so the numbers are comparable. Raw output is in
[`results/`](results/); the method is at the bottom, including what was thrown
away.

## Tested models

Everything below was run end to end on the hardware in the next section, served
through the node. `auto` is what `--moe-backend` resolved to on this host — it
is a property of the machine, not of the model, so [re-run `ft bench
bw`](#the-bench-profile-predicts-this-before-you-download-anything) on yours.

| Model | Checkpoint | On disk | Needs | MoE backend | Servable context | Decode 1 / 4 conc |
| --- | --- | ---: | --- | --- | ---: | ---: |
| **DeepSeek-V4-Flash** | [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | 156 GB | FreeToken 0.1.2 | `hybrid` | 64,128 | 24.5 / 51.1 tok/s |
| **Qwen3.8-Flash-Next** NVFP4 | [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) | 126 GB | main build † | `hybrid` | 8,192 → 262,144 ‡ | 53.8 / 147 tok/s |
| Qwen3.8-Flash-Next FP8 | [`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) | 173 GB | main build † | `offload` only | 8,320 | 34.4 / 63.9 tok/s |

† `qwen4_exp` is not in the 0.1.2 release on PyPI. A source build of `main` is
required, and the two can live in separate virtualenvs — the node picks between
them with `freetoken.binary`.

‡ Reconfigurable at runtime; see [Context is bought with expert
cache](#context-is-bought-with-expert-cache-and-it-is-cheap). The 262,144 figure
is the checkpoint's full advertised window, reached for an 11% throughput cost.

### What decides whether a checkpoint will load at all

Disk size is the number people check and it is the wrong one. The experts live
in host RAM, but everything that is *not* an expert has to be resident on the
card, and on these checkpoints that set is much larger than it looks:

| | dense, GPU-resident | experts + tables, host RAM |
| --- | ---: | ---: |
| Qwen3.8-Flash-Next NVFP4 | **16.0 GB** | 68.0 GB experts + 51.2 GB PLE |

16.0 GB is not a small residue of a 126 GB checkpoint — it is the whole reason
this model needs a 32 GB card rather than a 16 GB one. The NVFP4 quant config
leaves `*.self_attn.*`, `*.linear_attn.*` and `*.ple.*` in bf16 and the
248,320-entry vocab head is untied, so the unquantised part is most of it.

On a 10 GiB RTX 3080 this fails as `torch.OutOfMemoryError` inside
`load_state_dict`, before the KV pool or the expert cache are sized at all —
which means **no `--moe-cache-size`, `--moe-cache-rate`, `--memory-ratio` or
`--moe-backend` setting can rescue it.** Those trade against the pools. The
weights are not negotiable.

You can compute it from the repo listing before downloading anything, because
the shards are named by role:

```sh
# sums the non-expert, non-PLE shards — what has to fit on the card
curl -s https://huggingface.co/api/models/$REPO/tree/main \
  | python3 -c 'import json,sys; print(sum((f.get("lfs") or {}).get("size",0)
      for f in json.load(sys.stdin)
      if f["path"].endswith(".safetensors") and "experts" not in f["path"]
      and "ple" not in f["path"])/1e9, "GB")'
```

**Dense models are the same trap with no escape hatch.** `auto` resolves a dense
model to `fused`, so every weight is resident and the offload machinery does not
apply at all: `RedHatAI/Muse-Glimmer-30B-NVFP4` is 23.4 GB on disk and needs
23.4 GB of VRAM, because its parameters are one MLP stack rather than experts.
The snippet above is the check: on a dense model it returns the whole
checkpoint. Look for `num_experts` in the config too, before assuming something
benefits from this engine.

**Recommendation for this class of host:** prefer NVFP4 over FP8. It is not
close — 2.3x the throughput on 50 GB less host RAM — and the reason is
structural, not a tuning accident. See [FP8 vs NVFP4](#fp8-vs-nvfp4-measured).

Not tested here, and listed only so the gap is explicit: multi-GPU (the engine
serves one card per process), `bf16` and `mxfp4` checkpoints, and any model
outside FreeToken's supported list.

## The machine

| | |
| --- | --- |
| GPU | RTX 5090, 32,607 MiB, PCIe gen4 x16, 550 W board limit |
| CPU | 64 cores |
| Host RAM | 251 GiB |
| Driver / CUDA | 595.84 / 13.3 |
| Node | viiwork-freetoken, one backend, 4 slots |
| Engine | FreeToken 0.1.2 (PyPI), and a source build of `main` where noted |

Measured ceilings, from `ft bench bw` on this host:

```
CPU STREAM read 143.3 GB/s   |   PCIe H2D 27.5, D2H 28.7 GB/s
```

That ratio — host memory roughly 5x the PCIe link — is the whole reason an
offload MoE engine is viable here, and it is what `--moe-backend auto` reads
when it picks a backend.

## Results

Both served through the node, `--moe-backend` left to `auto`.

| | DeepSeek-V4-Flash | Qwen3.8-Flash-Next |
| --- | ---: | ---: |
| Checkpoint | 156 GB (fp8 blocks, fp4 experts) | 126 GB (NVFP4) |
| | | *FP8 build: see below* |
| Experts | 256 x 43 layers, top-6 | 512 x 48 layers, top-10 |
| MoE backend chosen | `hybrid` | `hybrid` |
| **Decode, 1 stream** | **24.5 tok/s** | **53.8 tok/s** |
| **Decode, 4 concurrent** | **51.1 tok/s** | **147 tok/s** |
| Per stream at 4 concurrent | 12.8 | 36.8 |
| TTFT floor | 5.2 s | 2.4 s |
| Prefill, marginal | ~1,000 tok/s | too fast to measure (see below) |
| **Servable context, default** | **64,128** | **8,192** |
| Advertised context | 1,048,576 | 262,144 |
| Energy at 4 concurrent | 3.8 J/tok | 1.6 J/tok |
| Power, idle → load | 13.4 → 195 W | 14.1 → 229 W |
| Load time (safetensors) | ~2 min | ~4.5 min |

Qwen is roughly **2.2x faster single-stream and 2.9x faster under
concurrency**, at half the usable context out of the box. Both leave the GPU
mostly idle in power terms — 195-230 W of a 550 W budget — because most of a
decode step is host memory and CPU, not the card.

### Prefill behaves completely differently

DeepSeek's time-to-first-token scales with the prompt; Qwen's does not move at
all inside its context window.

| Prompt | DSV4 cold | DSV4 warm | Qwen cold | Qwen warm |
| ---: | ---: | ---: | ---: | ---: |
| ~2,000 tok | 6.4 s | 5.2 s | 2.46 s | 2.47 s |
| ~8,000 tok | 11.2 s | 5.2 s | 2.50 s | 2.49 s |
| ~32,000 tok | 37.4 s | 5.3 s | — (over its limit) | — |

For DeepSeek the prefix cache is the difference between usable and not: a
repeated 32k prompt costs 5.3 s instead of 37.4 s. For Qwen there is nothing to
cache — a 7,336-token prompt adds under 50 ms to a 2.46 s floor, so cold and
warm are indistinguishable. Each figure is the median of three independent cold
prompts with fresh random filler.

## Context is bought with expert cache, and it is cheap

The single most useful thing measured here. KV and the MoE expert cache share
one VRAM budget (18.2 GiB on this card), and the engine's default split is very
conservative about context. You can move it at runtime, with no restart:

```sh
ft ctl cache --kv 65536 --moe 5000
```

Qwen3.8-Flash-Next, decode at concurrency 1:

| KV context | Expert slots | Decode | Cost |
| ---: | ---: | ---: | --- |
| 8,192 (default) | 5,932 (24.1%) | 53.8 tok/s | — |
| 65,536 | 5,000 (20.3%) | 53.0 tok/s | **8x context for 1.5%** |
| 262,144 (the full advertised window) | 3,000 (12.2%) | 48.1 tok/s | **32x context for 11%** |

So the advertised 262k context *is* reachable on a 32 GB card, and the default
8,192 is leaving a great deal on the table for most workloads. The node tracked
every successful rebuild live — `slot_ctx` went 8,192 → 65,536 → 262,144 with
no restart.

Two ways it refuses, both safe:

- Asking for more than the budget is rejected up front: *"needs 16.86 GiB >
  budget 15.51 GiB; old cache kept, still serving"*.
- Shrinking KV back while regrowing the expert cache hit a CUDA OOM, rolled
  back, and kept serving at the previous split. Growing the expert cache back
  runs into fragmentation; restart the backend instead.

## `hybrid` is not always the right backend

`auto` picks `hybrid` for both models given a bench profile. For Qwen that is
right under load and wrong for a single user:

| | `offload` | `hybrid` |
| --- | ---: | ---: |
| 1 stream | **64.0 tok/s** | 53.8 tok/s |
| 2 concurrent | 84.6 | 87.8 |
| 4 concurrent | 122.1 | **147.0** |
| J/token at 4 | 1.8 | **1.6** |

`hybrid` computes most expert misses on the CPU and fetches 18.1% over PCIe,
which batches well; `offload` streams every miss and wins when there is only one
token in flight. If this node serves one interactive user, pin
`freetoken.moe_backend: offload` and take the 19% single-stream gain. If it
serves a queue, leave `auto` alone.

## FP8 vs NVFP4, measured

Both are the same model. `Qwen/Qwen3.8-Flash-Next-FP8` is 173 GB on disk; the
`RadixArk` NVFP4 conversion is 126 GB. On this card the difference is not a
rounding error in either direction:

| Qwen3.8-Flash-Next | NVFP4 `hybrid` | NVFP4 `offload` | FP8 (`offload` only) |
| --- | ---: | ---: | ---: |
| Decode, 1 stream | 53.8 tok/s | **64.0 tok/s** | 34.4 tok/s |
| Decode, 4 concurrent | **147.0 tok/s** | 122.1 tok/s | 63.9 tok/s |
| TTFT floor | 2.43 s | 2.42 s | 4.29 s |
| J/token at 4 concurrent | **1.6** | 1.8 | 2.6 |
| Expert slots cached | 5,932 (24.1%) | 5,932 | **3,349 (13.6%)** |
| Host RAM | 125 GB | 125 GB | **177 GB** |
| Disk | 126 GB | 126 GB | 173 GB |

**NVFP4 is 2.3x faster under concurrency and 1.9x faster single-stream**, on 50 GB
less host RAM. FP8 loses twice over, and the two losses compound:

1. **Fewer experts fit in the cache.** The VRAM budget is the same ~18 GiB
   either way, but FP8 experts are larger, so 3,349 of them fit where 6,000
   NVFP4 experts do — 13.6% of the model resident instead of 24.1%. More misses.
2. **Every miss must cross PCIe.** `ft bench bw` reports no CPU path for this
   format, so `hybrid` is unavailable and each miss is a 26 GB/s transfer rather
   than 111 GB/s of CPU arithmetic.

The power figures are the tell that this is the right explanation: FP8 draws
*less* power under load than NVFP4 (167 W against 229 W) while producing under
half the tokens. It is not working harder and losing — it is waiting on the bus.

### The bench profile predicts this before you download anything

`ft bench bw` measures each expert format against both paths, in a couple of
minutes, without needing the model:

```
format      expert       CPU-MoE   PCIe-gather  CPU/PCIe  backend
bf16       9.00 MB    128.3 GB/s     26.1 GB/s     4.92x  hybrid
nvfp4      7.61 MB    111.3 GB/s     26.1 GB/s     4.27x  hybrid
ds_fp4    12.75 MB    101.0 GB/s     26.2 GB/s     3.86x  hybrid
mxfp4     12.62 MB     49.3 GB/s     26.2 GB/s     1.88x  offload
fp8        3.00 MB           n/a     26.1 GB/s         -  offload
       └─ CPU MoE has no fp8_block weight path; hybrid unavailable
```

That table said `offload` for fp8 and `hybrid` for nvfp4 before either model was
fetched, and the engine picked exactly that at load time in both cases. Run it
first and let it choose the checkpoint — it is far cheaper than downloading
173 GB to find out.

`mxfp4` is a milder warning of the same kind: at 1.88x it falls *under* the 2.0x
threshold and resolves to `offload` even though the CPU path exists.

**None of this transfers to other hardware.** The table is a property of one
CPU, one PCIe link and one card; a host with faster PCIe or slower memory would
reorder it.

## Method

Everything is measured **through the node** on `:9801`, not against the engine,
so the figures include what the proxy costs. Facts (cache geometry, advertised
context) come from the engine's own control plane. One script,
[`bench_final.py`](bench_final.py), one uncontended pass per configuration; any
concurrent model download was stopped first.

Three earlier passes were discarded rather than published, and the reasons are
worth repeating because each produced a plausible-looking number:

- **Shared filler across prompt sizes.** A "cold" 32k prompt was served from the
  prefix cache an earlier 8k prompt had populated, and came out *faster* than
  the 8k run. Fixed by generating unique random filler per prompt and per
  repetition.
- **Idle power sampled straight after load**, giving 107 W for a card that
  actually idles at 13.4 W once it has settled.
- **A counting prompt that stopped after 19 tokens**, which made
  aggregate-throughput-over-wall almost entirely prefill.

**Do not try to measure the CPU side with `perf`'s cache events.** On this
workload `cache-misses` reports ~11 GB/s and the AMD L1D fill events
(`ls_refills_from_sys`, `ls_hw_pf_dc_fill`, `ls_sw_pf_dc_fill`, all
`.ls_mabresp_lcl_dram`) report ~3.9 GB/s, against a real figure north of
37 GB/s. They do not see the streaming path, and `nps1_die_to_dram` needs the
`amd_df` PMU, which these kernels do not expose. What does work is a
multi-threaded DRAM read probe run idle and then again during a decode: it
reproduces `ft bench bw`'s STREAM figure to within 0.3%, and the drop is the
bandwidth the decode is consuming. Stealing bandwidth that way also drops GPU
utilisation from ~50% to 38%, which is the cleanest demonstration that host
memory paces the card rather than the other way round.

Token counts are the engine's own (`/v1/messages/count_tokens`), not a
words-times-a-constant estimate — the two differ by more than 2x for random
dictionary words.

Power is the node's `power_watts`, which on these hosts is the **sum of per-GPU
board power**, not a chassis reading. It excludes CPU, RAM and PSU losses, and
on an offload MoE workload that omission is large: the CPU is doing most of the
expert arithmetic. Read the J/token figures as GPU-side only.
