# A frontier MoE model on one consumer card

A complete deployment of DeepSeek-V4-Flash on a single RTX 5090, joining a mesh
over a tailnet. Addresses and hostnames are placeholders — see
[`../README.md`](../README.md).

The model is 156 GB. The card has 31.8 GiB. That is not a mistake, it is the
entire point of this engine, and it is what every decision below follows from.

## The machine

Measured on the host this sample is drawn from, because the numbers matter:

| | |
| --- | --- |
| GPU | RTX 5090, 32,607 MiB, PCIe gen4 x16, 550 W board limit |
| Driver / CUDA | 595.84 / 13.3 |
| CPU | AMD EPYC 7B12, 64 cores / 128 threads |
| Host RAM | 251 GiB |
| Engine | FreeToken 0.1.2 |
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731`, 156 GB in 48 safetensors shards |

Host RAM, not VRAM, is the requirement people miss. On an offload MoE backend
the expert weights live in system memory: 156 GB of them here, on a 251 GiB
box, which leaves under 100 GiB for everything else on the machine. Check that
before ordering the download.

## Files

| File | |
| --- | --- |
| `viiwork-freetoken.yaml.example` | The node config, with the reasoning for every non-default value. |
| `viiwork-freetoken.service.example` | A native systemd unit, which is the recommended shape here. |
| `fetch-model.sh` | Downloads a HF repo with an outer resume loop. At 156 GB this is not optional. |

Copy the `.example` files without the suffix; the real names are gitignored
because they are deployment-specific.

## Order

```sh
# 1. Driver first. CUDA 13 and driver r580+ are FreeToken's floor.
nvidia-smi
nvcc --version                     # must be on PATH at RUN time, not just install time

# 2. The engine, in its own virtualenv.
python3 -m venv /opt/freetoken/.venv
/opt/freetoken/.venv/bin/pip install "freetoken[accel]"
/opt/freetoken/.venv/bin/ft --version

# 3. Weights. Hours, and it will be interrupted. That is what the loop is for.
DEST=/srv/models/DeepSeek-V4-Flash-0731 ./fetch-model.sh

# 4. Calibrate this machine's MoE backend. Once per GPU. See below.
/opt/freetoken/.venv/bin/ft bench bw

# 5. Optional but worth it: convert to FTW so respawns do not re-read 156 GB.
/opt/freetoken/.venv/bin/ft checkpoint \
  --model /srv/models/DeepSeek-V4-Flash-0731 \
  --out   /srv/models/DeepSeek-V4-Flash-0731-ftw

# 6. The node.
make build && install -m755 bin/viiwork-freetoken /usr/local/bin/
install -m644 viiwork-freetoken.yaml /etc/viiwork/
systemctl enable --now viiwork-freetoken && journalctl -fu viiwork-freetoken
```

Expect minutes, and the *very* first start longer still: FreeToken JIT-compiles
its CUDA kernels on first use. The node logs each load phase as `/health`
reports it, so you can tell progress from a hang. Measured on this hardware,
from a warm kernel cache and unconverted safetensors:

```
[manager] gpu-0 spawned (pid 196029, GPU [0], port 9901)
[manager] gpu-0 loading (other)                     +5s
[manager] gpu-0 loading (expert_banks, 2%)         +20s
[manager] gpu-0 loading (warmup)                  +1m55s
[manager] gpu-0 healthy                           +2m00s
```

`expert_banks` is the long pole and the one to watch: it is the 156 GB going
into host RAM.

Then:

```sh
curl -s localhost:9801/v1/models
curl -s localhost:9801/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"DeepSeek-V4-Flash","messages":[{"role":"user","content":"hi"}]}'
```

## Measured

One uncontended run on the machine above, through the node rather than against
the engine, so the figures include what the proxy costs. FreeToken 0.1.2,
`--moe-backend` auto-resolved to `hybrid`, safetensors (not FTW).

| | |
| --- | --- |
| Decode, single stream | **24.5 tok/s** |
| Decode, 4 concurrent | **51.1 tok/s** aggregate (12.8 per stream) |
| Prefill, marginal | **~1,000 tok/s** |
| Servable context | **64,128 tokens** |
| Power, idle → load | **13.4 W → 195 W** on a 550 W card |
| Energy, 4 concurrent | **3.8 J/token** |
| VRAM / host RAM | 29.4 GB of 32.6 / 148 GB of 251 |

Side by side with Qwen3.8-Flash-Next on the same card, plus the
context-vs-expert-cache trade: [`../benchmarks`](../benchmarks/).

Five things in that table are worth more than the number itself.

**The GPU is not the bottleneck.** A 5090 is a 550 W card and this workload
draws 195 W, peaking at 230. That is the signature of `hybrid`: most of each
decode step is host memory and CPU, with 17.7% of expert misses crossing PCIe.
Undervolting or a lower power cap costs nothing here; more system memory
bandwidth would help.

**Concurrency is nearly free after the second stream.** Per-stream throughput
halves going from one stream to two — 24.6 to 13.7 — and then barely moves
going to four, 13.7 to 12.8. So aggregate throughput doubles from 1 to 4
concurrent while each caller sees roughly half the tokens per second. Energy
per token improves in step, 7.8 J to 3.8 J.

**The fifth concurrent request is refused, not queued.** At concurrency 8, four
requests completed and four came back 429, with aggregate throughput unchanged.
That is `max_in_flight_per_backend` defaulting to the engine's
`--max-running-requests` of 4. In a mesh that is what you want — another node
takes the request. On a single node with no peers it is a behaviour change from
running `ft serve` bare, which would have queued them.

**The prefix cache is the difference between usable and not.** A repeat prompt
is served in ~5.2 s *regardless of length*: 32k tokens costs 37.4 s cold and
5.3 s warm. For an agent replaying a long conversation this is the whole
experience.

| Prompt | Cold | Warm |
| ---: | ---: | ---: |
| 2,000 tok | 6.4 s | 5.2 s |
| 7,979 tok | 11.2 s | 5.2 s |
| 31,998 tok | 37.4 s | 5.3 s |

There is a ~5.2 s floor under every first token, which is why the marginal
prefill rate (~1,000 tok/s, from the slope) is the honest figure rather than
prompt ÷ TTFT.

**The advertised context is not the servable one, and the gap is 16x.**

```
/v1/models advertises   1,048,576 tokens
the backend accepts        64,128 tokens
   62,975 tok -> 200 OK
   70,590 tok -> 400 context_length_exceeded
                 "70590 tokens > 64128 maximum (prompt + generation)"
```

64,128 is the KV pool: 501 pages x 128. The engine sizes it from the VRAM left
after the weights and the expert cache, so on a card serving a model five times
its size the pool is a small fraction of what the checkpoint allows. The node
publishes the servable number to the mesh for exactly this reason.

You can buy context back, at a price the same table shows you:

```sh
ft ctl cache --kv 200000     # KV is resizable 3,968 .. 2,802,310 tokens
```

The budget is shared. Today it is 19.3 GB split as 15.4 GB of expert cache
(1,239 of 11,008 slots, 11.3%), 1.7 GB of full-attention window and 0.4 GB of
KV. KV is cheap per token here — 6.9 KB against 136 KB for the window pool — so
a large context increase costs expert slots, and expert slots are what the
throughput above is made of. Measure both sides before keeping the change.

## Choosing the MoE backend, which is the whole deployment

`fused` — every expert resident in VRAM — is off the table at 156 GB against
31.8 GiB, and the engine never auto-selects it anyway. The real choice is
inside the offload family:

| | Where the misses go |
| --- | --- |
| `offload` | Streamed from host RAM over PCIe, with an LRU cache of hot experts on the card. |
| `cpu` | Computed on the CPU. Nothing crosses PCIe. |
| `hybrid` | Some fetched, the rest computed, overlapped. Needs a bandwidth profile to size the split. |

Which wins is a property of the machine — specifically, whether host memory
bandwidth beats the PCIe link — and it is measurable rather than arguable:

```sh
ft bench bw            # writes ~/.cache/freetoken/benchbw/<gpu-uuid>.json
                       # (older engines: a single ~/.cache/freetoken/benchbw.json,
                       #  still honoured when the GPU name matches)
```

You can confirm it took effect. The engine says so at startup, and the node
passes its output through:

```
benchbw profile recommends hybrid for 'ds_fp4' experts on this GPU
Auto-selected MoE backend: hybrid
--moe-hybrid-max-fetch auto: fetching 17.7% of each decode step's expert
  misses over PCIe (benched PCIe/CPU bandwidth ratio), the rest on the CPU
```

If you see `offload` there instead, the profile was not found — check `HOME`.

It runs the real MoE kernels against both paths and recommends `hybrid` when
the CPU wins by more than 2x. On a host of this shape — 8-channel DDR4 against
one gen4 x16 link — it usually does, by a wide margin. **Run it anyway.** The
profile is keyed on GPU and expert format, so one from other hardware is
ignored rather than misapplied, and with a profile present you can leave
`freetoken.moe_backend` unset and let `auto` read it.

The profile lives under `$HOME/.cache`. A systemd unit whose user has no
writable home silently loses it on every restart and quietly serves the slower
backend forever — which is why the unit in this directory sets `HOME`.

## What to watch, and where

Cache occupancy does not cross the mesh yet — `meshapi.BackendInfo` has no
field for it — so the dashboard will not show you the number that matters most
here. Read it from the engine directly:

```sh
ft ctl cache      # pool table: KV pages, MoE expert slots
ft ctl stats      # throughput, VRAM, p95, active requests
```

Two knobs trade against each other in the same VRAM, and the dashboard is where
you see the trade land:

- `freetoken.max_seq_len` — context. The checkpoint declares 1,048,576
  positions (yarn, factor 16 over a 65,536 base). Serving that would spend on
  KV the VRAM that the expert cache is paying for. The sample uses 32k.
- `freetoken.max_running_requests` — concurrency, 4 by default. Each slot costs
  KV capacity. This is also the node's backpressure ceiling, so raising it
  raises how much this node will accept before returning 429 to the mesh.

Neither has a right answer that a config file can carry: measure with your
model and your traffic.

## Verified on this host

Behaviours worth knowing before you rely on them, each exercised against the
real engine rather than a test double.

**A crashed engine recovers on its own.** `kill -9` on the `ft` process:

```
request during the outage        ->  HTTP 503
[manager] gpu-0 respawning (attempt 1/3)
[manager] gpu-0 loading (expert_banks, 2%)
[manager] gpu-0 healthy                       ~210 s after the kill
```

A 503, not a hang and not a 500 — the node drops the backend out of the picker
the moment a probe fails, and answers honestly while there is nothing to route
to. Recovery costs a full model load, which is why `max_respawns` is bounded: a
model that cannot load would otherwise keep the machine busy failing for hours.

**No orphaned workers.** After the kill and respawn, `nvidia-smi` shows exactly
one compute process holding 29,146 MiB. FreeToken runs its scheduler, tokenizer
and detokenizer as separate processes, so the node signals the process *group*;
signalling the pid alone leaves those resident, holding VRAM, and the respawn
then fails for want of memory.

**Both engine builds work, unchanged.** The same node config was run against
the PyPI release (0.1.2, which has no `--gpu` flag at all) and against a source
build of `main` (which has one). Identical behaviour in both: the same
`hybrid` backend selected from the bench profile, the same servable context
published, working inference. That is the point of pinning the card through
`CUDA_VISIBLE_DEVICES` instead of the flag — see the Dockerfile and
`internal/freetoken.Backend.Env`.

## Why not the container

The repo ships a Dockerfile and it works, but this sample deploys natively, and
that is the recommendation for this engine rather than a shortcut. FreeToken
wants host RAM, a fast PCIe link and a persistent JIT kernel cache. In a
container all three become mounts and arguments you have to get right —
a *devel* CUDA base so `nvcc` exists at run time, a volume for
`~/.cache/freetoken` or kernels recompile every start, a volume for the
bandwidth profile, `shm_size` large enough for expert staging — to arrive back
where a native install started. Containerise this if the rest of your fleet is
containerised; otherwise do not.

## Energy

`energy.enabled` is on in this sample, which is only correct because this host
runs **one** node. Node wattage is a whole-host measurement: two recorders on
one machine record the same draw twice, and nothing detects that for you.

The gap between the recorded figure and the wall figure is wider here than on
most nodes, and worth saying out loud. With no BMC the node figure is the sum
of per-GPU board power, and this workload is deliberately CPU- and
memory-heavy — experts are streamed from or computed in host RAM, and a 64-core
part doing that draws real power a per-GPU sum cannot see at all. The store
records `nvidia-smi` as its source so a history read outside its mesh context
still says what it measured. Read it as a GPU figure. Read it with
[`../energy-dump`](../energy-dump).

Mount the store directory. Without it the rolling 24h and 30d totals reset on
every deploy, which is worse than reporting nothing.
