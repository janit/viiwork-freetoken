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

Expect the first start to take many minutes, and the *very* first to take
longer still: FreeToken JIT-compiles its CUDA kernels on first use. The node
logs each load phase as `/health` reports it, so you can tell progress from a
hang:

```
[gpu-0] loading (weights, 34%)
[gpu-0] loading (cuda_graphs)
[gpu-0] gpu-0 healthy
```

Then:

```sh
curl -s localhost:9801/v1/models
curl -s localhost:9801/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"DeepSeek-V4-Flash","messages":[{"role":"user","content":"hi"}]}'
```

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
```

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
