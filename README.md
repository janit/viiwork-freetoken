# viiwork-freetoken

A [viiwork](https://github.com/janit/viiwork) mesh node backed by
[**FreeToken**](https://github.com/FlashML-org/FreeToken) on **NVIDIA** GPUs.

viiwork is an LLM inference load balancer built for AMD Radeon VII cards: it
spawns one `llama-server` per GPU and fronts them with an OpenAI-compatible API
that load-balances across a mesh of hosts. This project is a third
implementation of the same mesh protocol, alongside
[viiwork-nvidia](https://github.com/janit/viiwork-nvidia), which drives vLLM.

What FreeToken adds to the fleet is a different *class* of node. vLLM and
llama.cpp both want the model to fit the hardware. FreeToken is an edge-native
MoE engine: expert weights live in host RAM and are either streamed over PCIe or
computed on the CPU, so the card only ever holds a cache of the hot ones. On the
host this was built against that means a 156 GB checkpoint served from 32 GB of
VRAM, at 24.5 tok/s. Such a host joins the mesh as a peer and serves a model no
other node in the fleet can hold.

It is a **separate repository on purpose**. The node types differ where the
hardware and the inference engine differ — and nowhere else.

```
                       ┌─────────────────────────┐
   any client ────────▶│  any node in the mesh   │  one entry point,
   (OpenAI API)        │  routes by model name   │  every model
                       └────────────┬────────────┘
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌───────────────────────┐       ┌───────────────────────┐
        │ viiwork (AMD hosts)   │       │ viiwork-freetoken     │
        │ llama.cpp · ROCm      │◀─────▶│ FreeToken · CUDA      │
        │ models that fit VRAM  │ mesh  │ experts in host RAM   │
        └───────────────────────┘       └───────────────────────┘
                    └───────────────┬───────────────┘
                                    ▼
                        one /mesh dashboard, served
                          identically by every node
```

## What it does

- Spawns and supervises one `ft serve` process per GPU, pinned with
  `CUDA_VISIBLE_DEVICES`, with health checks, bounded respawns and graceful
  shutdown.
- Reads readiness from FreeToken's `/health` **document**, not its status code,
  so a model still loading is never advertised to the mesh as ready — and logs
  each load phase (`expert_banks`, `warmup`, …) while it waits.
- Publishes the context the backend will actually **accept**, not the one the
  checkpoint advertises. Those differ by 16x on one of the models measured
  here, and the advertised figure is one a client would get a 400 for.
- Load-balances across local backends, and across the mesh by model name.
- Publishes `/v1/status` and `/v1/cluster` in the mesh wire format, so viiwork
  nodes see it as a peer.
- Serves the **same** `/mesh` dashboard, `/prompt` page and per-node dashboard
  as viiwork — the pages are imported from viiwork's `web` package, not forked.
- Collects per-GPU utilisation, VRAM and power from `nvidia-smi`, and publishes
  node wattage as the sum of per-GPU board power — labelled as such, see
  [Power is a GPU sum](#power-is-a-gpu-sum).
- Optionally records a durable per-host, per-model kWh history, using viiwork's
  shared `energy` package rather than a second implementation of it.
- Records recent prompts and outputs in memory for the dashboard's prompt view.

## What this achieves

One RTX 5090 — a gaming card with 32 GB — serving frontier MoE checkpoints
several times its own size, as a full member of the mesh. Measured through the
node on that host, with FreeToken's `hybrid` backend keeping experts in system
memory:

| | DeepSeek-V4-Flash | Qwen3.8-Flash-Next |
| --- | ---: | ---: |
| Checkpoint | 156 GB | 126 GB (NVFP4) |
| Decode, single stream | 24.5 tok/s | **53.8 tok/s** |
| Decode, 4 concurrent | 51.1 tok/s | **147 tok/s** |
| Servable context, default | 64,128 | 8,192 |
| ...and reconfigured | — | **262,144**, for 11% throughput |
| Power under load | 195 W | 229 W (card limit 550 W) |
| Energy at 4 concurrent | 3.8 J/tok | 1.6 J/tok |

The card is not the constraint — 195-230 W of a 550 W budget, because most of a
decode step is host memory and CPU. What you need is **system RAM**: the experts
have to live somewhere, and that host has 251 GB.

Three results worth reading twice:

- **Advertised context is not servable context.** Both models advertise far more
  than the backend accepts, because the engine sizes its KV pool from the VRAM
  left after the expert cache — 1,048,576 vs 64,128 for DeepSeek, 262,144 vs
  8,192 for Qwen. The node publishes the servable figure to the mesh, so the
  dashboard shows a number a client can actually use.
- **Context is cheap to buy back.** `ft ctl cache --kv 262144 --moe 3000` gets
  Qwen its full 262k window at an 11% throughput cost, live, with no restart.
- **Quantisation decides which MoE backend you get.** The same Qwen model as
  FP8 is 2.3x slower under concurrency than its NVFP4 conversion, on 50 GB more
  host RAM — the CPU expert kernel has no fp8_block path, so `hybrid` is
  unavailable and every miss crosses PCIe. `ft bench bw` predicts this in two
  minutes without either checkpoint present.

**[`samples/benchmarks`](samples/benchmarks/)** — which models have been tested
and what they need, the hardware they were measured on, full results, the
method, and the raw output. **[`samples/dsv4-flash-rtx5090`](samples/dsv4-flash-rtx5090/)**
— a complete worked deployment: config, systemd unit, model fetch.

## The protocol lives in viiwork

viiwork owns the wire contract, published as
[`github.com/janit/viiwork/meshapi`](https://github.com/janit/viiwork) — a
public, stdlib-only Go package. Both node types import it, so there is exactly
one definition of every field that crosses a host boundary, including the
grammar of the activity messages the dashboard parses to reconstruct in-flight
requests.

Adding a field to the mesh means adding it there first. `meshapi`'s tests pin
the field names and the compatibility rules; the mesh is a fleet of
independently upgraded machines, so a rename is a breaking change with no
migration path.

## Quick start

```bash
cp viiwork-freetoken.yaml.example viiwork-freetoken.yaml
$EDITOR viiwork-freetoken.yaml        # set model.name, model.path, gpus.devices
make build && ./bin/viiwork-freetoken --config viiwork-freetoken.yaml
```

FreeToken itself is installed separately — `pip install "freetoken[accel]"` into
a virtualenv, with a CUDA 13 toolkit and driver r580+ on the host. Point
`freetoken.binary` at its `ft`.

There is a Dockerfile, but read its header first: the image has to install the
engine itself and must be built on a *devel* CUDA base, because FreeToken
JIT-compiles its kernels at run time. Native beside a systemd unit is the
simpler deployment, and
[`samples/dsv4-flash-rtx5090`](samples/dsv4-flash-rtx5090/) does it that way.

Then:

- `http://<host>:9801/` — this node's dashboard
- `http://<host>:8086/` — the whole fleet
- `http://<host>:9801/v1/chat/completions` — the API

A complete worked deployment — a 156 GB model on a 32 GB card, with config,
systemd unit, model fetch and how to choose the MoE backend — is in
[`samples/dsv4-flash-rtx5090`](samples/dsv4-flash-rtx5090/).

## Joining an existing mesh

Peering is static and symmetric by convention. Add this node to the peer list
of the viiwork nodes, and them to its:

```yaml
# viiwork-freetoken.yaml — one entry per node, not per host: a host running
# several models runs several nodes, each on its own port.
peers:
  hosts: [amd1:9102, amd1:9302, amd2:9201]
```

```yaml
# viiwork.yaml on each AMD node
peers:
  hosts: [..., gpu1:9801]
```

Both directions matter: a node only routes to peers it knows about, so listing
one side only gives you routing that works one way. Once peered, `/v1/models`
on any node lists every model in the fleet and a request for any of them can be
sent to any node.

Two things to get right:

- **`node.hostname` must be what other hosts resolve this machine by.** Peers
  key per-host deduplication on it. If instances are peered by IP and the
  hostname is wrong, the dashboard splits one machine into several.
- **`server.mesh_port` should match across the fleet** (8086). The port is
  contended: whichever node on a host wins it serves that host's fleet view,
  and every node type contends for it equally.

## How it differs from the other nodes

The mesh-facing half is identical by construction. What differs is what sits
underneath:

| | viiwork | viiwork-nvidia | viiwork-freetoken |
|---|---|---|---|
| Inference engine | `llama-server` | `vllm serve` | `ft serve` |
| Model must fit VRAM | yes | yes | **no** — experts live in host RAM |
| Unit of deployment | one server per GPU | one per tensor-parallel group | one server per GPU |
| GPU pinning | `ROCR_VISIBLE_DEVICES` | `CUDA_VISIBLE_DEVICES` | `CUDA_VISIBLE_DEVICES` + `CUDA_DEVICE_ORDER` |
| Metrics source | `rocm-smi` | `nvidia-smi` | `nvidia-smi` |
| Backend stats | llama.cpp `/slots` | Prometheus `/metrics` | JSON `/v1/stats` |
| Readiness | HTTP status | HTTP status | a **field** in `/health` |
| Default concurrency | slots | 256 sequences | **4** requests |
| Startup | seconds | minutes | tens of minutes |

Four of those rows are worth expanding, because each is a place where porting
the vLLM node's behaviour verbatim would produce something subtly broken.

**Readiness is a field, not a status code.** FreeToken answers `/health` with
HTTP 200 in every lifecycle state — loading, serving, mid-cache-rebuild, failed
— while the generation endpoints 503 in all but one of them. A node that
treated 200 as ready would advertise a backend to the mesh twenty minutes
before it could serve anything, and every routed request would fail. So the
probe parses the document and reads `status` and `maintenance`. The upside is
that the wait is no longer silent: `/health` reports a load phase and a byte
count, and the node logs each phase change.

**Concurrency is small, and that is not a default to raise.** `ft serve` runs 4
requests concurrently by default, against vLLM's 256. On an edge-native engine
every concurrent slot costs KV capacity in a consumer card's VRAM, competing
with the expert cache that is doing the real work. The node's backpressure
ceiling follows it, so this node returns 429 to the mesh at ordinary
concurrency rather than only under abuse — which is correct: a node that queues
past what the scheduler will run is hiding latency the mesh could have routed
around.

**There is no queue-depth gauge, so the node derives one.** FreeToken's
`/v1/stats` reports how many requests are running and never how many are
waiting. Rather than publish a zero indistinguishable from an idle queue, the
node subtracts: backends bind loopback and this node is their only client, so
in-flight minus running *is* the queue, measured on this side of the socket.

**GPU pinning has to work across two engine generations.** FreeToken 0.1.2, the
current PyPI release, has no `--gpu` flag at all; later builds add one but still
honour a preset `CUDA_VISIBLE_DEVICES`. The environment is the mechanism correct
on both, so that is what the node uses — together with
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, without which CUDA's default fastest-first
enumeration can make `gpus.devices: [1]` select a different card than the one
the dashboard shows it against.

And the startup difference is the one that bites in practice:
`freetoken.startup_timeout` defaults to **20 minutes**, because loading here is
weights, then cache-pool sizing, then CUDA graph capture, then filling a GPU
expert cache from host memory — on models chosen precisely because they do not
fit in VRAM.

### Fields that stay blank

`tok_decoded` and `tok_remain` are omitted rather than filled. FreeToken
publishes cumulative token counters and a sliding-window decode rate, not
per-request progress, and a cumulative total in a column the dashboard renders
as "tokens left in this request" would be worse than a blank. `slot_active`
carries the engine's active-request count instead, which is the honest
equivalent.

Cache occupancy is *collected* as the fullest of the engine's pools rather than
as "KV cache" — which pools exist depends on the model, and reading only `kv`
would report a permanent zero for a sliding-window or linear-attention one. It
does not cross the mesh: `meshapi.BackendInfo` has no field for it, and adding
one means releasing viiwork first. The vLLM node is in the same position. Read
it from the engine directly with `ft ctl cache` meanwhile.

## Not implemented

Deliberately, for now — all of it is `omitempty` in the protocol, so the
dashboard degrades to blanks for this host rather than breaking:

- **Electricity cost.** Energy is recorded; pricing it is not. No
  `cost_eur_per_hour`, `cost_today_eur` or `cost_breakdown`.
- **Node power over IPMI.** There is no uniform BMC across these hosts, which
  is why `power_watts` is a GPU sum rather than a chassis reading — see below.
- **Tensor parallelism.** FreeToken has the flag and does not yet place more
  than one card, so `gpus` here is a list of independent backends.
- **Chassis power control.** This node publishes no `power_control` allowlist,
  so no power row is offered for it.
- **Pipelines.** viiwork's multi-step YAML pipelines are not ported.

## Power is a GPU sum

`power_watts` does not mean quite the same thing here as on a viiwork node, and
it is the one place the two implementations put physically different quantities
in one wire field.

viiwork measures the chassis over IPMI or DCMI — CPU, fans, drives and PSU
losses included, around 400 W on an idle 10-GPU host. These hosts have no
uniform BMC, so this node publishes the total of per-GPU board power, which
excludes all of that and understates the wall figure substantially.

The gap is wider on this node than on the others, and worth naming. An offload
MoE backend is deliberately CPU- and memory-heavy: experts are streamed from or
computed in host RAM, and a many-core part doing that draws real power that a
per-GPU sum cannot see at all.

That is a deliberate trade rather than an approximation waiting to be improved.
An operator-supplied overhead constant would put a fabricated number into a
field that is measured everywhere else in the fleet, and a wrong guess becomes
invisible the moment it is summed into a fleet total. What makes it honest is
the label: `power_source` reports `nvidia-smi`, and the same string is recorded
in the energy store's directory, so neither a dashboard nor a history copied off
the host can mistake it for a chassis reading.

Two consequences worth knowing:

- **The sum spans the host, not this instance's cards**, because node wattage is
  a per-host measurement. So `energy.enabled` belongs on **exactly one instance
  per host** — two recorders on one machine would record the same draw twice,
  and no node can detect that on its own.
- **Attribution is `energy.Direct`.** viiwork has to divide one chassis figure
  between the models causing it and a baseline drawn anyway. Here the node
  figure *is* the sum of the per-GPU readings, so there is nothing to divide:
  each card is charged its own measured draw, the shares reconcile with the node
  total exactly, and the baseline is honestly zero.

Only whole-node `energy_kwh_24h` and `energy_kwh_30d` cross the mesh. The
per-GPU and per-model detail is recorded but local — in both implementations —
so [`samples/energy-dump`](samples/energy-dump) reads it.

## Development

The test suite needs no GPU and no FreeToken: backends are `httptest` servers
answering with recorded `/health` and `/v1/stats` documents, and `nvidia-smi`
output is fixtures.

```bash
make test     # unit tests
make race     # the useful gate — the mesh is concurrent by nature
make vet
```

### Working against an unreleased protocol change

`go.mod` names a published viiwork version and carries **no `replace`
directive**, so the module builds for anyone who fetches it. To develop against
a `meshapi` change that is not published yet, put a viiwork checkout beside this
one and use a Go workspace:

```bash
cat > go.work <<'EOF'
go 1.27.0
use (
	.
	../viiwork-private
)
EOF
```

`go.work` is gitignored. `GOWORK=off go build ./...` builds the published form —
what a consumer actually gets — and is worth running before you rely on a field
you just added upstream.

## License

MIT
