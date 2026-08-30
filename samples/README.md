# Samples

Working examples, sanitised. Every address, hostname and path here is a
placeholder — see "Placeholders" below before copying anything.

| Sample | What it is |
| --- | --- |
| [`dsv4-flash-rtx5090/`](dsv4-flash-rtx5090/) | A complete single-GPU deployment of a 156 GB model on a 32 GB card: config, systemd unit, model fetch, and how to pick the MoE backend. |
| [`energy-dump/`](energy-dump/) | Reads a durable energy store and prints what is in it, including the per-model split that never crosses the wire. |

## Placeholders

Fleet topology is a privacy boundary in this project, so the samples carry
none. Substitute your own:

| Placeholder | Means |
| --- | --- |
| The peer addresses in `peers.hosts` | Other nodes. The samples use addresses from the Tailscale CGNAT range, because a tailnet is the deployment this is built for; substitute your own. |
| `gpu1`, `gpu2`, `gpu3` | Hosts in the mesh. |
| `/srv/models` | Wherever model weights live. Not `/`, on any host where that is small — a single checkpoint here is hundreds of gigabytes. |
| `/opt/freetoken/.venv` | Wherever the engine's virtualenv lives. |

The literal placeholder addresses live only in the `.example` files. This
README avoids spelling them out on purpose: `publish.sh` scans everything it
would publish for private address ranges and exempts `.example` files, so a
placeholder quoted in prose here fails the publish rather than the review.

## The port convention these assume

Node ports are `9<model-class><instance>`: class 1 might be a chat model at
`9101` on the first host and `9102` on the second, class 2 something else at
`9201`. The instance digit is a sequence number within that model's pool, not a
fixed per-host index.

`gpus.base_port` is separate and per-node, chosen so co-located instances do not
collide. None of this is enforced by the code — it is a fleet convention, and
the samples follow it so they read like the real thing.

Node defaults sit in their own band per implementation, so a host running more
than one kind does not have to renumber: viiwork 9001+/91xx-95xx,
viiwork-nvidia 9601/9701+, and this node 9801/9901+.
