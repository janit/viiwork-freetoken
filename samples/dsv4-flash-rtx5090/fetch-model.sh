#!/usr/bin/env bash
# Fetch a HuggingFace model with an outer resume loop.
#
# Why this is not `curl -L -o` or `hf download`:
#
# Some hosts corrupt large TLS transfers. The symptom is an abort several GB
# in, repeatably, with "SSL_read: decryption failed or bad record mac"
# (curl exit 56) or "tls: bad record MAC". curl's own --retry does not cover
# that class, so each file is wrapped in a loop that re-runs `curl -C -` until
# the bytes on disk match what the HF API reports.
#
# At this repo's size the loop stops being a nicety. DeepSeek-V4-Flash-0731 is
# 156 GB across 48 shards: a transfer that restarts from zero on any fault will
# not finish, and one that resumes per file will.
#
# Sizes come from the API rather than being hardcoded, so a repo revision does
# not silently leave a short file looking complete.
set -uo pipefail

REPO="${REPO:-deepseek-ai/DeepSeek-V4-Flash-0731}"
DEST="${DEST:-/srv/models/$(basename "$REPO")}"
ATTEMPTS="${ATTEMPTS:-40}"

mkdir -p "$DEST"

# Single instance per destination.
#
# Not defensive programming — this was hit. A download this long gets nudged:
# you background it, the connection drops, you start it again, and now two
# copies are running `curl -C -` against the same paths. Each resumes from the
# offset IT sees, they interleave writes, and the result is a file of exactly
# the right length and the wrong contents. The size check at the bottom passes,
# and the corruption surfaces an hour later as a tensor that will not load.
#
# The lock is held for the life of the script and released when it exits, so a
# second invocation exits immediately rather than joining in.
exec 9>"$DEST/.fetch.lock"
if ! flock -n 9; then
    echo "another fetch is already running for $DEST (lock: $DEST/.fetch.lock)"
    exit 1
fi
manifest=$(mktemp)
trap 'rm -f "$manifest"' EXIT

python3 - "$REPO" > "$manifest" <<'PY'
import json, sys, urllib.request
repo = sys.argv[1]
tree = json.load(urllib.request.urlopen(
    f"https://huggingface.co/api/models/{repo}/tree/main"))
for f in tree:
    if f.get("type") != "file":
        continue
    size = (f.get("lfs") or {}).get("size") or f.get("size") or 0
    print(f["path"], size)
PY

[ -s "$manifest" ] || { echo "manifest fetch failed"; exit 1; }
echo "--- $REPO -> $DEST ---"; cat "$manifest"

fail=0
while read -r path want; do
    # Repo furniture: not needed to serve, and model.sig is large.
    # .fetch.lock is ours, not the repo's.
    #
    # Note what is NOT skipped: inference/config.json. FreeToken reads the
    # authoritative DeepSeek-V4 model arguments from the inference/ subdir
    # rather than from the top-level config.json, and a checkout missing it
    # fails at load with an error that does not mention the missing file.
    case "$path" in .gitattributes|README.md|model.sig|.fetch.lock) continue;; esac
    out="$DEST/$path"
    mkdir -p "$(dirname "$out")"
    for attempt in $(seq 1 "$ATTEMPTS"); do
        have=$(stat -c %s "$out" 2>/dev/null || echo 0)
        [ "$have" = "$want" ] && break
        echo "[$path] attempt $attempt: $have / $want"
        curl -sSL -C - -o "$out" \
            "https://huggingface.co/$REPO/resolve/main/$path" || true
    done
    have=$(stat -c %s "$out" 2>/dev/null || echo 0)
    if [ "$have" = "$want" ]; then
        echo "OK   $path ($want)"
    else
        echo "FAIL $path ($have / $want)"; fail=1
    fi
done < "$manifest"

echo "=== done, fail=$fail ==="
du -sh "$DEST"
exit $fail
