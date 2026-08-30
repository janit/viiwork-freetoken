# viiwork-freetoken: a viiwork mesh node backed by FreeToken on NVIDIA GPUs.
#
# The Go binary is built in the pinned toolchain image and dropped into a CUDA
# image with FreeToken installed into a virtualenv. Nothing here is compiled
# against CUDA — this process only ever spawns and supervises `ft serve` — so
# the inference stack is entirely what the two ARGs below pin.
#
# Read this before building, because it differs from the vLLM node in ways that
# cost time and disk:
#
#   * There is no official FreeToken image to start from, so the engine is
#     installed here. That makes the build slow (torch and its CUDA wheels) and
#     the image large — several gigabytes.
#   * It must be a *devel* CUDA image, not a runtime one. FreeToken JIT-compiles
#     its kernels on first use and needs nvcc on PATH at RUN TIME, not just at
#     build time. A runtime image passes docker build and then fails on the
#     first request.
#   * The JIT cache and the model cache belong on volumes. Without them every
#     container start recompiles kernels and re-downloads weights — see
#     docker-compose.yaml.example.
#
# Most FreeToken deployments are not containers at all: the engine is an
# edge-native runtime that expects host RAM, a fast PCIe link and a persistent
# kernel cache, and running it natively beside a systemd unit for this node is
# the simpler path. The image exists for fleets that are already containerised.

# CUDA 13 is FreeToken's floor (driver r580+). Pinning the tag is how you pin
# the toolkit the kernels are compiled against.
ARG CUDA_TAG=13.3.1-devel-ubuntu24.04
ARG FREETOKEN_VERSION=

FROM golang:1.27.0 AS go-build
WORKDIR /src

# Dependencies first, so a source-only change does not re-download them.
COPY go.mod go.sum ./
RUN go mod download

COPY . .
ARG VERSION=dev
# CGO off: the binary is pure stdlib plus yaml, and a static one drops cleanly
# into any runtime image regardless of its libc.
RUN CGO_ENABLED=0 go build -buildvcs=false \
      -ldflags "-X main.version=${VERSION}" \
      -o /viiwork-freetoken ./cmd/viiwork-freetoken

FROM nvidia/cuda:${CUDA_TAG}
ARG FREETOKEN_VERSION

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Its own virtualenv, on PATH, so `ft` resolves the way the config file's
# default binary name expects.
ENV VIRTUAL_ENV=/opt/freetoken
ENV PATH="$VIRTUAL_ENV/bin:/usr/local/cuda/bin:$PATH"
RUN python3 -m venv "$VIRTUAL_ENV" \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "freetoken[accel]${FREETOKEN_VERSION:+==${FREETOKEN_VERSION}}" \
    && ft --version

# nvidia-smi comes from the NVIDIA container runtime rather than the image, so
# GPU metrics require the container to be started with GPU access (see
# docker-compose.yaml.example). Without it the node still serves — it just
# reports no GPUs.
COPY --from=go-build /viiwork-freetoken /usr/local/bin/viiwork-freetoken

CMD ["viiwork-freetoken", "--config", "/config/viiwork-freetoken.yaml"]
