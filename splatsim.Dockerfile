# Splatsim renderer container.
#
# Wraps the tier4/splatsim library (Gaussian Splatting based renderer) and
# exposes it through the alpasim SensorsimService gRPC contract, so Runtime
# can talk to it as a drop-in replacement for the NuRec renderer.
#
# Built separately from the alpasim-base image because splatsim pins
# Python 3.10 + CUDA 12.4 + PyTorch 2.4, while alpasim-base targets 3.11+.
# Standalone like trafficsim.Dockerfile.
#
# Build (from repo root):
#   docker build -f splatsim.Dockerfile -t alpasim-splatsim-renderer .
#
# Used by the `renderer=splatsim` wizard profile.

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=python3.10 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 \
      python3.10-dev \
      python3.10-venv \
      ca-certificates \
      curl \
      git \
      build-essential \
      ninja-build \
      libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY src/grpc /repo/src/grpc
COPY src/splatsim_renderer /repo/src/splatsim_renderer

# Compile alpasim_grpc protos first so the wrapper can resolve them as a path
# source.
WORKDIR /repo/src/grpc
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync
RUN uv run compile-protos --no-sync

# Install the splatsim_renderer wrapper + its `splatsim` extra (which pulls in
# the upstream tier4/splatsim package).
WORKDIR /repo/src/splatsim_renderer
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    uv sync --extra splatsim --extra dev

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    PATH="/repo/src/splatsim_renderer/.venv/bin:${PATH}" \
    UV_NO_SYNC=1

ENTRYPOINT ["uv", "run", "splatsim_renderer_server"]
