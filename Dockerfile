# Build command (run from repo root):
#   docker build -t alpasim-base .
#
# For private dependencies (requires ~/.netrc with credentials):
#   docker build --secret id=netrc,src=$HOME/.netrc -t alpasim-base .
#
# Automatically detects architecture:
#   x86_64  -> nvidia/cuda base
#   aarch64 -> NGC PyTorch base (only CUDA-enabled PyTorch source on ARM)
#
# CARLA Server (0.9.16, AMD64 only) is fetched in a separate stage and copied
# into the final image. ARM64 builds skip the CARLA stage.

FROM ubuntu:22.04 AS carla-fetch
ARG CARLA_VERSION=0.9.16
ARG TARGETARCH
RUN apt-get update && apt-get install -y curl tar ca-certificates xz-utils \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/carla && \
    if [ "$TARGETARCH" = "amd64" ]; then \
      curl -L "https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_${CARLA_VERSION}.tar.gz" \
        | tar -xz -C /opt/carla --strip-components=0 ; \
    else \
      echo "CARLA Server is not built for ${TARGETARCH}; skipping" > /opt/carla/UNSUPPORTED ; \
    fi

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base-amd64
FROM nvcr.io/nvidia/pytorch:25.08-py3 AS base-arm64
ARG TARGETARCH
FROM base-${TARGETARCH}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    curl \
    build-essential \
    libgl1 \
    libsdl2-2.0-0 \
    libomp5 \
    libvulkan1 \
    vulkan-tools \
    mesa-vulkan-drivers \
    libx11-6 \
    libxext6 \
    libxrandr2 \
    libxinerama1 \
    libxi6 \
    libxcursor1 \
    xdg-user-dirs \
    && rm -rf /var/lib/apt/lists/*

COPY --from=carla-fetch /opt/carla /opt/carla
ENV CARLA_ROOT=/opt/carla

# Install Rust toolchain (required for utils_rs maturin build)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

COPY . /repo

# Configure uv
ENV UV_LINK_MODE=copy

# Compile protos
WORKDIR /repo/src/grpc
RUN --mount=type=secret,id=netrc,target=/root/.netrc \
    --mount=type=cache,target=/root/.cache/uv \
    sh -c 'if [ -f /root/.netrc ]; then export NETRC=/root/.netrc; fi && uv sync'
RUN uv run compile-protos --no-sync

WORKDIR /repo

RUN --mount=type=secret,id=netrc,target=/root/.netrc \
    --mount=type=cache,target=/root/.cache/uv \
    sh -c 'if [ -f /root/.netrc ]; then export NETRC=/root/.netrc; fi && uv sync --extra all'

ENV UV_CACHE_DIR=/tmp/uv-cache
ENV UV_NO_SYNC=1
