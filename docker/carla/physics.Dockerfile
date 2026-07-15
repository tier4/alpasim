# syntax=docker/dockerfile:1.7
# Physics container image with CARLA 0.9.16 Server bundled.
#
# Built as a slim, multi-stage image matching grpc.Dockerfile:
#   - alpasim-base dependency dropped; runtime is the CUDA cuDNN runtime image
#     so warp-lang's GPU kernels run.
#   - CARLA 0.9.16 Server (Unreal binary) at /opt/carla, AMD64 only.
#   - Vulkan / SDL2 / libomp / X11 system libs CARLA needs at runtime.
#
# The entrypoint (docker/carla/entrypoint_physics.sh) launches CarlaUE4.sh in
# parallel with carla_physics_server (which lives here in docker/carla/ rather
# than in src/ — the alpasim service side must run without CARLA installed).
#
# Build (from repo root):
#   docker build -f docker/carla/physics.Dockerfile -t alpasim-physics-carla .
#
# Override the CUDA runtime base or Python version if needed:
#   docker build -f docker/carla/physics.Dockerfile \
#     --build-arg CUDA_RUNTIME_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 \
#     -t alpasim-physics-carla .
#
# Used by the `physics=carla` wizard profile.

ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ARG PYTHON_VERSION=3.11

# `alpasim_utils` depends on `trajdata-alpasim`, which is not on PyPI -- the
# workspace root pyproject.toml pins it to a git rev. The wheel METADATA does
# not encode that override, so we pass the git URL explicitly when installing
# the wheels into the runtime venv. Bump the rev here in lockstep with the
# workspace pyproject.toml entry.
ARG TRAJDATA_REQ="trajdata-alpasim @ git+https://github.com/NVlabs/trajdata@3caf3a8bd1a68f4a1545352aea82c535be9510b2"

# Pinned via Renovate / manual bump; COPY --from cannot resolve ARGs.
FROM ghcr.io/astral-sh/uv:0.5.18 AS uv

# ---- carla-fetch: download the CARLA Unreal binary on AMD64 ----------------
FROM ubuntu:22.04 AS carla-fetch
ARG CARLA_VERSION=0.9.16
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl tar ca-certificates xz-utils \
    && rm -rf /var/lib/apt/lists/*
# BuildKit cache mount keeps the ~7 GB CARLA tarball between builds so we do
# not re-download it whenever this stage is invalidated (e.g. base image
# refresh, CARLA_VERSION bump for the same arch already fetched, etc.).
RUN --mount=type=cache,target=/carla-cache,id=carla-${CARLA_VERSION}-${TARGETARCH},sharing=locked \
    mkdir -p /opt/carla && \
    arch="${TARGETARCH:-amd64}" ; \
    if [ "$arch" = "amd64" ]; then \
      tarball="/carla-cache/CARLA_${CARLA_VERSION}.tar.gz" ; \
      url="https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_${CARLA_VERSION}.tar.gz" ; \
      # The upstream S3 mirror throttles + drops connections mid-transfer.
      # Loop curl with -C - (resume) until gzip integrity passes so partial
      # progress in the buildkit cache accumulates across attempts.
      attempt=0 ; max_attempts=30 ; \
      while ! gzip -t "$tarball" 2>/dev/null; do \
        attempt=$((attempt + 1)) ; \
        if [ "$attempt" -gt "$max_attempts" ]; then \
          echo "Gave up downloading CARLA after $max_attempts attempts" >&2 ; \
          exit 1 ; \
        fi ; \
        have=$(stat -c %s "$tarball" 2>/dev/null || echo 0) ; \
        echo "CARLA download attempt #$attempt (have $have bytes)" >&2 ; \
        curl -fL -C - --retry 5 --retry-delay 10 --retry-all-errors \
             -o "$tarball" "$url" || true ; \
      done ; \
      echo "Extracting CARLA_${CARLA_VERSION}.tar.gz..." >&2 ; \
      tar -xzf "$tarball" -C /opt/carla --strip-components=0 ; \
    else \
      echo "CARLA Server is not built for ${arch}; skipping" > /opt/carla/UNSUPPORTED ; \
    fi

# ---- builder: produce wheels for alpasim_grpc + alpasim_utils + alpasim_physics
#                + carla_physics_server
FROM ubuntu:22.04 AS builder
ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PYTHON=${PYTHON_VERSION}

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

# alpasim_grpc's hatch_build.py runs grpc_tools.protoc, so .proto files are
# compiled into *_pb2.py / *_pb2_grpc.py and packed into the wheel. The other
# packages use setuptools and have no generated sources. Each package is built
# from its own source dir; `tool.uv.sources` workspace pins only matter at
# install/sync time. carla_physics_server lives in docker/carla/ because it
# imports the CARLA Python API — src/ must stay CARLA-free.
WORKDIR /build
COPY src/grpc /build/grpc
COPY src/utils /build/utils
COPY src/physics /build/physics
COPY docker/carla/carla_physics_server /build/carla_physics_server

RUN --mount=type=cache,target=/root/.cache/uv \
    uv build --wheel --out-dir /wheels /build/grpc \
 && uv build --wheel --out-dir /wheels /build/utils \
 && uv build --wheel --out-dir /wheels /build/physics \
 && uv build --wheel --out-dir /wheels /build/carla_physics_server

# ---- runtime: CUDA-enabled physics + CARLA ---------------------------------
FROM ${CUDA_RUNTIME_IMAGE} AS runtime
ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    PATH="/opt/alpasim-physics/bin:${PATH}"

# CARLA Unreal binary runtime libs (+ git so we can install trajdata-alpasim
# from the pinned source git URL alongside the wheels).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
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

COPY --from=uv /uv /usr/local/bin/

RUN uv venv --python ${PYTHON_VERSION} /opt/alpasim-physics

COPY --from=builder /wheels /wheels
ARG TRAJDATA_REQ
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/alpasim-physics/bin/python \
        "${TRAJDATA_REQ}" \
        /wheels/alpasim_grpc-*.whl \
        /wheels/alpasim_utils-*.whl \
        /wheels/alpasim_physics-*.whl \
        /wheels/carla_physics_server-*.whl \
    && rm -rf /wheels

COPY --from=carla-fetch /opt/carla /opt/carla
ENV CARLA_ROOT=/opt/carla

# Bake the CARLA + carla_physics_server launcher. Copied directly from
# docker/carla so updates to the script rebuild only this layer.
COPY docker/carla/entrypoint_physics.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

RUN /opt/alpasim-physics/bin/python -c \
    "import alpasim_physics, alpasim_grpc, alpasim_utils, carla_physics_server, numpy, warp"

# Run CARLA as a non-root user. CarlaUE4 writes into $CARLA_ROOT/CarlaUE4/Saved
# and reads xdg-user-dirs from $HOME, so we grant the user a writable home
# and ownership of the CARLA install tree.
ARG CARLA_UID=1000
ARG CARLA_GID=1000
RUN groupadd --system --gid ${CARLA_GID} carla \
 && useradd  --system --uid ${CARLA_UID} --gid ${CARLA_GID} \
             --home-dir /home/carla --create-home --shell /usr/sbin/nologin carla \
 && mkdir -p /work \
 && chown -R carla:carla /opt/carla /work /home/carla

USER carla:carla
ENV HOME=/home/carla

WORKDIR /work
ENTRYPOINT ["/opt/entrypoint.sh"]
CMD []
