# syntax=docker/dockerfile:1.7
# Slim physics image: runs physics_server alone on the alpasim-grpc-slim base.
# No CARLA Unreal binary; installs only the Python physics runtime on top of
# the gRPC slim image.
#
# Build (from repo root):
#   docker build -f docker/physics/simple.Dockerfile -t alpasim-physics-simple .
#
# Override the grpc slim base if needed:
#   docker build -f docker/physics/simple.Dockerfile \
#     --build-arg GRPC_SLIM_IMAGE=ghcr.io/tier4/alpasim-grpc-slim:latest \
#     -t alpasim-physics-simple .

ARG GRPC_SLIM_IMAGE=alpasim-grpc-slim:dev
ARG PYTHON_VERSION=3.11
ARG TRAJDATA_REQ="trajdata-alpasim @ git+https://github.com/NVlabs/trajdata@3caf3a8bd1a68f4a1545352aea82c535be9510b2"

# Pinned via Renovate / manual bump; COPY --from cannot resolve ARGs.
FROM ghcr.io/astral-sh/uv:0.5.18 AS uv

# ---- builder: produce wheels for alpasim_utils + alpasim_physics -----------
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

# Each package is built independently from its own source dir. `tool.uv.sources`
# workspace pins only matter at install/sync time -- wheel building uses the
# declared build backend (setuptools/hatchling) which does not need workspace
# resolution.
WORKDIR /build
COPY src/utils /build/utils
COPY src/physics /build/physics

RUN --mount=type=cache,target=/root/.cache/uv \
    uv build --wheel --out-dir /wheels /build/utils \
 && uv build --wheel --out-dir /wheels /build/physics

# ---- runtime: install wheels onto slim grpc base ---------------------------
FROM ${GRPC_SLIM_IMAGE} AS runtime
ARG PYTHON_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/
COPY --from=builder /wheels /wheels

ARG TRAJDATA_REQ
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/alpasim-grpc/bin/python \
        "${TRAJDATA_REQ}" \
        /wheels/alpasim_utils-*.whl \
        /wheels/alpasim_physics-*.whl \
    && rm -rf /wheels

RUN /opt/alpasim-grpc/bin/python -c \
    "import alpasim_physics, alpasim_grpc, alpasim_utils, numpy, warp"

WORKDIR /work
ENV PATH="/opt/alpasim-grpc/bin:${PATH}"
CMD ["physics_server"]
