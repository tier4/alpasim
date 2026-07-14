# Standalone image for the Alpasim trafficsim micro-service.
#
# Lives separately from the base alpasim image because trafficsim is pinned
# to Python 3.10 (autoware_carla_scenario depends on lanelet2-python-api-for-
# autoware which only ships a CPython 3.10 ABI wheel), while the rest of the
# workspace targets Python 3.11+.
#
# Build (from repo root):
#   docker build -f trafficsim.Dockerfile -t alpasim-trafficsim .

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=python3.10

# System deps:
#   - python3.10 + venv for the runtime
#   - libgl1 / libsdl2 needed by carla Python API client
#   - git for autoware_carla_scenario git+subdirectory install
#   - cmake + boost + eigen + geographic + pugixml to compile
#     lanelet2_python_api_for_autoware from git source
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 \
      python3.10-dev \
      python3.10-venv \
      ca-certificates \
      curl \
      git \
      build-essential \
      cmake \
      libboost-dev \
      libboost-filesystem-dev \
      libboost-iostreams-dev \
      libboost-program-options-dev \
      libboost-python-dev \
      libboost-serialization-dev \
      libboost-system-dev \
      libboost-thread-dev \
      libeigen3-dev \
      libgeographic-dev \
      libpugixml-dev \
      librange-v3-dev \
      libgl1 \
      libsdl2-2.0-0 \
      libomp5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY src/grpc /repo/src/grpc
COPY src/trafficsim /repo/src/trafficsim

# Compile protos against the grpc package first so the trafficsim install can
# resolve `alpasim_grpc` as a path source.
WORKDIR /repo/src/grpc
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync
RUN uv run compile-protos --no-sync

WORKDIR /repo/src/trafficsim
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra carla --extra dev

ENV PATH="/repo/src/trafficsim/.venv/bin:${PATH}"
ENV UV_NO_SYNC=1

ENTRYPOINT ["uv", "run", "trafficsim_server"]
