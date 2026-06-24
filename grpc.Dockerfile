# syntax=docker/dockerfile:1.7

# Slim image that ships only the alpasim_grpc Python package -- proto stubs and
# Python client/server bindings. No CUDA, no PyTorch, no Rust toolchain.
# Downstream E2E stacks that only need to speak gRPC to alpasim should depend
# on this image instead of the main alpasim-base image.
#
# Build (context is src/grpc, not the repo root):
#   docker build -f grpc.Dockerfile -t alpasim-grpc-slim src/grpc

ARG PYTHON_VERSION=3.11

# Pinned via Renovate / manual bump; COPY --from cannot resolve ARGs.
FROM ghcr.io/astral-sh/uv:0.5.18 AS uv

# ---- builder ---------------------------------------------------------------
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

WORKDIR /build
COPY . /build/

# hatch_build.py invokes grpc_tools.protoc, so .proto files are compiled into
# *_pb2.py / *_pb2_grpc.py / *.pyi and packed into the wheel automatically.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv build --wheel --out-dir /wheels

# ---- runtime ---------------------------------------------------------------
FROM ubuntu:22.04 AS runtime

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    PATH="/opt/alpasim-grpc/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/

RUN uv venv --python ${PYTHON_VERSION} /opt/alpasim-grpc

COPY --from=builder /wheels /wheels
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/alpasim-grpc/bin/python /wheels/*.whl \
    && rm -rf /wheels

RUN /opt/alpasim-grpc/bin/python -c "import alpasim_grpc; import alpasim_grpc.v0.egodriver_pb2"

WORKDIR /work
CMD ["python3"]
