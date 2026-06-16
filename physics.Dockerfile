# Physics container image with CARLA 0.9.16 Server bundled.
#
# Extends the base alpasim-base image (same uv/workspace setup) and overlays:
#   - CARLA 0.9.16 Server (Unreal binary) at /opt/carla, AMD64 only.
#   - Vulkan / SDL2 / libomp / X11 system libs CARLA needs at runtime.
#
# The entrypoint (src/physics/alpasim_physics/entrypoint.sh, baked into the
# repo copy at /repo/src) launches CarlaUE4.sh in parallel with physics_server
# when CARLA_ENABLED=true.
#
# Build (from repo root):
#   docker build -f physics.Dockerfile -t alpasim-physics-carla \
#     --build-arg BASE_IMAGE=alpasim-base:<tag> .
#
# Used by the `physics=carla` wizard profile.

ARG BASE_IMAGE=alpasim-base:latest

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

FROM ${BASE_IMAGE}

# Runtime libs for the CARLA Unreal binary. Apt cache is preserved by the
# build host; clean up at the end for a smaller layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
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
