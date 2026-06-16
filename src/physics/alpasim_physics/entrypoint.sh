#!/usr/bin/env bash
# Entrypoint for the physics container WITH the CARLA Server bundled.
#
# Only used when the container is built from physics.Dockerfile (i.e. the
# `physics=carla` wizard profile). The default base image has no CARLA binary
# and runs `physics_server` directly without this wrapper.
#
# Responsibilities:
#   - Start CarlaUE4.sh and physics_server in parallel.
#   - Forward SIGINT / SIGTERM to both children.
#   - Exit as soon as either child dies so docker can restart the container.
set -euo pipefail

CARLA_PORT="${CARLA_PORT:-2000}"
CARLA_ROOT="${CARLA_ROOT:-/opt/carla}"

if [ ! -x "$CARLA_ROOT/CarlaUE4.sh" ]; then
  echo "physics carla entrypoint requires $CARLA_ROOT/CarlaUE4.sh (is this image built from physics.Dockerfile?)" >&2
  exit 1
fi

pids=()

cleanup() {
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  wait
}
trap cleanup INT TERM

echo "[entrypoint] starting CARLA Server on port $CARLA_PORT"
"$CARLA_ROOT/CarlaUE4.sh" -RenderOffScreen -nosound -carla-rpc-port="$CARLA_PORT" &
pids+=("$!")

echo "[entrypoint] starting physics_server: $*"
uv run physics_server "$@" &
pids+=("$!")

wait -n "${pids[@]}"
exit_code=$?
cleanup
exit "$exit_code"
