#!/usr/bin/env bash
# Entrypoint for the physics container.
#
# When CARLA_ENABLED=true, launches the CARLA Server (Unreal binary at
# $CARLA_ROOT/CarlaUE4.sh) alongside the physics_server gRPC service. The two
# processes share a single container; this entrypoint forwards SIGTERM/SIGINT
# to both children and exits when either one dies.
set -euo pipefail

CARLA_ENABLED="${CARLA_ENABLED:-false}"
CARLA_PORT="${CARLA_PORT:-2000}"
CARLA_ROOT="${CARLA_ROOT:-/opt/carla}"

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

if [ "$CARLA_ENABLED" = "true" ]; then
  if [ ! -x "$CARLA_ROOT/CarlaUE4.sh" ]; then
    echo "CARLA_ENABLED=true but $CARLA_ROOT/CarlaUE4.sh is missing or not executable" >&2
    exit 1
  fi
  echo "[entrypoint] starting CARLA Server on port $CARLA_PORT"
  "$CARLA_ROOT/CarlaUE4.sh" -RenderOffScreen -nosound -carla-rpc-port="$CARLA_PORT" &
  pids+=("$!")
fi

echo "[entrypoint] starting physics_server: $*"
uv run physics_server "$@" &
pids+=("$!")

# Exit as soon as any child terminates so docker can restart the container.
wait -n "${pids[@]}"
exit_code=$?
cleanup
exit "$exit_code"
