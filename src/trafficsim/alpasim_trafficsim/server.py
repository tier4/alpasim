# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""gRPC entry point for the Alpasim trafficsim micro-service.

Implements the alpasim_grpc TrafficService contract backed by CARLA
TrafficManager. The scenario file passed via `--scenario` selects the CARLA
Town map and the traffic spawn rules; the actual spawning is delegated to
`scenario_runner` (which wraps autoware_carla_scenario).
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
from concurrent import futures
from typing import Optional

from alpasim_grpc.v0 import common_pb2, traffic_pb2, traffic_pb2_grpc

import grpc

from . import __version__ as ts_version
from .carla_session import CarlaSession
from .scenario_runner import ScenarioRunner

logger = logging.getLogger(__name__)


class TrafficSimServicer(traffic_pb2_grpc.TrafficServiceServicer):
    """CARLA-TrafficManager backed implementation of TrafficService."""

    def __init__(
        self,
        carla_host: str,
        carla_port: int,
        tm_port: int,
        scenario_path: Optional[str],
    ) -> None:
        self._carla_host = carla_host
        self._carla_port = carla_port
        self._tm_port = tm_port
        self._scenario_path = scenario_path
        self._sessions: dict[str, CarlaSession] = {}
        # gRPC server uses a ThreadPoolExecutor, so start/close/simulate may run
        # concurrently on different threads. The lock guards _sessions and the
        # session-creation critical section.
        self._sessions_lock = threading.Lock()
        try:
            import carla  # type: ignore
        except ImportError:
            logger.warning(
                "carla Python API not importable; server will reject start_session"
            )
            carla = None  # type: ignore
        self._carla_module = carla
        self._scenario_runner = ScenarioRunner(scenario_path) if scenario_path else None

    def start_session(self, request: traffic_pb2.TrafficSessionRequest, context):
        if self._carla_module is None:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, "carla Python API unavailable"
            )
            return common_pb2.SessionRequestStatus()  # unreachable; defensive

        with self._sessions_lock:
            if request.session_uuid in self._sessions:
                context.abort(
                    grpc.StatusCode.ALREADY_EXISTS,
                    f"session {request.session_uuid} already open",
                )
                return common_pb2.SessionRequestStatus()  # unreachable; defensive

        # Resolve the effective CARLA tick interval so ``session.open()`` applies
        # the right value to the world settings: the caller's request wins (so
        # alpasim can force lock-step); otherwise fall back to the scenario's
        # value; otherwise CarlaSession's dataclass default.
        if request.tick_interval_us > 0:
            fixed_delta_seconds = request.tick_interval_us / 1e6
        elif self._scenario_runner is not None:
            fixed_delta_seconds = self._scenario_runner.fixed_delta_seconds
        else:
            fixed_delta_seconds = CarlaSession.__dataclass_fields__[
                "fixed_delta_seconds"
            ].default

        session = CarlaSession(
            session_uuid=request.session_uuid,
            map_id=request.map_id,
            carla_host=self._carla_host,
            carla_port=self._carla_port,
            tm_port=self._tm_port,
            fixed_delta_seconds=fixed_delta_seconds,
        )
        session.open(self._carla_module)

        if self._scenario_runner is not None:
            self._scenario_runner.apply(
                session=session,
                request=request,
                carla_module=self._carla_module,
            )

        with self._sessions_lock:
            self._sessions[request.session_uuid] = session
        logger.info(
            "session %s started (%d actors)", request.session_uuid, len(session.actors)
        )
        return common_pb2.SessionRequestStatus()

    def simulate(self, request: traffic_pb2.TrafficRequest, context):
        with self._sessions_lock:
            session = self._sessions.get(request.session_uuid)
        if session is None:
            context.abort(
                grpc.StatusCode.NOT_FOUND, f"unknown session {request.session_uuid}"
            )
            return traffic_pb2.TrafficReturn()  # unreachable; defensive

        for update in request.object_trajectory_updates:
            session.apply_pose_update(update)

        session.tick_until(request.time_query_us)
        return session.snapshot()

    def close_session(self, request: traffic_pb2.TrafficSessionCloseRequest, context):
        with self._sessions_lock:
            session = self._sessions.pop(request.session_uuid, None)
        if session is not None:
            session.close()
        return common_pb2.Empty()

    def get_metadata(self, request: common_pb2.Empty, context):
        metadata = traffic_pb2.TrafficModuleMetadata(
            minimum_history_length_us=int(1e6),
        )
        metadata.version_id.version_id = ts_version
        metadata.version_id.git_hash = os.environ.get("ALPASIM_GIT_HASH", "")
        if self._scenario_runner is not None:
            for map_id in self._scenario_runner.supported_map_ids():
                metadata.supported_map_ids.append(map_id)
        return metadata


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Alpasim trafficsim gRPC server (CARLA TrafficManager backend)"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="bind address for the gRPC server"
    )
    parser.add_argument(
        "--port", type=int, required=True, help="bind port for the gRPC server"
    )
    parser.add_argument(
        "--carla-host", default="physics-0", help="CARLA Server hostname"
    )
    parser.add_argument(
        "--carla-port", type=int, default=2000, help="CARLA Server RPC port"
    )
    parser.add_argument(
        "--tm-port", type=int, default=8000, help="CARLA TrafficManager port"
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="path to a Hydra YAML scenario file (autoware_carla_scenario format)",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    parser.add_argument(
        "--max-workers", type=int, default=8, help="gRPC thread-pool size"
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    servicer = TrafficSimServicer(
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        tm_port=args.tm_port,
        scenario_path=args.scenario,
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.max_workers))
    traffic_pb2_grpc.add_TrafficServiceServicer_to_server(servicer, server)
    bind_addr = f"{args.host}:{args.port}"
    server.add_insecure_port(bind_addr)
    server.start()
    logger.info("trafficsim_server listening on %s", bind_addr)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("received SIGINT, shutting down")
        server.stop(grace=5.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
