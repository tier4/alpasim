# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import argparse
import functools
import logging
import threading
from concurrent import futures

from alpasim_grpc.v0.common_pb2 import (
    AvailableScenesReturn,
    Empty,
    Pose,
    PoseAtTime,
    SessionRequestStatus,
    Trajectory,
    VersionId,
)
from alpasim_grpc.v0.physics_pb2 import (
    PhysicsGroundIntersectionRequest,
    PhysicsGroundIntersectionReturn,
    PhysicsSessionCloseRequest,
    PhysicsSessionRequest,
)
from alpasim_grpc.v0.physics_pb2_grpc import (
    PhysicsServiceServicer,
    add_PhysicsServiceServicer_to_server,
)
from alpasim_physics import VERSION_MESSAGE
from alpasim_physics.backend import PhysicsBackend
from alpasim_physics.carla_clock import CarlaClock
from alpasim_physics.utils import (
    aabb_to_ndarray,
    ndarray_to_vec3,
    pose_grpc_to_ndarray,
    pose_status_to_grpc,
    scipy_to_quat,
)
from alpasim_utils.artifact import Artifact
from scipy.spatial.transform import Rotation as R

import grpc

logger = logging.getLogger(__name__)


class PhysicsSimService(PhysicsServiceServicer):
    def __init__(
        self,
        artifact_glob: str,
        carla_host: str,
        carla_port: int,
        cache_size: int = 2,
        use_ground_mesh: bool = False,
        visualize: bool = False,
    ) -> None:
        self._carla_host = carla_host
        self._carla_port = carla_port
        self._carla_module = None  # lazily imported on first session
        self._clocks_lock = threading.Lock()
        self._clocks: dict[str, CarlaClock] = {}
        self.artifacts = Artifact.discover_from_glob(
            artifact_glob, use_ground_mesh=use_ground_mesh
        )

        self.visualize = visualize
        logger.info(f"Available scenes: {list(self.artifacts.keys())}.")

        # instantiate the method here to avoid caching `self`
        @functools.lru_cache(maxsize=cache_size)
        def get_backend(scene_id: str) -> PhysicsBackend:
            if scene_id not in self.artifacts:
                raise KeyError(f"Scene {scene_id=} not available.")

            artifact = self.artifacts[scene_id]
            logger.info(f"Cache miss, loading {artifact.scene_id=}")
            mesh_ply = artifact.mesh_ply

            # don't keep the .ply file once the backend is evicted from the lru_cache
            artifact.clear_cache()

            return PhysicsBackend(
                mesh_ply,
                visualize=self.visualize,
            )

        self.get_backend = get_backend

    def _ensure_carla_module(self):
        if self._carla_module is None:
            import carla  # heavy; imported only when a session actually needs it

            self._carla_module = carla
        return self._carla_module

    def start_session(
        self, request: PhysicsSessionRequest, context: grpc.ServicerContext
    ) -> SessionRequestStatus:
        if request.tick_interval_us == 0:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "tick_interval_us must be non-zero — physics owns CARLA's tick cadence",
            )
        clock = CarlaClock(
            session_uuid=request.session_uuid,
            carla_host=self._carla_host,
            carla_port=self._carla_port,
            fixed_delta_seconds=request.tick_interval_us / 1e6,
        )
        clock.open(self._ensure_carla_module())
        with self._clocks_lock:
            if request.session_uuid in self._clocks:
                context.abort(
                    grpc.StatusCode.ALREADY_EXISTS,
                    f"session {request.session_uuid} already open",
                )
            self._clocks[request.session_uuid] = clock
        logger.info(
            "session %s: physics clock opened (fixed_delta=%.6fs)",
            request.session_uuid,
            clock.fixed_delta_seconds,
        )
        return SessionRequestStatus()

    def close_session(
        self, request: PhysicsSessionCloseRequest, context: grpc.ServicerContext
    ) -> SessionRequestStatus:
        with self._clocks_lock:
            clock = self._clocks.pop(request.session_uuid, None)
        if clock is not None:
            clock.close()
            logger.info("session %s: physics clock closed", request.session_uuid)
        return SessionRequestStatus()

    def ground_intersection(
        self, request: PhysicsGroundIntersectionRequest, context: grpc.ServicerContext
    ) -> PhysicsGroundIntersectionReturn:
        logger.debug(f"Received request for scene_id={request.scene_id}")
        logger.debug(f"full request={request}")
        try:
            backend = self.get_backend(request.scene_id)

            other_updates = []
            for other in request.other_objects:
                other_updates.append(
                    backend.update_pose(
                        pose_grpc_to_ndarray(other.pose_pair.future_pose),
                        aabb_to_ndarray(other.aabb),
                        request.future_us,
                    )
                )

            if request.HasField("ego_data"):
                ego_aabb = aabb_to_ndarray(request.ego_data.aabb)
                corrected_poses = []
                ego_statuses = []
                for pat in request.ego_data.ego_trajectory_aabb.poses:
                    pose_nd = pose_grpc_to_ndarray(pat.pose)
                    updated_pose, status = backend.update_pose(
                        pose_nd, ego_aabb, pat.timestamp_us
                    )
                    quat = R.from_matrix(updated_pose[:3, :3]).as_quat(canonical=False)
                    corrected_poses.append(
                        PoseAtTime(
                            pose=Pose(
                                vec=ndarray_to_vec3(updated_pose[:3, 3]),
                                quat=scipy_to_quat(quat),
                            ),
                            timestamp_us=pat.timestamp_us,
                        )
                    )
                    ego_statuses.append(status.to_grpc())

                response = PhysicsGroundIntersectionReturn(
                    ego_trajectory_aabb=Trajectory(poses=corrected_poses),
                    ego_status=ego_statuses,
                    other_poses=[
                        pose_status_to_grpc(pose, status)
                        for pose, status in other_updates
                    ],
                )
            else:
                response = PhysicsGroundIntersectionReturn(
                    other_poses=[
                        pose_status_to_grpc(pose, status)
                        for pose, status in other_updates
                    ],
                )
            if request.advance_world_to_us != 0:
                with self._clocks_lock:
                    clock = self._clocks.get(request.session_uuid)
                if clock is None:
                    context.abort(
                        grpc.StatusCode.FAILED_PRECONDITION,
                        f"advance_world_to_us set but session {request.session_uuid} "
                        "has no open CARLA clock; call start_session first",
                    )
                clock.advance_to(request.advance_world_to_us)
            logger.debug("sending response")
            return response
        except Exception as e:
            context.set_code(grpc.StatusCode.UNKNOWN)
            context.set_details(str(e))
            raise

    def get_version(self, request: Empty, context: grpc.ServicerContext) -> VersionId:
        logger.info("get_version")
        try:
            return VERSION_MESSAGE
        except Exception as e:
            context.set_code(grpc.StatusCode.UNKNOWN)
            context.set_details(str(e))
            raise

    def get_available_scenes(
        self, request: Empty, context: grpc.ServicerContext
    ) -> AvailableScenesReturn:
        logger.info("get_available_scenes")
        return AvailableScenesReturn(scene_ids=list(self.artifacts.keys()))


def parse_args(
    arg_list: list[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-glob",
        type=str,
        help="Glob expression to find artifacts. Must end in .usdz to find relevant files.",
        required=True,
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--carla-host",
        type=str,
        default="localhost",
        help="Host where the CARLA server listens (same container in the default topology).",
    )
    parser.add_argument(
        "--carla-port",
        type=int,
        default=2000,
        help="Port where the CARLA server listens.",
    )
    parser.add_argument("--use-ground-mesh", type=bool, default=False)
    parser.add_argument("--visualize", type=bool, default=False)
    parser.add_argument(
        "--cache-size",
        type=int,
        default=16,
        help="Number of scene backends to keep in LRU cache. Set to match or exceed concurrent scenes.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    args, overrides = parser.parse_known_args(arg_list)
    return args, overrides


def main(arg_list: list[str] | None = None) -> None:
    args, _ = parse_args(arg_list)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info(f"Identifying as\n{VERSION_MESSAGE}")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    address = f"{args.host}:{args.port}"
    server.add_insecure_port(address)

    service = PhysicsSimService(
        args.artifact_glob,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        cache_size=args.cache_size,
        use_ground_mesh=args.use_ground_mesh,
        visualize=args.visualize,
    )
    add_PhysicsServiceServicer_to_server(service, server)

    logger.info(f"Serving on {address}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    main()
