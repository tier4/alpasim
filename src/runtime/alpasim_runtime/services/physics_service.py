# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Physics service implementation."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Type

from alpasim_grpc.v0 import common_pb2
from alpasim_grpc.v0.logging_pb2 import LogEntry
from alpasim_grpc.v0.physics_pb2 import (
    PhysicsGroundIntersectionRequest,
    PhysicsSessionCloseRequest,
    PhysicsSessionRequest,
)
from alpasim_grpc.v0.physics_pb2_grpc import PhysicsServiceStub
from alpasim_runtime.services.service_base import ServiceBase, SessionInfo
from alpasim_runtime.services.session_configs import PhysicsSessionConfig
from alpasim_runtime.telemetry.rpc_wrapper import profiled_rpc_call
from alpasim_utils import geometry
from alpasim_utils.scenario import AABB

logger = logging.getLogger(__name__)


class PhysicsService(ServiceBase[PhysicsServiceStub]):
    """
    Physics service implementation that handles both real and skip modes.

    Physics is responsible for ground intersection calculations,
    determining how objects interact with the ground plane. It also owns
    CARLA's world clock: ``start_session`` configures CARLA's
    ``fixed_delta_seconds`` and each ego-facing ``ground_intersection``
    call advances the world by exactly one control step.
    """

    @property
    def stub_class(self) -> Type[PhysicsServiceStub]:
        return PhysicsServiceStub

    async def _initialize_session(self, session_info: SessionInfo) -> None:
        """Open the physics-side CARLA client so it owns the world tick.

        Skip mode is a no-op — physics won't tick CARLA and trafficsim
        won't either; the runtime is replaying poses and never advances
        the CARLA world.
        """
        if self.skip:
            logger.debug("Skip mode: physics start_session no-op")
            return

        cfg = session_info.session_config
        if not isinstance(cfg, PhysicsSessionConfig):
            raise TypeError(
                "PhysicsService._initialize_session requires a PhysicsSessionConfig "
                f"via session_config, got {type(cfg).__name__}."
            )

        session_request = PhysicsSessionRequest(
            session_uuid=session_info.uuid,
            tick_interval_us=cfg.control_timestep_us,
        )
        await session_info.broadcaster.broadcast(
            LogEntry(physics_session_request=session_request)
        )
        await profiled_rpc_call(
            "start_session", "physics", self.stub.start_session, session_request
        )

    async def _cleanup_session(self, session_info: SessionInfo) -> None:
        """Close the physics-side CARLA client."""
        if self.skip:
            logger.debug("Skip mode: physics close_session no-op")
            return

        close_request = PhysicsSessionCloseRequest(session_uuid=session_info.uuid)
        await profiled_rpc_call(
            "close_session", "physics", self.stub.close_session, close_request
        )

    async def ground_intersection(
        self,
        scene_id: str,
        delta_start_us: int,
        delta_end_us: int,
        ego_trajectory_aabb: geometry.Trajectory,
        traffic_poses: Dict[str, geometry.Pose],
        ego_aabb: AABB,
        skip: bool = False,
        advance_world_to_us: int = 0,
    ) -> Tuple[geometry.Trajectory, Dict[str, geometry.Pose]]:
        """
        Calculate ground intersection for ego and traffic vehicles.

        Args:
            ego_trajectory_aabb: Ego trajectory in AABB coordinates to
                ground-correct.
            skip: If True, return traffic poses unchanged without
                making a gRPC call. Use this when objects are following
                trajectories that already have correct physics applied (e.g.,
                recorded ground truth or when traffic sim is skipped).
            advance_world_to_us: When non-zero, ask the physics server to
                also tick CARLA to this timestamp as part of this RPC. Set
                by the caller on exactly one call per control step (the
                ego-facing one, before trafficsim.simulate reads the CARLA
                snapshot).

        Returns:
            Tuple of (ego_trajectory, traffic_poses) after ground intersection.
            The ego trajectory preserves the input timestamps.
        """
        if self.skip or skip:
            return ego_trajectory_aabb, traffic_poses

        session_info = self._require_session_info()

        traffic_poses = traffic_poses or {}

        request = self._prepare_request(
            scene_id,
            delta_start_us,
            delta_end_us,
            ego_trajectory_aabb=ego_trajectory_aabb,
            other_poses=[geometry.pose_to_grpc(p) for p in traffic_poses.values()],
            ego_aabb=ego_aabb,
            session_uuid=session_info.uuid,
            advance_world_to_us=advance_world_to_us,
        )

        await session_info.broadcaster.broadcast(LogEntry(physics_request=request))

        response = await profiled_rpc_call(
            "ground_intersection", "physics", self.stub.ground_intersection, request
        )

        await session_info.broadcaster.broadcast(LogEntry(physics_return=response))

        ego_trajectory = geometry.trajectory_from_grpc(response.ego_trajectory_aabb)
        traffic_responses = {
            k: geometry.pose_from_grpc(v.pose)
            for k, v in zip(traffic_poses.keys(), response.other_poses, strict=True)
        }

        return ego_trajectory, traffic_responses

    def _prepare_request(
        self,
        scene_id: str,
        delta_start_us: int,
        delta_end_us: int,
        ego_trajectory_aabb: geometry.Trajectory,
        other_poses: List[common_pb2.Pose],
        ego_aabb: AABB,
        session_uuid: str,
        advance_world_to_us: int,
    ) -> PhysicsGroundIntersectionRequest:
        """Prepare the physics ground intersection request."""
        return PhysicsGroundIntersectionRequest(
            scene_id=scene_id,
            now_us=delta_start_us,
            future_us=delta_end_us,
            ego_data=PhysicsGroundIntersectionRequest.EgoData(
                aabb=ego_aabb.to_grpc(),
                ego_trajectory_aabb=geometry.trajectory_to_grpc(ego_trajectory_aabb),
            ),
            other_objects=[
                PhysicsGroundIntersectionRequest.OtherObject(
                    # TODO[RDL] extract AABB from NRE reconstruction, this
                    # is placeholder assuming all cars are equally sized
                    aabb=ego_aabb.to_grpc(),
                    pose_pair=PhysicsGroundIntersectionRequest.PosePair(
                        now_pose=other_pose, future_pose=other_pose
                    ),
                )
                for other_pose in other_poses
            ],
            session_uuid=session_uuid,
            advance_world_to_us=advance_world_to_us,
        )
