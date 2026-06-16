"""Per-Alpasim-session CARLA client + TrafficManager lifecycle.

One CarlaSession instance corresponds to one TrafficSessionRequest from
Runtime. It holds the CARLA Client/World/TrafficManager handles and the list
of spawned actors so they can be cleaned up on close_session.

This module is structured so that unit tests can pass a mocked `carla`
module via the `carla_module` argument.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from alpasim_grpc.v0 import traffic_pb2

from .grpc_adapter import (
    actor_bounding_box_to_grpc,
    carla_transform_to_world_pose,
    grpc_pose_to_carla_transform,
    world_pose_to_grpc,
)

logger = logging.getLogger(__name__)


@dataclass
class SpawnedActor:
    """Bookkeeping for one actor controlled by this session."""

    object_id: str
    actor: Any
    is_ego: bool = False
    is_static: bool = False


@dataclass
class CarlaSession:
    """Holds the CARLA-side state for a single Alpasim traffic session."""

    session_uuid: str
    map_id: str
    carla_host: str
    carla_port: int
    tm_port: int
    fixed_delta_seconds: float = 0.05

    client: Any = None
    world: Any = None
    traffic_manager: Any = None
    actors: list[SpawnedActor] = field(default_factory=list)
    last_time_query_us: int = 0

    def open(self, carla_module) -> None:
        """Connect to CARLA, load the map and switch to synchronous mode."""
        logger.info(
            "session %s: connecting to CARLA at %s:%d (tm=%d)",
            self.session_uuid,
            self.carla_host,
            self.carla_port,
            self.tm_port,
        )
        self.client = carla_module.Client(self.carla_host, self.carla_port)
        self.client.set_timeout(30.0)
        # The map_id Runtime sends is a scene UUID; the concrete CARLA Town
        # is selected by the scenario YAML in scenario_runner. Here we just
        # take the world that's already loaded.
        self.world = self.client.get_world()
        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        self.traffic_manager.set_synchronous_mode(True)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(settings)

    def register_actor(self, object_id: str, actor: Any, *, is_ego: bool = False, is_static: bool = False) -> None:
        self.actors.append(SpawnedActor(object_id=object_id, actor=actor, is_ego=is_ego, is_static=is_static))

    def apply_ego_update(self, update: traffic_pb2.ObjectTrajectoryUpdate) -> None:
        """Set the ego actor's transform from the last pose in the update."""
        ego = next((a for a in self.actors if a.is_ego), None)
        if ego is None or not update.trajectory.poses:
            return
        target_pose = update.trajectory.poses[-1].pose
        ego.actor.set_transform(grpc_pose_to_carla_transform(target_pose))

    def tick_until(self, target_time_us: int) -> None:
        """Advance the CARLA world until `target_time_us`."""
        if self.last_time_query_us == 0:
            # First tick after start_session — single tick to materialise spawns.
            self.world.tick()
            self.last_time_query_us = target_time_us
            return

        delta_us = max(target_time_us - self.last_time_query_us, 0)
        step_us = int(self.fixed_delta_seconds * 1e6)
        steps = max(1, delta_us // step_us) if step_us > 0 else 1
        for _ in range(steps):
            self.world.tick()
        self.last_time_query_us = target_time_us

    def snapshot(self) -> traffic_pb2.TrafficReturn:
        """Collect the current pose of every registered actor."""
        from alpasim_grpc.v0 import common_pb2

        updates: list[traffic_pb2.ObjectTrajectoryUpdate] = []
        for entry in self.actors:
            transform = entry.actor.get_transform()
            world_pose = carla_transform_to_world_pose(transform)
            pose_msg = world_pose_to_grpc(world_pose)
            updates.append(
                traffic_pb2.ObjectTrajectoryUpdate(
                    object_id=entry.object_id,
                    trajectory=common_pb2.Trajectory(
                        poses=[
                            common_pb2.PoseAtTime(
                                pose=pose_msg,
                                timestamp_us=self.last_time_query_us,
                            )
                        ]
                    ),
                )
            )
        return traffic_pb2.TrafficReturn(object_trajectory_updates=updates)

    def bounding_box_for(self, object_id: str):
        """Return the AABB proto for a registered actor (used by metadata)."""
        for entry in self.actors:
            if entry.object_id == object_id:
                return actor_bounding_box_to_grpc(entry.actor)
        return None

    def close(self) -> None:
        """Destroy actors and restore async mode. Safe to call multiple times."""
        logger.info("session %s: closing (%d actors)", self.session_uuid, len(self.actors))
        for entry in self.actors:
            try:
                entry.actor.destroy()
            except Exception:  # noqa: BLE001
                logger.exception("failed to destroy actor %s", entry.object_id)
        self.actors.clear()
        if self.world is not None:
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
            except Exception:  # noqa: BLE001
                logger.exception("failed to restore async world settings")
        if self.traffic_manager is not None:
            try:
                self.traffic_manager.set_synchronous_mode(False)
            except Exception:  # noqa: BLE001
                logger.exception("failed to disable TM sync mode")
