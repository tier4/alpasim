# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

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
from typing import Any, Optional

from alpasim_grpc.v0 import common_pb2, traffic_pb2

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
    # True when this actor's transform is overwritten from incoming
    # ObjectTrajectoryUpdate messages each simulate() tick (ControlMode
    # GRPC_REPLAY). Otherwise the actor is driven by CARLA TrafficManager
    # autopilot and incoming updates are ignored.
    is_grpc_driven: bool = False


@dataclass
class CarlaSession:
    """Holds the CARLA-side state for a single Alpasim traffic session."""

    session_uuid: str
    map_id: str
    carla_host: str
    carla_port: int
    tm_port: int

    client: Any = None
    world: Any = None
    traffic_manager: Any = None
    actors: list[SpawnedActor] = field(default_factory=list)
    last_time_query_us: int | None = None
    _actors_by_id: dict[str, SpawnedActor] = field(default_factory=dict)

    def open(self, carla_module) -> None:
        """Connect to CARLA for spawning + pose read/write.

        The physics container owns CARLA's synchronous-mode and
        ``fixed_delta_seconds`` settings (and calls ``world.tick()`` each
        control step). This client attaches to the already-configured world
        and never ticks it.
        """
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

    def register_actor(
        self,
        object_id: str,
        actor: Any,
        *,
        is_ego: bool = False,
        is_static: bool = False,
        is_grpc_driven: Optional[bool] = None,
    ) -> None:
        # Back-compat: pre-ControlMode scenarios call register_actor without
        # is_grpc_driven and expect EGO to be gRPC-driven, everything else TM.
        if is_grpc_driven is None:
            is_grpc_driven = is_ego
        entry = SpawnedActor(
            object_id=object_id,
            actor=actor,
            is_ego=is_ego,
            is_static=is_static,
            is_grpc_driven=is_grpc_driven,
        )
        self.actors.append(entry)
        self._actors_by_id[object_id] = entry
        # Centralize the physics-disable invariant so scenarios don't each
        # have to remember it: an actor whose transform we overwrite each
        # tick must not also be simulated by CARLA's physics solver.
        if is_grpc_driven and not is_static and hasattr(actor, "set_simulate_physics"):
            actor.set_simulate_physics(False)

    def apply_pose_update(self, update: traffic_pb2.ObjectTrajectoryUpdate) -> None:
        """Overwrite a gRPC-driven actor's transform from the update's last pose.

        Silently ignored when (a) the object_id isn't registered, (b) the
        registered actor is TM-driven, or (c) the update has no poses. This
        lets clients send updates for the entire fleet without the server
        needing to know which subset is gRPC-driven.
        """
        entry = self._actors_by_id.get(update.object_id)
        if entry is None or not entry.is_grpc_driven or not update.trajectory.poses:
            return
        target_pose = update.trajectory.poses[-1].pose
        entry.actor.set_transform(grpc_pose_to_carla_transform(target_pose))

    def note_time_query(self, target_time_us: int) -> None:
        """Record the caller's target time so snapshot() can stamp poses.

        Trafficsim no longer advances CARLA — the physics container ticks the
        world. We still track ``target_time_us`` so the snapshot's per-pose
        ``timestamp_us`` matches the caller's clock rather than CARLA's
        internal one.
        """
        self.last_time_query_us = target_time_us

    def snapshot(self) -> traffic_pb2.TrafficReturn:
        """Collect the current pose of every registered actor."""
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
                                timestamp_us=self.last_time_query_us or 0,
                            )
                        ]
                    ),
                )
            )
        return traffic_pb2.TrafficReturn(object_trajectory_updates=updates)

    def bounding_box_for(self, object_id: str):
        """Return the AABB proto for a registered actor (used by metadata)."""
        entry = self._actors_by_id.get(object_id)
        return actor_bounding_box_to_grpc(entry.actor) if entry is not None else None

    def close(self) -> None:
        """Destroy actors and restore async mode. Safe to call multiple times."""
        logger.info(
            "session %s: closing (%d actors)", self.session_uuid, len(self.actors)
        )
        for entry in self.actors:
            try:
                entry.actor.destroy()
            except Exception:  # noqa: BLE001
                logger.exception("failed to destroy actor %s", entry.object_id)
        self.actors.clear()
        self._actors_by_id.clear()
        # Synchronous-mode / fixed_delta_seconds are owned by the physics
        # container; it restores async mode on its own close_session.
        if self.traffic_manager is not None:
            try:
                self.traffic_manager.set_synchronous_mode(False)
            except Exception:  # noqa: BLE001
                logger.exception("failed to disable TM sync mode")
