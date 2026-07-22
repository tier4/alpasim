# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""LiDAR render events for the event-based simulation loop."""

from __future__ import annotations

import logging
from typing import Any

from alpasim_runtime.broadcaster import MessageBroadcaster
from alpasim_runtime.events.base import Event, EventPriority, EventQueue
from alpasim_runtime.events.camera import _traffic_trajectories
from alpasim_runtime.events.state import RolloutState
from alpasim_runtime.services.driver_service import DriverService
from alpasim_runtime.services.sensorsim_service import SensorsimService
from alpasim_runtime.types import Clock, RuntimeLidar

logger = logging.getLogger(__name__)


class LidarFrameEvent(Event):
    """Render one LiDAR sweep and submit the point cloud to the driver.

    LiDAR is treated as instantaneous (``duration_us=0``): the trigger's
    time range collapses to a single simulated timestamp equal to the sweep
    time. Bundling is not implemented here — each LiDAR fires its own
    ``render_lidar`` RPC; when we adopt ``aggregated_render`` we will bundle
    LiDAR triggers alongside camera triggers on the flush event.
    """

    priority: int = EventPriority.LIDAR

    def __init__(
        self,
        lidar: RuntimeLidar,
        trigger: Clock.Trigger,
        sensorsim: SensorsimService,
        driver: DriverService,
    ):
        super().__init__(timestamp_us=trigger.time_range_us.stop)
        self.lidar = lidar
        self.trigger = trigger
        self.sensorsim = sensorsim
        self.driver = driver

    def description(self) -> str:
        return (
            f"LidarFrameEvent({self.lidar.logical_id}, "
            f"{self.trigger.time_range_us.stop:_}us)"
        )

    async def handle(self, rollout_state: RolloutState, queue: EventQueue) -> None:
        assert (
            rollout_state.step_context is not None
        ), "StepContext must exist before render"
        point_cloud = await self.sensorsim.render_lidar(
            ego_trajectory=rollout_state.ego_trajectory,
            traffic_trajectories=_traffic_trajectories(rollout_state),
            lidar_logical_id=self.lidar.logical_id,
            lidar_type=self.lidar.device_type,
            sensor_pose_delta=self.lidar.rig_to_lidar,
            trigger=self.trigger,
            scene_id=rollout_state.unbound.scene_id,
        )
        rollout_state.step_context.track_task(self.driver.submit_lidar(point_cloud))
        self._schedule_next(rollout_state, queue)

    def _schedule_next(self, state: RolloutState, queue: EventQueue) -> None:
        next_trigger = self.lidar.clock.ith_trigger(self.trigger.sequential_idx + 1)
        if next_trigger.time_range_us.stop > state.unbound.end_timestamp_us:
            return
        queue.submit(
            LidarFrameEvent(
                lidar=self.lidar,
                trigger=next_trigger,
                sensorsim=self.sensorsim,
                driver=self.driver,
            )
        )


def make_initial_lidar_render_events(
    *,
    scene_start_us: int,
    simulation_end_us: int,
    runtime_lidars: list[RuntimeLidar],
    renderer_service: Any,
    driver: DriverService,
    broadcaster: MessageBroadcaster,
) -> list[Event]:
    """Built-in factory for the first LiDAR render events of a rollout."""
    del broadcaster
    return [
        LidarFrameEvent(
            lidar=lidar,
            trigger=trigger,
            sensorsim=renderer_service,
            driver=driver,
        )
        for lidar in runtime_lidars
        for trigger in [lidar.clock.ith_trigger(0)]
        if scene_start_us <= trigger.time_range_us.stop <= simulation_end_us
    ]
