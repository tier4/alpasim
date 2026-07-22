# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from alpasim_grpc.v0.sensorsim_pb2 import LidarDeviceType
from alpasim_runtime.config import RuntimeCameraConfig, RuntimeLidarConfig

from utils_rs import Pose


@dataclass
class Clock:
    """
    Represents a clock which ticks triggers with a given interval (`interval_us`)
    and a given duration (`duration_us`).
    """

    @dataclass
    class Trigger:
        # start and end of sensor acquisition
        time_range_us: range

        # unique and consecutive within camera_id,
        # equivalent to sorting all CameraTriggers by time_range_us.start
        sequential_idx: int

    interval_us: int
    duration_us: int
    start_us: int = 0
    first_end_us: int | None = None

    def __post_init__(self) -> None:
        if self.interval_us <= 0:
            raise ValueError("interval_us must be positive")
        if self.duration_us < 0:
            raise ValueError("duration_us must be non-negative")

    def ith_trigger(self, i: int) -> Trigger:
        """Returns the i-th trigger of the clock since self.start_us"""
        if i < 0:
            raise ValueError(f"Trigger index must be non-negative, got {i}")
        if i == 0 and self.first_end_us is not None:
            return Clock.Trigger(
                range(self.start_us, self.first_end_us),
                sequential_idx=i,
            )
        first_end_us = (
            self.first_end_us
            if self.first_end_us is not None
            else self.start_us + self.duration_us
        )
        end_us = first_end_us + i * self.interval_us
        return Clock.Trigger(
            range(
                end_us - self.duration_us,
                end_us,
            ),
            sequential_idx=i,
        )


@dataclass
class RuntimeCamera:
    """This class defines which cameras are rendered and how to render them.

    - `logical_id` is the unique identifier for the camera. This references a
        `CameraDefinition` in the camera catalog.
    - `render_resolution_hw` is the resolution of the camera in pixels.
    - `clock` is the clock that determines the timing of the camera.
    """

    logical_id: str
    render_resolution_hw: tuple[int, int]
    clock: Clock

    @classmethod
    def from_camera_config(
        cls, camera_cfg: RuntimeCameraConfig, first_frame_range_us: range
    ) -> RuntimeCamera:
        """Build a `RuntimeCamera` from a scenario `CameraConfig`."""

        first_frame_duration_us = first_frame_range_us.stop - first_frame_range_us.start
        duration_us = camera_cfg.shutter_duration_us or first_frame_duration_us
        clock = Clock(
            interval_us=camera_cfg.frame_interval_us,
            duration_us=duration_us,
            start_us=first_frame_range_us.start,
            first_end_us=first_frame_range_us.stop,
        )
        return cls(
            logical_id=camera_cfg.logical_id,
            render_resolution_hw=(camera_cfg.height, camera_cfg.width),
            clock=clock,
        )


@dataclass
class RuntimeLidar:
    """This class defines which LiDAR sensors are rendered and how to render them.

    - ``logical_id`` uniquely identifies the sensor; the driver receives point
      clouds keyed by this id.
    - ``device_type`` picks the physical model (e.g. ``PANDAR128``) that the
      renderer applies.
    - ``clock`` determines the timing of point-cloud emission. LiDAR is treated
      as instantaneous (``duration_us=0``) — the trigger's ``time_range_us``
      collapses to a single timestamp.
    """

    logical_id: str
    device_type: LidarDeviceType
    clock: Clock
    rig_to_lidar: Pose = field(
        default_factory=lambda: Pose(
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )
    )

    @classmethod
    def from_lidar_config(
        cls,
        lidar_cfg: RuntimeLidarConfig,
        first_frame_end_us: int,
        t_sensor_rig: np.ndarray | None = None,
    ) -> RuntimeLidar:
        """Build a ``RuntimeLidar`` from a scenario ``RuntimeLidarConfig``.

        ``first_frame_end_us`` anchors the first LiDAR sweep so the initial
        cloud lines up with the first camera frame's shutter-close (this keeps
        camera + LiDAR arriving at the driver in the same simulated
        millisecond).
        """
        try:
            device_type = LidarDeviceType.Value(lidar_cfg.device_type)
        except ValueError as exc:
            raise ValueError(
                f"Unknown LidarDeviceType {lidar_cfg.device_type!r} for lidar "
                f"{lidar_cfg.logical_id!r}"
            ) from exc
        clock = Clock(
            interval_us=lidar_cfg.frame_interval_us,
            duration_us=0,
            start_us=first_frame_end_us,
            first_end_us=first_frame_end_us,
        )
        kwargs: dict = {
            "logical_id": lidar_cfg.logical_id,
            "device_type": device_type,
            "clock": clock,
        }
        if t_sensor_rig is not None:
            # USDZ stores rig→sensor (Autoware convention for LiDAR). Invert to
            # get sensor-in-rig (the pose the renderer expects).
            R = t_sensor_rig[:3, :3]
            t = t_sensor_rig[:3, 3]
            T_rig_to_lidar = np.eye(4, dtype=np.float32)
            T_rig_to_lidar[:3, :3] = R.T
            T_rig_to_lidar[:3, 3] = -R.T @ t
            kwargs["rig_to_lidar"] = Pose.from_se3(T_rig_to_lidar)
        return cls(**kwargs)
