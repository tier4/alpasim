# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Shared data types used across alpasim packages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ImageWithMetadata:
    """Metadata for a camera image captured during simulation.

    This class is defined in alpasim_utils to avoid circular dependencies
    between runtime and eval packages.
    """

    start_timestamp_us: int
    end_timestamp_us: int
    image_bytes: bytes
    camera_logical_id: str

    def __repr__(self) -> str:
        return (
            "ImageWithMetadata("
            f"start_timestamp_us={self.start_timestamp_us:_d}, "
            f"end_timestamp_us={self.end_timestamp_us:_d}, "
            f"camera_logical_id={self.camera_logical_id}, "
            f"len(image_bytes)={len(self.image_bytes)})"
        )


@dataclass
class LidarPointCloudWithMetadata:
    """Metadata for a LiDAR point cloud captured during simulation.

    ``point_xyzs`` is a flat little-endian float32 ``[x1, y1, z1, x2, y2,
    z2, ...]`` buffer in the end-of-spin lidar frame; ``point_intensities``
    is one float32 per point in ``[0, 1]``; ``point_ring_ids`` is a packed
    little-endian uint16 buffer (one per point) identifying the laser
    channel. All three correspond field-for-field to ``LidarRenderReturn``
    in ``alpasim_grpc.v0.sensorsim_pb2``.
    """

    start_timestamp_us: int
    end_timestamp_us: int
    point_xyzs: bytes
    point_intensities: bytes
    point_ring_ids: bytes
    num_points: int
    lidar_logical_id: str

    def __repr__(self) -> str:
        return (
            "LidarPointCloudWithMetadata("
            f"start_timestamp_us={self.start_timestamp_us:_d}, "
            f"end_timestamp_us={self.end_timestamp_us:_d}, "
            f"lidar_logical_id={self.lidar_logical_id}, "
            f"num_points={self.num_points})"
        )
