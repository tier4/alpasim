# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Session-specific LiDAR point cloud buffer for driver service."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
from threading import RLock
from typing import Callable, List, TypeVar

import numpy as np

F = TypeVar("F", bound=Callable)


@dataclass
class PointCloudEntry:
    """Represents a single LiDAR point cloud observation."""

    timestamp_us: int
    points_xyz: np.ndarray  # (N, 3) float32, in end-of-spin lidar frame
    intensities: np.ndarray  # (N,) float32 in [0, 1]
    ring_ids: np.ndarray  # (N,) uint16


def synchronized(method: F) -> F:
    @wraps(method)
    def wrapper(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        with self._lock:  # noqa: SLF001
            return method(self, *args, **kwargs)

    return wrapper  # type: ignore[return-value]


@dataclass
class PointCloudCache:
    """Bounded, time-ordered buffer of LiDAR point clouds for a session.

    Mirrors ``FrameCache``: context_length is the number of clouds requested by
    the model per inference, and subsample_factor allows keeping every Nth
    sample when the sensor produces at a higher rate than the model expects.
    """

    context_length: int
    lidar_id: str = ""
    subsample_factor: int = 1
    entries: List[PointCloudEntry] = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    @synchronized
    def add_point_cloud(
        self,
        timestamp_us: int,
        points_xyz: np.ndarray,
        intensities: np.ndarray,
        ring_ids: np.ndarray,
    ) -> None:
        """Insert a point cloud while keeping entries ordered by timestamp."""
        inserted = False
        for offset, entry in enumerate(reversed(self.entries)):
            if entry.timestamp_us == timestamp_us:
                raise ValueError(f"Point cloud {timestamp_us} already exists in cache")
            if entry.timestamp_us < timestamp_us:
                insert_at = len(self.entries) - offset
                self.entries.insert(
                    insert_at,
                    PointCloudEntry(timestamp_us, points_xyz, intensities, ring_ids),
                )
                inserted = True
                break
        if not inserted:
            self.entries.insert(
                0,
                PointCloudEntry(timestamp_us, points_xyz, intensities, ring_ids),
            )

        self._prune()

    @synchronized
    def frame_count(self) -> int:
        """Total number of point clouds currently cached."""
        return len(self.entries)

    def min_frames_required(self) -> int:
        """Minimum number of clouds needed for inference."""
        return (self.context_length - 1) * self.subsample_factor + 1

    @synchronized
    def has_enough_frames(self) -> bool:
        """Check if there are enough point clouds for inference."""
        return len(self.entries) >= self.min_frames_required()

    @synchronized
    def latest_frame_entries(self, count: int) -> List[PointCloudEntry]:
        """Return the newest ``count`` point clouds with subsampling (oldest first)."""
        min_required = (count - 1) * self.subsample_factor + 1
        if len(self.entries) < min_required:
            raise ValueError(
                f"Insufficient point clouds: have {len(self.entries)}, need at least "
                f"{min_required} (count={count}, subsample_factor={self.subsample_factor})"
            )

        selected_indices = []
        idx = len(self.entries) - 1
        for _ in range(count):
            selected_indices.append(idx)
            idx -= self.subsample_factor

        selected_indices = selected_indices[::-1]
        return [self.entries[i] for i in selected_indices]

    def _prune(self) -> None:
        """Bound the cache to accommodate subsampled context queries."""
        max_entries = self.context_length * self.subsample_factor
        excess = len(self.entries) - max_entries
        if excess <= 0:
            return
        self.entries = self.entries[excess:]
