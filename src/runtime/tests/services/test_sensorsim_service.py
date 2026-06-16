# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Unit tests for SensorsimService batch (batch_render_rgb) response mapping."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from alpasim_grpc.v0.sensorsim_pb2 import (
    BatchRGBRenderReturnItem,
    LidarDeviceType,
    LidarRenderReturn,
    RGBRenderReturn,
)
from alpasim_runtime.services.sensorsim_service import SensorsimService
from alpasim_runtime.types import Clock, RuntimeCamera


def _item(camera_name: str, image: bytes = b"img", success: bool = True, err: str = ""):
    return BatchRGBRenderReturnItem(
        camera_name=camera_name,
        result=RGBRenderReturn(image_bytes=image),
        success=success,
        error_message=err,
    )


def _triggers() -> dict[str, Clock.Trigger]:
    return {
        "cam_a": Clock.Trigger(range(0, 33_000), sequential_idx=0),
        "cam_b": Clock.Trigger(range(100_000, 133_000), sequential_idx=1),
    }


def test_batch_return_maps_images_by_camera_name():
    items = [_item("cam_a", b"a"), _item("cam_b", b"b")]

    images = SensorsimService._batch_return_to_images(items, _triggers())

    by_cam = {img.camera_logical_id: img for img in images}
    assert set(by_cam) == {"cam_a", "cam_b"}
    assert by_cam["cam_a"].image_bytes == b"a"
    # Timestamps come from the request-side trigger, not the response.
    assert by_cam["cam_a"].start_timestamp_us == 0
    assert by_cam["cam_a"].end_timestamp_us == 33_000
    assert by_cam["cam_b"].start_timestamp_us == 100_000


def test_batch_return_raises_on_failed_item():
    items = [
        _item("cam_a"),
        _item("cam_b", success=False, err="actor editing disabled"),
    ]

    with pytest.raises(RuntimeError, match=r"cam_b.*actor editing disabled"):
        SensorsimService._batch_return_to_images(items, _triggers())


def test_batch_return_raises_on_unknown_camera():
    items = [_item("cam_a"), _item("cam_UNEXPECTED")]

    with pytest.raises(RuntimeError, match=r"unknown camera 'cam_UNEXPECTED'"):
        SensorsimService._batch_return_to_images(items, _triggers())


def test_batch_return_raises_on_duplicate_camera():
    # cam_b would be missing the next loop, but the duplicate must fail first.
    items = [_item("cam_a"), _item("cam_a")]

    with pytest.raises(RuntimeError, match=r"duplicate camera 'cam_a'"):
        SensorsimService._batch_return_to_images(items, _triggers())


def test_batch_return_raises_on_missing_camera():
    # Requested cam_a and cam_b, but NRE only returned cam_a.
    items = [_item("cam_a")]

    with pytest.raises(RuntimeError, match=r"omitted requested camera.*cam_b"):
        SensorsimService._batch_return_to_images(items, _triggers())


@pytest.mark.asyncio
async def test_batch_render_skip_mode_returns_empty_images():
    """In skip mode batch_render returns placeholder frames like render()."""
    svc = SensorsimService("addr:0", skip=True, camera_catalog=MagicMock())
    cam = RuntimeCamera(
        logical_id="cam_a",
        render_resolution_hw=(2, 2),
        clock=Clock(interval_us=100_000, duration_us=33_000, start_us=0),
    )
    trigger = cam.clock.ith_trigger(0)

    images, driver_data = await svc.batch_render(
        [(cam, trigger)],
        ego_trajectory=MagicMock(),
        traffic_trajectories={},
        scene_id="scene",
        image_format=MagicMock(),
    )

    assert driver_data is None
    assert [img.camera_logical_id for img in images] == ["cam_a"]
    assert images[0].image_bytes == b""


def test_lidar_return_to_point_cloud_prefers_buffer_fields():
    ret = LidarRenderReturn(
        num_points=2,
        point_xyzs_buffer=np.array([1, 2, 3, 4, 5, 6], dtype=np.float32).tobytes(),
        point_intensities_buffer=np.array([0.5, 0.25], dtype=np.float32).tobytes(),
        point_ring_ids_buffer=np.array([3, 7], dtype=np.uint16).tobytes(),
    )
    trigger = Clock.Trigger(range(100, 200), sequential_idx=0)

    out = SensorsimService._lidar_return_to_point_cloud(ret, trigger, "lidar_top")

    assert out.lidar_logical_id == "lidar_top"
    assert out.num_points == 2
    assert out.start_timestamp_us == 100
    assert out.end_timestamp_us == 200
    assert np.frombuffer(out.point_ring_ids, dtype=np.uint16).tolist() == [3, 7]
    assert np.isclose(
        np.frombuffer(out.point_intensities, dtype=np.float32).tolist(), [0.5, 0.25]
    ).all()


def test_lidar_return_to_point_cloud_falls_back_to_repeated_fields():
    # Sender populated the repeated forms; ensure they round-trip into the
    # same little-endian buffer layout the buffer fields produce.
    ret = LidarRenderReturn(num_points=1)
    ret.point_xyzs.extend([1.0, 2.0, 3.0])
    ret.point_intensities.extend([0.75])
    ret.point_ring_ids.extend([5])
    trigger = Clock.Trigger(range(0, 10), sequential_idx=0)

    out = SensorsimService._lidar_return_to_point_cloud(ret, trigger, "lidar_top")

    assert np.frombuffer(out.point_xyzs, dtype=np.float32).tolist() == [1.0, 2.0, 3.0]
    assert np.frombuffer(out.point_ring_ids, dtype=np.uint16).tolist() == [5]


@pytest.mark.asyncio
async def test_render_lidar_skip_mode_returns_empty_point_cloud():
    """In skip mode render_lidar returns a zero-point placeholder cloud."""
    svc = SensorsimService("addr:0", skip=True, camera_catalog=MagicMock())
    trigger = Clock.Trigger(range(0, 10_000), sequential_idx=0)

    cloud = await svc.render_lidar(
        ego_trajectory=MagicMock(),
        traffic_trajectories={},
        lidar_logical_id="lidar_top",
        lidar_type=LidarDeviceType.PANDAR128,
        sensor_pose_delta=None,
        trigger=trigger,
        scene_id="scene",
    )

    assert cloud.lidar_logical_id == "lidar_top"
    assert cloud.num_points == 0
    assert cloud.point_xyzs == b""
    assert cloud.point_ring_ids == b""
