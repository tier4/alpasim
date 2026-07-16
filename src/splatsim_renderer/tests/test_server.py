# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""gRPC server-level tests with a stub SceneHandle.

The real SceneHandle pulls in torch + splatsim + CUDA; for unit tests we
inject a stub that returns a constant RGB array.
"""

from __future__ import annotations

import io
from unittest import mock

import numpy as np
import pytest
from alpasim_grpc.v0 import common_pb2, sensorsim_pb2
from PIL import Image


class _StubScene:
    """SceneHandle stand-in that returns a fixed gradient image."""

    def __init__(self, h=8, w=12, centroid=(0.0, 0.0, 0.0)):
        self._h = h
        self._w = w
        self.render_calls = []
        self._centroid = np.asarray(centroid, dtype=np.float32)
        self.device = "cpu"
        self.scene = mock.Mock(name="stub_scene")

    def render(self, viewmat, K):
        self.render_calls.append((viewmat.copy(), K.copy()))
        x = np.linspace(0.0, 1.0, self._w, dtype=np.float32)
        return np.broadcast_to(x[None, :, None], (self._h, self._w, 3)).astype(
            np.float32
        )

    @property
    def tile_local_centroid(self):
        return self._centroid


@pytest.fixture
def servicer_with_stub():
    from alpasim_splatsim_renderer.server import SplatsimSensorsimServicer

    scene = _StubScene()
    return SplatsimSensorsimServicer(scene=scene, scene_id="test-scene"), scene


def _pinhole_spec():
    spec = sensorsim_pb2.CameraSpec(resolution_w=12, resolution_h=8)
    p = spec.opencv_pinhole_param
    p.focal_length_x = 100.0
    p.focal_length_y = 100.0
    p.principal_point_x = 6.0
    p.principal_point_y = 4.0
    return spec


def _identity_pose_pair():
    pose = common_pb2.Pose(
        vec=common_pb2.Vec3(x=0.0, y=0.0, z=0.0),
        quat=common_pb2.Quat(w=1.0),
    )
    return sensorsim_pb2.PosePair(start_pose=pose, end_pose=pose)


def test_render_rgb_returns_png(servicer_with_stub):
    servicer, scene = servicer_with_stub
    req = sensorsim_pb2.RGBRenderRequest(
        scene_id="test-scene",
        camera_intrinsics=_pinhole_spec(),
        sensor_pose=_identity_pose_pair(),
        image_format=sensorsim_pb2.PNG,
    )
    resp = servicer.render_rgb(req, context=mock.Mock())
    assert resp.image_bytes
    img = Image.open(io.BytesIO(resp.image_bytes))
    assert img.size == (12, 8)
    # Exactly one render call was issued.
    assert len(scene.render_calls) == 1
    viewmat, K = scene.render_calls[0]
    assert viewmat.shape == (4, 4)
    assert K.shape == (3, 3)


def test_render_rgb_unsupported_camera_aborts(servicer_with_stub):
    servicer, _ = servicer_with_stub
    spec = sensorsim_pb2.CameraSpec(resolution_w=12, resolution_h=8)
    spec.ftheta_param.principal_point_x = 1.0
    req = sensorsim_pb2.RGBRenderRequest(
        scene_id="test-scene",
        camera_intrinsics=spec,
        sensor_pose=_identity_pose_pair(),
    )
    ctx = mock.Mock()
    ctx.abort.side_effect = RuntimeError("abort")
    with pytest.raises(RuntimeError):
        servicer.render_rgb(req, context=ctx)
    ctx.abort.assert_called_once()


def _make_lidar_request(device_type=sensorsim_pb2.LidarDeviceType.PANDAR128):
    req = sensorsim_pb2.LidarRenderRequest(scene_id="test-scene")
    req.lidar_config.lidar_type = device_type
    req.sensor_pose.CopyFrom(_identity_pose_pair())
    return req


def test_render_lidar_pandar128_returns_points(servicer_with_stub, monkeypatch):
    servicer, _ = servicer_with_stub

    def fake_render(scene_handle, base_to_world_np, device_type):
        xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        intensity = np.array([0.5, 0.8], dtype=np.float32)
        ring_ids = np.array([0, 7], dtype=np.uint16)
        return xyz, intensity, ring_ids

    monkeypatch.setattr(
        "alpasim_splatsim_renderer.server.render_lidar_panorama_from_scene",
        fake_render,
    )
    resp = servicer.render_lidar(_make_lidar_request(), context=mock.Mock())
    assert resp.num_points == 2
    assert len(resp.point_xyzs_buffer) == 2 * 3 * 4  # 2 pts * 3 floats * float32
    assert len(resp.point_intensities_buffer) == 2 * 4
    assert len(resp.point_ring_ids_buffer) == 2 * 2  # 2 pts * uint16


def test_render_lidar_at128_aborts_unimplemented(servicer_with_stub):
    servicer, _ = servicer_with_stub
    req = _make_lidar_request(device_type=sensorsim_pb2.LidarDeviceType.AT128)
    ctx = mock.Mock()
    ctx.abort.side_effect = RuntimeError("abort")
    with pytest.raises(RuntimeError):
        servicer.render_lidar(req, context=ctx)
    ctx.abort.assert_called_once()


def test_render_aggregated_isolates_lidar_failure(servicer_with_stub, monkeypatch):
    servicer, _ = servicer_with_stub

    # First lidar sub-request succeeds, second raises (mocking a splatsim error).
    calls = {"n": 0}

    def flaky_render(scene_handle, base_to_world_np, device_type):
        calls["n"] += 1
        if calls["n"] == 1:
            xyz = np.zeros((1, 3), dtype=np.float32)
            intensity = np.zeros((1,), dtype=np.float32)
            ring_ids = np.zeros((1,), dtype=np.uint16)
            return xyz, intensity, ring_ids
        raise RuntimeError("simulated splatsim failure")

    monkeypatch.setattr(
        "alpasim_splatsim_renderer.server.render_lidar_panorama_from_scene",
        flaky_render,
    )

    agg = sensorsim_pb2.AggregatedRenderRequest()
    agg.rgb_requests.add().CopyFrom(
        sensorsim_pb2.RGBRenderRequest(
            scene_id="test-scene",
            camera_intrinsics=_pinhole_spec(),
            sensor_pose=_identity_pose_pair(),
            image_format=sensorsim_pb2.PNG,
        )
    )
    agg.lidar_requests.add().CopyFrom(_make_lidar_request())
    agg.lidar_requests.add().CopyFrom(_make_lidar_request())

    resp = servicer.render_aggregated(agg, context=mock.Mock())
    assert len(resp.rgb_returns) == 1
    assert resp.rgb_returns[0].image_bytes  # RGB succeeded
    assert len(resp.lidar_returns) == 2
    assert resp.lidar_returns[0].num_points == 1  # first lidar succeeded
    assert resp.lidar_returns[1].num_points == 0  # second failed → empty


def test_batch_render_rgb_returns_per_camera_success(servicer_with_stub):
    servicer, _ = servicer_with_stub
    batch = sensorsim_pb2.BatchRGBRenderRequest()
    for name in ("front", "rear"):
        item = batch.items.add()
        item.camera_name = name
        item.request.CopyFrom(
            sensorsim_pb2.RGBRenderRequest(
                scene_id="test-scene",
                camera_intrinsics=_pinhole_spec(),
                sensor_pose=_identity_pose_pair(),
                image_format=sensorsim_pb2.PNG,
            )
        )
    resp = servicer.batch_render_rgb(batch, context=mock.Mock())
    assert [item.camera_name for item in resp.items] == ["front", "rear"]
    assert all(item.success for item in resp.items)
    assert all(item.result.image_bytes for item in resp.items)


def test_get_available_scenes_reports_loaded_scene(servicer_with_stub):
    servicer, _ = servicer_with_stub
    resp = servicer.get_available_scenes(common_pb2.Empty(), context=mock.Mock())
    assert list(resp.scene_ids) == ["test-scene"]


def test_get_version_includes_renderer_version(servicer_with_stub):
    from alpasim_splatsim_renderer import __version__ as expected

    servicer, _ = servicer_with_stub
    resp = servicer.get_version(common_pb2.Empty(), context=mock.Mock())
    assert resp.version_id == expected
