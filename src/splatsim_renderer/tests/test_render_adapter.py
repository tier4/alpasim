# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Unit tests for render_adapter conversions.

These run without torch / splatsim / CUDA — pure numpy + PIL.
"""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
from alpasim_grpc.v0 import common_pb2, sensorsim_pb2
from alpasim_splatsim_renderer.render_adapter import (
    camera_spec_to_intrinsics,
    encode_image,
    pose_pair_to_viewmat,
    pose_to_viewmat,
)
from PIL import Image


def _opencv_pinhole_spec(fx=500.0, fy=510.0, cx=480.0, cy=270.0, w=960, h=540):
    spec = sensorsim_pb2.CameraSpec(resolution_w=w, resolution_h=h)
    p = spec.opencv_pinhole_param
    p.focal_length_x = fx
    p.focal_length_y = fy
    p.principal_point_x = cx
    p.principal_point_y = cy
    return spec


def test_camera_spec_to_intrinsics_pinhole():
    spec = _opencv_pinhole_spec()
    intr = camera_spec_to_intrinsics(spec)
    assert (intr.fx, intr.fy, intr.cx, intr.cy) == (500.0, 510.0, 480.0, 270.0)
    assert (intr.width, intr.height) == (960, 540)

    K = intr.k_matrix()
    assert K.shape == (3, 3)
    assert K.dtype == np.float32
    assert K[0, 0] == 500.0
    assert K[1, 1] == 510.0
    assert K[0, 2] == 480.0
    assert K[1, 2] == 270.0
    assert K[2, 2] == 1.0


def test_camera_spec_to_intrinsics_unsupported_branch():
    spec = sensorsim_pb2.CameraSpec(resolution_w=960, resolution_h=540)
    # ftheta is intentionally unsupported in the initial integration.
    spec.ftheta_param.principal_point_x = 1.0
    with pytest.raises(NotImplementedError):
        camera_spec_to_intrinsics(spec)


def test_camera_spec_to_intrinsics_unset_oneof():
    spec = sensorsim_pb2.CameraSpec(resolution_w=960, resolution_h=540)
    with pytest.raises(NotImplementedError, match="camera_param is unset"):
        camera_spec_to_intrinsics(spec)


def test_camera_spec_to_intrinsics_rejects_zero_resolution():
    spec = sensorsim_pb2.CameraSpec(resolution_w=0, resolution_h=540)
    spec.opencv_pinhole_param.focal_length_x = 500.0
    with pytest.raises(ValueError, match="resolution must be positive"):
        camera_spec_to_intrinsics(spec)


def test_pose_identity_yields_translation_only_viewmat():
    pose = common_pb2.Pose(
        vec=common_pb2.Vec3(x=1.0, y=2.0, z=3.0),
        quat=common_pb2.Quat(w=1.0, x=0.0, y=0.0, z=0.0),
    )
    viewmat = pose_to_viewmat(pose)
    # Alpasim sends camera-in-world as ENU Z-up; splatsim's tile-local is
    # Y-up, so ``pose_to_viewmat`` applies R_ENU_TO_TILE = [[1,0,0],[0,0,1],
    # [0,-1,0]] before inversion. For identity rotation this yields
    # viewmat[:3,:3] = R_ENU_TO_TILE.T and viewmat[:3,3] = -R.T @ R @ t = -t.
    expected_R = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    np.testing.assert_allclose(viewmat[:3, :3], expected_R, atol=1e-6)
    np.testing.assert_allclose(viewmat[:3, 3], np.array([-1.0, -2.0, -3.0]), atol=1e-6)
    np.testing.assert_allclose(viewmat[3], [0, 0, 0, 1], atol=1e-6)


def test_pose_90deg_yaw_yields_orthonormal_inverse():
    # 90deg yaw around z-axis: quat = (cos(45), 0, 0, sin(45))
    half = math.pi / 4
    pose = common_pb2.Pose(
        vec=common_pb2.Vec3(x=0.0, y=0.0, z=0.0),
        quat=common_pb2.Quat(w=math.cos(half), x=0.0, y=0.0, z=math.sin(half)),
    )
    viewmat = pose_to_viewmat(pose)
    R = viewmat[:3, :3]
    # Inverse of a rotation matrix is its transpose; check orthonormality.
    np.testing.assert_allclose(R @ R.T, np.eye(3, dtype=np.float32), atol=1e-6)


def test_pose_pair_uses_start_pose():
    start = common_pb2.Pose(vec=common_pb2.Vec3(x=5.0), quat=common_pb2.Quat(w=1.0))
    end = common_pb2.Pose(vec=common_pb2.Vec3(x=99.0), quat=common_pb2.Quat(w=1.0))
    viewmat = pose_pair_to_viewmat(
        sensorsim_pb2.PosePair(start_pose=start, end_pose=end)
    )
    # Translation comes from start_pose: cam at x=5 -> world->cam tx=-5.
    assert viewmat[0, 3] == pytest.approx(-5.0)


def test_encode_image_png_roundtrip():
    rgb = np.zeros((4, 6, 3), dtype=np.float32)
    rgb[0, 0] = [1.0, 0.0, 0.0]
    rgb[1, 1] = [0.0, 1.0, 0.0]
    data = encode_image(rgb, sensorsim_pb2.PNG, 0.0)
    img = Image.open(io.BytesIO(data))
    arr = np.array(img)
    assert arr.shape == (4, 6, 3)
    assert tuple(arr[0, 0]) == (255, 0, 0)
    assert tuple(arr[1, 1]) == (0, 255, 0)


def test_encode_image_jpeg_uses_quality():
    rgb = np.full((8, 8, 3), 0.5, dtype=np.float32)
    data_low = encode_image(rgb, sensorsim_pb2.JPEG, 0.1)
    data_high = encode_image(rgb, sensorsim_pb2.JPEG, 0.95)
    # Higher quality JPEG should produce a larger byte string for the same input.
    assert len(data_low) <= len(data_high)


def test_encode_image_planar_uint8():
    rgb = np.zeros((2, 3, 3), dtype=np.float32)
    rgb[0, 0, 0] = 1.0
    data = encode_image(rgb, sensorsim_pb2.RGB_UINT8_PLANAR, 0.0)
    # Raw planar: 3 * H * W bytes.
    assert len(data) == 3 * 2 * 3


def test_encode_image_invalid_shape():
    rgb = np.zeros((4, 6), dtype=np.float32)
    with pytest.raises(ValueError):
        encode_image(rgb, sensorsim_pb2.PNG, 0.0)


def test_encode_image_uint8_input_passes_through():
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[0, 0] = [200, 100, 50]
    data = encode_image(rgb, sensorsim_pb2.PNG, 0.0)
    img = Image.open(io.BytesIO(data))
    arr = np.array(img)
    assert tuple(arr[0, 0]) == (200, 100, 50)


def test_encode_image_rejects_unsupported_dtype():
    rgb = np.zeros((2, 3, 3), dtype=np.int32)
    with pytest.raises(TypeError, match="unsupported RGB dtype"):
        encode_image(rgb, sensorsim_pb2.PNG, 0.0)


def test_encode_image_undefined_format_defaults_to_png():
    rgb = np.zeros((2, 3, 3), dtype=np.float32)
    data = encode_image(rgb, sensorsim_pb2.UNDEFINED, 0.0)
    # Should be decodable as PNG.
    img = Image.open(io.BytesIO(data))
    assert img.format == "PNG"


def test_identity_quat_takes_fast_path():
    """Identity quaternion should hit the fast path and yield exact R_ENU_TO_TILE^T.

    The frame rotation is a permutation with exact ±1 entries, so combining it
    with the identity-quat fast path in ``_quat_to_rotation_matrix`` must
    remain exact (no floating-point drift).
    """
    pose = common_pb2.Pose(
        vec=common_pb2.Vec3(),
        quat=common_pb2.Quat(w=1.0),
    )
    viewmat = pose_to_viewmat(pose)
    expected_R = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    np.testing.assert_array_equal(viewmat[:3, :3], expected_R)


def test_world_origin_offset_shifts_camera_position():
    """``world_origin`` (in tile-local Y-up) zeroes the recentered translation.

    Camera at world ENU (10, 20, 30) rotates to tile-local Y-up (10, 30, -20);
    passing that as ``world_origin`` recenters to the tile origin.
    """
    from alpasim_splatsim_renderer.render_adapter import pose_to_viewmat

    pose = common_pb2.Pose(
        vec=common_pb2.Vec3(x=10.0, y=20.0, z=30.0),
        quat=common_pb2.Quat(w=1.0),
    )
    viewmat = pose_to_viewmat(
        pose, world_origin=np.array([10.0, 30.0, -20.0], dtype=np.float32)
    )
    np.testing.assert_allclose(viewmat[:3, 3], np.zeros(3), atol=1e-6)


def test_zup_to_yup_axis_mapping():
    """+Z (up) in alpasim ENU must become +Y (up) in splatsim tile-local.

    Regression guard for the frame conversion. Verified at the sensor-to-world
    level (no inversion) so the assertion reads as a direct position swap.
    """
    from alpasim_splatsim_renderer.render_adapter import pose_to_sensor_to_world

    pose = common_pb2.Pose(
        vec=common_pb2.Vec3(x=0.0, y=0.0, z=5.0),
        quat=common_pb2.Quat(w=1.0),
    )
    s2w = pose_to_sensor_to_world(pose)
    np.testing.assert_allclose(s2w[:3, 3], np.array([0.0, 5.0, 0.0]), atol=1e-6)


def test_pose_to_sensor_to_world_is_inverse_of_viewmat():
    """``pose_to_sensor_to_world`` returns the un-inverted sensor→world 4x4."""
    from alpasim_splatsim_renderer.render_adapter import (
        pose_to_sensor_to_world,
        pose_to_viewmat,
    )

    half = math.pi / 4
    pose = common_pb2.Pose(
        vec=common_pb2.Vec3(x=1.0, y=2.0, z=3.0),
        quat=common_pb2.Quat(w=math.cos(half), x=0.0, y=0.0, z=math.sin(half)),
    )
    s2w = pose_to_sensor_to_world(pose)
    w2s = pose_to_viewmat(pose)
    # s2w and w2s should be inverse of each other.
    np.testing.assert_allclose(s2w @ w2s, np.eye(4, dtype=np.float32), atol=1e-6)
