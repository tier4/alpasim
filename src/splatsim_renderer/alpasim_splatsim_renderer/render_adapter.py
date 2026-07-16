# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Pure-numpy conversions between alpasim sensorsim proto and splatsim arrays.

Kept torch-free so unit tests run on the standard alpasim_grpc test deps.
Callers pass the resulting numpy arrays into ``torch.from_numpy`` at the
boundary inside the gRPC handler.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np
from alpasim_grpc.v0 import sensorsim_pb2
from PIL import Image


@dataclass(frozen=True)
class CameraIntrinsics:
    """Resolved intrinsics ready to be uploaded to splatsim."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def k_matrix(self) -> np.ndarray:
        """3x3 pinhole K matrix in row-major float32."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )


def camera_spec_to_intrinsics(spec: sensorsim_pb2.CameraSpec) -> CameraIntrinsics:
    """Extract pinhole intrinsics from a CameraSpec.

    Only OpenCVPinholeCameraParam is supported for now; ftheta and fisheye
    models need a CARLA-style distortion pipeline that splatsim doesn't
    expose. Raises NotImplementedError for unsupported one-of branches so
    callers can map it to ``grpc.StatusCode.UNIMPLEMENTED``. Raises
    ValueError for malformed inputs (zero / negative resolution).
    """
    width = int(spec.resolution_w)
    height = int(spec.resolution_h)
    if width <= 0 or height <= 0:
        raise ValueError(
            f"CameraSpec resolution must be positive; got {width}x{height}"
        )
    branch = spec.WhichOneof("camera_param")
    if branch == "opencv_pinhole_param":
        p = spec.opencv_pinhole_param
        return CameraIntrinsics(
            fx=float(p.focal_length_x),
            fy=float(p.focal_length_y),
            cx=float(p.principal_point_x),
            cy=float(p.principal_point_y),
            width=width,
            height=height,
        )
    if branch is None:
        raise NotImplementedError(
            "CameraSpec.camera_param is unset; splatsim renderer needs "
            "opencv_pinhole_param to be populated"
        )
    raise NotImplementedError(
        f"splatsim renderer only supports opencv_pinhole_param; got {branch!r}"
    )


def _quat_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert a wxyz quaternion to a 3x3 rotation matrix."""
    # Identity rotation is the common case for stationary cameras; skip the
    # sqrt + 4 divides + 9-term matrix construction.
    if qw == 1.0 and qx == 0.0 and qy == 0.0 and qz == 0.0:
        return np.eye(3, dtype=np.float32)
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0:
        return np.eye(3, dtype=np.float32)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.array(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float32,
    )


def pose_to_viewmat(pose, world_origin: np.ndarray | None = None) -> np.ndarray:
    """Convert a common.Pose (camera-in-world) to a 4x4 world->camera viewmat.

    Alpasim's `common.Pose` convention is "translation then rotation",
    interpreted as the camera's pose in the world frame (camera-to-world).
    splatsim's `Renderer.render` wants the inverse — world-to-camera. The
    inverse of a rigid transform ``[R | t]`` is ``[R^T | -R^T t]``.

    ``world_origin`` is subtracted from the position before inversion so
    that world-frame poses land in splatsim's tile-local frame (the frame
    ``Renderer.render`` actually operates in). Pass ``Background.tile_local_centroid``
    from the loaded scene here.
    """
    R = _quat_to_rotation_matrix(pose.quat.w, pose.quat.x, pose.quat.y, pose.quat.z)
    t = np.array([pose.vec.x, pose.vec.y, pose.vec.z], dtype=np.float32)
    if world_origin is not None:
        t = t - np.asarray(world_origin, dtype=np.float32)
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = R.T
    viewmat[:3, 3] = -R.T @ t
    return viewmat


def pose_pair_to_viewmat(
    pose_pair: sensorsim_pb2.PosePair,
    world_origin: np.ndarray | None = None,
) -> np.ndarray:
    """Use the start_pose; rolling-shutter (end_pose) is not modelled yet."""
    return pose_to_viewmat(pose_pair.start_pose, world_origin=world_origin)


def pose_to_sensor_to_world(
    pose,
    world_origin: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a common.Pose (camera-in-world) to a 4x4 sensor-to-tile-local matrix.

    Used by the LiDAR path (splatsim's ``LidarRenderer.render`` takes
    ``sensor_to_world`` directly, not its inverse). Applies the same
    ``world_origin`` offset as :func:`pose_to_viewmat`.
    """
    R = _quat_to_rotation_matrix(pose.quat.w, pose.quat.x, pose.quat.y, pose.quat.z)
    t = np.array([pose.vec.x, pose.vec.y, pose.vec.z], dtype=np.float32)
    if world_origin is not None:
        t = t - np.asarray(world_origin, dtype=np.float32)
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = R
    m[:3, 3] = t
    return m


def encode_image(
    rgb: np.ndarray,
    image_format: int,
    image_quality: float,
) -> bytes:
    """Encode a (H, W, 3) RGB image to the requested wire format.

    Accepted dtypes:
      - float32 / float64 in [0, 1] (values outside the range are clipped)
      - uint8 in [0, 255] (used directly)

    `image_quality` is the proto's JPEG quality knob in 0.0-1.0 range; values
    <= 0 fall back to PIL's default of 85.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB array, got shape {rgb.shape}")

    if rgb.dtype == np.uint8:
        rgb_uint8 = rgb
    elif np.issubdtype(rgb.dtype, np.floating):
        rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    else:
        raise TypeError(f"unsupported RGB dtype {rgb.dtype}; expected float or uint8")

    # Direct enum comparison — proto3 defaults to UNDEFINED (0); we treat that
    # as "default to PNG" since UNDEFINED has no on-wire semantics.
    if image_format == sensorsim_pb2.RGB_UINT8_PLANAR:
        # Planar layout = (3, H, W) bytes. Skip the PIL Image allocation.
        return np.ascontiguousarray(rgb_uint8.transpose(2, 0, 1)).tobytes()

    img = Image.fromarray(rgb_uint8, mode="RGB")
    buf = io.BytesIO()
    if image_format == sensorsim_pb2.JPEG:
        if image_quality > 0:
            quality = max(1, min(95, int(round(image_quality * 95))))
        else:
            quality = 85
        img.save(buf, format="JPEG", quality=quality)
    else:
        # PNG path covers PNG, UNDEFINED, and any not-yet-supported format.
        img.save(buf, format="PNG")
    return buf.getvalue()
