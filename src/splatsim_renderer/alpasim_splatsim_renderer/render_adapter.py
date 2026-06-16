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
    callers can map it to ``grpc.StatusCode.UNIMPLEMENTED``.
    """
    width = int(spec.resolution_w)
    height = int(spec.resolution_h)
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
    raise NotImplementedError(
        f"splatsim renderer only supports opencv_pinhole_param; got {branch!r}"
    )


def _quat_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert a wxyz quaternion to a 3x3 rotation matrix."""
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0:
        return np.eye(3, dtype=np.float32)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def pose_to_viewmat(pose) -> np.ndarray:
    """Convert a common.Pose (camera-in-world) to a 4x4 world->camera viewmat.

    Alpasim's `common.Pose` convention is "translation then rotation",
    interpreted as the camera's pose in the world frame (camera-to-world).
    splatsim's `Renderer.render` wants the inverse — world-to-camera. The
    inverse of a rigid transform ``[R | t]`` is ``[R^T | -R^T t]``.
    """
    R = _quat_to_rotation_matrix(pose.quat.w, pose.quat.x, pose.quat.y, pose.quat.z)
    t = np.array([pose.vec.x, pose.vec.y, pose.vec.z], dtype=np.float32)
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = R.T
    viewmat[:3, 3] = -R.T @ t
    return viewmat


def pose_pair_to_viewmat(pose_pair: sensorsim_pb2.PosePair) -> np.ndarray:
    """Use the start_pose; rolling-shutter (end_pose) is not modelled yet."""
    return pose_to_viewmat(pose_pair.start_pose)


def encode_image(
    rgb_float: np.ndarray,
    image_format: int,
    image_quality: float,
) -> bytes:
    """Encode a float32 (H, W, 3) RGB image (0-1) to the requested bytes.

    `image_quality` is the proto's JPEG quality knob in 0.0-1.0 range; we
    re-scale to PIL's 1-95 range.
    """
    if rgb_float.dtype != np.uint8:
        rgb_uint8 = np.clip(rgb_float * 255.0, 0, 255).astype(np.uint8)
    else:
        rgb_uint8 = rgb_float

    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB array, got shape {rgb_uint8.shape}")

    fmt = sensorsim_pb2.ImageFormat.Name(image_format) if image_format else "PNG"
    buf = io.BytesIO()
    img = Image.fromarray(rgb_uint8, mode="RGB")
    if fmt == "JPEG":
        quality = max(1, min(95, int(round(image_quality * 95)))) if image_quality > 0 else 85
        img.save(buf, format="JPEG", quality=quality)
    elif fmt == "RGB_UINT8_PLANAR":
        # Planar layout = (3, H, W) bytes. Used by some downstream consumers
        # that prefer un-encoded buffers.
        return np.ascontiguousarray(rgb_uint8.transpose(2, 0, 1)).tobytes()
    else:
        # Default + UNDEFINED + PNG path.
        img.save(buf, format="PNG")
    return buf.getvalue()
