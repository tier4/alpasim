# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Conversion helpers between Alpasim gRPC proto messages and CARLA actor state.

Kept dependency-light on purpose: trafficsim runs on Python 3.10 (constrained
by autoware_carla_scenario) while alpasim_utils currently requires the
utils_rs extension built only for 3.11+. We re-implement just what we need.

CARLA conventions:
  - carla.Transform.location is (x, y, z) in meters (left-handed, y-up).
  - carla.Transform.rotation is in degrees (pitch, yaw, roll).

Alpasim conventions (see CONTRIBUTING.md "Coordinate Systems"):
  - common.Pose holds (vec, quat) where quat is (w, x, y, z) and the active
    transform is translation first, then rotation.
  - Trajectory pose is local->aabb (i.e. world->local active transform).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from alpasim_grpc.v0 import common_pb2

try:
    import carla  # type: ignore
except ImportError:  # carla is optional so unit tests can run without it
    carla = None


@dataclass(frozen=True)
class WorldPose:
    """Minimal world-frame pose used as an intermediate representation."""

    x: float
    y: float
    z: float
    qw: float
    qx: float
    qy: float
    qz: float


def _euler_deg_to_quat(
    pitch_deg: float, yaw_deg: float, roll_deg: float
) -> tuple[float, float, float, float]:
    """Convert CARLA-style Euler (pitch, yaw, roll, degrees) to (w, x, y, z)."""
    half_p = math.radians(pitch_deg) * 0.5
    half_y = math.radians(yaw_deg) * 0.5
    half_r = math.radians(roll_deg) * 0.5

    cp, sp = math.cos(half_p), math.sin(half_p)
    cy, sy = math.cos(half_y), math.sin(half_y)
    cr, sr = math.cos(half_r), math.sin(half_r)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (w, x, y, z)


def carla_transform_to_world_pose(transform) -> WorldPose:
    """Convert a `carla.Transform` into a WorldPose."""
    loc = transform.location
    rot = transform.rotation
    qw, qx, qy, qz = _euler_deg_to_quat(rot.pitch, rot.yaw, rot.roll)
    return WorldPose(x=loc.x, y=loc.y, z=loc.z, qw=qw, qx=qx, qy=qy, qz=qz)


def world_pose_to_grpc(world_pose: WorldPose) -> common_pb2.Pose:
    """Convert a WorldPose to a common.Pose message.

    NOTE: Alpasim consumers expect the local->aabb active transform. Callers
    should invert/compose with the AABB origin before serialising in
    production; for the initial scaffold we forward the world pose directly
    and let downstream replay tests detect the gap.
    """
    return common_pb2.Pose(
        vec=common_pb2.Vec3(x=world_pose.x, y=world_pose.y, z=world_pose.z),
        quat=common_pb2.Quat(
            w=world_pose.qw, x=world_pose.qx, y=world_pose.qy, z=world_pose.qz
        ),
    )


def grpc_pose_to_carla_transform(pose: common_pb2.Pose):
    """Convert a common.Pose into a `carla.Transform`.

    Used to apply ego pose updates received from Runtime. Inverse of
    `_euler_deg_to_quat` — full quaternion -> Euler conversion (XYZ intrinsic).
    """
    if carla is None:
        raise RuntimeError("carla Python API is not installed in this environment")

    qw, qx, qy, qz = pose.quat.w, pose.quat.x, pose.quat.y, pose.quat.z

    # Quaternion to ZYX Euler matching CARLA's pitch/yaw/roll convention
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)

    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return carla.Transform(
        carla.Location(x=pose.vec.x, y=pose.vec.y, z=pose.vec.z),
        carla.Rotation(
            pitch=math.degrees(pitch),
            yaw=math.degrees(yaw),
            roll=math.degrees(roll),
        ),
    )


def actor_bounding_box_to_grpc(actor) -> common_pb2.AABB:
    """Extract a `common.AABB` from a CARLA actor (full extent, not half)."""
    bb = actor.bounding_box
    return common_pb2.AABB(
        size_x=bb.extent.x * 2.0,
        size_y=bb.extent.y * 2.0,
        size_z=bb.extent.z * 2.0,
    )
