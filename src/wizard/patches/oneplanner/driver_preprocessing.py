"""Map alpasim EgoDriver observations to OnePlanner sample tensors.

The session-state observations submit_* RPCs push into the servicer arrive in
alpasim's frame conventions:
  - ego trajectory poses are rig poses in the *local* world frame
  - LiDAR xyz is in the LiDAR (≈ rig) frame at the end-of-spin
  - route waypoints are in the rig frame at the route timestamp

The planner consumes everything in the current ego frame at ``time_now_us``.
This module owns that frame conversion and the construction of the sample
dict that ``E2EDataset.__getitem__`` produces — so the model sees the same
contract whether the batch came from the training set or live alpasim.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from oneplanner import constants as C
from oneplanner.deployment.hdmap import (
    FrameAlignment,
    LaneletMap,
    build_map_tensors,
)

_ALIGN_DEBUG_LOG = logging.getLogger("oneplanner.deployment.align_debug")
_ALIGN_DEBUG_LOG.setLevel(logging.INFO)
_ALIGN_DEBUG_TICK = 0
_ALIGN_DEBUG_EVERY_N = int(os.environ.get("ONEPLANNER_ALIGN_DEBUG_EVERY_N", "0") or 0)
_SAMPLE_DUMP_DIR = os.environ.get("ONEPLANNER_SAMPLE_DUMP_DIR", "")
_SAMPLE_DUMP_EVERY_N = int(os.environ.get("ONEPLANNER_SAMPLE_DUMP_EVERY_N", "0") or 0)


@dataclass
class EgoPoseSample:
    """One rig pose in local frame at ``timestamp_us``, plus its dynamic state."""

    timestamp_us: int
    # 4x4 active transform local -> rig (column-major math; we store the matrix
    # form to avoid carrying a quaternion library at the data layer).
    local_to_rig: np.ndarray
    # [vx, vy, ax, ay, yaw_rate] in rig frame; missing entries default to 0.
    dynamic_state: np.ndarray


@dataclass
class SessionObservations:
    """Cumulative state populated by submit_* RPCs and read by ``drive``."""

    ego_history: list[EgoPoseSample] = field(default_factory=list)
    lidar_xyz: np.ndarray | None = None  # [N, 3] in LiDAR frame
    lidar_intensity: np.ndarray | None = None  # [N]
    route_waypoints_rig: np.ndarray | None = None  # [K, 3] in rig frame
    route_timestamp_us: int | None = None
    ego_shape: np.ndarray = field(
        default_factory=lambda: np.array([2.7, 4.5, 1.85], dtype=np.float32)
    )


# ----------------------------------------------------------------------------
# Wire-format decoders
# ----------------------------------------------------------------------------


def decode_lidar_buffers(
    point_xyzs_buffer: bytes,
    point_intensities_buffer: bytes,
    point_ring_ids_buffer: bytes,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (xyz[N,3] float32, intensity[N] float32, ring[N] uint16)."""
    if num_points == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.uint16),
        )
    xyz = np.frombuffer(point_xyzs_buffer, dtype=np.float32).reshape(num_points, 3)
    intensity = np.frombuffer(point_intensities_buffer, dtype=np.float32)
    ring = np.frombuffer(point_ring_ids_buffer, dtype=np.uint16)
    if intensity.shape[0] != num_points or ring.shape[0] != num_points:
        raise ValueError(
            f"LiDAR buffer length mismatch: xyz={num_points}, "
            f"intensity={intensity.shape[0]}, ring={ring.shape[0]}"
        )
    return xyz.astype(np.float32, copy=False), intensity.astype(np.float32, copy=False), ring


def quat_to_rotmat(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Hamilton-quaternion -> 3x3 rotation matrix (right-handed, active)."""
    if w == 0.0 and x == 0.0 and y == 0.0 and z == 0.0:
        return np.eye(3, dtype=np.float64)
    return Rotation.from_quat([x, y, z, w]).as_matrix().astype(np.float64)


def pose_proto_to_matrix(pose) -> np.ndarray:
    """common.Pose -> 4x4 active transform (translate then rotate per proto convention)."""
    q = pose.quat
    v = pose.vec
    R = quat_to_rotmat(q.w, q.x, q.y, q.z)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [v.x, v.y, v.z]
    return T


def yaw_from_rotmat(R: np.ndarray) -> float:
    """Extract yaw (rotation about +Z) from a 3x3 rotation matrix."""
    return float(np.arctan2(R[1, 0], R[0, 0]))


def vec3_proto_to_xyz(v) -> tuple[float, float, float]:
    return float(v.x), float(v.y), float(v.z)


def dynamic_state_to_vec(dyn_state) -> np.ndarray:
    """common.DynamicState -> [vx, vy, ax, ay, yaw_rate] in rig frame."""
    lv = dyn_state.linear_velocity
    la = dyn_state.linear_acceleration
    av = dyn_state.angular_velocity
    return np.array(
        [lv.x, lv.y, la.x, la.y, av.z],
        dtype=np.float32,
    )


# ----------------------------------------------------------------------------
# Sample construction
# ----------------------------------------------------------------------------

_PAST_LEN = C.INPUT_T + 1  # 31: 30 past + current
_FUTURE_LEN = C.OUTPUT_T  # 80


def _pick_current(history: Sequence[EgoPoseSample], time_now_us: int) -> EgoPoseSample:
    """Most recent ego pose at or before ``time_now_us``; falls back to newest."""
    if not history:
        raise ValueError("ego_history is empty — submit_egomotion_observation before drive()")
    eligible = [s for s in history if s.timestamp_us <= time_now_us]
    return eligible[-1] if eligible else history[-1]


def build_ego_past_in_current_frame(
    history: Sequence[EgoPoseSample],
    current: EgoPoseSample,
) -> np.ndarray:
    """Construct ``ego_agent_past[31, 4]`` = [x_ego, y_ego, cos_yaw, sin_yaw], 10 Hz.

    The current pose is placed at index 30 as the origin; older poses are
    sampled at 100 ms steps backward and re-expressed in the current rig
    frame. When a step has no observation, the nearest earlier sample is
    used; samples before any observation are zero-filled (origin).
    """
    out = np.zeros((_PAST_LEN, 4), dtype=np.float32)
    out[-1] = [0.0, 0.0, 1.0, 0.0]
    if not history:
        return out

    rig_to_local_now = current.local_to_rig
    local_to_rig_now = np.linalg.inv(rig_to_local_now)

    # Build a quick searchable timeline.
    ts = np.array([s.timestamp_us for s in history], dtype=np.int64)
    order = np.argsort(ts)
    ts_sorted = ts[order]
    history_sorted = [history[i] for i in order]

    step_us = 100_000  # 10 Hz
    for k in range(_PAST_LEN - 1):
        target_us = current.timestamp_us - (_PAST_LEN - 1 - k) * step_us
        idx = int(np.searchsorted(ts_sorted, target_us, side="right") - 1)
        if idx < 0:
            continue
        sample = history_sorted[idx]
        local_to_rig_k = sample.local_to_rig  # rig_k in local
        # Express rig_k in current rig frame:
        rig_k_in_rig_now = local_to_rig_now @ local_to_rig_k
        x, y = rig_k_in_rig_now[0, 3], rig_k_in_rig_now[1, 3]
        yaw = yaw_from_rotmat(rig_k_in_rig_now[:3, :3])
        out[k] = [x, y, np.cos(yaw), np.sin(yaw)]
    return out


def build_ego_current_state(current: EgoPoseSample) -> np.ndarray:
    """``ego_current_state[10]`` = [0, 0, 1, 0, vx, vy, ax, ay, steer=0, yaw_rate]."""
    dyn = current.dynamic_state
    if dyn.shape[0] < 5:
        dyn = np.pad(dyn, (0, 5 - dyn.shape[0]))
    vx, vy, ax, ay, yaw_rate = dyn[:5]
    return np.array(
        [0.0, 0.0, 1.0, 0.0, vx, vy, ax, ay, 0.0, yaw_rate],
        dtype=np.float32,
    )


def build_route_lanes_from_waypoints(
    route_waypoints_rig: np.ndarray | None,
    route_timestamp_us: int | None,
    current: EgoPoseSample,
    route_history_for_timestamp: Sequence[EgoPoseSample] = (),
) -> np.ndarray:
    """``route_lanes[25, 20, 33]`` derived from the alpasim Route message.

    The route is given in the rig frame *at its own* timestamp; we re-express
    waypoints in the current rig frame by composing through `local`. Each
    20-point lanelet is one stride along the route; we pad to
    NUM_SEGMENTS_IN_ROUTE with zeros. Only the X/Y position and tangent
    direction channels are filled — lane boundaries, traffic-light state, and
    line-type one-hots stay zero because the alpasim wire format doesn't
    carry them.
    """
    out = np.zeros(
        (C.NUM_SEGMENTS_IN_ROUTE, C.POINTS_PER_LANELET, C.SEGMENT_POINT_DIM),
        dtype=np.float32,
    )
    if route_waypoints_rig is None or len(route_waypoints_rig) == 0:
        return out

    # Re-express route waypoints into the current rig frame.
    if route_timestamp_us is not None and route_history_for_timestamp:
        route_anchor = _pick_current(route_history_for_timestamp, route_timestamp_us)
        local_to_rig_route = route_anchor.local_to_rig
    else:
        # No anchor — assume route is already in the current rig frame.
        local_to_rig_route = current.local_to_rig

    local_to_rig_now_inv = np.linalg.inv(current.local_to_rig)
    rig_route_to_rig_now = local_to_rig_now_inv @ local_to_rig_route

    wp_h = np.concatenate(
        [
            route_waypoints_rig,
            np.ones((route_waypoints_rig.shape[0], 1), dtype=route_waypoints_rig.dtype),
        ],
        axis=1,
    )  # [K, 4]
    wp_now = (rig_route_to_rig_now @ wp_h.T).T[:, :3]  # [K, 3]

    # HOTFIX: alpasim sends only NUM_WAYPOINTS=20 points but this function
    # expects NUM_SEGMENTS_IN_ROUTE*POINTS_PER_LANELET=500 to fill all 25
    # lanelet slots. With 20 points we fill only segment 0 and pad the rest
    # with zeros, which trains OnePlanner to see "no route" and predict a
    # stationary trajectory. Upsample by arc-length linear interpolation so
    # every segment gets meaningful X/Y/tangent signal.
    pts_per = C.POINTS_PER_LANELET
    target_pts = C.NUM_SEGMENTS_IN_ROUTE * pts_per
    if wp_now.shape[0] >= 2 and wp_now.shape[0] < target_pts:
        seg_lens = np.linalg.norm(np.diff(wp_now[:, :2], axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg_lens)])  # cumulative arc length
        total_arc = float(s[-1])
        if total_arc > 1e-6:
            s_new = np.linspace(0.0, total_arc, target_pts)
            wp_new = np.empty((target_pts, 3), dtype=wp_now.dtype)
            for axis in range(3):
                wp_new[:, axis] = np.interp(s_new, s, wp_now[:, axis])
            wp_now = wp_new

    # Stride into NUM_SEGMENTS_IN_ROUTE lanelets of POINTS_PER_LANELET points.
    total_pts = wp_now.shape[0]
    max_pts_used = min(total_pts, C.NUM_SEGMENTS_IN_ROUTE * pts_per)
    used = wp_now[:max_pts_used]

    # Compute tangent directions via centered differences (forward at endpoints).
    if used.shape[0] >= 2:
        diffs = np.zeros_like(used[:, :2])
        diffs[1:-1] = used[2:, :2] - used[:-2, :2]
        diffs[0] = used[1, :2] - used[0, :2]
        diffs[-1] = used[-1, :2] - used[-2, :2]
        norms = np.linalg.norm(diffs, axis=1, keepdims=True)
        norms = np.where(norms > 1e-6, norms, 1.0)
        tangents = diffs / norms
    else:
        tangents = np.zeros((used.shape[0], 2), dtype=used.dtype)

    for seg_idx in range(C.NUM_SEGMENTS_IN_ROUTE):
        lo, hi = seg_idx * pts_per, (seg_idx + 1) * pts_per
        if lo >= used.shape[0]:
            break
        seg = used[lo : min(hi, used.shape[0])]
        seg_tan = tangents[lo : min(hi, used.shape[0])]
        n = seg.shape[0]
        out[seg_idx, :n, 0] = seg[:, 0]  # X
        out[seg_idx, :n, 1] = seg[:, 1]  # Y
        out[seg_idx, :n, 2] = seg_tan[:, 0]  # dX
        out[seg_idx, :n, 3] = seg_tan[:, 1]  # dY
    return out


def build_goal_pose_from_route(
    route_waypoints_rig: np.ndarray | None,
    route_timestamp_us: int | None,
    current: EgoPoseSample,
    route_history_for_timestamp: Sequence[EgoPoseSample] = (),
) -> np.ndarray:
    """``goal_pose[4]`` = [gx, gy, cos_h, sin_h] in current ego frame.

    Heading is derived from the tangent at the last waypoint. Returns zeros
    when no route has been submitted.
    """
    out = np.zeros((4,), dtype=np.float32)
    if route_waypoints_rig is None or len(route_waypoints_rig) == 0:
        return out

    if route_timestamp_us is not None and route_history_for_timestamp:
        route_anchor = _pick_current(route_history_for_timestamp, route_timestamp_us)
        local_to_rig_route = route_anchor.local_to_rig
    else:
        local_to_rig_route = current.local_to_rig

    rig_route_to_rig_now = np.linalg.inv(current.local_to_rig) @ local_to_rig_route
    wp_h = np.concatenate(
        [
            route_waypoints_rig,
            np.ones((route_waypoints_rig.shape[0], 1), dtype=route_waypoints_rig.dtype),
        ],
        axis=1,
    )
    wp_now = (rig_route_to_rig_now @ wp_h.T).T[:, :3]

    gx, gy = wp_now[-1, 0], wp_now[-1, 1]
    if wp_now.shape[0] >= 2:
        tan = wp_now[-1, :2] - wp_now[-2, :2]
        heading = float(np.arctan2(tan[1], tan[0]))
    else:
        heading = 0.0
    out[0], out[1] = gx, gy
    out[2], out[3] = np.cos(heading), np.sin(heading)
    return out


def build_sample(
    obs: SessionObservations,
    time_now_us: int,
    *,
    lanelet_map: LaneletMap | None = None,
    alignment: FrameAlignment | None = None,
) -> dict[str, np.ndarray]:
    """Construct the per-sample dict ``e2e_collate_fn`` consumes.

    Only the keys needed by ``E2EPlanner.forward`` at eval time are populated;
    map fidelity is intentionally lossy (only ``route_lanes`` is non-zero)
    because the alpasim contract doesn't carry an HD-map graph. See module
    docstring.
    """
    current = _pick_current(obs.ego_history, time_now_us)

    ego_past = build_ego_past_in_current_frame(obs.ego_history, current)
    ego_now = build_ego_current_state(current)
    route_lanes = build_route_lanes_from_waypoints(
        obs.route_waypoints_rig, obs.route_timestamp_us, current, obs.ego_history
    )
    goal = build_goal_pose_from_route(
        obs.route_waypoints_rig, obs.route_timestamp_us, current, obs.ego_history
    )

    # LiDAR -> [N, 5] (x, y, z, intensity, ring_as_float). Empty cloud
    # passes a single zero row so downstream voxelization doesn't choke on
    # an all-empty batch.
    if obs.lidar_xyz is None or obs.lidar_xyz.shape[0] == 0:
        points = np.zeros((1, 5), dtype=np.float32)
    else:
        n = obs.lidar_xyz.shape[0]
        points = np.zeros((n, 5), dtype=np.float32)
        points[:, :3] = obs.lidar_xyz
        if obs.lidar_intensity is not None and obs.lidar_intensity.shape[0] == n:
            points[:, 3] = obs.lidar_intensity
        # ring channel kept as zero — alpasim ring ids are uint16 and the
        # trained voxelizer treats this column as a feature, not an index.

    # Map tokens: driven by the USDZ-bundled lanelet2 map when the caller
    # supplies both ``lanelet_map`` and ``alignment`` (the standard deployment
    # path — the driver servicer loads them once per scene at startup). When
    # either is missing we fall back to zero-padded tokens; the encoder handles
    # zero-padded segments through its standard segment-mask path (used for
    # short-context training scenes), so this is a graceful degradation rather
    # than an invalid input.
    if lanelet_map is not None and alignment is not None:
        map_tensors = build_map_tensors(
            lanelet_map,
            alignment,
            current.local_to_rig,
        )
        lanes = map_tensors["lanes"]
        lanes_sl = map_tensors["lanes_speed_limit"]
        lanes_has_sl = map_tensors["lanes_has_speed_limit"]
        polygons = map_tensors["polygons"]
        line_strings = map_tensors["line_strings"]

        # Debug: verify alpasim's "local" frame matches OnePlanner's alignment "local".
        # Root cause suspicion: rerun shows ego_history + prediction crossing road_borders,
        # which implies road_border draws in a different frame than ego. Log the pieces so
        # we can compare numerically. Print every N-th call (default off unless env set).
        global _ALIGN_DEBUG_TICK
        if _ALIGN_DEBUG_EVERY_N > 0 and _ALIGN_DEBUG_TICK % _ALIGN_DEBUG_EVERY_N == 0:
            local_to_rig_mat = np.asarray(current.local_to_rig, dtype=np.float64)
            ego_in_local = local_to_rig_mat[:3, 3]
            _yaw_deg_local_to_rig = float(np.degrees(np.arctan2(local_to_rig_mat[1, 0], local_to_rig_mat[0, 0])))
            _ALIGN_DEBUG_LOG.info(
                "ALIGN_DEBUG tick=%d local_to_rig yaw_deg=%.3f rot=%s translation=%s",
                _ALIGN_DEBUG_TICK, _yaw_deg_local_to_rig,
                np.round(local_to_rig_mat[:3, :3], 4).tolist(),
                np.round(local_to_rig_mat[:3, 3], 3).tolist(),
            )
            map_origin_in_local = np.asarray(alignment.local_to_map, dtype=np.float64)[:3, 3]
            local_origin_in_map = np.asarray(alignment.map_to_local, dtype=np.float64)[:3, 3]
            ep = np.asarray(ego_past, dtype=np.float64)
            ep_xy_min = ep[..., :2].min(axis=tuple(range(ep.ndim - 1))).tolist()
            ep_xy_max = ep[..., :2].max(axis=tuple(range(ep.ndim - 1))).tolist()
            ls = np.asarray(line_strings, dtype=np.float64)
            # first non-zero road_border point (any slot, any point) in ego-rig frame
            mask = np.any(ls[..., :2] != 0, axis=-1)
            if mask.any():
                idx = np.argwhere(mask)[0]
                first_ls_pt = ls[idx[0], idx[1], :2]
                ls_xy_min = ls[..., :2][mask].min(axis=0)
                ls_xy_max = ls[..., :2][mask].max(axis=0)
            else:
                first_ls_pt = np.zeros(2)
                ls_xy_min = np.zeros(2)
                ls_xy_max = np.zeros(2)
            _ALIGN_DEBUG_LOG.info(
                "ALIGN_DEBUG tick=%d ego_in_local=%s map_origin_in_local=%s "
                "local_origin_in_map=%s dist_ego_to_map_origin_in_local=%.3fm "
                "ego_past xy_range_in_rig=[%s .. %s] "
                "line_strings first_pt_in_rig=%s xy_range_in_rig=[%s .. %s] "
                "n_nonzero_pts=%d",
                _ALIGN_DEBUG_TICK,
                np.round(ego_in_local, 3).tolist(),
                np.round(map_origin_in_local, 3).tolist(),
                np.round(local_origin_in_map, 3).tolist(),
                float(np.linalg.norm(ego_in_local - map_origin_in_local)),
                [round(v, 3) for v in ep_xy_min],
                [round(v, 3) for v in ep_xy_max],
                np.round(first_ls_pt, 3).tolist(),
                np.round(ls_xy_min, 3).tolist(),
                np.round(ls_xy_max, 3).tolist(),
                int(mask.sum()),
            )
        _ALIGN_DEBUG_TICK += 1
    else:
        lanes = np.zeros(
            (C.NUM_SEGMENTS_IN_LANE, C.POINTS_PER_LANELET, C.SEGMENT_POINT_DIM),
            dtype=np.float32,
        )
        lanes_sl = np.zeros((C.NUM_SEGMENTS_IN_LANE, 1), dtype=np.float32)
        lanes_has_sl = np.zeros((C.NUM_SEGMENTS_IN_LANE, 1), dtype=np.float32)
        polygons = np.zeros((C.NUM_POLYGONS, C.POINTS_PER_POLYGON, 3), dtype=np.float32)
        line_strings = np.zeros((C.NUM_LINE_STRINGS, C.POINTS_PER_LINE_STRING, 4), dtype=np.float32)

    # route_lanes speed-limit stays zero either way — route_lanes are built
    # from ego waypoints (build_route_lanes_from_waypoints above), not looked
    # up against the HD map, so we have no per-route-segment limit to attach.
    route_sl = np.zeros((C.NUM_SEGMENTS_IN_ROUTE, 1), dtype=np.float32)
    route_has_sl = np.zeros((C.NUM_SEGMENTS_IN_ROUTE, 1), dtype=np.float32)

    sample = {
        "ego_agent_past": ego_past,
        "ego_current_state": ego_now,
        "ego_agent_future": np.zeros((_FUTURE_LEN, 3), dtype=np.float32),
        "lanes": lanes,
        "lanes_speed_limit": lanes_sl,
        "lanes_has_speed_limit": lanes_has_sl,
        "route_lanes": route_lanes,
        "route_lanes_speed_limit": route_sl,
        "route_lanes_has_speed_limit": route_has_sl,
        "polygons": polygons,
        "line_strings": line_strings,
        "ego_shape": obs.ego_shape.astype(np.float32, copy=False),
        "turn_indicators": np.zeros((_PAST_LEN,), dtype=np.int32),
        "goal_pose": goal,
        "points": points,
    }
    if _SAMPLE_DUMP_DIR and _SAMPLE_DUMP_EVERY_N > 0:
        # Save the sample dict as NPZ every Nth call so we can render offline
        # with viz.py and compare with what appears in the .rrd file. Uses the
        # module-level tick counter that ALIGN_DEBUG also increments upstream.
        _tick_for_dump = _ALIGN_DEBUG_TICK - 1 if _ALIGN_DEBUG_TICK > 0 else 0
        if _tick_for_dump % _SAMPLE_DUMP_EVERY_N == 0:
            try:
                dump_dir = _SAMPLE_DUMP_DIR
                os.makedirs(dump_dir, exist_ok=True)
                # NPZ-safe: cast non-numpy leaves to arrays as needed
                out_path = os.path.join(dump_dir, f"sample_tick_{_tick_for_dump:06d}.npz")
                np.savez(out_path, **{k: np.asarray(v) for k, v in sample.items()})
            except Exception as exc:  # noqa: BLE001 — dumps must not break RPC
                _ALIGN_DEBUG_LOG.warning("sample dump failed: %s", exc)
    return sample
