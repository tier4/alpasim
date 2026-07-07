# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Unit tests for route waypoints flowing into ``PredictionInput``."""

from __future__ import annotations

import numpy as np
from alpasim_grpc.v0.egodriver_pb2 import Route

from ..main import Session, _session_to_route_obs


def _make_session() -> Session:
    return Session(
        uuid="test-session",
        seed=0,
        debug_scene_id="scene-0",
        frame_caches={},
        available_cameras_logical_ids=set(),
        desired_cameras_logical_ids=set(),
        camera_specs={},
    )


def _make_route(
    waypoints: list[tuple[float, float, float]], timestamp_us: int
) -> Route:
    route = Route()
    route.timestamp_us = timestamp_us
    for x, y, z in waypoints:
        wp = route.waypoints.add()
        wp.x = x
        wp.y = y
        wp.z = z
    return route


def test_store_route_populates_waypoint_buffer() -> None:
    session = _make_session()
    route = _make_route(
        waypoints=[(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)],
        timestamp_us=123_456,
    )

    session.store_route(route)

    assert session.latest_route_timestamp_us == 123_456
    buf = session.latest_route_waypoints_rig
    assert buf is not None
    assert buf.shape == (3, 3)
    assert buf.dtype == np.float32
    np.testing.assert_array_equal(
        buf,
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32),
    )


def test_store_route_with_empty_waypoints_resets_cache() -> None:
    session = _make_session()
    session.store_route(_make_route([(1.0, 2.0, 3.0)], timestamp_us=1))
    assert session.latest_route_waypoints_rig is not None

    session.store_route(_make_route([], timestamp_us=2))

    assert session.latest_route_waypoints_rig is None
    assert session.latest_route_timestamp_us is None


def test_store_route_overwrites_previous_route() -> None:
    session = _make_session()
    session.store_route(_make_route([(1.0, 2.0, 3.0)], timestamp_us=100))
    session.store_route(
        _make_route([(10.0, 20.0, 30.0), (40.0, 50.0, 60.0)], timestamp_us=200)
    )

    assert session.latest_route_timestamp_us == 200
    buf = session.latest_route_waypoints_rig
    assert buf is not None
    assert buf.shape == (2, 3)
    np.testing.assert_array_equal(
        buf,
        np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32),
    )


def test_session_to_route_obs_returns_none_before_submit() -> None:
    session = _make_session()
    assert _session_to_route_obs(session) is None


def test_session_to_route_obs_shares_waypoint_buffer() -> None:
    session = _make_session()
    route = _make_route([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)], timestamp_us=42)
    session.store_route(route)

    obs = _session_to_route_obs(session)

    assert obs is not None
    assert obs.timestamp_us == 42
    assert obs.waypoints_rig is session.latest_route_waypoints_rig


def test_store_route_defaults_z_to_zero() -> None:
    """Waypoints with unset z (default 0.0) are still stored verbatim."""
    session = _make_session()
    route = Route()
    route.timestamp_us = 1
    wp = route.waypoints.add()
    wp.x = 1.5
    wp.y = 2.5

    session.store_route(route)

    buf = session.latest_route_waypoints_rig
    assert buf is not None
    np.testing.assert_array_equal(buf, np.array([[1.5, 2.5, 0.0]], dtype=np.float32))
