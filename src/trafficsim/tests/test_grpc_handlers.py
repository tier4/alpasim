"""Unit tests for the trafficsim gRPC servicer.

The real CARLA Python API is not installed in CI, so we inject a fake `carla`
module + fake World/Actor/TrafficManager before importing the servicer. The
tests cover session lifecycle (start → simulate → close) and a few error paths
that map to gRPC status codes.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest import mock

import pytest


def _install_fake_carla(monkeypatch) -> types.ModuleType:
    """Insert a minimal fake `carla` module into sys.modules."""

    fake = types.ModuleType("carla")

    class _Vec:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x = x
            self.y = y
            self.z = z

    class _Rot:
        def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
            self.pitch = pitch
            self.yaw = yaw
            self.roll = roll

    class _Tf:
        def __init__(self, location=None, rotation=None):
            self.location = location or _Vec()
            self.rotation = rotation or _Rot()

    fake.Location = _Vec
    fake.Rotation = _Rot
    fake.Transform = _Tf
    fake.Vector3D = _Vec  # nothing depends on this but mimic real API
    monkeypatch.setitem(sys.modules, "carla", fake)
    return fake


class _FakeActor:
    def __init__(self, transform):
        self._transform = transform
        self.bounding_box = types.SimpleNamespace(extent=types.SimpleNamespace(x=2.0, y=1.0, z=0.7))
        self.destroyed = False
        self.autopilot_calls: list[tuple[bool, int]] = []

    def set_transform(self, t):
        self._transform = t

    def get_transform(self):
        return self._transform

    def set_autopilot(self, enabled, tm_port):
        self.autopilot_calls.append((enabled, tm_port))

    def destroy(self):
        self.destroyed = True


class _FakeBlueprintLibrary:
    def filter(self, _filter):
        return [object()]


class _FakeWorld:
    def __init__(self, fake_carla, actors_by_id: dict[str, _FakeActor]):
        self._fake_carla = fake_carla
        self._actors_by_id = actors_by_id
        self._settings = types.SimpleNamespace(synchronous_mode=False, fixed_delta_seconds=None)

    def get_settings(self):
        return self._settings

    def apply_settings(self, settings):
        self._settings = settings

    def get_map(self):
        return None

    def get_blueprint_library(self):
        return _FakeBlueprintLibrary()

    def try_spawn_actor(self, _bp, _transform):
        # Hand out a fresh actor instance per call so the session can register
        # distinct objects.
        return _FakeActor(self._fake_carla.Transform())

    def tick(self):
        pass


class _FakeTrafficManager:
    def __init__(self):
        self.sync_mode = False

    def set_synchronous_mode(self, enabled):
        self.sync_mode = enabled

    def vehicle_percentage_speed_difference(self, _actor, _pct):
        pass

    def distance_to_leading_vehicle(self, _actor, _d):
        pass


class _FakeClient:
    def __init__(self, fake_carla):
        self.fake_carla = fake_carla
        self.world = _FakeWorld(fake_carla, {})
        self.tm = _FakeTrafficManager()

    def set_timeout(self, _t):
        pass

    def get_world(self):
        return self.world

    def load_world(self, _name):
        return self.world

    def get_trafficmanager(self, _port):
        return self.tm


@pytest.fixture
def fake_carla(monkeypatch):
    fake = _install_fake_carla(monkeypatch)
    fake.Client = lambda host, port: _FakeClient(fake)
    return fake


@pytest.fixture
def servicer(fake_carla):
    # Import after carla is faked so the module-level `import carla` succeeds.
    from alpasim_trafficsim.server import TrafficSimServicer

    s = TrafficSimServicer(
        carla_host="physics-0",
        carla_port=2000,
        tm_port=8000,
        scenario_path=None,
    )
    return s


def _make_session_request(uuid="sess-1", n_traffic=2):
    from alpasim_grpc.v0 import common_pb2, traffic_pb2

    def _logged(object_id):
        return traffic_pb2.ObjectTrajectory(
            object_id=object_id,
            aabb=common_pb2.AABB(size_x=4.0, size_y=2.0, size_z=1.5),
            trajectory=common_pb2.Trajectory(
                poses=[
                    common_pb2.PoseAtTime(
                        pose=common_pb2.Pose(
                            vec=common_pb2.Vec3(x=0.0, y=0.0, z=0.0),
                            quat=common_pb2.Quat(w=1.0),
                        ),
                        timestamp_us=0,
                    )
                ]
            ),
            is_static=False,
        )

    request = traffic_pb2.TrafficSessionRequest(
        session_uuid=uuid,
        map_id="Town01",
        random_seed=42,
        handover_time_us=int(1e6),
    )
    request.logged_object_trajectories.append(_logged("EGO"))
    for i in range(n_traffic):
        request.logged_object_trajectories.append(_logged(f"npc-{i}"))
    return request


def test_get_metadata_returns_minimum_history(servicer):
    from alpasim_grpc.v0 import common_pb2

    md = servicer.get_metadata(common_pb2.Empty(), context=mock.Mock())
    assert md.minimum_history_length_us == int(1e6)
    assert md.version_id.version_id


def test_start_session_without_scenario_skips_spawning(servicer):
    ctx = mock.Mock()
    status = servicer.start_session(_make_session_request(), context=ctx)
    assert status is not None
    # No scenario_runner -> no actors get spawned.
    assert "sess-1" in servicer._sessions
    assert servicer._sessions["sess-1"].actors == []


def test_close_session_idempotent(servicer):
    ctx = mock.Mock()
    servicer.start_session(_make_session_request("sess-x"), context=ctx)
    servicer.close_session(
        _close_request("sess-x"), context=ctx
    )
    # Second close on an unknown session is a no-op (not an abort).
    servicer.close_session(_close_request("sess-x"), context=ctx)


def test_duplicate_start_session_aborts(servicer):
    ctx = mock.Mock()
    ctx.abort.side_effect = RuntimeError("abort called")
    servicer.start_session(_make_session_request("dup"), context=ctx)
    with pytest.raises(RuntimeError):
        servicer.start_session(_make_session_request("dup"), context=ctx)
    ctx.abort.assert_called_once()


def _close_request(uuid: str):
    from alpasim_grpc.v0 import traffic_pb2

    return traffic_pb2.TrafficSessionCloseRequest(session_uuid=uuid)
