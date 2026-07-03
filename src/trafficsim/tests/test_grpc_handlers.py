# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

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

# Module-level so ScenarioRunner tests can construct fake carla namespaces.


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
        self.bounding_box = types.SimpleNamespace(
            extent=types.SimpleNamespace(x=2.0, y=1.0, z=0.7)
        )
        self.destroyed = False
        self.autopilot_calls: list[tuple[bool, int]] = []
        self.set_transform_calls: list[Any] = []
        self.physics_enabled: bool = True

    def set_transform(self, t):
        self._transform = t
        self.set_transform_calls.append(t)

    def get_transform(self):
        return self._transform

    def set_autopilot(self, enabled, tm_port):
        self.autopilot_calls.append((enabled, tm_port))

    def set_simulate_physics(self, enabled):
        self.physics_enabled = enabled

    def destroy(self):
        self.destroyed = True


class _FakeBlueprintLibrary:
    def filter(self, _filter):
        return [object()]


class _FakeWorld:
    def __init__(self, fake_carla, actors_by_id: dict[str, _FakeActor]):
        self._fake_carla = fake_carla
        self._actors_by_id = actors_by_id
        self._settings = types.SimpleNamespace(
            synchronous_mode=False, fixed_delta_seconds=None
        )

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
    servicer.close_session(_close_request("sess-x"), context=ctx)
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


# =============================================================================
# ScenarioRunner: Python + YAML loading
# =============================================================================


def test_scenario_runner_loads_python_module_and_sibling_yaml(tmp_path):
    """A `.py` scenario is dynamically imported and a sibling `.yaml` becomes params."""
    from alpasim_trafficsim.scenario_runner import ScenarioRunner

    scenario_py = tmp_path / "demo.py"
    scenario_py.write_text(
        "MAP_ID = 'Town02'\n"
        "FIXED_DELTA_SECONDS = 0.025\n"
        "SUPPORTED_MAP_IDS = ['Town02']\n"
        "calls = []\n"
        "def apply(session, request, carla_module, params):\n"
        "    calls.append({'params': params, 'objects': [o.object_id for o in request.logged_object_trajectories]})\n"
    )
    (tmp_path / "demo.yaml").write_text("ego:\n  blueprint: vehicle.tesla.model3\n")

    runner = ScenarioRunner(str(scenario_py))

    assert runner.map_id == "Town02"
    assert runner.fixed_delta_seconds == 0.025
    assert list(runner.supported_map_ids()) == ["Town02"]

    # Drive a single apply() — use a stub session that captures registrations.
    request = _make_session_request("sess", n_traffic=2)
    runner._module.apply.__wrapped__ = None  # type: ignore[attr-defined]  # no-op sentinel
    fake_session = mock.MagicMock()
    fake_session.world.get_map.return_value = None
    fake_carla = types.SimpleNamespace()
    runner.apply(session=fake_session, request=request, carla_module=fake_carla)

    captured = runner._module.calls  # type: ignore[attr-defined]
    assert len(captured) == 1
    assert captured[0]["objects"] == ["EGO", "npc-0", "npc-1"]
    assert captured[0]["params"]["ego"]["blueprint"] == "vehicle.tesla.model3"


def test_scenario_runner_rejects_python_without_apply(tmp_path):
    """A `.py` file without an `apply` callable fails at load time, not at run time."""
    from alpasim_trafficsim.scenario_runner import ScenarioRunner

    bad = tmp_path / "broken.py"
    bad.write_text("MAP_ID = 'Town01'\n")  # no apply()

    with pytest.raises(RuntimeError, match="missing a top-level `apply"):
        ScenarioRunner(str(bad))


def test_scenario_runner_python_without_sibling_yaml_passes_empty_params(tmp_path):
    """If no sibling yaml exists, params is an empty dict."""
    from alpasim_trafficsim.scenario_runner import ScenarioRunner

    scenario_py = tmp_path / "demo.py"
    scenario_py.write_text(
        "received = {}\n"
        "def apply(session, request, carla_module, params):\n"
        "    received['params'] = params\n"
    )

    runner = ScenarioRunner(str(scenario_py))
    fake_session = mock.MagicMock()
    fake_session.world.get_map.return_value = None
    runner.apply(
        session=fake_session,
        request=_make_session_request("s", n_traffic=0),
        carla_module=types.SimpleNamespace(),
    )

    assert runner._module.received["params"] == {}  # type: ignore[attr-defined]


def test_scenario_runner_rejects_unknown_extension(tmp_path):
    from alpasim_trafficsim.scenario_runner import ScenarioRunner

    bad = tmp_path / "scenario.txt"
    bad.write_text("anything")

    with pytest.raises(ValueError, match="unsupported scenario extension"):
        ScenarioRunner(str(bad))


def test_scenario_runner_loads_plain_yaml(tmp_path):
    """A `.yaml` scenario without a Python file uses the declarative spawner."""
    from alpasim_trafficsim.scenario_runner import ScenarioRunner

    yaml_path = tmp_path / "demo.yaml"
    yaml_path.write_text(
        "map:\n  id: Town03\n" "simulation:\n  fixed_delta_seconds: 0.1\n"
    )

    runner = ScenarioRunner(str(yaml_path))
    assert runner._module is None
    assert runner.map_id == "Town03"
    assert runner.fixed_delta_seconds == 0.1


# =============================================================================
# ControlMode: per-actor gRPC vs TrafficManager dispatch
# =============================================================================


def test_resolve_grpc_driven_back_compat_unset():
    """CONTROL_MODE_UNSPECIFIED falls back to 'EGO is gRPC, rest is TM'."""
    from alpasim_grpc.v0 import traffic_pb2
    from alpasim_trafficsim.scenario_runner import resolve_grpc_driven

    ego = traffic_pb2.ObjectTrajectory(object_id="EGO")
    npc = traffic_pb2.ObjectTrajectory(object_id="npc-0")

    assert resolve_grpc_driven(ego) is True
    assert resolve_grpc_driven(npc) is False


def test_resolve_grpc_driven_explicit_mode_wins():
    """Explicit control_mode overrides the EGO-vs-rest heuristic."""
    from alpasim_grpc.v0 import traffic_pb2
    from alpasim_trafficsim.scenario_runner import resolve_grpc_driven

    # An NPC explicitly marked as GRPC_REPLAY (e.g. log-replay) -> True.
    npc_replay = traffic_pb2.ObjectTrajectory(
        object_id="npc-replay",
        control_mode=traffic_pb2.CONTROL_MODE_GRPC_REPLAY,
    )
    # EGO explicitly marked as TM (e.g. open-loop traffic study) -> False.
    ego_tm = traffic_pb2.ObjectTrajectory(
        object_id="EGO",
        control_mode=traffic_pb2.CONTROL_MODE_TRAFFIC_MANAGER,
    )

    assert resolve_grpc_driven(npc_replay) is True
    assert resolve_grpc_driven(ego_tm) is False


def test_apply_pose_update_targets_only_grpc_driven_actors(fake_carla):
    """simulate()'s per-actor dispatch: gRPC actors get set_transform, TM ones don't."""
    from alpasim_grpc.v0 import common_pb2, traffic_pb2
    from alpasim_trafficsim.carla_session import CarlaSession

    session = CarlaSession(
        session_uuid="s",
        map_id="Town01",
        carla_host="h",
        carla_port=1,
        tm_port=2,
    )
    ego_actor = _FakeActor(fake_carla.Transform())
    tm_actor = _FakeActor(fake_carla.Transform())
    session.register_actor("EGO", ego_actor, is_ego=True, is_grpc_driven=True)
    session.register_actor("npc-0", tm_actor, is_grpc_driven=False)

    def _update(object_id, x):
        return traffic_pb2.ObjectTrajectoryUpdate(
            object_id=object_id,
            trajectory=common_pb2.Trajectory(
                poses=[
                    common_pb2.PoseAtTime(
                        pose=common_pb2.Pose(
                            vec=common_pb2.Vec3(x=x, y=0.0, z=0.0),
                            quat=common_pb2.Quat(w=1.0),
                        ),
                        timestamp_us=0,
                    )
                ],
            ),
        )

    session.apply_pose_update(_update("EGO", 10.0))
    session.apply_pose_update(_update("npc-0", 20.0))
    session.apply_pose_update(_update("unknown", 30.0))  # silently ignored

    assert len(ego_actor.set_transform_calls) == 1
    # TM-driven actor and unknown id must not be touched via gRPC.
    assert tm_actor.set_transform_calls == []


def test_resolve_grpc_driven_static_actor_is_never_grpc_driven():
    """is_static=True objects ignore control_mode and are never gRPC-driven."""
    from alpasim_grpc.v0 import traffic_pb2
    from alpasim_trafficsim.scenario_runner import resolve_grpc_driven

    static_replay = traffic_pb2.ObjectTrajectory(
        object_id="cone-0",
        is_static=True,
        control_mode=traffic_pb2.CONTROL_MODE_GRPC_REPLAY,
    )
    static_ego = traffic_pb2.ObjectTrajectory(object_id="EGO", is_static=True)

    assert resolve_grpc_driven(static_replay) is False
    assert resolve_grpc_driven(static_ego) is False


def test_register_actor_disables_physics_for_grpc_driven(fake_carla):
    """register_actor centralizes set_simulate_physics(False) for gRPC actors."""
    from alpasim_trafficsim.carla_session import CarlaSession

    session = CarlaSession(
        session_uuid="s",
        map_id="Town01",
        carla_host="h",
        carla_port=1,
        tm_port=2,
    )
    grpc_actor = _FakeActor(fake_carla.Transform())
    tm_actor = _FakeActor(fake_carla.Transform())
    static_actor = _FakeActor(fake_carla.Transform())

    session.register_actor("ego", grpc_actor, is_grpc_driven=True)
    session.register_actor("npc", tm_actor, is_grpc_driven=False)
    session.register_actor("cone", static_actor, is_static=True, is_grpc_driven=True)

    assert grpc_actor.physics_enabled is False
    assert tm_actor.physics_enabled is True
    # Static actors keep physics — register_actor must not flip them off.
    assert static_actor.physics_enabled is True


def test_register_actor_back_compat_defaults_grpc_driven_from_is_ego(fake_carla):
    """Pre-ControlMode scenarios pass is_ego only; EGO must still receive updates."""
    from alpasim_trafficsim.carla_session import CarlaSession

    session = CarlaSession(
        session_uuid="s",
        map_id="Town01",
        carla_host="h",
        carla_port=1,
        tm_port=2,
    )
    ego = _FakeActor(fake_carla.Transform())
    npc = _FakeActor(fake_carla.Transform())
    session.register_actor("EGO", ego, is_ego=True)
    session.register_actor("npc", npc)

    assert session._actors_by_id["EGO"].is_grpc_driven is True
    assert session._actors_by_id["npc"].is_grpc_driven is False
