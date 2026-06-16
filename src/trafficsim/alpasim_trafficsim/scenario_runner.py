"""Thin adapter around autoware_carla_scenario.

The autoware_carla_scenario package (from tier4/autoware_lanelet2_to_opendrive)
exposes a Hydra CLI that loads a Town map, configures TrafficManager and
spawns traffic. We do not call its CLI: that would seize control of the
event loop and the synchronous-mode tick. Instead we treat the YAML files
as a passive declarative config that we apply ourselves via this wrapper.

This file is intentionally tolerant of the package being absent at import
time so the rest of trafficsim_server can be unit-tested without CARLA or
autoware_carla_scenario installed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterable, Optional

from alpasim_grpc.v0 import traffic_pb2
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class ScenarioRunner:
    """Loads a Hydra YAML scenario and applies it to a CarlaSession."""

    def __init__(self, scenario_path: str) -> None:
        path = Path(scenario_path)
        if not path.exists():
            raise FileNotFoundError(f"scenario file not found: {scenario_path}")
        self._path = path
        self._cfg: DictConfig = OmegaConf.load(path)  # type: ignore[assignment]
        logger.info("loaded scenario %s", path)

        # Optional: pre-import the autoware_carla_scenario module so we can
        # delegate spawn details to its built-in helpers. We tolerate ImportError
        # because PR#2 may run before PR#3 has the dependency installed.
        try:
            import autoware_carla_scenario as acs  # type: ignore

            self._acs: Optional[Any] = acs
        except ImportError:
            self._acs = None
            logger.warning(
                "autoware_carla_scenario is not installed; falling back to "
                "a minimal built-in spawner"
            )

    @property
    def map_id(self) -> str:
        return str(OmegaConf.select(self._cfg, "map.id", default="Town01"))

    @property
    def fixed_delta_seconds(self) -> float:
        return float(OmegaConf.select(self._cfg, "simulation.fixed_delta_seconds", default=0.05))

    def supported_map_ids(self) -> Iterable[str]:
        # When autoware_carla_scenario is loaded, return its registered Towns.
        if self._acs is not None and hasattr(self._acs, "SUPPORTED_MAPS"):
            return list(self._acs.SUPPORTED_MAPS)
        return [self.map_id]

    def apply(
        self,
        session,
        request: traffic_pb2.TrafficSessionRequest,
        carla_module,
    ) -> None:
        """Spawn ego + traffic actors into `session`.

        Order matters: we MUST register actors in the same order as
        `request.logged_object_trajectories` so that TrafficReturn preserves
        the order the Runtime expects (see proto comment on TrafficReturn).
        """
        session.fixed_delta_seconds = self.fixed_delta_seconds
        self._load_world(session, carla_module)

        for obj in request.logged_object_trajectories:
            spawn_pose = self._initial_world_transform(obj, carla_module)
            blueprint = self._pick_blueprint(session, obj)
            actor = session.world.try_spawn_actor(blueprint, spawn_pose)
            if actor is None:
                logger.warning("spawn failed for %s (skipping)", obj.object_id)
                continue
            is_ego = obj.object_id == "EGO"
            session.register_actor(
                obj.object_id, actor, is_ego=is_ego, is_static=obj.is_static
            )
            if not is_ego and not obj.is_static:
                self._enrol_in_traffic_manager(session, actor)

    # ----- helpers -----

    def _load_world(self, session, carla_module) -> None:
        target_map = self.map_id
        current = session.world.get_map().name if session.world.get_map() else ""
        if target_map in current:
            return
        logger.info("loading CARLA map %s", target_map)
        session.world = session.client.load_world(target_map)

    def _pick_blueprint(self, session, obj: traffic_pb2.ObjectTrajectory):
        blueprints = session.world.get_blueprint_library()
        if obj.object_id == "EGO":
            filt = OmegaConf.select(self._cfg, "ego.blueprint", default="vehicle.tesla.model3")
        else:
            filt = OmegaConf.select(self._cfg, "traffic.default_blueprint", default="vehicle.audi.a2")
        candidates = blueprints.filter(filt)
        if not candidates:
            raise RuntimeError(f"no CARLA blueprints match {filt!r}")
        return candidates[0]

    def _initial_world_transform(self, obj: traffic_pb2.ObjectTrajectory, carla_module):
        """Take the first PoseAtTime from the logged trajectory.

        The proto stores active local->aabb; for the initial scaffold we
        forward translation as-is to world coordinates. PR#3 will plug in the
        proper frame conversion using autoware_carla_scenario's map helpers.
        """
        if not obj.trajectory.poses:
            return carla_module.Transform()
        pose = obj.trajectory.poses[0].pose
        return carla_module.Transform(
            carla_module.Location(x=pose.vec.x, y=pose.vec.y, z=pose.vec.z + 0.5),
            carla_module.Rotation(),
        )

    def _enrol_in_traffic_manager(self, session, actor) -> None:
        actor.set_autopilot(True, session.tm_port)
        # TM customisation knobs from the scenario YAML
        tm_cfg = OmegaConf.select(self._cfg, "traffic.manager", default=None)
        if tm_cfg is None:
            return
        speed_pct = float(OmegaConf.select(tm_cfg, "speed_difference_pct", default=0.0))
        if speed_pct:
            session.traffic_manager.vehicle_percentage_speed_difference(actor, speed_pct)
        distance_m = float(OmegaConf.select(tm_cfg, "distance_to_leading_vehicle_m", default=0.0))
        if distance_m:
            session.traffic_manager.distance_to_leading_vehicle(actor, distance_m)


def _expand_scenario_path(scenario_arg: str) -> str:
    """Resolve `scenario=foo/bar` or a raw path to an absolute file path."""
    return os.path.expandvars(scenario_arg)
