"""Thin adapter around autoware_carla_scenario.

The autoware_carla_scenario package (from tier4/autoware_lanelet2_to_opendrive)
exposes a Hydra CLI that loads a Town map, configures TrafficManager and
spawns traffic. We do not call its CLI: that would seize control of the
event loop and the synchronous-mode tick. Instead we treat the YAML files
as a passive declarative config that we apply ourselves via this wrapper.

The autoware_carla_scenario module import is optional: only the
SUPPORTED_MAPS list is consumed when present. Spawning itself uses CARLA's
own blueprint library and TrafficManager APIs and works without it.
"""

from __future__ import annotations

import logging
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

        # Optional: pre-import autoware_carla_scenario to consume SUPPORTED_MAPS.
        # Spawning itself does not depend on the package.
        try:
            import autoware_carla_scenario as acs  # type: ignore

            self._acs: Optional[Any] = acs
        except ImportError:
            self._acs = None
            logger.info("autoware_carla_scenario is not installed; using local map_id only")

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
        A failed spawn is fatal here — silently skipping would shorten the
        TrafficReturn list and break that ordering invariant.
        """
        session.fixed_delta_seconds = self.fixed_delta_seconds
        self._load_world(session, carla_module)

        # Hoist out per-actor work that doesn't depend on the loop index.
        blueprints = session.world.get_blueprint_library()
        ego_filter = str(OmegaConf.select(self._cfg, "ego.blueprint", default="vehicle.tesla.model3"))
        traffic_filter = str(
            OmegaConf.select(self._cfg, "traffic.default_blueprint", default="vehicle.audi.a2")
        )
        tm_cfg = OmegaConf.select(self._cfg, "traffic.manager", default=None)

        for obj in request.logged_object_trajectories:
            is_ego = obj.object_id == "EGO"
            spawn_pose = self._initial_world_transform(obj, carla_module)
            blueprint = self._pick_blueprint(blueprints, ego_filter if is_ego else traffic_filter)
            actor = session.world.try_spawn_actor(blueprint, spawn_pose)
            if actor is None:
                raise RuntimeError(
                    f"CARLA refused to spawn {obj.object_id}; "
                    "ordered TrafficReturn requires every logged object to spawn"
                )
            session.register_actor(
                obj.object_id, actor, is_ego=is_ego, is_static=obj.is_static
            )
            if not is_ego and not obj.is_static:
                self._enrol_in_traffic_manager(session, actor, tm_cfg)

    # ----- helpers -----

    def _load_world(self, session, carla_module) -> None:
        target_map = self.map_id
        carla_map = session.world.get_map()
        current = carla_map.name if carla_map is not None else ""
        # CARLA map names look like "/Game/Carla/Maps/Town01"; match the final
        # path segment so Town01 doesn't false-match Town10 / Town01_Opt.
        current_basename = current.rsplit("/", 1)[-1]
        if current_basename == target_map:
            return
        logger.info("loading CARLA map %s (was %s)", target_map, current_basename or "<none>")
        session.world = session.client.load_world(target_map)

    @staticmethod
    def _pick_blueprint(blueprints, filt: str):
        candidates = blueprints.filter(filt)
        if not candidates:
            raise RuntimeError(f"no CARLA blueprints match {filt!r}")
        return candidates[0]

    def _initial_world_transform(self, obj: traffic_pb2.ObjectTrajectory, carla_module):
        """Take the first PoseAtTime from the logged trajectory.

        FIXME(carla-frames): the proto stores the active local->aabb transform;
        for the initial scaffold we forward translation as-is to world
        coordinates. Plug in the proper frame conversion (probably via
        autoware_carla_scenario's map helpers) before turning this on outside
        of smoke tests.
        """
        if not obj.trajectory.poses:
            return carla_module.Transform()
        pose = obj.trajectory.poses[0].pose
        return carla_module.Transform(
            carla_module.Location(x=pose.vec.x, y=pose.vec.y, z=pose.vec.z + 0.5),
            carla_module.Rotation(),
        )

    def _enrol_in_traffic_manager(self, session, actor, tm_cfg) -> None:
        actor.set_autopilot(True, session.tm_port)
        if tm_cfg is None:
            return
        speed_pct = float(OmegaConf.select(tm_cfg, "speed_difference_pct", default=0.0))
        if speed_pct:
            session.traffic_manager.vehicle_percentage_speed_difference(actor, speed_pct)
        distance_m = float(OmegaConf.select(tm_cfg, "distance_to_leading_vehicle_m", default=0.0))
        if distance_m:
            session.traffic_manager.distance_to_leading_vehicle(actor, distance_m)
