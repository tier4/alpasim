"""Scenario loader for the trafficsim micro-service.

Supports two input shapes — both are read from the host-mounted scenarios
directory at container start, no rebuild required:

1. **Python scenario** (``foo.py``)

   The file is loaded dynamically with ``importlib.util`` and must expose:

   - ``apply(session, request, carla_module, params)`` — required. Called
     once during ``start_session`` after the CARLA world is loaded. Use
     ``session.register_actor(object_id, actor, is_ego=..., is_static=...)``
     in the **same order** as ``request.logged_object_trajectories``.
   - ``MAP_ID`` (str) — optional. CARLA Town to load. Defaults to the
     ``map.id`` field of the sibling YAML if present, else ``"Town01"``.
   - ``FIXED_DELTA_SECONDS`` (float) — optional. Sync-mode tick length.
   - ``SUPPORTED_MAP_IDS`` (Iterable[str]) — optional. Reported by
     ``TrafficService.get_metadata``.

   A sibling ``foo.yaml`` (same basename, in the same directory) is loaded
   into a plain ``dict`` and forwarded as the ``params`` argument so the
   scenario can read tuning knobs without re-parsing the file.

2. **YAML scenario** (``foo.yaml`` / ``foo.yml``)

   Falls back to a built-in declarative spawner that reads ``map.id``,
   ``ego.blueprint``, ``traffic.default_blueprint`` and ``traffic.manager``.
   Adequate for "one ego + N TM-controlled NPCs" scenarios.

One trafficsim container == one scenario. To swap, change what the host
directory contains and restart the container.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

from alpasim_grpc.v0 import traffic_pb2
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


_PYTHON_SUFFIXES = {".py"}
_YAML_SUFFIXES = {".yaml", ".yml"}


def _load_python_scenario(path: Path):
    """Dynamically import a `.py` file outside of PYTHONPATH.

    Each ScenarioRunner instance gets a fresh module, so editing the file on
    the host and restarting the container picks up the new code without any
    Python import-cache concerns.
    """
    spec = importlib.util.spec_from_file_location(f"_alpasim_scenario_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "apply", None)):
        raise RuntimeError(
            f"scenario {path} is missing a top-level `apply(session, request, "
            "carla_module, params)` function"
        )
    return module


class ScenarioRunner:
    """Loads a Python or YAML scenario and applies it to a CarlaSession."""

    def __init__(self, scenario_path: str) -> None:
        path = Path(scenario_path)
        if not path.exists():
            raise FileNotFoundError(f"scenario file not found: {scenario_path}")
        self._path = path

        if path.suffix in _PYTHON_SUFFIXES:
            self._module = _load_python_scenario(path)
            yaml_sibling = path.with_suffix(".yaml")
            if yaml_sibling.exists():
                self._cfg: DictConfig = OmegaConf.load(yaml_sibling)  # type: ignore[assignment]
                logger.info("loaded scenario %s (+ params %s)", path, yaml_sibling)
            else:
                self._cfg = OmegaConf.create({})
                logger.info("loaded scenario %s (no sibling yaml)", path)
        elif path.suffix in _YAML_SUFFIXES:
            self._module = None
            self._cfg = OmegaConf.load(path)  # type: ignore[assignment]
            logger.info("loaded declarative scenario %s", path)
        else:
            raise ValueError(
                f"unsupported scenario extension {path.suffix!r}; expected .py or .yaml"
            )

        # Optional: pre-import autoware_carla_scenario to consume SUPPORTED_MAPS.
        try:
            import autoware_carla_scenario as acs  # type: ignore

            self._acs: Optional[Any] = acs
        except ImportError:
            self._acs = None

    # ----- metadata -----

    @property
    def map_id(self) -> str:
        if self._module is not None:
            mod_map = getattr(self._module, "MAP_ID", None)
            if mod_map:
                return str(mod_map)
        return str(OmegaConf.select(self._cfg, "map.id", default="Town01"))

    @property
    def fixed_delta_seconds(self) -> float:
        if self._module is not None:
            mod_dt = getattr(self._module, "FIXED_DELTA_SECONDS", None)
            if mod_dt is not None:
                return float(mod_dt)
        return float(OmegaConf.select(self._cfg, "simulation.fixed_delta_seconds", default=0.05))

    def supported_map_ids(self) -> Iterable[str]:
        if self._module is not None:
            mod_maps = getattr(self._module, "SUPPORTED_MAP_IDS", None)
            if mod_maps:
                return list(mod_maps)
        if self._acs is not None and hasattr(self._acs, "SUPPORTED_MAPS"):
            return list(self._acs.SUPPORTED_MAPS)
        return [self.map_id]

    # ----- apply -----

    def apply(
        self,
        session,
        request: traffic_pb2.TrafficSessionRequest,
        carla_module,
    ) -> None:
        """Drive `start_session` spawning.

        For Python scenarios we delegate to the user-supplied `apply()`. For
        YAML scenarios we run the built-in declarative spawner.
        """
        session.fixed_delta_seconds = self.fixed_delta_seconds
        self._load_world(session, carla_module)

        if self._module is not None:
            params = OmegaConf.to_container(self._cfg, resolve=True) if len(self._cfg) else {}
            self._module.apply(
                session=session,
                request=request,
                carla_module=carla_module,
                params=params,
            )
            return

        self._apply_declarative(session, request, carla_module)

    # ----- declarative (YAML-only) spawner -----

    def _apply_declarative(
        self,
        session,
        request: traffic_pb2.TrafficSessionRequest,
        carla_module,
    ) -> None:
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
