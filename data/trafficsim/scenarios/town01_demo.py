"""Example trafficsim scenario: ego + N logged NPCs on Town01, TM-controlled.

This file is mounted into the trafficsim container at /mnt/scenarios/ and
loaded dynamically by ScenarioRunner. Sibling town01_demo.yaml supplies
tunable parameters that the runner forwards via the `params` argument.

See src/trafficsim/alpasim_trafficsim/scenario_runner.py for the contract:
- `apply(session, request, carla_module, params)` — required
- `MAP_ID`, `FIXED_DELTA_SECONDS`, `SUPPORTED_MAP_IDS` — optional metadata
"""

from __future__ import annotations

import logging

MAP_ID = "Town01"
FIXED_DELTA_SECONDS = 0.05
SUPPORTED_MAP_IDS = ["Town01"]

logger = logging.getLogger(__name__)


def apply(session, request, carla_module, params):
    """Spawn one ego + every logged traffic object, register them in proto order.

    The CarlaSession already has `world`, `traffic_manager` and an empty
    `actors` list. We MUST call session.register_actor for every entry in
    request.logged_object_trajectories in the same order so TrafficReturn
    keeps the order Runtime expects.
    """
    blueprints = session.world.get_blueprint_library()
    ego_filter = params.get("ego", {}).get("blueprint", "vehicle.tesla.model3")
    npc_filter = params.get("traffic", {}).get("default_blueprint", "vehicle.audi.a2")
    tm_cfg = params.get("traffic", {}).get("manager", {})

    for obj in request.logged_object_trajectories:
        is_ego = obj.object_id == "EGO"
        spawn_pose = _pose_to_transform(obj, carla_module)
        bp_filter = ego_filter if is_ego else npc_filter
        candidates = blueprints.filter(bp_filter)
        if not candidates:
            raise RuntimeError(f"no blueprints match {bp_filter!r}")
        actor = session.world.try_spawn_actor(candidates[0], spawn_pose)
        if actor is None:
            raise RuntimeError(
                f"CARLA refused to spawn {obj.object_id}; ordered TrafficReturn "
                "requires every logged object to spawn"
            )
        session.register_actor(
            obj.object_id, actor, is_ego=is_ego, is_static=obj.is_static
        )
        if is_ego or obj.is_static:
            continue

        actor.set_autopilot(True, session.tm_port)
        speed_pct = float(tm_cfg.get("speed_difference_pct", 0.0))
        if speed_pct:
            session.traffic_manager.vehicle_percentage_speed_difference(actor, speed_pct)
        distance_m = float(tm_cfg.get("distance_to_leading_vehicle_m", 0.0))
        if distance_m:
            session.traffic_manager.distance_to_leading_vehicle(actor, distance_m)

    logger.info("spawned %d actors on %s", len(session.actors), MAP_ID)


def _pose_to_transform(obj, carla_module):
    if not obj.trajectory.poses:
        return carla_module.Transform()
    p = obj.trajectory.poses[0].pose
    return carla_module.Transform(
        carla_module.Location(x=p.vec.x, y=p.vec.y, z=p.vec.z + 0.5),
        carla_module.Rotation(),
    )
