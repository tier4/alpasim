# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Owns CARLA's world clock inside the physics container.

The trafficsim container also holds a CARLA client for spawning actors and
reading pose snapshots, but only the physics container calls
``world.apply_settings(synchronous_mode=True, fixed_delta_seconds=...)`` and
``world.tick()``. Keeping the tick here means alpasim's control loop drives
the physics clock via a single service (the same one that already runs the
CARLA server binary) rather than through the traffic simulator.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CarlaClock:
    """Minimal CARLA client that owns synchronous-mode tick cadence."""

    session_uuid: str
    carla_host: str
    carla_port: int
    fixed_delta_seconds: float

    client: Any = None
    world: Any = None
    last_time_query_us: Optional[int] = None
    _prev_settings: Any = field(default=None)

    def open(self, carla_module) -> None:
        """Connect and switch CARLA to synchronous mode at ``fixed_delta_seconds``."""
        step_us = int(self.fixed_delta_seconds * 1e6)
        if step_us <= 0:
            raise ValueError(
                "fixed_delta_seconds must be positive, got "
                f"{self.fixed_delta_seconds!r}"
            )
        logger.info(
            "session %s: connecting to CARLA at %s:%d (fixed_delta=%.6fs)",
            self.session_uuid,
            self.carla_host,
            self.carla_port,
            self.fixed_delta_seconds,
        )
        self.client = carla_module.Client(self.carla_host, self.carla_port)
        self.client.set_timeout(30.0)
        self.world = self.client.get_world()
        settings = self.world.get_settings()
        # Remember the original settings so close() can restore them even if
        # apply_settings changes more than one field.
        self._prev_settings = settings
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(settings)

    def advance_to(self, target_time_us: int) -> None:
        """Tick CARLA to ``target_time_us`` in units of ``fixed_delta_seconds``.

        The first call seeds the clock without ticking — alpasim log
        timestamps and CARLA's world clock use different epochs, so we can't
        assume the target equals ``step_us * n``. Every subsequent call must
        land on a step boundary from that origin or the two rates drift.
        """
        step_us = int(self.fixed_delta_seconds * 1e6)
        if self.last_time_query_us is None:
            self.last_time_query_us = target_time_us
            return
        delta_us = target_time_us - self.last_time_query_us
        if delta_us <= 0:
            return
        if delta_us % step_us != 0:
            raise ValueError(
                f"target_time_us={target_time_us} is not aligned to "
                f"fixed_delta={step_us}us "
                f"(last_time_query_us={self.last_time_query_us})"
            )
        steps = delta_us // step_us
        for _ in range(steps):
            self.world.tick()
        self.last_time_query_us += steps * step_us

    def close(self) -> None:
        """Restore async mode; safe to call multiple times."""
        if self.world is None:
            return
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        except Exception:  # noqa: BLE001
            logger.exception("failed to restore async world settings")
        self.world = None
        self.client = None
