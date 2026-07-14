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
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# CARLA is always co-located in the physics container in the alpasim topology
# (trafficsim reaches it as `physics-0:2000`, we reach it as localhost).
CARLA_HOST = "localhost"
CARLA_PORT = 2000


@dataclass
class CarlaClock:
    """Minimal CARLA client that owns synchronous-mode tick cadence."""

    session_uuid: str
    tick_interval_us: int

    client: Any = None
    world: Any = None
    last_time_query_us: Optional[int] = None

    def open(self, carla_module) -> None:
        """Connect and switch CARLA to synchronous mode at ``tick_interval_us``."""
        if self.tick_interval_us <= 0:
            raise ValueError(
                f"tick_interval_us must be positive, got {self.tick_interval_us!r}"
            )
        fixed_delta_seconds = self.tick_interval_us / 1e6
        logger.info(
            "session %s: connecting to CARLA at %s:%d (fixed_delta=%.6fs)",
            self.session_uuid,
            CARLA_HOST,
            CARLA_PORT,
            fixed_delta_seconds,
        )
        self.client = carla_module.Client(CARLA_HOST, CARLA_PORT)
        self.client.set_timeout(30.0)
        self.world = self.client.get_world()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = fixed_delta_seconds
        self.world.apply_settings(settings)

    def advance_to(self, target_time_us: int) -> None:
        """Tick CARLA to ``target_time_us`` in units of ``tick_interval_us``.

        The first call seeds the clock without ticking — alpasim log
        timestamps and CARLA's world clock use different epochs, so we can't
        assume the target equals ``tick_interval_us * n``. Every subsequent
        call must land on a step boundary from that origin or the two rates
        drift.
        """
        if self.world is None:
            raise RuntimeError(
                f"session {self.session_uuid}: CarlaClock.advance_to called on a "
                "closed clock; call open() first"
            )
        if self.last_time_query_us is None:
            self.last_time_query_us = target_time_us
            return
        delta_us = target_time_us - self.last_time_query_us
        if delta_us <= 0:
            return
        if delta_us % self.tick_interval_us != 0:
            raise ValueError(
                f"target_time_us={target_time_us} is not aligned to "
                f"tick_interval_us={self.tick_interval_us} "
                f"(last_time_query_us={self.last_time_query_us})"
            )
        steps = delta_us // self.tick_interval_us
        for _ in range(steps):
            self.world.tick()
        self.last_time_query_us += steps * self.tick_interval_us

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
