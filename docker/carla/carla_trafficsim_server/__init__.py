# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""CARLA TrafficManager backed traffic gRPC server.

Lives in ``docker/carla/`` (not ``src/``) because it imports the CARLA Python
API and depends on ``autoware_carla_scenario``. The alpasim service side is
CARLA-free.
"""

__version__ = "0.1.0"
