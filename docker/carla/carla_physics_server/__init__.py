# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""CARLA-backed physics gRPC server.

Lives in ``docker/carla/`` (not ``src/``) because it imports the CARLA Python
API. The alpasim service side treats CARLA integration as a container concern
— everything in ``src/`` runs whether CARLA is available or not.
"""

__version__ = "0.1.0"
