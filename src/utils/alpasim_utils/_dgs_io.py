# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Re-export shim for the ``3dgs_io`` package.

The upstream package is named ``3dgs-io`` (PyPI) / ``3dgs_io`` (module dir);
the leading digit makes ``import 3dgs_io`` a Python ``SyntaxError``. Importing
through :func:`importlib.import_module` is the only way to reach it from
regular Python source. We hide that ugliness here and re-export the symbols
Alpasim actually needs.
"""

from __future__ import annotations

import importlib

_mod = importlib.import_module("3dgs_io")

# USDZ I/O + scene bundle
save_scene_usdz = _mod.save_scene_usdz
SceneUsdzOptions = _mod.SceneUsdzOptions
SceneUsdzResult = _mod.SceneUsdzResult

# Lanelet2 -> ClipGT converter (uvx-driven)
lanelet2_to_clipgt = _mod.lanelet2_to_clipgt
mgrs_overrides_from_root_transform = _mod.mgrs_overrides_from_root_transform
DEFAULT_LANELET2_CONVERTER_PACKAGE = _mod.DEFAULT_LANELET2_CONVERTER_PACKAGE

# Alpasim-format sidecars (USDZ extras)
parse_alpasim_rig_trajectories = _mod.parse_alpasim_rig_trajectories
parse_alpasim_sequence_tracks = _mod.parse_alpasim_sequence_tracks
RigTrajectory = _mod.RigTrajectory
Track = _mod.Track

__all__ = [
    "DEFAULT_LANELET2_CONVERTER_PACKAGE",
    "RigTrajectory",
    "SceneUsdzOptions",
    "SceneUsdzResult",
    "Track",
    "lanelet2_to_clipgt",
    "mgrs_overrides_from_root_transform",
    "parse_alpasim_rig_trajectories",
    "parse_alpasim_sequence_tracks",
    "save_scene_usdz",
]
