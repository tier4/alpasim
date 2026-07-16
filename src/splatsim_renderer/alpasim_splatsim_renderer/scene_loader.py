# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""USDZ scene loader for the splatsim renderer.

Loads a splatsim scene from a single ``.usdz`` file (3D Gaussian tileset with
``EXT_3dgs_spz`` chunks) via ``SceneConfig.from_source`` from splatsim v0.2.0,
and exposes the tile-local / ECEF metadata that the RPC layer needs to
transform world-frame sensor poses into the tile-local frame the renderer
operates in.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from splatsim import Renderer, Scene
from splatsim.dataclass.scene_config import SceneConfig

logger = logging.getLogger(__name__)


class SceneHandle:
    """Owns the splatsim Scene + Renderer for a single loaded USDZ.

    One container hosts one scene: the USDZ path is fixed for the lifetime of
    the process. Poses in world (map) frame are translated into the
    tile-local frame using ``tile_local_centroid`` before rendering.
    """

    def __init__(
        self,
        usdz_path: Path,
        default_resolution: tuple[int, int] = (960, 540),
        device: str = "cuda",
    ) -> None:
        if not usdz_path.is_file() or usdz_path.suffix.lower() != ".usdz":
            raise FileNotFoundError(
                f"Splatsim scene must point at a .usdz file: {usdz_path}"
            )

        self._device = device
        self._default_resolution = (
            int(default_resolution[0]),
            int(default_resolution[1]),
        )

        cfg = SceneConfig.from_source(usdz_path)
        # Override render resolution / device before instantiating the
        # Renderer so the wizard config is authoritative.
        cfg.renderer.width = self._default_resolution[0]
        cfg.renderer.height = self._default_resolution[1]
        cfg.renderer.device = device

        logger.info("loading splatsim scene from %s", usdz_path)
        self._scene = Scene.from_config(cfg, device=torch.device(device))
        self._config = cfg

        # Cache ECEF metadata for the RPC layer. ``tile_local_centroid`` is a
        # torch tensor on ``device``; the RPC layer expects host-side numpy so
        # it can compose translations with pose data coming off the wire.
        bg = self._scene.background
        if bg is None:
            raise RuntimeError(
                f"Loaded scene has no Background; USDZ may be missing a tileset: {usdz_path}"
            )
        self._tile_local_centroid = (
            bg.tile_local_centroid.detach().cpu().numpy().astype(np.float32)
        )
        self._ecef_translation = np.asarray(bg.ecef_translation, dtype=np.float64)
        self._ecef_rotation = np.asarray(bg.ecef_rotation, dtype=np.float64)

        self._renderer = Renderer(
            width=self._default_resolution[0],
            height=self._default_resolution[1],
            device=device,
            background_color=tuple(cfg.renderer.background_color),
            near_plane=cfg.renderer.near_plane,
            far_plane=cfg.renderer.far_plane,
            radius_clip=cfg.renderer.radius_clip,
        )

    # ----- properties -----

    @property
    def scene(self) -> Scene:
        return self._scene

    @property
    def config(self) -> SceneConfig:
        return self._config

    @property
    def default_resolution(self) -> tuple[int, int]:
        return self._default_resolution

    @property
    def device(self) -> str:
        return self._device

    @property
    def tile_local_centroid(self) -> np.ndarray:
        """(3,) float32 offset to subtract from world-frame positions."""
        return self._tile_local_centroid

    @property
    def ecef_translation(self) -> np.ndarray:
        """(3,) float64 ECEF translation of the tile root (metadata only)."""
        return self._ecef_translation

    @property
    def ecef_rotation(self) -> np.ndarray:
        """(3, 3) float64 ECEF rotation of the tile root (metadata only)."""
        return self._ecef_rotation

    # ----- rendering -----

    def render(self, viewmat_np: np.ndarray, k_np: np.ndarray) -> np.ndarray:
        """Render RGB frame.

        ``viewmat_np`` must already be in the tile-local frame (i.e. the RPC
        layer has subtracted ``tile_local_centroid`` from the world-frame
        position before inverting to world-to-camera).

        Returns (H, W, 3) float32 in [0, 1].
        """
        viewmat = torch.from_numpy(viewmat_np).to(self._device)
        k = torch.from_numpy(k_np).to(self._device)
        rgb = self._renderer.render(viewmat, k, scene=self._scene)
        return rgb.detach().to("cpu").numpy()
