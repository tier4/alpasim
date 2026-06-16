"""Build a splatsim ``Scene`` from a Cesium 3D Tiles directory.

Layout we expect on the bind-mounted host directory:

    /mnt/cesium-tiles/
      ├── tileset.json            # Cesium 3D Tiles root
      ├── ...                     # tile payload (.b3dm / .pnts / .glb)
      └── scene.yaml              # optional: bypass the default config

If ``scene.yaml`` is present it's passed straight to ``Scene.from_config`` so
the user can override the renderer block (resolution, device, SH on/off,
extra rigid_bodies). Otherwise we synthesise a minimal config that points
splatsim at ``tileset.json`` with no rigid bodies — the static Cesium
background only.

Dynamic objects from the gRPC requests are intentionally ignored (NOP).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


def _synthesise_scene_yaml(tiles_dir: Path, resolution: tuple[int, int]) -> Path:
    """Write a temp YAML pointing splatsim at the directory's tileset.json."""
    tileset = tiles_dir / "tileset.json"
    if not tileset.exists():
        raise FileNotFoundError(
            f"Cesium tileset not found at {tileset}; pass a directory containing "
            "tileset.json or provide a scene.yaml override."
        )

    cfg = {
        "background_tileset": str(tileset),
        "use_sh": False,
        "rigid_bodies": [],
        "renderer": {
            "width": int(resolution[0]),
            "height": int(resolution[1]),
            "device": "cuda",
        },
    }
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="splatsim_scene_", delete=False
    )
    yaml.safe_dump(cfg, tmp)
    tmp.close()
    logger.info("synthesised splatsim scene config at %s for %s", tmp.name, tileset)
    return Path(tmp.name)


def resolve_scene_yaml(tiles_dir: Path, resolution: tuple[int, int]) -> Path:
    """Find or synthesise the YAML splatsim's Scene.from_config will consume."""
    override = tiles_dir / "scene.yaml"
    if override.exists():
        logger.info("using scene.yaml override at %s", override)
        return override
    return _synthesise_scene_yaml(tiles_dir, resolution)


class SceneHandle:
    """Lazy splatsim scene loader.

    splatsim + torch + CUDA are heavy to import, so we defer the actual
    construction until the first render call. This keeps unit tests free of
    the torch dependency and means we can boot the gRPC server without a GPU
    if the host doesn't have one yet (handy for smoke testing).
    """

    def __init__(self, tiles_dir: Path, default_resolution: tuple[int, int]) -> None:
        self._tiles_dir = tiles_dir
        self._default_resolution = default_resolution
        self._scene: Any = None
        self._renderer: Any = None
        self._device: Any = None

    def _ensure_loaded(self) -> None:
        if self._scene is not None:
            return
        # Heavy imports kept local so server.py can import this module on a
        # box with no CUDA / no torch (unit tests).
        import torch  # type: ignore
        from splatsim.renderer import Renderer  # type: ignore
        from splatsim.scene import Scene  # type: ignore

        yaml_path = resolve_scene_yaml(self._tiles_dir, self._default_resolution)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._scene = Scene.from_config(str(yaml_path), device=self._device)
        self._renderer = Renderer(
            width=self._default_resolution[0],
            height=self._default_resolution[1],
            device=self._device,
        )
        logger.info("splatsim scene loaded (device=%s)", self._device)

    def render(self, viewmat_np, k_np):
        """Render a single frame and return an ndarray (H, W, 3) float32 [0, 1]."""
        self._ensure_loaded()
        import torch  # type: ignore

        viewmat = torch.from_numpy(viewmat_np).to(self._device)
        K = torch.from_numpy(k_np).to(self._device)
        rgb = self._renderer.render(viewmat, K, scene=self._scene)
        return rgb.detach().to("cpu").numpy()

    @property
    def device(self) -> Optional[Any]:
        return self._device
