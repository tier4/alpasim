"""LiDAR device mapping and panorama-to-point-cloud helpers for splatsim.

alpasim's ``sensorsim.proto`` exposes two device types (``PANDAR128``,
``AT128``). splatsim v0.2.0 ships two named sensor models (``OT128``, ``XT32``).
We only map the physically-corresponding one (Hesai Pandar OT128) and leave
AT128 unsupported until splatsim adds a matching model.

The panorama returned by :class:`splatsim.LidarRenderer` does not include
ring indices, but the row index of a panorama pixel *is* the ring index.
:func:`lidar_panorama_to_point_cloud` reconstructs that mapping so alpasim's
``LidarRenderReturn.point_ring_ids`` can carry the channel each point came from.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from alpasim_grpc.v0 import sensorsim_pb2
from splatsim.lidar_renderer import LidarRenderer, LidarSensorSpec

if TYPE_CHECKING:
    from .scene_loader import SceneHandle

logger = logging.getLogger(__name__)

# Map alpasim LidarDeviceType → splatsim LidarSensorSpec ``sensor_type``.
# Hesai Pandar OT128 is a 128-beam spinning LiDAR; alpasim's ``PANDAR128``
# refers to the same family. AT128 is a solid-state design without a splatsim
# equivalent as of v0.2.0.
_LIDAR_DEVICE_TO_SENSOR_TYPE: dict[int, str] = {
    sensorsim_pb2.LidarDeviceType.PANDAR128: "OT128",
}

# Public re-export so tests can import a single mapping table.
LIDAR_DEVICE_SPECS = _LIDAR_DEVICE_TO_SENSOR_TYPE


def _build_lidar_spec(device_type: int) -> LidarSensorSpec:
    sensor_type = _LIDAR_DEVICE_TO_SENSOR_TYPE.get(device_type)
    if sensor_type is None:
        name = sensorsim_pb2.LidarDeviceType.Name(device_type)
        raise NotImplementedError(
            f"splatsim v0.2.0 has no LiDAR model for {name}; "
            f"supported: {sorted(_LIDAR_DEVICE_TO_SENSOR_TYPE)}"
        )
    # Identity sensor-to-base — alpasim passes the sensor pose directly, so
    # we treat sensor frame == base frame and let the splatsim renderer bake
    # the spinning model itself.
    return LidarSensorSpec(
        name=sensor_type,
        sensor_type=sensor_type,
        s2b=np.eye(4, dtype=np.float64),
    )


def build_lidar_renderer(device_type: int, device: str) -> LidarRenderer:
    spec = _build_lidar_spec(device_type)
    return LidarRenderer(spec, device=device)


def lidar_panorama_to_point_cloud(
    renderer: LidarRenderer,
    panorama: dict[str, torch.Tensor],
    *,
    drop_threshold: float = 0.5,
    alpha_threshold: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (xyz (N, 3) float32, intensity (N,) float32, ring_ids (N,) uint16).

    Sensor-frame coordinates (+x forward, +y left, +z up), matching splatsim's
    built-in :meth:`LidarRenderer.panorama_to_point_cloud`. Ring index = panorama
    row = elevation channel.
    """
    distance = panorama["distance"]
    intensity = panorama["intensity"]
    alpha = panorama["alpha"]
    raydrop = torch.sigmoid(panorama["raydrop_logit"])

    valid = (
        (alpha > alpha_threshold)
        & (raydrop < drop_threshold)
        & (distance > renderer.min_range_m)
        & (distance < renderer.max_range_m)
    )

    n_rows = renderer.n_rows
    n_cols = renderer.n_columns
    # Reuse the elevation/azimuth tables the renderer already precomputed to
    # avoid re-materialising them per call.
    el_grid = renderer._elevs[:, None].expand(-1, n_cols)  # (H, W)
    az_grid = renderer._azimuths[None, :].expand(n_rows, -1)  # (H, W)

    cos_el = torch.cos(el_grid)
    x = distance * cos_el * torch.cos(az_grid)
    y = distance * cos_el * torch.sin(az_grid)
    z = distance * torch.sin(el_grid)

    xyz_stacked = torch.stack([x, y, z], dim=-1)  # (H, W, 3)
    xyz = xyz_stacked[valid].detach().cpu().numpy().astype(np.float32)
    intensity_valid = intensity[valid].detach().cpu().numpy().astype(np.float32)

    # Row index of every valid pixel = ring id.
    row_idx = (
        torch.arange(n_rows, device=valid.device).unsqueeze(1).expand(n_rows, n_cols)
    )
    ring_ids = row_idx[valid].detach().cpu().numpy().astype(np.uint16)

    return xyz, intensity_valid, ring_ids


def render_lidar_panorama_from_scene(
    scene_handle: "SceneHandle",
    base_to_world_np: np.ndarray,
    device_type: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience wrapper: build a renderer, invoke it, and unpack points.

    ``base_to_world_np`` must already be in the scene's tile-local frame
    (i.e. the caller applied ``SceneHandle.tile_local_centroid``).
    """
    renderer = build_lidar_renderer(device_type, device=scene_handle.device)
    base_to_world = torch.from_numpy(base_to_world_np).to(scene_handle.device)
    panorama = renderer.render(base_to_world, scene=scene_handle.scene)
    return lidar_panorama_to_point_cloud(renderer, panorama)
