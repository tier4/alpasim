# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""gRPC entry point for the splatsim-backed renderer.

Implements the alpasim ``SensorsimService`` contract so Runtime can swap NuRec
for splatsim without changing the client. RGB rendering is delegated to
splatsim's :class:`splatsim.Renderer`; LiDAR is delegated to splatsim's
:class:`splatsim.LidarRenderer` (both introduced in v0.2.0). ``dynamic_objects``
in the request is currently ignored — only the static background is rendered.
"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent import futures
from pathlib import Path
from typing import Optional

from alpasim_grpc.v0 import common_pb2, sensorsim_pb2, sensorsim_pb2_grpc

import grpc

from . import __version__ as renderer_version
from .lidar_adapter import render_lidar_panorama_from_scene
from .render_adapter import (
    camera_spec_to_intrinsics,
    encode_image,
    pose_pair_to_viewmat,
    pose_to_sensor_to_world,
)
from .scene_loader import SceneHandle

logger = logging.getLogger(__name__)


class SplatsimSensorsimServicer(sensorsim_pb2_grpc.SensorsimServiceServicer):
    """Splatsim-backed implementation of SensorsimService."""

    def __init__(self, scene: SceneHandle, scene_id: str) -> None:
        self._scene = scene
        # One container == one scene for now. We accept any scene_id but log
        # mismatches so misconfigured Runtime requests are visible.
        self._scene_id = scene_id

    # ----- rendering (internal, raise on error) -----

    def _do_render_rgb(
        self, request: sensorsim_pb2.RGBRenderRequest
    ) -> sensorsim_pb2.RGBRenderReturn:
        if request.scene_id and request.scene_id != self._scene_id:
            logger.warning(
                "render_rgb scene_id mismatch (requested=%s, loaded=%s); rendering loaded scene",
                request.scene_id,
                self._scene_id,
            )
        intrinsics = camera_spec_to_intrinsics(request.camera_intrinsics)
        viewmat = pose_pair_to_viewmat(
            request.sensor_pose,
            world_origin=self._scene.tile_local_centroid,
        )
        rgb = self._scene.render(viewmat, intrinsics.k_matrix())
        image_bytes = encode_image(rgb, request.image_format, request.image_quality)
        return sensorsim_pb2.RGBRenderReturn(image_bytes=image_bytes)

    def _do_render_lidar(
        self, request: sensorsim_pb2.LidarRenderRequest
    ) -> sensorsim_pb2.LidarRenderReturn:
        if request.scene_id and request.scene_id != self._scene_id:
            logger.warning(
                "render_lidar scene_id mismatch (requested=%s, loaded=%s); rendering loaded scene",
                request.scene_id,
                self._scene_id,
            )
        device_type = request.lidar_config.lidar_type
        base_to_world = pose_to_sensor_to_world(
            request.sensor_pose.start_pose,
            world_origin=self._scene.tile_local_centroid,
        )
        xyz, intensity, ring_ids = render_lidar_panorama_from_scene(
            self._scene, base_to_world, device_type
        )
        # ring_ids on the wire are packed little-endian uint16.
        ring_ids_u16 = ring_ids.astype("<u2", copy=False)
        return sensorsim_pb2.LidarRenderReturn(
            num_points=int(xyz.shape[0]),
            point_xyzs_buffer=xyz.tobytes(order="C"),
            point_intensities_buffer=intensity.tobytes(order="C"),
            point_ring_ids_buffer=ring_ids_u16.tobytes(order="C"),
        )

    # ----- rendering (public RPC entry points) -----

    def render_rgb(self, request: sensorsim_pb2.RGBRenderRequest, context):
        try:
            return self._do_render_rgb(request)
        except NotImplementedError as exc:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, str(exc))
            return sensorsim_pb2.RGBRenderReturn()  # unreachable

    def render_lidar(self, request: sensorsim_pb2.LidarRenderRequest, context):
        try:
            return self._do_render_lidar(request)
        except NotImplementedError as exc:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, str(exc))
            return sensorsim_pb2.LidarRenderReturn()  # unreachable

    def render_aggregated(
        self, request: sensorsim_pb2.AggregatedRenderRequest, context
    ):
        # Isolate per-item failures so one bad sub-request doesn't cancel the
        # rest of the aggregate. Uses the internal helpers directly to avoid
        # ``context.abort()`` from a sub-call terminating this outer RPC.
        rgb_returns: list[sensorsim_pb2.RGBRenderReturn] = []
        for req in request.rgb_requests:
            try:
                rgb_returns.append(self._do_render_rgb(req))
            except Exception:  # noqa: BLE001
                logger.exception("render_aggregated: rgb sub-request failed")
                rgb_returns.append(sensorsim_pb2.RGBRenderReturn())
        lidar_returns: list[sensorsim_pb2.LidarRenderReturn] = []
        for req in request.lidar_requests:
            try:
                lidar_returns.append(self._do_render_lidar(req))
            except Exception:  # noqa: BLE001
                logger.exception("render_aggregated: lidar sub-request failed")
                lidar_returns.append(sensorsim_pb2.LidarRenderReturn())
        return sensorsim_pb2.AggregatedRenderReturn(
            rgb_returns=rgb_returns,
            lidar_returns=lidar_returns,
        )

    def batch_render_rgb(self, request: sensorsim_pb2.BatchRGBRenderRequest, context):
        items = []
        for item in request.items:
            try:
                result = self._do_render_rgb(item.request)
                items.append(
                    sensorsim_pb2.BatchRGBRenderReturnItem(
                        camera_name=item.camera_name,
                        result=result,
                        success=True,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("batch render failed for %s", item.camera_name)
                items.append(
                    sensorsim_pb2.BatchRGBRenderReturnItem(
                        camera_name=item.camera_name,
                        success=False,
                        error_message=str(exc),
                    )
                )
        return sensorsim_pb2.BatchRGBRenderReturn(items=items)

    # ----- metadata -----

    def get_version(self, request: common_pb2.Empty, context):
        version = common_pb2.VersionId(
            version_id=renderer_version,
            git_hash=os.environ.get("ALPASIM_GIT_HASH", ""),
        )
        return version

    def get_available_scenes(self, request: common_pb2.Empty, context):
        return common_pb2.AvailableScenesReturn(scene_ids=[self._scene_id])

    def get_available_cameras(
        self, request: sensorsim_pb2.AvailableCamerasRequest, context
    ):
        # No catalog in NOP mode — Runtime is expected to supply intrinsics in
        # render requests. Returning an empty list is the documented "unknown"
        # answer.
        return sensorsim_pb2.AvailableCamerasReturn()

    def get_available_trajectories(
        self, request: sensorsim_pb2.AvailableTrajectoriesRequest, context
    ):
        return sensorsim_pb2.AvailableTrajectoriesReturn()

    def get_available_ego_masks(self, request: common_pb2.Empty, context):
        return sensorsim_pb2.AvailableEgoMasksReturn()


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Alpasim splatsim renderer gRPC server (SensorsimService backend)"
    )
    parser.add_argument("--host", default="0.0.0.0", help="bind address")
    parser.add_argument("--port", type=int, required=True, help="bind port")
    parser.add_argument(
        "--scene-usdz",
        required=True,
        help="path to a .usdz file (3D Gaussian tileset with EXT_3dgs_spz chunks)",
    )
    parser.add_argument(
        "--scene-id",
        default="splatsim-default",
        help="scene_id reported via get_available_scenes and accepted by render_rgb",
    )
    parser.add_argument("--width", type=int, default=960, help="default render width")
    parser.add_argument("--height", type=int, default=540, help="default render height")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    usdz_path = Path(args.scene_usdz)
    if not usdz_path.is_file() or usdz_path.suffix.lower() != ".usdz":
        raise SystemExit(f"--scene-usdz must point at a .usdz file: {usdz_path}")

    scene = SceneHandle(
        usdz_path=usdz_path, default_resolution=(args.width, args.height)
    )
    servicer = SplatsimSensorsimServicer(scene=scene, scene_id=args.scene_id)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.max_workers))
    sensorsim_pb2_grpc.add_SensorsimServiceServicer_to_server(servicer, server)
    bind_addr = f"{args.host}:{args.port}"
    server.add_insecure_port(bind_addr)
    server.start()
    logger.info(
        "splatsim_renderer_server listening on %s (scene=%s)", bind_addr, usdz_path
    )
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("received SIGINT, shutting down")
        server.stop(grace=5.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
