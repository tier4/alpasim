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
    CameraIntrinsics,
    camera_spec_to_intrinsics,
    encode_image,
    pose_pair_to_viewmat,
    pose_to_sensor_to_world,
)
from .scene_loader import SceneHandle

logger = logging.getLogger(__name__)


def _build_available_cameras_from_usdz(
    usdz_path: Optional[Path],
) -> list["sensorsim_pb2.AvailableCamerasReturn.AvailableCamera"]:
    """Read camera_calibrations from the USDZ rig_trajectories.json.

    Returns AvailableCamera entries with camera-in-rig extrinsics (inverted
    from the USDZ's rig→sensor ``T_sensor_rig`` field, see below) and pinhole
    intrinsics from the USDZ. Non-fatal on any parse failure — an empty list
    falls back to the original "no catalog" behavior.
    """
    if usdz_path is None:
        return []

    import json
    import zipfile
    import numpy as np

    try:
        with zipfile.ZipFile(str(usdz_path)) as zf:
            with zf.open("rig_trajectories.json") as f:
                doc = json.load(f)
    except (KeyError, zipfile.BadZipFile, json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read camera_calibrations from %s: %s", usdz_path, exc)
        return []

    calibrations = doc.get("camera_calibrations") or {}
    if not isinstance(calibrations, dict):
        return []

    cameras: list = []
    for logical_id, cam in calibrations.items():
        try:
            model = cam["camera_model"]
            if model.get("type") != "pinhole":
                logger.warning(
                    "Skipping camera %s: unsupported camera_model type %r",
                    logical_id,
                    model.get("type"),
                )
                continue
            params = model["parameters"]
            width, height = params["resolution"]
            fx, fy, cx, cy = params["fx"], params["fy"], params["cx"], params["cy"]

            # ``T_sensor_rig`` in NuRec/3dgs_io exports is the rig→sensor
            # matrix (OpenCV +Z forward), NOT sensor-in-rig — verified
            # empirically both against splatsim's own ``iter_world_to_camera``
            # and in splatsim/_usdz.py's implementation. Invert it here so
            # that alpasim's downstream ``ego.compose(rig_to_camera)`` sees
            # a proper camera-in-rig transform.
            T_raw = np.asarray(cam["T_sensor_rig"], dtype=np.float64)
            R = T_raw[:3, :3].T
            t = -R @ T_raw[:3, 3]
            # Matrix -> quaternion (xyzw). Uses Shepperd's method to avoid
            # scipy dependency here.
            trace = R[0, 0] + R[1, 1] + R[2, 2]
            if trace > 0.0:
                s = np.sqrt(trace + 1.0) * 2.0
                qw = 0.25 * s
                qx = (R[2, 1] - R[1, 2]) / s
                qy = (R[0, 2] - R[2, 0]) / s
                qz = (R[1, 0] - R[0, 1]) / s
            elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s

            entry = sensorsim_pb2.AvailableCamerasReturn.AvailableCamera(
                logical_id=logical_id,
                intrinsics=sensorsim_pb2.CameraSpec(
                    logical_id=logical_id,
                    resolution_w=int(width),
                    resolution_h=int(height),
                    opencv_pinhole_param=sensorsim_pb2.OpenCVPinholeCameraParam(
                        focal_length_x=float(fx),
                        focal_length_y=float(fy),
                        principal_point_x=float(cx),
                        principal_point_y=float(cy),
                    ),
                ),
                rig_to_camera=common_pb2.Pose(
                    vec=common_pb2.Vec3(x=float(t[0]), y=float(t[1]), z=float(t[2])),
                    quat=common_pb2.Quat(
                        w=float(qw),
                        x=float(qx),
                        y=float(qy),
                        z=float(qz),
                    ),
                ),
            )
            cameras.append(entry)
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Could not parse camera %s: %s", logical_id, exc)
            continue

    logger.info(
        "Loaded %d camera(s) from USDZ rig_trajectories.json: %s",
        len(cameras),
        [c.logical_id for c in cameras],
    )
    return cameras


class SplatsimSensorsimServicer(sensorsim_pb2_grpc.SensorsimServiceServicer):
    """Splatsim-backed implementation of SensorsimService."""

    def __init__(
        self,
        scene: SceneHandle,
        scene_id: str,
        usdz_path: Optional[Path] = None,
    ) -> None:
        self._scene = scene
        # One container == one scene for now. We accept any scene_id but log
        # mismatches so misconfigured Runtime requests are visible.
        self._scene_id = scene_id
        self._available_cameras = _build_available_cameras_from_usdz(usdz_path)

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
        # The K matrix on the wire is calibrated for `intrinsics.width x .height`
        # (the sensor's native resolution, e.g. 2880x1860 for CAM_FRONT_WIDE),
        # but this Renderer's canvas is fixed at `self._scene.default_resolution`
        # (e.g. 960x540). Feeding the unscaled K to gsplat produces principal
        # points outside the small canvas, so every Gaussian projects off-screen
        # and the image comes back all zeros. Scale K to the canvas.
        canvas_w, canvas_h = self._scene.default_resolution
        scale_x = canvas_w / float(intrinsics.width)
        scale_y = canvas_h / float(intrinsics.height)
        intrinsics = CameraIntrinsics(
            width=canvas_w,
            height=canvas_h,
            fx=intrinsics.fx * scale_x,
            fy=intrinsics.fy * scale_y,
            cx=intrinsics.cx * scale_x,
            cy=intrinsics.cy * scale_y,
        )
        # DEBUG: log incoming pose + scene stats to diagnose black-frame issue.
        import numpy as _np
        _sp = request.sensor_pose.start_pose
        _sv = _np.array([_sp.vec.x, _sp.vec.y, _sp.vec.z], dtype=_np.float64)
        _sq = _np.array([_sp.quat.x, _sp.quat.y, _sp.quat.z, _sp.quat.w], dtype=_np.float64)
        _tlc = _np.asarray(self._scene.tile_local_centroid, dtype=_np.float64)
        try:
            _means = self._scene._bg.means.detach().cpu().numpy()
            _bbox_min = _means.min(axis=0)
            _bbox_max = _means.max(axis=0)
        except Exception:
            _bbox_min = _bbox_max = None
        logger.info(
            "RENDER_DBG pose_t=%s pose_q_xyzw=%s tile_local_centroid=%s bbox=[%s..%s] "
            "K_scaled=fx=%.2f fy=%.2f cx=%.2f cy=%.2f canvas=%dx%d",
            _sv.tolist(), _sq.tolist(), _tlc.tolist(),
            None if _bbox_min is None else _bbox_min.tolist(),
            None if _bbox_max is None else _bbox_max.tolist(),
            intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
            canvas_w, canvas_h,
        )
        viewmat = pose_pair_to_viewmat(
            request.sensor_pose,
            world_origin=self._scene.tile_local_centroid,
        )
        logger.info("RENDER_DBG viewmat=\n%s", _np.asarray(viewmat).tolist())
        rgb = self._scene.render(viewmat, intrinsics.k_matrix())
        try:
            _rgb_np = _np.asarray(rgb)
            logger.info(
                "RENDER_DBG rgb shape=%s dtype=%s min=%s max=%s mean=%s",
                _rgb_np.shape, str(_rgb_np.dtype),
                float(_rgb_np.min()), float(_rgb_np.max()), float(_rgb_np.mean()),
            )
        except Exception as _e:
            logger.info("RENDER_DBG rgb stats failed: %s", _e)
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
        # Runtime's CameraCatalog requires local overrides to be a subset of
        # what sensorsim reports. Advertise the USDZ rig's cameras so the
        # local extra_cameras merge check passes.
        return sensorsim_pb2.AvailableCamerasReturn(
            available_cameras=list(self._available_cameras)
        )

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
    servicer = SplatsimSensorsimServicer(
        scene=scene, scene_id=args.scene_id, usdz_path=usdz_path
    )

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
