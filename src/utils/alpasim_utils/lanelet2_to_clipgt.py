# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Convert Autoware Lanelet2 ``.osm`` maps into ClipGT parquet bundles usable
inside Alpasim.

Map conversion is delegated to the external ``autoware_lanelet2_to_clipgt``
library (https://github.com/hakuturu583/autoware_lanelet2_to_clipgt). Because
that library targets Python ``<3.11`` while Alpasim runs on Python ``>=3.11``,
the CLI is invoked through ``uvx``, which provisions an isolated environment
with the correct interpreter.

The library writes ``lane.parquet``, ``lane_line.parquet``,
``road_boundary.parquet``, ``wait_line.parquet``, ``crosswalk.parquet`` and a
handful of others, but it does **not** emit ``association.parquet`` or
``clip.parquet`` -- both of which ``trajdata`` requires when loading a ClipGT
map into a :class:`VectorMap`. We synthesise those two files here:

* ``clip.parquet``: single-row metadata pinned to the ``clip_id`` used during
  conversion.
* ``association.parquet``: lane adjacency relations recovered from rail
  endpoint geometry. This is an approximation of Lanelet2's native routing
  graph, sufficient for typical planning queries on Autoware-style maps.

The ``wait_line`` rows emitted by the upstream library use a ``key.map_id``
that does not match trajdata's ``"{wait_line_id}-{lane_id}"`` convention, so
we clear ``wait_line.parquet`` to avoid downstream ``IndexError`` during
``populate_vector_map``. Wait-line ingestion can be added once the upstream
library learns about lanelet-to-stopline regulatory relations.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from trajdata.maps import VectorMap

logger = logging.getLogger(__name__)

DEFAULT_LIB_GIT_URL = "git+https://github.com/hakuturu583/autoware_lanelet2_to_clipgt"
DEFAULT_LIB_REF = "master"
DEFAULT_PYTHON = "3.10"

LABEL_CLASS_ID = "lanelet2:autoware:v0"
SCHEMA_VERSION = 1

# Endpoint-matching tolerance (metres). Autoware vector maps are typically
# authored in metre-precise UTM, so 0.5 m comfortably absorbs digitisation
# noise without merging distinct lanelets.
_ENDPOINT_TOL_M = 0.5


@dataclass(frozen=True)
class LatLonOrigin:
    """Map origin specified directly as latitude/longitude in degrees."""

    latitude: float
    longitude: float
    altitude: float = 0.0


@dataclass(frozen=True)
class MgrsOrigin:
    """Map origin specified as an MGRS grid square with optional metre offset.

    Either ``grid`` alone (e.g. ``"54SUE815501"``) or ``grid`` plus
    ``offset_x`` / ``offset_y`` may be supplied -- mirroring the convention
    used by ``autoware_lanelet2_to_opendrive``.
    """

    grid: str
    offset_x: float | None = None
    offset_y: float | None = None
    offset_z: float = 0.0


Origin = Union[LatLonOrigin, MgrsOrigin]


def origin_from_xodr(xodr_source: str | Path) -> LatLonOrigin:
    """Recover the Lanelet2 ``LatLonOrigin`` from an OpenDRIVE map.

    The 3dgs_io ``splatsim`` scene bundle ships an Autoware Lanelet2
    ``map.osm`` alongside a CARLA-style ``map.xodr``. The Lanelet2 file alone
    carries no global anchor (just metre-scale local coordinates), so we read
    OpenDRIVE's ``<header geoReference="...">`` PROJ4 string and project the
    local ``(0, 0)`` back to WGS84 lat/lon. Both maps are authored against
    the same local frame, so that lat/lon is the origin Lanelet2 wants.

    ``xodr_source`` may be a path or the raw XML string.
    """
    import pyproj

    proj4 = _read_xodr_geo_reference(xodr_source)
    transformer = pyproj.Transformer.from_crs(proj4, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(0.0, 0.0)
    return LatLonOrigin(latitude=float(lat), longitude=float(lon))


def _read_xodr_geo_reference(xodr_source: str | Path) -> str:
    """Extract the PROJ4 string from an OpenDRIVE ``<header geoReference>``."""
    if isinstance(xodr_source, Path) or (
        isinstance(xodr_source, str) and Path(xodr_source).is_file()
    ):
        xml = Path(xodr_source).read_text()
    else:
        xml = str(xodr_source)
    root = ET.fromstring(xml)
    header = root.find("header")
    if header is None:
        raise ValueError("OpenDRIVE has no <header> element")
    # Prefer the `<geoReference>` child element; OpenDRIVE 1.x allows both an
    # attribute on `<header>` and a child element wrapping a CDATA PROJ4 string.
    elem = header.find("geoReference")
    if elem is not None and elem.text and elem.text.strip():
        return elem.text.strip()
    attr = header.attrib.get("geoReference")
    if attr:
        return attr.strip()
    raise ValueError("OpenDRIVE header is missing geoReference (PROJ4 string)")


def convert_osm_to_clipgt_dir(
    osm_path: str | Path,
    out_dir: str | Path,
    origin: Origin,
    *,
    clip_id: str | None = None,
    library_git_url: str = DEFAULT_LIB_GIT_URL,
    library_ref: str = DEFAULT_LIB_REF,
    python_version: str = DEFAULT_PYTHON,
) -> Path:
    """Convert an Autoware Lanelet2 ``.osm`` file to a ClipGT parquet bundle.

    The bundle layout matches what :func:`trajdata.dataset_specific.mads.\
mads_utils.populate_vector_map` expects, after the ``association.parquet`` /
    ``clip.parquet`` augmentation done here.
    """
    osm_path = Path(osm_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    if not osm_path.is_file():
        raise FileNotFoundError(f"Lanelet2 .osm file not found: {osm_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    _run_library_cli(
        osm_path=osm_path,
        out_dir=out_dir,
        origin=origin,
        clip_id=clip_id,
        library_git_url=library_git_url,
        library_ref=library_ref,
        python_version=python_version,
    )

    resolved_clip_id = _resolve_clip_id(out_dir, clip_id)
    _build_clip_parquet(out_dir, clip_id=resolved_clip_id)
    _build_association_parquet(out_dir, clip_id=resolved_clip_id)
    _clear_wait_line_parquet(out_dir)
    return out_dir


def load_vector_map_from_lanelet2_osm(
    osm_path: str | Path,
    origin: Origin,
    *,
    map_id: str | None = None,
    clip_id: str | None = None,
) -> "VectorMap":
    """Convert ``osm_path`` and return a fully-populated trajdata ``VectorMap``.

    The intermediate ClipGT bundle is written to a temporary directory that is
    cleaned up before this function returns.
    """
    from trajdata.dataset_specific.mads.mads_utils import populate_vector_map
    from trajdata.maps import VectorMap

    osm_path = Path(osm_path).expanduser().resolve()
    resolved_map_id = map_id or f"lanelet2:{osm_path.stem}"
    vector_map = VectorMap(map_id=resolved_map_id)

    with tempfile.TemporaryDirectory() as tmp:
        clipgt_dir = convert_osm_to_clipgt_dir(
            osm_path, Path(tmp) / "clipgt", origin, clip_id=clip_id
        )
        populate_vector_map(vector_map, str(clipgt_dir))

    vector_map.__post_init__()
    vector_map.compute_search_indices()
    # trajdata exposes adjacency lists; Alpasim expects sets.
    for lane in vector_map.lanes:
        lane.next_lanes = set(lane.next_lanes)
        lane.prev_lanes = set(lane.prev_lanes)
        lane.adj_lanes_right = set(lane.adj_lanes_right)
        lane.adj_lanes_left = set(lane.adj_lanes_left)
    return vector_map


# --- Library CLI invocation ---------------------------------------------------


def _run_library_cli(
    *,
    osm_path: Path,
    out_dir: Path,
    origin: Origin,
    clip_id: str | None,
    library_git_url: str,
    library_ref: str,
    python_version: str,
) -> None:
    """Invoke ``autoware_lanelet2_to_clipgt`` through ``uvx`` and check it succeeded."""
    if shutil.which("uvx") is None:
        raise RuntimeError(
            "`uvx` not found on PATH. `autoware_lanelet2_to_clipgt` requires "
            "Python 3.10, which we provision through uvx. Install uv from "
            "https://docs.astral.sh/uv/."
        )

    cmd = [
        "uvx",
        "--from",
        f"{library_git_url}@{library_ref}",
        "--python",
        python_version,
        "python",
        "-m",
        "autoware_lanelet2_to_clipgt",
        "map=example",
        f"input_map_path={osm_path}",
        f"output_dir={out_dir}",
        *_origin_overrides(origin),
    ]
    if clip_id is not None:
        cmd.append(f"clip_id={clip_id}")

    logger.info("Invoking lanelet2-to-clipgt: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _origin_overrides(origin: Origin) -> list[str]:
    """Render Hydra overrides that fully replace the ``map`` config block."""
    if isinstance(origin, LatLonOrigin):
        # `+map.*=` would conflict with the bundled `example` schema; use `++`
        # to force-override.
        return [
            f"++map.lat_lon.latitude={origin.latitude!r}",
            f"++map.lat_lon.longitude={origin.longitude!r}",
            f"++map.lat_lon.altitude={origin.altitude!r}",
            "~map.mgrs_grid",
        ]
    if isinstance(origin, MgrsOrigin):
        overrides = [f"++map.mgrs_grid={origin.grid}"]
        if origin.offset_x is not None and origin.offset_y is not None:
            overrides.extend(
                [
                    f"++map.offset.x={origin.offset_x!r}",
                    f"++map.offset.y={origin.offset_y!r}",
                    f"++map.offset.z={origin.offset_z!r}",
                ]
            )
        return overrides
    raise TypeError(f"unsupported origin type: {type(origin).__name__}")


# --- ClipGT bundle augmentation ----------------------------------------------


def _resolve_clip_id(out_dir: Path, requested: str | None) -> str:
    """Return the clip id stamped into the library output, or the requested override."""
    if requested is not None:
        return requested
    lane_table = pq.read_table(out_dir / "lane.parquet")
    if lane_table.num_rows == 0:
        return "lanelet2"
    return str(lane_table.column("key")[0]["clip_id"].as_py())


def _build_clip_parquet(out_dir: Path, *, clip_id: str) -> None:
    """Write a minimal ``clip.parquet`` so trajdata can pick the clip metadata up."""
    schema = pa.schema(
        [
            (
                "key",
                pa.struct(
                    [
                        ("clip_id", pa.string()),
                        ("label_class_id", pa.string()),
                    ]
                ),
            ),
            (
                "clip",
                pa.struct(
                    [
                        ("label_class_id", pa.string()),
                    ]
                ),
            ),
            ("version", pa.uint64()),
        ]
    )
    table = pa.table(
        {
            "key": [{"clip_id": clip_id, "label_class_id": LABEL_CLASS_ID}],
            "clip": [{"label_class_id": LABEL_CLASS_ID}],
            "version": [SCHEMA_VERSION],
        },
        schema=schema,
    )
    pq.write_table(table, out_dir / "clip.parquet")


def _build_association_parquet(out_dir: Path, *, clip_id: str) -> None:
    """Synthesise lane adjacency relations from rail-endpoint geometry.

    The upstream library does not preserve Lanelet2's routing-graph relations,
    so we recover them heuristically:

    * ``NEXT_LANE``    - end of A's left+right rail meets start of B's left+right rail.
    * ``PREVIOUS_LANE``- inverse of ``NEXT_LANE``.
    * ``LEFT_LANE``    - A's ``left_rail`` matches B's ``right_rail`` polyline.
    * ``RIGHT_LANE``   - A's ``right_rail`` matches B's ``left_rail`` polyline.

    ``SIGN_TO_LANE`` is left empty -- the converter does not emit lane <-> sign
    regulatory links yet.
    """
    schema = _association_schema()
    lane_path = out_dir / "lane.parquet"
    lane_table = pq.read_table(lane_path)
    if lane_table.num_rows == 0:
        pq.write_table(_empty_table(schema), out_dir / "association.parquet")
        return

    lanes: list[tuple[str, np.ndarray, np.ndarray]] = []
    for i in range(lane_table.num_rows):
        key = lane_table.column("key")[i].as_py()
        lane = lane_table.column("lane")[i].as_py()
        lanes.append(
            (
                str(key["map_id"]),
                _rail_to_array(lane["left_rail"]),
                _rail_to_array(lane["right_rail"]),
            )
        )

    rows: list[dict] = []
    seq = 0
    for a_id, a_left, a_right in lanes:
        next_ids: list[str] = []
        for b_id, b_left, b_right in lanes:
            if b_id == a_id or len(b_left) == 0 or len(b_right) == 0:
                continue
            if (
                np.linalg.norm(a_left[-1] - b_left[0]) < _ENDPOINT_TOL_M
                and np.linalg.norm(a_right[-1] - b_right[0]) < _ENDPOINT_TOL_M
            ):
                next_ids.append(b_id)
        if next_ids:
            rows.append(_assoc_row(clip_id, seq, "NEXT_LANE", a_id, next_ids))
            seq += 1
            for nxt in next_ids:
                rows.append(_assoc_row(clip_id, seq, "PREVIOUS_LANE", nxt, [a_id]))
                seq += 1

    for a_id, a_left, a_right in lanes:
        for b_id, _b_left, b_right in lanes:
            if b_id == a_id:
                continue
            if _polylines_match(a_left, b_right):
                rows.append(_assoc_row(clip_id, seq, "LEFT_LANE", a_id, [b_id]))
                seq += 1
        for b_id, b_left, _b_right in lanes:
            if b_id == a_id:
                continue
            if _polylines_match(a_right, b_left):
                rows.append(_assoc_row(clip_id, seq, "RIGHT_LANE", a_id, [b_id]))
                seq += 1

    if rows:
        table = pa.Table.from_pylist(rows, schema=schema)
    else:
        table = _empty_table(schema)
    pq.write_table(table, out_dir / "association.parquet")


def _clear_wait_line_parquet(out_dir: Path) -> None:
    """Empty ``wait_line.parquet``.

    The upstream library writes ``key.map_id = <linestring_id>``, but trajdata
    expects ``"{wait_line_id}-{lane_id}"`` -- mismatched values crash the
    loader with ``IndexError``. Until per-lane stop-line attribution is
    implemented in the converter, we drop wait-line rows entirely so map
    loading remains robust.
    """
    path = out_dir / "wait_line.parquet"
    if not path.is_file():
        return
    schema = pq.read_table(path).schema
    pq.write_table(_empty_table(schema), path)


def _association_schema() -> pa.Schema:
    return pa.schema(
        [
            (
                "key",
                pa.struct(
                    [
                        ("clip_id", pa.string()),
                        ("label_class_id", pa.string()),
                        ("map_id", pa.string()),
                        ("kind", pa.string()),
                    ]
                ),
            ),
            (
                "Association",
                pa.struct(
                    [
                        ("subjects", pa.string()),
                        ("objects", pa.list_(pa.string())),
                    ]
                ),
            ),
            ("version", pa.uint64()),
        ]
    )


def _empty_table(schema: pa.Schema) -> pa.Table:
    return pa.table({field.name: [] for field in schema}, schema=schema)


def _assoc_row(
    clip_id: str, seq: int, kind: str, subject: str, objects: list[str]
) -> dict:
    return {
        "key": {
            "clip_id": clip_id,
            "label_class_id": LABEL_CLASS_ID,
            "map_id": f"{kind}-{seq}",
            "kind": kind,
        },
        "Association": {"subjects": subject, "objects": objects},
        "version": SCHEMA_VERSION,
    }


def _rail_to_array(rail: list[dict]) -> np.ndarray:
    if not rail:
        return np.zeros((0, 3), dtype=float)
    return np.asarray([[p["x"], p["y"], p["z"]] for p in rail], dtype=float)


def _polylines_match(
    a: np.ndarray, b: np.ndarray, *, tol: float = _ENDPOINT_TOL_M
) -> bool:
    """Test whether two polylines coincide -- same or opposite traversal."""
    if len(a) < 2 or len(b) < 2:
        return False
    same = np.linalg.norm(a[0] - b[0]) < tol and np.linalg.norm(a[-1] - b[-1]) < tol
    reversed_ = (
        np.linalg.norm(a[0] - b[-1]) < tol and np.linalg.norm(a[-1] - b[0]) < tol
    )
    return same or reversed_
