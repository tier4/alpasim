# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""trajdata-compat post-processing for ``autoware_lanelet2_to_clipgt`` output.

The upstream converter (invoked through :mod:`alpasim_utils._dgs_io`) does
not emit ``association.parquet`` or ``clip.parquet``, and the
``wait_line.parquet`` it does emit uses a ``key.map_id`` format that
trajdata's ``populate_vector_map`` cannot parse (it expects
``"{wait_line_id}-{lane_id}"`` and slices on ``-``).

Until those gaps land upstream in ``3dgs_io.converters`` /
``autoware_lanelet2_to_clipgt`` proper, we run this thin post-process over
the converter output:

* synthesise ``clip.parquet`` (single-row metadata),
* recover lane adjacency heuristically from rail-endpoint geometry into
  ``association.parquet``,
* clear ``wait_line.parquet`` so the slicing crash is avoided.

The heuristic is approximate — it cannot reproduce Lanelet2's routing
graph exactly — but is sufficient for typical Autoware HD maps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

LABEL_CLASS_ID = "lanelet2:autoware:v0"
SCHEMA_VERSION = 1

# Endpoint-matching tolerance (metres). Autoware vector maps are typically
# authored in metre-precise UTM, so 0.5 m comfortably absorbs digitisation
# noise without merging distinct lanelets.
_ENDPOINT_TOL_M = 0.5


def finalize_clipgt_bundle(out_dir: str | Path, *, clip_id: str | None = None) -> str:
    """Augment a converter-produced ClipGT directory for trajdata consumption.

    Adds ``clip.parquet`` + ``association.parquet`` and empties
    ``wait_line.parquet`` so :func:`trajdata.dataset_specific.mads.\
mads_utils.populate_vector_map` can ingest the bundle.

    Returns the resolved ``clip_id`` actually stamped into the synthesised
    metadata (auto-derived from the converter's lane keys when not given).
    """
    out_dir = Path(out_dir)
    resolved = _resolve_clip_id(out_dir, clip_id)
    _build_clip_parquet(out_dir, clip_id=resolved)
    _build_association_parquet(out_dir, clip_id=resolved)
    _clear_wait_line_parquet(out_dir)
    return resolved


# ---------------------------------------------------------------------------


def _resolve_clip_id(out_dir: Path, requested: str | None) -> str:
    if requested is not None:
        return requested
    lane_table = pq.read_table(out_dir / "lane.parquet")
    if lane_table.num_rows == 0:
        return "lanelet2"
    return lane_table.column("key")[0]["clip_id"].as_py()


def _build_clip_parquet(out_dir: Path, *, clip_id: str) -> None:
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
    """Heuristically recover NEXT/PREV/LEFT/RIGHT lane relations.

    The upstream converter writes rail polylines but no routing edges, so we
    approximate them by endpoint geometry:

    * ``NEXT_LANE`` — end of A's left+right rail meets start of B's left+right rail.
    * ``PREVIOUS_LANE`` — inverse of ``NEXT_LANE``.
    * ``LEFT_LANE`` — A's ``left_rail`` matches B's ``right_rail`` polyline.
    * ``RIGHT_LANE`` — A's ``right_rail`` matches B's ``left_rail`` polyline.

    ``SIGN_TO_LANE`` stays empty.
    """
    schema = _association_schema()
    lane_path = out_dir / "lane.parquet"
    lane_table = pq.read_table(lane_path)

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

    if not rows:
        # trajdata's df_expand_json only materialises ``key.clip_id`` etc.
        # when the struct columns have at least one row to normalise. Emit a
        # sentinel row whose ``kind`` matches none of the kinds trajdata
        # filters on so it contributes no relations but keeps the parquet
        # readable.
        rows.append(_assoc_row(clip_id, 0, "NONE", "__sentinel__", []))
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema), out_dir / "association.parquet"
    )


def _clear_wait_line_parquet(out_dir: Path) -> None:
    """Empty ``wait_line.parquet`` to avoid trajdata's ``IndexError``."""
    path = out_dir / "wait_line.parquet"
    if not path.is_file():
        return
    schema = pq.read_table(path).schema
    pq.write_table(pa.table({field.name: [] for field in schema}, schema=schema), path)


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
    if len(a) < 2 or len(b) < 2:
        return False
    same = np.linalg.norm(a[0] - b[0]) < tol and np.linalg.norm(a[-1] - b[-1]) < tol
    reversed_ = (
        np.linalg.norm(a[0] - b[-1]) < tol and np.linalg.norm(a[-1] - b[0]) < tol
    )
    return same or reversed_
