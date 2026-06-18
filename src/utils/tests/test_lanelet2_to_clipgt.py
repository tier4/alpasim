# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Tests for the Lanelet2-compat layer.

The OSM → ClipGT conversion itself is owned by ``3dgs_io.lanelet2_to_clipgt``
and exercised in that project's test suite. Here we only cover the thin
post-process that augments the converter output for trajdata consumption
(``alpasim_utils.lanelet2_postprocess``) plus the importlib shim.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from alpasim_utils import _dgs_io
from alpasim_utils import lanelet2_postprocess as lpp


def _write_synthetic_lane_parquet(out_dir: Path, lanes: list[dict]) -> None:
    rows = []
    for lane in lanes:
        rows.append(
            {
                "key": {
                    "clip_id": lane["clip_id"],
                    "label_class_id": "lanelet2:autoware:v0",
                    "map_id": lane["map_id"],
                    "map_id_version": "1",
                },
                "lane": {
                    "left_rail": [
                        {"x": float(x), "y": float(y), "z": 0.0}
                        for x, y in lane["left_rail"]
                    ],
                    "right_rail": [
                        {"x": float(x), "y": float(y), "z": 0.0}
                        for x, y in lane["right_rail"]
                    ],
                },
                "version": 1,
            }
        )
    point = pa.struct([("x", pa.float64()), ("y", pa.float64()), ("z", pa.float64())])
    schema = pa.schema(
        [
            (
                "key",
                pa.struct(
                    [
                        ("clip_id", pa.string()),
                        ("label_class_id", pa.string()),
                        ("map_id", pa.string()),
                        ("map_id_version", pa.string()),
                    ]
                ),
            ),
            (
                "lane",
                pa.struct(
                    [
                        ("left_rail", pa.list_(point)),
                        ("right_rail", pa.list_(point)),
                    ]
                ),
            ),
            ("version", pa.uint64()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out_dir / "lane.parquet")


# ---------------------------------------------------------------------------
# Shim
# ---------------------------------------------------------------------------


def test_dgs_io_shim_reexports_required_symbols():
    """Cover the public surface Alpasim actually imports."""
    assert callable(_dgs_io.lanelet2_to_clipgt)
    assert callable(_dgs_io.mgrs_overrides_from_root_transform)
    assert callable(_dgs_io.parse_alpasim_rig_trajectories)
    assert callable(_dgs_io.parse_alpasim_sequence_tracks)
    assert callable(_dgs_io.save_scene_usdz)
    assert _dgs_io.DEFAULT_LANELET2_CONVERTER_PACKAGE.startswith("git+")


# ---------------------------------------------------------------------------
# Post-process
# ---------------------------------------------------------------------------


def test_finalize_resolves_clip_id_from_lane_keys(tmp_path: Path):
    _write_synthetic_lane_parquet(
        tmp_path,
        [
            {
                "clip_id": "clip-from-lane",
                "map_id": "1",
                "left_rail": [(0, 1), (1, 1)],
                "right_rail": [(0, 0), (1, 0)],
            }
        ],
    )
    resolved = lpp.finalize_clipgt_bundle(tmp_path)
    assert resolved == "clip-from-lane"
    df = pq.read_table(tmp_path / "clip.parquet").to_pandas()
    assert df.iloc[0]["key"]["clip_id"] == "clip-from-lane"


def test_finalize_overrides_clip_id_when_supplied(tmp_path: Path):
    _write_synthetic_lane_parquet(
        tmp_path,
        [
            {
                "clip_id": "ignored",
                "map_id": "1",
                "left_rail": [(0, 1), (1, 1)],
                "right_rail": [(0, 0), (1, 0)],
            }
        ],
    )
    resolved = lpp.finalize_clipgt_bundle(tmp_path, clip_id="explicit")
    assert resolved == "explicit"
    df = pq.read_table(tmp_path / "clip.parquet").to_pandas()
    assert df.iloc[0]["key"]["clip_id"] == "explicit"


def test_association_recovers_next_prev_and_left(tmp_path: Path):
    """Collinear A→B yields NEXT/PREV; parallel C to A's left yields LEFT."""
    _write_synthetic_lane_parquet(
        tmp_path,
        [
            {
                "clip_id": "clip-A",
                "map_id": "A",
                "left_rail": [(0, 1), (10, 1)],
                "right_rail": [(0, 0), (10, 0)],
            },
            {
                "clip_id": "clip-A",
                "map_id": "B",
                "left_rail": [(10, 1), (20, 1)],
                "right_rail": [(10, 0), (20, 0)],
            },
            {
                "clip_id": "clip-A",
                "map_id": "C",
                "left_rail": [(0, 2), (10, 2)],
                "right_rail": [(0, 1), (10, 1)],  # matches A.left → A.LEFT == C
            },
        ],
    )
    lpp.finalize_clipgt_bundle(tmp_path)
    df = pq.read_table(tmp_path / "association.parquet").to_pandas()

    by_kind = df["key"].map(lambda k: k["kind"])
    next_rows = df[by_kind == "NEXT_LANE"]
    assert len(next_rows) == 1
    assert next_rows.iloc[0]["Association"]["subjects"] == "A"
    assert list(next_rows.iloc[0]["Association"]["objects"]) == ["B"]

    prev_rows = df[by_kind == "PREVIOUS_LANE"]
    assert len(prev_rows) == 1
    assert prev_rows.iloc[0]["Association"]["subjects"] == "B"
    assert list(prev_rows.iloc[0]["Association"]["objects"]) == ["A"]

    left_pairs = {
        (row["Association"]["subjects"], tuple(row["Association"]["objects"]))
        for _, row in df[by_kind == "LEFT_LANE"].iterrows()
    }
    assert ("A", ("C",)) in left_pairs


def test_association_emits_sentinel_when_no_relations(tmp_path: Path):
    """Empty/disconnected bundles must still produce a parquet whose
    ``key.clip_id`` column survives trajdata's ``df_expand_json``; we do
    that by emitting a sentinel row with an inert ``kind``.
    """
    _write_synthetic_lane_parquet(tmp_path, [])
    lpp.finalize_clipgt_bundle(tmp_path, clip_id="clip-empty")
    df = pq.read_table(tmp_path / "association.parquet").to_pandas()
    assert len(df) == 1
    sentinel = df.iloc[0]
    assert sentinel["key"]["kind"] == "NONE"
    assert sentinel["key"]["kind"] not in {
        "NEXT_LANE",
        "PREVIOUS_LANE",
        "LEFT_LANE",
        "RIGHT_LANE",
        "SIGN_TO_LANE",
    }


def test_finalize_clears_wait_line(tmp_path: Path):
    """``wait_line.parquet`` from the upstream converter uses a key.map_id
    format trajdata can't parse; we drop the rows but keep the schema.
    """
    _write_synthetic_lane_parquet(
        tmp_path,
        [
            {
                "clip_id": "c",
                "map_id": "1",
                "left_rail": [(0, 1), (1, 1)],
                "right_rail": [(0, 0), (1, 0)],
            }
        ],
    )
    point = pa.struct([("x", pa.float64()), ("y", pa.float64()), ("z", pa.float64())])
    schema = pa.schema(
        [
            ("key", pa.struct([("clip_id", pa.string()), ("map_id", pa.string())])),
            (
                "wait_line",
                pa.struct([("category", pa.string()), ("location", pa.list_(point))]),
            ),
            ("version", pa.uint64()),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "key": {"clip_id": "c", "map_id": "wl-1"},
                    "wait_line": {
                        "category": "STOP",
                        "location": [{"x": 0.0, "y": 0.0, "z": 0.0}],
                    },
                    "version": 1,
                }
            ],
            schema=schema,
        ),
        tmp_path / "wait_line.parquet",
    )
    lpp.finalize_clipgt_bundle(tmp_path)
    out = pq.read_table(tmp_path / "wait_line.parquet")
    assert out.num_rows == 0
    assert out.schema == schema


def test_finalize_is_idempotent_when_wait_line_missing(tmp_path: Path):
    _write_synthetic_lane_parquet(
        tmp_path,
        [
            {
                "clip_id": "c",
                "map_id": "1",
                "left_rail": [(0, 1), (1, 1)],
                "right_rail": [(0, 0), (1, 0)],
            }
        ],
    )
    lpp.finalize_clipgt_bundle(tmp_path)  # must not raise
    assert not (tmp_path / "wait_line.parquet").exists()
