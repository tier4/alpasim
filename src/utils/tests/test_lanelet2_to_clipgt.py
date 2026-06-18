# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Tests for ``alpasim_utils.lanelet2_to_clipgt``.

The library invocation itself is exercised via ``subprocess.run`` monkeypatch:
spinning up a real ``uvx`` job in CI would be flaky and slow. Instead we
verify (a) the CLI command we build, and (b) the parquet augmentation that
runs over the converter output.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from alpasim_utils import lanelet2_to_clipgt as l2c


def _write_synthetic_lane_parquet(out_dir: Path, lanes: list[dict]) -> None:
    """Write a minimal ``lane.parquet`` mirroring the upstream library schema."""
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
                        (
                            "left_rail",
                            pa.list_(
                                pa.struct(
                                    [
                                        ("x", pa.float64()),
                                        ("y", pa.float64()),
                                        ("z", pa.float64()),
                                    ]
                                )
                            ),
                        ),
                        (
                            "right_rail",
                            pa.list_(
                                pa.struct(
                                    [
                                        ("x", pa.float64()),
                                        ("y", pa.float64()),
                                        ("z", pa.float64()),
                                    ]
                                )
                            ),
                        ),
                    ]
                ),
            ),
            ("version", pa.uint64()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out_dir / "lane.parquet")


def test_origin_overrides_latlon():
    overrides = l2c._origin_overrides(
        l2c.LatLonOrigin(latitude=35.6895, longitude=139.6917, altitude=42.5)
    )
    joined = " ".join(overrides)
    assert "++map.lat_lon.latitude=35.6895" in joined
    assert "++map.lat_lon.longitude=139.6917" in joined
    assert "++map.lat_lon.altitude=42.5" in joined


def test_origin_overrides_mgrs_with_offset():
    overrides = l2c._origin_overrides(
        l2c.MgrsOrigin(grid="54SUE", offset_x=1.5, offset_y=2.5, offset_z=3.5)
    )
    assert "++map.mgrs_grid=54SUE" in overrides
    assert "++map.offset.x=1.5" in overrides
    assert "++map.offset.y=2.5" in overrides
    assert "++map.offset.z=3.5" in overrides


def test_origin_overrides_mgrs_no_offset():
    overrides = l2c._origin_overrides(l2c.MgrsOrigin(grid="54SUE815501"))
    assert overrides == ["++map.mgrs_grid=54SUE815501"]


def test_origin_overrides_rejects_unknown_type():
    with pytest.raises(TypeError):
        l2c._origin_overrides("not-an-origin")  # type: ignore[arg-type]


def test_build_clip_parquet(tmp_path: Path):
    _write_synthetic_lane_parquet(
        tmp_path,
        [
            {
                "clip_id": "clip-A",
                "map_id": "1",
                "left_rail": [(0, 0), (1, 0)],
                "right_rail": [(0, -1), (1, -1)],
            }
        ],
    )
    l2c._build_clip_parquet(tmp_path, clip_id="clip-A")
    df = pq.read_table(tmp_path / "clip.parquet").to_pandas()
    assert len(df) == 1
    assert df.iloc[0]["key"]["clip_id"] == "clip-A"
    assert df.iloc[0]["key"]["label_class_id"] == l2c.LABEL_CLASS_ID


def test_build_association_parquet_recovers_next_and_left(tmp_path: Path):
    """Two collinear lanes (`A` then `B`) should yield NEXT_LANE/PREVIOUS_LANE,
    and a parallel lane `C` to the left of `A` should yield LEFT_LANE/RIGHT_LANE.
    """
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
                # C's right_rail matches A's left_rail -> A.LEFT_LANE == C.
                "right_rail": [(0, 1), (10, 1)],
            },
        ],
    )
    l2c._build_association_parquet(tmp_path, clip_id="clip-A")
    df = pq.read_table(tmp_path / "association.parquet").to_pandas()

    next_rows = df[df["key"].map(lambda k: k["kind"]) == "NEXT_LANE"]
    assert len(next_rows) == 1
    assert next_rows.iloc[0]["Association"]["subjects"] == "A"
    assert list(next_rows.iloc[0]["Association"]["objects"]) == ["B"]

    prev_rows = df[df["key"].map(lambda k: k["kind"]) == "PREVIOUS_LANE"]
    assert len(prev_rows) == 1
    assert prev_rows.iloc[0]["Association"]["subjects"] == "B"
    assert list(prev_rows.iloc[0]["Association"]["objects"]) == ["A"]

    left_rows = df[df["key"].map(lambda k: k["kind"]) == "LEFT_LANE"]
    left_pairs = {
        (row["Association"]["subjects"], tuple(row["Association"]["objects"]))
        for _, row in left_rows.iterrows()
    }
    assert ("A", ("C",)) in left_pairs


def test_build_association_parquet_emits_sentinel_when_no_relations(tmp_path: Path):
    """Even when the heuristic finds no relations (or there are no lanes),
    the parquet must carry at least one row -- otherwise trajdata's
    df_expand_json fails to materialise ``key.clip_id`` from empty struct
    columns and ``find_lane_polylines_parquet`` raises KeyError.
    The sentinel uses an unused ``kind`` so it filters out as a no-op.
    """
    _write_synthetic_lane_parquet(tmp_path, [])
    l2c._build_association_parquet(tmp_path, clip_id="clip-empty")
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


def test_clear_wait_line_parquet_emits_empty_table(tmp_path: Path):
    schema = pa.schema(
        [
            ("key", pa.struct([("clip_id", pa.string()), ("map_id", pa.string())])),
            (
                "wait_line",
                pa.struct(
                    [
                        ("category", pa.string()),
                    ]
                ),
            ),
            ("version", pa.uint64()),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "key": {"clip_id": "c", "map_id": "wl-1"},
                    "wait_line": {"category": "STOP"},
                    "version": 1,
                }
            ],
            schema=schema,
        ),
        tmp_path / "wait_line.parquet",
    )
    l2c._clear_wait_line_parquet(tmp_path)
    table = pq.read_table(tmp_path / "wait_line.parquet")
    assert table.num_rows == 0
    assert table.schema == schema


def test_clear_wait_line_parquet_is_a_noop_when_missing(tmp_path: Path):
    l2c._clear_wait_line_parquet(tmp_path)
    assert not (tmp_path / "wait_line.parquet").exists()


def test_convert_invokes_uvx_with_expected_command(tmp_path: Path, monkeypatch):
    osm = tmp_path / "lanelet2.osm"
    osm.write_text("<osm/>")  # contents irrelevant — we mock subprocess.

    captured: dict = {}

    def fake_which(name: str) -> str | None:
        return "/usr/bin/uvx" if name == "uvx" else None

    def fake_run(cmd, check):
        captured["cmd"] = cmd
        captured["check"] = check
        # Pretend the upstream library wrote a minimal lane.parquet.
        _write_synthetic_lane_parquet(
            tmp_path / "out",
            [
                {
                    "clip_id": "clip-A",
                    "map_id": "lane-1",
                    "left_rail": [(0, 1), (1, 1)],
                    "right_rail": [(0, 0), (1, 0)],
                }
            ],
        )

    monkeypatch.setattr(l2c.shutil, "which", fake_which)
    monkeypatch.setattr(l2c.subprocess, "run", fake_run)

    out_dir = l2c.convert_osm_to_clipgt_dir(
        osm,
        tmp_path / "out",
        l2c.MgrsOrigin(grid="54SUE", offset_x=1.0, offset_y=2.0),
        clip_id="explicit-clip",
    )
    assert out_dir == (tmp_path / "out").resolve()
    assert captured["check"] is True
    cmd = captured["cmd"]
    assert cmd[0] == "uvx"
    assert "autoware_lanelet2_to_clipgt" in cmd
    assert f"input_map_path={osm.resolve()}" in cmd
    assert "++map.mgrs_grid=54SUE" in cmd
    assert "clip_id=explicit-clip" in cmd

    # Augmentation outputs.
    assert (out_dir / "clip.parquet").is_file()
    assert (out_dir / "association.parquet").is_file()


def test_convert_raises_when_uvx_missing(tmp_path: Path, monkeypatch):
    osm = tmp_path / "lanelet2.osm"
    osm.write_text("<osm/>")
    monkeypatch.setattr(l2c.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="uvx"):
        l2c.convert_osm_to_clipgt_dir(
            osm,
            tmp_path / "out",
            l2c.MgrsOrigin(grid="54SUE"),
        )


def test_polylines_match_directional_and_reversed():
    import numpy as np

    a = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
    b_same = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
    b_rev = np.array([[10, 0, 0], [0, 0, 0]], dtype=float)
    b_off = np.array([[0, 5, 0], [10, 5, 0]], dtype=float)

    assert l2c._polylines_match(a, b_same)
    assert l2c._polylines_match(a, b_rev)
    assert not l2c._polylines_match(a, b_off)


def _xodr_with_proj4(proj4: str) -> str:
    return f"""<?xml version="1.0"?>
<OpenDRIVE>
  <header revMajor="1" revMinor="6" name="" version="1.00">
    <geoReference><![CDATA[{proj4}]]></geoReference>
  </header>
</OpenDRIVE>
"""


def test_origin_from_xodr_recovers_latlon():
    # UTM zone 54N anchored such that (0,0) corresponds to ~Tokyo bay.
    # `+lat_0`/`+lon_0` make the inverse projection deterministic for this test.
    proj4 = (
        "+proj=tmerc +lat_0=35.6895 +lon_0=139.6917 +k=1 +x_0=0 +y_0=0 "
        "+ellps=WGS84 +units=m +no_defs"
    )
    origin = l2c.origin_from_xodr(_xodr_with_proj4(proj4))
    assert isinstance(origin, l2c.LatLonOrigin)
    assert origin.latitude == pytest.approx(35.6895, abs=1e-6)
    assert origin.longitude == pytest.approx(139.6917, abs=1e-6)


def test_origin_from_xodr_accepts_attribute_form(tmp_path: Path):
    proj4 = (
        "+proj=tmerc +lat_0=10.0 +lon_0=20.0 +k=1 +x_0=0 +y_0=0 "
        "+ellps=WGS84 +units=m +no_defs"
    )
    xml = (
        '<?xml version="1.0"?>\n'
        "<OpenDRIVE>\n"
        f'  <header revMajor="1" revMinor="6" geoReference="{proj4}"/>\n'
        "</OpenDRIVE>\n"
    )
    xodr_path = tmp_path / "map.xodr"
    xodr_path.write_text(xml)
    origin = l2c.origin_from_xodr(xodr_path)
    assert origin.latitude == pytest.approx(10.0, abs=1e-6)
    assert origin.longitude == pytest.approx(20.0, abs=1e-6)


def test_origin_from_xodr_missing_georeference_raises():
    xml = '<?xml version="1.0"?><OpenDRIVE><header/></OpenDRIVE>'
    with pytest.raises(ValueError, match="geoReference"):
        l2c.origin_from_xodr(xml)


def test_origin_from_xodr_missing_header_raises():
    xml = '<?xml version="1.0"?><OpenDRIVE/>'
    with pytest.raises(ValueError, match="<header>"):
        l2c.origin_from_xodr(xml)
