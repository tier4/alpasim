"""Unit tests for the LiDAR device mapping helper.

Full renderer tests are covered end-to-end at the gRPC layer; here we only
verify the alpasim → splatsim device-type mapping so an accidental proto
addition (say a third LiDAR type) surfaces as a clear NotImplementedError
rather than a silent NOP.
"""

from __future__ import annotations

import pytest
from alpasim_grpc.v0 import sensorsim_pb2
from alpasim_splatsim_renderer.lidar_adapter import (
    LIDAR_DEVICE_SPECS,
    _build_lidar_spec,
)


def test_pandar128_maps_to_ot128():
    assert LIDAR_DEVICE_SPECS[sensorsim_pb2.LidarDeviceType.PANDAR128] == "OT128"


def test_at128_is_unsupported():
    with pytest.raises(NotImplementedError, match="AT128"):
        _build_lidar_spec(sensorsim_pb2.LidarDeviceType.AT128)
