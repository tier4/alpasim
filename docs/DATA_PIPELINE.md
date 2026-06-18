# Alpasim data

This document is meant to describe data handling in Alpasim to help test and data engineers build
the test cases they want and troubleshoot issues.

## HD maps

Alpasim consumes vector maps packed inside the scene `.usdz`. 3dgs_io's
[`save_scene_usdz`](https://github.com/autowarefoundation/3dgs_io/blob/feat/usdz-io/src/3dgs_io/scene_usdz.py)
produces a single-file USDZ archive that bundles the 3D-GS payload together
with non-gaussian `extras` (Lanelet2, OpenDRIVE, tracks, rigs).
`Artifact.map` probes the following sources, in order:

1. **ClipGT parquet bundle** inside the USDZ -- `clipgt/map_data/` (or
   `map_data/`) directory. The canonical Alpasim format.
2. **Autoware Lanelet2** inside the USDZ -- `map.osm`, with the global
   origin recovered from a sibling `map.xodr` via its OpenDRIVE
   `<header geoReference>` PROJ4 string.
3. **OpenDRIVE** inside the USDZ -- `map.xodr` (used standalone when
   Lanelet2 isn't present), with `rig_trajectories.json` supplying the
   OpenDRIVE-ENU → simulation transform.

### Bringing in an Autoware Lanelet2 map

Pack the Lanelet2 source into the USDZ as a splatsim `extras` entry. The
origin is taken from `tileset.json`'s `root.transform` (ECEF anchor), so a
sibling OpenDRIVE map is not strictly required:

```
<scene>.usdz
├── default.usda
├── scene.json          # extras.map_lanelet2 → "map.osm"
├── tileset.json        # root.transform supplies the ECEF anchor
├── chunks/...spz
└── map.osm             # Autoware Lanelet2 vector map
```

Conversion is fully delegated to
[`3dgs_io.lanelet2_to_clipgt`](https://github.com/autowarefoundation/3dgs_io/blob/feat/usdz-io/src/3dgs_io/converters.py)
(itself a thin wrapper over
[`autoware_lanelet2_to_clipgt`](https://github.com/hakuturu583/autoware_lanelet2_to_clipgt)
invoked through `uvx`). It auto-derives the UTM projection origin
(`map.mgrs_grid` + `map.offset.{x,y,z}`) from `tileset.json`'s
`root.transform`, so the resulting ClipGT parquet bundle aligns with the
3D-GS scene origin. Install `uv` (https://docs.astral.sh/uv/) once and
conversion is hands-off thereafter.

Alpasim only adds a small trajdata-compat post-process on top of the
converter output (`alpasim_utils.lanelet2_postprocess.finalize_clipgt_bundle`):
- synthesises `association.parquet` (lane adjacency from rail-endpoint
  geometry) and `clip.parquet` — neither is emitted by the upstream
  converter today;
- clears `wait_line.parquet`, whose `key.map_id` format is currently
  incompatible with trajdata's `"{wait_line_id}-{lane_id}"` convention.

These workarounds will retire when their respective pieces land upstream
in `3dgs_io.converters`.

## asl files

The output of simulation in alpasim are `asl` files (it stands for AlpaSim Log). These are a
size-delimited protobuf stream with a custom schema defined
[here](/src/grpc/alpasim_grpc/v0/logging.proto). Each rollout will create its own `asl` file with
three types of messages:

- A metadata header (see `RolloutMetadata`) aiming to help with reproducibility and book keeping
- Actor poses (see `ActorPoses`) messages which inform about the location of all actors (including
  `'EGO'`) in global coordinate space
- Microservice requests and responses (see `*_request`/`*_return` messages) which enable reproducing
  behavior of a given service in replay mode without starting up the entire simulator

> :green_book: `RolloutCameraImage` requests allow for assembling an `.mp4` video out of an `.asl`
> log.

> :warning: The simulation header doesn't specify the `usdz` file uuid.

### Reading asl logs

`alpasim-grpc` provides [async_read_pb_log](/src/utils/alpasim_utils/logs.py) for reading `asl`
logs as a stream of messages. An example usage to print the first 20 messages in a log (since
`async_read_pb_log` is an async function it needs to be executed from a jupyter notebook or
submitted to an async runtime loop):

```python
from alpasim_grpc.utils.logs import async_read_pb_log

i = 0
async for log_entry in async_read_pb_log("<path_to_log>.asl"):
    print(log_entry)
    i += 1
    if i == 20:
        break
```

results in

```
rollout_metadata {
  session_metadata {
    session_uuid: "a5823758-a782-11ef-aa43-0242c0a89003"
    scene_id: "clipgt-3055a5c9-53e8-4e20-b41a-19c0f917b081"
    batch_size: 1
    n_sim_steps: 120
    start_timestamp_us: 1689697803493732
    control_timestep_us: 99000
  }
  actor_definitions {
  }
  force_gt_duration: 1700000
  version_ids {
    runtime_version {
      version_id: "0.3.0"
      git_hash: "83bf78502c43dabac683d68b3712cdca17f6a810+dirty"
      grpc_api_version {
        minor: 24
      }
    }
    egodriver_version {
      version_id: "0.0.0"
      git_hash: "mock"
      grpc_api_version {
        minor: 23
...
    image_bytes: "\377\330\377\340\000\020JFIF\000\001..."
  }
}

Output is truncated.
```
