# Alpasim data

This document is meant to describe data handling in Alpasim to help test and data engineers build
the test cases they want and troubleshoot issues.

## HD maps

Alpasim consumes vector maps either packed inside the scene `.usdz` or as
sidecars in the splatsim scene bundle directory that surrounds the USDZ
(see [autowarefoundation/3dgs_io](https://github.com/autowarefoundation/3dgs_io)).
`Artifact.map` probes the following sources, in order:

1. **ClipGT parquet bundle** inside the USDZ -- `clipgt/map_data/` (or
   `map_data/`) directory. The canonical Alpasim format.
2. **OpenDRIVE** inside the USDZ -- `map.xodr` plus `rig_trajectories.json`
   for the OpenDRIVE-ENU → simulation transform.
3. **Autoware Lanelet2** in the splatsim scene bundle directory -- `map.osm`
   next to the USDZ, with the origin recovered from a sibling `map.xodr`.

### Bringing in an Autoware Lanelet2 map

Place the splatsim scene bundle sidecars next to the USDZ:

```
<bundle_dir>/
  <scene>.usdz       # 3D-GS (Gaussian cloud)
  map.osm            # Autoware Lanelet2 vector map
  map.xodr           # OpenDRIVE map -- supplies the geo-anchor (PROJ4)
```

The Lanelet2 file itself only stores metre-scale local coordinates, so we
read the geographic anchor from `map.xodr`'s `<header geoReference>` PROJ4
string and project the local `(0, 0)` back to WGS84 lat/lon. Both maps are
authored against the same local frame, so that lat/lon is the origin
Lanelet2 wants.

Lanelet2 → ClipGT conversion is delegated to the external
[`autoware_lanelet2_to_clipgt`](https://github.com/hakuturu583/autoware_lanelet2_to_clipgt)
library. Because that library requires Python 3.10 while Alpasim targets
Python 3.11+, conversion runs through `uvx`, which provisions an isolated
interpreter. Install `uv` (https://docs.astral.sh/uv/) once and conversion
is hands-off thereafter.

Two extra entry points are provided when you want to do the conversion
ahead of time or outside the artifact pipeline:

* `lanelet2-to-clipgt` CLI (from `alpasim-tools`) -- produce a ClipGT parquet
  bundle from an `.osm` and an explicit origin.

  ```bash
  uv run lanelet2-to-clipgt \
      --osm path/to/map.osm \
      --output-dir out/clipgt \
      --mgrs-grid 54SUE \
      --offset-x 81655.73 --offset-y 50137.43 --offset-z 42.5
  ```

* `alpasim_utils.lanelet2_to_clipgt.load_vector_map_from_lanelet2_osm()` --
  convert and load into a `trajdata` `VectorMap` in one call.

The Lanelet2 → ClipGT path also synthesises `association.parquet` (lane
adjacency derived from rail-endpoint geometry) and a minimal `clip.parquet`,
since the upstream converter does not emit them. `wait_line.parquet` is
cleared on the way through because the upstream `key.map_id` is incompatible
with `trajdata`'s `"{wait_line_id}-{lane_id}"` convention.

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
