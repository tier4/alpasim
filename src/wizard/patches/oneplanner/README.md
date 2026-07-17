# OnePlanner container patches

These files bind-mount over the corresponding paths inside the
`oneplanner:local` docker image at wizard runtime (see
`src/wizard/configs/driver/oneplanner.yaml`). Each still exists because the
shipped image predates the fix, so we override the file until the upstream
change lands and the image is rebuilt.

Each file mirrors the layout inside the container so the mapping is obvious:

| File in this dir | Overrides in container |
|---|---|
| `encoder.py` | `/opt/oneplanner/src/oneplanner/models/planner/encoder.py` |
| `model_loader.py` | `/opt/oneplanner/src/oneplanner/deployment/model_loader.py` |
| `driver_preprocessing.py` | `/opt/oneplanner/src/oneplanner/deployment/driver_preprocessing.py` |

## Purpose per patch

- **encoder.py** — HOTFIX: cast `has_speed_limit` to bool before `torch.where`
  (upstream stores it as float32 but the encoder expects a bool condition).
  Delete once upstream is fixed.
- **model_loader.py** — HOTFIX: upstream defaults `StateNormalizer` and
  `ObservationNormalizer` to identity/empty because the shipped checkpoint has
  no embedded `config` key. This loader instead reads
  `configs/planner/normalization.json` (ego mean/std) so predictions aren't
  scaled by identity. See memory `oneplanner_deployment_normalizer_missing`.
- **driver_preprocessing.py** — HOTFIX: upsample 20 route waypoints (from
  alpasim's RouteGenerator) to 500 (`NUM_SEGMENTS_IN_ROUTE=25 *
  POINTS_PER_LANELET=20`) via arc-length interpolation. Without this, only
  1/25 route_lanes segments carries signal. See memory
  `oneplanner_route_needs_500_waypoints`.

## Maintenance

When upstream OnePlanner ships a fix for any of these, delete the file here
and remove its bind-mount from `oneplanner.yaml`. Verify by extracting the
same path from the current image (`docker create --name op-extract
oneplanner:local sh && docker cp op-extract:/opt/... /tmp/...`) and diffing.
