<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TruckDrive to NCore V4 Converter

Converts [TruckDrive](http://torc-ai.github.io/TruckDrive) (Torc Robotics / Princeton) scenes
to NCore V4 format. TruckDrive is a long-range highway truck dataset with a custom
file-per-frame layout (one directory per scene). This converter mirrors the
`tools/data_converter/man_truckscenes` converter (multiple raw lidars + radar as ray bundles,
single `rig` frame) and the file-based `tools/data_converter/kitti` converter (scene glob + a
standalone trajectory file).

It is **self-contained** — it re-implements the small set of devkit primitives it needs
(`truckdrive/utils.py`) and depends only on numpy + pyquaternion, so **no pip-lock change or
devkit runtime dependency is required**.

## Usage

```bash
bazel run //tools/data_converter/truckdrive:truckdrive -- \
    --root-dir /path/to/TruckDrive \
    --output-dir /path/to/output \
    truckdrive-v4
```

Convert a single scene with `--scene-name scene_28_1`. `convert_one_scene.sh` wraps this for a
quick smoke test.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--scene-glob` | `scene_*` | Glob for scene directories under `--root-dir` |
| `--scene-name` | None | Convert only this scene directory |
| `--store-type` | itar | Output store format (`itar` or `directory`) |
| `--profile` | separate-sensors | Component group assignment profile |
| `--sequence-meta` / `--no-sequence-meta` | enabled | Emit sequence-meta JSON |
| `--annotated-window-only` / `--full-scene` | annotated-window-only | Convert only the per-scene span where 3D cuboid annotations exist (drop the unlabeled head/tail); `--full-scene` keeps every frame |

Plus the shared `--no-cameras` / `--camera-id` / `--no-lidars` / `--lidar-id` / `--no-radars` /
`--radar-id` sensor-selection options.

## What it stores (V1)

By default the converter trims each scene to the **per-scene time span where 3D cuboid annotations
exist** (`[first_box_ts, last_box_ts]`) and drops the unlabeled head/tail — TruckDrive labels only a
central span of each scene. The window is computed from each scene's own box timestamps, so it adapts
to any scene duration; sensor frames, the pose track, and the sequence interval are all restricted to
it. Pass `--full-scene` to keep every frame instead.

- **Poses**: dynamic `rig → world` from `poses/gt_trajectory.txt` (scene-local). The NCore `rig` is
  anchored to the devkit **`velodyne`** frame (x-forward, y-left, **z-up** — the devkit's canonical
  display + annotation frame). **Not** the devkit `vehicle` frame, which is y-right/**z-down** (NED-style)
  and would render the whole scene upside down. The raw trajectory is anchored to the **Aeva** frame
  (`lidar_aeva_forward_center_wide → world`), so it is re-anchored via the tf tree
  (`T_rig_world = T_aeva_world @ T_velodyne→aeva`) before storing; a static `world → world_global`
  carries the first-pose anchor. Static `<sensor> → rig` extrinsics are resolved from the per-scene
  tf tree (`calib_tf_tree_full.json`) by BFS to the `velodyne` frame.
- **Cameras (11)**: RGB JPEGs (`camera/leopard/<pos>/images`). Model from the calib `distortion_model`:
  `plumb_bob` → `OpenCVPinhole`, `equidistant` → `OpenCVFisheye` (zero distortion; images are
  delivered rectified to their model). Global shutter; intrinsics from the projection matrix `P`.
- **LiDARs (3 Ouster + 1 Aeva)**: raw ray bundles (no structured model). Per-point timestamps =
  filename timestamp + per-point offset. Ouster carries `reflectivity`/`ring`; **Aeva** carries
  the FMCW Doppler `radial_velocity_m_s`, a 3D `velocity_{x,y,z}_m_s` vector, `reflectivity`,
  `sensor_id`, and raw `intensity_raw` in `generic_data` (its native intensity is signed dB-like,
  so the first-class intensity channel is left at 0).
- **Radar (Continental conti542)**: ray bundle + `radial_velocity_m_s`, `range_rate_m_s`, `rcs`,
  `amplitude`, `azimuth/elevation`, the velocity vector, and `sensor_id` in `generic_data`.
- **Cuboids**: 3D boxes from `annotations/bounding_boxes/*.json` in the `velodyne` frame (a static
  `velodyne → rig` pose is stored so they can be interpreted), mapped to 9 coarse classes via the
  vendored `metainfo.json` label mapping; ignore/ego classes (id `-1`) are dropped. The coarse
  class names are normalised to the NCore snake_case convention: `bike`, `passenger_car`,
  `person`, `road_obstruction`, `semi_truck_cab`, `semi_truck_trailer`, `vehicle`, `traffic_sign`,
  `emergency_vehicle`.

## Deferred to V2 (not in V1)

- **Accumulated GT depth** (`accumulated_gt_depth/<pos>/*.npy.gz`, float32 to ~1 km): would store via
  `CameraLabelsComponent`, but (a) it'd be the first production use of the RAW-float + uint16
  quantization path and (b) whether the values are along-optical-axis `z` or along-ray range is
  not documented in the devkit and must be confirmed first. Deferred to avoid a silent
  correctness bug; easy fast-follow once z-vs-ray is confirmed.
- **Lane lines** (`annotations/lane_lines/*.json` polylines): no schema-correct V4 destination.
- **Aeva per-unit re-transform**: assumed already merged into the
  `lidar_aeva_forward_center_wide` reference frame; the per-point `sensor_id` is preserved so a
  correction would be non-destructive.

## Open items to confirm (flagged by the investigation)

- Aeva velocity semantics: raw FMCW Doppler (`radial_velocity_m_s`, ego motion not removed) vs the
  decomposed `velocity_{x,y,z}` — both are stored verbatim; pick the learning target downstream.
- Wide cameras: `distortion_model='equidistant'` with `D=0` → stored as fisheye with zero
  distortion (pure equidistant). Confirm the JPEGs are not pre-rectified to pinhole.
- Pose vs sensor clock: `gt_trajectory` seconds and filename nanosecond timestamps are treated as a
  common normalized clock (constant ~28 ms inter-sensor latency observed); confirm no clock offset.

## Testing

```bash
TRUCKDRIVE_DIR=/path/to/TruckDrive \
  bazel test //tools/data_converter/truckdrive:pytest_converter_3_11 --test_output=all
```

(`manual`-tagged; needs the dataset, excluded from `bazel test //...`.)

## NOTICE

`truckdrive-devkit/` is the upstream TruckDrive devkit (Apache-2.0), vendored unmodified. This
converter does not import it at runtime; it only reads the vendored `metainfo.json` class mapping.
