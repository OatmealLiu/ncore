<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MAN TruckScenes to NCore V4 Converter

Converts [MAN TruckScenes](https://www.man.eu/truckscenes/) scenes to NCore V4 format.
TruckScenes follows the nuScenes schema (its devkit is a fork of the nuScenes devkit), so
this converter mirrors the sibling `tools/data_converter/nuscenes` converter and adapts the
truck-specific sensor suite and conventions.

## Build prerequisite (one-time)

The vendored devkit decodes PCD point clouds with `pypcd4`, which is **not yet in the pinned
pip lock**. After checking out this converter you must regenerate the lock once (needs network):

```bash
# 1. recompile the 3.11 pip lock to include pypcd4 (+ transitive deps)
bazel run //deps/pip:requirements_3_11
# 2. refresh the bzlmod lock so --lockfile_mode=error is satisfied
bazel build //... --lockfile_mode=update --nobuild
# 3. commit the updated deps/pip/requirements_3_11.txt and MODULE.bazel.lock
```

The dependency itself is declared in `deps/pip/requirements_man_truckscenes.in` (wired into
`requirements_3_11.in` and the `//deps/pip:requirements_3_11` compile rule).

## Usage

```bash
bazel run //tools/data_converter/man_truckscenes:man_truckscenes -- \
    --root-dir /path/to/man-truckscenes \
    --output-dir /path/to/output \
    man-truckscenes-v4 \
    --version v1.2-trainval
```

### Convert a single scene (by name)

```bash
bazel run //tools/data_converter/man_truckscenes:man_truckscenes -- \
    --root-dir /path/to/man-truckscenes \
    --output-dir /path/to/output \
    man-truckscenes-v4 \
    --version v1.2-trainval \
    --scene-name scene-<...>
```

`convert_one_scene.sh` wraps this for a quick single-scene smoke test.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--version` | v1.2-trainval | TruckScenes version string |
| `--scene-token` / `--scene-name` | None | Convert a single scene |
| `--store-type` | itar | Output store format (`itar` or `directory`) |
| `--profile` | separate-sensors | Component group assignment profile |
| `--keyframes-only` / `--all-sweeps` | all-sweeps | Convert only ~2 Hz keyframes vs the full sweep cadence |
| `--keep-ego-trailer` / `--drop-ego-trailer` | keep | Keep the recording vehicle's own trailer as a distinct `ego_trailer` cuboid (not `trailer`) |
| `--sequence-meta` / `--no-sequence-meta` | enabled | Emit sequence-meta JSON |

The shared `--no-cameras` / `--camera-id` / `--no-lidars` / `--lidar-id` / `--no-radars` /
`--radar-id` options select a subset of sensors.

## Sensor suite & mapping

| Modality | TruckScenes channels | NCore sensor ids |
|----------|----------------------|------------------|
| Camera (4) | `CAMERA_{LEFT,RIGHT}_{FRONT,BACK}` | `camera_left_front`, … |
| LiDAR (6) | `LIDAR_TOP_{FRONT,LEFT,RIGHT}`, `LIDAR_{LEFT,RIGHT}`, `LIDAR_REAR` | `lidar_top_front`, … |
| Radar (6) | `RADAR_{LEFT,RIGHT}_{FRONT,BACK,SIDE}` | `radar_left_front`, … |

The two IMUs (`XSENSE_CABIN`, `XSENSE_CHASSIS`) are not NCore sensors; they back the
`ego_motion_cabin` / `ego_motion_chassis` tables (see caveats).

## What it stores

- **Poses**: dynamic `rig -> world` from `ego_pose` (UTM), anchored via a static
  `world -> world_global`; static `<sensor> -> rig` extrinsics from `calibrated_sensor`.
- **Cameras**: JPEG frames (1980×943) + `OpenCVPinholeCameraModelParameters` (global shutter,
  zero distortion — images are delivered rectified).
- **LiDARs**: raw per-sensor ray bundles (direction / distance / intensity) with **absolute
  per-point timestamps**. Stored **without** a structured spinning lidar model (see below).
- **Radars**: ray bundles + `radial_velocity_m_s` (relative velocity projected on the ray) and
  `rcs`. One timestamp per frame (radar has no per-point time).
- **Cuboids**: keyframe `sample_annotation` boxes in the global (`world_global`) frame, mapped
  to 12 detection classes via the devkit taxonomy.

## Design notes & caveats

- **No structured lidar model.** Unlike nuScenes (single HDL-32E modelled as a
  `RowOffsetStructuredSpinningLidarModel`), TruckScenes has 6 heterogeneous lidars with no
  beam/ring index and raw (un-compensated) clouds, so each is stored as a generic ray bundle
  with no intrinsics. This avoids forcing an inapplicable structured-grid model.
- **Single `rig` frame (dual-body collapse).** TruckScenes provides one static extrinsic set
  per sensor (into a single ego body frame) and separate `ego_motion_cabin` /
  `ego_motion_chassis` kinematics. The devkit treats the vehicle as one rigid frame; this
  converter does the same. The cabin tilts/rolls on its suspension relative to the chassis, so
  cab-mounted sensors deviate slightly from the static rig pose during accel/braking — this is
  not recoverable from the provided static calibration and is left for a possible V2.
- **Undocumented units.** Lidar `intensity` is normalised by 255 and clipped to `[0, 1]` (units
  unconfirmed); radar `rcs` is stored as-is under the key `rcs` (dBsm vs linear unconfirmed).

## Testing

```bash
MAN_TRUCKSCENES_DIR=/path/to/man-truckscenes \
  bazel test //tools/data_converter/man_truckscenes:pytest_converter_3_11 --test_output=all
```

(The test is tagged `manual`; it needs the dataset and is excluded from `bazel test //...`.)

## NOTICE

The `man-truckscenes-devkit/` subdirectory is the upstream MAN TruckScenes devkit (a fork of the
nuScenes devkit), vendored unmodified and used under its own license; see its `LICENSE`/`README`.
