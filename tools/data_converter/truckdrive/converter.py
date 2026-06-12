# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TruckDrive dataset to NCore V4 converter.

TruckDrive (Torc Robotics / Princeton) is a long-range highway truck dataset with a
custom file-per-frame layout (one directory per scene). This converter mirrors the
sibling ``man_truckscenes`` converter (multiple raw lidars + radar stored as generic
ray bundles, single ``rig`` frame) and the file-based ``kitti`` converter (scene glob +
a standalone trajectory file), and resolves per-sensor extrinsics from the per-scene
tf tree (``calib_tf_tree_full.json``).

V1 scope: 11 RGB cameras, 3 Ouster + 1 Aeva lidars, 1 Continental radar, ego poses, and
3D cuboids. The accumulated GT-depth maps and lane-line polylines are NOT yet converted
(no schema-correct V4 destination resolved); see README for the deferral rationale.

By default each scene is trimmed to the per-scene time span where 3D cuboid annotations
exist ([first_box_ts, last_box_ts]); the unlabeled head/tail (TruckDrive labels only a
central span of each scene) is dropped. The window is derived from each scene's own box
timestamps, so it adapts to any scene duration. Pass --full-scene to keep all frames.
"""

from __future__ import annotations

import json
import logging

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import click
import numpy as np
import tqdm

from upath import UPath

from ncore.impl.common.transformations import HalfClosedInterval, se3_inverse
from ncore.impl.data.types import (
    BBox3,
    CuboidTrackObservation,
    LabelSource,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    ShutterType,
)
from ncore.impl.data.v4.components import (
    CameraSensorComponent,
    CuboidsComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    MasksComponent,
    PosesComponent,
    RadarSensorComponent,
    SequenceComponentGroupsReader,
    SequenceComponentGroupsWriter,
)
from ncore.impl.data.v4.types import ComponentGroupAssignments
from ncore.impl.data_converter.base import FileBasedDataConverter, FileBasedDataConverterConfig
from tools.data_converter.cli import cli
from tools.data_converter.truckdrive import utils


# Safety buffer (us) around the computed sequence time span (pose timeline + active sensor
# per-frame times + active lidar per-point extrema). See man_truckscenes for rationale.
SEQUENCE_INTERVAL_MARGIN_US = 200_000
# Drop returns nearer than this (m) so a unit ray direction is well-defined.
MIN_RANGE_M = 1e-3


@dataclass(kw_only=True, slots=True)
class TruckDriveConverter4Config(FileBasedDataConverterConfig):
    """Configuration for TruckDrive to NCore V4 conversion."""

    scene_glob: str = "scene_*"
    scene_name: Optional[str] = None
    store_type: Literal["itar", "directory"] = "itar"
    component_group_profile: Literal["default", "separate-sensors", "separate-all"] = "separate-sensors"
    store_sequence_meta: bool = True
    # Convert only the per-scene time span where 3D cuboid annotations exist (drop the
    # unlabeled head/tail). The window is derived from each scene's own first/last box
    # timestamps, so it adapts to any scene duration. Set False to keep all sensor frames.
    trim_to_annotated_window: bool = True


class TruckDriveConverter4(FileBasedDataConverter):
    """Dataset preprocessing class for converting TruckDrive data to NCore V4 format.

    On-disk layout (post-reorg, see folder_structure_report.txt): per-scene sensor dirs are
    grouped under camera/, lidar/, radar/; ego poses under poses/; labels under annotations/;
    calibration under calibrations/. The root-level mmdet_annotations/ and sensor_sync/ folders
    (from the devkit's generate_training_data) are NOT consumed -- this converter reads the raw
    per-sensor files directly for full fidelity (per-point timestamps, intensity, Doppler, ring).

    Sensor suite & assumptions:
    - Cameras (11 leopard positions): RGB JPEGs under camera/leopard/<pos>/images. Per-camera
      projection model chosen from the calib distortion_model: 'plumb_bob' -> OpenCV
      pinhole, 'equidistant' -> OpenCV fisheye. Distortion coeffs are zero on disk
      (images delivered rectified to their model); global shutter.
    - LiDARs: 3 Ouster spinning under lidar/ouster/<pos>/points (.bin float32, 7 cols
      x,y,z,intensity,rel_time_ns,reflectivity,ring) + 1 Aeva FMCW joint under
      lidar/aeva/joint_lidars/points (.bin float64, 11 cols incl. per-point Doppler velocity +
      a vx,vy,vz vector). Stored as generic ray bundles (LidarSensorComponent, no structured
      model). Per-point timestamps come from the per-point time offset added to the frame's
      filename timestamp.
    - Radar: Continental conti542 joint under radar/conti542/joint_radars/detections (.bin
      float64, 33 cols). Stored as a ray bundle; radial velocity + rcs/amplitude/velocity-vectors
      carried in generic_data.
    - Ego motion: single 'rig' frame anchored to the devkit 'velodyne' frame (x-forward, y-left,
      z-UP -- the devkit's canonical/annotation frame; NOT the devkit 'vehicle' frame, which is
      y-right/z-down and would flip the scene upside down). Dynamic rig->world from
      poses/gt_trajectory.txt (scene-local), anchored via static world->world_global. Per-sensor
      static extrinsics resolved from the per-scene tf tree (calibrations/) via BFS.
    - Cuboids: 3D boxes from annotations/bounding_boxes/*.json in the 'velodyne' frame (== rig,
      so the stored velodyne->rig pose is identity); coarse class from metainfo.
    """

    def __init__(self, config: TruckDriveConverter4Config) -> None:
        super().__init__(config)
        self.scene_glob = config.scene_glob
        self.component_group_profile = config.component_group_profile
        self.store_type = config.store_type
        self.store_sequence_meta = config.store_sequence_meta
        self.trim_to_annotated_window = config.trim_to_annotated_window
        self._scene_name = config.scene_name
        # Per-scene annotated time window [lo, hi] us, set in convert_sequence; None = full scene.
        self._keep_window_us: Optional[tuple[int, int]] = None
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def get_sequence_ids(config: TruckDriveConverter4Config) -> List[str]:
        """Discover scene directory names to convert."""
        if config.scene_name is not None:
            return [config.scene_name]
        return utils.list_scene_dirs(str(config.root_dir), config.scene_glob)

    @staticmethod
    def from_config(config: TruckDriveConverter4Config) -> TruckDriveConverter4:
        return TruckDriveConverter4(config)

    # -------------------------------------------------------------------------

    def convert_sequence(self, sequence_id: str) -> None:
        """Convert a single TruckDrive scene to NCore V4 format."""
        scene_name = sequence_id
        scene_dir = UPath(str(self.root_dir)) / scene_name
        self.logger.info(f"Converting scene {scene_name}")

        # --- Transform graph + ego trajectory ---------------------------------
        tf_graph = utils.build_transform_graph(utils.load_tf_tree(scene_dir))
        pose_timestamps_us, t_aeva_world_all = utils.load_gt_trajectory(str(scene_dir / utils.TRAJECTORY_FILE))
        if len(pose_timestamps_us) < 2:
            raise AssertionError(f"Scene {scene_name}: gt_trajectory has < 2 poses.")
        if not np.all(np.diff(pose_timestamps_us.astype(np.int64)) > 0):
            raise AssertionError(f"Scene {scene_name}: gt_trajectory timestamps are not strictly increasing.")

        # gt_trajectory poses are anchored to the Aeva reference lidar frame
        # (lidar_aeva_forward_center_wide -> world). Re-anchor to the NCore rig frame
        # (RIG_NODE == 'velodyne', x-forward/y-left/z-UP -- see utils.RIG_NODE) so the
        # sensor->rig->world chain is correct AND the world is z-up: T_rig_world =
        # T_aeva_world @ T_rig_to_aeva. (Anchoring to the devkit 'vehicle' frame instead would
        # make world z-down -- vehicle is a y-right/z-down NED frame.)
        t_rig_to_aeva = utils.find_transform(tf_graph, utils.RIG_NODE, utils.LIDAR_SENSORS["aeva"]["node"])
        t_rig_world_all = t_aeva_world_all @ t_rig_to_aeva

        # --- Active sensors (intersect maps with present-on-disk + CLI selection) ----
        present_cameras = [p for p in utils.CAMERA_POSITIONS if (scene_dir / utils.camera_images_dir(p)).is_dir()]
        present_lidars = [k for k, v in utils.LIDAR_SENSORS.items() if (scene_dir / v["dir"]).is_dir()]
        present_radars = [k for k, v in utils.RADAR_SENSORS.items() if (scene_dir / v["dir"]).is_dir()]
        camera_ids = self.get_active_camera_ids(present_cameras)
        lidar_ids = self.get_active_lidar_ids(present_lidars)
        radar_ids = self.get_active_radar_ids(present_radars)

        # --- Annotated-window trim (per-scene; TruckDrive labels only a central span) ----
        # Derived from THIS scene's own first/last cuboid timestamps, so it adapts to any
        # scene duration. None => keep the full scene. All sensor decoders + the sequence
        # interval honour self._keep_window_us.
        self._keep_window_us = self._annotated_window(scene_dir) if self.trim_to_annotated_window else None

        # --- Sequence interval -------------------------------------------------
        if self._keep_window_us is not None:
            a_lo, a_hi = self._keep_window_us
            seq_start_us = max(0, a_lo - SEQUENCE_INTERVAL_MARGIN_US)
            seq_end_us = a_hi + SEQUENCE_INTERVAL_MARGIN_US
        else:
            seq_start_us, seq_end_us = self._compute_sequence_interval(
                scene_dir, camera_ids, lidar_ids, radar_ids, pose_timestamps_us
            )
        sequence_timestamp_interval_us = HalfClosedInterval.from_start_end(seq_start_us, seq_end_us)

        # --- Ego pose track: clip to the window (when trimming), anchor, pad ----
        if self._keep_window_us is not None:
            t_rig_world_all, pose_timestamps_us = self._clip_poses_to_interval(
                t_rig_world_all, pose_timestamps_us, seq_start_us, seq_end_us
            )
            if len(pose_timestamps_us) < 2:
                raise AssertionError(
                    f"Scene {scene_name}: < 2 ego poses within the annotated window "
                    f"[{seq_start_us}, {seq_end_us}] us."
                )

        # Anchor to the first (kept) pose (the trajectory is scene-local; keeps the established
        # two-frame world/world_global convention and float32-safe local coordinates).
        t_world_world_global = t_rig_world_all[0].copy()
        t_rig_world_relative = (se3_inverse(t_world_world_global) @ t_rig_world_all).astype(np.float32)
        t_rig_world_relative, pose_timestamps_us = self._pad_pose_timeline(
            t_rig_world_relative, pose_timestamps_us, seq_start_us, seq_end_us
        )

        # --- Writer + poses ---------------------------------------------------
        component_groups = ComponentGroupAssignments.create(
            camera_ids=camera_ids,
            lidar_ids=lidar_ids,
            radar_ids=radar_ids,
            point_clouds_ids=[],
            camera_labels_ids=[],
            profile=self.component_group_profile,
        )
        store_writer = SequenceComponentGroupsWriter(
            output_dir_path=self.output_dir / scene_name,
            store_base_name=scene_name,
            sequence_id=scene_name,
            sequence_timestamp_interval_us=sequence_timestamp_interval_us,
            store_type=self.store_type,
            generic_meta_data={"source_dataset": "truckdrive", "truckdrive_scene_name": scene_name},
        )
        poses_writer = store_writer.register_component_writer(
            PosesComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.poses_component_group,
            generic_meta_data={"calibration_type": "truckdrive:tf_tree", "egomotion_type": "truckdrive:gt_trajectory"},
        )
        intrinsics_writer = store_writer.register_component_writer(
            IntrinsicsComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.intrinsics_component_group,
        )
        masks_writer = store_writer.register_component_writer(
            MasksComponent.Writer, component_instance_name="default", group_name=component_groups.masks_component_group
        )
        poses_writer.store_dynamic_pose(
            source_frame_id="rig", target_frame_id="world", poses=t_rig_world_relative, timestamps_us=pose_timestamps_us
        )
        poses_writer.store_static_pose(
            source_frame_id="world", target_frame_id="world_global", pose=t_world_world_global
        )

        # --- Decode sensors ---------------------------------------------------
        self._decode_lidars(scene_dir, tf_graph, store_writer, poses_writer, component_groups, lidar_ids)
        self._decode_cameras(
            scene_dir,
            tf_graph,
            store_writer,
            poses_writer,
            intrinsics_writer,
            masks_writer,
            component_groups,
            camera_ids,
        )
        self._decode_radars(scene_dir, tf_graph, store_writer, poses_writer, component_groups, radar_ids)
        self._decode_cuboids(scene_dir, tf_graph, store_writer, poses_writer, component_groups)

        # --- Finalize ---------------------------------------------------------
        ncore_4_paths = store_writer.finalize()
        if self.store_sequence_meta:
            reader = SequenceComponentGroupsReader(ncore_4_paths)
            meta_path = self.output_dir / scene_name / f"{reader.sequence_id}.json"
            with meta_path.open("w") as f:
                json.dump(reader.get_sequence_meta().to_dict(), f, indent=2)
            self.logger.info(f"Wrote sequence meta data {str(meta_path)}")

    # -------------------------------------------------------------------------
    # Timeline helpers
    # -------------------------------------------------------------------------

    def _compute_sequence_interval(
        self, scene_dir, camera_ids, lidar_ids, radar_ids, pose_timestamps_us
    ) -> tuple[int, int]:
        """Interval covering the pose timeline, active sensor frame times, and active lidar per-point extrema."""
        lo = int(pose_timestamps_us[0])
        hi = int(pose_timestamps_us[-1])

        def fold_frame_times(frames):
            nonlocal lo, hi
            for _, ts_ns, _ in frames:
                ts_us = ts_ns // 1000
                lo, hi = min(lo, ts_us), max(hi, ts_us)

        for cam in camera_ids:
            fold_frame_times(utils.list_frames(scene_dir / utils.camera_images_dir(cam), ".jpg"))
        for rid in radar_ids:
            fold_frame_times(utils.list_frames(scene_dir / utils.RADAR_SENSORS[rid]["dir"], ".bin"))
        # Cuboid (bounding-box) frame timestamps must be covered too -- store_observations
        # asserts each observation timestamp lies within the sequence interval.
        fold_frame_times(utils.list_frames(scene_dir / utils.BOUNDING_BOXES_DIR, ".json"))
        for lid in lidar_ids:
            spec = utils.LIDAR_SENSORS[lid]
            frames = self._scene_frames(scene_dir / spec["dir"], ".bin")
            fold_frame_times(frames)
            # per-point extrema from the first & last frame (frames are time-ordered)
            for _, ts_ns, path in dict.fromkeys([frames[0], frames[-1]]) if frames else []:
                _, ppts = self._load_lidar_frame(str(path), spec, int(ts_ns))
                if ppts.size:
                    lo, hi = min(lo, int(ppts.min())), max(hi, int(ppts.max()))
        # Clamp the lower bound at 0 so the (uint64) timeline can never wrap negative.
        return max(0, lo - SEQUENCE_INTERVAL_MARGIN_US), hi + SEQUENCE_INTERVAL_MARGIN_US

    @staticmethod
    def _pad_pose_timeline(poses, timestamps_us, seq_start_us, seq_end_us) -> tuple[np.ndarray, np.ndarray]:
        """Extend the dynamic pose track to the sequence boundaries via constant-velocity extrapolation."""
        if seq_start_us < int(timestamps_us[0]):
            boundary = (
                (poses[0] @ (se3_inverse(poses[1]) @ poses[0])).astype(np.float32) if len(poses) >= 2 else poses[0]
            )
            poses = np.concatenate([boundary[np.newaxis], poses], axis=0)
            timestamps_us = np.concatenate([np.array([seq_start_us], dtype=np.uint64), timestamps_us])
        if seq_end_us > int(timestamps_us[-1]):
            boundary = (
                (poses[-1] @ (se3_inverse(poses[-2]) @ poses[-1])).astype(np.float32) if len(poses) >= 2 else poses[-1]
            )
            poses = np.concatenate([poses, boundary[np.newaxis]], axis=0)
            timestamps_us = np.concatenate([timestamps_us, np.array([seq_end_us], dtype=np.uint64)])
        return poses, timestamps_us

    # -------------------------------------------------------------------------
    # Annotated-window trim helpers
    # -------------------------------------------------------------------------

    def _annotated_window(self, scene_dir) -> Optional[tuple[int, int]]:
        """[first, last] cuboid timestamp (us) for this scene, or None if no usable span.

        Computed per-scene from the scene's own bounding-box file timestamps, so it adapts to
        each scene's duration. Falls back to None (full scene) if a scene has <2 annotations.
        """
        box_frames = utils.list_frames(scene_dir / utils.BOUNDING_BOXES_DIR, ".json")
        if len(box_frames) < 2:
            self.logger.warning(
                f"Annotated-window trim requested but scene has {len(box_frames)} annotation frame(s); "
                "converting the full scene."
            )
            return None
        lo = int(box_frames[0][1] // 1000)  # frames are time-sorted by list_frames
        hi = int(box_frames[-1][1] // 1000)
        if hi <= lo:
            self.logger.warning("Annotated-window trim: degenerate annotation span; converting the full scene.")
            return None
        self.logger.info(
            f"Annotated window: [{lo}, {hi}] us ({(hi - lo) / 1e6:.1f}s) from {len(box_frames)} annotation "
            "frames; dropping the unlabeled head/tail."
        )
        return (lo, hi)

    @staticmethod
    def _clip_poses_to_interval(
        poses: np.ndarray, timestamps_us: np.ndarray, lo_us: int, hi_us: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Keep poses within [lo, hi] us plus one bracketing sample each side (clean boundary interp)."""
        ts = timestamps_us.astype(np.int64)
        inside = np.nonzero((ts >= lo_us) & (ts <= hi_us))[0]
        if inside.size == 0:
            return poses[:0], timestamps_us[:0]
        i0 = max(0, int(inside[0]) - 1)
        i1 = min(len(poses), int(inside[-1]) + 2)
        return poses[i0:i1], timestamps_us[i0:i1]

    def _scene_frames(self, dir_path, suffix):
        """utils.list_frames, restricted to the annotated window (self._keep_window_us) when trimming."""
        frames = utils.list_frames(dir_path, suffix)
        if self._keep_window_us is None:
            return frames
        lo, hi = self._keep_window_us
        return [f for f in frames if lo <= (f[1] // 1000) <= hi]

    # -------------------------------------------------------------------------
    # Lidars (ray bundles; no structured model)
    # -------------------------------------------------------------------------

    @staticmethod
    def _load_lidar_frame(path: str, spec: Dict, frame_ts_ns: int) -> tuple[np.ndarray, np.ndarray]:
        """Load a lidar .bin -> (points [N,cols], per-point timestamp_us [N] uint64).

        Per-point absolute time = frame filename timestamp (ns) + per-point time offset (ns):
        Ouster col4 = rel_time_ns, Aeva col6 = time_offset_ns.
        """
        points = utils.load_point_bin(path, spec["dtype"], spec["cols"])
        time_col = 4 if spec["kind"] == "ouster" else 6
        offset_ns = points[:, time_col].astype(np.int64) if points.size else np.zeros(0, dtype=np.int64)
        # Clamp absolute time at 0 BEFORE the uint64 cast: the first synced frame can have
        # frame_ts_ns=0 (e.g. aeva 0000_0.bin), so a stray negative per-point offset would
        # otherwise floor-divide negative and wrap to a garbage ~1.8e19 us timestamp.
        abs_ns = np.maximum(np.int64(frame_ts_ns) + offset_ns, np.int64(0))
        timestamp_us = (abs_ns // 1000).astype(np.uint64)
        return points, timestamp_us

    def _decode_lidars(self, scene_dir, tf_graph, store_writer, poses_writer, component_groups, lidar_ids) -> None:
        for ncore_id in lidar_ids:
            spec = utils.LIDAR_SENSORS[ncore_id]
            frames = self._scene_frames(scene_dir / spec["dir"], ".bin")
            if not frames:
                self.logger.warning(f"No frames for lidar {ncore_id}")
                continue
            self.logger.info(f"Processing lidar {ncore_id}: {len(frames)} frames")

            poses_writer.store_static_pose(
                source_frame_id=ncore_id,
                target_frame_id="rig",
                pose=utils.find_transform(tf_graph, spec["node"], utils.RIG_NODE).astype(np.float32),
            )
            lidar_writer = store_writer.register_component_writer(
                LidarSensorComponent.Writer,
                component_instance_name=ncore_id,
                group_name=component_groups.lidar_component_groups.get(ncore_id),
                generic_meta_data={"sensor_kind": spec["kind"], "tf_node": spec["node"]},
            )

            for _, ts_ns, path in tqdm.tqdm(frames, desc=f"Process {ncore_id}"):
                points, timestamp_us = self._load_lidar_frame(str(path), spec, int(ts_ns))
                if not points.size:
                    continue
                xyz = points[:, 0:3].astype(np.float32)
                distance_m = np.linalg.norm(xyz, axis=1).astype(np.float32)
                valid = distance_m > MIN_RANGE_M
                if not valid.any():
                    continue
                xyz, distance_m, timestamp_us, points = (
                    xyz[valid],
                    distance_m[valid],
                    timestamp_us[valid],
                    points[valid],
                )
                direction = (xyz / distance_m[:, np.newaxis]).astype(np.float32)

                if spec["kind"] == "ouster":
                    # cols: x,y,z,intensity,rel_time_ns,reflectivity,ring
                    intensity = np.clip(points[:, 3].astype(np.float32) / 255.0, 0.0, 1.0)
                    generic_data = {
                        "reflectivity": points[:, 5].astype(np.float32),
                        "ring": points[:, 6].astype(np.float32),
                    }
                else:
                    # aeva cols: x,y,z,intensity,velocity,reflectivity,time_offset_ns,sensor_id,vx,vy,vz
                    # intensity is signed dB-like (not [0,1]); store zeros in the first-class channel
                    # and keep the raw value (plus FMCW Doppler + 3D velocity) in generic_data.
                    intensity = np.zeros(len(xyz), dtype=np.float32)
                    generic_data = {
                        "intensity_raw": points[:, 3].astype(np.float32),
                        "radial_velocity_m_s": points[:, 4].astype(np.float32),
                        "reflectivity": points[:, 5].astype(np.float32),
                        "sensor_id": points[:, 7].astype(np.float32),
                        "velocity_x_m_s": points[:, 8].astype(np.float32),
                        "velocity_y_m_s": points[:, 9].astype(np.float32),
                        "velocity_z_m_s": points[:, 10].astype(np.float32),
                    }

                lidar_writer.store_frame(
                    direction=direction,
                    timestamp_us=timestamp_us,
                    model_element=None,
                    distance_m=distance_m.reshape(1, -1),
                    intensity=intensity.reshape(1, -1),
                    frame_timestamps_us=np.array([int(timestamp_us.min()), int(timestamp_us.max())], dtype=np.uint64),
                    generic_data=generic_data,
                    generic_meta_data={},
                )

    # -------------------------------------------------------------------------
    # Cameras
    # -------------------------------------------------------------------------

    def _decode_cameras(
        self,
        scene_dir,
        tf_graph,
        store_writer,
        poses_writer,
        intrinsics_writer,
        masks_writer,
        component_groups,
        camera_ids,
    ) -> None:
        for ncore_id in camera_ids:
            frames = self._scene_frames(scene_dir / utils.camera_images_dir(ncore_id), ".jpg")
            if not frames:
                self.logger.warning(f"No frames for camera {ncore_id}")
                continue
            self.logger.info(f"Processing camera {ncore_id}: {len(frames)} frames")

            calib_path = utils.resolve_scene_file(scene_dir, utils.camera_calib_file(ncore_id))
            if calib_path is None:
                self.logger.warning(f"No calibration file for camera {ncore_id}; skipping")
                continue
            with calib_path.open() as f:
                calib = json.load(f)

            # _camera_model_from_calib emits a zero-distortion model (assumes images are delivered
            # rectified, matching the devkit which uses the projection matrix P directly). If the calib
            # actually carries non-zero distortion coefficients, surface it loudly so it is not silently
            # dropped into a wrong projection model.
            distortion = calib.get("D", calib.get("distortion_coefficients"))
            if distortion is not None and np.any(np.abs(np.asarray(distortion, dtype=np.float64)) > 1e-9):
                self.logger.warning(
                    f"Camera {ncore_id}: calib has NON-ZERO distortion {np.asarray(distortion).ravel().tolist()[:8]} "
                    f"(model='{calib.get('distortion_model', '')}'), but the converter emits a zero-distortion model "
                    f"(assumes rectified images). Verify the JPEGs are rectified, else intrinsics are wrong."
                )

            poses_writer.store_static_pose(
                source_frame_id=ncore_id,
                target_frame_id="rig",
                pose=utils.find_transform(tf_graph, utils.camera_tf_node(ncore_id), utils.RIG_NODE).astype(np.float32),
            )
            intrinsics_writer.store_camera_intrinsics(
                camera_id=ncore_id, camera_model_parameters=self._camera_model_from_calib(calib)
            )
            masks_writer.store_camera_masks(camera_id=ncore_id, mask_images={})

            camera_writer = store_writer.register_component_writer(
                CameraSensorComponent.Writer,
                component_instance_name=ncore_id,
                group_name=component_groups.camera_component_groups.get(ncore_id),
                generic_meta_data={"distortion_model": calib.get("distortion_model", "")},
            )
            for _, ts_ns, path in tqdm.tqdm(frames, desc=f"Process {ncore_id}"):
                with path.open("rb") as f:
                    image_binary = f.read()
                frame_ts = int(ts_ns // 1000)
                camera_writer.store_frame(
                    image_binary_data=image_binary,
                    image_format="jpeg",
                    frame_timestamps_us=np.array([frame_ts, frame_ts], dtype=np.uint64),
                    generic_data={},
                    generic_meta_data={},
                )

    @staticmethod
    def _camera_model_from_calib(calib: Dict):
        """Build the V4 camera model from a ROS CameraInfo calib dict (projection P, distortion_model)."""
        projection = np.array(calib["P"], dtype=np.float64).reshape(3, 4)
        focal_length = np.array([projection[0, 0], projection[1, 1]], dtype=np.float32)
        principal_point = np.array([projection[0, 2], projection[1, 2]], dtype=np.float32)
        resolution = np.array([int(calib["width"]), int(calib["height"])], dtype=np.uint64)
        model = str(calib.get("distortion_model", "")).lower()
        # Wide cameras carry an equidistant (fisheye) lens model; their FOV is too wide to be
        # a pinhole projection. D is zero on disk -> pure equidistant. (Guard against a
        # degenerate zero projection, which would fail the fisheye focal>0 invariant.)
        if model == "equidistant" and float(focal_length.min()) > 0.0:
            radial_coeffs = np.zeros(4, dtype=np.float32)
            max_angle = OpenCVFisheyeCameraModelParameters.compute_max_angle(
                resolution=resolution,
                focal_length=focal_length,
                principal_point=principal_point,
                radial_coeffs=radial_coeffs,
            )
            return OpenCVFisheyeCameraModelParameters(
                resolution=resolution,
                shutter_type=ShutterType.GLOBAL,
                external_distortion_parameters=None,
                principal_point=principal_point,
                focal_length=focal_length,
                radial_coeffs=radial_coeffs,
                max_angle=float(max_angle),
            )
        return OpenCVPinholeCameraModelParameters(  # plumb_bob (or unknown) -> rectified pinhole
            resolution=resolution,
            shutter_type=ShutterType.GLOBAL,
            external_distortion_parameters=None,
            principal_point=principal_point,
            focal_length=focal_length,
            radial_coeffs=np.zeros(6, dtype=np.float32),
            tangential_coeffs=np.zeros(2, dtype=np.float32),
            thin_prism_coeffs=np.zeros(4, dtype=np.float32),
        )

    # -------------------------------------------------------------------------
    # Radar
    # -------------------------------------------------------------------------

    def _decode_radars(self, scene_dir, tf_graph, store_writer, poses_writer, component_groups, radar_ids) -> None:
        for ncore_id in radar_ids:
            spec = utils.RADAR_SENSORS[ncore_id]
            frames = self._scene_frames(scene_dir / spec["dir"], ".bin")
            if not frames:
                self.logger.warning(f"No frames for radar {ncore_id}")
                continue
            self.logger.info(f"Processing radar {ncore_id}: {len(frames)} frames")

            poses_writer.store_static_pose(
                source_frame_id=ncore_id,
                target_frame_id="rig",
                pose=utils.find_transform(tf_graph, spec["node"], utils.RIG_NODE).astype(np.float32),
            )
            radar_writer = store_writer.register_component_writer(
                RadarSensorComponent.Writer,
                component_instance_name=ncore_id,
                group_name=component_groups.radar_component_groups.get(ncore_id),
                generic_meta_data={"tf_node": spec["node"]},
            )
            for _, ts_ns, path in tqdm.tqdm(frames, desc=f"Process {ncore_id}"):
                points = utils.load_point_bin(str(path), spec["dtype"], spec["cols"])
                if not points.size:
                    continue
                xyz = points[:, 0:3].astype(np.float32)
                distance_m = np.linalg.norm(xyz, axis=1).astype(np.float32)
                valid = distance_m > 0.1
                if not valid.any():
                    continue
                xyz, distance_m, points = xyz[valid], distance_m[valid], points[valid]
                direction = (xyz / distance_m[:, np.newaxis]).astype(np.float32)
                # Conti542 33-col layout (devkit dataset_details.py / colorize.py / README): the primary
                # velocity triple is cols 27-29 (vx,vy,vz); cols 30-32 (vx0,vy0,vz0) are a second triple
                # whose ego-compensation semantics the devkit does not document -- carry BOTH verbatim and
                # let the downstream consumer choose. radial_velocity projects the primary triple onto the ray.
                velocity_vec = points[:, 27:30].astype(np.float32)
                velocity_vec0 = points[:, 30:33].astype(np.float32)
                radial_velocity = np.sum(velocity_vec * direction, axis=1).astype(np.float32)

                frame_ts = int(ts_ns // 1000)
                radar_writer.store_frame(
                    direction=direction,
                    timestamp_us=np.full(len(xyz), frame_ts, dtype=np.uint64),
                    distance_m=distance_m.reshape(1, -1),
                    frame_timestamps_us=np.array([frame_ts, frame_ts], dtype=np.uint64),
                    generic_data={
                        "radial_velocity_m_s": radial_velocity,
                        "range_rate_m_s": points[:, 3].astype(np.float32),
                        "rcs": points[:, 4].astype(np.float32),
                        "amplitude": points[:, 5].astype(np.float32),
                        "azimuth_rad": points[:, 7].astype(np.float32),
                        "elevation_rad": points[:, 8].astype(np.float32),
                        "sensor_id": points[:, 26].astype(np.float32),
                        "velocity_x_m_s": velocity_vec[:, 0],
                        "velocity_y_m_s": velocity_vec[:, 1],
                        "velocity_z_m_s": velocity_vec[:, 2],
                        "velocity_x0_m_s": velocity_vec0[:, 0],
                        "velocity_y0_m_s": velocity_vec0[:, 1],
                        "velocity_z0_m_s": velocity_vec0[:, 2],
                    },
                    generic_meta_data={},
                )

    # -------------------------------------------------------------------------
    # Cuboids
    # -------------------------------------------------------------------------

    @staticmethod
    def _box_geometry(obj: Dict):
        """Return (cx, cy, cz, length, width, height, yaw) for a box, or None if unparseable.

        Handles the bare ego/velodyne fields (x, y, z, l, w, h, yaw) and the nested 'lidar'
        form (x_c, y_c, z_c, ...) that the devkit also supports.
        """
        candidates = [(obj, ("x", "y", "z", "l", "w", "h", "yaw"))]
        nested = obj.get("lidar")
        if isinstance(nested, dict):
            candidates.append((nested, ("x_c", "y_c", "z_c", "l", "w", "h", "yaw")))
        for source, keys in candidates:
            try:
                cx, cy, cz, length, width, height, yaw = (float(source[k]) for k in keys)
            except (KeyError, TypeError, ValueError):
                continue
            return cx, cy, cz, abs(length), abs(width), abs(height), yaw
        return None

    def _decode_cuboids(self, scene_dir, tf_graph, store_writer, poses_writer, component_groups) -> None:
        box_frames = utils.list_frames(scene_dir / utils.BOUNDING_BOXES_DIR, ".json")
        if not box_frames:
            self.logger.info("No bounding-box annotations found")
            return

        # The 3D boxes live in the 'velodyne' frame; store its static extrinsic so the V4
        # pose graph can interpret them (no sensor streams in this frame).
        poses_writer.store_static_pose(
            source_frame_id=utils.ANNOTATION_NODE,
            target_frame_id="rig",
            pose=utils.find_transform(tf_graph, utils.ANNOTATION_NODE, utils.RIG_NODE).astype(np.float32),
        )

        observations: List[CuboidTrackObservation] = []
        for _, ts_ns, path in tqdm.tqdm(box_frames, desc="Process cuboids"):
            timestamp_us = int(ts_ns // 1000)
            with path.open() as f:
                raw = json.load(f)
            objects = list(raw.values()) if isinstance(raw, dict) else raw

            geometry_failures = 0
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                class_name = utils.class_name_for_label(str(obj.get("class-id", "")))
                if class_name is None:  # unmapped or ignore (ego cab/trailer, *DontCare groups)
                    continue
                geometry = self._box_geometry(obj)
                if geometry is None:
                    geometry_failures += 1
                    continue
                cx, cy, cz, length, width, height, yaw = geometry
                bbox3 = BBox3.from_array(np.array([cx, cy, cz, length, width, height, 0.0, 0.0, yaw], dtype=np.float32))
                observations.append(
                    CuboidTrackObservation(
                        track_id=str(obj.get("Tracking_ID", obj.get("id", ""))),
                        class_id=class_name,
                        timestamp_us=timestamp_us,
                        reference_frame_id=utils.ANNOTATION_NODE,
                        reference_frame_timestamp_us=timestamp_us,
                        bbox3=bbox3,
                        # LabelSource.EXTERNAL for dataset-provided GT boxes, matching the
                        # nuScenes/waymo/man_truckscenes convention.
                        source=LabelSource.EXTERNAL,
                    )
                )
            if geometry_failures:
                self.logger.warning(
                    f"Box file {path.name}: {geometry_failures} mapped objects had unparseable geometry (schema mismatch?)"
                )

        if observations:
            store_writer.register_component_writer(
                CuboidsComponent.Writer, "default", component_groups.cuboid_track_observations_component_group
            ).store_observations(observations)
            self.logger.info(f"Stored {len(observations)} cuboid observations")
        else:
            self.logger.info("No mapped cuboid observations to store")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


@cli.command(name="truckdrive-v4")
@click.option(
    "--scene-glob", type=str, default="scene_*", show_default=True, help="Glob for scene directories under --root-dir"
)
@click.option("--scene-name", type=str, default=None, help="Convert only this scene directory")
@click.option(
    "--store-type",
    type=click.Choice(["itar", "directory"], case_sensitive=False),
    default="itar",
    show_default=True,
    help="Output store type",
)
@click.option(
    "component_group_profile",
    "--profile",
    type=click.Choice(["default", "separate-sensors", "separate-all"], case_sensitive=False),
    default="separate-sensors",
    show_default=True,
    help="Output profile for component group assignment",
)
@click.option(
    "store_sequence_meta", "--sequence-meta/--no-sequence-meta", default=True, help="Generate sequence meta-data JSON?"
)
@click.option(
    "trim_to_annotated_window",
    "--annotated-window-only/--full-scene",
    default=True,
    show_default=True,
    help="Convert only the per-scene span where 3D cuboid annotations exist (drop the unlabeled "
    "head/tail); --full-scene keeps every sensor frame.",
)
@click.pass_context
def truckdrive_v4(ctx, scene_glob, scene_name, **kwargs):
    """TruckDrive data conversion (V4 format)"""
    config = TruckDriveConverter4Config(
        **{**vars(ctx.obj), "scene_glob": scene_glob, "scene_name": scene_name, **kwargs}
    )
    TruckDriveConverter4.convert(config)
