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

"""MAN TruckScenes dataset to NCore V4 converter."""

from __future__ import annotations

import json
import logging

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import click
import numpy as np
import tqdm

from pyquaternion import Quaternion
from truckscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from upath import UPath

from ncore.impl.common.transformations import HalfClosedInterval, se3_inverse
from ncore.impl.data.types import (
    BBox3,
    CuboidTrackObservation,
    LabelSource,
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
from tools.data_converter.man_truckscenes.utils import (
    CAMERA_MAP,
    EGO_TRAILER_CATEGORY,
    LIDAR_MAP,
    RADAR_MAP,
    REF_LIDAR_CHANNEL,
    TRUCKSCENES_CATEGORY_MAP,
    get_sample_records,
    get_sweep_tokens,
    get_truckscenes,
    resolve_scene_token,
)


# Safety buffer (microseconds) added on each side of the computed sequence time span. The
# span itself already covers the pose timeline, every active sensor's per-frame timestamps,
# and the active lidars' raw per-point timestamp extrema (see _compute_sequence_interval),
# so this is only a small guard for the half-closed interval boundary and minor clock effects.
SEQUENCE_INTERVAL_MARGIN_US = 200_000

# Minimum range (m) below which lidar/radar returns are treated as degenerate (drops
# zero-range origin points so a unit ray direction can be computed).
MIN_RANGE_M = 1e-3


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass(kw_only=True, slots=True)
class ManTruckScenesConverter4Config(FileBasedDataConverterConfig):
    """Configuration for MAN TruckScenes to NCore V4 conversion."""

    version: str = "v1.2-trainval"
    scene_token: Optional[str] = None
    scene_name: Optional[str] = None
    store_type: Literal["itar", "directory"] = "itar"
    component_group_profile: Literal["default", "separate-sensors", "separate-all"] = "separate-sensors"
    store_sequence_meta: bool = True
    # Convert only annotated keyframes (~2 Hz) instead of the full sweep cadence (~10-20 Hz).
    keyframes_only: bool = False
    # Keep the recording vehicle's own (rigidly attached) trailer as a 'trailer' cuboid.
    keep_ego_trailer: bool = True


# -----------------------------------------------------------------------------
# Converter
# -----------------------------------------------------------------------------


class ManTruckScenesConverter4(FileBasedDataConverter):
    """Dataset preprocessing class for converting MAN TruckScenes data to NCore V4 format.

    MAN TruckScenes follows the nuScenes schema; this converter mirrors the nuScenes
    converter but adapts every truck-specific assumption.

    Sensor suite & assumptions:
    - Cameras (4: CAMERA_{LEFT,RIGHT}_{FRONT,BACK}): treated as global shutter
      (ShutterType.GLOBAL); images are delivered undistorted/rectified, so all
      distortion coefficients are zero. Intrinsics come from calibrated_sensor.
    - LiDARs (6: LIDAR_TOP_{FRONT,LEFT,RIGHT}, LIDAR_{LEFT,RIGHT}, LIDAR_REAR): the
      on-disk PCD clouds are RAW (not motion-compensated) with absolute per-point
      timestamps and no beam/ring index. They are stored as generic ray bundles
      (LidarSensorComponent without any structured spinning lidar model), NOT via the
      HDL-32E structured-model path used for nuScenes. PCD files are LZF
      ``binary_compressed`` and are decoded with the devkit's pypcd4-based loader.
    - Radars (6: RADAR_{LEFT,RIGHT}_{FRONT,BACK,SIDE}): 7-field PCD
      (x,y,z,vrel_x,vrel_y,vrel_z,rcs); radial velocity is the relative velocity vector
      projected onto the ray direction. No per-point timestamp (one per-frame time).
    - Ego motion: modelled as a single unified ``rig`` body frame (the frame
      calibrated_sensor extrinsics are expressed in), exactly as the devkit does. The
      dual-body cabin/chassis kinematics (ego_motion_cabin / ego_motion_chassis) are NOT
      modelled as separate geometric frames in V1.
    - Cuboid annotations: stored in the global (UTM) frame, keyframes only.
    """

    def __init__(self, config: ManTruckScenesConverter4Config) -> None:
        super().__init__(config)

        self.component_group_profile = config.component_group_profile
        self.store_type = config.store_type
        self.store_sequence_meta = config.store_sequence_meta
        self.keyframes_only = config.keyframes_only
        self.keep_ego_trailer = config.keep_ego_trailer

        self._version = config.version
        self._scene_token = config.scene_token
        self._scene_name = config.scene_name

        self.logger = logging.getLogger(__name__)

    @staticmethod
    def get_sequence_ids(config: ManTruckScenesConverter4Config) -> List[str]:
        """Discover scene tokens to convert."""
        trucksc = get_truckscenes(version=config.version, dataroot=config.root_dir)

        resolved = resolve_scene_token(trucksc, config.scene_token, config.scene_name)
        if resolved is not None:
            return [resolved]

        return [s["token"] for s in trucksc.scene]

    @staticmethod
    def from_config(config: ManTruckScenesConverter4Config) -> ManTruckScenesConverter4:
        return ManTruckScenesConverter4(config)

    def _ego_pose_matrix(self, trucksc, sample_data_record: Dict) -> np.ndarray:
        """Build the 4x4 rig->world (UTM) transform for a sample_data's ego_pose."""
        ego_pose = trucksc.get("ego_pose", sample_data_record["ego_pose_token"])
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = Quaternion(ego_pose["rotation"]).rotation_matrix
        transform[:3, 3] = ego_pose["translation"]
        return transform

    @staticmethod
    def _extrinsic_matrix(calibrated_sensor: Dict) -> np.ndarray:
        """Build the 4x4 sensor->rig transform from a calibrated_sensor record."""
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = Quaternion(calibrated_sensor["rotation"]).rotation_matrix
        transform[:3, 3] = calibrated_sensor["translation"]
        return transform

    def convert_sequence(self, sequence_id: str) -> None:
        """Convert a single MAN TruckScenes scene to NCore V4 format."""
        scene_token = sequence_id
        trucksc = get_truckscenes(version=self._version, dataroot=str(self.root_dir))
        scene_record = trucksc.get("scene", scene_token)
        scene_name = scene_record["name"]

        self.logger.info(f"Converting scene {scene_name} ({scene_token})")
        sequence_output_name = scene_name

        # --- Determine active sensors ------------------------------------------
        camera_ids = self.get_active_camera_ids(list(CAMERA_MAP.keys()))
        lidar_ids = self.get_active_lidar_ids(list(LIDAR_MAP.keys()))
        radar_ids = self.get_active_radar_ids(list(RADAR_MAP.keys()))

        # --- Master pose timeline from the reference lidar ---------------------
        ref_sweep_tokens = get_sweep_tokens(trucksc, scene_record, REF_LIDAR_CHANNEL, self.keyframes_only)
        if len(ref_sweep_tokens) < 2:
            raise AssertionError(
                f"Scene {scene_name} has < 2 {REF_LIDAR_CHANNEL} frames ({len(ref_sweep_tokens)}); cannot build a pose timeline."
            )
        ref_sweep_data = [trucksc.get("sample_data", t) for t in ref_sweep_tokens]
        pose_timestamps_us = np.array([sd["timestamp"] for sd in ref_sweep_data], dtype=np.uint64)
        if not np.all(np.diff(pose_timestamps_us.astype(np.int64)) > 0):
            raise AssertionError(
                f"Scene {scene_name}: {REF_LIDAR_CHANNEL} sample_data timestamps are not strictly increasing; "
                "cannot use them as the dynamic-pose timeline."
            )

        # rig -> world (UTM) per reference-lidar sweep, anchored to the first pose so
        # local coordinates stay small enough for float32 (UTM coords are ~1e6 m).
        t_rig_world_all = np.stack([self._ego_pose_matrix(trucksc, sd) for sd in ref_sweep_data])  # [N,4,4] f64
        t_world_world_global = t_rig_world_all[0].copy()  # f64 high-precision anchor
        t_rig_world_relative = (se3_inverse(t_world_world_global) @ t_rig_world_all).astype(np.float32)

        # --- Sequence interval over all active sensors -------------------------
        seq_start_us, seq_end_us = self._compute_sequence_interval(
            trucksc, scene_record, camera_ids, lidar_ids, radar_ids, pose_timestamps_us
        )
        sequence_timestamp_interval_us = HalfClosedInterval.from_start_end(seq_start_us, seq_end_us)

        # Pad the pose timeline (constant-velocity extrapolation) to cover the interval
        # boundaries, so the dynamic rig->world track spans the full sequence.
        t_rig_world_relative, pose_timestamps_us = self._pad_pose_timeline(
            t_rig_world_relative, pose_timestamps_us, seq_start_us, seq_end_us
        )

        # --- Component group assignments + writer ------------------------------
        component_groups = ComponentGroupAssignments.create(
            camera_ids=camera_ids,
            lidar_ids=lidar_ids,
            radar_ids=radar_ids,
            point_clouds_ids=[],
            camera_labels_ids=[],
            profile=self.component_group_profile,
        )

        store_writer = SequenceComponentGroupsWriter(
            output_dir_path=self.output_dir / sequence_output_name,
            store_base_name=sequence_output_name,
            sequence_id=sequence_output_name,
            sequence_timestamp_interval_us=sequence_timestamp_interval_us,
            store_type=self.store_type,
            generic_meta_data={
                "source_dataset": "man_truckscenes",
                "truckscenes_version": self._version,
                "truckscenes_scene_token": scene_token,
                "truckscenes_scene_name": scene_name,
                "scene_description": scene_record.get("description", ""),
            },
        )

        poses_writer = store_writer.register_component_writer(
            PosesComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.poses_component_group,
            generic_meta_data={
                "calibration_type": "truckscenes:calibrated_sensor",
                "egomotion_type": "truckscenes:ego_pose",
            },
        )
        intrinsics_writer = store_writer.register_component_writer(
            IntrinsicsComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.intrinsics_component_group,
        )
        masks_writer = store_writer.register_component_writer(
            MasksComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.masks_component_group,
        )

        # --- Store ego poses ---------------------------------------------------
        poses_writer.store_dynamic_pose(
            source_frame_id="rig",
            target_frame_id="world",
            poses=t_rig_world_relative,
            timestamps_us=pose_timestamps_us,
        )
        poses_writer.store_static_pose(
            source_frame_id="world",
            target_frame_id="world_global",
            pose=t_world_world_global,
        )

        # --- Decode sensors ----------------------------------------------------
        self._decode_lidars(trucksc, scene_record, store_writer, poses_writer, component_groups, lidar_ids)
        self._decode_cameras(
            trucksc,
            scene_record,
            store_writer,
            poses_writer,
            intrinsics_writer,
            masks_writer,
            component_groups,
            camera_ids,
        )
        self._decode_radars(trucksc, scene_record, store_writer, poses_writer, component_groups, radar_ids)
        self._decode_cuboids(trucksc, scene_record, store_writer, component_groups)

        # --- Finalize ----------------------------------------------------------
        ncore_4_paths = store_writer.finalize()

        if self.store_sequence_meta:
            sequence_component_reader = SequenceComponentGroupsReader(ncore_4_paths)
            sequence_meta_path = (
                self.output_dir / sequence_output_name / f"{sequence_component_reader.sequence_id}.json"
            )
            with sequence_meta_path.open("w") as f:
                json.dump(sequence_component_reader.get_sequence_meta().to_dict(), f, indent=2)
            self.logger.info(f"Wrote sequence meta data {str(sequence_meta_path)}")

    # -------------------------------------------------------------------------
    # Timeline helpers
    # -------------------------------------------------------------------------

    def _compute_sequence_interval(
        self, trucksc, scene_record, camera_ids, lidar_ids, radar_ids, pose_timestamps_us: np.ndarray
    ) -> tuple[int, int]:
        """Sequence time interval [start, end] (microseconds), expanded by a safety buffer.

        Must cover (a) the dynamic rig->world pose timeline (built unconditionally from the
        reference lidar, so it must be covered even when that lidar is not an active output
        sensor), (b) every active sensor's per-frame sample_data timestamps, and (c) the
        absolute per-point timestamp extrema of the active lidars -- their raw acquisition
        windows can extend slightly beyond the representative per-frame timestamps. For (c)
        only the first and last frame of each lidar are scanned, since frames are stored in
        acquisition order.
        """
        # (a) the pose timeline.
        lo = int(pose_timestamps_us[0])
        hi = int(pose_timestamps_us[-1])

        # (b) per-frame sample_data timestamps of all active sensors.
        active_channels = (
            [LIDAR_MAP[i] for i in lidar_ids] + [CAMERA_MAP[i] for i in camera_ids] + [RADAR_MAP[i] for i in radar_ids]
        )
        for channel in active_channels:
            for token in get_sweep_tokens(trucksc, scene_record, channel, self.keyframes_only):
                ts = int(trucksc.get("sample_data", token)["timestamp"])
                lo, hi = min(lo, ts), max(hi, ts)

        # (c) raw per-point timestamp extrema of each active lidar (first & last frame only).
        for ncore_id in lidar_ids:
            tokens = get_sweep_tokens(trucksc, scene_record, LIDAR_MAP[ncore_id], self.keyframes_only)
            for token in dict.fromkeys([tokens[0], tokens[-1]]) if tokens else []:
                sd = trucksc.get("sample_data", token)
                point_timestamps = np.asarray(
                    LidarPointCloud.from_file(str(UPath(str(self.root_dir)) / sd["filename"])).timestamps,
                    dtype=np.uint64,
                )
                if point_timestamps.size:
                    lo, hi = min(lo, int(point_timestamps.min())), max(hi, int(point_timestamps.max()))

        return lo - SEQUENCE_INTERVAL_MARGIN_US, hi + SEQUENCE_INTERVAL_MARGIN_US

    @staticmethod
    def _pad_pose_timeline(
        poses: np.ndarray, timestamps_us: np.ndarray, seq_start_us: int, seq_end_us: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extend a dynamic pose track to the sequence boundaries via constant-velocity extrapolation."""
        if seq_start_us < int(timestamps_us[0]):
            if len(poses) >= 2:
                delta_inv = se3_inverse(poses[1]) @ poses[0]
                boundary = (poses[0] @ delta_inv).astype(np.float32)
            else:
                boundary = poses[0]
            poses = np.concatenate([boundary[np.newaxis], poses], axis=0)
            timestamps_us = np.concatenate([np.array([seq_start_us], dtype=np.uint64), timestamps_us])

        if seq_end_us > int(timestamps_us[-1]):
            if len(poses) >= 2:
                delta = se3_inverse(poses[-2]) @ poses[-1]
                boundary = (poses[-1] @ delta).astype(np.float32)
            else:
                boundary = poses[-1]
            poses = np.concatenate([poses, boundary[np.newaxis]], axis=0)
            timestamps_us = np.concatenate([timestamps_us, np.array([seq_end_us], dtype=np.uint64)])

        return poses, timestamps_us

    # -------------------------------------------------------------------------
    # Lidar (generic ray bundle -- no structured spinning model)
    # -------------------------------------------------------------------------

    def _decode_lidars(
        self,
        trucksc,
        scene_record: Dict,
        store_writer: SequenceComponentGroupsWriter,
        poses_writer: PosesComponent.Writer,
        component_groups: ComponentGroupAssignments,
        lidar_ids: List[str],
    ) -> None:
        """Decode and store all active lidars as raw ray bundles.

        TruckScenes lidar PCDs are raw (un-compensated) sensor-frame clouds with absolute
        per-point timestamps and no beam/ring index, so they are stored without any
        structured spinning lidar model (model_element=None, no stored intrinsics). The
        rig->world dynamic pose plus per-point timestamps let any downstream consumer
        deskew if desired.
        """
        for ncore_id, channel in LIDAR_MAP.items():
            if ncore_id not in lidar_ids:
                continue

            sweep_tokens = get_sweep_tokens(trucksc, scene_record, channel, self.keyframes_only)
            sweep_data = [trucksc.get("sample_data", t) for t in sweep_tokens]
            if not sweep_data:
                self.logger.warning(f"No data for lidar {channel}")
                continue

            self.logger.info(f"Processing lidar {ncore_id} ({channel}): {len(sweep_data)} frames")

            calibrated_sensor = trucksc.get("calibrated_sensor", sweep_data[0]["calibrated_sensor_token"])
            poses_writer.store_static_pose(
                source_frame_id=ncore_id, target_frame_id="rig", pose=self._extrinsic_matrix(calibrated_sensor)
            )

            lidar_writer = store_writer.register_component_writer(
                LidarSensorComponent.Writer,
                component_instance_name=ncore_id,
                group_name=component_groups.lidar_component_groups.get(ncore_id),
                generic_meta_data={"channel": channel},
            )

            logged_intensity = False
            for sd in tqdm.tqdm(sweep_data, desc=f"Process {ncore_id}"):
                pc = LidarPointCloud.from_file(str(UPath(str(self.root_dir)) / sd["filename"]))
                xyz = pc.points[:3].T.astype(np.float32)  # [N, 3]
                intensity_raw = np.asarray(pc.points[3], dtype=np.float32)  # [N]
                # Per-point absolute timestamps (uint64 microseconds).
                timestamp_us = np.asarray(pc.timestamps, dtype=np.uint64).reshape(-1)

                distance_m = np.linalg.norm(xyz, axis=1).astype(np.float32)
                valid = distance_m > MIN_RANGE_M
                if not valid.any():
                    continue
                xyz, intensity_raw, timestamp_us, distance_m = (
                    xyz[valid],
                    intensity_raw[valid],
                    timestamp_us[valid],
                    distance_m[valid],
                )

                direction = (xyz / distance_m[:, np.newaxis]).astype(np.float32)
                # Intensity units are undocumented; normalise by 255 (8-bit return) and clip
                # to the [0, 1] range required by store_frame.
                intensity = np.clip(intensity_raw / 255.0, 0.0, 1.0).astype(np.float32)

                if not logged_intensity:
                    self.logger.info(
                        f"  {ncore_id}: raw intensity max={float(intensity_raw.max()):.1f} (normalised by /255)"
                    )
                    logged_intensity = True

                lidar_writer.store_frame(
                    direction=direction,
                    timestamp_us=timestamp_us,
                    model_element=None,
                    distance_m=distance_m.reshape(1, -1),
                    intensity=intensity.reshape(1, -1),
                    frame_timestamps_us=np.array([int(timestamp_us.min()), int(timestamp_us.max())], dtype=np.uint64),
                    generic_data={},
                    generic_meta_data={},
                )

    # -------------------------------------------------------------------------
    # Cameras
    # -------------------------------------------------------------------------

    def _decode_cameras(
        self,
        trucksc,
        scene_record: Dict,
        store_writer: SequenceComponentGroupsWriter,
        poses_writer: PosesComponent.Writer,
        intrinsics_writer: IntrinsicsComponent.Writer,
        masks_writer: MasksComponent.Writer,
        component_groups: ComponentGroupAssignments,
        camera_ids: List[str],
    ) -> None:
        """Decode and store all camera frames (undistorted global-shutter pinhole)."""
        for ncore_id, channel in CAMERA_MAP.items():
            if ncore_id not in camera_ids:
                continue

            sweep_tokens = get_sweep_tokens(trucksc, scene_record, channel, self.keyframes_only)
            sweep_data = [trucksc.get("sample_data", t) for t in sweep_tokens]
            if not sweep_data:
                self.logger.warning(f"No data for camera {channel}")
                continue

            self.logger.info(f"Processing camera {ncore_id} ({channel}): {len(sweep_data)} frames")

            calibrated_sensor = trucksc.get("calibrated_sensor", sweep_data[0]["calibrated_sensor_token"])
            poses_writer.store_static_pose(
                source_frame_id=ncore_id, target_frame_id="rig", pose=self._extrinsic_matrix(calibrated_sensor)
            )

            intrinsic = np.array(calibrated_sensor["camera_intrinsic"], dtype=np.float32)  # [3, 3]
            width = int(sweep_data[0]["width"])
            height = int(sweep_data[0]["height"])

            # Images are delivered undistorted/rectified, so all distortion coeffs are zero.
            # ShutterType.GLOBAL: a single capture timestamp per image, no rolling-shutter metadata.
            intrinsics_writer.store_camera_intrinsics(
                camera_id=ncore_id,
                camera_model_parameters=OpenCVPinholeCameraModelParameters(
                    resolution=np.array([width, height], dtype=np.uint64),
                    shutter_type=ShutterType.GLOBAL,
                    external_distortion_parameters=None,
                    principal_point=np.array([intrinsic[0, 2], intrinsic[1, 2]], dtype=np.float32),
                    focal_length=np.array([intrinsic[0, 0], intrinsic[1, 1]], dtype=np.float32),
                    radial_coeffs=np.zeros(6, dtype=np.float32),
                    tangential_coeffs=np.zeros(2, dtype=np.float32),
                    thin_prism_coeffs=np.zeros(4, dtype=np.float32),
                ),
            )
            masks_writer.store_camera_masks(camera_id=ncore_id, mask_images={})

            camera_writer = store_writer.register_component_writer(
                CameraSensorComponent.Writer,
                component_instance_name=ncore_id,
                group_name=component_groups.camera_component_groups.get(ncore_id),
                generic_meta_data={"channel": channel},
            )

            for sd in tqdm.tqdm(sweep_data, desc=f"Process {ncore_id}"):
                with (UPath(str(self.root_dir)) / sd["filename"]).open("rb") as f:
                    image_binary = f.read()
                frame_ts = int(sd["timestamp"])
                camera_writer.store_frame(
                    image_binary_data=image_binary,
                    image_format="jpeg",
                    frame_timestamps_us=np.array([frame_ts, frame_ts], dtype=np.uint64),
                    generic_data={},
                    generic_meta_data={},
                )

    # -------------------------------------------------------------------------
    # Radars
    # -------------------------------------------------------------------------

    def _decode_radars(
        self,
        trucksc,
        scene_record: Dict,
        store_writer: SequenceComponentGroupsWriter,
        poses_writer: PosesComponent.Writer,
        component_groups: ComponentGroupAssignments,
        radar_ids: List[str],
    ) -> None:
        """Decode and store all radar frames.

        TruckScenes radar PCDs carry 7 fields (x, y, z, vrel_x, vrel_y, vrel_z, rcs). The
        radial velocity is the full 3D relative-velocity vector projected onto the ray
        direction (positive = moving away). There is no per-point timestamp, so all
        detections in a frame share the per-frame sample_data timestamp.
        """
        for ncore_id, channel in RADAR_MAP.items():
            if ncore_id not in radar_ids:
                continue

            sweep_tokens = get_sweep_tokens(trucksc, scene_record, channel, self.keyframes_only)
            sweep_data = [trucksc.get("sample_data", t) for t in sweep_tokens]
            if not sweep_data:
                self.logger.warning(f"No data for radar {channel}")
                continue

            self.logger.info(f"Processing radar {ncore_id} ({channel}): {len(sweep_data)} frames")

            calibrated_sensor = trucksc.get("calibrated_sensor", sweep_data[0]["calibrated_sensor_token"])
            poses_writer.store_static_pose(
                source_frame_id=ncore_id, target_frame_id="rig", pose=self._extrinsic_matrix(calibrated_sensor)
            )

            radar_writer = store_writer.register_component_writer(
                RadarSensorComponent.Writer,
                component_instance_name=ncore_id,
                group_name=component_groups.radar_component_groups.get(ncore_id),
                generic_meta_data={"channel": channel},
            )

            for sd in tqdm.tqdm(sweep_data, desc=f"Process {ncore_id}"):
                pc = RadarPointCloud.from_file(str(UPath(str(self.root_dir)) / sd["filename"]))
                pts = pc.points.T  # [N, 7]
                if len(pts) == 0:
                    continue

                xyz = pts[:, 0:3].astype(np.float32)
                velocity_vec = pts[:, 3:6].astype(np.float32)
                rcs = pts[:, 6].astype(np.float32)

                distance_m = np.linalg.norm(xyz, axis=1).astype(np.float32)
                valid = distance_m > 0.1
                if not valid.any():
                    continue
                xyz, velocity_vec, rcs, distance_m = xyz[valid], velocity_vec[valid], rcs[valid], distance_m[valid]

                direction = (xyz / distance_m[:, np.newaxis]).astype(np.float32)
                radial_velocity = np.sum(velocity_vec * direction, axis=1).astype(np.float32)

                frame_ts = int(sd["timestamp"])
                radar_writer.store_frame(
                    direction=direction,
                    timestamp_us=np.full(len(xyz), frame_ts, dtype=np.uint64),
                    distance_m=distance_m.reshape(1, -1),
                    frame_timestamps_us=np.array([frame_ts, frame_ts], dtype=np.uint64),
                    generic_data={"radial_velocity_m_s": radial_velocity, "rcs": rcs},
                    generic_meta_data={},
                )

    # -------------------------------------------------------------------------
    # Cuboid annotations
    # -------------------------------------------------------------------------

    def _decode_cuboids(
        self,
        trucksc,
        scene_record: Dict,
        store_writer: SequenceComponentGroupsWriter,
        component_groups: ComponentGroupAssignments,
    ) -> None:
        """Decode TruckScenes 3D annotations (keyframes) as global-frame cuboid observations."""
        cuboid_observations: List[CuboidTrackObservation] = []

        for sample in tqdm.tqdm(get_sample_records(trucksc, scene_record), desc="Process cuboids"):
            timestamp_us = int(sample["timestamp"])
            for ann_token in sample["anns"]:
                record = trucksc.get("sample_annotation", ann_token)
                category_name = record["category_name"]

                if category_name not in TRUCKSCENES_CATEGORY_MAP:
                    continue
                if category_name == EGO_TRAILER_CATEGORY and not self.keep_ego_trailer:
                    continue

                # sample_annotation: translation=[x,y,z] (global/UTM), size=[w,l,h], rotation quat.
                # BBox3 expects [cx,cy,cz, size_x(length), size_y(width), size_z(height), rx, ry, rz].
                width, length, height = record["size"]
                yaw = Quaternion(record["rotation"]).yaw_pitch_roll[0]
                bbox3 = BBox3.from_array(
                    np.array(
                        [
                            record["translation"][0],
                            record["translation"][1],
                            record["translation"][2],
                            length,
                            width,
                            height,
                            0.0,
                            0.0,
                            yaw,
                        ],
                        dtype=np.float32,
                    )
                )

                cuboid_observations.append(
                    CuboidTrackObservation(
                        track_id=record["instance_token"],
                        class_id=TRUCKSCENES_CATEGORY_MAP[category_name],
                        timestamp_us=timestamp_us,
                        reference_frame_id="world_global",
                        reference_frame_timestamp_us=timestamp_us,
                        bbox3=bbox3,
                        source=LabelSource.EXTERNAL,
                    )
                )

        if cuboid_observations:
            store_writer.register_component_writer(
                CuboidsComponent.Writer,
                "default",
                component_groups.cuboid_track_observations_component_group,
            ).store_observations(cuboid_observations)
            self.logger.info(f"Stored {len(cuboid_observations)} cuboid observations")
        else:
            self.logger.info("No cuboid annotations found")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


@cli.command(name="man-truckscenes-v4")
@click.option(
    "--version",
    "truckscenes_version",
    type=str,
    default="v1.2-trainval",
    show_default=True,
    help="MAN TruckScenes dataset version (e.g. v1.2-trainval, v1.2-mini, v1.2-test)",
)
@click.option(
    "--scene-token",
    type=str,
    default=None,
    help="Convert only the scene with this token (mutually exclusive with --scene-name)",
)
@click.option(
    "--scene-name",
    type=str,
    default=None,
    help="Convert only the scene with this name (mutually exclusive with --scene-token)",
)
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
    "store_sequence_meta",
    "--sequence-meta/--no-sequence-meta",
    default=True,
    help="Generate sequence meta-data JSON?",
)
@click.option(
    "keyframes_only",
    "--keyframes-only/--all-sweeps",
    default=False,
    show_default=True,
    help="Convert only annotated keyframes (~2 Hz) instead of the full sweep cadence.",
)
@click.option(
    "keep_ego_trailer",
    "--keep-ego-trailer/--drop-ego-trailer",
    default=True,
    show_default=True,
    help="Keep the recording vehicle's own trailer (vehicle.ego_trailer) as a 'trailer' cuboid.",
)
@click.pass_context
def man_truckscenes_v4(ctx, truckscenes_version, scene_token, scene_name, **kwargs):
    """MAN TruckScenes data conversion (V4 format)"""

    config = ManTruckScenesConverter4Config(
        **{
            **vars(ctx.obj),
            "version": truckscenes_version,
            "scene_token": scene_token,
            "scene_name": scene_name,
            **kwargs,
        }
    )

    ManTruckScenesConverter4.convert(config)
