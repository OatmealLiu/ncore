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

"""Integration tests for the MAN TruckScenes data converter (V4 format).

Requires the MAN_TRUCKSCENES_DIR environment variable pointing to a TruckScenes dataset
root directory (the dir containing ``v1.2-trainval/`` plus ``samples/`` and ``sweeps/``).
Set MAN_TRUCKSCENES_VERSION to override the default version. The test converts a single
scene (keyframes only) for speed.
"""

import os
import tempfile
import unittest

from typing import Literal, cast

from parameterized import parameterized_class
from upath import UPath

from ncore.impl.data.types import OpenCVPinholeCameraModelParameters
from ncore.impl.data.v4.components import (
    CameraSensorComponent,
    CuboidsComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    PosesComponent,
    RadarSensorComponent,
    SequenceComponentGroupsReader,
)
from tools.data_converter.man_truckscenes.converter import ManTruckScenesConverter4, ManTruckScenesConverter4Config
from tools.data_converter.man_truckscenes.utils import CAMERA_MAP, LIDAR_MAP, RADAR_MAP, get_truckscenes


@parameterized_class(
    ("store_type",),
    [
        ("itar",),
        ("directory",),
    ],
)
class TestManTruckScenesConverter(unittest.TestCase):
    """Integration tests for the MAN TruckScenes data converter.

    Requires MAN_TRUCKSCENES_DIR. Uses the first scene in the dataset (keyframes only).
    """

    store_type: Literal["itar", "directory"]

    @classmethod
    def setUpClass(cls):
        cls.dataset_dir = os.environ.get("MAN_TRUCKSCENES_DIR")
        if cls.dataset_dir is None:
            raise unittest.SkipTest("MAN_TRUCKSCENES_DIR not set -- skipping TruckScenes integration tests")

        cls.version = os.environ.get("MAN_TRUCKSCENES_VERSION", "v1.2-trainval")

        cls._tempdir = tempfile.TemporaryDirectory(prefix="man_truckscenes_test_")
        cls.output_dir = cls._tempdir.name

        trucksc = get_truckscenes(version=cls.version, dataroot=cls.dataset_dir)
        cls.scene_token = trucksc.scene[0]["token"]
        cls.scene_name = trucksc.scene[0]["name"]

        config = ManTruckScenesConverter4Config(
            root_dir=cls.dataset_dir,
            output_dir=cls.output_dir,
            no_cameras=False,
            camera_ids=None,
            no_lidars=False,
            lidar_ids=None,
            no_radars=False,
            radar_ids=None,
            verbose=False,
            debug=False,
            debug_port=5678,
            version=cls.version,
            scene_token=cls.scene_token,
            scene_name=None,
            store_type=cls.store_type,
            component_group_profile="separate-sensors",
            store_sequence_meta=True,
            keyframes_only=True,
        )
        ManTruckScenesConverter4.convert(config)

        seq_dirs = [d for d in UPath(cls.output_dir).iterdir() if d.is_dir()]
        assert len(seq_dirs) == 1, f"Expected 1 sequence dir, found {len(seq_dirs)}: {seq_dirs}"
        cls.seq_dir = seq_dirs[0]

        meta_files = list(cls.seq_dir.glob("*.json"))
        assert len(meta_files) == 1, f"Expected 1 meta JSON, found {len(meta_files)}"
        cls.reader = SequenceComponentGroupsReader([meta_files[0]])

    @classmethod
    def tearDownClass(cls):
        cls._tempdir.cleanup()

    # --- Poses ----------------------------------------------------------------

    def test_sequence_has_dynamic_rig_to_world_pose(self):
        poses_reader = list(self.reader.open_component_readers(PosesComponent.Reader).values())[0]
        poses, timestamps = poses_reader.get_dynamic_pose("rig", "world")
        self.assertEqual(poses.shape[1:], (4, 4))
        self.assertGreater(poses.shape[0], 0)
        self.assertEqual(timestamps.shape[0], poses.shape[0])

    def test_sequence_has_static_world_to_world_global(self):
        poses_reader = list(self.reader.open_component_readers(PosesComponent.Reader).values())[0]
        static_poses = dict(poses_reader.get_static_poses())
        self.assertIn(("world", "world_global"), static_poses)
        self.assertEqual(static_poses[("world", "world_global")].shape, (4, 4))

    # --- Cameras --------------------------------------------------------------

    def test_all_cameras_exist(self):
        camera_readers = self.reader.open_component_readers(CameraSensorComponent.Reader)
        self.assertEqual(set(camera_readers.keys()), set(CAMERA_MAP.keys()))
        for cam_id, cam_reader in camera_readers.items():
            self.assertGreater(cam_reader.frames_count, 0, f"{cam_id} should have frames")

    def test_camera_intrinsics_zero_distortion(self):
        intrinsics_reader = list(self.reader.open_component_readers(IntrinsicsComponent.Reader).values())[0]
        for cam_id in CAMERA_MAP:
            params = intrinsics_reader.get_camera_model_parameters(cam_id)
            self.assertIsInstance(params, OpenCVPinholeCameraModelParameters)
            params = cast(OpenCVPinholeCameraModelParameters, params)
            self.assertTrue((params.radial_coeffs == 0).all())
            self.assertTrue((params.tangential_coeffs == 0).all())
            self.assertTrue((params.focal_length > 0).all())

    def test_camera_extrinsics_stored_as_static_poses(self):
        poses_reader = list(self.reader.open_component_readers(PosesComponent.Reader).values())[0]
        static_poses = dict(poses_reader.get_static_poses())
        for cam_id in CAMERA_MAP:
            self.assertIn((cam_id, "rig"), static_poses, f"Missing static pose for {cam_id}")

    # --- Lidars ---------------------------------------------------------------

    def test_all_lidars_exist(self):
        lidar_readers = self.reader.open_component_readers(LidarSensorComponent.Reader)
        self.assertEqual(set(lidar_readers.keys()), set(LIDAR_MAP.keys()))
        for lidar_id, lidar_reader in lidar_readers.items():
            self.assertGreater(lidar_reader.frames_count, 0, f"{lidar_id} should have frames")

    def test_lidar_extrinsics_stored_as_static_poses(self):
        poses_reader = list(self.reader.open_component_readers(PosesComponent.Reader).values())[0]
        static_poses = dict(poses_reader.get_static_poses())
        for lidar_id in LIDAR_MAP:
            self.assertIn((lidar_id, "rig"), static_poses, f"Missing static pose for {lidar_id}")

    def test_lidars_have_no_structured_model(self):
        """The 6-lidar rig is stored as raw ray bundles, so no lidar intrinsics are written."""
        intrinsics_reader = list(self.reader.open_component_readers(IntrinsicsComponent.Reader).values())[0]
        for lidar_id in LIDAR_MAP:
            self.assertIsNone(intrinsics_reader.get_lidar_model_parameters(lidar_id))

    # --- Radars ---------------------------------------------------------------

    def test_all_radars_exist(self):
        radar_readers = self.reader.open_component_readers(RadarSensorComponent.Reader)
        self.assertEqual(set(radar_readers.keys()), set(RADAR_MAP.keys()))
        for radar_id, radar_reader in radar_readers.items():
            self.assertGreater(radar_reader.frames_count, 0, f"{radar_id} should have frames")

    def test_radar_extrinsics_stored_as_static_poses(self):
        poses_reader = list(self.reader.open_component_readers(PosesComponent.Reader).values())[0]
        static_poses = dict(poses_reader.get_static_poses())
        for radar_id in RADAR_MAP:
            self.assertIn((radar_id, "rig"), static_poses, f"Missing static pose for {radar_id}")

    # --- Cuboids --------------------------------------------------------------

    def test_cuboid_observations_exist(self):
        cuboid_readers = self.reader.open_component_readers(CuboidsComponent.Reader)
        if not cuboid_readers:
            self.skipTest("No cuboid component (possibly an unannotated split)")
        cuboid_reader = list(cuboid_readers.values())[0]
        observations = list(cuboid_reader.get_observations())
        self.assertGreater(len(observations), 0)
        obs = observations[0]
        self.assertIsInstance(obs.track_id, str)
        self.assertIsInstance(obs.class_id, str)
        self.assertEqual(obs.reference_frame_id, "world_global")

    # --- Sequence meta --------------------------------------------------------

    def test_sequence_id_matches_scene_name(self):
        self.assertEqual(self.reader.sequence_id, self.scene_name)


if __name__ == "__main__":
    unittest.main()
