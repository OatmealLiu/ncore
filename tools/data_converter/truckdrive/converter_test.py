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

"""Integration tests for the TruckDrive data converter (V4 format).

Requires the TRUCKDRIVE_DIR environment variable pointing to the dataset root (the dir
containing the scene_* directories). Converts a single scene (the first by default, or
TRUCKDRIVE_SCENE).
"""

import os
import tempfile
import unittest

from typing import Literal

from upath import UPath

from ncore.impl.data.types import OpenCVFisheyeCameraModelParameters, OpenCVPinholeCameraModelParameters
from ncore.impl.data.v4.components import (
    CameraSensorComponent,
    CuboidsComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    PosesComponent,
    RadarSensorComponent,
    SequenceComponentGroupsReader,
)
from tools.data_converter.truckdrive import utils
from tools.data_converter.truckdrive.converter import TruckDriveConverter4, TruckDriveConverter4Config


class TestTruckDriveConverter(unittest.TestCase):
    """Integration test for the TruckDrive data converter (uses one scene)."""

    store_type: Literal["itar", "directory"] = "itar"

    @classmethod
    def setUpClass(cls):
        cls.dataset_dir = os.environ.get("TRUCKDRIVE_DIR")
        if cls.dataset_dir is None:
            raise unittest.SkipTest("TRUCKDRIVE_DIR not set -- skipping TruckDrive integration tests")

        scenes = utils.list_scene_dirs(cls.dataset_dir, "scene_*")
        if not scenes:
            raise unittest.SkipTest(f"No scene_* dirs under {cls.dataset_dir}")
        cls.scene_name = os.environ.get("TRUCKDRIVE_SCENE", scenes[0])

        cls._tempdir = tempfile.TemporaryDirectory(prefix="truckdrive_test_")
        cls.output_dir = cls._tempdir.name

        config = TruckDriveConverter4Config(
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
            scene_name=cls.scene_name,
            store_type=cls.store_type,
            component_group_profile="separate-sensors",
            store_sequence_meta=True,
        )
        TruckDriveConverter4.convert(config)

        seq_dirs = [d for d in UPath(cls.output_dir).iterdir() if d.is_dir()]
        assert len(seq_dirs) == 1, f"Expected 1 sequence dir, found {len(seq_dirs)}"
        cls.seq_dir = seq_dirs[0]
        meta_files = list(cls.seq_dir.glob("*.json"))
        assert len(meta_files) == 1, f"Expected 1 meta JSON, found {len(meta_files)}"
        cls.reader = SequenceComponentGroupsReader([meta_files[0]])

    @classmethod
    def tearDownClass(cls):
        cls._tempdir.cleanup()

    def _poses_reader(self):
        return list(self.reader.open_component_readers(PosesComponent.Reader).values())[0]

    def test_dynamic_and_anchor_poses(self):
        poses, timestamps = self._poses_reader().get_dynamic_pose("rig", "world")
        self.assertEqual(poses.shape[1:], (4, 4))
        self.assertGreater(poses.shape[0], 0)
        self.assertEqual(timestamps.shape[0], poses.shape[0])
        self.assertIn(("world", "world_global"), dict(self._poses_reader().get_static_poses()))

    def test_cameras(self):
        camera_readers = self.reader.open_component_readers(CameraSensorComponent.Reader)
        self.assertGreater(len(camera_readers), 0)
        intrinsics_reader = list(self.reader.open_component_readers(IntrinsicsComponent.Reader).values())[0]
        static_poses = dict(self._poses_reader().get_static_poses())
        for cam_id, cam_reader in camera_readers.items():
            self.assertGreater(cam_reader.frames_count, 0)
            params = intrinsics_reader.get_camera_model_parameters(cam_id)
            self.assertIsInstance(params, (OpenCVPinholeCameraModelParameters, OpenCVFisheyeCameraModelParameters))
            self.assertIn((cam_id, "rig"), static_poses)

    def test_lidars_raybundle_no_model(self):
        lidar_readers = self.reader.open_component_readers(LidarSensorComponent.Reader)
        self.assertGreater(len(lidar_readers), 0)
        intrinsics_reader = list(self.reader.open_component_readers(IntrinsicsComponent.Reader).values())[0]
        static_poses = dict(self._poses_reader().get_static_poses())
        for lidar_id, lidar_reader in lidar_readers.items():
            self.assertGreater(lidar_reader.frames_count, 0)
            self.assertIsNone(intrinsics_reader.get_lidar_model_parameters(lidar_id))
            self.assertIn((lidar_id, "rig"), static_poses)

    def test_radars(self):
        radar_readers = self.reader.open_component_readers(RadarSensorComponent.Reader)
        static_poses = dict(self._poses_reader().get_static_poses())
        for radar_id, radar_reader in radar_readers.items():
            self.assertGreater(radar_reader.frames_count, 0)
            self.assertIn((radar_id, "rig"), static_poses)

    def test_cuboids_in_velodyne_frame(self):
        cuboid_readers = self.reader.open_component_readers(CuboidsComponent.Reader)
        if not cuboid_readers:
            self.skipTest("No cuboids in this scene")
        observations = list(list(cuboid_readers.values())[0].get_observations())
        self.assertGreater(len(observations), 0)
        self.assertEqual(observations[0].reference_frame_id, "velodyne")
        # the velodyne annotation frame must have a static extrinsic to rig
        self.assertIn(("velodyne", "rig"), dict(self._poses_reader().get_static_poses()))

    def test_sequence_id_matches_scene(self):
        self.assertEqual(self.reader.sequence_id, self.scene_name)


if __name__ == "__main__":
    unittest.main()
