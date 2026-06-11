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

"""MAN TruckScenes-specific utilities for the NCore V4 converter.

MAN TruckScenes follows the nuScenes schema (its devkit is a fork of the nuScenes
devkit) but with a truck-specific sensor suite: 6 LiDARs, 6 radars, 4 cameras, plus
two IMUs that back the dual-body (cabin vs chassis) ego-motion tables. See the
sibling ``tools/data_converter/nuscenes`` converter for the reference structure.
"""

from __future__ import annotations

from functools import cache
from typing import Any, Dict, List, Optional

from truckscenes import TruckScenes


# --- Sensor ID mappings --------------------------------------------------------
# Mapping from NCore sensor ID (lowercase) -> TruckScenes channel name.

CAMERA_MAP: Dict[str, str] = {
    "camera_left_front": "CAMERA_LEFT_FRONT",
    "camera_right_front": "CAMERA_RIGHT_FRONT",
    "camera_left_back": "CAMERA_LEFT_BACK",
    "camera_right_back": "CAMERA_RIGHT_BACK",
}

LIDAR_MAP: Dict[str, str] = {
    "lidar_top_front": "LIDAR_TOP_FRONT",
    "lidar_top_left": "LIDAR_TOP_LEFT",
    "lidar_top_right": "LIDAR_TOP_RIGHT",
    "lidar_left": "LIDAR_LEFT",
    "lidar_right": "LIDAR_RIGHT",
    "lidar_rear": "LIDAR_REAR",
}

# Reference lidar whose sweep timeline drives the master pose / sequence timeline.
# LIDAR_LEFT is the devkit's reference channel (eval/visualization use it as ref_chan)
# and is one of the dense side lidars, so it is well sampled in every scene.
REF_LIDAR_ID = "lidar_left"
REF_LIDAR_CHANNEL = "LIDAR_LEFT"

RADAR_MAP: Dict[str, str] = {
    "radar_left_front": "RADAR_LEFT_FRONT",
    "radar_right_front": "RADAR_RIGHT_FRONT",
    "radar_left_back": "RADAR_LEFT_BACK",
    "radar_right_back": "RADAR_RIGHT_BACK",
    "radar_left_side": "RADAR_LEFT_SIDE",
    "radar_right_side": "RADAR_RIGHT_SIDE",
}

# TruckScenes raw category name -> NCore class_id.
# Derived from the devkit detection mapping (eval/detection/utils.py + constants.py):
# the 12 detection classes car/truck/bus/trailer/other_vehicle/pedestrian/motorcycle/
# bicycle/traffic_cone/barrier/animal/traffic_sign. Raw categories absent from this map
# (e.g. emergency vehicles, bicycle racks, strollers, trains, debris) are dropped.
# Deviation from the devkit taxonomy: the recording vehicle's own rigidly-attached
# trailer (vehicle.ego_trailer) is kept as a DISTINCT 'ego_trailer' class rather than
# folded into 'trailer', so the ego trailer can be told apart from other vehicles'
# trailers. Whether it is emitted at all is gated by the --keep-ego-trailer flag.
TRUCKSCENES_CATEGORY_MAP: Dict[str, str] = {
    "animal": "animal",
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.construction": "other_vehicle",
    "vehicle.other": "other_vehicle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "static_object.traffic_sign": "traffic_sign",
    "vehicle.ego_trailer": "ego_trailer",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}

# The recording vehicle's own trailer (rigidly attached, always present). Whether to
# keep or drop it is controlled by a converter flag; see TRUCKSCENES_CATEGORY_MAP.
EGO_TRAILER_CATEGORY = "vehicle.ego_trailer"


# --- TruckScenes DB helpers ----------------------------------------------------


@cache
def get_truckscenes(version: str, dataroot: str) -> TruckScenes:
    """Cached TruckScenes DB loader to avoid reloading for multiple scenes."""
    return TruckScenes(version=version, dataroot=dataroot, verbose=False)


def get_sweep_tokens(
    trucksc: TruckScenes,
    scene_record: Dict[str, Any],
    channel: str,
    keyframes_only: bool = False,
) -> List[str]:
    """Return ordered sample_data tokens for a channel across an entire scene.

    Walks the per-sensor sample_data linked list starting from the first keyframe.
    Includes both keyframe and non-keyframe (sweep) frames unless ``keyframes_only``.
    Returns an empty list if the channel is absent from the scene.
    """
    result: List[str] = []
    sample_record = trucksc.get("sample", scene_record["first_sample_token"])
    if channel not in sample_record.get("data", {}):
        return result

    sweep_token = sample_record["data"][channel]
    while sweep_token:
        sample_data_record = trucksc.get("sample_data", sweep_token)
        if (not keyframes_only) or sample_data_record["is_key_frame"]:
            result.append(sweep_token)
        sweep_token = sample_data_record["next"]

    return result


def get_sample_records(trucksc: TruckScenes, scene_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return ordered list of keyframe sample records for a scene."""
    result: List[Dict[str, Any]] = []
    sample_token = scene_record["first_sample_token"]

    while sample_token:
        sample_record = trucksc.get("sample", sample_token)
        result.append(sample_record)
        sample_token = sample_record["next"]

    return result


def resolve_scene_token(trucksc: TruckScenes, scene_token: Optional[str], scene_name: Optional[str]) -> Optional[str]:
    """Resolve a scene identifier to its token.

    If scene_token is provided, validate it exists and return it.
    If scene_name is provided, look it up and return its token.
    If neither provided, return None (meaning: convert all scenes).
    """
    if scene_token is not None and scene_name is not None:
        raise ValueError("Specify at most one of --scene-token or --scene-name, not both.")

    if scene_token is not None:
        all_tokens = {s["token"] for s in trucksc.scene}
        if scene_token not in all_tokens:
            raise ValueError(
                f"Scene token '{scene_token}' not found in dataset. Available: {sorted(all_tokens)[:5]}..."
            )
        return scene_token

    if scene_name is not None:
        name_to_token = {s["name"]: s["token"] for s in trucksc.scene}
        if scene_name not in name_to_token:
            raise ValueError(f"Scene name '{scene_name}' not found. Available: {sorted(name_to_token.keys())[:5]}...")
        return name_to_token[scene_name]

    return None
