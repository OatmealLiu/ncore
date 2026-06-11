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

"""MAN TruckDrive-specific utilities for the NCore V4 converter.

TruckDrive (Torc Robotics / Princeton) is a long-range highway truck dataset with a
fully custom, file-per-frame layout (one directory per scene; per-sensor subdirs of
``<sync>_<timestamp>.<ext>`` files). This module re-implements the small set of
primitives the converter needs from the vendored devkit (``truckdrive-devkit``) so the
converter depends only on numpy + pyquaternion (no mmdet/open3d at runtime):

  * the 9-column ``gt_trajectory.txt`` ego-pose parser,
  * the tf-tree transform graph + BFS (``calib_tf_tree_full.json``),
  * the typed ``.bin`` point-cloud readers (Ouster / Aeva / Continental),
  * the gzipped-numpy depth reader,
  * the metainfo class mapping.

See the sibling ``tools/data_converter/man_truckscenes`` converter for the reference style.
"""

from __future__ import annotations

import gzip
import io
import json
import re

from collections import defaultdict, deque
from functools import cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pyquaternion import Quaternion
from upath import UPath


# --- Scene-relative file/dir conventions ---------------------------------------

TF_TREE_FILE = "calib_tf_tree_full.json"
TRAJECTORY_FILE = "gt_trajectory.txt"
BOUNDING_BOXES_DIR = "bounding_boxes"
RIG_NODE = "vehicle"  # the devkit ego/vehicle frame == NCore "rig"
ANNOTATION_NODE = "velodyne"  # frame the 3D bounding boxes are expressed in

# <sync>_<normalized-ns-timestamp> stem, e.g. "0061_3072067608".
_STEM_RE = re.compile(r"^(?P<sync>\d+)_(?P<ts>\d+)$")


# --- Sensor maps (NCore id -> on-disk + tf-tree info) --------------------------

# 11 RGB camera positions present on disk (== NCore camera id). Lens type (narrow/
# medium/wide) is encoded in the name; the projection model is selected per-camera from
# the calib file's distortion_model (plumb_bob -> pinhole, equidistant -> fisheye).
CAMERA_POSITIONS: Tuple[str, ...] = (
    "forward_center_medium",
    "forward_left_narrow",
    "forward_left_wide",
    "forward_right_narrow",
    "forward_right_wide",
    "rearward_left_bottom_medium",
    "rearward_right_bottom_medium",
    "sideward_left_back_wide",
    "sideward_left_front_wide",
    "sideward_right_back_wide",
    "sideward_right_front_wide",
)

# NCore lidar id -> {dir (relative to scene), tf node, dtype, cols, kind}
LIDAR_SENSORS: Dict[str, Dict] = {
    "ouster_forward_center": {
        "dir": "ouster/forward_center/points",
        "node": "lidar_ouster_forward_center",
        "dtype": np.float32,
        "cols": 7,
        "kind": "ouster",
    },
    "ouster_sideward_left": {
        "dir": "ouster/sideward_left/points",
        "node": "lidar_ouster_sideward_left",
        "dtype": np.float32,
        "cols": 7,
        "kind": "ouster",
    },
    "ouster_sideward_right": {
        "dir": "ouster/sideward_right/points",
        "node": "lidar_ouster_sideward_right",
        "dtype": np.float32,
        "cols": 7,
        "kind": "ouster",
    },
    "aeva": {
        "dir": "aeva/joint_lidars/points",
        "node": "lidar_aeva_forward_center_wide",
        "dtype": np.float64,
        "cols": 11,
        "kind": "aeva",
    },
}

# NCore radar id -> {dir, tf node, dtype, cols}
RADAR_SENSORS: Dict[str, Dict] = {
    "conti542": {
        "dir": "conti542/joint_radars/detections",
        "node": "radar_conti542_forward_left_high",
        "dtype": np.float64,
        "cols": 33,
    },
}


def camera_tf_node(position: str) -> str:
    return f"camera_leopard_{position}"


def camera_images_dir(position: str) -> str:
    return f"leopard/{position}/images"


def camera_calib_file(position: str) -> str:
    return f"calib_camera_leopard_{position}.json"


def camera_depth_dir(position: str) -> str:
    # Accumulated GT depth lives at the scene root under the bare position name.
    return position


# --- Scene / frame enumeration -------------------------------------------------


def list_scene_dirs(root: str, pattern: str = "scene_*") -> List[str]:
    """Return sorted scene directory names matching the glob pattern."""
    return sorted(p.name for p in Path(root).glob(pattern) if p.is_dir())


def parse_stem(name: str) -> Optional[Tuple[int, int]]:
    """Parse '<sync>_<ts>' from a filename, returning (sync_key, timestamp_ns) or None."""
    stem = name.split(".")[0]
    m = _STEM_RE.match(stem)
    if not m:
        return None
    return int(m.group("sync")), int(m.group("ts"))


def list_frames(dir_path: UPath, suffix: str) -> List[Tuple[int, int, UPath]]:
    """List (sync_key, timestamp_ns, path) for files in dir_path ending in suffix, time-sorted."""
    if not dir_path.is_dir():
        return []
    frames: List[Tuple[int, int, UPath]] = []
    for path in dir_path.iterdir():
        if not path.name.endswith(suffix):
            continue
        parsed = parse_stem(path.name)
        if parsed is not None:
            frames.append((parsed[0], parsed[1], path))
    return sorted(frames, key=lambda x: x[1])


# --- Ego trajectory ------------------------------------------------------------


def load_gt_trajectory(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse gt_trajectory.txt -> (timestamps_us [N] uint64, T_rig_world [N,4,4] float64).

    Columns (whitespace-delimited, one header line): SYNC_KEY, TIMESTAMP (seconds),
    X, Y, Z, R_X, R_Y, R_Z, R_W (quaternion, xyzw). The trajectory is scene-local
    (starts near the origin). Header rows (non-integer first token) are skipped.
    """
    timestamps_us: List[int] = []
    poses: List[np.ndarray] = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                int(parts[0])  # SYNC_KEY; skip the header line where this fails
            except ValueError:
                continue
            t_sec = float(parts[1])
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            qx, qy, qz, qw = float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = Quaternion(w=qw, x=qx, y=qy, z=qz).rotation_matrix
            transform[:3, 3] = (x, y, z)
            timestamps_us.append(int(round(t_sec * 1e6)))
            poses.append(transform)
    if not poses:
        raise ValueError(f"No trajectory rows parsed from {path}")
    return np.array(timestamps_us, dtype=np.uint64), np.stack(poses)


# --- tf-tree transform graph (ported from devkit load_utils.py) ----------------


def _dict_to_4x4(transform: Dict) -> np.ndarray:
    """ROS TransformStamped 'transform' dict -> 4x4 (maps child-frame points into parent)."""
    rotation = transform["rotation"]
    translation = transform["translation"]
    quaternion = Quaternion(w=rotation["w"], x=rotation["x"], y=rotation["y"], z=rotation["z"])
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion.rotation_matrix
    matrix[:3, 3] = (translation["x"], translation["y"], translation["z"])
    return matrix


def build_transform_graph(tf_tree: Dict) -> Dict[str, Dict[str, np.ndarray]]:
    """Build graph[A][B] = 4x4 transform mapping a point from frame A to frame B."""
    graph: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
    for entry in tf_tree.values():
        parent = entry["header"]["frame_id"]
        child = entry["child_frame_id"]
        t_child_to_parent = _dict_to_4x4(entry["transform"])
        graph[child][parent] = t_child_to_parent
        graph[parent][child] = np.linalg.inv(t_child_to_parent)
    return graph


def find_transform(graph: Dict[str, Dict[str, np.ndarray]], src_node: str, tgt_node: str) -> np.ndarray:
    """BFS the transform graph and return T mapping a point from src_node to tgt_node."""
    if src_node not in graph:
        raise ValueError(f"Source frame '{src_node}' not in tf tree.")
    if tgt_node not in graph:
        raise ValueError(f"Target frame '{tgt_node}' not in tf tree.")
    visited = set()
    queue = deque([(src_node, np.eye(4, dtype=np.float64))])
    while queue:
        frame, t_src_to_frame = queue.popleft()
        if frame == tgt_node:
            return t_src_to_frame
        visited.add(frame)
        for neighbor, t_frame_to_neighbor in graph[frame].items():
            if neighbor not in visited:
                queue.append((neighbor, t_frame_to_neighbor @ t_src_to_frame))
    raise ValueError(f"No transform path from '{src_node}' to '{tgt_node}'.")


def resolve_scene_file(scene_dir: UPath, name: str) -> Optional[UPath]:
    """Locate a scene file at the scene root or a 'calibrations/' subdir (both devkit layouts)."""
    for candidate in (scene_dir / name, scene_dir / "calibrations" / name):
        if candidate.exists():
            return candidate
    return None


def load_tf_tree(scene_dir: UPath) -> Dict:
    path = resolve_scene_file(scene_dir, TF_TREE_FILE)
    if path is None:
        raise FileNotFoundError(f"{TF_TREE_FILE} not found in {scene_dir} (or its calibrations/ subdir).")
    with path.open() as f:
        return json.load(f)


# --- Point-cloud / depth readers (ported from devkit load_utils.py) ------------


def load_point_bin(path: str, dtype, cols: int) -> np.ndarray:
    """Read a flat binary point cloud into [N, cols] of the given dtype."""
    arr = np.fromfile(path, dtype=dtype)
    if arr.size == 0:
        return arr.reshape((0, cols))
    if arr.size % cols != 0:
        raise ValueError(f"{path}: {arr.size} values not divisible by {cols} columns.")
    return arr.reshape((-1, cols))


def load_depth_npy_gz(path: str) -> np.ndarray:
    """Read a gzip-compressed .npy accumulated-depth map -> float32 [H, W] (meters)."""
    with gzip.open(path, "rb") as f:
        return np.load(io.BytesIO(f.read()))


# --- Class mapping (from the vendored devkit metainfo.json) --------------------


@cache
def _metainfo() -> Dict:
    # Do NOT .resolve(): under bazel runfiles (a symlink forest) the data dep is co-located
    # with this module's runfiles copy, so keep the path inside that tree.
    path = Path(__file__).parent / "truckdrive-devkit" / "metainfo.json"
    with open(path) as f:
        return json.load(f)


@cache
def _fine_to_class_name() -> Dict[str, str]:
    """Map each fine label string -> coarse class NAME, dropping ignore (id < 0) labels."""
    info = _metainfo()
    id_to_name = {idx: name for name, idx in info["mapped_categories"].items() if idx >= 0}
    return {fine: id_to_name[idx] for fine, idx in info["label_mapping"].items() if idx >= 0 and idx in id_to_name}


def class_name_for_label(fine_label: str) -> Optional[str]:
    """Coarse class name for a fine label, or None if it maps to ignore / is unknown."""
    return _fine_to_class_name().get(fine_label)
