import os
import numpy as np
import json
from pathlib import Path
from pyquaternion import Quaternion
from collections import defaultdict, deque
from vis_utils.dataset_details import camera_setup, sensor_path
from vis_utils.transforms import (
    transform_points_4x4,
    match_timestamps_one_to_one,
    lidar_timestamps_to_camera,
)
import open3d as o3d
from numpy import cos, sin
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import re

import math

_SYNCED_STEM_RE = re.compile(r"^(?P<sync>\d+)_(?P<ts>\d+)$", re.ASCII)
_LEGACY_TS_STEM_RE = re.compile(r"^(?P<ts>\d+)$", re.ASCII)
_ID_SUFFIX_RE = re.compile(r"_ID\d+$", re.IGNORECASE)
DEPTH_FILENAME_RE = re.compile(
    r"^(?P<sync>\d+)_(?P<ts>\d+)\.npy(?:\.gz)?$",
    re.IGNORECASE,
)


def load_depth_files_with_sync_from_folder(folder_path: str):
    """
    Load accumulated GT depth filenames.

    Supports both:
      <sync>_<timestamp>.npy
      <sync>_<timestamp>.npy.gz

    Returns:
      files, timestamps, sync_ids
    """
    if not os.path.isdir(folder_path):
        return [], [], []

    files = []
    timestamps = []
    sync_ids = []

    for fn in os.listdir(folder_path):
        m = DEPTH_FILENAME_RE.match(fn)
        if not m:
            continue

        files.append(fn)
        sync_ids.append(int(m.group("sync")))
        timestamps.append(int(m.group("ts")))

    order = sorted(
        range(len(files)),
        key=lambda i: (sync_ids[i], timestamps[i], files[i]),
    )

    files = [files[i] for i in order]
    timestamps = [timestamps[i] for i in order]
    sync_ids = [sync_ids[i] for i in order]

    return files, timestamps, sync_ids


def strip_sensor_id(name: str) -> str:
    return _ID_SUFFIX_RE.sub("", str(name or ""))


def is_nan_like(value: Any) -> bool:
    """True for float NaN and string NaN sentinels that appear in JSONs."""
    if value is None:
        return False
    try:
        if isinstance(value, float) and math.isnan(value):
            return True
    except Exception:
        pass
    if isinstance(value, str) and value.strip().lower() == "nan":
        return True
    return False


def parse_synced_filename(filename: str) -> Tuple[Optional[int], Optional[int], str]:
    """
    Parse dataset file names.

    Returns (sync_id, normalized_timestamp, extension_without_dot).

    Supported stems:
      <sync>_<normalized_timestamp>.<ext>
      <timestamp>.<ext>
    """
    base = os.path.basename(filename)
    stem, ext = os.path.splitext(base)
    ext = ext.lower().lstrip(".")
    m = _SYNCED_STEM_RE.match(stem)
    if m:
        return int(m.group("sync")), int(m.group("ts")), ext
    m = _LEGACY_TS_STEM_RE.match(stem)
    if m:
        return None, int(m.group("ts")), ext
    return None, None, ext


def load_files_with_sync_from_folder(
    folder_path: str,
    extensions: Sequence[str],
) -> Tuple[List[str], List[int], List[Optional[int]]]:
    """
    Load files from a folder and parse synchronized names.

    Returns:
      files, timestamps, sync_ids

    Files are sorted by (sync_id if present else timestamp, timestamp, name).
    Unsupported names are ignored instead of crashing the visualizer.
    """
    if not os.path.isdir(folder_path):
        return [], [], []

    ext_set = {e.lower().lstrip(".") for e in extensions}
    rows: List[Tuple[int, int, str, Optional[int]]] = []
    for fn in os.listdir(folder_path):
        if fn.startswith("."):
            continue
        sync_id, ts, ext = parse_synced_filename(fn)
        if ext not in ext_set or ts is None:
            continue
        sort0 = sync_id if sync_id is not None else ts
        rows.append((sort0, ts, fn, sync_id))

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    files = [r[2] for r in rows]
    timestamps = [int(r[1]) for r in rows]
    sync_ids = [r[3] for r in rows]
    return files, timestamps, sync_ids


def match_files_by_sync_or_time(
    sync1: Sequence[Optional[int]],
    sync2: Sequence[Optional[int]],
    ts1: Sequence[int],
    ts2: Sequence[int],
    max_diff: float,
) -> List[Tuple[int, int]]:
    """
    Match two file lists for the dataset.

    If both sides have synchronized <sync>_<timestamp> names, matching is done
    by sync_id first. This is safer than matching only normalized timestamps,
    because synchronized files can intentionally have different normalized
    timestamps but the same sync key.
    """
    have_sync1 = any(s is not None for s in sync1)
    have_sync2 = any(s is not None for s in sync2)
    if have_sync1 and have_sync2:
        by_sync2: Dict[int, List[int]] = defaultdict(list)
        for j, s2 in enumerate(sync2):
            if s2 is not None:
                by_sync2[int(s2)].append(j)

        used2 = set()
        matches: List[Tuple[int, int]] = []
        for i, s1 in enumerate(sync1):
            if s1 is None:
                continue
            candidates = [j for j in by_sync2.get(int(s1), []) if j not in used2]
            if not candidates:
                continue
            # Pick the nearest normalized timestamp if a sync key has duplicates.
            j = min(candidates, key=lambda jj: abs(int(ts1[i]) - int(ts2[jj])))
            used2.add(j)
            matches.append((i, j))
        matches.sort(key=lambda ij: (ij[1], ij[0]))
        return matches

    return match_timestamps_one_to_one(ts1, ts2, max_diff=max_diff)


def _reshape_binary_scan(file_path: str, dtype: Any, columns: int) -> np.ndarray:
    arr = np.fromfile(file_path, dtype=dtype)
    if arr.size == 0:
        return arr.reshape((0, columns))
    if arr.size % columns != 0:
        raise ValueError(
            f"{file_path} has {arr.size} values, not divisible by expected columns={columns}."
        )
    return arr.reshape((-1, columns))


def dict_to_4x4(tf_dict):
    """
    Convert a 'transform' dict with keys:
       tf_dict["translation"] = {x, y, z}
       tf_dict["rotation"]    = {w, x, y, z}  (unit quaternion in standard form)
    into a 4x4 homogeneous transform matrix.
    """
    q = Quaternion(
        w=tf_dict["rotation"]["w"],
        x=tf_dict["rotation"]["x"],
        y=tf_dict["rotation"]["y"],
        z=tf_dict["rotation"]["z"],
    )
    R = q.rotation_matrix
    t = np.array(
        [
            tf_dict["translation"]["x"],
            tf_dict["translation"]["y"],
            tf_dict["translation"]["z"],
        ]
    )

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def build_graph_of_transforms(data_extrinsics):
    """
    data_extrinsics is a dict of dicts (like your example).
    Returns a dict-of-dict 'graph':
        graph[A][B] = 4x4 transform from frame A -> frame B
    """
    graph = defaultdict(dict)

    for _, val in data_extrinsics.items():
        parent = val["header"]["frame_id"]
        child = val["child_frame_id"]
        T_parent_to_child = dict_to_4x4(val["transform"])

        graph[parent][child] = np.linalg.inv(T_parent_to_child)
        graph[child][parent] = T_parent_to_child

    return graph


def find_transform(graph, src_node, tgt_node):
    """
    Given a graph of frames (from build_graph_of_transforms),
    find a path from src_node -> tgt_node (if it exists),
    and return the 4x4 matrix T_src_to_tgt.
    """
    if src_node not in graph:
        raise ValueError(f"Source frame '{src_node}' not in graph.")
    if tgt_node not in graph:
        raise ValueError(f"Target frame '{tgt_node}' not in graph.")

    visited = set()
    queue = deque()
    # Each queue item: (current_frame, T_src_to_current)
    queue.append((src_node, np.eye(4)))

    while queue:
        current_frame, T_src_to_current = queue.popleft()

        if current_frame == tgt_node:
            return T_src_to_current  # Found the path!

        visited.add(current_frame)

        # Explore neighbors
        for neighbor_frame, T_current_to_neighbor in graph[current_frame].items():
            if neighbor_frame not in visited:
                # Transform from src_node -> neighbor_frame
                T_src_to_neighbor = T_current_to_neighbor @ T_src_to_current
                queue.append((neighbor_frame, T_src_to_neighbor))

    raise ValueError(f"No path from '{src_node}' to '{tgt_node}' found in the graph.")


def load_extrinsic_between_nodes(data_extrinsics, src_node, tgt_node):
    """
    Build a graph from the extrinsic transforms, then find
    a 4x4 homogeneous transform from `src_node` to `tgt_node`.
    """

    # 1) Build the adjacency from the given JSON (list or dict["transforms"])
    graph = build_graph_of_transforms(data_extrinsics)

    # 2) Run BFS/DFS to find T_src_to_tgt
    T_src_to_tgt = find_transform(graph, src_node, tgt_node)

    return T_src_to_tgt


def load_pointclouds_from_folder(folder_path):
    files, timestamps, _sync_ids = load_files_with_sync_from_folder(
        folder_path, ["bin"]
    )
    return files, timestamps


def load_annos_from_folder(folder_path):
    files, timestamps, _sync_ids = load_files_with_sync_from_folder(
        folder_path, ["json"]
    )
    return files, timestamps


def load_images_from_folder(folder_path):
    files, timestamps, _sync_ids = load_files_with_sync_from_folder(
        folder_path, ["jpg", "jpeg", "png"]
    )
    return files, timestamps


def load_radar_joint_point_cloud(file_path):
    """
    Joint radar binary format.

    Source radar files: the expected shape is (-1, 33).
    """
    arr = np.fromfile(file_path, dtype=np.float64)
    if arr.size == 0:
        return arr.reshape((0, 33))
    if arr.size % 33 == 0:
        return arr.reshape((-1, 33))
    raise ValueError(f"{file_path} is not divisible by 33 radar columns")


def load_lidar_joint_point_cloud(file_path):
    """
    Aeva joint lidar format.

    Expected columns (float64, 11):
      x, y, z, intensity, velocity, reflectivity, time_offset_ns,
      sensor_id, vx, vy, vz
    """
    arr = np.fromfile(file_path, dtype=np.float64)
    if arr.size == 0:
        return arr.reshape((0, 11))
    if arr.size % 11 == 0:
        return arr.reshape((-1, 11))
    raise ValueError(f"{file_path} is not divisible by 11 Aeva columns")


def load_single_ouster_lidar_point_cloud(file):
    """
    Ouster lidar format.

    Expected columns (float32, 7):
      x, y, z, intensity, rel_time_ns, reflectivity, ring
    """
    arr = np.fromfile(file, dtype=np.float32)
    if arr.size == 0:
        return arr.reshape((0, 7))
    if arr.size % 7 == 0:
        return arr.reshape((-1, 7))
    raise ValueError(f"{file} is not divisible by 7 Ouster columns")


def inverse_transformation_matrix(rotation_matrix, translation_vector):
    inv_rotation_matrix = np.transpose(rotation_matrix)
    inv_translation_vector = -np.dot(inv_rotation_matrix, translation_vector)

    inv_transformation_matrix = np.eye(4)
    return inv_rotation_matrix, inv_translation_vector


def euler_to_rotation_matrix(rotation):
    roll, pitch, yaw = rotation
    R = np.array(
        [
            [
                cos(yaw) * cos(pitch),
                cos(yaw) * sin(pitch) * sin(roll) - sin(yaw) * cos(roll),
                cos(yaw) * sin(pitch) * cos(roll) + sin(yaw) * sin(roll),
            ],
            [
                sin(yaw) * cos(pitch),
                sin(yaw) * sin(pitch) * sin(roll) + cos(yaw) * cos(roll),
                sin(yaw) * sin(pitch) * cos(roll) - cos(yaw) * sin(roll),
            ],
            [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(roll)],
        ]
    )

    return R


def load_poses(filename):
    poses = []
    with open(filename, "r") as f:
        for line in f:
            data = line.strip().split()
            timestamp = float(data[0])
            x, y, z, roll, pitch, yaw = map(float, data[1:])
            poses.append((timestamp, x, y, z, roll, pitch, yaw))
    return poses


def read_trajectory(filepath: str, read_global_info: bool = False):
    """
    Read a trajectory text file.

    Data rows format per line:
        TIMESTAMP X Y Z R_X R_Y R_Z R_W    # quat is [x, y, z, w]

    First line format when read_global_info is True:
        METHOD_NAME X_GLOBAL Y_GLOBAL Z_GLOBAL R_X_GLOBAL R_Y_GLOBAL R_Z_GLOBAL R_W_GLOBAL

    Returns
    -------
    poses, timestamps                          (if read_global_info == False)
    poses, timestamps, global_info_dict        (if read_global_info == True)

    where:
      poses: List[Tuple[np.ndarray(3,), np.ndarray(3,)]]
             -> (position_xyz, euler_rpy) with radians
      timestamps: List[int]  (nanoseconds)
      global_info_dict: {"method_name", "xyz_global", "rot_global"}  # rot_global is quaternion [x,y,z,w]
    """
    from scipy.spatial.transform import Rotation as R

    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")

    method_name: Optional[str] = None
    xyz_global: Optional[np.ndarray] = None
    rot_global: Optional[np.ndarray] = None

    # Read first line (header) separately
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip()

    if read_global_info:
        parts = [p for p in re.split(r"[,\s]+", header) if p]
        if len(parts) < 8:
            raise ValueError(
                "Header must contain METHOD_NAME and 7 numeric values for globals."
            )
        method_name = parts[0]
        try:
            gvals = np.array([float(x) for x in parts[1:8]], dtype=np.float64)
        except ValueError as e:
            raise ValueError("Global values in header must be numeric.") from e
        xyz_global = gvals[:3]
        rot_global = gvals[3:7]  # quaternion [x,y,z,w]

    # Load numeric rows (skip header line)
    try:
        data = np.loadtxt(str(path), skiprows=1)
    except Exception as e:
        raise ValueError(f"Failed to load numeric rows from {path}: {e}") from e

    if data.ndim == 1:
        data = data[np.newaxis, :]

    if data.shape[1] < 8:
        raise ValueError(
            "Each data row must have at least 8 columns: t x y z qx qy qz qw."
        )

    # Parse columns
    t_sec = data[:, 0].astype(np.float64)
    xyz = data[:, 1:4].astype(np.float64)
    quats = data[:, 4:8].astype(np.float64)  # [x,y,z,w]

    # Quaternion -> Euler (roll, pitch, yaw), radians
    rpy = R.from_quat(quats).as_euler("xyz", degrees=False)  # shape (N,3)

    # Build outputs exactly like load_poses()
    poses: List[Tuple[np.ndarray, np.ndarray]] = [
        (xyz[i].copy(), rpy[i].copy()) for i in range(xyz.shape[0])
    ]
    # timestamps in nanoseconds
    timestamps: List[int] = (t_sec * 1e9).astype(np.int64).tolist()

    if read_global_info:
        global_info: Dict[str, Any] = {
            "method_name": method_name,
            "xyz_global": xyz_global,
            "rot_global": rot_global,  # quaternion [x,y,z,w], unchanged
        }
        return poses, timestamps, global_info

    return poses, timestamps


def _iter_annotation_objects(data: Any) -> List[dict]:
    if isinstance(data, list):
        raw = data
    elif isinstance(data, dict):
        raw = list(data.values())
    else:
        raw = []
    return [obj for obj in raw if isinstance(obj, dict) and not is_nan_like(obj)]


def _safe_tracking_id(obj: dict) -> int:
    val = obj.get("Tracking_ID", obj.get("tracking_id", -1))
    try:
        if is_nan_like(val):
            return -9999
        return int(val)
    except Exception:
        oid = str(obj.get("object-id", obj.get("id", "")) or "")
        try:
            return int(oid.split(":")[-1])
        except Exception:
            return -9999


def load_boxes_from_json(json_path):
    """
    Read cleaned bounding-box annotations.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    boxes = []
    for obj in _iter_annotation_objects(data):
        if "lidar" in obj and isinstance(obj.get("lidar"), dict):
            lidar_info = obj["lidar"]
            tracking_id = _safe_tracking_id(obj)
            box_params = {
                "x": lidar_info.get("x_c", -1000),
                "y": lidar_info.get("y_c", -1000),
                "z": lidar_info.get("z_c", -1000),
                "l": lidar_info.get("l", -1),
                "w": lidar_info.get("w", -1),
                "h": lidar_info.get("h", -1),
                "yaw": lidar_info.get("yaw", -1000),
                "Parent_id": obj.get("Parent_id", -1),
                "hardLabel_3d": obj.get("hardLabel_3d", -1),
                "Tracking_ID": tracking_id,
                "object-id": obj.get("object-id", obj.get("id", "")),
                "class-id": obj.get("class-id", ""),
            }
        else:
            tracking_id = _safe_tracking_id(obj)
            box_params = {
                "x": obj.get("x", obj.get("x_ego", -1000)),
                "y": obj.get("y", obj.get("y_ego", -1000)),
                "z": obj.get("z", obj.get("z_ego", -1000)),
                "l": obj.get("l", obj.get("length", -1)),
                "w": obj.get("w", obj.get("width", -1)),
                "h": obj.get("h", obj.get("height", obj.get("h_ego", -1))),
                "yaw": obj.get("yaw", obj.get("yaw_ego", obj.get("yaw_rad", -1000))),
                "Parent_id": obj.get("Parent_id", -1),
                "hardLabel_3d": obj.get("hardLabel_3d", -1),
                "Tracking_ID": tracking_id,
                "object-id": obj.get("object-id", obj.get("id", "")),
                "class-id": obj.get("class-id", ""),
            }

        cameras: Dict[str, Any] = {}

        for i in range(32):
            cam_obj = obj.get(f"camera-{i}")
            if not isinstance(cam_obj, dict):
                continue
            image_path = cam_obj.get("image_path", "")
            camera_name = cam_obj.get("camera_name")
            if not camera_name and image_path:
                camera_name = "_".join(
                    os.path.basename(image_path).split(".jpg")[0].split("_")[1:]
                )
            camera_name = strip_sensor_id(camera_name or "")
            if not camera_name:
                continue
            cameras[camera_name] = {
                "image_path": image_path,
                "timestamp": cam_obj.get("timestamp", ""),
                "camera_name": camera_name,
                "x0": cam_obj.get("bbox_2d_x0", cam_obj.get("x0", -1)),
                "x1": cam_obj.get("bbox_2d_x1", cam_obj.get("x1", -1)),
                "y0": cam_obj.get("bbox_2d_y0", cam_obj.get("y0", -1)),
                "y1": cam_obj.get("bbox_2d_y1", cam_obj.get("y1", -1)),
                "occlusion": obj.get(
                    f"occlusion_camera_{i}", cam_obj.get("occlusion", "N/A")
                ),
                "truncation": obj.get(
                    f"truncation_camera_{i}", cam_obj.get("truncation", "N/A")
                ),
            }

        for cam_name in camera_setup:
            cam_obj = obj.get(cam_name)
            if isinstance(cam_obj, dict):
                cam_obj = dict(cam_obj)
                cam_obj["camera_name"] = strip_sensor_id(
                    cam_obj.get("camera_name", cam_name)
                )
                cameras[cam_name] = cam_obj

        box_params.update(cameras)
        boxes.append(box_params)

    return boxes


_TRAJECTORY_SKIP_TRACKING_IDS = frozenset({-9999})


def trajectory_track_key_from_box(b: dict) -> Optional[str]:
    """
    Stable key for grouping box centers across frames.
    Returns None if the box should not contribute to trajectories.
    """
    oid = str(b.get("object-id", "") or "").strip()
    if oid.endswith(":-9999"):
        return None
    tid_raw = b.get("Tracking_ID", -(10**9))
    try:
        tid = int(tid_raw)
    except (TypeError, ValueError):
        return None
    if tid in _TRAJECTORY_SKIP_TRACKING_IDS or tid < 0:
        return None
    if oid:
        return oid
    cid = str(b.get("class-id", "") or "").strip()
    if cid:
        return f"{cid}:{tid}"
    return str(tid)


_REF_LIDAR_NODE_VELO2GLOBAL = "lidar_aeva_forward_center_wide"


def velodyne2global_from_check_labels_pose(
    pose_position_xyz: np.ndarray,
    pose_rpy_rad: np.ndarray,
    data_extrinsics: dict,
    anno_ref_key: str = "annotations/bounding_boxes",
) -> np.ndarray:
    rotation_matrix = euler_to_rotation_matrix(pose_rpy_rad)
    a = np.identity(4, dtype=np.float64)
    a[:3, :3] = rotation_matrix
    a[:3, 3] = np.asarray(pose_position_xyz, dtype=np.float64).reshape(3)
    T_velodyne_to_lidar = load_extrinsic_between_nodes(
        data_extrinsics,
        sensor_path[anno_ref_key],
        _REF_LIDAR_NODE_VELO2GLOBAL,
    )
    return a @ T_velodyne_to_lidar


def _nearest_pose_index(
    ts_query: int, ts_pose_ns: List[int], max_diff_ns: float = 50e6
) -> Optional[int]:
    best_j: Optional[int] = None
    best_d = float("inf")
    for j, t in enumerate(ts_pose_ns):
        d = abs(int(t) - int(ts_query))
        if d < best_d:
            best_d = d
            best_j = j
    if best_j is None or best_d > max_diff_ns:
        return None
    return best_j


def load_boxes_sequence(
    annos_folder: str, annos_files: Sequence[str]
) -> List[List[dict]]:
    """
    Load ``load_boxes_from_json`` for each filename, in order.

    Missing or unreadable files yield an empty list for that index so lengths stay
    aligned with ``annos_files``.
    """
    out: List[List[dict]] = []
    for fname in annos_files:
        json_path = os.path.join(annos_folder, fname)
        if not os.path.isfile(json_path):
            print(f"[load_boxes_sequence] missing file -> [] : {json_path}")
            out.append([])
            continue
        try:
            out.append(load_boxes_from_json(json_path))
        except Exception as e:
            print(f"[load_boxes_sequence] skip {json_path}: {e}")
            out.append([])
    return out


def load_json_for_each_image(image_files, json_folder):
    """
    For each image file in image_files, check if a JSON file with the same stem exists in json_folder.
    Returns a dictionary mapping the image filename (or its stem) to the loaded JSON data.
    If the JSON file does not exist, the mapping value is an empty dict.
    """
    json_data = {}
    for image_path in image_files:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(json_folder, f"{stem}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error loading JSON {json_path}: {e}. Using empty dict.")
                data = {}
        else:
            data = {}
        json_data[stem] = data
    return json_data
