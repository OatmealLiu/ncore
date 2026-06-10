import os
import sys
import numpy as np
import open3d as o3d
import json
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math, colorsys
import imageio
import re
import gzip

from vis_utils.load_utils import (
    load_pointclouds_from_folder,
    load_annos_from_folder,
    load_radar_joint_point_cloud,
    load_lidar_joint_point_cloud,
    load_boxes_from_json,
    load_extrinsic_between_nodes,
    load_single_ouster_lidar_point_cloud,
    load_depth_files_with_sync_from_folder,
    load_images_from_folder,
    load_files_with_sync_from_folder,
    match_files_by_sync_or_time,
    parse_synced_filename,
)
from vis_utils.colorize import colorize_pointcloud, get_color_fields_for_sensor
from vis_utils.dataset_details import *
from vis_utils.transforms import (
    get_3d_box_corners,
    transform_points_4x4,
    from_corners_to_obb,
    match_timestamps_one_to_one,
    lidar_timestamps_to_camera,
    radar_timestamps_to_camera,
    clip_box_against_camera_fov,
    extract_fx_fy_cx_cy,
)

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QSlider,
    QVBoxLayout,
    QComboBox,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QDoubleSpinBox,
    QGroupBox,
    QFormLayout,
    QDialog,
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage


def _norm_cam(s: str) -> str:
    s = (s or "").strip().lower()
    # remove common path-like prefixes if any
    s = s.replace("camera/leopard/", "").replace("camera/", "")
    # remove trailing pipeline suffixes
    s = re.sub(r"(?:/.*)$", "", s)  # anything after a slash
    # compress underscores
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def resolve_release_cam_key(camera_topic: str, cam_map: dict) -> str:
    """
    cam_map keys are release camera names for a specific timestamp.
    Returns the best-matching key or None.
    """
    if not cam_map:
        return None

    # Query name from topic
    parts = (camera_topic or "").split("/")
    q = parts[2] if len(parts) >= 3 else (camera_topic or "")
    q_raw = q
    qn = _norm_cam(q_raw)

    # 1) exact match
    if q_raw in cam_map:
        return q_raw
    if q_raw.lower() in {k.lower(): k for k in cam_map}.keys():
        # return original key that matches case-insensitively
        for k in cam_map:
            if k.lower() == q_raw.lower():
                return k

    # 2) normalized match
    norm_to_key = {}
    for k in cam_map.keys():
        kn = _norm_cam(k)
        # keep first occurrence
        if kn not in norm_to_key:
            norm_to_key[kn] = k

    if qn in norm_to_key:
        return norm_to_key[qn]

    # 3) substring / startswith heuristics (useful when release names include extra suffixes)
    # pick the shortest key that still contains qn (most specific without extra junk)
    candidates = []
    for k in cam_map.keys():
        kn = _norm_cam(k)
        if qn and (qn in kn or kn in qn):
            candidates.append((len(kn), k))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    return None


# --------------------------------------------------------------------------------
# The main visualizer
# --------------------------------------------------------------------------------


def quat_xyzw_to_R(qx, qy, qz, qw):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def transform_points_4x4_np(pts_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_xyz, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 3)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.concatenate([pts, ones], axis=1)  # Nx4
    out = (T @ pts_h.T).T  # Nx4
    return out[:, :3]


def find_timestamp_in_url(url: str):
    # release urls usually contain ".../<timestamp>/..."
    for seg in (url or "").split("/"):
        if seg.isdigit() and len(seg) >= 16:
            return seg
    return None


def index_release_json(release_json_path: str):
    """
    Returns:
      release_idx[scene_name][timestamp][camera_name] = {"K": 3x3, "T_ego_cam": 4x4}

    where T_ego_cam is built from release extrinsics translation+quaternion.
    """
    with open(release_json_path, "r") as f:
        rel = json.load(f)

    out = {}
    samples = rel["dataset"]["samples"]
    for s in samples:
        scene_name = s["name"]
        frames = s.get("attributes", {}).get("frames", [])
        scene_map = {}

        for fr in frames:
            images = fr.get("images", [])
            if not images:
                continue

            ts = find_timestamp_in_url(
                images[0].get("url", "")
            ) or find_timestamp_in_url(fr.get("pcd", {}).get("url", ""))
            if ts is None:
                continue

            cam_map = {}
            for im in images:
                cam_name = im.get("name", "")
                K = np.array(
                    im["intrinsics"]["intrinsic_matrix"], dtype=np.float64
                ).reshape(3, 3)

                ex = im["extrinsics"]
                t = np.array(
                    [
                        ex["translation"]["x"],
                        ex["translation"]["y"],
                        ex["translation"]["z"],
                    ],
                    dtype=np.float64,
                )

                q = ex["rotation"]  # qx,qy,qz,qw
                R = quat_xyzw_to_R(q["qx"], q["qy"], q["qz"], q["qw"])
                T_ego_cam = make_T(R, t)

                cam_map[cam_name] = {"K": K, "T_ego_cam": T_ego_cam}

            scene_map[ts] = cam_map

        out[scene_name] = scene_map

    return out


def is_wide_camera_name(cam_name: str) -> bool:
    s = (cam_name or "").lower()
    return ("wide" in s) and ("forward" in s)


def rotate_uv_upright_to_stored_90(
    u0: np.ndarray, v0: np.ndarray, Hs: int, Ws: int, direction: str
):
    """
    Upright (intrinsics space) -> Stored image coords. Stored shape is (Hs, Ws).
    Upright shape is (Hu, Wu) = (Ws, Hs).

    stored = rotate(upright, CW):   u_s = Hu-1 - v0,   v_s = u0
    stored = rotate(upright, CCW):  u_s = v0,          v_s = Wu-1 - u0
    """
    Hu = Ws
    Wu = Hs
    if direction == "cw":
        return (Hu - 1) - v0, u0
    if direction == "ccw":
        return v0, (Wu - 1) - u0
    raise ValueError(direction)


def score_inside(u: np.ndarray, v: np.ndarray, Hs: int, Ws: int) -> int:
    inside = (u >= 0) & (u < Ws) & (v >= 0) & (v < Hs)
    return int(np.count_nonzero(inside))


def choose_best_rotation_upright_to_stored(
    u0: np.ndarray, v0: np.ndarray, Hs: int, Ws: int
) -> str:
    u_cw, v_cw = rotate_uv_upright_to_stored_90(u0, v0, Hs, Ws, "cw")
    u_ccw, v_ccw = rotate_uv_upright_to_stored_90(u0, v0, Hs, Ws, "ccw")
    return (
        "cw"
        if score_inside(u_cw, v_cw, Hs, Ws) >= score_inside(u_ccw, v_ccw, Hs, Ws)
        else "ccw"
    )


def create_obb_lineset(x, y, z, l, w, h, yaw_deg, color=[1.0, 0.0, 0.0]):
    """
    Creates and returns an OrientedBoundingBox, converted to a wireframe LineSet.
    Rotation is assumed around Z-axis by `yaw_deg`.
    """
    obb = o3d.geometry.OrientedBoundingBox()

    # Center
    obb.center = np.array([x, y, z], dtype=np.float64)

    # Extent = (length, width, height)
    obb.extent = np.array([l, w, h], dtype=np.float64)

    # Rotation around Z
    yaw_rad = np.deg2rad(yaw_deg)
    R = obb.get_rotation_matrix_from_xyz((0, 0, yaw_rad))
    obb.R = R

    # Convert OBB to a wireframe LineSet
    lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    lineset.colors = o3d.utility.Vector3dVector([color] * len(lineset.lines))

    return lineset


def _bit_reverse(x: int, bits: int) -> int:
    y = 0
    for _ in range(bits):
        y = (y << 1) | (x & 1)
        x >>= 1
    return y


def distinct_colors_rgb(n: int, s: float = 0.65, v: float = 0.95, seed_h: float = 0.0):
    """
    Generate n visually distinct RGB colors (0–255 tuples) where
    adjacent indices are spread apart in hue using bit-reversal order.
    """
    if not (1 <= n <= 255):
        raise ValueError("n must be between 1 and 255")

    bits = math.ceil(math.log2(max(2, n)))  # at least 1 bit
    m = 1 << bits  # next power of two
    order = [_bit_reverse(i, bits) for i in range(m)]
    order = [x for x in order if x < n]

    colors_rgb = []
    for i in range(n):
        j = order[i]
        hue = (seed_h + j / n) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, s, v)
        colors_rgb.append(
            (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))
        )

    index_to_rgb = {i: colors_rgb[i] for i in range(n)}
    return colors_rgb, index_to_rgb


def create_color_from_tracking_id(tracking_id):
    # Create a colormap with as many colors as indices
    colors_rgb, index_to_rgb = distinct_colors_rgb(255)

    # cmap = plt.cm.get_cmap("tab10", 255)
    # values = cmap(tracking_id)[:3]
    values = index_to_rgb[int(np.clip(tracking_id, 0, 254))]
    # values = tuple([int(c*255) for c in values])
    # print(f"Tracking ID: {tracking_id} -> Color: {values}")
    return values


class ImagePopup(QDialog):
    """
    A QDialog that displays an image with projected 2D points or 2D boxes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Viewer")
        self.label = QLabel(self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        self.resize(1280, 720)

    def set_image(self, qimage):
        """
        Set the QImage or QPixmap to display in the label.
        """
        pixmap = QPixmap.fromImage(qimage)
        self.label.setPixmap(pixmap)
        self.label.adjustSize()


class PointCloudVisualizer(QWidget):
    def __init__(self, root_dir, recording, ptype="lidar"):
        super().__init__()
        self.root_dir = root_dir
        self.recording = recording
        self.scene_dir = os.path.join(root_dir, recording)
        self.ptype = ptype

        if not os.path.isdir(self.scene_dir):
            raise FileNotFoundError(f"Scene folder not found: {self.scene_dir}")

        public_bbox = os.path.join(self.scene_dir, annotation_bounding_boxes_relpath)

        self.existing_label_folder = []
        if os.path.isdir(public_bbox):
            self.existing_label_folder.append(annotation_bounding_boxes_relpath)

        if not self.existing_label_folder:
            raise FileNotFoundError(
                f"No annotation folder found under {self.scene_dir}. Expected "
                f"{annotation_bounding_boxes_relpath}/ for the TruckDrive dataset."
            )

        self.annos_folder = os.path.join(self.scene_dir, self.existing_label_folder[0])
        self.annos, self.timestamps_annos, self.sync_annos = (
            load_files_with_sync_from_folder(self.annos_folder, ["json"])
        )
        if not self.annos:
            print(f"Warning: no annotation JSON files found in {self.annos_folder}")

        # ------------------------------------------------------------------
        # Calibration tree
        # ------------------------------------------------------------------
        tf_candidates = [
            os.path.join(self.scene_dir, tf_tree_relpath),
            os.path.join(self.scene_dir, "calib_tf_tree_full.json"),
        ]
        tf_file = next((c for c in tf_candidates if os.path.isfile(c)), None)
        if tf_file is None:
            raise FileNotFoundError(
                "Could not find calib_tf_tree_full.json. Checked: "
                + ", ".join(tf_candidates)
            )
        with open(tf_file, "r") as f:
            self.data_extrinsics = json.load(f)
        self.tf_file = tf_file

        # Keep track of bounding boxes from JSON.
        self.box_linesets = []

        self.lane_linesets_3d = []

        self.sensors_data = {}
        self.pointcloud_names = []
        self.camera_names = []
        self.short_camera_names = []

        # ------------------------------------------------------------------
        # LiDAR sensors
        # ------------------------------------------------------------------
        for lidar_topic, lidar_tf_node in zip(lidar_topics, lidar_tf_nodes):
            rel_topic = lidar_topic[1:]
            lidar_folder = os.path.join(self.scene_dir, rel_topic)
            files_lidar, ts_lidar, sync_lidar = load_files_with_sync_from_folder(
                lidar_folder, ["bin"]
            )
            matches_lidar = self._match_sensor_to_annotations(
                sync_lidar, ts_lidar, max_diff=75e6
            )
            print(f"Found {len(files_lidar)} lidar bin files in {lidar_folder}")
            self.sensors_data[rel_topic] = {
                "folder": lidar_folder,
                "files": files_lidar,
                "timestamps": ts_lidar,
                "sync_ids": sync_lidar,
                "matches": matches_lidar,
            }
            if files_lidar:
                self.pointcloud_names.append(rel_topic)

        # ------------------------------------------------------------------
        # Radar sensors
        # ------------------------------------------------------------------
        for radar_topic, radar_tf_node in zip(radar_topics, radar_tf_nodes):
            rel_topic = radar_topic[1:]
            radar_folder = os.path.join(self.scene_dir, rel_topic)
            files_radar, ts_radar, sync_radar = load_files_with_sync_from_folder(
                radar_folder, ["bin"]
            )
            matches_radar = self._match_sensor_to_annotations(
                sync_radar, ts_radar, max_diff=75e6
            )
            print(f"Found {len(files_radar)} radar bin files in {radar_folder}")
            self.sensors_data[rel_topic] = {
                "folder": radar_folder,
                "files": files_radar,
                "timestamps": ts_radar,
                "sync_ids": sync_radar,
                "matches": matches_radar,
            }
            if files_radar:
                self.pointcloud_names.append(rel_topic)

        # ------------------------------------------------------------------
        # Camera image folders
        # ------------------------------------------------------------------
        for image_topic, calib, short_camera_name in zip(
            image_topics, calib_file, camera_setup
        ):
            rel_topic = image_topic[1:]
            image_folder = os.path.join(self.scene_dir, rel_topic)
            files_image, ts_image, sync_image = load_files_with_sync_from_folder(
                image_folder, ["jpg", "jpeg", "png"]
            )
            matches_image = self._match_sensor_to_annotations(
                sync_image, ts_image, max_diff=75e6
            )
            matches_dict = {}
            for image_idx, label_idx in matches_image:
                if matches_dict.get(label_idx, None) is None:
                    matches_dict[label_idx] = image_idx
                else:
                    print(
                        f"Warning: label_idx={label_idx} matched more than once in {image_folder}"
                    )
            print(f"Found {len(files_image)} image files in {image_folder}")
            self.sensors_data[rel_topic] = {
                "folder": image_folder,
                "files": files_image,
                "timestamps": ts_image,
                "sync_ids": sync_image,
                "matches": matches_dict,
            }
            if files_image:
                self.camera_names.append(rel_topic)
                self.short_camera_names.append(short_camera_name)

        self.current_camera_topic = self.camera_names[0] if self.camera_names else None

        # ------------------------------------------------------------------
        # Accumulated GT depth folders
        # scene/accumulated_gt_depth/<camera_name>/<sync>_<timestamp>.npy
        # ------------------------------------------------------------------
        self.depth_data = {}
        self.depth_root = os.path.join(self.scene_dir, "accumulated_gt_depth")

        if os.path.isdir(self.depth_root):
            for cam_topic in self.camera_names:
                short = self._camera_short_name(cam_topic)
                depth_folder = os.path.join(self.depth_root, short)

                files_depth, ts_depth, sync_depth = (
                    load_depth_files_with_sync_from_folder(depth_folder)
                )

                sync_to_idx = {}
                ts_to_idx = {}

                for i, (fn, ts, sid) in enumerate(
                    zip(files_depth, ts_depth, sync_depth)
                ):
                    if sid is not None and int(sid) not in sync_to_idx:
                        sync_to_idx[int(sid)] = i
                    if ts is not None:
                        ts_to_idx[int(ts)] = i

                self.depth_data[cam_topic] = {
                    "folder": depth_folder,
                    "files": files_depth,
                    "timestamps": ts_depth,
                    "sync_ids": sync_depth,
                    "sync_to_idx": sync_to_idx,
                    "ts_to_idx": ts_to_idx,
                }

                print(
                    f"[DEPTH] Found {len(files_depth)} depth npy files in {depth_folder}"
                )
        else:
            print(f"[DEPTH] accumulated_gt_depth folder not found: {self.depth_root}")

        # ------------------------------------------------------------------
        # Lane-line annotations in the dataset
        # ------------------------------------------------------------------
        self.bev_lane_scene_dir = os.path.join(
            self.scene_dir, annotation_lane_lines_relpath
        )
        self.lane_annos = []
        self.lane_timestamps = []
        self.lane_sync_ids = []
        self.lane_sync_to_file = {}
        self.lane_ts_to_file = {}
        self.lane_ts_set = set()
        if os.path.isdir(self.bev_lane_scene_dir):
            self.lane_annos, self.lane_timestamps, self.lane_sync_ids = (
                load_files_with_sync_from_folder(self.bev_lane_scene_dir, ["json"])
            )
            for fn, ts, sid in zip(
                self.lane_annos, self.lane_timestamps, self.lane_sync_ids
            ):
                if sid is not None and sid not in self.lane_sync_to_file:
                    self.lane_sync_to_file[int(sid)] = fn
                    self.lane_ts_set.add(str(int(sid)))
                self.lane_ts_to_file[int(ts)] = fn
            print(
                f"[LANE] Found {len(self.lane_annos)} lane JSON files in {self.bev_lane_scene_dir}"
            )
        else:
            print(f"[LANE] Lane-line folder not found: {self.bev_lane_scene_dir}")
            self.bev_lane_scene_dir = None

        self.lane_label_indices = []
        self.label_to_slider_idx = {}

        # Open3D visualizer.
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name="Open3D PointCloud Viewer",
            width=1920,
            height=1080,
        )

        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)

        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=5.0, origin=[0, 0, 0]
        )
        self.vis.add_geometry(self.coordinate_frame)

        self.current_sensor = (
            self.pointcloud_names[0] if self.pointcloud_names else None
        )
        self.bb_lineset = None

        self.active_3d_mode = "single"
        self.active_fusion_group = None

        self.init_ui()

        if self.current_sensor is not None:
            QTimer.singleShot(0, self.load_pointcloud)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_visualizer)
        self.timer.start(100)
        self.set_initial_view()

    def _match_sensor_to_annotations(self, sync_sensor, ts_sensor, max_diff=75e6):
        return match_files_by_sync_or_time(
            sync_sensor,
            self.sync_annos,
            ts_sensor,
            self.timestamps_annos,
            max_diff=max_diff,
        )

    def _annotation_file_sync_and_timestamp(self, label_idx: int):
        if label_idx < 0 or label_idx >= len(self.annos):
            return None, None
        return parse_synced_filename(self.annos[label_idx])[:2]

    def _camera_short_name(self, camera_topic: str) -> str:
        parts = (camera_topic or "").split("/")
        if len(parts) >= 3:
            return parts[2]
        return camera_topic or ""

    def camera_calibration_path(self, camera_topic: str) -> str:
        short = self._camera_short_name(camera_topic)
        candidates = [
            os.path.join(
                self.scene_dir, "calibrations", f"calib_camera_leopard_{short}.json"
            ),
            os.path.join(self.scene_dir, f"calib_camera_leopard_{short}.json"),
            os.path.join(
                self.scene_dir, "calib_" + camera_topic.replace("/", "_") + ".json"
            ),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        return candidates[0]

    def load_camera_projection(self, camera_topic: str):
        calib_path = self.camera_calibration_path(camera_topic)
        if not os.path.isfile(calib_path):
            raise FileNotFoundError(
                f"Camera calibration not found for {camera_topic}: {calib_path}"
            )
        with open(calib_path, "r") as f:
            d = json.load(f)
        P = np.asarray(d["P"], dtype=np.float64).reshape((3, 4))
        height = int(d.get("height", 0))
        width = int(d.get("width", 0))
        return P, height, width, calib_path

    def on_recording_changed(self, idx):
        # self.recording = self.available_recordings[idx]
        self.update_annos_folder()

    def on_anno_changed(self, idx):
        self.update_annos_folder()

    def update_annos_folder(self):
        anno_choice = self.anno_combo.currentText()
        self.annos_folder = os.path.join(self.scene_dir, anno_choice)
        self.annos, self.timestamps_annos, self.sync_annos = (
            load_files_with_sync_from_folder(self.annos_folder, ["json"])
        )
        print("Using annotations from:", self.annos_folder)

        # Rebuild sensor/annotation matches because label indices may have changed.
        for topic, data in self.sensors_data.items():
            sync_ids = data.get("sync_ids", [])
            timestamps = data.get("timestamps", [])
            matches = self._match_sensor_to_annotations(
                sync_ids, timestamps, max_diff=75e6
            )
            if topic in self.camera_names:
                md = {}
                for sensor_idx, label_idx in matches:
                    if label_idx not in md:
                        md[label_idx] = sensor_idx
                data["matches"] = md
            else:
                data["matches"] = matches

        self.label_to_slider_idx = {}
        self.lane_label_indices = []
        if self.current_sensor in self.sensors_data:
            n_matches = len(self.sensors_data[self.current_sensor].get("matches", []))
            self.slider.setMinimum(0)
            self.slider.setMaximum(max(0, n_matches - 1))

    def init_ui(self):
        main_layout = QVBoxLayout()

        # ------------------------------
        # Sensor type dropdown
        # ------------------------------
        sensor_label = QLabel("Select 3D Sensor:")
        main_layout.addWidget(sensor_label)
        self.sensor_combo = QComboBox()
        self.sensor_combo.addItems(self.pointcloud_names)
        # Connect to a callback that changes the current sensor
        self.sensor_combo.currentIndexChanged.connect(self.on_sensor_changed)
        main_layout.addWidget(self.sensor_combo)

        # ------------------------------
        # 1) Controls for browsing point clouds
        # ------------------------------
        pc_controls_layout = QVBoxLayout()
        label = QLabel("Select Point Cloud and Color Column")
        pc_controls_layout.addWidget(label)

        self.slider = QSlider(Qt.Horizontal)
        n_matches = (
            len(self.sensors_data[self.current_sensor]["matches"])
            if self.current_sensor in self.sensors_data
            else 0
        )
        self.slider.setMinimum(0)
        if n_matches > 0:
            self.slider.setMinimum(0)
            self.slider.setMaximum(n_matches - 1)
        self.slider.valueChanged.connect(self.on_slider_changed)
        pc_controls_layout.addWidget(self.slider)

        self.combo = QComboBox()
        self.update_color_combo_for_sensor()
        self.combo.currentIndexChanged.connect(self.update_colorization)
        pc_controls_layout.addWidget(self.combo)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_pointcloud)
        pc_controls_layout.addWidget(self.next_button)

        main_layout.addLayout(pc_controls_layout)

        # ------------------------------
        # 3D LiDAR Section
        # ------------------------------

        ddd_label = QLabel("3D Visualization Tools:")
        main_layout.addWidget(ddd_label)

        self.visualize_all_ouster_button = QPushButton("Visualize all Ouster sensors")
        self.visualize_all_ouster_button.clicked.connect(
            self.on_visualize_all_ouster_clicked
        )
        main_layout.addWidget(self.visualize_all_ouster_button)

        self.visualize_all_lidar_button = QPushButton("Visualize all LiDAR sensors")
        self.visualize_all_lidar_button.clicked.connect(
            self.on_visualize_all_lidar_clicked
        )
        main_layout.addWidget(self.visualize_all_lidar_button)

        self.visualize_all_3d_button = QPushButton("Visualize all 3D sensors")
        self.visualize_all_3d_button.clicked.connect(self.on_visualize_all_3d_clicked)
        main_layout.addWidget(self.visualize_all_3d_button)

        # Add a button to load your JSON boxes
        self.load_boxes_button = QPushButton("Load 3D Bounding Boxes")
        self.load_boxes_button.clicked.connect(self.on_load_boxes_clicked)
        main_layout.addWidget(self.load_boxes_button)

        self.load_lane_lines_3d_button = QPushButton("Load 3D Lane Lines")
        self.load_lane_lines_3d_button.clicked.connect(
            self.on_load_3d_lane_lines_clicked
        )
        main_layout.addWidget(self.load_lane_lines_3d_button)

        # ------------------------------
        # Generate LiDAR Video Button
        # ------------------------------
        self.generate_video_button = QPushButton("Generate 3D Sensor Video")
        self.generate_video_button.clicked.connect(self.on_generate_video_clicked)
        main_layout.addWidget(self.generate_video_button)

        # ---------------------------------------------------------
        # Add camera combo box for image selection (NEW)
        # ---------------------------------------------------------
        camera_label = QLabel("Select 2D Camera:")
        main_layout.addWidget(camera_label)
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(self.camera_names)
        self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)
        main_layout.addWidget(self.camera_combo)

        # ------------------------------
        # 2D Camera Section
        # ------------------------------

        dd_label = QLabel("2D Visualization Tools:")
        main_layout.addWidget(dd_label)

        # ---------------------------------------------------------
        # Button to visualize current frame in 2D image (NEW)
        # ---------------------------------------------------------
        self.visualize_boxes_2d_button = QPushButton("Visualize boxes in 2D")
        self.visualize_boxes_2d_button.clicked.connect(self.on_visualize_boxes_2d)

        self.visualize_lanes_2d_button = QPushButton("Visualize lane lines in 2D")
        self.visualize_lanes_2d_button.clicked.connect(self.on_visualize_lane_lines_2d)

        self.visualize_depth_2d_button = QPushButton("Visualize accumulated GT depth")
        self.visualize_depth_2d_button.clicked.connect(
            self.on_visualize_accumulated_gt_depth_2d
        )

        main_layout.addWidget(self.visualize_boxes_2d_button)
        main_layout.addWidget(self.visualize_lanes_2d_button)
        main_layout.addWidget(self.visualize_depth_2d_button)

        # Button to generate a video from camera images
        self.generate_camera_video_button = QPushButton("Generate Video (Camera)")
        self.generate_camera_video_button.clicked.connect(
            self.on_generate_video_camera_clicked
        )
        main_layout.addWidget(self.generate_camera_video_button)

        # ------------------------------
        # Generate Video Button (Lane lines only)
        # ------------------------------
        self.generate_lane_video_button = QPushButton("Generate Video (Lane lines)")
        self.generate_lane_video_button.clicked.connect(
            self.on_generate_video_lane_lines_clicked
        )
        main_layout.addWidget(self.generate_lane_video_button)

        self.setLayout(main_layout)

    def get_current_label_idx(self):
        """
        Return the annotation label_idx corresponding to the current slider value
        for the current 3D sensor.
        """
        if self.current_sensor not in self.sensors_data:
            return None

        data = self.sensors_data[self.current_sensor]
        matches = data.get("matches", [])
        if not matches:
            return None

        idx = self.slider.value()
        if idx < 0 or idx >= len(matches):
            return None

        pair = matches[idx]
        if not pair or len(pair) < 2:
            return None

        return pair[1]

    def _get_topic_frame(self, topic: str):
        """
        Resolve a topic into its TF frame using sensor_path.
        Topics are stored without leading slash in this tool.
        """
        if topic in sensor_path:
            return sensor_path[topic]

        topic_no_slash = topic[1:] if topic.startswith("/") else topic
        if topic_no_slash in sensor_path:
            return sensor_path[topic_no_slash]

        topic_with_slash = "/" + topic_no_slash
        if topic_with_slash in sensor_path:
            return sensor_path[topic_with_slash]

        raise KeyError(f"No TF frame found in sensor_path for topic={topic}")

    def _find_sensor_file_for_label(self, topic: str, label_idx: int):
        """
        For a point-cloud topic, find the sensor file matched to label_idx.
        Returns full path or None.
        """
        data = self.sensors_data.get(topic)
        if data is None:
            return None

        matches = data.get("matches", [])
        files = data.get("files", [])
        folder = data.get("folder", "")

        for pair in matches:
            if not pair or len(pair) < 2:
                continue

            sensor_idx, matched_label_idx = pair
            if matched_label_idx != label_idx:
                continue

            if sensor_idx < 0 or sensor_idx >= len(files):
                return None

            return os.path.join(folder, files[sensor_idx])

        return None

    def on_slider_changed(self):
        """
        Render the current frame according to the active 3D mode.

        single mode:
        reload only self.current_sensor

        fusion mode:
        reload the active fused sensor group at the current slider frame
        """
        self.render_current_3d_frame()

    def render_current_3d_frame(self):
        """
        Render the current slider frame using either:
        - the selected single sensor
        - the active fused sensor group
        """
        if self.active_3d_mode == "fusion" and self.active_fusion_group is not None:
            self.visualize_sensor_group(self.active_fusion_group)
        else:
            self.load_pointcloud()

    def _load_pointcloud_for_topic(self, topic: str, file_path: str):
        """
        Load one point cloud using the correct binary layout.
        """
        if "lidar" in topic:
            if "joint" in topic:
                return load_lidar_joint_point_cloud(file_path)
            elif "ouster" in topic:
                return load_single_ouster_lidar_point_cloud(file_path)
            else:
                return None

        if "radar" in topic:
            if "joint" in topic:
                return load_radar_joint_point_cloud(file_path)
            else:
                return None

        raise ValueError(f"Unknown point-cloud topic: {topic}")

    def update_pcd_xyz_rgb(self, points_xyz: np.ndarray, colors_rgb: np.ndarray):
        """
        Replace the Open3D point cloud with already-transformed XYZ points and RGB colors.
        Used for fused multi-sensor visualization.
        """
        points_xyz = np.asarray(points_xyz, dtype=np.float64)
        colors_rgb = np.asarray(colors_rgb, dtype=np.float64)

        if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
            print("[FUSION] Invalid points array.")
            return

        if colors_rgb.ndim != 2 or colors_rgb.shape[1] != 3:
            print("[FUSION] Invalid colors array.")
            return

        if points_xyz.shape[0] == 0:
            print("[FUSION] Empty fused point cloud.")
            return

        if colors_rgb.shape[0] != points_xyz.shape[0]:
            print("[FUSION] Color count does not match point count.")
            return

        view_ctl = self.vis.get_view_control()
        cam_params = view_ctl.convert_to_pinhole_camera_parameters()

        try:
            self.vis.remove_geometry(self.pcd, reset_bounding_box=False)
        except Exception:
            pass

        self.remove_all_bboxes()
        self.remove_all_lane_lines_3d()

        self.pcd.clear()
        self.pcd.points = o3d.utility.Vector3dVector(points_xyz)
        self.pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

        self.vis.add_geometry(self.pcd, reset_bounding_box=False)
        self.vis.add_geometry(self.coordinate_frame, reset_bounding_box=False)

        try:
            view_ctl.convert_from_pinhole_camera_parameters(cam_params)
        except Exception:
            pass

        self.vis.poll_events()
        self.vis.update_renderer()

    def get_display_frame(self) -> str:
        """
        Canonical Open3D display frame.

        Bounding-box annotations are already expressed in this frame, so all 3D
        sensors should be transformed into this frame before visualization.
        """
        return sensor_path.get("annotations/bounding_boxes", "velodyne")

    def transform_xyz_from_topic_to_display_frame(
        self, xyz: np.ndarray, topic: str
    ) -> np.ndarray:
        """
        Transform XYZ points from a sensor topic frame into the display frame.
        """
        xyz = np.asarray(xyz, dtype=np.float64)

        if xyz.size == 0:
            return xyz.reshape(0, 3)

        sensor_frame = self._get_topic_frame(topic)
        display_frame = self.get_display_frame()

        if sensor_frame == display_frame:
            return xyz

        T_display_to_sensor = load_extrinsic_between_nodes(
            self.data_extrinsics,
            display_frame,
            sensor_frame,
        )

        T_sensor_to_display = invert_T(T_display_to_sensor)

        return transform_points_4x4_np(xyz, T_sensor_to_display)

    def transform_pointcloud_to_display_frame(
        self, pc_data: np.ndarray, topic: str
    ) -> np.ndarray:
        """
        Transform first 3 columns of a point cloud into the display frame.
        Non-XYZ columns are preserved for colorization.
        """
        if pc_data is None or pc_data.shape[0] == 0:
            return pc_data

        out = np.asarray(pc_data).copy()
        out[:, :3] = self.transform_xyz_from_topic_to_display_frame(
            out[:, :3],
            topic,
        )
        return out

    def _norm_label_value(self, value) -> str:
        """
        Normalize annotation string values for robust comparisons.
        """
        if value is None:
            return ""

        s = str(value).strip().lower()
        s = s.replace("-", "_").replace(" ", "_").replace("/", "_")
        s = re.sub(r"_+", "_", s)
        return s.strip("_")

    def is_ego_box(self, box: dict) -> bool:
        """
        Return True if this annotation box corresponds to the ego vehicle.

        This is intentionally defensive because different annotation exports may
        encode ego with different fields/names.

        Common patterns covered:
        id: ego
        id: ego_vehicle
        class-id: Ego
        class-id: EgoVehicle
        class-id: Vehicle-Ego
        mapped_class: Ego
        label: ego
        name: ego

        If your dataset uses a known Tracking_ID for ego, add it below.
        """
        if not isinstance(box, dict):
            return False

        string_fields = [
            "id",
            "class-id",
            "class_id",
            "class",
            "label",
            "name",
            "mapped_class",
            "object_class",
            "obj_class",
            "category",
            "category_name",
        ]

        ego_tokens = {
            "ego",
            "ego_vehicle",
            "vehicle_ego",
        }

        for key in string_fields:
            value = box.get(key, None)
            if value is None:
                continue

            norm = self._norm_label_value(value)

            if norm in ego_tokens:
                return True

            if "ego" in norm:
                return True

        return False

    def _find_depth_file_for_label(self, camera_topic: str, label_idx: int):
        """
        Find accumulated_gt_depth npy file for the same sync/timestamp as label_idx
        and the selected camera.
        """
        info = self.depth_data.get(camera_topic)
        if info is None:
            print(f"[DEPTH] No depth data registered for camera={camera_topic}")
            return None

        sync_id, ts = self._annotation_file_sync_and_timestamp(label_idx)

        depth_idx = None

        if sync_id is not None:
            depth_idx = info["sync_to_idx"].get(int(sync_id))

        if depth_idx is None and ts is not None:
            depth_idx = info["ts_to_idx"].get(int(ts))

        if depth_idx is None:
            print(
                f"[DEPTH] No depth file matched camera={camera_topic}, "
                f"label_idx={label_idx}, sync_id={sync_id}, ts={ts}"
            )
            return None

        files = info["files"]
        folder = info["folder"]

        if depth_idx < 0 or depth_idx >= len(files):
            return None

        path = os.path.join(folder, files[depth_idx])
        if not os.path.isfile(path):
            print(f"[DEPTH] Depth file missing: {path}")
            return None

        return path

    def draw_accumulated_gt_depth_on_image(
        self,
        pil_image,
        label_idx: int,
        camera_topic: str,
        alpha: float = 0.55,
        cmap_name: str = "turbo",
    ):
        """
        Overlay accumulated_gt_depth/<camera>/<sync>_<timestamp>.npy on the camera image.

        Expected depth npy:
        H x W float array
        or:
        H x W x 1 array

        Depth values <= 0, NaN, or inf are treated as invalid.
        """
        depth_path = self._find_depth_file_for_label(camera_topic, label_idx)
        if depth_path is None:
            return pil_image

        try:
            with gzip.open(depth_path, "rb") as f:
                depth = np.load(f)
        except Exception as e:
            print(f"[DEPTH] Failed to load {depth_path}: {e}")
            return pil_image

        depth = np.asarray(depth)

        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        elif depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]
        elif depth.ndim != 2:
            print(f"[DEPTH] Unsupported depth shape {depth.shape} for {depth_path}")
            return pil_image

        depth = depth.astype(np.float64)

        valid = np.isfinite(depth) & (depth > 0)
        if np.count_nonzero(valid) == 0:
            print(f"[DEPTH] No valid depth values in {depth_path}")
            return pil_image

        vals = depth[valid]

        lo = np.percentile(vals, 2)
        hi = np.percentile(vals, 98)

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(vals))
            hi = float(np.nanmax(vals))

        if hi <= lo:
            print(f"[DEPTH] Degenerate depth range in {depth_path}")
            return pil_image

        norm = np.zeros_like(depth, dtype=np.float64)
        norm[valid] = np.clip((depth[valid] - lo) / (hi - lo), 0.0, 1.0)

        try:
            cmap = plt.get_cmap(cmap_name)
        except Exception:
            cmap = plt.get_cmap("viridis")

        rgb = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)

        alpha_map = np.zeros(depth.shape, dtype=np.uint8)
        alpha_map[valid] = int(np.clip(alpha, 0.0, 1.0) * 255)

        rgba = np.dstack([rgb, alpha_map])

        overlay = Image.fromarray(rgba, mode="RGBA")

        # Resize depth overlay if it does not exactly match the image.
        Ws, Hs = pil_image.size
        if overlay.size != (Ws, Hs):
            overlay = overlay.resize((Ws, Hs), resample=Image.BILINEAR)

        base = pil_image.convert("RGBA")
        out = Image.alpha_composite(base, overlay).convert("RGB")

        draw = ImageDraw.Draw(out)
        draw.text(
            (20, 20),
            f"Accumulated GT depth | {os.path.basename(depth_path)} | range={lo:.2f}-{hi:.2f}",
            fill=(255, 255, 255),
        )

        print(
            f"[DEPTH] Overlayed {depth_path}, "
            f"shape={depth.shape}, valid={np.count_nonzero(valid)}, "
            f"range={lo:.3f}-{hi:.3f}"
        )

        return out

    def visualize_sensor_group(self, group: str):
        """
        Visualize multiple 3D sensors at the same synchronized label frame.

        The current selected sensor defines the target visualization frame.
        Example:
        if current_sensor == lidar/aeva/joint_lidars/points,
        all selected sensors are transformed into the Aeva joint-lidar frame.

        group:
        "ouster"    -> all Ouster sensors
        "all_lidar" -> Aeva + Ouster
        "all_3d"    -> LiDAR + radar
        """
        label_idx = self.get_current_label_idx()
        if label_idx is None:
            print("[FUSION] Could not determine current label_idx.")
            return

        if self.current_sensor not in self.sensors_data:
            print("[FUSION] No current 3D sensor selected.")
            return

        if group == "ouster":
            topics = [t for t in self.pointcloud_names if t.startswith("lidar/ouster/")]
        elif group == "aeva":
            topics = [t for t in self.pointcloud_names if t.startswith("lidar/aeva/")]
        elif group == "all_lidar":
            topics = [t for t in self.pointcloud_names if t.startswith("lidar/")]
        elif group == "all_3d":
            topics = list(self.pointcloud_names)
        else:
            raise ValueError(f"Unknown fusion group: {group}")

        if not topics:
            print(f"[FUSION] No topics found for group={group}")
            return

        target_topic = self.current_sensor
        target_frame = self.get_display_frame()

        print(f"[FUSION] Visualizing group={group}")
        print(f"[FUSION] Timeline/current topic={target_topic}")
        print(f"[FUSION] Display target frame={target_frame}")
        print(f"[FUSION] Topics={topics}")

        all_points = []
        all_colors = []

        colors_rgb, _ = distinct_colors_rgb(max(1, len(topics)))

        for topic_idx, topic in enumerate(topics):
            file_path = self._find_sensor_file_for_label(topic, label_idx)
            if file_path is None:
                print(
                    f"[FUSION] No matched file for topic={topic}, label_idx={label_idx}"
                )
                continue

            try:
                pc_data = self._load_pointcloud_for_topic(topic, file_path)
            except Exception as e:
                print(f"[FUSION] Failed to load {file_path}: {e}")
                continue

            if pc_data is None or pc_data.shape[0] == 0:
                print(f"[FUSION] Empty cloud for topic={topic}")
                continue

            xyz = np.asarray(pc_data[:, :3], dtype=np.float64)

            try:
                xyz = self.transform_xyz_from_topic_to_display_frame(
                    xyz,
                    topic,
                )
            except Exception as e:
                print(
                    f"[FUSION] Could not transform topic={topic} "
                    f"into target_frame={target_frame}: {e}"
                )
                continue

            color = np.asarray(colors_rgb[topic_idx], dtype=np.float64) / 255.0
            color_arr = np.tile(color.reshape(1, 3), (xyz.shape[0], 1))

            all_points.append(xyz)
            all_colors.append(color_arr)

            print(
                f"[FUSION] Added topic={topic}, "
                f"points={xyz.shape[0]}, file={os.path.basename(file_path)}"
            )

        if not all_points:
            print("[FUSION] No point clouds could be fused.")
            return

        fused_points = np.vstack(all_points)
        fused_colors = np.vstack(all_colors)

        self.update_pcd_xyz_rgb(fused_points, fused_colors)

        print(
            f"[FUSION] Displayed fused point cloud with {fused_points.shape[0]} points."
        )

    def on_visualize_all_ouster_clicked(self):
        self.active_3d_mode = "fusion"
        self.active_fusion_group = "ouster"
        self.visualize_sensor_group("ouster")

    def on_visualize_all_aeva_clicked(self):
        self.active_3d_mode = "fusion"
        self.active_fusion_group = "aeva"
        self.visualize_sensor_group("aeva")

    def on_visualize_all_lidar_clicked(self):
        self.active_3d_mode = "fusion"
        self.active_fusion_group = "all_lidar"
        self.visualize_sensor_group("all_lidar")

    def on_visualize_all_3d_clicked(self):
        self.active_3d_mode = "fusion"
        self.active_fusion_group = "all_3d"
        self.visualize_sensor_group("all_3d")

    def on_visualize_accumulated_gt_depth_2d(self):
        """
        Visualize accumulated GT depth on top of the currently selected camera image.
        Uses the current slider's annotation label_idx, then finds the matching camera
        image and matching accumulated_gt_depth npy by sync key.
        """
        if not self.current_camera_topic:
            print("[DEPTH] No camera selected.")
            return

        label_idx = self.get_current_label_idx()
        if label_idx is None:
            print("[DEPTH] Could not determine current label_idx.")
            return

        data_camera = self.sensors_data.get(self.current_camera_topic)
        if data_camera is None:
            print(f"[DEPTH] Camera topic not found: {self.current_camera_topic}")
            return

        matches_camera = data_camera.get("matches", {})
        camera_idx = matches_camera.get(label_idx, None)

        if camera_idx is None:
            print(
                f"[DEPTH] No camera image matched for "
                f"camera={self.current_camera_topic}, label_idx={label_idx}"
            )
            return

        files_camera = data_camera.get("files", [])
        folder_camera = data_camera.get("folder", "")

        if camera_idx < 0 or camera_idx >= len(files_camera):
            print(f"[DEPTH] Invalid camera_idx={camera_idx}")
            return

        image_path = os.path.join(folder_camera, files_camera[camera_idx])
        image = self.load_image_for_camera(image_path)

        if image is None:
            print(f"[DEPTH] Failed to load image: {image_path}")
            return

        overlay = self.draw_accumulated_gt_depth_on_image(
            image.copy(),
            label_idx,
            self.current_camera_topic,
        )

        def numpy_to_qimage(image_pil) -> QImage:
            image_pil = image_pil.resize((1920, 1024))
            rgb_array = np.array(image_pil)

            if rgb_array.dtype != np.uint8:
                rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)

            rgb_array = np.ascontiguousarray(rgb_array)

            h, w, ch = rgb_array.shape
            bytes_per_line = ch * w

            return QImage(
                rgb_array.tobytes(),
                w,
                h,
                bytes_per_line,
                QImage.Format_RGB888,
            )

        qimg = numpy_to_qimage(overlay)
        self.show_single_image_popup(qimg)

    def set_initial_view(self):
        # 1) Create a large AxisAlignedBoundingBox
        large_aabb = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(-300, -300, -300), max_bound=(300, 300, 300)
        )

        # 2) Add with reset_bounding_box=True (this will zoom out once)
        self.vis.add_geometry(large_aabb, reset_bounding_box=True)

        # 3) Remove it without resetting the view
        self.vis.remove_geometry(large_aabb, reset_bounding_box=False)

        # 4) Poll / update
        self.vis.poll_events()
        self.vis.update_renderer()

    def on_generate_video_lane_lines_clicked(self):
        """
        Generate a video for the currently selected camera using ONLY frames where
        lane json exists (1 Hz) AND the camera has a matched image.
        Draw ONLY lane lines (no 2D/3D boxes), then write frames to an mp4.
        """
        import imageio
        import numpy as np

        if not self.current_camera_topic:
            print("[LANE-VID] No camera selected.")
            return

        # Ensure lane_label_indices is built for the current camera
        self.rebuild_lane_label_indices_for_current_camera()
        if not self.lane_label_indices:
            print(
                "[LANE-VID] No synced lane frames available (lane json + camera match)."
            )
            return

        cam_topic = self.current_camera_topic
        data_camera = self.sensors_data.get(cam_topic, None)
        if data_camera is None:
            print(f"[LANE-VID] Camera topic not found in sensors_data: {cam_topic}")
            return

        matches_camera = data_camera.get("matches", {})  # dict {label_idx: camera_idx}
        files_camera = data_camera.get("files", [])
        folder_camera = data_camera.get("folder", "")

        if not matches_camera or not files_camera or not os.path.isdir(folder_camera):
            print(f"[LANE-VID] Camera data incomplete for {cam_topic}")
            return

        # Output path
        sensor_name = cam_topic.replace("/", "_")
        output_name = f"output_{self.recording}_{sensor_name}_LANES_ONLY.mp4"

        fps = 5  # lane frames are 1 Hz, but video fps can be anything
        print(
            f"[LANE-VID] Generating lane-only video for '{cam_topic}' -> {output_name} (fps={fps})"
        )
        writer = imageio.get_writer(output_name, fps=fps)

        written = 0

        try:
            for label_idx in self.lane_label_indices:
                camera_idx = matches_camera.get(label_idx, None)
                if camera_idx is None:
                    continue
                if camera_idx < 0 or camera_idx >= len(files_camera):
                    continue

                file_name = files_camera[camera_idx]
                image_path = os.path.join(folder_camera, file_name)

                image = self.load_image_for_camera(image_path)
                if image is None:
                    print(f"[LANE-VID] Failed to load: {image_path}")
                    continue

                overlay = image.copy()

                # Draw ONLY lanes
                try:
                    overlay = self.draw_lanes_on_image_release(
                        overlay, label_idx, cam_topic
                    )
                except Exception as e:
                    print("[LANE-VID] draw_lanes_on_image_release failed:", e)
                    continue

                # imageio wants uint8 array
                np_img = np.array(overlay)
                if np_img.dtype != np.uint8:
                    np_img = np.clip(np_img, 0, 255).astype(np.uint8)

                writer.append_data(np_img)
                written += 1

        finally:
            writer.close()

        print(f"[LANE-VID] Saved {written} frames -> '{output_name}'")

    def on_generate_video_clicked(self):
        """
        Generate a 3D video using the currently active 3D visualization mode.

        If active_3d_mode == "single":
        video shows the currently selected single sensor.

        If active_3d_mode == "fusion":
        video shows the active fused group, e.g. all_lidar or all_3d.

        The current selected sensor still defines:
        - the timeline / slider matches
        - the target coordinate frame for fused visualization
        - the frame used for 3D bounding boxes
        """
        if self.current_sensor not in self.sensors_data:
            print("[VIDEO] No current 3D sensor selected.")
            return

        data = self.sensors_data[self.current_sensor]
        matches = data.get("matches", [])

        if not matches:
            print("[VIDEO] No data to generate a video.")
            return

        fps = 10

        if self.active_3d_mode == "fusion" and self.active_fusion_group is not None:
            mode_name = f"FUSED_{self.active_fusion_group}"
        else:
            mode_name = self.current_sensor.replace("/", "_")

        anno_name = os.path.basename(self.annos_folder.rstrip("/"))
        output_name = f"output_{self.recording}_{mode_name}_{anno_name}.mp4"

        print(
            f"[VIDEO] Creating video '{output_name}' at {fps} FPS. "
            f"mode={self.active_3d_mode}, "
            f"fusion_group={self.active_fusion_group}, "
            f"timeline_sensor={self.current_sensor}"
        )

        view_ctl = self.vis.get_view_control()

        # Important: grab the camera after the user has manually positioned the view.
        fixed_cam_params = view_ctl.convert_to_pinhole_camera_parameters()

        with imageio.get_writer(output_name, fps=fps) as writer:
            for i in range(len(matches)):
                print(f"[VIDEO] Rendering frame {i + 1}/{len(matches)}")

                # Avoid double-render from slider signal; render explicitly below.
                self.slider.blockSignals(True)
                self.slider.setValue(i)
                self.slider.blockSignals(False)

                # Render current frame according to active mode:
                #   single -> load_pointcloud()
                #   fusion -> visualize_sensor_group(...)
                self.render_current_3d_frame()

                # Add 3D boxes for this frame.
                self.remove_all_bboxes()
                self.on_load_boxes_clicked()
                self.on_load_3d_lane_lines_clicked()

                # Re-apply the fixed view before capture.
                try:
                    view_ctl.convert_from_pinhole_camera_parameters(fixed_cam_params)
                except Exception as e:
                    print("[VIDEO] Failed to restore fixed view:", e)

                self.vis.poll_events()
                self.vis.update_renderer()

                float_img = self.vis.capture_screen_float_buffer(do_render=True)
                if float_img is None:
                    print(f"[VIDEO] Warning: failed to capture screen for frame {i}")
                    continue

                np_img = (np.asarray(float_img) * 255).astype(np.uint8)
                writer.append_data(np_img)

        # Restore view at end.
        try:
            view_ctl.convert_from_pinhole_camera_parameters(fixed_cam_params)
        except Exception:
            pass

        print(f"[VIDEO] Video saved to '{output_name}'")

    def on_generate_video_camera_clicked(self):
        """
        Generate a video for the currently selected camera by iterating over
        all available annotation indices, loading the matching camera image,
        drawing bounding boxes, and appending frames to a video using imageio.
        """

        import imageio

        # 1) Identify which camera is selected
        camera_topic = self.current_camera_topic  # e.g. 'camera/front_center'
        data_camera = self.sensors_data[camera_topic]
        # matches_camera is a dict of { label_idx : camera_idx }
        matches_camera = data_camera["matches"]
        files_camera = data_camera["files"]
        folder_camera = data_camera["folder"]

        if not matches_camera:
            print(f"No camera matches found for topic: {camera_topic}")
            return

        # We'll write a .mp4 video with a certain FPS.
        # Make sure you have installed imageio[ffmpeg]: `pip install imageio imageio-ffmpeg`
        sensor_name = camera_topic.replace("/", "_")
        output_name = f"output_{self.recording}_{sensor_name}_{self.annos_folder.split('/')[-1]}.mp4"
        fps = 5
        writer = imageio.get_writer(output_name, fps=fps)

        print(f"Generating camera video for topic '{camera_topic}' -> {output_name}")

        # 2) Iterate over all annotation indices in sorted order
        #    (or you could just do for i in range(len(self.annos)) if that’s your use-case)
        all_label_indices = sorted(matches_camera.keys())
        for label_idx in all_label_indices:
            camera_idx = matches_camera[label_idx]  # which camera file index
            if camera_idx is None:
                # This means no valid camera frame matched that annotation index
                continue

            # 2a) Load the camera image
            file_name = files_camera[camera_idx]
            image_path = os.path.join(folder_camera, file_name)
            image_rgb = self.load_image_for_camera(
                image_path
            )  # returns a PIL Image or None
            if image_rgb is None:
                print(f"Failed to load camera image: {image_path}")
                continue

            # 2b) Draw bounding boxes onto the image if you want them
            #     Reuse logic from `on_visualize_2d`, but do it in memory (no popup).
            # image_with_boxes = self.draw_boxes_on_image(image_rgb, label_idx, camera_topic)
            image_with_boxes = self.draw_boxes_on_image(
                image_rgb, label_idx, camera_topic
            )
            # image_with_boxes = self.draw_lanes_on_image_release(image_with_boxes, label_idx, camera_topic)

            # 2c) Convert the result to a NumPy array (H, W, 3) in [0..255] for imageio
            np_img = np.array(image_with_boxes)  # from PIL to NumPy
            # Make sure it’s uint8. If it's already an 8-bit image, this is fine.

            # 2d) Append this frame to the video
            writer.append_data(np_img)

        # 3) Close the video writer
        writer.close()
        print(f"Camera video saved to '{output_name}'")

    def show_single_image_popup(self, qimg: QImage):
        """
        Show the image in a single reusable popup window.
        If one is already open, close it before opening a new one.
        """
        # Close previous window if it exists
        old = getattr(self, "image_popup", None)
        if old is not None:
            try:
                old.close()
                old.deleteLater()
            except Exception:
                pass
            self.image_popup = None

        # Create new window (parented to main widget)
        win = ImagePopup(self)
        win.setAttribute(Qt.WA_DeleteOnClose, True)

        # When user closes it, clear the reference
        win.destroyed.connect(lambda *_: setattr(self, "image_popup", None))

        win.set_image(qimg)
        win.show()

        self.image_popup = win

    def rebuild_label_to_slider_map(self):
        """
        For the currently selected sensor, map label_idx -> first slider index where it appears.
        This lets us jump the slider when user selects a lane frame.
        """
        data = self.sensors_data[self.current_sensor]
        matches = data.get("matches", [])
        m = {}
        for i, pair in enumerate(matches):
            if not pair or len(pair) < 2:
                continue
            _, label_idx = pair
            if label_idx not in m:
                m[label_idx] = i
        self.label_to_slider_idx = m

    def rebuild_lane_label_indices_for_current_camera(self):
        """
        Build label indices where both the current camera image and a
        lane-line JSON exist for the same sync key.
        """
        if not self.lane_sync_to_file and not self.lane_ts_to_file:
            self.lane_label_indices = []
            return

        cam_topic = self.current_camera_topic
        data_camera = self.sensors_data.get(cam_topic, None)
        if data_camera is None:
            self.lane_label_indices = []
            return

        matches_camera = data_camera.get("matches", {})
        if not matches_camera:
            self.lane_label_indices = []
            return

        lane_idxs = []
        for label_idx in sorted(matches_camera.keys()):
            sync_id, ts = self._annotation_file_sync_and_timestamp(label_idx)
            has_lane = False
            if sync_id is not None and int(sync_id) in self.lane_sync_to_file:
                has_lane = True
            elif ts is not None and int(ts) in self.lane_ts_to_file:
                has_lane = True
            if has_lane:
                lane_idxs.append(label_idx)

        self.lane_label_indices = lane_idxs
        print(
            f"[LANE] Synced lane frames for {cam_topic}: {len(self.lane_label_indices)}"
        )

    def on_lane_frames_dialog(self):
        """
        Show a dialog listing only the frames (timestamps) where lane json exists
        AND the current camera has a matched image. On selection: jump + visualize.
        """
        self.rebuild_lane_label_indices_for_current_camera()
        if not self.lane_label_indices:
            print("[LANE] No synced lane frames available (lane json + camera match).")
            return

        # Ensure we can jump the slider
        if not self.label_to_slider_idx:
            self.rebuild_label_to_slider_map()

        # Build display strings
        items = []
        for label_idx in self.lane_label_indices:
            ts = os.path.splitext(os.path.basename(self.annos[label_idx]))[0]
            items.append(f"{ts}  |  label_idx={label_idx}")

        dlg = QDialog(self)
        dlg.setWindowTitle("Synced lane frames (lane json + camera match)")
        layout = QVBoxLayout(dlg)

        combo = QComboBox(dlg)
        combo.addItems(items)
        layout.addWidget(combo)

        btn_row = QHBoxLayout()
        show_btn = QPushButton("Jump + Visualize 2D", dlg)
        close_btn = QPushButton("Close", dlg)
        btn_row.addWidget(show_btn)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        def _do_show():
            s = combo.currentText()
            # parse label_idx from " ... label_idx=NNN"
            try:
                label_idx = int(s.split("label_idx=")[-1].strip())
            except Exception:
                print("[LANE] Could not parse label_idx from selection.")
                return

            # Jump slider to the corresponding frame for current sensor
            slider_idx = self.label_to_slider_idx.get(label_idx, None)
            if slider_idx is None:
                # rebuild and try again (sensor may have changed)
                self.rebuild_label_to_slider_map()
                slider_idx = self.label_to_slider_idx.get(label_idx, None)

            if slider_idx is None:
                print(
                    f"[LANE] Could not find slider index for label_idx={label_idx} in current sensor."
                )
                return

            self.slider.setValue(slider_idx)

            # Use your existing 2D viewer (this will draw boxes + lanes as you implemented)
            self.on_visualize_lane_lines_2d()

        show_btn.clicked.connect(_do_show)
        close_btn.clicked.connect(dlg.accept)

        dlg.exec_()

    # --------------------------------------------------------------------------
    # Sensor-aware color field menu
    # --------------------------------------------------------------------------
    def update_color_combo_for_sensor(self):
        """Refresh color-field choices for the active sensor.

        RCS is radar-only; Ouster has no velocity; Aeva has velocity and
        reflectivity. The dropdown is rebuilt on sensor changes so the label
        always matches the actual column being visualized.
        """
        if not hasattr(self, "combo"):
            return
        previous = self.combo.currentText() if self.combo.count() else None
        fields = get_color_fields_for_sensor(self.current_sensor)
        if not fields:
            fields = ["Distance"]
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItems(fields)
        if previous in fields:
            self.combo.setCurrentText(previous)
        self.combo.blockSignals(False)

    # --------------------------------------------------------------------------
    # Sensor Selection
    # --------------------------------------------------------------------------
    def on_sensor_changed(self):
        """
        Callback when the sensor combo changes (lidar/radar).
        We'll reset self.current_sensor, update slider range, reload the cloud, etc.
        """
        self.current_sensor = self.sensor_combo.currentText()

        self.active_3d_mode = "single"
        self.active_fusion_group = None
        if self.current_sensor not in self.sensors_data:
            print(f"Sensor not found in sensors_data: {self.current_sensor}")
            return

        # Reset the slider range based on how many matches we have
        data = self.sensors_data[self.current_sensor]
        n_matches = len(data["matches"])
        self.slider.setValue(0)
        self.slider.setMinimum(0)
        if n_matches > 0:
            self.slider.setMaximum(n_matches - 1)
        else:
            self.slider.setMaximum(0)

        # Optionally clear old bounding box linesets if you don't want them
        for ls in self.box_linesets:
            self.vis.remove_geometry(ls, reset_bounding_box=False)
        self.box_linesets.clear()

        self.remove_all_lane_lines_3d()

        # removing boxes
        self.box_linesets = list()

        # Refresh legal color fields before reloading. For example, RCS is radar-only.
        self.update_color_combo_for_sensor()

        # Reload the cloud for the new sensor at index=0
        self.load_pointcloud()
        self.label_to_slider_idx = {}
        self.lane_label_indices = []

    def on_camera_changed(self):
        self.current_camera_topic = self.camera_combo.currentText()
        print(f"Camera changed to: {self.current_camera_topic}")
        self.label_to_slider_idx = {}
        self.lane_label_indices = []

    # --------------------------------------------------------------------------
    # Loading the PointCloud
    # --------------------------------------------------------------------------
    def load_pointcloud(self):
        if self.current_sensor not in self.sensors_data:
            print("No point-cloud sensor is available.")
            return

        print(f"[SINGLE] Loading single sensor: {self.current_sensor}")
        data = self.sensors_data[self.current_sensor]
        matches = data["matches"]
        if not matches:
            print(f"No matches found for sensor {self.current_sensor}.")
            return

        idx = self.slider.value()
        idx_file = matches[idx][0]  # index of pointcloud is first element
        file_name = data["files"][idx_file]
        file_path = os.path.join(data["folder"], file_name)
        print("Loading:", file_path)

        if "lidar" in self.current_sensor:
            if "joint" in self.current_sensor:
                pc_data = load_lidar_joint_point_cloud(file_path)
            elif "accumulated" in self.current_sensor:
                pc_data = load_lidar_accumulated_point_cloud(file_path)
            elif "ouster" in self.current_sensor:
                pc_data = load_single_ouster_lidar_point_cloud(file_path)
            else:
                pc_data = load_single_lidar_point_cloud(file_path)
        elif "radar" in self.current_sensor:
            if "joint" in self.current_sensor:
                pc_data = load_radar_joint_point_cloud(file_path)
            else:
                pc_data = load_single_radar_point_cloud(file_path)
        else:
            print(f"Unknown sensor {self.current_sensor}, no load function!")
            return

        if pc_data.shape[0] == 0:
            print("Warning: Empty point cloud.")
            return

        try:
            pc_data = self.transform_pointcloud_to_display_frame(
                pc_data,
                self.current_sensor,
            )
        except Exception as e:
            print(
                f"[FRAME] Failed to transform point cloud from "
                f"{self.current_sensor} into display frame {self.get_display_frame()}: {e}"
            )
            return

        self.update_pcd(pc_data)

    def load_lane_objects_for_label(self, label_idx: int):
        """
        Load lane-line annotation for the same sync key as the selected
        bounding-box annotation.
        """
        if self.bev_lane_scene_dir is None:
            return None

        sync_id, ts = self._annotation_file_sync_and_timestamp(label_idx)
        lane_file = None
        if sync_id is not None:
            lane_file = self.lane_sync_to_file.get(int(sync_id))
        if lane_file is None and ts is not None:
            lane_file = self.lane_ts_to_file.get(int(ts))
        if lane_file is None:
            return None

        lane_path = os.path.join(self.bev_lane_scene_dir, lane_file)
        if not os.path.isfile(lane_path):
            return None

        try:
            with open(lane_path, "r") as f:
                objs = json.load(f)
            if isinstance(objs, list):
                return objs
            if isinstance(objs, dict):
                # Some exports wrap objects under a top-level key; accept either.
                for key in ("objects", "annotations", "data"):
                    if isinstance(objs.get(key), list):
                        return objs[key]
                return [v for v in objs.values() if isinstance(v, dict)]
            return None
        except Exception as e:
            print("[LANE] Failed to read lane json:", lane_path, e)
            return None

    def get_release_K_and_Tcam(
        self,
        label_idx: int,
        camera_topic: str,
        allow_nearest: bool = True,
        max_nearest_dt_ns: int = 50_000_000,
    ):
        """

        Returns:
          K (3x3) from calibrations/calib_camera_leopard_<camera>.json,
          T_cam_cab (4x4) that maps lane points from cab frame into camera frame,
          and the annotation timestamp stem used for logging.
        """
        try:
            P, _im_h, _im_w, _calib_path = self.load_camera_projection(camera_topic)
            K = P[:, :3].copy()
            T_cam_cab = load_extrinsic_between_nodes(
                self.data_extrinsics,
                "velodyne",
                sensor_path[camera_topic],
            )
            ts = os.path.splitext(os.path.basename(self.annos[label_idx]))[0]
            return K, T_cam_cab, ts
        except Exception as e:
            print(f"[LANE] Could not load camera calibration/T for {camera_topic}: {e}")
            return None, None, None

    def draw_lanes_on_image_release(self, pil_image, label_idx: int, camera_topic: str):
        """
        Draw lane_line and lane_segment polylines onto a PIL image using:
          - points in CAB frame (treated as ego/cab)
          - K + extrinsics from release json (per timestamp)
          - wide-camera stored rotation correction
        """
        print(
            f"[LANE] ENTER draw_lanes_on_image_release cam_topic={camera_topic} label_idx={label_idx}"
        )

        lane_objs = self.load_lane_objects_for_label(label_idx)
        if lane_objs is None:
            return pil_image

        K, T_cam_cab, used_ts = self.get_release_K_and_Tcam(
            label_idx, camera_topic, allow_nearest=True
        )
        if K is None or T_cam_cab is None:
            print(
                f"[LANE] Missing K/T for cam_topic={camera_topic} label_idx={label_idx} (returning early)"
            )
            return pil_image

        draw = ImageDraw.Draw(pil_image)
        Ws, Hs = pil_image.size  # PIL: (W,H)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        # camera name used for wide fix
        parts = (camera_topic or "").split("/")
        cam_name = parts[2] if len(parts) >= 3 else camera_topic
        need_rot = False
        rot_dir = None  # decide once for this frame (stable)

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        for obj in lane_objs:
            cls = obj.get("obj_class", "")
            if cls not in ("lane_line", "lane_segment"):
                continue

            pts = np.array(obj.get("points", []), dtype=np.float64)
            if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 2:
                continue

            pts_cam = transform_points_4x4_np(pts, T_cam_cab)  # Nx3
            Z = pts_cam[:, 2]
            valid = Z > 1e-3
            if np.count_nonzero(valid) < 2:
                continue

            pts_cam = pts_cam[valid]
            X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]

            u0 = fx * (X / Z) + cx
            v0 = fy * (Y / Z) + cy

            if need_rot:
                if rot_dir is None:
                    rot_dir = choose_best_rotation_upright_to_stored(u0, v0, Hs, Ws)
                u, v = rotate_uv_upright_to_stored_90(u0, v0, Hs, Ws, rot_dir)
            else:
                u, v = u0, v0

            inside = (u >= 0) & (u < Ws - 1) & (v >= 0) & (v < Hs - 1)
            if np.count_nonzero(inside) < 2:
                continue

            u = u[inside]
            v = v[inside]

            poly = [(int(round(ui)), int(round(vi))) for ui, vi in zip(u, v)]
            if len(poly) < 2:
                continue

            obj_id = int(obj.get("obj_id", 0))
            color = create_color_from_tracking_id(
                obj_id
            )  # your function returns RGB 0..255
            width = 4 if cls == "lane_line" else 2

            draw.line(poly, fill=color, width=width)

        print(
            f"[LANE] cam={cam_name} ts={used_ts} imgWxH={Ws}x{Hs} "
            f"fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}"
        )

        return pil_image

    def load_image_for_camera(self, image_path):
        if not os.path.exists(image_path):
            return None
        image_rgb = Image.open(image_path)  # shape (H, W, 3)
        return image_rgb

    def draw_boxes_on_image(self, pil_image, label_idx, camera_topic):
        draw = ImageDraw.Draw(pil_image)

        json_file_path = os.path.join(self.annos_folder, self.annos[label_idx])
        if not os.path.exists(json_file_path):
            print(f"JSON file not found: {json_file_path}")
            return pil_image

        boxes = load_boxes_from_json(json_file_path)
        T_src_to_tgt = load_extrinsic_between_nodes(
            self.data_extrinsics,
            sensor_path["annotations/bounding_boxes"],
            sensor_path[camera_topic],
        )
        camera_idx = self.camera_names.index(self.current_camera_topic)
        short_camera_name = self.short_camera_names[camera_idx]
        # Camera calibrations live under scene/calibrations/.
        try:
            P, im_h, im_w, calib_path = self.load_camera_projection(camera_topic)
        except Exception as e:
            print(f"Camera calibration does not exist for {camera_topic}: {e}")
            return pil_image
        for b in boxes:
            if self.is_ego_box(b):
                continue

            tracking_id = int(b["Tracking_ID"])
            color = create_color_from_tracking_id(tracking_id)
            # print(color)
            x_c = b["x"]
            y_c = b["y"]
            z_c = b["z"]
            l = b["l"]
            w = b["w"]
            h = b["h"]
            yaw_deg = b["yaw"]
            if not (l == -1 and w == -1 and h == -1):
                corners_src = get_3d_box_corners(x_c, y_c, z_c, l, w, h, yaw_deg)

                existing_edges = list()
                stored_points = dict()
                edges = [
                    (0, 1),
                    (1, 3),
                    (2, 3),
                    (2, 0),
                    (4, 5),
                    (5, 7),
                    (6, 7),
                    (6, 4),
                    (0, 4),
                    (1, 5),
                    (2, 6),
                    (3, 7),
                ]

                fx, fy, cx, cy = extract_fx_fy_cx_cy(P)
                corners_tgt = transform_points_4x4(
                    corners_src, T_src_to_tgt
                )  # [:,(1,2,0,3)]
                clipped_pts = corners_tgt
                a, edges = clip_box_against_camera_fov(
                    corners_tgt[:, :3], edges, im_w, im_h, fx, fy, cx, cy
                )
                clipped_pts = np.ones((a.shape[0], 4))
                clipped_pts[:, :3] = a

                c_2d = (P @ clipped_pts.T).T
                zc = c_2d[:, 2] + 1e-6
                mean_distance = np.mean(zc)
                c_2d[:, 0] /= zc + +1e-6
                c_2d[:, 1] /= zc + 1e-6
            else:
                mean_distance = -1

            # 4. Draw lines for each edge
            if mean_distance > -1:
                for start, end in edges:
                    x1, y1 = c_2d[start][:2]
                    x2, y2 = c_2d[end][:2]

                    # Check for NaN or Inf
                    if (
                        math.isnan(x1)
                        or math.isnan(y1)
                        or math.isnan(x2)
                        or math.isnan(y2)
                        or math.isinf(x1)
                        or math.isinf(y1)
                        or math.isinf(x2)
                        or math.isinf(y2)
                    ):
                        print(f"Skipping invalid corners: {(x1, y1)}, {(x2, y2)}")
                        continue

                    # Convert to int and clamp within the image if necessary
                    p1 = (int(round(x1)), int(round(y1)))
                    p2 = (int(round(x2)), int(round(y2)))
                    draw.line([p1, p2], fill=color, width=2)
                font = ImageFont.truetype("DejaVuSans.ttf", 60)
                draw.text(p1, str(tracking_id), color, font=font)
            best_box_2d = b.get(short_camera_name, None)
            w_c, h_c = center_crop_lookup[short_camera_name]
            if best_box_2d is not None:
                x0 = best_box_2d["x0"]
                y0 = best_box_2d["y0"]
                x1 = best_box_2d["x1"]
                y1 = best_box_2d["y1"]
                draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            w, h = pil_image.size

        return pil_image

    def on_visualize_boxes_2d(self):
        self.on_visualize_2d(draw_boxes=True, draw_lane_lines=False)

    def on_visualize_lane_lines_2d(self):
        self.on_visualize_2d(draw_boxes=False, draw_lane_lines=True)

    def on_visualize_2d(self, draw_boxes=False, draw_lane_lines=False):
        if not self.current_camera_topic:
            print("No camera selected!")
            return

        if self.current_sensor not in self.sensors_data:
            print("No point-cloud sensor is selected.")
            return

        data = self.sensors_data[self.current_sensor]
        matches = data["matches"]
        idx = self.slider.value()
        if not matches or idx >= len(matches):
            print("No valid match or slider index out of range.")
            return

        lidar_idx, label_idx = matches[idx]

        data_camera = self.sensors_data[self.current_camera_topic]
        matches_camera = data_camera["matches"]
        camera_idx = matches_camera.get(label_idx, None)
        if camera_idx is None:
            print("No valid label in camera frame available for this timestamp.")
            return

        file_name = data_camera["files"][camera_idx]
        folder = data_camera["folder"]
        image_path = os.path.join(folder, file_name)

        image = self.load_image_for_camera(image_path)
        if image is None:
            print("Failed to load image.")
            return

        overlay = image.copy()

        # --- Draw boxes (may early-return internally for some cameras) ---
        if draw_boxes:
            try:
                overlay = self.draw_boxes_on_image(
                    overlay, label_idx, self.current_camera_topic
                )
            except Exception as e:
                print("[2D] draw_boxes_on_image failed:", e)

        if draw_lane_lines:
            # --- ALWAYS draw lanes using release intr/extr (independent of calib_*.json) ---
            try:
                overlay = self.draw_lanes_on_image_release(
                    overlay, label_idx, self.current_camera_topic
                )
            except Exception as e:
                print("[2D] draw_lanes_on_image_release failed:", e)

        def numpy_to_qimage(image_pil) -> QImage:
            image_pil = image_pil.resize((1920, 1024))
            rgb_array = np.array(image_pil)
            if rgb_array.dtype != np.uint8:
                raise ValueError("Only uint8 images are supported.")
            rgb_array = np.ascontiguousarray(rgb_array)
            h, w, ch = rgb_array.shape
            bytesPerLine = ch * w
            return QImage(rgb_array.tobytes(), w, h, bytesPerLine, QImage.Format_RGB888)

        qimg = numpy_to_qimage(overlay)

        self.show_single_image_popup(qimg)

        # self.open_image_windows.append(img_win)

    def remove_all_bboxes(self):
        # For each lineset reference, remove it with reset_bounding_box=False
        for ls in self.box_linesets:
            # if self.vis.has_geometry(ls):
            self.vis.remove_geometry(ls, reset_bounding_box=False)

        # Now clear out our list since they're no longer in the scene
        self.box_linesets.clear()

        self.vis.poll_events()
        self.vis.update_renderer()

    def remove_all_lane_lines_3d(self):
        """
        Remove all currently displayed 3D lane-line geometries.
        """
        for ls in self.lane_linesets_3d:
            try:
                self.vis.remove_geometry(ls, reset_bounding_box=False)
            except Exception:
                pass

        self.lane_linesets_3d.clear()

        self.vis.poll_events()
        self.vis.update_renderer()

    def on_load_3d_lane_lines_clicked(self):
        """
        Load lane-line annotations for the current synchronized frame and visualize
        them as 3D polylines in the current selected 3D sensor frame.
        """
        if self.current_sensor not in self.sensors_data:
            print("[LANE-3D] No current 3D sensor selected.")
            return

        label_idx = self.get_current_label_idx()
        if label_idx is None:
            print("[LANE-3D] Could not determine current label_idx.")
            return

        lane_objs = self.load_lane_objects_for_label(label_idx)
        if not lane_objs:
            print(f"[LANE-3D] No lane-line objects found for label_idx={label_idx}.")
            return

        # Source frame for lane points.
        lane_source_frame = sensor_path.get("annotations/lane_lines", None)
        if lane_source_frame is None:
            lane_source_frame = sensor_path.get(
                "annotations/bounding_boxes", "velodyne"
            )

        # Target frame is the current selected 3D sensor frame.
        try:
            target_frame = self.get_display_frame()
        except Exception as e:
            print(
                f"[LANE-3D] Could not resolve target frame for {self.current_sensor}: {e}"
            )
            return

        try:
            if lane_source_frame == target_frame:
                T_lane_to_target = np.eye(4, dtype=np.float64)
            else:
                T_lane_to_target = load_extrinsic_between_nodes(
                    self.data_extrinsics,
                    lane_source_frame,
                    target_frame,
                )
        except Exception as e:
            print(
                f"[LANE-3D] Could not compute transform "
                f"{lane_source_frame} -> {target_frame}: {e}"
            )
            return

        # Remove old 3D lane lines before adding the new frame.
        self.remove_all_lane_lines_3d()

        added = 0
        skipped = 0

        for obj in lane_objs:
            cls = obj.get("obj_class", "")
            if cls not in ("lane_line", "lane_segment"):
                continue

            pts = np.asarray(obj.get("points", []), dtype=np.float64)

            if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 2:
                skipped += 1
                continue

            pts_target = transform_points_4x4_np(pts, T_lane_to_target)

            # Remove invalid points.
            finite = np.all(np.isfinite(pts_target), axis=1)
            pts_target = pts_target[finite]

            if pts_target.shape[0] < 2:
                skipped += 1
                continue

            lines = [[i, i + 1] for i in range(pts_target.shape[0] - 1)]

            if not lines:
                skipped += 1
                continue

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(pts_target)
            line_set.lines = o3d.utility.Vector2iVector(lines)

            obj_id = int(obj.get("obj_id", added))
            color_rgb_255 = create_color_from_tracking_id(obj_id)
            color_rgb = [float(c) / 255.0 for c in color_rgb_255]

            # Open3D legacy LineSet colors are per-line.
            line_set.colors = o3d.utility.Vector3dVector([color_rgb for _ in lines])

            self.vis.add_geometry(line_set, reset_bounding_box=False)
            self.lane_linesets_3d.append(line_set)
            added += 1

        self.vis.poll_events()
        self.vis.update_renderer()

        print(
            f"[LANE-3D] Added {added} lane polylines, skipped {skipped}. "
            f"label_idx={label_idx}, source_frame={lane_source_frame}, "
            f"target_frame={target_frame}, sensor={self.current_sensor}"
        )

    def on_load_boxes_clicked(self):
        """
        Slot called when user clicks "Load JSON Boxes" button.
        For demonstration, let's pick a path or use a file dialog.
        """
        data = self.sensors_data[self.current_sensor]
        matches = data["matches"]
        if not matches:
            print(f"No matches found for sensor {self.current_sensor}.")
            return

        idx = self.slider.value()
        json_file_path = os.path.join(self.annos_folder, self.annos[matches[idx][1]])
        if not os.path.exists(json_file_path):
            print(f"JSON file not found: {json_file_path}")
            return

        # Optionally clear old bounding box linesets if you don't want them
        self.remove_all_bboxes()

        # Save current camera parameters
        view_ctl = self.vis.get_view_control()
        cam_params = view_ctl.convert_to_pinhole_camera_parameters()

        # Load a list of boxes from the JSON
        boxes = load_boxes_from_json(json_file_path)
        annotation_frame = sensor_path.get("annotations/bounding_boxes", "velodyne")
        display_frame = self.get_display_frame()

        if annotation_frame == display_frame:
            T_src_to_tgt = np.eye(4, dtype=np.float64)
        else:
            T_src_to_tgt = load_extrinsic_between_nodes(
                self.data_extrinsics,
                annotation_frame,
                display_frame,
            )
        # Create a lineset for each box
        for b in boxes:
            x_c = b["x"]
            y_c = b["y"]
            z_c = b["z"]
            tracking_id = int(b["Tracking_ID"])
            color = [float(c) / 255 for c in create_color_from_tracking_id(tracking_id)]
            l = b["l"]
            w = b["w"]
            h = b["h"]

            yaw_deg = b["yaw"]

            if (
                np.isnan(x_c)
                or np.isnan(y_c)
                or np.isnan(z_c)
                or np.isnan(l)
                or np.isnan(w)
                or np.isnan(h)
                or np.isnan(yaw_deg)
            ):
                continue
            corners_src = get_3d_box_corners(x_c, y_c, z_c, l, w, h, yaw_deg)
            corners_tgt = transform_points_4x4(corners_src, T_src_to_tgt)
            b = from_corners_to_obb(corners_tgt)
            x = b["x"]
            y = b["y"]
            z = b["z"]
            l = b["l"]
            w = b["w"]
            h = b["h"]
            yaw = b["yaw"]
            lineset = create_obb_lineset(x, y, z, l, w, h, yaw, color=color)
            self.box_linesets.append(lineset)

        for ls in self.box_linesets:
            self.vis.add_geometry(ls, reset_bounding_box=False)

        view_ctl.convert_from_pinhole_camera_parameters(cam_params)

        self.vis.poll_events()
        self.vis.update_renderer()
        print(f"Loaded {len(self.box_linesets)} boxes from JSON.")

    def _colors_for_current_sensor(self, pc_data: np.ndarray, cidx: int) -> np.ndarray:
        """
        Robust color extraction for the binary layouts.

        Combo order: ID, Velocity, RCS, Intensity.
        """
        n = pc_data.shape[0]
        if n == 0:
            return np.zeros((0, 3), dtype=np.float64)

        sensor = self.current_sensor or ""
        col = None
        if "radar" in sensor:
            # Radar 33 columns: rangerate=3, rcs=4, amplitude=5, sensor_id=26.
            col_map = {0: 26, 1: 3, 2: 4, 3: 5}
            col = col_map.get(cidx)
        elif "ouster" in sensor:
            # Ouster 7 columns: intensity=3, rel_time_ns=4, reflectivity=5, ring=6.
            col_map = {0: 6, 1: 4, 2: 5, 3: 3}
            col = col_map.get(cidx)
        elif "lidar" in sensor:
            # Aeva joint 11 columns: intensity=3, velocity=4, reflectivity=5, sensor_id=7.
            col_map = {0: 7, 1: 4, 2: 5, 3: 3}
            col = col_map.get(cidx)

        if col is None or col >= pc_data.shape[1]:
            return np.tile(np.array([[0.65, 0.65, 0.65]], dtype=np.float64), (n, 1))

        vals = np.asarray(pc_data[:, col], dtype=np.float64)
        finite = np.isfinite(vals)
        if not np.any(finite):
            return np.tile(np.array([[0.65, 0.65, 0.65]], dtype=np.float64), (n, 1))

        lo = np.nanpercentile(vals[finite], 2)
        hi = np.nanpercentile(vals[finite], 98)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0
        norm = np.clip((vals - lo) / (hi - lo), 0.0, 1.0)
        norm[~finite] = 0.0
        cmap = plt.get_cmap("viridis")
        return np.asarray(cmap(norm)[:, :3], dtype=np.float64)

    def update_pcd(self, pc_data):
        """
        Update the main point cloud geometry and color in Open3D.
        """

        view_ctl = self.vis.get_view_control()
        cam_params = view_ctl.convert_to_pinhole_camera_parameters()

        self.vis.remove_geometry(self.pcd, reset_bounding_box=False)

        self.remove_all_bboxes()
        self.remove_all_lane_lines_3d()

        points = pc_data[:, :3]
        if points.shape[0] == 0:
            print("Warning: No points to visualize!")
            return

        self.pcd.clear()
        self.pcd.points = o3d.utility.Vector3dVector(points)

        field_name = self.combo.currentText() if self.combo.count() else "ID"
        colors = colorize_pointcloud(
            pc_data, field_name, sensor_name=self.current_sensor
        )
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

        self.vis.add_geometry(self.pcd, reset_bounding_box=False)
        self.vis.add_geometry(self.coordinate_frame, reset_bounding_box=False)

        # Return camera view to original position
        view_ctl.convert_from_pinhole_camera_parameters(cam_params)

        self.vis.poll_events()
        self.vis.update_renderer()

    def update_colorization(self):
        """
        Called when the combo box changes (ID/Velocity/RCS/Intensity).
        Simply reload the current cloud with new color scheme.
        """
        self.load_pointcloud()

    def next_pointcloud(self):
        """
        Go to the next point cloud file by incrementing slider.
        """
        current_value = self.slider.value()
        if current_value < self.slider.maximum():
            self.slider.setValue(current_value + 1)

    # --------------------------------------------------------------------------
    # Periodic update (keeps O3D window alive)
    # --------------------------------------------------------------------------
    def update_visualizer(self):
        self.vis.poll_events()
        self.vis.update_renderer()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize a TruckDrive scene.")
    parser.add_argument(
        "--root-dir",
        default="datasets/TruckDrive",
        help="Dataset root containing scene_XX_N folders.",
    )
    parser.add_argument(
        "--recording",
        default="scene_28_1",
        help="Scene folder name, for example scene_28_1.",
    )
    parser.add_argument(
        "--ptype",
        default="lidar",
        choices=["lidar", "radar"],
        help="Initial point-cloud type label retained for compatibility.",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = PointCloudVisualizer(args.root_dir, args.recording, ptype=args.ptype)
    window.show()
    sys.exit(app.exec_())
