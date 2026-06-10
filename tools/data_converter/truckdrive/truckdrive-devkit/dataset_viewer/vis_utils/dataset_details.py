"""
Dataset layout description for the public TruckDrive scenes.

Expected scene layout:

scene_XX_N/
  calibrations/
    calib_tf_tree_full.json
    calib_camera_leopard_<camera>.json
  camera/leopard/<camera>/images/*.jpg
  lidar/aeva/joint_lidars/points/*.bin
  lidar/ouster/<ouster>/points/*.bin
  radar/conti542/joint_radars/detections/*.bin
  annotations/bounding_boxes/*.json
  annotations/lane_lines/*.json
  poses/gt_trajectory.txt

File names are synchronized as <sync_number>_<normalized_timestamp>.<ext>.
"""

# -----------------------------------------------------------------------------
# Dataset relative folders
# -----------------------------------------------------------------------------
calibrations_relpath = "calibrations"
tf_tree_relpath = "calibrations/calib_tf_tree_full.json"
annotation_bounding_boxes_relpath = "annotations/bounding_boxes"
annotation_lane_lines_relpath = "annotations/lane_lines"
gt_trajectory_relpath = "poses/gt_trajectory.txt"

# -----------------------------------------------------------------------------
# LiDAR setup
# -----------------------------------------------------------------------------
# Aeva joint lidar points are 11 float64 columns:
# x, y, z, intensity, velocity, reflectivity, time_offset_ns, sensor_id, vx, vy, vz
lidar_topics = ["/lidar/aeva/joint_lidars/points"]
lidar_tf_nodes = ["lidar_aeva_forward_center_wide"]
lidar_names = ["joint_lidars"]

# Ouster points are 7 float32 columns:
# x, y, z, intensity, rel_time_ns, reflectivity, ring
ouster_names = ["forward_center", "sideward_left", "sideward_right"]
ouster_topics = [f"/lidar/ouster/{name}/points" for name in ouster_names]
ouster_tf_nodes = [f"lidar_ouster_{name}" for name in ouster_names]

lidar_topics.extend(ouster_topics)
lidar_tf_nodes.extend(ouster_tf_nodes)
lidar_names.extend(ouster_names)

# -----------------------------------------------------------------------------
# Radar setup
# -----------------------------------------------------------------------------
# Joint radar points are 33 float64. The remaining order is:
# x, y, z, rangerate, rcs, amplitude, range_radar, azimuth, elevation,
# scan_id, detection_id, quality, is_rangerate_ambiguous, rangerate_ambiguity,
# cluster_range_spread, cluster_azimuth_spread, cluster_elevation_spread,
# visited, clustered, pose_x, pose_y, pose_z,
# orientation_x, orientation_y, orientation_z, orientation_w,
# sensor_id, vx, vy, vz, vx0, vy0, vz0
radar_names = ["joint_radars"]
radar_topics = ["/radar/conti542/joint_radars/detections"]
radar_tf_nodes = ["radar_conti542_forward_left_high"]

# -----------------------------------------------------------------------------
# Camera setup.
# -----------------------------------------------------------------------------
camera_setup = [
    "forward_center_medium",
    "forward_left_medium",
    "forward_left_narrow",
    "forward_left_wide",
    "forward_right_medium",
    "forward_right_narrow",
    "forward_right_wide",
    "rearward_left_bottom_medium",
    "rearward_left_top_medium",
    "rearward_right_bottom_medium",
    "rearward_right_top_medium",
    "sideward_left_front_wide",
    "sideward_left_back_wide",
    "sideward_right_front_wide",
    "sideward_right_back_wide",
]

correlation_group_front = [
    "forward_center_medium",
    "forward_left_medium",
    "forward_left_narrow",
    "forward_right_medium",
    "forward_right_narrow",
    "forward_left_wide",
    "forward_right_wide",
]
correletion_group_left = [
    "rearward_left_bottom_medium",
    "rearward_left_top_medium",
    "sideward_left_front_wide",
    "sideward_left_back_wide",
]
correletion_group_right = [
    "rearward_right_bottom_medium",
    "rearward_right_top_medium",
    "sideward_right_front_wide",
    "sideward_right_back_wide",
]

center_crop_lookup = {
    "forward_center_medium": [(200, -200), (200, -200)],
    "forward_left_medium": [(200, -200), (200, -200)],
    "forward_left_narrow": [(100, -100), (100, -100)],
    "forward_left_wide": [(350, -350), (300, -300)],
    "forward_right_medium": [(200, -200), (200, -200)],
    "forward_right_narrow": [(100, -100), (100, -100)],
    "forward_right_wide": [(350, -350), (300, -300)],
    "rearward_left_bottom_medium": [(200, -200), (200, -200)],
    "rearward_left_top_medium": [(200, -200), (200, -200)],
    "rearward_right_bottom_medium": [(200, -200), (200, -200)],
    "rearward_right_top_medium": [(200, -200), (200, -200)],
    "sideward_left_front_wide": [(350, -350), (300, -300)],
    "sideward_left_back_wide": [(350, -350), (300, -300)],
    "sideward_right_front_wide": [(350, -350), (300, -300)],
    "sideward_right_back_wide": [(350, -350), (300, -300)],
}

image_topics = [f"/camera/leopard/{name}/images" for name in camera_setup]
calib_file = [f"calibrations/calib_camera_leopard_{name}.json" for name in camera_setup]
camera_tf_nodes = [f"camera_leopard_{name}" for name in camera_setup]

# -----------------------------------------------------------------------------
# Transform-frame lookup. Keys are dataset-relative sensor paths without a
# leading slash; values are frame names in calibrations/calib_tf_tree_full.json.
# -----------------------------------------------------------------------------
sensor_path = {
    "annotations/bounding_boxes": "velodyne",
}

for lidar_topic, lidar_tf_node in zip(lidar_topics, lidar_tf_nodes):
    sensor_path[lidar_topic[1:]] = lidar_tf_node

for radar_topic, radar_tf_node in zip(radar_topics, radar_tf_nodes):
    sensor_path[radar_topic[1:]] = radar_tf_node

for image_topic, camera_tf_node in zip(image_topics, camera_tf_nodes):
    sensor_path[image_topic[1:]] = camera_tf_node

vehicle_frame = "vehicle"
cab_frame = "cab"
