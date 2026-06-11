# TruckDrive Dataset Viewer

Interactive visualization tools for the public TruckDrive dataset release.

The viewer supports synchronized 3D and 2D inspection of TruckDrive scenes, including LiDAR, radar, camera images, 3D bounding boxes, lane lines, accumulated depth overlays and video export.

## Expected Dataset Layout

The viewer expects a public TruckDrive scene directory with the following structure:

```text
TruckDrivePublic/
  scene_28_1/
    calibrations/
      calib_tf_tree_full.json
      calib_camera_leopard_<camera>.json

    camera/
      leopard/
        <camera>/
          images/
            <sync_id>_<normalized_timestamp>.jpg

    lidar/
      aeva/
        joint_lidars/
          points/
            <sync_id>_<normalized_timestamp>.bin

      ouster/
        <ouster_sensor>/
          points/
            <sync_id>_<normalized_timestamp>.bin

    radar/
      conti542/
        joint_radars/
          detections/
            <sync_id>_<normalized_timestamp>.bin

    annotations/
      bounding_boxes/
        <sync_id>_<normalized_timestamp>.json
      lane_lines/
        <sync_id>_<normalized_timestamp>.json

    accumulated_gt_depth/
      <camera>/
        <sync_id>_<normalized_timestamp>.npy

    poses/
      gt_trajectory.txt
```

File names use the synchronized public naming format:

```text
<sync_id>_<normalized_timestamp>.<extension>
```

For example:

```text
0063_3172004958.bin
0063_3172004958.jpg
0063_3172004958.json
```

## Coordinate Frames

The public scene calibration file:

```text
calibrations/calib_tf_tree_full.json
```

contains the static transform tree used by the viewer. Camera intrinsic and projection parameters are stored in:

```text
calibrations/calib_camera_leopard_<camera>.json
```

Frame names are defined in `vis_utils/dataset_details.py` through the `sensor_path` mapping. The viewer uses `velodyne` as the canonical Open3D display frame, because the public 3D bounding-box annotations are expressed in `velodyne`.

| Dataset component | Original frame | Frame used by the viewer |
|---|---|---|
| 3D bounding boxes | `velodyne` | `velodyne` |
| 3D lane lines | `velodyne` | `velodyne` |
| Aeva joint LiDAR | `lidar_aeva_forward_center_wide` | transformed to `velodyne` |
| Ouster forward-center LiDAR | `lidar_ouster_forward_center` | transformed to `velodyne` |
| Ouster sideward-left LiDAR | `lidar_ouster_sideward_left` | transformed to `velodyne` |
| Ouster sideward-right LiDAR | `lidar_ouster_sideward_right` | transformed to `velodyne` |
| Joint radar detections | `radar_conti542_forward_left_high` | transformed to `velodyne` |
| Camera images | `camera_leopard_<camera>` plus image pixel coordinates | selected camera image frame for 2D visualization |
| 2D bounding-box projection | `velodyne` | projected into the selected `camera_leopard_<camera>` frame |
| 2D lane-line projection | `velodyne` | projected into the selected `camera_leopard_<camera>` frame |
| Accumulated GT depth | image-aligned to the corresponding camera | selected camera image frame |

In the 3D Open3D window, the selected 3D sensor still controls the timeline, synchronization, and file matching. It does **not** define the final Open3D coordinate frame. All 3D point clouds, 3D boxes, and 3D lane lines are displayed in `velodyne` after the viewer applies the calibration-tree transforms.

For 2D visualization, the viewer does **not** use the Open3D display frame. Instead, it projects directly from the annotation or lane-line frame into the selected camera frame using `calib_tf_tree_full.json` and the corresponding camera calibration JSON.

## Environment Setup

Create a Conda environment:

```bash
conda create -n truckdrive_visualizer python=3.11 -y
conda activate truckdrive_visualizer
```

Configure Conda to use `conda-forge` for this environment:

```bash
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
```

Install the main dependencies:

```bash
conda install -y \
  numpy \
  pandas \
  scipy \
  scikit-learn \
  matplotlib \
  pillow \
  shapely \
  pyquaternion \
  pyqt \
  imageio
```

Install Open3D with pip:

```bash
python -m pip install open3d
```

Optional video export plugins:

```bash
python -m pip install "imageio[ffmpeg]"
python -m pip install "imageio[pyav]"
```

## Running the Viewer

From the dataset viewer folder, run:

```bash
python entrypoint.py \
  --root-dir /your_path_to/TruckDrivePublic \
  --recording scene_28_1
```

## Viewer Controls

### 3D Visualization

Use the **Select 3D Sensor** dropdown to choose a point-cloud source, such as:

```text
lidar/aeva/joint_lidars/points
lidar/ouster/forward_center/points
lidar/ouster/sideward_left/points
lidar/ouster/sideward_right/points
radar/conti542/joint_radars/detections
```

The point-cloud slider moves through synchronized frames.

Available 3D tools include:

```text
Load 3D Bounding Boxes
Visualize all Ouster sensors
Visualize all LiDAR sensors
Visualize all 3D sensors
Generate 3D Sensor Video
```

### 2D Visualization

Use the **Select 2D Camera** dropdown to choose a camera.

Available 2D tools include:

```text
Visualize boxes in 2D
Visualize lane lines in 2D
Visualize accumulated GT depth
Generate Video (Camera)
Generate Video (Lane lines)
```

## Synchronization Notes

The viewer matches files primarily by synchronized frame ID, not by exact timestamp. This is important because synchronized sensors may share the same `sync_id` while having slightly different normalized timestamps.

For example:

```text
lidar:   0063_3099941376.bin
camera:  0063_3172004958.jpg
radar:   0063_311698688.bin
```

These are treated as synchronized because they share:

```text
0063
```

# Point Cloud Binary Formats

This release uses cleaned binary point-cloud formats.

### Aeva Joint LiDAR

Loaded as `float64` with 11 columns:

```text
x, y, z,
intensity,
velocity,
reflectivity,
time_offset_ns,
sensor_id,
vx, vy, vz
```

### Ouster LiDAR

Loaded as `float32` with 7 columns:

```text
x, y, z,
intensity,
rel_time_ns,
reflectivity,
ring
```

### Conti542 Joint Radar

Loaded as `float64` with 33 columns.

```text
x, y, z,
rangerate,
rcs,
amplitude,
range_radar,
azimuth,
elevation,
scan_id,
detection_id,
quality,
is_rangerate_ambiguous,
rangerate_ambiguity,
cluster_range_spread,
cluster_azimuth_spread,
cluster_elevation_spread,
visited,
clustered,
pose_x,
pose_y,
pose_z,
orientation_x,
orientation_y,
orientation_z,
orientation_w,
sensor_id,
vx, vy, vz,
vx0, vy0, vz0
```

## Troubleshooting

If the Open3D window opens but no point cloud is visible, select another sensor once or press **Next**. The viewer should then load the current synchronized frame.

If video export fails, install one of the optional imageio backends:

```bash
python -m pip install "imageio[ffmpeg]"
```

or:

```bash
python -m pip install "imageio[pyav]"
```

If PyQt fails to start on a remote machine, make sure your display forwarding or virtual display setup is configured correctly.
