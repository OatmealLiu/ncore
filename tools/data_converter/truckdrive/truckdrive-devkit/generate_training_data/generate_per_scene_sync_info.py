import argparse
import json
import os
from glob import glob
from pathlib import Path
from tqdm import tqdm
import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial.transform import Rotation as R


KEY_MISSING = None  # 'Dropped' or 'None'


def parse_arguments():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description="Time synchronize all sensors and annotations."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./TruckDrive/",
        help="The data root directory.",
    )
    parser.add_argument(
        "--sync-info-root",
        type=str,
        default="./TruckDrive/sensor_sync/",
        help="The output directory for synchronized data.",
    )
    parser.add_argument(
        "--multiprocessing",
        action="store_true",
        help="Enable multiprocessing.",
    )
    args = parser.parse_args()
    return args


def sync_sensors(scene_path, data_root, output_root):
    output_file = os.path.join(output_root, "sync_" + Path(scene_path).name + ".json")

    # -- load cameras --
    cam_paths = glob(os.path.join(scene_path, "camera/leopard/*"))
    cameras_data = {}
    for cam_path in cam_paths:
        k = os.path.basename(cam_path)
        camera_samples = sorted(glob(os.path.join(cam_path, "images/*.jpg")))
        camera_data = []
        for sample in camera_samples:
            sync_key, timestamp = os.path.basename(sample).split(".")[0].split("_")
            path = os.path.relpath(sample, data_root)
            data = dict(sync_key=int(sync_key), timestamp=int(timestamp), path=path)
            camera_data.append(data)
        cameras_data[k] = camera_data

    # -- load lidar --
    aeva_data = []
    aeva_samples = sorted(
        glob(os.path.join(scene_path, "lidar/aeva/joint_lidars/points/*.bin"))
    )
    for sample in aeva_samples:
        sync_key, timestamp = os.path.basename(sample).split(".")[0].split("_")
        path = os.path.relpath(sample, data_root)
        data = dict(sync_key=int(sync_key), timestamp=int(timestamp), path=path)
        aeva_data.append(data)

    # -- load ousters --
    ousters_paths = [p for p in glob(os.path.join(scene_path, "lidar/ouster/*"))]
    ousters_data = {}
    for ousters_path in ousters_paths:
        k = os.path.basename(ousters_path)
        v = sorted(glob(os.path.join(ousters_path, "points/*.bin")))
        ouster_data = []
        for sample in v:
            sync_key, timestamp = os.path.basename(sample).split(".")[0].split("_")
            path = os.path.relpath(sample, data_root)
            data = dict(sync_key=int(sync_key), timestamp=int(timestamp), path=path)
            ouster_data.append(data)
        ousters_data[k] = ouster_data

    # -- load radars --
    radar_data = []
    radar_samples = sorted(
        glob(os.path.join(scene_path, "radar/conti542/joint_radars/detections/*.bin"))
    )
    for sample in radar_samples:
        sync_key, timestamp = os.path.basename(sample).split(".")[0].split("_")
        path = os.path.relpath(sample, data_root)
        data = dict(sync_key=int(sync_key), timestamp=int(timestamp), path=path)
        radar_data.append(data)

    # -- load annos --
    anno_data = []
    anno_samples = [
        p
        for p in sorted(
            glob(os.path.join(scene_path, "annotations/bounding_boxes/*.json"))
        )
    ]
    for sample in anno_samples:
        sync_key, timestamp = os.path.basename(sample).split(".")[0].split("_")
        path = os.path.relpath(sample, data_root)
        data = dict(sync_key=int(sync_key), timestamp=int(timestamp), path=path)
        anno_data.append(data)

    # -- load global poses --
    global_pose_data = []
    global_poses_path = os.path.join(scene_path, "poses/gt_trajectory.txt")
    global_poses_raw = np.loadtxt(global_poses_path, skiprows=1)
    for gp in global_poses_raw:  # SYNC_KEY, TIMESTAMP, X, Y, Z, R_X, R_Y, R_Z, R_W
        timestamp = (gp[1] * 1e9).astype(np.int64)
        pos = gp[2:5]
        rot = gp[5:9]
        rot = R.from_quat(rot).as_matrix()
        pose_homogeneous = np.eye(4)
        pose_homogeneous[:3, :3] = rot
        pose_homogeneous[:3, 3] = pos
        data = dict(
            sync_key=int(gp[0]), timestamp=int(timestamp), global_pose=pose_homogeneous
        )
        global_pose_data.append(data)

    # -- find max sync id --

    max_sync_id = max(
        [v[-1]["sync_key"] for _, v in cameras_data.items()]
        + [aeva_data[-1]["sync_key"]]
        + [v[-1]["sync_key"] for _, v in ousters_data.items()]
        + [radar_data[-1]["sync_key"]]
    )

    # -- sync sensors --
    synced_list = []
    for sync_id in range(max_sync_id + 1):
        data = {}

        timestamp = None
        data["images"] = {}
        for cam, cam_data in cameras_data.items():
            if len(cam_data) == 0:
                data["images"][cam] = KEY_MISSING
            else:
                cam_sync_id = cam_data[0]["sync_key"]
                if cam_sync_id == sync_id:
                    data["images"][cam] = {}
                    data["images"][cam]["path"] = cam_data[0]["path"]
                    data["images"][cam]["sync_key"] = sync_id
                    data["images"][cam]["timestamp"] = cam_data[0]["timestamp"]
                    if timestamp is None:
                        timestamp = cam_data[0]["timestamp"]
                    data["images"][cam]["delta_t_sync"] = (
                        cam_data[0]["timestamp"] - timestamp
                    )
                    cam_data.pop(0)
                elif cam_sync_id < sync_id:
                    raise ValueError(
                        f"Sync ID CAMERA {cam_sync_id} is grater then current sync id {sync_id}"
                    )
                else:
                    data["images"][cam] = KEY_MISSING

        data["aeva"] = {}
        if len(aeva_data) == 0:
            data["aeva"] = KEY_MISSING
        else:
            aeva_sync_id = aeva_data[0]["sync_key"]
            if aeva_sync_id == sync_id:
                data["aeva"]["path"] = aeva_data[0]["path"]
                data["aeva"]["sync_key"] = sync_id
                data["aeva"]["timestamp"] = aeva_data[0]["timestamp"]
                if timestamp is None:
                    timestamp = aeva_data[0]["timestamp"]
                data["aeva"]["delta_t_sync"] = aeva_data[0]["timestamp"] - timestamp
                aeva_data.pop(0)
            elif aeva_sync_id < sync_id:
                raise ValueError(
                    f"Sync ID AEVA {aeva_sync_id} is grater then current sync id {sync_id}"
                )
            else:
                data["aeva"] = KEY_MISSING

        data["ouster"] = {}
        for ous, ous_data in ousters_data.items():
            if len(ous_data) == 0:
                data["ouster"][ous] = KEY_MISSING
            else:
                ous_sync_id = ous_data[0]["sync_key"]
                if ous_sync_id == sync_id:
                    data["ouster"][ous] = {}
                    data["ouster"][ous]["path"] = ous_data[0]["path"]
                    data["ouster"][ous]["sync_key"] = sync_id
                    data["ouster"][ous]["timestamp"] = ous_data[0]["timestamp"]
                    if timestamp is None:
                        timestamp = ous_data[0]["timestamp"]
                    data["ouster"][ous]["delta_t_sync"] = (
                        ous_data[0]["timestamp"] - timestamp
                    )
                    ous_data.pop(0)
                elif ous_sync_id < sync_id:
                    raise ValueError(
                        f"Sync ID OUSTER {ous_sync_id} is grater then current sync id {sync_id}"
                    )
                else:
                    data["ouster"][ous] = KEY_MISSING

        data["radar"] = {}
        if len(radar_data) == 0:
            data["radar"] = KEY_MISSING
        else:
            radar_sync_id = radar_data[0]["sync_key"]
            if radar_sync_id == sync_id:
                data["radar"]["path"] = radar_data[0]["path"]
                data["radar"]["sync_key"] = sync_id
                data["radar"]["timestamp"] = radar_data[0]["timestamp"]
                if timestamp is None:
                    timestamp = radar_data[0]["timestamp"]
                data["radar"]["delta_t_sync"] = radar_data[0]["timestamp"] - timestamp
                radar_data.pop(0)
            elif radar_sync_id < sync_id:
                raise ValueError(
                    f"Sync ID RADAR {radar_sync_id} is grater then current sync id {sync_id}"
                )
            else:
                data["radar"] = KEY_MISSING

        if timestamp is None:
            print(
                f"Timestamp is None for sync id {sync_id} in scene {scene_path}, skipping..."
            )
            continue

        data["annos"] = {}
        if len(anno_data) == 0:
            data["annos"] = KEY_MISSING
        else:
            annos_sync_id = anno_data[0]["sync_key"]
            if annos_sync_id == sync_id:
                data["annos"]["path"] = anno_data[0]["path"]
                data["annos"]["sync_key"] = sync_id
                data["annos"]["timestamp"] = anno_data[0]["timestamp"]
                data["annos"]["delta_t_sync"] = anno_data[0]["timestamp"] - timestamp
                anno_data.pop(0)
            elif annos_sync_id < sync_id:
                raise ValueError(
                    f"Sync ID ANNOS {annos_sync_id} is grater then current sync id {sync_id}"
                )
            else:
                data["annos"] = KEY_MISSING

        data["global_pose"] = {}
        if len(global_pose_data) == 0:
            data["global_pose"] = KEY_MISSING
        else:
            global_pose_sync_id = global_pose_data[0]["sync_key"]
            if global_pose_sync_id == sync_id:
                data["global_pose"]["global_pose"] = global_pose_data[0][
                    "global_pose"
                ].tolist()
                data["global_pose"]["sync_key"] = sync_id
                data["global_pose"]["timestamp"] = global_pose_data[0]["timestamp"]
                data["global_pose"]["delta_t_sync"] = (
                    global_pose_data[0]["timestamp"] - timestamp
                )
                global_pose_data.pop(0)
            elif global_pose_sync_id < sync_id:
                raise ValueError(
                    f"Sync ID GLOBAL_POSE {global_pose_sync_id} is grater then current sync id {sync_id}"
                )
            else:
                data["global_pose"] = KEY_MISSING

        data["timestamp"] = timestamp
        synced_list.append(data)

    with open(output_file, "w") as f:
        json.dump(synced_list, f, indent=4)


if __name__ == "__main__":
    args = parse_arguments()
    print(args)

    output_root = Path(args.sync_info_root)
    output_root.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    paths = sorted(
        [path for path in glob(str(data_root) + "/*") if "scene_" in str(path)]
    )
    print(f"Syncing data list. Total number of scenes: {len(paths)}")

    if args.multiprocessing:
        with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
            future_to_path = {
                executor.submit(sync_sensors, path, data_root, output_root): path
                for path in paths
            }

            for f in tqdm(
                as_completed(future_to_path), total=len(future_to_path), desc=" scenes"
            ):
                path = future_to_path[f]
                try:
                    f.result()
                except Exception as e:
                    print(f"Error in worker {Path(path).name}: {e}")
    else:
        print("----- DEBUG MODE: only one worker, only one scene -----")
        for path in tqdm(paths[:1], desc=" scenes"):
            print(path)
            p = sync_sensors(path, data_root, output_root)
