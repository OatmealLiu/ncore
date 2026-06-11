import json
import argparse
from pathlib import Path
from tqdm import tqdm
import os
import pickle as pkl
from typing import Dict
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

sys.path.append(str(Path(__file__).parent.parent))
from dataset_viewer.vis_utils.load_utils import load_extrinsic_between_nodes


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
        help="The directory for synchronized data.",
    )
    parser.add_argument(
        "--metainfo-path",
        type=str,
        default="./TruckDrive/sensor_sync/metainfo.json",
        help="The path to the metainfo JSON file.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="./TruckDrive/mmdet_annotations/",
        help="The output directory for mmdet annotations.",
    )
    parser.add_argument(
        "--multiprocessing",
        action="store_true",
        help="Enable multiprocessing.",
    )
    args = parser.parse_args()
    return args


metainfo_annotations = {
    "classes": [
        "DontCare",
        "Bike",
        "Passenger-Car",
        "Person",
        "RoadObstruction",
        "SemiTruck-Cab",
        "SemiTruck-Trailer",
        "Vehicle",
        "TrafficSign",
        "EmergencyVehicle",
    ],
    "categories": {
        "DontCare": -1,
        "Bike": 0,
        "Passenger-Car": 1,
        "Person": 2,
        "RoadObstruction": 3,
        "SemiTruck-Cab": 4,
        "SemiTruck-Trailer": 5,
        "Vehicle": 6,
        "TrafficSign": 7,
        "EmergencyVehicle": 8,
    },
}


def custom_encoder(obj):
    """JSON encoder function to serialize NumPy arrays, Posix paths, and other custom objects."""
    if isinstance(obj, (np.ndarray, np.int64, np.float64, np.int32, np.float32)):  # type: ignore
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    else:
        raise TypeError(f"Object of type '{type(obj)}' is not JSON serializable")


def save_dict_to_json(output_dict: Dict, output_json_path: Path):
    """Save the dictionary to a JSON file."""
    with open(output_json_path, "w") as json_file:
        json.dump(output_dict, json_file, default=custom_encoder, indent=2)
    print("Saved the dictionary to ", output_json_path)


def save_dict_to_pickle(output_dict: Dict, output_pickle_path: Path):
    """Save the dictionary to a pickle file."""
    with open(output_pickle_path, "wb") as pickle_file:
        pkl.dump(output_dict, pickle_file, protocol=pkl.HIGHEST_PROTOCOL)
    print("Saved the dictionary to ", output_pickle_path)


def load_instances_refined_with_vel(
    anno_path, cam_keys, next_anno_path, prev_vel, delta_t, label_mapping
):
    instances = []
    new_prev_vel = {}
    with open(anno_path, "r") as f:
        anno_file = json.load(f)
    if next_anno_path is not None:
        with open(next_anno_path, "r") as f:
            next_anno_file = json.load(f)
        next_pos = {}
        for d in next_anno_file:
            if "lidar" in d:
                name = d["class-id"] + str(d["Tracking_ID"])
                next_pos[name] = (d["lidar"]["x_c"], d["lidar"]["y_c"])
    else:
        next_pos = {}
    for j, d in enumerate(anno_file):
        box_2d = {}
        for ck in cam_keys:
            box_2d[ck] = []
            if ck in d:
                box_2d[ck] = [d[ck]["x0"], d[ck]["y0"], d[ck]["x1"], d[ck]["y1"]]
        ann = {
            "object-id": str(d["Tracking_ID"]),
            "class-id": d["class-id"],
            "bbox_2d": box_2d,
            "bbox_label_2d": label_mapping[d["class-id"]],
            "object-id": -1000,
            "bbox_3d": [-1000, -1000, -1000, -1000, -1000, -1000, -1000],
            "bbox_label_3d": -1,
            "velocity": [0, 0],
            "num_lidar_pts": 0,
            "num_radar_pts": 0,
        }
        fields = [
            "x",
            "y",
            "z",
            "l",
            "w",
            "h",
            "yaw",
            "vel_x_ego",
            "vel_y_ego",
            "vel_z_ego",
        ]
        if (
            fields[0] in d
            and fields[1] in d
            and fields[2] in d
            and fields[3] in d
            and fields[4] in d
            and fields[5] in d
            and fields[6] in d
            and isinstance(d[fields[0]], float)
            and not np.isnan(d[fields[0]])
            and isinstance(d[fields[1]], float)
            and not np.isnan(d[fields[1]])
            and isinstance(d[fields[2]], float)
            and not np.isnan(d[fields[2]])
            and isinstance(d[fields[3]], float)
            and not np.isnan(d[fields[3]])
            and isinstance(d[fields[4]], float)
            and not np.isnan(d[fields[4]])
            and isinstance(d[fields[5]], float)
            and not np.isnan(d[fields[5]])
            and isinstance(d[fields[6]], float)
            and not np.isnan(d[fields[6]])
        ):
            name = d["class-id"] + str(d["Tracking_ID"])
            if name in next_pos:
                x, y = next_pos[name]
                vel_x = (d["lidar"]["x_c"] - x) / delta_t
                vel_y = (d["lidar"]["y_c"] - y) / delta_t
            elif name not in next_pos and name in prev_vel:
                vel_x, vel_y = prev_vel[name]
            elif name not in next_pos and name not in prev_vel:
                vel_x, vel_y = 0.0, 0.0

            ann.update(
                {
                    "object-id": str(d["Tracking_ID"]),
                    "class-id": d["class-id"],
                    "bbox_3d": [
                        d[fields[0]],
                        d[fields[1]],
                        d[fields[2]],
                        abs(d[fields[3]]),
                        abs(d[fields[4]]),
                        abs(d[fields[5]]),
                        d[fields[6]],
                    ],
                    "bbox_label_3d": label_mapping[d["class-id"]],
                    "velocity": [vel_x, vel_y],
                    "num_lidar_pts": 10,
                    "num_radar_pts": 10,
                }
            )
            new_prev_vel[name] = [vel_x, vel_y]
        instances.append(ann)
    return instances, new_prev_vel


def process_scene(json_file, data_root_dir, metainfo):
    """Process one sync JSON file and return (training, unlabelled, validation) samples lists."""
    scene_id = os.path.basename(json_file).replace(".json", "").replace("sync_", "")
    with open(json_file, "r") as f:
        sync_data_sensor = json.load(f)
    sync_data_sensor = sorted(sync_data_sensor, key=lambda item: int(item["timestamp"]))
    sample_count = len(sync_data_sensor)

    scene_path = os.path.join(data_root_dir, scene_id)
    tf_file_path = os.path.join(scene_path, "calibrations/calib_tf_tree_full.json")
    with open(tf_file_path, "r") as f:
        data_extrinsics = json.load(f)
    aeva2velo = load_extrinsic_between_nodes(
        data_extrinsics, "lidar_aeva_forward_center_wide", "velodyne"
    )
    per_scene_sensor_keys = metainfo["per_scene_sensor_keys"]
    CALIB = {
        "aeva2velo": aeva2velo,
        "velo2cam": {},
        "aeva2cam": {},
        "cam2img": {},
        "height": {},
        "width": {},
    }
    for cam in per_scene_sensor_keys[scene_id]["cam_keys"]:
        CALIB["velo2cam"][cam] = load_extrinsic_between_nodes(
            data_extrinsics, "velodyne", f"camera_leopard_{cam}"
        )
        CALIB["aeva2cam"][cam] = load_extrinsic_between_nodes(
            data_extrinsics, "lidar_aeva_forward_center_wide", f"camera_leopard_{cam}"
        )
        calib_file_path = os.path.join(
            scene_path, f"calibrations/calib_camera_leopard_{cam}.json"
        )
        with open(calib_file_path, "r") as cf:
            calib_data = json.load(cf)
        cam2img = np.eye(4)
        cam2img[:3, :4] = np.array(calib_data["P"]).reshape((3, 4))
        CALIB["cam2img"][cam] = cam2img
        CALIB["height"][cam] = calib_data["height"]
        CALIB["width"][cam] = calib_data["width"]
    CALIB.update(
        {"velo2ouster": {}, "aeva2ouster": {}, "ouster2velo": {}, "ouster2aeva": {}}
    )
    per_scene_sensor_keys[scene_id]["ou_keys"] = [
        k for k in per_scene_sensor_keys[scene_id]["ou_keys"] if "join" not in k
    ]
    for ou in per_scene_sensor_keys[scene_id]["ou_keys"]:
        CALIB["velo2ouster"][ou] = load_extrinsic_between_nodes(
            data_extrinsics, "velodyne", f"lidar_ouster_{ou}"
        )
        CALIB["aeva2ouster"][ou] = load_extrinsic_between_nodes(
            data_extrinsics, "lidar_aeva_forward_center_wide", f"lidar_ouster_{ou}"
        )
        CALIB["ouster2velo"][ou] = load_extrinsic_between_nodes(
            data_extrinsics, f"lidar_ouster_{ou}", "velodyne"
        )
        CALIB["ouster2aeva"][ou] = load_extrinsic_between_nodes(
            data_extrinsics, f"lidar_ouster_{ou}", "lidar_aeva_forward_center_wide"
        )

    CALIB["velo2radar"] = load_extrinsic_between_nodes(
        data_extrinsics, "velodyne", "radar_conti542_forward_left_high"
    )
    CALIB["aeva2radar"] = load_extrinsic_between_nodes(
        data_extrinsics,
        "lidar_aeva_forward_center_wide",
        "radar_conti542_forward_left_high",
    )
    CALIB["radar2velo"] = load_extrinsic_between_nodes(
        data_extrinsics, "radar_conti542_forward_left_high", "velodyne"
    )
    CALIB["radar2aeva"] = load_extrinsic_between_nodes(
        data_extrinsics,
        "radar_conti542_forward_left_high",
        "lidar_aeva_forward_center_wide",
    )

    scene_id_validations = metainfo["val_scenes"]
    (
        training_samples,
        training_samples_unlabelled,
        validation_samples,
        validation_samples_unlabelled,
    ) = ([], [], [], [])
    prev_vel = {}
    for i in range(sample_count):
        data = {}
        data["timestamp"] = sync_data_sensor[i]["timestamp"]
        data["scene_id"] = scene_id
        if sync_data_sensor[i]["global_pose"] is not None:
            data["lidar2global"] = sync_data_sensor[i]["global_pose"]["global_pose"]
        else:
            data["lidar2global"] = None
        if sync_data_sensor[i]["aeva"] is None:
            data["lidar_points"] = {
                "lidar_path": None,
                "timestamp": data["timestamp"],
                "delta_t_sync": 0.0,
                "num_pts_feats": 5,
                "lidar2velo": CALIB["aeva2velo"],
            }
        else:
            data["lidar_points"] = {
                "lidar_path": sync_data_sensor[i]["aeva"]["path"],
                "timestamp": sync_data_sensor[i]["aeva"]["timestamp"],
                "delta_t_sync": sync_data_sensor[i]["aeva"]["delta_t_sync"],
                "num_pts_feats": 5,
                "lidar2velo": CALIB["aeva2velo"],
            }
        data["images"] = {}
        for cam in per_scene_sensor_keys[scene_id]["cam_keys"]:
            if sync_data_sensor[i]["images"][cam] is None:
                velo2cam = CALIB["velo2cam"][cam]
                lidar2cam = CALIB["aeva2cam"][cam]
                cam2img = CALIB["cam2img"][cam]
                height = CALIB["height"][cam]
                width = CALIB["width"][cam]
                data["images"][cam] = {
                    "img_path": None,
                    "timestamp": data["timestamp"],
                    "delta_t_sync": 0.0,
                    "img_h": height,
                    "img_w": width,
                    "lidar2cam": lidar2cam,
                    "velo2cam": velo2cam,
                    "cam2img": cam2img,
                    "lidar2img": cam2img @ lidar2cam,
                    "velo2img": cam2img @ velo2cam,
                }
            else:
                velo2cam = CALIB["velo2cam"][cam]
                lidar2cam = CALIB["aeva2cam"][cam]
                cam2img = CALIB["cam2img"][cam]
                height = CALIB["height"][cam]
                width = CALIB["width"][cam]
                data["images"][cam] = {
                    "img_path": sync_data_sensor[i]["images"][cam]["path"],
                    "timestamp": sync_data_sensor[i]["images"][cam]["timestamp"],
                    "delta_t_sync": sync_data_sensor[i]["images"][cam]["delta_t_sync"],
                    "img_h": height,
                    "img_w": width,
                    "lidar2cam": lidar2cam,
                    "velo2cam": velo2cam,
                    "cam2img": cam2img,
                    "lidar2img": cam2img @ lidar2cam,
                    "velo2img": cam2img @ velo2cam,
                }
        if len(per_scene_sensor_keys[scene_id]["ou_keys"]) > 0:
            data["short_range_lidar_points"] = {}
            for ou in per_scene_sensor_keys[scene_id]["ou_keys"]:
                if sync_data_sensor[i]["ouster"][ou] is None:
                    data["short_range_lidar_points"][ou] = {
                        "short_range_lidar_path": None,
                        "timestamp": data["timestamp"],
                        "delta_t_sync": 0.0,
                        "velo2ouster": CALIB["velo2ouster"][ou],
                        "lidar2ouster": CALIB["aeva2ouster"][ou],
                        "ouster2velo": CALIB["ouster2velo"][ou],
                        "ouster2lidar": CALIB["ouster2aeva"][ou],
                    }
                else:
                    data["short_range_lidar_points"][ou] = {
                        "short_range_lidar_path": sync_data_sensor[i]["ouster"][ou][
                            "path"
                        ],
                        "timestamp": sync_data_sensor[i]["ouster"][ou]["timestamp"],
                        "delta_t_sync": sync_data_sensor[i]["ouster"][ou][
                            "delta_t_sync"
                        ],
                        "velo2ouster": CALIB["velo2ouster"][ou],
                        "lidar2ouster": CALIB["aeva2ouster"][ou],
                        "ouster2velo": CALIB["ouster2velo"][ou],
                        "ouster2lidar": CALIB["ouster2aeva"][ou],
                    }
        else:
            data["short_range_lidar_points"] = None
        if sync_data_sensor[i]["radar"] is None:
            data["radar_points"] = {
                "radar_path": None,
                "timestamp": data["timestamp"],
                "delta_t_sync": 0.0,
                "num_pts_feats": 5,
                "velo2radar": CALIB["velo2radar"],
                "lidar2radar": CALIB["aeva2radar"],
                "radar2velo": CALIB["radar2velo"],
                "radar2lidar": CALIB["radar2aeva"],
            }
        else:
            data["radar_points"] = {
                "radar_path": sync_data_sensor[i]["radar"]["path"],
                "timestamp": sync_data_sensor[i]["radar"]["timestamp"],
                "delta_t_sync": sync_data_sensor[i]["radar"]["delta_t_sync"],
                "num_pts_feats": 5,
                "velo2radar": CALIB["velo2radar"],
                "lidar2radar": CALIB["aeva2radar"],
                "radar2velo": CALIB["radar2velo"],
                "radar2lidar": CALIB["radar2aeva"],
            }

        data["instances"] = []
        if scene_id in scene_id_validations:
            validation_samples_unlabelled.append(data.copy())
        else:
            training_samples_unlabelled.append(data.copy())

        if sync_data_sensor[i]["annos"] is not None:
            anno_path = os.path.join(
                data_root_dir, sync_data_sensor[i]["annos"]["path"]
            )
            if i == sample_count - 1 or sync_data_sensor[i + 1]["annos"] is None:
                next_anno_path = None
                delta_t = 0
            else:
                next_anno_path = os.path.join(
                    data_root_dir, sync_data_sensor[i + 1]["annos"]["path"]
                )
                delta_t = (
                    sync_data_sensor[i + 1]["annos"]["timestamp"]
                    - sync_data_sensor[i]["annos"]["timestamp"]
                ) / 1e9

            instances, prev_vel = load_instances_refined_with_vel(
                anno_path,
                per_scene_sensor_keys[scene_id]["cam_keys"],
                next_anno_path,
                prev_vel,
                delta_t,
                metainfo["label_mapping"],
            )
            data["instances"] = instances
            if scene_id in scene_id_validations:
                validation_samples.append(data)
            else:
                training_samples.append(data)

    return (
        training_samples,
        training_samples_unlabelled,
        validation_samples,
        validation_samples_unlabelled,
    )


if __name__ == "__main__":
    args = parse_arguments()
    print(args)

    data_root_dir = Path(args.data_root)
    sync_info_dir = Path(args.sync_info_root)
    metainfo_path = Path(args.metainfo_path)
    data_output_dir = Path(args.output_root)
    data_output_dir.mkdir(parents=True, exist_ok=True)

    with open(metainfo_path, "r") as f:
        metainfo = json.load(f)
    json_files = list(sync_info_dir.glob("*.json"))

    (
        training_samples,
        training_samples_unlabelled,
        validation_samples,
        validation_samples_unlabelled,
    ) = ([], [], [], [])
    if args.multiprocessing:
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [
                executor.submit(process_scene, jf, data_root_dir, metainfo)
                for jf in json_files
            ]
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing scenes in parallel",
            ):
                tr, unlab, val, unlab_val = fut.result()
                training_samples.extend(tr)
                training_samples_unlabelled.extend(unlab)
                validation_samples.extend(val)
                validation_samples_unlabelled.extend(unlab_val)
    else:
        print("----- DEBUG MODE: only one worker, only one scene -----")
        for path in tqdm(json_files[:1], desc=" scenes"):
            tr, unlab, val, unlab_val = process_scene(path, data_root_dir, metainfo)
            training_samples.extend(tr)
            training_samples_unlabelled.extend(unlab)
            validation_samples.extend(val)
            validation_samples_unlabelled.extend(unlab_val)

    print("Total labelled training samples:", len(training_samples))
    print(
        "Total labelled + unlabelled training samples:",
        len(training_samples_unlabelled),
    )
    print("Total labelled validation samples:", len(validation_samples))
    print(
        "Total labelled + unlabelled validation samples:",
        len(validation_samples_unlabelled),
    )

    save_dict_to_json(
        output_dict={"metainfo": metainfo_annotations, "data_list": training_samples},
        output_json_path=data_output_dir / "annotations_train_labelled.json",
    )
    save_dict_to_pickle(
        output_dict={"metainfo": metainfo_annotations, "data_list": training_samples},
        output_pickle_path=data_output_dir / "annotations_train_labelled.pkl",
    )

    save_dict_to_json(
        output_dict={"metainfo": metainfo_annotations, "data_list": validation_samples},
        output_json_path=data_output_dir / "annotations_val_labelled.json",
    )
    save_dict_to_pickle(
        output_dict={"metainfo": metainfo_annotations, "data_list": validation_samples},
        output_pickle_path=data_output_dir / "annotations_val_labelled.pkl",
    )

    save_dict_to_json(
        output_dict={
            "metainfo": metainfo_annotations,
            "data_list": training_samples_unlabelled,
        },
        output_json_path=data_output_dir / "annotations_train_unlabelled.json",
    )
    save_dict_to_pickle(
        output_dict={
            "metainfo": metainfo_annotations,
            "data_list": training_samples_unlabelled,
        },
        output_pickle_path=data_output_dir / "annotations_train_unlabelled.pkl",
    )

    save_dict_to_json(
        output_dict={
            "metainfo": metainfo_annotations,
            "data_list": validation_samples_unlabelled,
        },
        output_json_path=data_output_dir / "annotations_val_unlabelled.json",
    )
    save_dict_to_pickle(
        output_dict={
            "metainfo": metainfo_annotations,
            "data_list": validation_samples_unlabelled,
        },
        output_pickle_path=data_output_dir / "annotations_val_unlabelled.pkl",
    )
