import argparse
import json
from pathlib import Path
from tqdm import tqdm
import os


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
        "--metainfo-path",
        type=str,
        default="./TruckDrive/sensor_sync/metainfo.json",
        help="The path to the metainfo JSON file.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    path_to_data = Path(args.data_root)
    scenes_list = os.listdir(path_to_data)
    path_to_metainfo = Path(args.metainfo_path)

    with open(path_to_metainfo, "r") as f:
        metainfo = json.load(f)
    metainfo["per_scene_sensor_keys"] = {}
    for scene in tqdm(scenes_list):
        bs = os.path.basename(scene)
        metainfo["per_scene_sensor_keys"][bs] = {}
        path_to_cam = path_to_data / scene / "camera" / "leopard"
        cam_keys = os.listdir(path_to_cam)
        metainfo["per_scene_sensor_keys"][bs]["cam_keys"] = cam_keys
        path_to_aeva = path_to_data / scene / "lidar" / "aeva"
        aeva_keys = os.listdir(path_to_aeva)
        metainfo["per_scene_sensor_keys"][bs]["aeva_keys"] = sorted(aeva_keys)
        path_to_ou = path_to_data / scene / "lidar" / "ouster"
        ou_keys = os.listdir(path_to_ou)
        metainfo["per_scene_sensor_keys"][bs]["ou_keys"] = sorted(ou_keys)
        path_to_rad = path_to_data / scene / "radar" / "conti542"
        rad_keys = os.listdir(path_to_rad)
        metainfo["per_scene_sensor_keys"][bs]["rad_keys"] = sorted(rad_keys)

    with open(path_to_metainfo, "w") as f:
        json.dump(metainfo, f, indent=4)
