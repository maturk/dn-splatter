import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import tyro
from dn_splatter.utils.camera_utils import OPENGL_TO_OPENCV
from rich.progress import track


def generate_kinect_pointcloud_within_sequence(
    data_path: Path, num_points: int = 1_000_000
):
    print("Generating pointclouds from kinect data...")

    info_file = json.load(open(os.path.join(data_path, "transformations_colmap.json")))

    frames = info_file["frames"]
    frame_names = [frame["file_path"].split("/")[-1].split(".")[0] for frame in frames]

    num_images = len(frames)
    i_all = np.arange(num_images)

    with open(os.path.join(data_path, "test.txt")) as f:
        lines = f.readlines()
    i_eval_name = [num.split("\n")[0] for num in lines]

    # only select images that exist in frame_names
    i_eval_name = [name for name in i_eval_name if name in frame_names]
    i_eval = [frame_names.index(name) for name in i_eval_name]
    i_train = np.setdiff1d(i_all, i_eval)  # type: ignore

    index = i_train

    points_list = []
    colors_list = []
    normals_list = []

    samples_per_frame = (num_points + len(index)) // (len(index))

    for item in track(index, description="processing ... "):
        frame = frames[item]
        name = frame["file_path"].split("/")[-1].split(".")[0]
        pcd = o3d.io.read_point_cloud(
            os.path.join(data_path, "PointCloud", name + ".ply")
        )

        # change the pd from spectacularAI pose world coordination to colmap pose world coordination
        original_pose = np.loadtxt(
            os.path.join(data_path, "pose", name + ".txt")
        ).reshape(4, 4)
        original_pose = np.matmul(original_pose, OPENGL_TO_OPENCV)
        pcd = pcd.transform(np.linalg.inv(original_pose))

        colmap_pose = frame["transform_matrix"]
        pcd = pcd.transform(colmap_pose)

        samples_per_frame = min(samples_per_frame, len(pcd.points))

        mask = random.sample(range(len(pcd.points)), samples_per_frame)
        mask = np.asarray(mask)
        color = np.asarray(pcd.colors)[mask]
        point = np.asarray(pcd.points)[mask]
        normal = np.asarray(pcd.normals)[mask]

        points_list.append(np.asarray(point))
        colors_list.append(np.asarray(color))
        normals_list.append(np.asarray(normal))

    cloud = o3d.geometry.PointCloud()
    points = o3d.utility.Vector3dVector(np.vstack(points_list))
    colors = o3d.utility.Vector3dVector(np.vstack(colors_list))
    normals = o3d.utility.Vector3dVector(np.vstack(normals_list))

    cloud.points = points
    cloud.colors = colors
    cloud.normals = normals
    o3d.io.write_point_cloud(os.path.join(data_path, "kinect_pointcloud.ply"), cloud)


def generate_iPhone_pointcloud_within_sequence(
    data_path: Path, num_points: int = 1_000_000
):
    print("Generating pointcloud from iPhone data...")
    info_file = json.load(open(os.path.join(data_path, "transformations_colmap.json")))

    frames = info_file["frames"]
    frame_names = [frame["file_path"].split("/")[-1].split(".")[0] for frame in frames]

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.04,
        sdf_trunc=0.2,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    num_images = len(frames)
    i_all = np.arange(num_images)

    with open(os.path.join(data_path, "test.txt")) as f:
        lines = f.readlines()
    i_eval_name = [num.split("\n")[0] for num in lines]
    i_eval_name = [name for name in i_eval_name if name in frame_names]
    i_eval = [frame_names.index(name) for name in i_eval_name]
    i_train = np.setdiff1d(i_all, i_eval)

    index = i_train

    points_list = []
    colors_list = []

    if "fl_x" in info_file:
        fx, fy, cx, cy = (
            float(info_file["fl_x"]),
            float(info_file["fl_y"]),
            float(info_file["cx"]),
            float(info_file["cy"]),
        )
        H = int(info_file["h"])
        W = int(info_file["w"])
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

    samples_per_frame = (num_points + len(index)) // (len(index))

    for item in track(index, description="processing ... "):
        frame = frames[item]
        if "fl_x" in frame:
            fx, fy, cx, cy = (
                float(frame["fl_x"]),
                float(frame["fl_y"]),
                float(frame["cx"]),
                float(frame["cy"]),
            )
            H = int(frame["h"])
            W = int(frame["w"])
            camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

        color = cv2.imread(os.path.join(data_path, frame["file_path"]))
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color = o3d.geometry.Image(color)

        # pose
        pose = frame["transform_matrix"]
        pose = np.matmul(np.array(pose), OPENGL_TO_OPENCV)

        depth = cv2.imread(
            os.path.join(data_path, frame["depth_file_path"]), cv2.IMREAD_ANYDEPTH
        )
        depth = cv2.resize(depth, (W, H))  # type: ignore
        depth = o3d.geometry.Image(depth)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False
        )

        volume.integrate(
            rgbd,
            camera_intrinsics,  # type: ignore
            np.linalg.inv(pose),
        )

        pcd = volume.extract_point_cloud()

        # randomly select samples_per_frame points from points
        samples_per_frame = min(samples_per_frame, len(pcd.points))
        mask = random.sample(range(len(pcd.points)), samples_per_frame)
        mask = np.asarray(mask)
        color = np.asarray(pcd.colors)[mask]
        point = np.asarray(pcd.points)[mask]

        points_list.append(np.asarray(point))
        colors_list.append(np.asarray(color))

    points = np.vstack(points_list)
    colors = np.vstack(colors_list)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(os.path.join(data_path, "iphone_pointcloud.ply"), pcd)


if __name__ == "__main__":
    tyro.cli(generate_kinect_pointcloud_within_sequence)
