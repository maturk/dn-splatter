import random

import cv2
import numpy as np
import open3d as o3d
from utils.graphics_utils import getWorld2View2

OPENGL_TO_OPENCV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


def generate_ply_from_rgbd(train_cam_infos, meta, num_points, ply_path):
    print("Generating ply from rgbd")
    train_example = train_cam_infos[0]
    w, h = train_example.width, train_example.height

    samples_per_frame = (num_points + len(train_cam_infos)) // len(train_cam_infos)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.04,
        sdf_trunc=0.2,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    points_list = []
    colors_list = []
    if "fl_x" in meta:
        fx, fy, cx, cy = meta["fl_x"], meta["fl_y"], meta["cx"], meta["cy"]
    else:
        try:
            fx, fy, cx, cy = meta["frames"][0]["fl_x"], meta["frames"][0]["fl_y"], meta["frames"][0]["cx"], meta["frames"][0]["cy"]
        except KeyError:
            # raise exception
            print("Error: No intrinsics found")
            quit()
    for train_cam in train_cam_infos:
        w2c = getWorld2View2(train_cam.R, train_cam.T)
        c2w = np.linalg.inv(w2c)

        image_path = train_cam.image_path
        color = cv2.imread(image_path)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color = o3d.geometry.Image(color)

        depth = (train_cam.depth * 1000).astype(np.uint16)
        depth = o3d.geometry.Image(depth)

        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False
        )

        volume.integrate(
            rgbd,
            camera_intrinsics,  # type: ignore
            np.linalg.inv(c2w),
        )

        pcd = volume.extract_point_cloud()

        samples_per_frame = min(samples_per_frame, len(pcd.points))
        mask = random.sample(range(len(pcd.points)), samples_per_frame)
        mask = np.asarray(mask)
        color = np.asarray(pcd.colors)[mask]
        point = np.asarray(pcd.points)[mask]

        points_list.append(np.asarray(point))
        colors_list.append(np.asarray(color))

    points = np.concatenate(points_list, axis=0)
    colors = np.concatenate(colors_list, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(ply_path, pcd)
