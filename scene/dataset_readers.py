#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from calendar import c
from genericpath import exists
from re import I
import open3d as o3d
from cgi import test
import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import (
    read_extrinsics_text,
    read_intrinsics_text,
    qvec2rotmat,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
    read_points3D_text,
)
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.pointcloud_utils import generate_iPhone_pointcloud
from natsort import natsorted
import glob
from utils.camera_utils import load_from_json
from utils.pd_utils import generate_ply_from_rgbd
from utils.general_utils import write_json
import cv2
import math


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    depth: np.array
    normal: np.array
    depth_confidence: np.array
    image_path: str
    image_name: str
    width: int
    height: int


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, load_every=5):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        if idx % load_every != 0:
            continue
        if idx % 10 == 0:
            continue
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        rotation = qvec2rotmat(extr.qvec)
        translation = np.array(extr.tvec).reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([0, 2, 1, 3]), :]
        c2w[2, :] *= -1

        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = np.transpose(
            w2c[:3, :3]
        )  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)

        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        image_path = os.path.join(images_folder, "rgb", os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        # depth
        depth_path = os.path.join(images_folder, "depth", image_name + ".png")
        depth = Image.open(depth_path)
        depth = np.array(depth) / 1000.0
        depth = cv2.resize(depth, (width, height))

        # normal
        normal_path = os.path.join(
            images_folder, "normals_from_pretrain", image_name + ".png"
        )
        normal = Image.open(normal_path)
        normal = np.array(normal) / 255.0
        h, w, _ = normal.shape
        normal = normal.reshape(-1, 3).transpose(1, 0)
        normal = (normal - 0.5) * 2
        normal = (R @ normal).T
        normal = normal / np.linalg.norm(normal, axis=1, keepdims=True)
        normal = normal.reshape(h, w, 3)

        # confidence
        confidence_path = os.path.join(
            images_folder, "depth_normals_mask", image_name + ".jpg"
        )
        if not exists(confidence_path):
            depth_confidence = np.ones_like(np.array(depth), dtype=np.float32)
        else:
            depth_confidence = 1 - (cv2.imread(confidence_path) / 255)[..., 0]

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            depth=depth,
            normal=normal,
            depth_confidence=depth_confidence,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
        )
        cam_infos.append(cam_info)

    sys.stdout.write("\n")
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    print(vertices.shape)
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=10):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        print("Reading text files")
        cameras_extrinsic_file = os.path.join(path, "colmap", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "colmap", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=path,
        load_every=5,
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
    test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]

    write_json(train_cam_infos, path)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "point_cloud.ply")
    num_pts = 100_000
    # pcd = fetchPly(ply_path)
    # generate_ply_from_rgbd(train_cam_infos, num_pts, ply_path)
    # try:
    #     ply = o3d.io.read_point_cloud(ply_path)
    #     positions = np.array(ply.points)
    #     colors = np.vstack(ply.colors) / 255.0
    #     normals = np.vstack(ply.normals)
    #     pcd = BasicPointCloud(points=positions, colors=colors, normals=normals)
    # except:
    #     pcd = None
    # ply_path = os.path.join(path, "points3d.ply")
    # if not os.path.exists(ply_path):
    #     # Since this data set has no colmap data, we start with random points
    # num_pts = 100_000
    # print(f"Generating random point cloud ({num_pts})...")

    # # We create random points inside the bounds of the synthetic Blender scenes
    # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
    # shs = np.random.random((num_pts, 3)) / 255.0
    # pcd = BasicPointCloud(
    #     points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
    # )

    # storePly(ply_path, xyz, SH2RGB(shs) * 255)
    # # try:
    # pcd = fetchPly(ply_path)
    # except:
    #     pcd = None
    try:
        ply = o3d.io.read_point_cloud(ply_path)
        # estimate normal for point cloud
        ply.estimate_normals()
        positions = np.array(ply.points)
        colors = np.vstack(ply.colors)
        normals = np.vstack(ply.normals)
        pcd = BasicPointCloud(points=positions, colors=colors, normals=normals)
        print("loaded point cloud")
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    train_cam_infos = []
    test_cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        if "fl_x" in contents:
            import math

            fx = contents["fl_x"]
            fy = contents["fl_y"]
            cx = contents["cx"]
            cy = contents["cy"]
            width = contents["w"]
            height = contents["h"]

            fovx = 2 * math.atan(width / (2 * fx))
            fovy = 2 * math.atan(height / (2 * fy))
        else:
            fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        print(len(frames))
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"])

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            four = np.eye(4)
            four[:3, :4] = c2w
            w2c = np.linalg.inv(four)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            confidence_path = os.path.join(
                path,
                cam_name.replace("rgb", "depth_normals_mask").replace("png", "jpg"),
            )
            assert os.path.exists(
                confidence_path
            ), f"Path {confidence_path} does not exist"
            depth_confidence = 1 - (cv2.imread(confidence_path) / 255)[..., 0]

            normal_path = os.path.join(
                path,
                "long_capture",
                cam_name.replace("rgb", "normals_from_pretrain").replace("jpg", "png"),
            )
            if not os.path.exists(normal_path):
                normal = np.zeros_like(np.array(image))
            else:
                normal = Image.open(normal_path)
                normal = np.array(normal) / 255.0

                h, w, _ = normal.shape
                normal = normal.reshape(-1, 3).transpose(1, 0)
                normal = (normal - 0.5) * 2
                normal = (R @ normal).T
                normal = normal / np.linalg.norm(normal, axis=1, keepdims=True)
                normal = normal.reshape(h, w, 3)

            depth = Image.open(
                os.path.join(
                    path,
                    cam_name.replace("rgb", "depth").replace("jpg", "png"),
                )
            )
            depth = np.array(depth) / 1000.0
            depth = cv2.resize(depth, (width, height))

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (
                1 - norm_data[:, :, 3:4]
            )
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            train_cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    depth=depth,
                    normal=normal,
                    depth_confidence=depth_confidence,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                )
            )

    return train_cam_infos, test_cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", white_background, extension
    )
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", white_background, extension
    )

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(
            points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
        )

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readCamerasFromTransformations(
    path, transformsfile, load_every, white_background, extension=".png"
):
    train_cam_infos = []
    test_cam_infos = []

    with open(os.path.join(path, "long_capture", transformsfile)) as json_file:
        contents = json.load(json_file)
        if "fl_x" in contents:
            fl_x = contents["fl_x"]
            fl_y = contents["fl_y"]
            w = contents["w"]
            h = contents["h"]
            cx, cy = contents["cx"], contents["cy"]

        frames = contents["frames"]

        id_list = np.arange(len(frames))

        id_list = id_list[::load_every]

        test_split_path = os.path.join(path, "long_capture", "test.txt")
        if os.path.exists(test_split_path):
            test_ids = []
            if os.path.exists(test_split_path):
                with open(test_split_path) as f:
                    test_frames = f.readlines()

                    test_ids = [x.strip() for x in test_frames]

        for idx, frame in enumerate(frames):
            if idx not in id_list:
                continue
            cam_name = os.path.join(path, "long_capture", frame["file_path"])
            depth_name = os.path.join(path, "long_capture", frame["depth_file_path"])
            if "fl_x" in frame:
                fl_x = frame["fl_x"]
                fl_y = frame["fl_y"]
                w = frame["w"]
                h = frame["h"]
                cx, cy = frame["cx"], frame["cy"]

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            if c2w.shape[0] == 3:
                c2w = np.vstack([c2w, np.array([0, 0, 0, 1])])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, "long_capture", cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            depth_path = os.path.join(path, "long_capture", depth_name)
            depth = Image.open(depth_path)
            depth = np.array(depth) / 1000.0

            confidence_path = os.path.join(
                path,
                "long_capture",
                depth_name.replace("depth", "depth_normals_mask").replace("png", "jpg"),
            )

            assert os.path.exists(
                confidence_path
            ), f"Path {confidence_path} does not exist"
            depth_confidence = 1 - (cv2.imread(confidence_path) / 255)[..., 0]

            normal_path = os.path.join(
                path,
                "long_capture",
                cam_name.replace("images", "normals_from_pretrain").replace(
                    "jpg", "png"
                ),
            )
            if not os.path.exists(normal_path):
                normal = np.zeros_like(np.array(image))
            else:
                normal = Image.open(normal_path)
                normal = np.array(normal) / 255.0

                h, w, _ = normal.shape
                normal = normal.reshape(-1, 3).transpose(1, 0)
                normal = (normal - 0.5) * 2
                normal = (R @ normal).T
                normal = normal / np.linalg.norm(normal, axis=1, keepdims=True)
                normal = normal.reshape(h, w, 3)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (
                1 - norm_data[:, :, 3:4]
            )
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovx = focal2fov(fl_x, w)
            fovy = focal2fov(fl_y, h)
            FovY = fovy
            FovX = fovx

            if image_name.split(".")[0] in test_ids:
                test_cam_infos.append(
                    CameraInfo(
                        uid=idx,
                        R=R,
                        T=T,
                        FovY=FovY,
                        FovX=FovX,
                        image=image,
                        depth=depth,
                        normal=normal,
                        depth_confidence=depth_confidence,
                        image_path=image_path,
                        image_name="long_" + image_name,
                        width=image.size[0],
                        height=image.size[1],
                    )
                )
            else:
                train_cam_infos.append(
                    CameraInfo(
                        uid=idx,
                        R=R,
                        T=T,
                        FovY=FovY,
                        FovX=FovX,
                        image=image,
                        depth=depth,
                        normal=normal,
                        depth_confidence=depth_confidence,
                        image_path=image_path,
                        image_name="long_" + image_name,
                        width=image.size[0],
                        height=image.size[1],
                    )
                )
    with open(os.path.join(path, "short_capture", transformsfile)) as json_file:
        contents = json.load(json_file)

        test_frames = contents["frames"]
    for idx, frame in enumerate(test_frames):
        cam_name = os.path.join(path, "short_capture", frame["file_path"])
        depth_name = os.path.join(path, "short_capture", frame["depth_file_path"])
        if "fl_x" in frame:
            fl_x = frame["fl_x"]
            fl_y = frame["fl_y"]
            w = frame["w"]
            h = frame["h"]
            cx, cy = frame["cx"], frame["cy"]

        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        if c2w.shape[0] == 3:
            c2w = np.vstack([c2w, np.array([0, 0, 0, 1])])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(
            w2c[:3, :3]
        )  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_path = os.path.join(path, "short_capture", cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)

        depth_path = os.path.join(path, "short_capture", depth_name)
        depth = Image.open(depth_path)
        depth = np.array(depth) / 1000.0

        confidence_path = os.path.join(
            path,
            "short_capture",
            depth_name.replace("depth", "depth_normals_mask").replace("png", "npy"),
        )
        if not exists(confidence_path):
            depth_confidence = np.ones_like(np.array(depth), dtype=np.float32)
        else:
            depth_confidence = (np.load(confidence_path) / 1000.0)[..., 0]

        normal_path = os.path.join(
            path,
            "short_capture",
            cam_name.replace("images", "normals_from_pretrain").replace("jpg", "png"),
        )
        if not exists(normal_path):
            normal = np.zeros_like(np.array(image))
        else:
            normal = Image.open(normal_path)
            normal = np.array(normal) / 255.0

            h, w, _ = normal.shape
            normal = normal.reshape(-1, 3).transpose(1, 0)
            normal = (normal - 0.5) * 2
            normal = (R @ normal).T
            normal = normal / np.linalg.norm(normal, axis=1, keepdims=True)
            normal = normal.reshape(h, w, 3)

        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (
            1 - norm_data[:, :, 3:4]
        )
        image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

        fovx = focal2fov(fl_x, w)
        fovy = focal2fov(fl_y, h)
        FovY = fovy
        FovX = fovx

        test_cam_infos.append(
            CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                depth=depth,
                normal=normal,
                depth_confidence=depth_confidence,
                image_path=image_path,
                image_name="short_" + image_name,
                width=image.size[0],
                height=image.size[1],
            )
        )

    return train_cam_infos, test_cam_infos


def readMuSHRoomInfo(path, white_background, eval, extension=".jpg"):
    print("Reading Training Transforms")
    train_cam_infos, test_cam_infos = readCamerasFromTransformations(
        path, "transformations_colmap.json", 1, white_background, extension
    )

    nerf_normalization = getNerfppNorm(train_cam_infos)
    num_pts = 1_000_000

    ply_path = os.path.join(path, "long_capture/iphone_pointcloud.ply")
    if not os.path.exists(ply_path):
        meta = json.load(
            open(os.path.join(path, "long_capture/transformations_colmap.json"), "r")
        )
        generate_ply_from_rgbd(train_cam_infos, meta, num_pts, ply_path)
    try:
        ply = o3d.io.read_point_cloud(ply_path)
        # estimate normal for point cloud
        ply.estimate_normals()
        positions = np.array(ply.points)
        colors = np.vstack(ply.colors)
        normals = np.vstack(ply.normals)
        pcd = BasicPointCloud(points=positions, colors=colors, normals=normals)
        print("loaded point cloud")
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readCamerasFromCalibration(
    path, transformsfile, white_background, extension=".png"
):
    cam_infos = []
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        for idx, frame in enumerate(contents):

            # NeRF 'transform_matrix' is a camera-to-world transform
            w2c = np.array(contents[frame]["R"]).reshape(3, 3)
            T = np.array(contents[frame]["T"])

            R = np.transpose(w2c)  # R is stored transposed due to 'glm' in CUDA code
            K = np.array(contents[frame]["K"]).reshape(3, 3)
            focal_length_x = K[0, 0]
            focal_length_y = K[1, 1]

            image_path = os.path.join(path, "images", frame + ".jpg")
            image_name = frame
            image = Image.open(image_path)
            mask = Image.open(image_path.replace("images", "mask"))
            im_data = np.array(image.convert("RGBA"))
            mask = np.array(mask) / 255
            im_data = im_data * mask[..., np.newaxis]
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (
                1 - norm_data[:, :, 3:4]
            )
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
            mask = Image.fromarray((mask * 255).astype(np.uint8), "L")
            image_size = contents[frame]["imgSize"]
            width = image_size[0]
            height = image_size[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    mask=mask,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                )
            )

    return cam_infos


def readTransformsInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos, test_cam_infos = readCamerasFromTransforms(
        path, "transforms.json", white_background, extension
    )
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "point_cloud.ply")
    num_pts = 100_000

    if not os.path.exists(ply_path):
        meta = json.load(open(os.path.join(path, "transforms.json"), "r"))
        generate_ply_from_rgbd(train_cam_infos, meta, num_pts, ply_path)

    try:
        ply = o3d.io.read_point_cloud(ply_path)
        ply.estimate_normals()
        positions = np.array(ply.points)
        colors = np.vstack(ply.colors) / 255.0
        normals = np.vstack(ply.normals)
        pcd = BasicPointCloud(points=positions, colors=colors, normals=normals)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "MuSHRoom": readMuSHRoomInfo,
    "Transforms": readTransformsInfo,
}
