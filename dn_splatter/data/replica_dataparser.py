"""Replica dataparser

to download preprocessed Replica dataset:
    python dn_splatter/data/replica_utils/replica_download.py
"""

from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import numpy as np
import open3d as o3d
import torch
from dn_splatter.scripts.normals_from_pretrain import (
    NormalsFromPretrained,
    normals_from_depths,
)
from dn_splatter.utils import camera_utils as dn_splatter_camera_utils
from natsort import natsorted
from PIL import Image
from rich.console import Console
from typing_extensions import Literal

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfstudio.utils.io import load_from_json

CONSOLE = Console()


@dataclass
class ReplicaDataParserConfig(DataParserConfig):
    """Scene dataset parser config"""

    _target: Type = field(default_factory=lambda: ReplicaDataparser)
    """target class to instantiate"""
    data: Path = Path("datasets/Replica/")
    """Root directory specifying location of replica dataset."""
    sequence: Literal[
        "office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2"
    ] = "room0"
    """room name"""
    depth_mode: Literal["sensor", "mono", "all"] = "sensor"
    """Which depth data to load"""
    is_euclidean_depth: bool = False
    """Whether input depth maps are Euclidean distances (or z-distances)."""
    load_depths: bool = True
    """Whether to load depth maps"""
    load_normals: bool = True
    """Set to true to use ground truth normal maps"""
    normal_format: Literal["opencv", "opengl"] = "opengl"
    """Which format the normal maps in camera frame are saved in."""
    normals_from: Literal["depth", "pretrained"] = "depth"
    """If no ground truth normals, generate normals either from sensor depths or from pretrained model."""
    load_pcd_normals: bool = True
    """Whether to load pcd normals for normal initialisation"""
    initialisation_type: Literal["mesh", "rgbd"] = "rgbd"
    """Which method to generate initial point clouds from"""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "none"
    """The method to use for orientation."""
    auto_scale_poses: bool = False
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    load_every: int = 25
    """load every n'th frame from the dense trajectory"""
    skip_every_for_val_split: int = 5
    """sub sampling validation images"""
    auto_orient: bool = False
    """automatically orient the scene such that the up direction is the same as the viewer's up direction"""
    center_method: Literal["poses", "focus", "none"] = "none"
    """The method to use to center the poses."""
    num_sfm_points: int = 100_000
    """Number of sfm points to init"""
    save_for_inria: bool = False
    """Save train images to /input folder for running with inria code"""


@dataclass
class ReplicaDataparser(DataParser):
    config: ReplicaDataParserConfig

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/results/depth*.png"))
        return color_paths, depth_paths

    def get_normal_filepaths(self):
        return natsorted(glob.glob(f"{self.normal_save_dir}/*.png"))

    def load_poses(self):
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        for i in range(self.num_imgs):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
        return torch.stack(poses)

    def _generate_dataparser_outputs(
        self, split="train"
    ):  # pylint: disable=unused-argument,too-many-statements
        self.input_folder = self.config.data / self.config.sequence
        cam_data = load_from_json(self.config.data / "cam_params.json")["camera"]
        self.pose_path = self.config.data / self.config.sequence / "traj.txt"
        self.mesh_path = str(
            self.config.data / Path(str(self.config.sequence) + "_mesh.ply")
        )
        assert self.pose_path.exists()

        color_paths, depth_paths = self.get_filepaths()
        self.num_imgs = len(color_paths)
        poses = self.load_poses()
        all_indices = list(range(self.num_imgs))
        indices = all_indices[:: self.config.load_every]
        assert self.config.skip_every_for_val_split >= 1
        if split != "train":
            # eval split
            indices = indices[:: self.config.skip_every_for_val_split]
        elif split == "train":
            # train split
            eval_indices = indices[:: self.config.skip_every_for_val_split]
            indices = [i for i in indices if i not in eval_indices]

        image_filenames = color_paths
        sensor_depth_filenames = depth_paths
        self.width = cam_data["w"]
        self.height = cam_data["h"]
        cam_data["scale"]
        self.fx = cam_data["fx"]
        self.fy = cam_data["fy"]
        self.cx = cam_data["cx"]
        self.cy = cam_data["cy"]

        camera_to_worlds = poses

        fx = torch.tensor(self.fx)
        fy = torch.tensor(self.fy)
        cx = torch.tensor(self.cx)
        cy = torch.tensor(self.cy)

        # Convert from COLMAP's/OPENCV's camera coordinate system to nerfstudio
        camera_to_worlds[:, 0:3, 1:3] *= -1
        self.camera_to_worlds = camera_to_worlds

        if self.config.auto_orient:
            camera_to_worlds, transform = camera_utils.auto_orient_and_center_poses(
                camera_to_worlds,
                method=self.config.orientation_method,
                center_method=self.config.center_method,
            )
        else:
            transform = torch.eye(4)[:3, :4]

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(camera_to_worlds[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        camera_to_worlds[:, :3, 3] *= scale_factor

        distort = camera_utils.get_distortion_params()

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distort,
            height=self.height,
            width=self.width,
            camera_to_worlds=camera_to_worlds[indices, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )
        metadata = {}

        self.path_to_point_cloud = (
            self.config.data / self.config.sequence / Path("point_cloud.ply")
        )
        if split == "train":
            if not self.path_to_point_cloud.exists():
                CONSOLE.print(
                    f"[bold yellow]Could not find point cloud, generating it into {str(self.path_to_point_cloud)}"
                )
                if self.config.initialisation_type == "mesh":
                    self._generate_ply_from_mesh()
                elif self.config.initialisation_type == "rgbd":
                    self._generate_ply_from_rgbd()

        metadata.update(
            self._load_3D_points(
                ply_file_path=self.path_to_point_cloud,
                transform_matrix=transform,
                scale_factor=scale_factor,
            )
        )

        if self.config.load_pcd_normals:
            metadata.update(
                self._load_points3D_normals(
                    ply_file_path=self.path_to_point_cloud, transform_matrix=transform
                )
            )

        metadata.update(
            {
                "sensor_depth_filenames": [
                    Path(sensor_depth_filenames[idx]) for idx in indices
                ]
            }
        )
        metadata.update({"depth_mode": self.config.depth_mode})
        metadata.update({"is_euclidean_depth": self.config.is_euclidean_depth})
        metadata.update({"load_depths": self.config.load_depths})

        # write transforms.json
        if split == "train":
            self._write_json(
                [image_filenames[idx] for idx in indices],
                [Path(sensor_depth_filenames[idx]) for idx in indices],
                fx,
                fy,
                cx,
                cy,
                self.width,
                self.height,
                camera_to_worlds[indices, :3, :4],
            )

        if self.config.normals_from == "depth":
            self.normal_save_dir = self.input_folder / Path("normals_from_depth")
        else:
            self.normal_save_dir = self.input_folder / Path("normals_from_pretrain")

        if self.config.load_normals and (
            not (self.normal_save_dir).exists()
            or len(os.listdir(self.normal_save_dir)) == 0
        ):
            CONSOLE.print(
                f"[bold yellow]Could not find normals, generating them into {str(self.normal_save_dir)}"
            )
            self.normal_save_dir.mkdir(exist_ok=True, parents=True)
            if self.config.normals_from == "depth":
                normals_from_depths(
                    path_to_transforms=Path(image_filenames[0]).parent.parent
                    / "transforms.json",
                    normal_format=self.config.normal_format,
                )
            elif self.config.normals_from == "pretrained":
                NormalsFromPretrained(data_dir=self.input_folder).main()
            else:
                raise NotImplementedError

        if self.config.load_normals:
            normal_filenames = self.get_normal_filepaths()

        if self.config.load_normals:
            metadata.update({"normal_filenames": normal_filenames})

        # save global transform matrix after orientation changes
        metadata.update({"transform": transform})

        # scale is from:
        # https://github.com/cvg/nice-slam/blob/7af15cc33729aa5a8ca052908d96f495e34ab34c/configs/Replica/replica.yaml#L32C2-L32C2
        metadata.update({"depth_unit_scale_factor": 1 / 6553.5})
        metadata.update({"normal_format": self.config.normal_format})
        metadata.update({"load_normals": self.config.load_normals})

        scene_box = SceneBox(aabb=torch.tensor([[-1, -1, -1], [1, 1, 1]]))
        dataparser_outputs = DataparserOutputs(
            image_filenames=[image_filenames[idx] for idx in indices],
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale=scale_factor,
            dataparser_transform=transform,
            metadata=metadata,
        )

        if self.config.save_for_inria and split == "train":
            from shutil import copy2, rmtree

            images = [Path(image_filenames[idx]) for idx in indices]
            destination_folder = Path(images[0].parent.parent / "images")
            if destination_folder.is_dir():
                rmtree(destination_folder)
            for image in images:
                destination_folder.mkdir(exist_ok=True, parents=True)
                destination_path = destination_folder / image.name
                copy2(image, destination_path)
        return dataparser_outputs

    def _generate_ply_from_mesh(self):
        mesh = o3d.io.read_point_cloud(self.mesh_path)
        assert self.config.num_sfm_points < len(
            mesh.points
        ), f"chosen num of sfm points {self.config.num_sfm_points} is larger than gt mesh size {len(mesh.points)}"
        sampling_ratio = (self.config.num_sfm_points + 1) / len(mesh.points)
        pd = mesh.random_down_sample(sampling_ratio)
        o3d.io.write_point_cloud(str(self.path_to_point_cloud), pd)
        return pd

    def _generate_ply_from_rgbd(self):
        image_filenames, depth_filenames = self.get_filepaths()
        c2w = torch.from_numpy(np.array(self.camera_to_worlds)).float()
        c2w = torch.matmul(
            c2w, torch.from_numpy(dn_splatter_camera_utils.OPENGL_TO_OPENCV).float()
        )
        img_size = (self.width, self.height)
        point_list = []
        color_list = []

        pixels_per_frame = int(self.width * self.height)
        samples_per_frame = (self.config.num_sfm_points + len(image_filenames)) // len(
            image_filenames
        )
        indices = random.sample(range(pixels_per_frame), samples_per_frame)

        for i in range(len(c2w)):
            rgb = np.array(Image.open(Path(image_filenames[i]))) / 255.0
            depth = np.array(Image.open(Path(depth_filenames[i]))) / 6553.5
            rgb = torch.from_numpy(rgb).float()
            depth = torch.from_numpy(depth).float()
            points, colors = dn_splatter_camera_utils.get_colored_points_from_depth(
                depths=depth,
                rgbs=rgb,
                c2w=c2w[i],
                fx=self.fx,
                fy=self.fy,
                cx=self.cx,
                cy=self.cy,
                img_size=img_size,
                mask=indices,
            )
            point_list.append(points)
            color_list.append(colors)
        points = np.concatenate(point_list, axis=0)
        colors = np.concatenate(color_list, axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(str(self.path_to_point_cloud), pcd)
        return pcd

    def _load_3D_points(
        self, ply_file_path: Path, transform_matrix: torch.Tensor, scale_factor: float
    ):
        pcd = o3d.io.read_point_cloud(str(ply_file_path))

        points3D = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32))
        points3D = (
            torch.cat((points3D, torch.ones_like(points3D[..., :1])), -1)
            @ transform_matrix.T
        )
        points3D *= scale_factor
        points3D_rgb = torch.from_numpy((np.asarray(pcd.colors) * 255).astype(np.uint8))
        out = {"points3D_xyz": points3D, "points3D_rgb": points3D_rgb}
        return out

    def _load_points3D_normals(
        self, ply_file_path: Path, transform_matrix: torch.Tensor
    ):
        pcd = o3d.io.read_point_cloud(str(ply_file_path))
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.normalize_normals()
        points3D_normals = torch.from_numpy(np.asarray(pcd.normals, dtype=np.float32))
        points3D_normals = (
            torch.cat(
                (points3D_normals, torch.ones_like(points3D_normals[..., :1])), -1
            )
            @ transform_matrix.T
        )
        return {"points3D_normals": points3D_normals}

    def _write_json(
        self,
        image_filenames,
        depth_filenames,
        fx,
        fy,
        cx,
        cy,
        width,
        height,
        camera_to_worlds,
    ):
        frames = []
        base_dir = Path(image_filenames[0]).parent.parent
        for img, depth, c2w in zip(image_filenames, depth_filenames, camera_to_worlds):
            img = Path(img)
            depth = Path(depth)
            file_path = img.relative_to(base_dir)
            depth_path = depth.relative_to(base_dir)
            frame = {
                "file_path": file_path.as_posix(),
                "transform_matrix": c2w.cpu().numpy().tolist(),
                "depth_file_path": depth_path.as_posix(),
            }
            frames.append(frame)
        out = {}
        out["fl_x"] = fx.item()
        out["fl_y"] = fy.item()
        out["k1"] = 0
        out["k2"] = 0
        out["p1"] = 0
        out["p2"] = 0
        out["cx"] = cx.item()
        out["cy"] = cy.item()
        out["w"] = width
        out["h"] = height
        out["frames"] = frames
        with open(base_dir / "transforms.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=4)


def filter_list(list_to_filter, indices):
    """Returns a copy list with only selected indices"""
    if list_to_filter:
        return [list_to_filter[i] for i in indices]
    else:
        return []


ReplicaDataParserSpecification = DataParserSpecification(
    config=ReplicaDataParserConfig(), description="Replica dataparser"
)
