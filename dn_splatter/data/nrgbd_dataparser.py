"""NRGBD dataparser

Download data with: 
    python dn_splatter/data/download_scripts/nrgbd_download.py
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

CONSOLE = Console()


@dataclass
class NRGBDDataParserConfig(DataParserConfig):
    """Scene dataset parser config"""

    _target: Type = field(default_factory=lambda: NRGBDDataparser)
    """target class to instantiate"""
    data: Path = Path("datasets/NRGBD/")
    """Root directory specifying location of NRGBD dataset."""
    sequence: Literal[
        "breakfast_room",
        "complete_kitchen",
        "green_room",
        "grey_white_room",
        "kitchen",
        "morning_apartment",
        "staircase",
        "thin_geometry",
        "whiteroom",
    ] = "whiteroom"
    """room name"""
    depth_mode: Literal["sensor", "mono", "all"] = "sensor"
    """Which depth data to load"""
    is_euclidean_depth: bool = False
    """Whether input depth maps are Euclidean distances (or z-distances)."""
    depth_name: Literal["depth", "depth_with_noise"] = "depth"
    """Which sensor depth to load"""
    load_normals: bool = True
    """Set to true to use ground truth normal maps"""
    load_depths: bool = True
    """Whether to load depth maps"""
    normal_format: Literal["omnidata", "dsine"] = "omnidata"
    """Which monocular normal network was used to generate normals (they have different coordinate systems)."""
    normals_from: Literal["depth", "pretrained"] = "depth"
    """If no ground truth normals, generate normals either from sensor depths or from pretrained model."""
    load_pcd_normals: bool = False
    """Whether to load pcd normals for normal initialisation"""
    initialisation_type: Literal["mesh", "rgbd"] = "rgbd"
    """Which method to generate initial point clouds from"""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "none"
    """The method to use for orientation."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    load_every: int = 15
    """load every n'th frame from the dense trajectory"""
    skip_every_for_val_split: int = 10
    """sub sampling validation images"""
    auto_orient: bool = False
    """automatically orient the scene such that the up direction is the same as the viewer's up direction"""
    center_method: Literal["poses", "focus", "none"] = "none"
    """The method to use to center the poses."""
    depth_unit_scale_factor: float = 0.001  # 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""
    num_sfm_points: int = 100_000
    """Num points to load"""
    random_seed: bool = False
    """Random gaussian xyz initialisation"""


@dataclass
class NRGBDDataparser(DataParser):
    config: NRGBDDataParserConfig

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.images_root_path}/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.depth_root_path}/*.png"))
        return color_paths, depth_paths

    def _generate_dataparser_outputs(
        self, split="train"
    ):  # pylint: disable=unused-argument,too-many-statements
        self.input_folder = self.config.data / self.config.sequence
        self.images_root_path = self.input_folder / Path("images")
        self.depth_root_path = self.input_folder / Path(self.config.depth_name)

        image_filenames, sensor_depth_filenames = self.get_filepaths()
        assert len(image_filenames) == len(sensor_depth_filenames)
        gt_poses_path = self.input_folder / Path("poses.txt")
        gt_camera_to_worlds, valid = self.load_poses(gt_poses_path)
        poses_path = self.input_folder / Path("trainval_poses.txt")
        camera_to_worlds, valid = self.load_poses(poses_path)

        init_pose = np.array(camera_to_worlds[0]).astype(np.float32)
        init_gt_pose = np.array(gt_camera_to_worlds[0]).astype(np.float32)
        align_matrix = init_gt_pose @ np.linalg.inv(init_pose)
        self.camera_to_worlds = [
            align_matrix @ np.array(pose).astype(np.float32)
            for pose in camera_to_worlds
        ]

        # N-RGBD poses are already in OpenGl format. No need to convert them like other parsers.
        camera_to_worlds = torch.from_numpy(np.array(self.camera_to_worlds)).float()

        self.fx, self.fy = 554.2562584220408, 554.2562584220408
        self.height, self.width, _ = np.array(
            Image.open(Path(image_filenames[0]))
        ).shape
        self.cx, self.cy = self.width * 0.5, self.height * 0.5
        fx, fy, height, width, cx, cy = (
            torch.tensor([self.fy]),
            torch.tensor([self.fx]),
            torch.tensor([self.height]),
            torch.tensor([self.width]),
            torch.tensor([self.cx]),
            torch.tensor([self.cy]),
        )
        all_indices = list(range(len(image_filenames)))
        indices = all_indices[:: self.config.load_every]
        assert self.config.skip_every_for_val_split >= 1
        if split != "train":
            # eval split
            indices = indices[:: self.config.skip_every_for_val_split]
            print("eval", len(indices), indices)
        elif split == "train":
            # train split
            eval_indices = indices[:: self.config.skip_every_for_val_split]
            indices = [i for i in indices if i not in eval_indices]
            print("train", len(indices), indices)
        print(split, "number of images ", len(indices), indices)

        self.mesh_path = str(
            self.config.data / self.config.sequence / Path("gt_mesh.ply")
        )
        self.path_to_point_cloud = (
            self.config.data / self.config.sequence / "point_cloud.ply"
        )

        # generate xyz points
        if split == "train":
            self.camera_to_worlds = camera_to_worlds[indices]
            if self.config.initialisation_type == "rgbd":
                self._generate_ply_rgbd(
                    image_filenames=[image_filenames[i] for i in indices],
                    depth_filenames=[sensor_depth_filenames[i] for i in indices],
                )
            else:
                raise NotImplementedError
            # save transforms.json file, useful for generating normals from depth/mono-depth
            self._write_json(
                image_filenames=image_filenames,
                depth_filenames=sensor_depth_filenames,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                width=width,
                height=height,
                camera_to_worlds=camera_to_worlds,
            )

        if self.config.auto_orient:
            orientation_method = self.config.orientation_method
            camera_to_worlds, transform = camera_utils.auto_orient_and_center_poses(
                camera_to_worlds,
                method=orientation_method,
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
            height=height,
            width=width,
            camera_to_worlds=camera_to_worlds[indices, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )

        metadata = {}
        metadata.update(
            {
                "sensor_depth_filenames": [
                    Path(sensor_depth_filenames[idx]) for idx in indices
                ]
            }
        )
        metadata.update(
            {"depth_unit_scale_factor": self.config.depth_unit_scale_factor}
        )

        # save global transform matrix after orientation changes
        metadata.update({"transform": transform})

        metadata.update({"depth_mode": self.config.depth_mode})
        metadata.update({"is_euclidean_depth": self.config.is_euclidean_depth})
        metadata.update({"load_depths": self.config.load_depths})

        # process normals
        if self.config.load_normals:
            if self.config.normals_from == "depth":
                self.normal_save_dir = self.input_folder / Path("normals_from_depth")
            else:
                self.normal_save_dir = self.input_folder / Path("normals_from_pretrain")

        if self.config.load_normals and (
            not self.normal_save_dir.exists()
            or len(os.listdir(self.normal_save_dir)) == 0
        ):
            CONSOLE.print(
                f"[bold yellow]Could not find normals, generating them into {str(self.normal_save_dir)}"
            )
            (self.normal_save_dir).mkdir(exist_ok=True, parents=True)

            if self.config.normals_from == "pretrained":
                NormalsFromPretrained(data_dir=self.input_folder).main()
            else:
                normals_from_depths(
                    path_to_transforms=self.input_folder / "transforms.json",
                    save_path=self.normal_save_dir,
                    normal_format=self.config.normal_format,
                )

        # update metadata with normals
        if self.config.load_normals:
            normal_filenames = self.get_normal_filepaths()
            metadata.update(
                {"normal_filenames": [normal_filenames[idx] for idx in indices]}
            )
            metadata.update({"normal_format": self.config.normal_format})
        metadata.update({"load_normals": self.config.load_normals})

        # initialise gaussian means/colors and/or normals
        if not self.config.random_seed:
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
                        ply_file_path=self.path_to_point_cloud,
                        transform_matrix=transform,
                    )
                )

        scene_box = SceneBox(aabb=torch.tensor([[-1, -1, -1], [1, 1, 1]]))

        # TODO: check how to properly deal with masks
        dataparser_outputs = DataparserOutputs(
            image_filenames=[Path(image_filenames[idx]) for idx in indices],
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale=scale_factor,
            dataparser_transform=transform,
            metadata=metadata,
        )
        return dataparser_outputs

    def get_normal_filepaths(self):
        """Helper to load normal paths"""
        return natsorted(glob.glob(f"{self.normal_save_dir}/*.png"))

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

    def _generate_ply_rgbd(self, image_filenames, depth_filenames):
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
            rgb = np.array(Image.open(Path(image_filenames[i]))) / 255
            depth = (
                np.array(Image.open(Path(depth_filenames[i])))
                * self.config.depth_unit_scale_factor
            )
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
        out["w"] = width.item()
        out["h"] = height.item()
        out["frames"] = frames
        with open(base_dir / "transforms.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=4)

    def load_poses(self, posefile):
        file = open(posefile, "r")
        lines = file.readlines()
        file.close()
        poses = []
        valid = []
        lines_per_matrix = 4
        for i in range(0, len(lines), lines_per_matrix):
            if "nan" in lines[i]:
                valid.append(False)
                poses.append(np.eye(4, 4, dtype=np.float32).tolist())
            else:
                valid.append(True)
                pose_floats = [
                    [float(x) for x in line.split()]
                    for line in lines[i : i + lines_per_matrix]
                ]
                poses.append(pose_floats)

        return poses, valid


NRGBDDataParserSpecification = DataParserSpecification(
    config=NRGBDDataParserConfig(), description="NRGBD dataparser"
)

if __name__ == "__main__":
    parser = NRGBDDataparser(NRGBDDataParserConfig)._generate_dataparser_outputs()
