"""Dataparser for ScanNet++ dataset.

adapted form: https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/dataparsers/scannetpp_dataparser.py
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Type

import numpy as np
import open3d as o3d
import torch
from dn_splatter.data.scannetpp_utils.pointcloud_utils import generate_iPhone_pointcloud
from dn_splatter.scripts.align_depth import ColmapToAlignedMonoDepths
from dn_splatter.scripts.depth_from_pretrain import depth_from_pretrain
from dn_splatter.scripts.normals_from_pretrain import (
    NormalsFromPretrained,
    normals_from_depths,
)
from natsort import natsorted

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.colmap_dataparser import (
    ColmapDataParser,
    ColmapDataParserConfig,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class ScanNetppDataParserConfig(ColmapDataParserConfig):
    _target: Type = field(default_factory=lambda: ScanNetpp)
    """target class to instantiate"""
    data: Path = Path("datasets/scannetpp/data")
    """Directory to the root of the data."""
    sequence: Literal["8b5caf3398", "b20a261fdf"] = "b20a261fdf"
    """room name"""
    mode: Literal["dslr", "iphone"] = "iphone"
    """Which camera to use"""
    depth_mode: Literal["mono", "sensor", "none", "all"] = "sensor"
    """Which depth data to load"""
    is_euclidean_depth: bool = False
    """Whether input depth maps are Euclidean distances (or z-distances)."""
    load_depths: bool = True
    """Whether to load depth maps"""
    mono_pretrain: Literal["zoe"] = "zoe"
    """Which mono depth pretrain model to use."""
    load_normals: bool = True
    """Set to true to use ground truth normal maps"""
    normal_format: Literal["opencv", "opengl"] = "opengl"
    """Which format the normal maps in camera frame are saved in."""
    normals_from: Literal["pretrained", "none"] = "pretrained"
    """If no ground truth normals, generate normals either from sensor depths or from pretrained model."""
    load_pcd_normals: bool = False
    """Whether to load pcd normals for normal initialisation"""
    initialisation_type: Literal["mesh", "rgbd"] = "rgbd"
    """Which method to generate initial point clouds from"""

    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    scene_scale: float = 1.5
    """How much to scale the region of interest by. Default is 1.5 since the cameras are inside the rooms."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "none"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "none"
    """The method to use to center the poses."""
    auto_scale_poses: bool = False
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    load_every: int = 5
    """load every n'th frame from the dense trajectory"""
    skip_every_for_val_split: int = 10
    """sub sampling validation images"""
    load_3D_points: bool = True
    """Whether to load the 3D points from the colmap reconstruction."""
    num_sfm_points: int = 100_000
    """Num points to load"""
    assume_colmap_world_coordinate_convention: bool = True


@dataclass
class ScanNetpp(ColmapDataParser):
    """ScanNet++ DatasetParser"""

    config: ScanNetppDataParserConfig

    def _generate_dataparser_outputs(self, split="train", **kwargs):
        self.input_folder = self.config.data / self.config.sequence / self.config.mode
        assert (
            self.input_folder.exists()
        ), f"Data directory {self.input_folder} does not exist."

        if self.config.mode == "dslr":
            self.input_folder = (
                self.input_folder / "undistort_colmap" / self.config.sequence
            )
            colmap_path = self.input_folder / "colmap"
            data_dir = self.input_folder / "images"
            mask_dir = self.input_folder / "masks"

            if (colmap_path / "cameras.txt").exists():
                meta = self._get_all_images_and_cameras(colmap_path)
        elif self.config.mode == "iphone":
            colmap_path = self.input_folder / "colmap"
            data_dir = self.input_folder / "rgb"
            mask_dir = self.input_folder / "rgb_masks"

            if (colmap_path / "cameras.txt").exists():
                meta = self._get_all_images_and_cameras(colmap_path)
                depth_dir = self.input_folder / "depth"
        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []

        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        i_train = []
        i_eval = []
        # sort the frames by fname
        if self.config.mode == "dslr":
            frames = meta["frames"]
            train_test_split = json.load(
                open(
                    self.config.data
                    / self.config.sequence
                    / self.config.mode
                    / "train_test_lists.json",
                    "r",
                )
            )
            file_names = [f["file_path"].split("/")[-1] for f in frames]
            test_frames = [f for f in file_names if f in train_test_split["test"]]

        elif self.config.mode == "iphone":
            frames = meta["frames"]
        frames.sort(key=lambda x: x["file_path"])

        for idx, frame in enumerate(frames):
            filepath = Path(frame["file_path"]).name
            fname = data_dir / filepath

            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))

            fx.append(float(frame["fl_x"]))
            fy.append(float(frame["fl_y"]))
            cx.append(float(frame["cx"]))
            cy.append(float(frame["cy"]))
            height.append(int(frame["h"]))
            width.append(int(frame["w"]))
            distort.append(
                camera_utils.get_distortion_params(
                    k1=float(frame["k1"]) if "k1" in frame else 0.0,
                    k2=float(frame["k2"]) if "k2" in frame else 0.0,
                    k3=float(frame["k3"]) if "k3" in frame else 0.0,
                    k4=float(frame["k4"]) if "k4" in frame else 0.0,
                    p1=float(frame["p1"]) if "p1" in frame else 0.0,
                    p2=float(frame["p2"]) if "p2" in frame else 0.0,
                )
            )

            if meta.get("has_mask", True) and "mask_path" in frame:
                mask_filepath = Path(frame["mask_path"]).name
                mask_fname = mask_dir / mask_filepath
                mask_filenames.append(mask_fname)

            if self.config.mode == "dslr":
                if filepath in test_frames:
                    i_eval.append(idx)
                else:
                    i_train.append(idx)
            elif self.config.mode == "iphone":
                depth_filepath = filepath.replace("jpg", "png")
                depth_fname = depth_dir / depth_filepath
                depth_filenames.append(depth_fname)
        assert len(mask_filenames) == 0 or (
            len(mask_filenames) == len(image_filenames)
        ), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """

        if self.config.mode == "iphone":
            num_imgs = len(image_filenames)
            indices = list(range(num_imgs))
            assert self.config.skip_every_for_val_split >= 1
            eval_indices = indices[:: self.config.skip_every_for_val_split]
            i_eval = [i for i in indices if i in eval_indices]
            i_train = [i for i in indices if i not in eval_indices]

        if split == "train":
            indices = i_train
            if self.config.load_every > 1:
                indices = indices[:: self.config.load_every]
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")
        print(indices)

        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(
                f"[yellow] Dataset is overriding orientation method to {orientation_method}"
            )
        else:
            orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses, method=orientation_method, center_method=self.config.center_method
        )
        self.orient_transform = transform_matrix

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = (
            [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        )
        depth_filenames = (
            [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []
        )
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]
        self.poses = poses
        self.camera_to_worlds = poses[:, :3, :4]
        if split == "train":
            self._write_json(
                image_filenames,
                depth_filenames,
                fx[0],
                fy[0],
                cx[0],
                cy[0],
                width[0],
                height[0],
                poses[:, :3, :4],
            )

        metadata = {}

        # in x,y,z order
        # assumes that the scene is centered at the origin
        if not self.config.auto_scale_poses:
            # Set aabb_scale to scene_scale * the max of the absolute values of the poses
            aabb_scale = self.config.scene_scale * float(
                torch.max(torch.abs(poses[:, :3, 3]))
            )
        else:
            aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [
                    [-aabb_scale, -aabb_scale, -aabb_scale],
                    [aabb_scale, aabb_scale, aabb_scale],
                ],
                dtype=torch.float32,
            )
        )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        fx = torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = torch.tensor(width, dtype=torch.int32)[idx_tensor]
        distortion_params = torch.stack(distort, dim=0)[idx_tensor]
        self.width = width[0]
        self.height = height[0]
        self.fx = fx[0]
        self.fy = fy[0]
        self.cx = cx[0]
        self.cy = cy[0]

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
        )

        if "applied_transform" in meta:
            applied_transform = torch.tensor(
                meta["applied_transform"], dtype=transform_matrix.dtype
            )
            transform_matrix = transform_matrix @ torch.cat(
                [
                    applied_transform,
                    torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype),
                ],
                0,
            )
        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        # Load 3D points
        if self.config.load_3D_points and split == "train":
            if self.config.mode == "dslr":
                metadata.update(
                    self._load_3D_points(colmap_path, transform_matrix, scale_factor)
                )

            elif self.config.mode == "iphone":
                # iphone doesn't have sparse colmap points, need to build init point clouds from dense reconstruction
                self.path_to_point_cloud = self.input_folder / "point_cloud.ply"
                if not self.path_to_point_cloud.exists():
                    generate_iPhone_pointcloud(self.input_folder, meta, i_train)
                out, selected_indices = self._load_iphone_3D_points(
                    self.path_to_point_cloud, transform_matrix, scale_factor
                )
                metadata.update(out)

            if self.config.load_pcd_normals:
                metadata.update(
                    self._load_points3D_normals(
                        ply_file_path=self.path_to_point_cloud,
                        selected_indices=selected_indices,
                    )
                )

        metadata.update({"depth_unit_scale_factor": 1e-3})

        # process normal
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
                if self.config.mode == "dslr":
                    NormalsFromPretrained(data_dir=self.input_folder).main()
                elif self.config.mode == "iphone":
                    NormalsFromPretrained(data_dir=self.input_folder).main()
            else:
                normals_from_depths(
                    path_to_transforms=self.input_folder / self.config.transforms_path,
                    save_path=self.normal_save_dir,
                    normal_format=self.config.normal_format,
                )

        # update metadata with normals
        if self.config.load_normals and split == "train":
            normal_filenames = self.get_normal_filepaths()
            normal_filenames = [
                Path(filename)
                for filename in normal_filenames
                if Path(filename).stem in [n.stem for n in image_filenames]
            ]
            assert len(normal_filenames) == len(image_filenames)
            metadata.update({"normal_filenames": normal_filenames})
            metadata.update({"normal_format": self.config.normal_format})

        if split == "train":
            metadata.update({"load_normals": self.config.load_normals})
        else:
            metadata.update({"load_normals": False})

        # process depth
        metadata.update({"depth_mode": self.config.depth_mode})
        metadata.update({"is_euclidean_depth": self.config.is_euclidean_depth})
        metadata.update({"load_depths": self.config.load_depths})

        if self.config.depth_mode in ["mono", "all"]:
            self.depth_save_dir = self.input_folder / Path("mono_depth")
            if (
                not self.depth_save_dir.exists()
                or len(os.listdir(self.depth_save_dir)) == 0
            ):
                if self.config.mode == "dslr":
                    ColmapToAlignedMonoDepths(
                        data=self.input_folder,
                        sparse_path=Path("colmap"),
                        mono_depth_network=self.config.mono_pretrain,
                        img_dir_name="images",
                    ).main()
                elif self.config.mode == "iphone":
                    depth_from_pretrain(
                        input_folder=self.input_folder,
                        img_dir_name="rgb",
                        path_to_transforms=None,
                        save_path=self.depth_save_dir,
                        create_new_transforms=False,
                        return_mode="mono-aligned",
                    )

            self.mono_depth_filenames = self.get_depth_filepaths()
            if split == "train":
                metadata.update(
                    {
                        "mono_depth_filenames": self.mono_depth_filenames
                        if len(self.mono_depth_filenames) == len(image_filenames)
                        else [Path(self.mono_depth_filenames[idx]) for idx in indices]
                        if len(self.mono_depth_filenames) > 0
                        else None
                    }
                )
            else:
                # dummy data
                metadata.update(
                    {
                        "mono_depth_filenames": [
                            Path(idx)
                            for idx in self.mono_depth_filenames[: len(image_filenames)]
                        ]
                        if len(self.mono_depth_filenames) > 0
                        else None
                    }
                )

        if self.config.depth_mode in ["sensor", "all"]:
            metadata.update(
                {
                    "sensor_depth_filenames": depth_filenames
                    if len(depth_filenames) > 0
                    else None
                }
            )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata=metadata,
        )

        return dataparser_outputs

    def get_normal_filepaths(self):
        """Helper to load normal paths"""
        return natsorted(glob.glob(f"{self.normal_save_dir}/*.png"))

    def get_depth_filepaths(self):
        """Helper to load depth paths"""
        depth_list = natsorted(glob.glob(f"{self.depth_save_dir}/*_aligned.npy"))
        return depth_list

    def _load_iphone_3D_points(
        self, ply_file_path: Path, transform_matrix: torch.Tensor, scale_factor: float
    ):
        """load pointcloud from ply file"""
        pcd = o3d.io.read_point_cloud(str(ply_file_path))
        points3D = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32))

        points3D = (
            torch.cat((points3D, torch.ones_like(points3D[..., :1])), -1)
            @ self.orient_transform.T
        )
        points3D = points3D[..., :3]
        points3D *= scale_factor
        points3D_rgb = torch.from_numpy((np.asarray(pcd.colors) * 255).astype(np.uint8))
        selected_indices = torch.randperm(points3D.shape[0])[
            : self.config.num_sfm_points
        ]
        out = {
            "points3D_xyz": points3D[selected_indices],
            "points3D_rgb": points3D_rgb[selected_indices],
        }
        return out, selected_indices

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
            file_path = img.relative_to(base_dir)
            depth_path = depth.relative_to(base_dir)
            frame = {
                "file_path": file_path.as_posix(),
                "depth_file_path": depth_path.as_posix(),
                "transform_matrix": c2w.cpu().numpy().tolist(),
            }
            frames.append(frame)
        out = {}
        out["fl_x"] = fx
        out["fl_y"] = fy
        out["k1"] = 0
        out["k2"] = 0
        out["p1"] = 0
        out["p2"] = 0
        out["cx"] = cx
        out["cy"] = cy
        out["w"] = width
        out["h"] = height
        out["frames"] = frames
        with open(base_dir / "transforms.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=4)

    def _load_points3D_normals(self, ply_file_path: Path, selected_indices):
        pcd = o3d.io.read_point_cloud(str(ply_file_path))
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.normalize_normals()
        points3D_normals = torch.from_numpy(np.asarray(pcd.normals, dtype=np.float32))
        return {"points3D_normals": points3D_normals[selected_indices]}


ScanNetppDataParserSpecification = DataParserSpecification(
    config=ScanNetppDataParserConfig(), description="Scannet++ dataparser"
)
