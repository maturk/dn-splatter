from __future__ import annotations

import math
import glob
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Literal, Optional, Type, List

import cv2
import sys
import numpy as np
from PIL import Image
import open3d as o3d
import torch
from dn_splatter.scripts.align_depth import ColmapToAlignedMonoDepths
from dn_splatter.scripts.normals_from_pretrain import (
    NormalsFromPretrained,
    normals_from_depths,
)
from natsort import natsorted
from rich.console import Console
from rich.prompt import Confirm

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras
from nerfstudio.data.dataparsers.colmap_dataparser import (
    ColmapDataParser,
    ColmapDataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfstudio.data.utils import colmap_parsing_utils as colmap_utils
from nerfstudio.process_data.colmap_utils import colmap_to_json
from nerfstudio.utils.rich_utils import CONSOLE, status
from nerfstudio.utils.scripts import run_command


MAX_AUTO_RESOLUTION = 1600
CONSOLE = Console()


@dataclass
class CoolerMapDataParserConfig(ColmapDataParserConfig):
    _target: Type = field(default_factory=lambda: CoolerMapDataParser)

    depth_mode: Literal["mono", "none"] = "mono"
    """Which depth data to load"""
    is_euclidean_depth: bool = False
    """Whether input depth maps are Euclidean distances (or z-distances)."""
    load_depths: bool = False
    """Whether to load depth maps"""
    mono_pretrain: Literal["zoe"] = "zoe"
    """Which mono depth pretrain model to use."""
    load_normals: bool = False
    """Set to true to use ground truth normal maps"""
    normal_format: Literal["opencv", "opengl"] = "opengl"
    """Which format the normal maps in camera frame are saved in."""
    normals_from: Literal["depth", "pretrained"] = "pretrained"
    """If no ground truth normals, generate normals either from sensor depths or from pretrained model."""
    load_pcd_normals: bool = True
    """Whether to load pcd normals for normal initialisation"""
    load_3D_points: bool = True
    """Whether to load the 3D points from the colmap reconstruction."""
    eval_mode: Literal["fraction", "filename", "interval", "all"] = "interval"
    """
    Interval uses every nth frame for eval (used by most academic papers, e.g. MipNerf360, GSplat).
    """
    load_every: int = 1  # 30 for eval train split
    """load every n'th frame from the dense trajectory from the train split"""
    eval_interval: int = 8
    """eval interval"""
    depth_unit_scale_factor: float = 1
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""
    colmap_path: Path = Path("colmap/sparse/0")
    """Path to the colmap reconstruction directory relative to the data path."""
    images_path: Path = Path("images")
    """Path to images directory relative to the data path."""
    depths_path: Optional[Path] = None
    """Path to depth maps directory. If not set, depths are not loaded."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "none"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "none"
    """The method to use to center the poses."""
    auto_scale_poses: bool = False
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    downscale_factor: Optional[int] = None


class CoolerMapDataParser(ColmapDataParser):
    config: CoolerMapDataParserConfig

    def __init__(self, config: CoolerMapDataParserConfig):
        super().__init__(config)
        self.config = config
        self._downscale_factor = None

    def get_depth_filepaths(self):
        # TODO this only returns aligned monodepths right now
        depth_paths = natsorted(
            glob.glob(f"{self.config.data}/mono_depth/*_aligned.npy")
        )
        depth_paths = [Path(depth_path) for depth_path in depth_paths]
        return depth_paths

    def get_normal_filepaths(self):
        normal_paths = natsorted(glob.glob(f"{self.normal_save_dir}/*.png"))
        normal_paths = [Path(normal_path) for normal_path in normal_paths]
        return normal_paths

    def _generate_dataparser_outputs(self, split: str = "train", **kwargs):
        assert (
            self.config.data.exists()
        ), f"Data directory {self.config.data} does not exist."
        colmap_path = self.config.data / self.config.colmap_path
        assert colmap_path.exists(), f"Colmap path {colmap_path} does not exist."

        meta = self._get_all_images_and_cameras(colmap_path)
        camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]

        image_filenames = []
        mask_filenames = []
        poses = []

        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        for frame in meta["frames"]:
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
            image_filenames.append(Path(frame["file_path"]))
            poses.append(frame["transform_matrix"])
            if "mask_path" in frame:
                mask_filenames.append(Path(frame["mask_path"]))

        assert len(mask_filenames) == 0 or (
            len(mask_filenames) == len(image_filenames)
        ), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """

        poses = [
            pose for img, pose in natsorted(zip(image_filenames, poses), lambda x: x[0])
        ]
        image_filenames = natsorted(image_filenames)

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor
        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        indices = self._get_image_indices(image_filenames, split)
        if split == "train":
            indices = indices[:: self.config.load_every]

        metadata = {}

        # load depths
        depth_filenames = []
        if self.config.depth_mode != "none" and self.config.load_depths:
            if not (self.config.data / "mono_depth").exists():
                CONSOLE.print(
                    "Load depth has been set to true, but could not find mono_depth path. Trying to generate aligned mono depth frames."
                )
                ColmapToAlignedMonoDepths(
                    data=self.config.data, mono_depth_network=self.config.mono_pretrain
                ).main()
            depth_filenames = self.get_depth_filepaths()

        # load normals
        normal_filenames = []
        if self.config.normals_from == "depth":
            self.normal_save_dir = self.config.data / Path("normals_from_depth")
        else:
            self.normal_save_dir = self.config.data / Path("normals_from_pretrain")

        if self.config.load_normals:
            if not (self.normal_save_dir).exists() or len(os.listdir(self.normal_save_dir)) == 0:
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
                    NormalsFromPretrained(data_dir=self.config.data).main()
                else:
                    raise NotImplementedError
            normal_filenames = self.get_normal_filepaths()


        image_filenames, mask_filenames, depth_filenames, normal_filenames, downscale_factor = self._setup_downscale_factor(
            image_filenames, mask_filenames, depth_filenames, normal_filenames
        )

        if self.config.load_depths:
            metadata["mono_depth_filenames"] = [
                Path(depth_filenames[i]) for i in indices
            ]

        if self.config.load_normals:
            metadata["normal_filenames"] = [
                Path(normal_filenames[i]) for i in indices
            ]

        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = (
            [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        )
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

                # in x,y,z order
        # assumes that the scene is centered at the origin
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

        fx = torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = torch.tensor(width, dtype=torch.int32)[idx_tensor]
        distortion_params = torch.stack(distort, dim=0)[idx_tensor]

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

        cameras.rescale_output_resolution(
            scaling_factor=1.0 / downscale_factor, scale_rounding_mode=self.config.downscale_rounding_mode
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

        if self.config.load_3D_points:
            # Load 3D points
            metadata.update(
                self._load_3D_points(colmap_path, transform_matrix, scale_factor)
            )

        metadata.update({"depth_mode": self.config.depth_mode})
        metadata.update({"load_depths": self.config.load_depths})
        metadata.update({"is_euclidean_depth": self.config.is_euclidean_depth})
        metadata.update({"load_normals": self.config.load_normals})
        metadata.update({"normal_format": self.config.normal_format})

        if self.config.load_pcd_normals:
            metadata.update(
                self._load_points3D_normals(points_3d=metadata["points3D_xyz"])
            )

        # write json
        colmap_to_json(
            recon_dir=self.config.data / self.config.colmap_path,
            output_dir=self.config.data,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                **metadata,
            },
        )

        return dataparser_outputs
    
    def _downscale_numpy(
        self,
        paths,
        get_fname,
        downscale_factor: int,
        downscale_rounding_mode: str = "floor",
        nearest_neighbor: bool = False,
    ):
        def calculate_scaled_size(original_width, original_height, downscale_factor, mode="floor"):
            if mode == "floor":
                return math.floor(original_width / downscale_factor), math.floor(original_height / downscale_factor)
            elif mode == "round":
                return round(original_width / downscale_factor), round(original_height / downscale_factor)
            elif mode == "ceil":
                return math.ceil(original_width / downscale_factor), math.ceil(original_height / downscale_factor)
            else:
                raise ValueError("Invalid mode. Choose from 'floor', 'round', or 'ceil'.")

        with status(msg="[bold yellow]Downscaling images...", spinner="growVertical"):
            assert downscale_factor > 1
            assert isinstance(downscale_factor, int)
            filepath = next(iter(paths))
            img = np.load(filepath)
            w, h = img.shape[1], img.shape[0]
            w_scaled, h_scaled = calculate_scaled_size(w, h, downscale_factor, downscale_rounding_mode)
            # Using %05d ffmpeg commands appears to be unreliable (skips images).
            for path in paths:
                img = np.load(path)
                img = cv2.resize(
                    img, (w_scaled, h_scaled), interpolation=cv2.INTER_NEAREST
                )
                path_out = get_fname(path)
                path_out.parent.mkdir(parents=True, exist_ok=True)
                np.save(path_out, img)
                
        CONSOLE.log("[bold green]:tada: Done downscaling images.")

    def _setup_downscale_factor(
        self, image_filenames: List[Path], mask_filenames: List[Path], depth_filenames: List[Path], normal_filenames: List[Path]
    ):
        """
        Setup the downscale factor for the dataset. This is used to downscale the images and cameras.
        """

        def get_fname(parent: Path, filepath: Path) -> Path:
            """Returns transformed file name when downscale factor is applied"""
            rel_part = filepath.relative_to(parent)
            base_part = parent.parent / (str(parent.name) + f"_{self._downscale_factor}")
            return base_part / rel_part

        filepath = next(iter(image_filenames))
        if self._downscale_factor is None:
            if self.config.downscale_factor is None:
                test_img = Image.open(filepath)
                w, h = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) <= MAX_AUTO_RESOLUTION:
                        break
                    df += 1

                self._downscale_factor = 2**df
                CONSOLE.log(f"Using image downscale factor of {self._downscale_factor}")
            else:
                self._downscale_factor = self.config.downscale_factor
            if self._downscale_factor > 1 and not all(
                get_fname(self.config.data / self.config.images_path, fp).parent.exists() for fp in image_filenames
            ):
                # Downscaled images not found
                # Ask if user wants to downscale the images automatically here
                CONSOLE.print(
                    f"[bold red]Downscaled images do not exist for factor of {self._downscale_factor}.[/bold red]"
                )
                if Confirm.ask(
                    f"\nWould you like to downscale the images using '{self.config.downscale_rounding_mode}' rounding mode now?",
                    default=False,
                    console=CONSOLE,
                ):
                    # Install the method
                    self._downscale_images(
                        image_filenames,
                        partial(get_fname, self.config.data / self.config.images_path),
                        self._downscale_factor,
                        self.config.downscale_rounding_mode,
                        nearest_neighbor=False,
                    )
                    if len(mask_filenames) > 0:
                        assert self.config.masks_path is not None
                        self._downscale_images(
                            mask_filenames,
                            partial(get_fname, self.config.data / self.config.masks_path),
                            self._downscale_factor,
                            self.config.downscale_rounding_mode,
                            nearest_neighbor=True,
                        )
                    if len(depth_filenames) > 0:
                        self._downscale_numpy(
                            depth_filenames,
                            partial(get_fname, self.config.data / "mono_depth"),
                            self._downscale_factor,
                            self.config.downscale_rounding_mode,
                            nearest_neighbor=True,
                        )
                    if len(normal_filenames) > 0:
                        self._downscale_images(
                            normal_filenames,
                            partial(get_fname, self.normal_save_dir),
                            self._downscale_factor,
                            self.config.downscale_rounding_mode,
                            nearest_neighbor=True,
                        )
                else:
                    sys.exit(1)

        # Return transformed filenames
        if self._downscale_factor > 1:
            image_filenames = [get_fname(self.config.data / self.config.images_path, fp) for fp in image_filenames]
            if len(mask_filenames) > 0:
                assert self.config.masks_path is not None
                mask_filenames = [get_fname(self.config.data / self.config.masks_path, fp) for fp in mask_filenames]
            if len(depth_filenames) > 0:
                depth_filenames = [get_fname(self.config.data / "mono_depth", fp) for fp in depth_filenames]
            if len(normal_filenames) > 0:
                normal_filenames = [get_fname(self.normal_save_dir, fp) for fp in normal_filenames]
        assert isinstance(self._downscale_factor, int)
        return image_filenames, mask_filenames, depth_filenames, normal_filenames, self._downscale_factor


    def _load_points3D_normals(self, points_3d):
        transform_matrix = torch.eye(4, dtype=torch.float, device="cpu")[:3, :4]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d.cpu().numpy())
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


CoolerMapDataParserSpecification = DataParserSpecification(
    config=CoolerMapDataParserConfig(),
    description="CoolerMap: modified version of Colmap dataparser",
)
