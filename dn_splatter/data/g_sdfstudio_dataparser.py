"""DTU dataparser

Download data with:
    python dn_splatter/data/download_scripts/dtu_download.py
"""


from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import cv2
import numpy as np
import open3d as o3d
import torch
from dn_splatter.utils.utils import image_path_to_tensor
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
class GSDFStudioDataParserConfig(DataParserConfig):
    """Scene dataset parser config"""

    _target: Type = field(default_factory=lambda: GSDFStudioDataparser)
    """target class to instantiate"""
    data: Path = Path("datasets/dtu/")
    """Root directory specifying location of DTU dataset."""
    load_for_sdfstudio: bool = False
    """If true, the output will be in sdfstudio format. Usable with gneusfacto. If false, output will be for dn_splatter model and other nerfstudio models."""
    load_depths: bool = True
    """Whether to load depth maps"""
    load_normals: bool = True
    """Set to true to use load normal maps."""
    depth_mode: Literal["sensor", "mono", "all"] = "sensor"
    """Which depth data to load"""
    is_euclidean_depth: bool = False
    """Whether input depth maps are Euclidean distances (or z-distances)."""
    load_pcd_normals: bool = True
    """Whether to load pcd normals for normal initialisation"""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "none"
    """The method to use for orientation."""
    auto_scale_poses: bool = False
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    load_every: int = 1  # 50
    """load every n'th frame from the dense trajectory"""
    skip_every_for_val_split: int = 1000
    """sub sampling validation images"""
    auto_orient: bool = False
    """automatically orient the scene such that the up direction is the same as the viewer's up direction"""
    num_sfm_points: int = 10_000
    """Num points to load"""
    use_masks: bool = False
    """Whether to load masks or not. Disable when using Neus/Neurangelo/MonoSDF models"""


@dataclass
class GSDFStudioDataparser(DataParser):
    config: GSDFStudioDataParserConfig

    def _generate_dataparser_outputs(
        self, split="train"
    ):  # pylint: disable=unused-argument,too-many-statements
        self.input_folder = self.config.data
        meta = load_from_json(self.input_folder / "meta_data.json")

        all_indices = list(range(len(meta["frames"])))
        indices = all_indices[:: self.config.load_every]
        assert self.config.skip_every_for_val_split >= 1
        if split != "train":
            # eval split
            indices = indices[:: self.config.skip_every_for_val_split]
        elif split == "train":
            # train split
            eval_indices = indices[:: self.config.skip_every_for_val_split]
            indices = [i for i in indices if i not in eval_indices]

        self.height = meta["height"]
        self.width = meta["width"]
        mono_depth_filenames = []
        sensor_depth_filenames = []
        image_filenames = []
        normal_filenames = []
        mask_filenames = []
        sfm_points = []
        fx, fy, cx, cy, camera_to_worlds = [], [], [], [], []
        for i, frame in enumerate(meta["frames"]):
            if i not in indices:
                continue
            # load rgb
            image_filenames.append(self.input_folder / frame["rgb_path"])
            # load mono depth
            if "mono_depth_path" in frame:
                mono_depth_filenames.append(
                    self.input_folder / frame["mono_depth_path"]
                )
            # load sensor depth
            if "sensor_depth_path" in frame:
                sensor_depth_filenames.append(
                    self.input_folder / frame["sensor_depth_path"]
                )
            # load mono normal
            if "mono_normal_path" in frame:
                normal_filenames.append(
                    self.input_folder
                    / Path(frame["mono_normal_path"]).with_suffix(".png")
                )
            elif "normal_from_depth_path" in frame:
                normal_filenames.append(
                    self.input_folder
                    / Path(frame["normal_from_depth_path"]).with_suffix(".png")
                )
            # process and load mask
            if "foreground_mask" in frame:
                mask_fname = self.input_folder / Path(
                    frame["foreground_mask"]
                ).with_suffix(".jpeg")
                if not Path(mask_fname).exists():
                    # sdfstudio provides masks with 3 channels. Convert to 1 channel for nerfstudio compatibility
                    mask = image_path_to_tensor(
                        self.input_folder / Path(frame["foreground_mask"])
                    )
                    mask = mask.bool()[..., 0][..., None]
                    mask = mask.numpy()
                    mask = mask.astype(np.uint8)
                    cv2.imwrite(  # type: ignore
                        str(
                            self.input_folder
                            / Path(frame["foreground_mask"]).with_suffix(".jpeg")
                        ),
                        mask,
                    )
                mask_filenames.append(mask_fname)
            # sfm points
            if "sfm_sparse_points_view" in frame:
                sfm_points.append(
                    torch.from_numpy(
                        np.loadtxt(self.input_folder / frame["sfm_sparse_points_view"])
                    ).float()
                )

            # intrinsics
            intrinsics = torch.tensor(frame["intrinsics"])
            fx.append(intrinsics[0, 0])
            fy.append(intrinsics[1, 1])
            cx.append(intrinsics[0, 2])
            cy.append(intrinsics[1, 2])
            camera_to_worlds.append(torch.tensor(frame["camtoworld"]).float())
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        fx = torch.stack(fx)
        fy = torch.stack(fy)
        cx = torch.stack(cx)
        cy = torch.stack(cy)
        self.image_filenames = image_filenames
        self.mono_depth_filenames = mono_depth_filenames
        self.camera_to_worlds = torch.stack(camera_to_worlds)
        c2w_colmap = torch.stack(camera_to_worlds)
        # Convert from COLMAP's/OPENCV's camera coordinate system to nerfstudio
        self.camera_to_worlds[:, 0:3, 1:3] *= -1

        if self.config.auto_orient:
            (
                self.camera_to_worlds,
                transform,
            ) = camera_utils.auto_orient_and_center_poses(
                self.camera_to_worlds, method="up", center_method="none"
            )
        else:
            transform = torch.eye(4)[:3, :4]

        # scene box from meta data
        meta_scene_box = meta["scene_box"]
        aabb = torch.tensor(meta_scene_box["aabb"], dtype=torch.float32)
        scene_box = SceneBox(aabb=aabb)

        distort = camera_utils.get_distortion_params()

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distort,
            height=self.height,
            width=self.width,
            camera_to_worlds=self.camera_to_worlds[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )

        metadata = {}
        metadata.update({"load_depths": self.config.load_depths})
        metadata.update({"depth_mode": self.config.depth_mode})
        metadata.update({"normal_filenames": normal_filenames})
        metadata.update(
            {
                "mono_depth_filenames": mono_depth_filenames
                if len(mono_depth_filenames) > 0
                else None
            }
        )
        metadata.update(
            {
                "sensor_depth_filenames": mono_depth_filenames
                if len(mono_depth_filenames) > 0
                else None
            }
        )
        metadata.update({"is_euclidean_depth": self.config.is_euclidean_depth})
        metadata.update({"depth_unit_scale_factor": 1})  # TODO: check this is correct
        metadata.update({"load_normals": self.config.load_normals})
        if len(sfm_points) > 0:
            sfm_points = torch.cat(sfm_points)
            points3D = (
                torch.cat((sfm_points, torch.ones_like(sfm_points[..., :1])), -1)
                @ transform.T
            )
            metadata.update(
                {"points3D_xyz": points3D, "points3D_rgb": torch.rand_like(points3D)}
            )

        # save global transform matrix after orientation changes
        metadata.update({"transform": transform})
        metadata.update(
            {"camera_to_worlds": c2w_colmap if len(c2w_colmap) > 0 else None}
        )

        # Safety check to make sure we load correct normal conventions
        if self.config.load_for_sdfstudio:
            normal_format = "dsine"
            normal_frame = "world_frame"
        else:
            normal_format = "opengl"
            normal_frame = "camera_frame"
        metadata.update({"normal_format": normal_format})
        metadata.update({"normal_frame": normal_frame})
        # TODO: check how to properly deal with masks
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            mask_filenames=mask_filenames if self.config.use_masks else None,
            scene_box=scene_box,
            metadata=metadata,
        )
        return dataparser_outputs

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


GSDFStudioDataParserSpecification = DataParserSpecification(
    config=GSDFStudioDataParserConfig(),
    description="support any dataset in sdfstudio format",
)

if __name__ == "__main__":
    parser = GSDFStudioDataparser(
        GSDFStudioDataParserConfig
    )._generate_dataparser_outputs()
