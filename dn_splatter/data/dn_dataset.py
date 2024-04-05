"""
InputDataset that loads various Depth formats and Normal formats
"""

from pathlib import Path
from typing import Dict, Literal, Optional, Union

import numpy as np
import torch
from dn_splatter.utils.camera_utils import euclidean_to_z_depth
from PIL import Image

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_depth_image_from_path
from nerfstudio.utils.rich_utils import CONSOLE


class GDataset(InputDataset):
    """Dataset that loads various depth formats and normal formats"""

    def __init__(
        self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0
    ):
        """
        Args:
            dataparser_outputs: dataparser outputs that should have depth/normal data.
            scale_factor: Scale factor for depth data.
            depth_mode: What depth data to load if more than one present.
            normal_format: format that the normal data is stored in.
        """
        super().__init__(dataparser_outputs, scale_factor)

        # Configs
        if "load_depths" in self.metadata:
            self.load_depths = self.metadata["load_depths"]
        else:
            self.load_depths = True

        if "load_normals" in self.metadata:
            self.load_normals = self.metadata["load_normals"]
        else:
            self.load_normals = False

        if "depth_mode" in self.metadata:
            self.depth_mode = self.metadata["depth_mode"]
            assert self.depth_mode in ["sensor", "mono", "all", "none"]
        else:
            self.depth_mode = "sensor"

        if "transform" in self.metadata:
            self.transform = self.metadata["transform"]

        if "normal_format" in self.metadata:
            self.normal_format = self.metadata["normal_format"]
        else:
            self.normal_format = "opengl"

        if "normal_frame" in self.metadata:
            self.normal_frame = self.metadata["normal_frame"]
            assert self.normal_frame in ["camera_frame", "world_frame"]
        else:
            self.normal_frame = "camera_frame"

        if self.normal_frame == "world_frame":
            assert "camera_to_worlds" in self.metadata
            self.camera_to_worlds = self.metadata["camera_to_worlds"]
        else:
            self.camera_to_worlds = None
        if "is_euclidean_depth" in self.metadata:
            self.is_euclidean_depth = self.metadata["is_euclidean_depth"]
        else:
            self.is_euclidean_depth = False

        # load depths
        if self.load_depths:
            self.depth_unit_scale_factor = self.metadata["depth_unit_scale_factor"]
            if "sensor_depth_filenames" in self.metadata:
                self.sensor_depth_filenames = self.metadata["sensor_depth_filenames"]
            if "depth_filenames" in self.metadata:
                self.sensor_depth_filenames = self.metadata["depth_filenames"]

            self.mono_depth_filenames = None
            if self.depth_mode in ["mono", "all"]:
                if (
                    "mono_depth_filenames" in dataparser_outputs.metadata.keys()
                    and dataparser_outputs.metadata["mono_depth_filenames"] is not None
                ):
                    self.mono_depth_filenames = self.metadata["mono_depth_filenames"]

                else:
                    CONSOLE.print(
                        "[bold yellow] Could not find mono depth filenames in dataparser. Quitting!"
                    )
                    quit()
        # load normals
        if self.load_normals and (
            "normal_filenames" not in dataparser_outputs.metadata.keys()
            or dataparser_outputs.metadata["normal_filenames"] is None
        ):
            CONSOLE.print(
                "[bold yellow] No normal data found, although use normals has been set to True in datamanager! Quitting!"
            )
            quit()

        if self.load_normals:
            assert "normal_filenames" in self.metadata
            self.normal_filenames = self.metadata["normal_filenames"]

    def get_metadata(self, data: Dict) -> Dict:
        metadata = {}
        depth_data = {}
        normal_data = {}
        if self.load_depths:
            # try to load depth data
            height = int(self._dataparser_outputs.cameras.height[data["image_idx"]])
            width = int(self._dataparser_outputs.cameras.width[data["image_idx"]])
            # Scale depth images to meter units and also by scaling applied to cameras
            scale_factor = (
                self.depth_unit_scale_factor * self._dataparser_outputs.dataparser_scale
            )
            if self.depth_mode in ["sensor", "all"]:
                filepath = self.sensor_depth_filenames[data["image_idx"]]
                depth_image = get_depth_image_from_path(
                    filepath=filepath,
                    height=height,
                    width=width,
                    scale_factor=scale_factor,
                )
                if self.is_euclidean_depth:
                    fx = self._dataparser_outputs.cameras.fx[data["image_idx"]].item()
                    fy = self._dataparser_outputs.cameras.fy[data["image_idx"]].item()
                    cx = int(
                        self._dataparser_outputs.cameras.cx[data["image_idx"]].item()
                    )
                    cy = int(
                        self._dataparser_outputs.cameras.cy[data["image_idx"]].item()
                    )
                    depth_image = euclidean_to_z_depth(
                        depth_image, fx, fy, cx, cy, (width, height), depth_image.device
                    )

                depth_data.update({"sensor_depth": depth_image})

            if self.depth_mode in ["mono", "all"]:
                assert self.mono_depth_filenames is not None

                filepath = self.mono_depth_filenames[data["image_idx"]]
                mono_image = get_depth_image_from_path(
                    filepath=Path(filepath),
                    height=height,
                    width=width,
                    scale_factor=scale_factor,
                )
                depth_data.update({"mono_depth": mono_image})

        if self.load_normals:
            assert self.normal_filenames is not None
            filepath = self.normal_filenames[data["image_idx"]]
            camtoworld = None
            if self.normal_frame == "world_frame":
                camtoworld = self.camera_to_worlds[data["image_idx"]]
            normal_image = self.get_normal_image_from_path(
                Path(filepath),
                normal_format=self.normal_format,
                normal_frame=self.normal_frame,
                c2w=camtoworld,
            )
            normal_data.update({"normal": normal_image})
        metadata.update(depth_data)
        metadata.update(normal_data)
        return metadata

    def _find_transform(self, image_path: Path) -> Union[Path, None]:
        while image_path.parent != image_path:
            transform_path = image_path.parent / "transforms.json"
            if transform_path.exists():
                return transform_path
            image_path = image_path.parent
        return None

    def get_normal_image_from_path(
        self,
        path,
        normal_format: Literal["opencv", "opengl"],
        normal_frame: Literal["camera_frame", "world_frame"],
        c2w: Optional[None] = None,
    ):
        """Helper function to load normal data

        Args:
            path: path to normal file
            normal_format: which format "opencv" or "opengl" the normal data is stored in. We convert automatically to opencv
            c2w: optional c2w transform if normals should be in world frame
        """
        if path.suffix == ".png":
            normal_map = np.array(Image.open(path), dtype="uint8")[..., :3]
        else:
            # TODO: check if this is correct for .npy data
            normal_map = np.load(path)
            normal_map = normal_map.transpose(1, 2, 0)
            if normal_map.min() < 0:
                normal_map = (normal_map + 1) / 2

        normal_map = torch.from_numpy(normal_map.astype("float32") / 255.0).float()

        if normal_format == "opengl" and normal_frame == "camera_frame":
            # convert normal map from opengl to opencv
            h, w, _ = normal_map.shape
            normal_map = normal_map.view(-1, 3)
            normal_map = 2 * normal_map - 1
            normal_map = normal_map @ torch.diag(
                torch.tensor([1, -1, -1], device=normal_map.device, dtype=torch.float)
            )
            normal_map = normal_map.view(h, w, 3)
            if normal_map.min() < 0:
                normal_map = (normal_map + 1) / 2

        if normal_frame == "world_frame":
            # convert normals to world coordinates
            # Used for SDFStudio models
            # import os
            # from dn_splatter.utils.utils import save_img
            # save_img((normal_map), os.getcwd()+"/test_before_conversion.png")
            assert c2w is not None
            normal_map = 2 * normal_map - 1
            h, w, _ = normal_map.shape
            rot = c2w[:3, :3]
            normal_map = normal_map.permute(2, 0, 1).reshape(3, -1)
            normal_map = torch.nn.functional.normalize(normal_map, p=2, dim=0)
            normal_map = rot @ normal_map
            normal_map = normal_map.permute(1, 0).reshape(h, w, 3)
            # print(rot)
            # save_img((normal_map + 1) / 2, os.getcwd() + "/test_world_normal.png")

            if self.transform is not None:
                h, w, _ = normal_map.shape
                normal_map = self.transform[:3, :3] @ normal_map.reshape(-1, 3).permute(
                    1, 0
                )
                normal_map = normal_map.permute(1, 0).reshape(h, w, 3)

        if self._dataparser_outputs.mask_filenames is not None:
            # TODO: figure out what to do with normal data if masks present ...
            pass
        return normal_map
