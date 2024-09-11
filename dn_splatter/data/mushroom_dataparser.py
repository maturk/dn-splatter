"""Data parser for Mushroom dataset

for more info about the dataset, visit: https://github.com/TUTvision/MuSHRoom

Conventions:
    - Test and evaluation splits for the dataset are defined by the eval_mode flag.  The options are "Within" protocol which trains on long sequences but tests on short sequences. "With" protocol trains on long sequence and evaluates on a subset of long sequence images. "Both" uses both protocols.
    - Initial pointcloud: uses either the pointcloud genereated from iphone or kinect sequences for gaussian initialisation.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Type

import numpy as np
import open3d as o3d
import torch
from dn_splatter.data.download_scripts.mushroom_download import download_mushroom
from dn_splatter.data.mushroom_utils.pointcloud_utils import (
    generate_iPhone_pointcloud_within_sequence,
    generate_kinect_pointcloud_within_sequence,
)
from dn_splatter.data.mushroom_utils.reference_depth_download import (
    download_reference_depth,
)
from dn_splatter.scripts.depth_from_pretrain import depth_from_pretrain
from dn_splatter.scripts.depth_to_normal import DepthToNormal
from dn_splatter.scripts.normals_from_pretrain import (
    NormalsFromPretrained,
    normals_from_depths,
)
from natsort import natsorted
from PIL import Image

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
from nerfstudio.utils.rich_utils import CONSOLE, status
from nerfstudio.utils.scripts import run_command

MAX_AUTO_RESOLUTION = 1600


@dataclass
class MushroomDataParserConfig(DataParserConfig):
    """Mushroom dataset config"""

    _target: Type = field(default_factory=lambda: MushroomDataParser)
    """target class to instantiate"""

    # Mushroom configs
    mode: Literal["iphone", "kinect"] = "iphone"
    """load either the iphone or kinect mushroom data."""
    eval_mode: Literal["with", "within", "all"] = "all"
    """evaluation protocol. "Within" protocol trains on long sequences but tests on short sequences. "With" protocol trains on long sequence and evaluates on a subset of long sequence images. "Both" uses both protocols."""
    iphone_ply_name: Path = Path("iphone_pointcloud.ply")
    """Path to the polycam reconstruction directory relative to the data path."""
    kinect_ply_path: Path = Path("kinect_pointcloud.ply")
    """Path to kinect data point cloud relative to the data path."""
    load_3D_points: bool = True
    """Whether to load the 3D points from the polycam reconstruction."""
    use_faro_scanner_depths: bool = False
    """Whether to use faro scanner depth files in dataparser"""
    use_faro_scanner_pd: bool = False
    """Whether to use faro scanner point data in dataparser"""
    depth_mode: Literal["sensor", "all", "mono"] = "sensor"
    """Which depth data to load"""
    mono_pretrain: Literal["zoe"] = "zoe"
    """Which mono depth pretrain model to use."""
    is_euclidean_depth: bool = False
    """Whether input depth maps are Euclidean distances (or z-distances)."""
    load_depths: bool = True
    """Whether to load depth maps"""
    load_depth_confidence_masks: bool = False
    """Whether to load depth confidence masks"""
    load_normals: bool = True
    """Set to true to load normal maps"""
    normal_format: Literal["opencv", "opengl"] = "opengl"
    """Which format the normal maps in camera frame are saved in."""
    normals_from: Literal["depth", "pretrained"] = "pretrained"
    """If no ground truth normals, generate normals either from sensor depths or from pretrained model."""
    load_pcd_normals: bool = True
    """Whether to load pcd normals for normal initialisation"""
    create_pc_from_colmap: bool = False
    """Whether to create pointclouds from colmap for the mushroom data."""
    num_init_points: int = 1_000_000
    """Number of points in initial seed pointcloud. Does not apply to Colmap generated points."""

    # general configs
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "none"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "none"
    """The method to use to center the poses."""
    auto_scale_poses: bool = False  # no scaling, lets keep metric scale
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""


class MushroomDataParser(DataParser):
    """MUSHROOM DatasetParser."""

    config: MushroomDataParserConfig

    def __init__(self, config: MushroomDataParserConfig):
        super().__init__(config)
        self.config = config
        self._downscale_factor = None

    def _generate_dataparser_outputs(self, split: str = "train", **kwargs):
        assert (
            self.config.data.exists()
        ), f"Data directory {self.config.data} does not exist."

        long_data_dir = self.config.data / self.config.mode / "long_capture"
        short_data_dir = self.config.data / self.config.mode / "short_capture"

        if self.config.use_faro_scanner_depths:
            if not (long_data_dir / "reference_depth").exists():
                download_reference_depth()
        else:
            long_meta = load_from_json(long_data_dir / "transformations_colmap.json")
            short_meta = load_from_json(short_data_dir / "transformations_colmap.json")

        fx_fixed = "fl_x" in long_meta
        fy_fixed = "fl_y" in long_meta
        cx_fixed = "cx" in long_meta
        cy_fixed = "cy" in long_meta
        height_fixed = "h" in long_meta
        width_fixed = "w" in long_meta
        distort_fixed = False

        for distort_key in ["k1", "k2", "k3", "p1", "p2"]:
            if distort_key in long_meta:
                distort_fixed = True
                break

        # sort the frames by fname
        long_fnames = []
        for frame in long_meta["frames"]:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, long_data_dir)
            long_fnames.append(fname)

        if self.config.mode == "iphone":
            inds = np.argsort(long_fnames)
        else:
            inds = np.argsort([int(fname.stem) for fname in long_fnames])
        long_frames = [long_meta["frames"][ind] for ind in inds]

        short_fnames = []
        for frame in short_meta["frames"]:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, short_data_dir)
            short_fnames.append(fname)

        if self.config.mode == "iphone":
            inds = np.argsort(short_fnames)
        else:
            inds = np.argsort([int(fname.stem) for fname in short_fnames])
        short_frames = [short_meta["frames"][ind] for ind in inds]

        if self.config.load_depth_confidence_masks:
            long_data_dir = self.config.data / self.config.mode / "long_capture"
            short_data_dir = self.config.data / self.config.mode / "short_capture"
            if not (long_data_dir / "depth_normals_mask").exists():
                CONSOLE.log(
                    f"[yellow]Could not find depth confidence masks, trying to generate them into {str(long_data_dir / 'depth_normals_mask')}"
                )
                DepthToNormal(data_dir=long_data_dir).main()
            if not (short_data_dir / "depth_normals_mask").exists():
                CONSOLE.log(
                    f"[yellow]Could not find depth confidence masks, trying to generate them into {str(short_data_dir / 'depth_normals_mask')}"
                )
                DepthToNormal(data_dir=short_data_dir).main()
        (
            return_long_filenames,
            long_poses,
            long_fx,
            long_fy,
            long_cx,
            long_cy,
            long_height,
            long_width,
            long_distort,
        ) = self.get_ele_from_meta(
            long_frames,
            long_data_dir,
            fx_fixed,
            fy_fixed,
            cx_fixed,
            cy_fixed,
            height_fixed,
            width_fixed,
            distort_fixed,
            self.config.use_faro_scanner_depths,
            self.config.load_depth_confidence_masks,
        )
        (
            return_short_filenames,
            short_poses,
            short_fx,
            short_fy,
            short_cx,
            short_cy,
            short_height,
            short_width,
            short_distort,
        ) = self.get_ele_from_meta(
            short_frames,
            short_data_dir,
            fx_fixed,
            fy_fixed,
            cx_fixed,
            cy_fixed,
            height_fixed,
            width_fixed,
            distort_fixed,
            self.config.use_faro_scanner_depths,
            self.config.load_depth_confidence_masks,
        )
        long_filenames = return_long_filenames["image"]
        short_filenames = return_short_filenames["image"]
        long_depth_filenames = return_long_filenames["depth"]
        short_depth_filenames = return_short_filenames["depth"]
        long_mask_filenames = return_long_filenames["mask"]
        short_mask_filenames = return_short_filenames["mask"]

        image_filenames = long_filenames + short_filenames
        mask_filenames = long_mask_filenames + short_mask_filenames
        depth_filenames = long_depth_filenames + short_depth_filenames
        if self.config.load_depth_confidence_masks:
            confidence_filenames = (
                return_long_filenames["confidence"]
                + return_short_filenames["confidence"]
            )
        else:
            confidence_filenames = []
        poses = long_poses + short_poses
        fx = long_fx + short_fx
        fy = long_fy + short_fy
        cx = long_cx + short_cx
        cy = long_cy + short_cy
        height = long_height + short_height
        width = long_width + short_width
        distort = long_distort + short_distort

        poses = np.array(poses)

        assert len(mask_filenames) == 0 or (
            len(mask_filenames) == len(image_filenames)
        ), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """

        assert len(depth_filenames) == 0 or (
            len(depth_filenames) == len(image_filenames)
        ), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(confidence_filenames) == 0 or (
            len(confidence_filenames) == len(image_filenames)
        )

        # Mushroom eval images
        eval_image_txt_path = Path(long_data_dir / "test.txt")

        test_filenames = []
        if eval_image_txt_path.exists():
            with open(eval_image_txt_path) as fid:
                while True:
                    img_name = fid.readline()
                    if not img_name:
                        break
                    img_name = img_name.strip()
                    if self.config.mode == "iphone":
                        file_name = "images/" + img_name + ".jpg"
                    else:
                        file_name = "images/" + img_name + ".png"

                    test_filenames.append(
                        self._get_fname(
                            file_name,
                            data_dir=long_data_dir,
                            downsample_folder_prefix="images",
                        )
                    )
        else:
            CONSOLE.log(
                f"[yellow]Path to test images at {eval_image_txt_path} does not exist. Using zero test images."
            )

        # mushroom dataset train and eval split, load both eval images for eval with and within protocol

        i_train, i_eval = self.mushroom_get_train_eval_split_filename(
            long_filenames, test_filenames
        )

        if self.config.eval_mode == "within":
            i_eval = i_eval
        elif self.config.eval_mode == "with":
            i_eval = np.arange(len(long_filenames), len(image_filenames))
        elif self.config.eval_mode == "all":
            i_eval = np.concatenate(
                [i_eval, np.arange(len(long_filenames), len(image_filenames))]
            )
        else:
            raise ValueError(f"Unknown eval mode {self.config.eval_mode}")

        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        if self.config.mode == "iphone":
            if "orientation_override" in long_meta:
                orientation_method = long_meta["orientation_override"]
                CONSOLE.log(
                    f"[yellow] Dataset is overriding orientation method to {orientation_method}"
                )
            else:
                orientation_method = self.config.orientation_method
        else:
            orientation_method = self.config.orientation_method

        poses = torch.from_numpy(poses.astype(np.float32))

        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses, method=orientation_method, center_method=self.config.center_method
        )

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

        if self.config.load_depth_confidence_masks:
            confidence_filenames = (
                [confidence_filenames[i] for i in indices]
                if len(confidence_filenames) > 0
                else []
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

        # Mushroom dataset contains only perspective camera types
        camera_type = CameraType.PERSPECTIVE

        fx = (
            float(long_meta["fl_x"])
            if fx_fixed
            else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        )
        fy = (
            float(long_meta["fl_y"])
            if fy_fixed
            else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        )
        cx = (
            float(long_meta["cx"])
            if cx_fixed
            else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        )
        cy = (
            float(long_meta["cy"])
            if cy_fixed
            else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        )
        height = (
            int(long_meta["h"])
            if height_fixed
            else torch.tensor(height, dtype=torch.int32)[idx_tensor]
        )
        width = (
            int(long_meta["w"])
            if width_fixed
            else torch.tensor(width, dtype=torch.int32)[idx_tensor]
        )
        if distort_fixed:
            distortion_params = camera_utils.get_distortion_params(
                k1=float(long_meta["k1"]) if "k1" in long_meta else 0.0,
                k2=float(long_meta["k2"]) if "k2" in long_meta else 0.0,
                k3=float(long_meta["k3"]) if "k3" in long_meta else 0.0,
                k4=float(long_meta["k4"]) if "k4" in long_meta else 0.0,
                p1=float(long_meta["p1"]) if "p1" in long_meta else 0.0,
                p2=float(long_meta["p2"]) if "p2" in long_meta else 0.0,
            )
        else:
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

        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        if self.config.mode == "iphone":
            if "applied_transform" in long_meta:
                applied_transform = torch.tensor(
                    long_meta["applied_transform"], dtype=transform_matrix.dtype
                )
                transform_matrix = transform_matrix @ torch.cat(
                    [
                        applied_transform,
                        torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype),
                    ],
                    0,
                )
            if "applied_scale" in long_meta:
                applied_scale = float(long_meta["applied_scale"])
                scale_factor *= applied_scale
        # else:
        # change to 4x3
        # transform_matrix = torch.cat([transform_matrix, torch.zeros(3, 1)], dim=1)

        metadata = {}

        if split == "train":
            if self.config.use_faro_scanner_pd:
                metadata.update(
                    self._load_faro_scanner_point_data(
                        self.config.data,
                        transform_matrix,
                        scale_factor,
                        self.config.mode,
                    )
                )
            else:
                if self.config.mode == "iphone":
                    # load iphone polycam point data
                    if not self.config.create_pc_from_colmap:
                        iphone_ply_file_path = (
                            long_data_dir / self.config.iphone_ply_name
                        )
                        if not iphone_ply_file_path.exists():
                            CONSOLE.log(
                                f"[bold yellow] could not find polycam pointcloud path {iphone_ply_file_path}. Trying to reconstruct it from available data..."
                            )
                            generate_iPhone_pointcloud_within_sequence(long_data_dir)

                    else:
                        from nerfstudio.process_data.colmap_utils import (
                            create_ply_from_colmap,
                        )

                        ply_filename = "sparse_pc.ply"
                        applied_transform = np.eye(4)[:3, :]
                        applied_transform = applied_transform[np.array([0, 2, 1]), :]
                        applied_transform[2, :] *= -1
                        create_ply_from_colmap(
                            filename=ply_filename,
                            recon_dir=long_data_dir / "sparse/0",
                            output_dir=long_data_dir,
                            applied_transform=torch.tensor(
                                applied_transform,
                                dtype=torch.float32,
                            ),
                        )
                        iphone_ply_file_path = long_data_dir / ply_filename
                    # load iphone point data
                    metadata.update(
                        self._load_3D_points(
                            iphone_ply_file_path, transform_matrix, scale_factor
                        )
                    )
                    if metadata["points3D_xyz"].shape[0] != self.config.num_init_points:
                        CONSOLE.log(
                            f"[bold yellow] Found pointcloud with {metadata['points3D_xyz'].shape[0]} number of points, regenerating with with {self.config.num_init_points} points..."
                        )
                        generate_iPhone_pointcloud_within_sequence(
                            long_data_dir, num_points=self.config.num_init_points
                        )

                        metadata.update(
                            self._load_3D_points(
                                iphone_ply_file_path, transform_matrix, scale_factor
                            )
                        )

                else:
                    kinect_pointcloud_path = long_data_dir / self.config.kinect_ply_path
                    if not kinect_pointcloud_path.exists():
                        CONSOLE.log(
                            f"[bold yellow] could not find kinect pointcloud path {kinect_pointcloud_path}. Trying to reconstruct it from available data..."
                        )
                        PointCloud_path = long_data_dir / "PointCloud"
                        if PointCloud_path.exists():
                            generate_kinect_pointcloud_within_sequence(long_data_dir)
                        else:
                            CONSOLE.log(
                                "[bold red] could not find pointcloud data. Exiting..."
                            )
                            quit()
                    # load kinect point data
                    metadata.update(
                        self._load_3D_points(
                            kinect_pointcloud_path, transform_matrix, scale_factor
                        )
                    )
                    if metadata["points3D_xyz"].shape[0] != self.config.num_init_points:
                        CONSOLE.log(
                            f"[bold yellow] Found pointcloud with {metadata['points3D_xyz'].shape[0]} number of points, regenerating with with {self.config.num_init_points} points..."
                        )
                        generate_kinect_pointcloud_within_sequence(
                            long_data_dir, num_points=self.config.num_init_points
                        )

                        metadata.update(
                            self._load_3D_points(
                                iphone_ply_file_path, transform_matrix, scale_factor
                            )
                        )

            if self.config.load_pcd_normals:
                metadata.update(
                    self._load_points3D_normals(
                        points=metadata["points3D_xyz"],
                        colors=metadata["points3D_rgb"],
                        transform_matrix=transform_matrix,
                    )
                )

        # process depths
        if self.config.depth_mode == "all" or self.config.depth_mode == "mono":
            self.depth_save_dir = long_data_dir / Path("mono_depth")
            if (
                not self.depth_save_dir.exists()
                or len(os.listdir(self.depth_save_dir)) == 0
            ):
                depth_from_pretrain(
                    input_folder=long_data_dir,
                    img_dir_name="images",
                    path_to_transforms=None,
                    save_path=self.depth_save_dir,
                    create_new_transforms=False,
                    return_mode="mono-aligned",
                )
            long_name_list = [frame.name for frame in long_depth_filenames]
            long_mono_depth_filenames = self.get_depth_filepaths()
            long_mono_depth_filenames = [
                Path(frame)
                for frame in long_mono_depth_filenames
                if Path(frame).stem + ".png" in long_name_list
            ]

            self.depth_save_dir = short_data_dir / Path("mono_depth")
            if (
                not self.depth_save_dir.exists()
                or len(os.listdir(self.depth_save_dir)) == 0
            ):
                depth_from_pretrain(
                    input_folder=short_data_dir,
                    img_dir_name="images",
                    path_to_transforms=None,
                    save_path=self.depth_save_dir,
                    create_new_transforms=False,
                    return_mode="mono-aligned",
                )
            short_name_list = [frame.name for frame in short_depth_filenames]
            short_mono_depth_filenames = self.get_depth_filepaths()
            short_mono_depth_filenames = [
                Path(frame)
                for frame in short_mono_depth_filenames
                if Path(frame).stem + ".png" in short_name_list
            ]

            mono_depth_filenames = (
                long_mono_depth_filenames + short_mono_depth_filenames
            )
            metadata.update(
                {
                    "mono_depth_filenames": (
                        [Path(mono_depth_filenames[idx]) for idx in indices]
                        if len(mono_depth_filenames) > 0
                        else None
                    )
                }
            )
        if self.config.depth_mode == "all" or self.config.depth_mode == "sensor":
            metadata.update(
                {
                    "sensor_depth_filenames": (
                        depth_filenames if len(depth_filenames) > 0 else None
                    ),
                }
            )

        # process normals
        if self.config.normals_from == "depth":
            self.long_normal_save_dir = long_data_dir / Path("normals_from_depth")
        else:
            self.long_normal_save_dir = long_data_dir / Path("normals_from_pretrain")

        if self.config.load_normals and (
            not self.long_normal_save_dir.exists()
            or len(os.listdir(self.long_normal_save_dir)) == 0
        ):
            CONSOLE.print(
                f"[bold yellow]Could not find normals, generating them into {str(self.long_normal_save_dir)}"
            )
            (self.long_normal_save_dir).mkdir(exist_ok=True, parents=True)

            if self.config.normals_from == "pretrained":
                NormalsFromPretrained(
                    data_dir=long_data_dir,
                    save_path=self.long_normal_save_dir,
                    force_images_dir=True,
                ).main()
            else:
                normals_from_depths(
                    path_to_transforms=long_data_dir / "transformations_colmap.json",
                    save_path=self.long_normal_save_dir,
                    normal_format=self.config.normal_format,
                )

        if self.config.normals_from == "depth":
            self.short_normal_save_dir = short_data_dir / Path("normals_from_depth")
        else:
            self.short_normal_save_dir = short_data_dir / Path("normals_from_pretrain")

        if self.config.load_normals and (
            not self.short_normal_save_dir.exists()
            or len(os.listdir(self.short_normal_save_dir)) == 0
        ):
            CONSOLE.print(
                f"[bold yellow]Could not find normals, generating them into {str(self.short_normal_save_dir)}"
            )
            (self.short_normal_save_dir).mkdir(exist_ok=True, parents=True)

            if self.config.normals_from == "pretrained":
                NormalsFromPretrained(
                    data_dir=short_data_dir,
                    transforms_name="transformations_colmap.json",
                    save_path=self.short_normal_save_dir,
                ).main()
            else:
                normals_from_depths(
                    path_to_transforms=short_data_dir / "transformations_colmap.json",
                    save_path=self.short_normal_save_dir,
                    normal_format=self.config.normal_format,
                )
        metadata.update({"depth_mode": self.config.depth_mode})
        metadata.update({"is_euclidean_depth": self.config.is_euclidean_depth})
        metadata.update({"load_depths": self.config.load_depths})

        # update metadata with normals
        if self.config.load_normals:
            self.normal_save_dir = self.long_normal_save_dir
            long_normal_filenames = self.get_normal_filepaths()
            long_normal_filenames = [
                Path(filename)
                for filename in long_normal_filenames
                if Path(filename).stem in [n.stem for n in long_filenames]
            ]
            self.normal_save_dir = self.short_normal_save_dir
            short_normal_filenames = self.get_normal_filepaths()
            short_normal_filenames = [
                Path(filename)
                for filename in short_normal_filenames
                if Path(filename).stem in [n.stem for n in short_filenames]
            ]
            assert len(long_normal_filenames) == len(long_filenames)
            assert len(short_normal_filenames) == len(short_filenames)
            normal_filenames = long_normal_filenames + short_normal_filenames
            metadata.update(
                {"normal_filenames": [normal_filenames[idx] for idx in indices]}
            )
            metadata.update({"normal_format": self.config.normal_format})

        metadata.update({"load_normals": self.config.load_normals})

        metadata.update({"load_confidence": self.config.load_depth_confidence_masks})

        if self.config.load_depth_confidence_masks:
            metadata.update({"confidence_filenames": confidence_filenames})

        self.scale_factor = scale_factor
        self.transform_matrix = transform_matrix

        self.scale_factor = scale_factor
        self.transform_matrix = transform_matrix

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

    def _load_points3D_normals(self, points, colors, transform_matrix: torch.Tensor):
        """Initialise gaussian scales/rots with normals predicted from pcd"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.numpy())
        pcd.colors = o3d.utility.Vector3dVector(colors.numpy())
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

    def get_normal_filepaths(self):
        """Helper to load normal paths"""
        return natsorted(glob.glob(f"{self.normal_save_dir}/*.png"))

    def get_depth_filepaths(self):
        """Helper to load depth paths"""
        files = os.listdir(self.depth_save_dir)
        if any(file.endswith(".npy") for file in files):
            extension = "npy"
        else:
            extension = "png"

        return natsorted(glob.glob(f"{self.depth_save_dir}/*.{extension}"))

    def _load_3D_points(
        self, ply_file_path: Path, transform_matrix: torch.Tensor, scale_factor: float
    ):
        """load pointcloud from ply file"""
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

    def _load_faro_scanner_point_data(
        self,
        file_path: Path,
        transform_matrix: torch.Tensor,
        scale_factor: float,
        mode: str,
        num_points: int = 1_000_000,
    ):
        """load pointcloud from ground truth faro ply file"""
        CONSOLE.print("[bold yellow]loading faro scanner point cloud")
        pcd_path = file_path / "gt_pd.ply"
        if not pcd_path.exists():
            download_mushroom(file_path.parts[1], sequence="faro")
        pcd = o3d.io.read_point_cloud(str(file_path / "gt_pd.ply"))
        align_matrix = np.array(
            json.load(open(file_path / f"icp_{mode}.json"))["gt_transformation"]
        ).reshape(4, 4)
        align_matrix = np.linalg.inv(np.array(align_matrix))
        pcd = pcd.transform(align_matrix)
        points = np.asarray(pcd.points, dtype=np.float32)
        index = np.random.choice(points.shape[0], num_points, replace=False)
        points = points[index]
        colors = np.asarray(pcd.colors, dtype=np.float32)
        colors = colors[index]
        points3D = torch.from_numpy(np.asarray(points, dtype=np.float32))
        points3D = (
            torch.cat((points3D, torch.ones_like(points3D[..., :1])), -1)
            @ transform_matrix.T
        )
        points3D *= scale_factor
        points3D_rgb = torch.from_numpy((np.asarray(colors) * 255).astype(np.uint8))
        out = {"points3D_xyz": points3D.to(torch.float32), "points3D_rgb": points3D_rgb}
        return out

    def _downscale_images(
        self, paths, get_fname, downscale_factor: int, nearest_neighbor: bool = False
    ):
        with status(msg="[bold yellow]Downscaling images...", spinner="growVertical"):
            assert downscale_factor > 1
            assert isinstance(downscale_factor, int)
            # Using %05d ffmpeg commands appears to be unreliable (skips images).
            for path in paths:
                nn_flag = "" if not nearest_neighbor else ":flags=neighbor"
                path_out = get_fname(path)
                path_out.parent.mkdir(parents=True, exist_ok=True)
                ffmpeg_cmd = [
                    f'ffmpeg -y -noautorotate -i "{path}" ',
                    f"-q:v 2 -vf scale=iw/{downscale_factor}:ih/{downscale_factor}{nn_flag} ",
                    f'"{path_out}"',
                ]
                ffmpeg_cmd = " ".join(ffmpeg_cmd)
                run_command(ffmpeg_cmd)

        CONSOLE.log("[bold green]:tada: Done downscaling images.")

    def _get_fname(
        self, filepath: Path, data_dir: Path, downsample_folder_prefix="images_"
    ) -> Path:
        """Get the filename of the image file.
        downsample_folder_prefix can be used to point to auxiliary image data, e.g. masks

        filepath: the base file name of the transformations.
        data_dir: the directory of the data that contains the transform file
        downsample_folder_prefix: prefix of the newly generated downsampled images
        """

        if self.config.downscale_factor is None:
            test_img = Image.open(data_dir / filepath)
            h, w = test_img.size
            max_res = max(h, w)
            df = 0
            while True:
                if (max_res / 2 ** (df)) < MAX_AUTO_RESOLUTION:
                    break
                if not (
                    data_dir / f"{downsample_folder_prefix}{2**(df+1)}" / filepath.name
                ).exists():
                    break
                df += 1

            self.downscale_factor = 2**df
            # TODO check if there is a better way to inform user of downscale factor instead of printing so many lines
            # CONSOLE.print(f"Auto image downscale factor of {self.downscale_factor}")
        else:
            self.downscale_factor = self.config.downscale_factor

        if self.downscale_factor > 1:
            return (
                data_dir
                / f"{downsample_folder_prefix}{self.downscale_factor}"
                / filepath.name
            )
        return data_dir / filepath

    def mushroom_get_train_eval_split_filename(
        self, image_filenames: List, test_filenames: List
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the train/eval split based on the filename of the images.

        Args:
            image_filenames: list of image filenames
        """
        if not test_filenames:
            num_images = len(image_filenames)
            return np.arange(num_images), np.arange(0)
        num_images = len(image_filenames)
        basenames = [
            os.path.basename(image_filename) for image_filename in image_filenames
        ]
        test_basenames = [
            os.path.basename(test_filename) for test_filename in test_filenames
        ]
        i_all = np.arange(num_images)
        i_train = []
        i_eval = []
        for idx, basename in zip(i_all, basenames):
            # check the frame index
            if basename in test_basenames:
                i_eval.append(idx)
            else:
                i_train.append(idx)

        return np.array(i_train), np.array(i_eval)

    def get_ele_from_meta(
        self,
        frames,
        data_dir,
        fx_fixed,
        fy_fixed,
        cx_fixed,
        cy_fixed,
        height_fixed,
        width_fixed,
        distort_fixed,
        use_faro_scanner_depths,
        load_depth_confidence_masks,
    ):
        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []
        if load_depth_confidence_masks:
            confidence_filenames = []

        for frame in frames:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)

            if not fx_fixed:
                assert "fl_x" in frame, "fx not specified in frame"
                fx.append(float(frame["fl_x"]))
            if not fy_fixed:
                assert "fl_y" in frame, "fy not specified in frame"
                fy.append(float(frame["fl_y"]))
            if not cx_fixed:
                assert "cx" in frame, "cx not specified in frame"
                cx.append(float(frame["cx"]))
            if not cy_fixed:
                assert "cy" in frame, "cy not specified in frame"
                cy.append(float(frame["cy"]))
            if not height_fixed:
                assert "h" in frame, "height not specified in frame"
                height.append(int(frame["h"]))
            if not width_fixed:
                assert "w" in frame, "width not specified in frame"
                width.append(int(frame["w"]))
            if not distort_fixed:
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

            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            if "mask_path" in frame:
                mask_filepath = Path(frame["mask_path"])
                mask_fname = self._get_fname(
                    mask_filepath, data_dir, downsample_folder_prefix="masks_"
                )
                mask_filenames.append(mask_fname)

            if "depth_file_path" in frame:
                if use_faro_scanner_depths:
                    depth_filepath = Path(
                        frame["depth_file_path"].replace("depths", "reference_depth")
                    )
                else:
                    depth_filepath = Path(frame["depth_file_path"])
                depth_fname = self._get_fname(
                    depth_filepath, data_dir, downsample_folder_prefix="depths_"
                )
                depth_filenames.append(depth_fname)
            if load_depth_confidence_masks:
                confidence_filepath = Path(
                    frame["depth_file_path"]
                    .replace("depth", "depth_normals_mask")
                    .replace("png", "jpg")
                )
                confidence_fname = self._get_fname(
                    confidence_filepath,
                    data_dir,
                    downsample_folder_prefix="confidence_",
                )
                confidence_filenames.append(confidence_fname)

        if load_depth_confidence_masks:
            return_filenames = {
                "image": image_filenames,
                "mask": mask_filenames,
                "depth": depth_filenames,
                "confidence": confidence_filenames,
            }
        else:
            return_filenames = {
                "image": image_filenames,
                "mask": mask_filenames,
                "depth": depth_filenames,
            }

        return (
            return_filenames,
            poses,
            fx,
            fy,
            cx,
            cy,
            height,
            width,
            distort,
        )


MushroomDataParserSpecification = DataParserSpecification(
    config=MushroomDataParserConfig(),
    description="MuSHRoom dataparser for indoor scenes",
)
