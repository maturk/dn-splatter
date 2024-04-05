"""
Datamanager that processes optional depth and normal data.
"""

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union

import torch
import torchvision.transforms.functional as TF

from dn_splatter.data.dn_dataset import GDataset
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)
from nerfstudio.data.datasets.base_dataset import InputDataset


@dataclass
class DNSplatterManagerConfig(FullImageDatamanagerConfig):
    """DataManager Config"""

    _target: Type = field(default_factory=lambda: DNSplatterDataManager)

    camera_res_scale_factor: float = 1.0
    """Rescale cameras"""


class DNSplatterDataManager(FullImageDatamanager):
    """DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: DNSplatterManagerConfig
    train_dataset: GDataset
    eval_dataset: GDataset

    def __init__(
        self,
        config: DNSplatterManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.config = config
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            **kwargs,
        )
        metadata = self.train_dataparser_outputs.metadata
        self.load_depths = (
            True
            if ("depth_filenames" in metadata)
            or ("sensor_depth_filenames" in metadata)
            or ("mono_depth_filenames") in metadata
            else False
        )

        self.load_normals = True if ("normal_filenames" in metadata) else False
        self.image_idx = 0

    def create_train_dataset(self) -> InputDataset:
        """Sets up the data loaders for training"""
        return GDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> InputDataset:
        """Sets up the data loaders for evaluation"""
        return GDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(
                split=self.test_split
            ),
            scale_factor=self.config.camera_res_scale_factor,
        )

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch"""

        # Don't randomly sample train images (keep t-1, t, t+1 ordering).
        self.image_idx = self.train_unseen_cameras.pop(0)
        if len(self.train_unseen_cameras) == 0:
            self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]
        data = deepcopy(self.cached_train[self.image_idx])
        data["image"] = data["image"].to(self.device)

        if "mask" in data:
            data["mask"] = data["mask"].to(self.device)
            if data["mask"].dim() == 2:
                data["mask"] = data["mask"][..., None]

        if self.load_depths:
            if "sensor_depth" in data:
                data["sensor_depth"] = data["sensor_depth"].to(self.device)
                if data["sensor_depth"].shape != data["image"].shape:
                    data["sensor_depth"] = TF.resize(
                        data["sensor_depth"].permute(2, 0, 1),
                        data["image"].shape[:2],
                        antialias=None,
                    ).permute(1, 2, 0)
            if "mono_depth" in data:
                data["mono_depth"] = data["mono_depth"].to(self.device)
                if data["mono_depth"].shape != data["image"].shape:
                    data["mono_depth"] = TF.resize(
                        data["mono_depth"].permute(2, 0, 1),
                        data["image"].shape[:2],
                        antialias=None,
                    ).permute(1, 2, 0)

        if self.load_normals:
            assert "normal" in data
            data["normal"] = data["normal"].to(self.device)
            if data["normal"].shape != data["image"].shape:
                data["normal"] = TF.resize(
                    data["normal"].permute(2, 0, 1),
                    data["image"].shape[:2],
                    antialias=None,
                ).permute(1, 2, 0)

        assert (
            len(self.train_dataset.cameras.shape) == 1
        ), "Assumes single batch dimension"
        camera = self.train_dataset.cameras[self.image_idx : self.image_idx + 1].to(
            self.device
        )
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = self.image_idx
        return camera, data

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle"""
        image_idx = self.eval_unseen_cameras[
            random.randint(0, len(self.eval_unseen_cameras) - 1)
        ]

        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        data = deepcopy(self.cached_eval[image_idx])
        data["image"] = data["image"].to(self.device)
        if "mask" in data:
            data["mask"] = data["mask"].to(self.device)
            if data["mask"].dim() == 2:
                data["mask"] = data["mask"][..., None]
        if self.load_depths:
            if "sensor_depth" in data:
                data["sensor_depth"] = data["sensor_depth"].to(self.device)
                if data["sensor_depth"].shape != data["image"].shape:
                    data["sensor_depth"] = TF.resize(
                        data["sensor_depth"].permute(2, 0, 1),
                        data["image"].shape[:2],
                        antialias=None,
                    ).permute(1, 2, 0)
            if "mono_depth" in data:
                data["mono_depth"] = data["mono_depth"].to(self.device)
                if data["mono_depth"].shape != data["image"].shape:
                    data["mono_depth"] = TF.resize(
                        data["mono_depth"].permute(2, 0, 1),
                        data["image"].shape[:2],
                        antialias=None,
                    ).permute(1, 2, 0)
        if self.load_normals:
            assert "normal" in data
            data["normal"] = data["normal"].to(self.device)
            if data["normal"].shape != data["image"].shape:
                data["normal"] = TF.resize(
                    data["normal"].permute(2, 0, 1),
                    data["image"].shape[:2],
                    antialias=None,
                ).permute(1, 2, 0)

        assert (
            len(self.eval_dataset.cameras.shape) == 1
        ), "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx
        return camera, data

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next eval image"""

        image_idx = self.eval_unseen_cameras[
            random.randint(0, len(self.eval_unseen_cameras) - 1)
        ]
        data = deepcopy(self.cached_eval[image_idx])
        data["image"] = data["image"].to(self.device)
        assert (
            len(self.eval_dataset.cameras.shape) == 1
        ), "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        return camera, data
