"""Depths from monocular depths

Optional: Completes sparse or incomplete depth maps with scale aligned zoe-depth estimates

Run with:
    python dn_splatter/scripts/zoe_depth_completion.py 
        --path-to-transforms [PATH_TO_TRANSFORMS.JSON]
        --save-path [path to save dir]: default is data_root/mono_depth
        --create_new_transforms [True/False]: default True

TODO: currently assumes depth and images are equal sizes. This might be a problem for some datasets
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF
import tyro
from dn_splatter.utils.camera_utils import euclidean_to_z_depth
from dn_splatter.utils.utils import (
    SCALE_FACTOR,
    depth_path_to_tensor,
    get_filename_list,
    image_path_to_tensor,
)
from rich.console import Console
from rich.progress import track
from torch import Tensor

from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.misc import torch_compile

CONSOLE = Console(width=120)
BATCH_SIZE = 50

device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DepthsFromPretrained:
    """Generate monocular depth estimates for input images"""

    data_dir: Path
    """Path to data root"""
    img_dir_name: str = "images"
    """If transforms.json file is not found, use images in this folder"""
    pretrain_model: Literal["zoe"] = "zoe"
    """Which pretrained monocular depth model to use"""
    save_path: Optional[Path] = None
    """Default save path is /mono_depth"""

    def main(self):
        if os.path.exists(os.path.join(self.data_dir, "transforms.json")):
            CONSOLE.print(
                "Found path to transforms.json, using this to generate monodepths."
            )
            meta = load_from_json(self.data_dir / Path("transforms.json"))
            image_paths = [
                self.data_dir / Path(frame["file_path"]) for frame in meta["frames"]
            ]
            run_monocular_depths(
                images=image_paths,
                save_path=self.save_path,
                pretrain_model=self.pretrain_model,
            )

        elif len(os.listdir(self.data_dir / Path(self.img_dir_name))) != 0:
            CONSOLE.print(
                f"[bold yellow]Found images in /{self.img_dir_name}, using these to generate monodepths."
            )
            image_paths = get_filename_list(self.data_dir / Path(self.img_dir_name))
            run_monocular_depths(
                images=image_paths,
                save_path=self.save_path,
                pretrain_model=self.pretrain_model,
            )
        else:
            CONSOLE.print(
                f"Could not find tranforms.json or images in /{self.img_dir_name}, quitting."
            )
        CONSOLE.print("Completed!")


def depth_align(
    depths: Tensor,
    est_depths: Tensor,
    iterations: int = 1000,
    lr: float = 0.1,
    threshold: float = 0.0,  # treshold for masking invalid depths
) -> Tensor:
    """Depth align with real scale and shift

    solves: argmin_{scale}_{shift} || depth - (scale * depth_est + shift) ||^2

    Args:
        depths: sensor depths
        est_depths: estimated zoe/mono depths
        iterations: number of grad descent iterations
        lr: learning rate
        threshold: threshold for masking invalid depth values

    Returns:
        Tensor of depth aligned zoe depth estimates
    """
    assert (
        depths.shape[:] == est_depths.shape[:]
    ), f"incorrect depth {depths.shape} and estimated depth {est_depths.shape} shapes"
    mono_depths = []
    for idx in track(
        range(depths.shape[0]),
        description="Depth completion: solving for scale/shift pairs ...",
    ):
        scale = torch.nn.Parameter(torch.tensor([1], device=device, dtype=torch.float))
        shift = torch.nn.Parameter(torch.tensor([0], device=device, dtype=torch.float))
        depth = depths[idx, ...].float()
        est_depth = est_depths[idx, ...].float()

        # threshold invalid depths
        mask = depth > threshold
        depth_masked = depth[mask].to(device)
        est_masked = est_depth[mask].to(device)

        mse_loss = torch.nn.MSELoss()
        optimizer = torch.optim.Adam([scale, shift], lr=lr)
        avg_err = []
        for step in range(iterations):
            optimizer.zero_grad()
            loss = mse_loss(depth_masked, scale * est_masked + shift)
            loss.backward()
            optimizer.step()
        avg_err.append(loss.item())
        mono_depths.append(scale * est_depth + shift)
    avg = sum(avg_err) / len(avg_err)
    CONSOLE.print(
        f"[bold yellow]Average depth completion error for batch depths is: {avg:3f} which is {'good' if avg<0.2 else 'bad'}"
    )
    return torch.stack(mono_depths, dim=0)


def run_monocular_depths(
    images: List[Path],
    save_path: Optional[Path] = None,
    pretrain_model: Literal["zoe"] = "zoe",
):
    image_tensors = []
    batch_size = BATCH_SIZE
    num_frames = len(images)

    if pretrain_model == "zoe":
        repo = "isl-org/ZoeDepth"
        zoe = torch_compile(torch.hub.load(repo, "ZoeD_N", pretrained=True).to(device))
    else:
        raise NotImplementedError
    for batch_index in range(0, num_frames, batch_size):
        CONSOLE.print(
            f"[bold green]Processing batch {batch_index // batch_size} / {num_frames//batch_size}"
        )
        batch_frames = images[batch_index : batch_index + batch_size]
        with torch.no_grad():
            image_tensors = []
            for frame_index in range(len(batch_frames)):
                frame = images[batch_index : batch_index + batch_size][frame_index]
                image = image_path_to_tensor(frame).to(device)
                image_tensors.append(image)

            image_tensors = torch.stack(image_tensors, dim=0)

            # run mono pretrain model on input images
            mono_depths = []
            for i in track(
                range(image_tensors.shape[0]),
                description="Running Depth completion model {} for batch frames ...".format(
                    pretrain_model
                ),
            ):
                image = image_tensors[i]
                image = torch.permute(image, (2, 0, 1)).unsqueeze(0).to(device)
                if image.shape[1] == 4:
                    image = image[:, :3, :, :]
                if pretrain_model == "zoe":
                    mono_depth_tensor = zoe.infer(image).squeeze().unsqueeze(-1)
                else:
                    raise NotImplementedError

                mono_depths.append(mono_depth_tensor)
            mono_depths = torch.stack(mono_depths, dim=0)

        # save data
        if save_path is None:
            save_path = images[0].parent.parent / "mono_depth"
        save_path.mkdir(exist_ok=True, parents=True)
        for idx in track(
            range(image_tensors.shape[0]),
            description=f"saving depth images to {save_path} ...",
        ):
            # rescale depths back to original scale
            mono_depths[idx] = mono_depths[idx] / SCALE_FACTOR
            file_name = Path(images[batch_index : batch_index + batch_size][idx]).stem
            file_save_path = save_path / Path(file_name).with_suffix(".npy")
            # save the depth
            depth = mono_depths[idx].detach().cpu().numpy()
            np.save(str(file_save_path), depth)


def depth_from_pretrain(
    input_folder: Path,
    img_dir_name: str,
    path_to_transforms: Optional[Path],
    save_path: Optional[Path] = None,
    create_new_transforms: bool = False,
    is_euclidean_depth: bool = False,
    return_mode: Literal["mono", "mono-aligned"] = "mono",
    pretrain_model: Literal["zoe"] = "zoe",
):
    """
    Args:
        path_to_transforms: path to json file where paths to images and depths are specified
        save_path: path to save new completed mono depth images
        create_new_transforms: creates a "mono_depth_transforms.json" file containing updated meta deta with a new path for "mono_depth_file_path"
    """
    # Two methods 1) if transforms file is give, load info from that. Or 2) try to load images from input_folder/images dir.
    if path_to_transforms is not None:
        meta = load_from_json(path_to_transforms)
        meta["frames"] = meta["frames"]
        image_tensors = []
        depth_tensors = []

        batch_size = BATCH_SIZE
        num_frames = len(meta["frames"])
        if pretrain_model == "zoe":
            repo = "isl-org/ZoeDepth"
            zoe = torch_compile(
                torch.hub.load(repo, "ZoeD_N", pretrained=True).to(device)
            )
        else:
            raise NotImplementedError
        for batch_index in range(0, len(meta["frames"]), batch_size):
            CONSOLE.print(
                f"[bold green]Processing batch {batch_index // batch_size} / {num_frames//batch_size}"
            )

            batch_frames = meta["frames"][batch_index : batch_index + batch_size]
            with torch.no_grad():
                image_tensors = []
                depth_tensors = []
                for frame_index in range(len(batch_frames)):
                    frame = meta["frames"][batch_index : batch_index + batch_size][
                        frame_index
                    ]
                    image = image_path_to_tensor(input_folder / frame["file_path"]).to(
                        device
                    )

                    image_tensors.append(image)

                    # return aligned zoe estimates
                    if return_mode == "mono-aligned":
                        depth = depth_path_to_tensor(
                            input_folder / frame["depth_file_path"],
                            return_color=False,
                            scale_factor=SCALE_FACTOR,
                        )
                        if is_euclidean_depth:
                            if "fl_x" in meta:
                                fx = meta["fl_x"]
                                fy = meta["fl_y"]
                                cx = meta["cx"]
                                cy = meta["cy"]
                                h = meta["h"]
                                w = meta["w"]
                            elif "fl_x" in frame:
                                fx = frame["fl_x"]
                                fy = frame["fl_y"]
                                cx = frame["cx"]
                                cy = frame["cy"]
                                h = frame["h"]
                                w = frame["w"]

                                depth = euclidean_to_z_depth(
                                    depths=depth,
                                    fx=fx,
                                    fy=fy,
                                    cx=cx,
                                    cy=cy,
                                    img_size=(w, h),
                                    device="cpu",
                                )
                                depth_tensors.append(depth)
                        else:
                            depth_tensors.append(depth)

                image_tensors = torch.stack(image_tensors, dim=0)
                if return_mode == "mono-aligned":
                    depth_tensors = torch.stack(depth_tensors, dim=0)

                # run mono pretrain model on input images
                mono_depths = []
                for i in track(
                    range(image_tensors.shape[0]),
                    description="Running Depth completion model {} for batch frames ...".format(
                        pretrain_model
                    ),
                ):
                    image = image_tensors[i]
                    image = torch.permute(image, (2, 0, 1)).unsqueeze(0).to(device)
                    if image.shape[1] == 4:
                        image = image[:, :3, :, :]
                    if pretrain_model == "zoe":
                        mono_depth_tensor = zoe.infer(image).squeeze().unsqueeze(-1)
                    else:
                        raise NotImplementedError

                    mono_depths.append(mono_depth_tensor)
                mono_depths = torch.stack(mono_depths, dim=0)

            # align mono depths by solving for scale and shift
            if return_mode == "mono-aligned":
                if depth_tensors.shape[:] != mono_depths.shape[:]:
                    depth_tensors = TF.resize(
                        depth_tensors.permute(0, 3, 1, 2),
                        mono_depths.shape[1:3],
                        antialias=None,
                    ).permute(0, 2, 3, 1)
                depths_completed = depth_align(
                    depths=depth_tensors, est_depths=mono_depths
                )
            else:
                depths_completed = mono_depths

            # save data
            if save_path is None:
                save_path = input_folder / "mono_depth"
            save_path.mkdir(exist_ok=True, parents=True)
            for idx in track(
                range(image_tensors.shape[0]),
                description=f"saving depth images to {save_path} ...",
            ):
                # rescale depths back to original scale
                depths_completed[idx] = depths_completed[idx] / SCALE_FACTOR
                file_name = Path(
                    meta["frames"][batch_index : batch_index + batch_size][idx][
                        "file_path"
                    ]
                ).stem
                file_save_path = save_path / Path(file_name + "_aligned").with_suffix(
                    ".npy"
                )
                # save the depth
                depth = depths_completed[idx].detach().cpu().numpy()
                np.save(str(file_save_path), depth)

    else:
        assert len(os.listdir(input_folder / Path(img_dir_name))) != 0
        images = get_filename_list(input_folder / Path(img_dir_name))
        image_tensors = []
        depth_tensors = []
        num_frames = len(images)

        batch_size = BATCH_SIZE
        if pretrain_model == "zoe":
            repo = "isl-org/ZoeDepth"
            zoe = torch_compile(
                torch.hub.load(repo, "ZoeD_N", pretrained=True).to(device)
            )
        else:
            raise NotImplementedError
        for batch_index in range(0, len(images), batch_size):
            CONSOLE.print(
                f"[bold green]Processing batch {batch_index // batch_size} / {num_frames//batch_size}"
            )
            batch_frames = images[batch_index : batch_index + batch_size]
            with torch.no_grad():
                image_tensors = []
                depth_tensors = []
                for frame_index in range(len(batch_frames)):
                    frame = images[batch_index : batch_index + batch_size][frame_index]
                    image = image_path_to_tensor(frame).to(device)
                    image_tensors.append(image)

                    if return_mode == "mono-aligned":
                        # currently we only use image folder input for scannetpp iphone and mushroom
                        depth = depth_path_to_tensor(
                            (frame.parent.parent / "depth" / frame.name).with_suffix(
                                ".png"
                            ),
                            return_color=False,
                            scale_factor=SCALE_FACTOR,
                        )
                        if depth.shape[:2] != image.shape[:2]:
                            depth = TF.resize(
                                depth.permute(2, 0, 1), image.shape[:2], antialias=None
                            ).permute(1, 2, 0)

                        depth_tensors.append(depth)

                image_tensors = torch.stack(image_tensors, dim=0)
                if return_mode == "mono-aligned":
                    depth_tensors = torch.stack(depth_tensors, dim=0)
                # run zoe on input images
                mono_depths = []
                for i in track(
                    range(image_tensors.shape[0]),
                    description="Running depth completion {} for batch frames ...".format(
                        pretrain_model
                    ),
                ):
                    image = image_tensors[i]
                    image = torch.permute(image, (2, 0, 1)).unsqueeze(0).to(device)
                    if image.shape[1] == 4:
                        image = image[:, :3, :, :]
                    if pretrain_model == "zoe":
                        mono_depth_tensor = zoe.infer(image).squeeze().unsqueeze(-1)
                    else:
                        raise NotImplementedError

                    mono_depths.append(mono_depth_tensor)
                mono_depths = torch.stack(mono_depths, dim=0)

            # align mono depths by solving for scale and shift
            if return_mode == "mono-aligned":
                depths_completed = depth_align(
                    depths=depth_tensors, est_depths=mono_depths
                )
            else:
                depths_completed = mono_depths

            # save data
            if save_path is None:
                save_path = input_folder / "mono_depth"
            save_path.mkdir(exist_ok=True, parents=True)
            for idx in track(
                range(image_tensors.shape[0]),
                description=f"saving depth images to {save_path} ...",
            ):
                # rescale depths back to original scale
                depths_completed[idx] = depths_completed[idx] / SCALE_FACTOR

                file_name = Path(
                    images[batch_index : batch_index + batch_size][idx]
                ).stem

                file_save_path = save_path / Path(file_name).with_suffix(".npy")
                # save the depth
                depth = depths_completed[idx].detach().cpu().numpy()

                np.save(str(file_save_path), depth)

    if create_new_transforms:
        for idx in range(num_frames):
            file_name = Path(meta["frames"][idx]["depth_file_path"]).stem
            file_save_path = Path(save_path.stem) / Path(file_name).with_suffix(".png")
            meta["frames"][idx].update({"mono_depth_file_path": str(file_save_path)})

        CONSOLE.print(
            f"[bold yellow]saving mono_depth_transformations.json to {input_folder}/mono_depth_transformations.json"
        )
        with open(
            input_folder / Path("mono_depth_transformations.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(meta, f, indent=4)


if __name__ == "__main__":
    # tyro.cli(depth_from_pretrain)
    tyro.cli(DepthsFromPretrained).main()
