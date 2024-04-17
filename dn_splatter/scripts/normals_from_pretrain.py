"""Generate monocular normal estimates from Omnidata model.

Note, parts of this script are adapted from https://github.com/autonomousvision/monosdf
"""

import glob
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import tyro
from dn_splatter.utils.camera_utils import euclidean_to_z_depth
from dn_splatter.utils.normal_utils import normal_from_depth_image
from dn_splatter.utils.utils import (
    depth_path_to_tensor,
    get_filename_list,
    image_path_to_tensor,
    save_img,
    save_normal,
)
from dn_splatter.scripts.dsine.dsine_predictor import DSinePredictor
from omnidata_tools.torch.modules.midas.dpt_depth import DPTDepthModel
from rich.console import Console
from tqdm import tqdm

from nerfstudio.utils.io import load_from_json
from jaxtyping import Float

BATCH_SIZE = 15
CONSOLE = Console(width=120)

image_size = 384  # omnidata can only accept 384x384 images
step = 96  # need to be smaller than image_size


@dataclass
class NormalsFromPretrained:
    """Generate monocular normal estimates for input images"""

    data_dir: Path
    """Path to data root"""
    transforms_name: str = "transforms.json"
    """Name of transformation json file"""
    img_dir_name: str = "images"
    """If transforms.json file is not found, use images in this folder"""
    save_path: Optional[Path] = None
    """Default save path is /normals_from_pretrain"""
    resolution: Literal["low", "hd"] = "low"
    """Whether to make low resolution or full image resolution normal estimates"""
    force_images_dir: bool = False
    """Force to use img_dir_name instead of transforms.json if found"""
    model_type: Literal["omnidata", "dsine"] = "omnidata"

    def main(self):
        if (
            os.path.exists(os.path.join(self.data_dir, self.transforms_name))
            and not self.force_images_dir
        ):
            CONSOLE.print(
                f"Found path to {self.transforms_name}, using this to generate mononormals."
            )
            meta = load_from_json(self.data_dir / self.transforms_name)
            image_paths = [
                self.data_dir / Path(frame["file_path"]) for frame in meta["frames"]
            ]
            if self.model_type == "omnidata":
                if self.resolution == "low":
                    run_monocular_normals(images=image_paths, save_path=self.save_path)
                else:
                    run_monocular_normals_hd(
                        images=image_paths, save_path=self.save_path
                    )
            elif self.model_type == "dsine":
                run_monocular_dsine(images=image_paths, save_path=self.save_path)

        elif len(os.listdir(self.data_dir / Path(self.img_dir_name))) != 0:
            CONSOLE.print(
                f"[bold yellow]Found images in /{self.img_dir_name}, using these to generate mononormals."
            )
            image_paths = get_filename_list(self.data_dir / Path(self.img_dir_name))
            if self.model_type == "omnidata":
                if self.resolution == "low":
                    run_monocular_normals(images=image_paths, save_path=self.save_path)
                else:
                    run_monocular_normals_hd(
                        images=image_paths, save_path=self.save_path
                    )
            elif self.model_type == "dsine":
                run_monocular_dsine(images=image_paths, save_path=self.save_path)
        else:
            CONSOLE.print(
                f"Could not find {self.transforms_name} or images in /{self.img_dir_name}, quitting."
            )
        CONSOLE.print("Completed generating mono-normals!")


def run_monocular_dsine(
    images: List[Path],
    save_path: Path,
):
    """Generates normal maps from pretrained omnidata
    Args:
        images: list of image paths
        save_path: path to save directory

    Returns:
        None
    """
    if save_path is None:
        save_path = images[0].parent.parent / "normals_from_pretrain"
    save_path.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DSinePredictor(device=device)

    image_list = images

    for idx, single_path in enumerate(tqdm(image_list)):
        bgr = cv2.imread(str(single_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        normal_b3hw: Float[torch.Tensor, "b 3 h w"] = model(rgb)
        b, _, h, w = normal_b3hw.shape
        normal_np_bhw3 = normal_b3hw.permute(0, 2, 3, 1).numpy(force=True)
        # convert from LUF to RUF
        normal_np_bhw3 = normal_np_bhw3.reshape(-1, 3)
        transform_mat = np.diag([-1, 1, 1])
        normal_np_bhw3 = normal_np_bhw3 @ transform_mat

        normal_np_bhw3 = normal_np_bhw3.reshape(b, h, w, 3)
        # bring from [-1, 1] to [0, 1]
        normal_np_bhw3 = (normal_np_bhw3 + 1.0) / 2.0
        normal_np_hw3 = normal_np_bhw3.squeeze(0)

        # png is between 0-1, (h, w, 3), float32
        save_normal(
            normal_np_hw3,
            f"{save_path / single_path.stem}.png",
            verbose=False,
            format="png",
        )


def run_monocular_normals(
    images: List[Path],
    save_path: Path,
    omnidata_pretrained_weights_path: Path = Path("omnidata_ckpt"),
) -> None:
    """Generates normal maps from pretrained omnidata
    Args:
        images: list of image paths
        save_path: path to save directory
        omnidata_pretrained_weights_path: omnidata weights path

    Returns:
        None
    """
    if save_path is None:
        save_path = images[0].parent.parent / "normals_from_pretrain"
    save_path.mkdir(exist_ok=True, parents=True)

    map_location = (
        (lambda storage, loc: storage.cuda())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    omnidata_pretrained_weights_path = (
        omnidata_pretrained_weights_path / "omnidata_dpt_normal_v2.ckpt"
    )
    model = DPTDepthModel(backbone="vitb_rn50_384", num_channels=3)  # DPT Hybrid
    checkpoint = torch.load(omnidata_pretrained_weights_path, map_location=map_location)
    if "state_dict" in checkpoint:
        state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)

    image_list = images
    batch_size = BATCH_SIZE

    for batch_index in range(0, len(image_list), batch_size):
        CONSOLE.print(
            f"[bold green]Generating normals from imgs for batch {batch_index // batch_size} / {len(image_list)//batch_size}"
        )
        batch_images_path = image_list[batch_index : batch_index + batch_size]
        with torch.no_grad():
            image_tensors = []
            for path_index in range(len(batch_images_path)):
                single_path = batch_images_path[path_index]
                image = image_path_to_tensor(single_path).to(device)
                H, W = image.shape[:2]
                if image.shape[0] > image_size or image.shape[1] > image_size:
                    image = TF.resize(
                        image.permute(2, 0, 1), (image_size, image_size), antialias=None
                    ).permute(1, 2, 0)
                image_tensors.append(image)

            image_tensors = torch.stack(image_tensors, dim=0)
            image_tensors = image_tensors.permute(0, 3, 1, 2)
            output = model(image_tensors).clamp(min=0, max=1)

            # save data
            for path_index in range(len(batch_images_path)):
                single_path = batch_images_path[path_index]
                result_i = output[path_index].permute(1, 2, 0)
                if result_i.shape[:2] != (H, W):
                    result_i = TF.resize(
                        result_i.permute(2, 0, 1), (H, W), antialias=None
                    ).permute(1, 2, 0)

                save_normal(
                    result_i,
                    save_path
                    / (batch_images_path[path_index].name.split(".")[0] + ".png"),
                    verbose=False,
                    format="png",
                )
                save_normal(
                    result_i.permute(2, 0, 1).detach().cpu().numpy(),
                    save_path
                    / (batch_images_path[path_index].name.split(".")[0] + ".npy"),
                    verbose=False,
                    format="npy",
                )


def run_monocular_normals_hd(
    images: List[Path],
    save_path: Path,
    omnidata_pretrained_weights_path: Path = Path("omnidata_ckpt"),
) -> None:
    """Generates hd normal maps from pretrained omnidata
    Args:
        images: list of image paths
        save_path: path to save directory
        omnidata_pretrained_weights_path: omnidata weights path

    Returns:
        None
    """

    # temporary folders, used to save overlapped cropped images
    if save_path is None:
        save_path = images[0].parent.parent / "normals_from_pretrain"
    save_path.mkdir(exist_ok=True, parents=True)

    tmp_folder = save_path / Path("highres_tmp")
    image_pathces_folder = tmp_folder / "images"
    os.makedirs(image_pathces_folder, exist_ok=True)
    normal_patches_folder = tmp_folder / "normals_patches"
    os.makedirs(normal_patches_folder, exist_ok=True)

    rgbs = images
    image = cv2.imread(str(rgbs[0]))

    H, W = image.shape[:2]
    assert H > image_size and W > image_size
    assert step < image_size

    x = (W - image_size) // step  # number of patches in x direction
    y = (H - image_size) // step  # number of patches in y direction

    crop_patches(rgbs, image_pathces_folder, x, y, H, W)

    image_patches = get_filename_list(image_pathces_folder)

    run_monocular_normals(
        image_patches, normal_patches_folder, omnidata_pretrained_weights_path
    )

    merge_patches(rgbs, normal_patches_folder, save_path, x, y, H, W)

    shutil.rmtree(tmp_folder)


def normals_from_pretrain(
    image_folder: Path,
    save_path: Path,
    omnidata_pretrained_weights_path: Path = Path("omnidata_ckpt"),
) -> None:
    """Generates normal maps from pretrained omnidata
    Args:
        image_folder: path to image directory
        save_path: path to save directory

    Returns:
        None
    """
    map_location = (
        (lambda storage, loc: storage.cuda())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    omnidata_pretrained_weights_path = (
        omnidata_pretrained_weights_path / "omnidata_dpt_normal_v2.ckpt"
    )
    model = DPTDepthModel(backbone="vitb_rn50_384", num_channels=3)  # DPT Hybrid
    checkpoint = torch.load(omnidata_pretrained_weights_path, map_location=map_location)
    if "state_dict" in checkpoint:
        state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)

    image_list = sorted([file for file in image_folder.glob("*") if file.is_file()])

    batch_size = BATCH_SIZE

    for batch_index in range(0, len(image_list), batch_size):
        CONSOLE.print(
            f"[bold green]Processing batch {batch_index // batch_size} / {len(image_list)//batch_size}"
        )
        batch_images_path = image_list[batch_index : batch_index + batch_size]
        with torch.no_grad():
            image_tensors = []
            for path_index in range(len(batch_images_path)):
                single_path = batch_images_path[path_index]
                image = image_path_to_tensor(single_path).to(device)
                image_tensors.append(image)

            image_tensors = torch.stack(image_tensors, dim=0)
            image_tensors = image_tensors.permute(0, 3, 1, 2)
            output = model(image_tensors).clamp(min=0, max=1)

            # save data
            for path_index in range(len(batch_images_path)):
                single_path = batch_images_path[path_index]
                result_i = output[path_index].permute(1, 2, 0)
                save_normal(
                    result_i,
                    save_path
                    / (batch_images_path[path_index].name.split(".")[0] + ".png"),
                    verbose=False,
                    format="png",
                )

                save_normal(
                    result_i.permute(2, 0, 1).detach().cpu().numpy(),
                    save_path
                    / (batch_images_path[path_index].name.split(".")[0] + ".npy"),
                    verbose=False,
                    format="npy",
                )


def normals_from_omnidata_hd(
    save_path: Path,
    omnidata_pretrained_weights_path: Path = Path("omnidata_ckpt"),
    images_folder: Optional[Path] = None,  # path to the folder containing the images
    path_to_transforms: Optional[
        Path
    ] = None,  # path to the folder containing the transforms
):
    # temporary folders, used to save overlapped cropped images
    tmp_folder = save_path / Path("highres_tmp")
    image_pathces_folder = tmp_folder / "images"
    os.makedirs(image_pathces_folder, exist_ok=True)
    normal_patches_folder = tmp_folder / "normals_patches"
    os.makedirs(normal_patches_folder, exist_ok=True)

    # load poses and images
    # TODO: find better way to find image file names
    if images_folder is None:
        meta = json.load(open(path_to_transforms, "r"))
        frames = meta["frames"]
        rgbs = []
        for frame in frames:
            rgbs.append(path_to_transforms.parent / frame["file_path"])
    else:
        rgbs = sorted([file for file in images_folder.glob("*") if file.is_file()])

    image = cv2.imread(str(rgbs[0]))

    H, W = image.shape[:2]
    assert H > image_size and W > image_size
    assert step < image_size

    x = (W - image_size) // step  # number of patches in x direction
    y = (H - image_size) // step  # number of patches in y direction

    crop_patches(rgbs, image_pathces_folder, x, y, H, W)

    normals_from_pretrain(
        image_pathces_folder, normal_patches_folder, omnidata_pretrained_weights_path
    )

    merge_patches(rgbs, normal_patches_folder, save_path, x, y, H, W)

    try:
        shutil.rmtree(tmp_folder)
    except Exception as e:
        print(f"Error: {e}")


def normals_from_depths(
    path_to_transforms: Path,
    save_path: Optional[Path] = None,
    normal_format: Literal["opencv", "opengl"] = "opengl",
    is_euclidean_depth: bool = False,
) -> None:
    """Normal maps from depth maps

    Args:
        path_to_transforms: path to .json file
        normal_format: camera coordinate system to save normals in
        is_euclidean_depth: if depths are euclidian or not

    Returns:
        None
    """
    meta = load_from_json(path_to_transforms)
    frames = meta["frames"]
    assert "depth_file_path" in frames[0], "Oopsie no depth paths in the transforms!"
    depth_name = "depth_file_path"
    data_dir = path_to_transforms.parent
    if save_path is None:
        save_dir = Path(data_dir / "normals_from_depth")

    else:
        save_dir = save_path
    save_dir.mkdir(parents=True, exist_ok=True)

    if "fl_x" in meta:
        fx = meta["fl_x"]
        fy = meta["fl_y"]
        cx = meta["cx"]
        cy = meta["cy"]
        h = meta["h"]
        w = meta["w"]
    elif "fl_x" in frames[0]:
        fx = frames[0]["fl_x"]
        fy = frames[0]["fl_y"]
        cx = frames[0]["cx"]
        cy = frames[0]["cy"]
        h = frames[0]["h"]
        w = frames[0]["w"]
    else:
        raise LookupError("Cant find intrinsics data!")

    print(f"Saving normals to {save_dir}")
    for i, frame in enumerate(frames):
        depth = depth_path_to_tensor(path_to_transforms.parent / frame[depth_name])
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
        else:
            raise LookupError("Cant find intrinsics data!")
        if is_euclidean_depth:
            depth = euclidean_to_z_depth(
                depths=depth, fx=fx, fy=fy, cx=cx, cy=cy, img_size=(w, h), device="cpu"
            )

        image_name = Path(frame["file_path"]).stem
        image_name = image_name + ".png"
        c2w = torch.Tensor(frame["transform_matrix"]).float()

        # normals in world space
        normal = normal_from_depth_image(
            depth,
            fx,
            fy,
            cx,
            cy,
            (w, h),
            c2w=c2w
            @ torch.diag(
                torch.tensor([1, -1, -1, 1], device=c2w.device, dtype=c2w.dtype)
            )
            if normal_format == "opencv"
            else c2w,
            device="cpu",
            smooth=False,
        )
        # normals in camera space
        normal_cam = normal.cpu() @ c2w[:3, :3]

        # convert normals from [-1,1] to [0,1] and save normals as png files
        save_img((normal_cam + 1) / 2, save_dir / image_name, verbose=False)


# copy from vis-mvsnet
def find_files(dir):
    if os.path.isdir(dir):
        files_grabbed = glob.glob(os.path.join(dir, "*"))
        files_grabbed = sorted(files_grabbed)

        return files_grabbed
    else:
        return []


# adatpted from https://github.com/dakshaau/ICP/blob/master/icp.py#L4 for rotation only
def best_fit_transform(A, B):
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    AA = A
    BB = B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    return R


# align normal map in the x direction from left to right
def align_normal_x(normal1, normal2, s1, e1, s2, e2):
    assert normal1.shape[0] == normal2.shape[0]
    assert normal1.shape[1] == normal2.shape[1]

    assert (e1 - s1) == (e2 - s2)

    R = best_fit_transform(
        normal2[:, :, s2:e2].reshape(3, -1).T, normal1[:, :, s1:e1].reshape(3, -1).T
    )

    normal2_aligned = (R @ normal2.reshape(3, -1)).reshape(normal2.shape)
    result = np.ones(
        (3, normal1.shape[1], normal1.shape[2] + normal2.shape[2] - (e1 - s1))
    )

    result[:, :, :s1] = normal1[:, :, :s1]
    result[:, :, normal1.shape[2] :] = normal2_aligned[:, :, e2:]

    weight = np.linspace(1, 0, (e1 - s1))[None, None, :]

    result[:, :, s1 : normal1.shape[2]] = normal1[:, :, s1:] * weight + normal2_aligned[
        :, :, :e2
    ] * (1 - weight)
    result = result / (np.linalg.norm(result, axis=0) + 1e-15)[None]

    return result


# align normal map in the y direction from top to down
def align_normal_y(normal1, normal2, s1, e1, s2, e2):
    assert normal1.shape[0] == normal2.shape[0]
    assert normal1.shape[2] == normal2.shape[2]

    assert (e1 - s1) == (e2 - s2)

    R = best_fit_transform(
        normal2[:, s2:e2, :].reshape(3, -1).T, normal1[:, s1:e1, :].reshape(3, -1).T
    )

    normal2_aligned = (R @ normal2.reshape(3, -1)).reshape(normal2.shape)
    result = np.ones(
        (3, normal1.shape[1] + normal2.shape[1] - (e1 - s1), normal1.shape[2])
    )

    result[:, :s1, :] = normal1[:, :s1, :]
    result[:, normal1.shape[1] :, :] = normal2_aligned[:, e2:, :]

    weight = np.linspace(1, 0, (e1 - s1))[None, :, None]

    result[:, s1 : normal1.shape[1], :] = normal1[:, s1:, :] * weight + normal2_aligned[
        :, :e2, :
    ] * (1 - weight)
    result = result / (np.linalg.norm(result, axis=0) + 1e-15)[None]

    return result


def crop_patches(
    rgbs: list, image_pathces_folder: Path, x: int, y: int, H: int, W: int
):
    for image_path in tqdm(rgbs, desc="Cropping hd images into smaller patches"):
        image_name = os.path.basename(image_path)
        image_name = image_name.split(".")[0]

        image = image_path_to_tensor(image_path)

        # crop images every step pixels
        for j in range(y):  # height
            for i in range(x):  # width
                image_cur = image[
                    j * step : j * step + image_size,
                    i * step : i * step + image_size,
                    :,
                ]
                target_file = os.path.join(
                    image_pathces_folder, "%06s_%02d_%02d.jpg" % (image_name, j, i)
                )
                save_img(image_cur, target_file, verbose=False)

        # save the last row and column
        for j in range(y):
            image_cur = image[j * step : j * step + image_size, W - image_size : W, :]
            target_file = os.path.join(
                image_pathces_folder, "%06s_%02d_%02d.jpg" % (image_name, j, x)
            )
            save_img(image_cur, target_file, verbose=False)

        for i in range(x):
            image_cur = image[H - image_size : H, i * step : i * step + image_size, :]
            target_file = os.path.join(
                image_pathces_folder, "%06s_%02d_%02d.jpg" % (image_name, y, i)
            )
            save_img(image_cur, target_file, verbose=False)

        # save the last image
        image_cur = image[H - image_size : H, W - image_size : W, :]
        target_file = os.path.join(
            image_pathces_folder, "%06s_%02d_%02d.jpg" % (image_name, y, x)
        )
        save_img(image_cur, target_file, verbose=False)

        # save middle image for alignments
        image_cur = image[
            H // 2 - int(image_size / 2) : H // 2 + int(image_size / 2),
            W // 2 - int(image_size / 2) : W // 2 + int(image_size / 2),
        ]

        target_file = os.path.join(image_pathces_folder, "%06s_mid.jpg" % (image_name))
        save_img(image_cur, target_file, verbose=False)


def merge_patches(
    rgbs: list,
    normal_patches_folder: Path,
    out_path_for_training: Path,
    x: int,
    y: int,
    H: int,
    W: int,
):
    for image_path in tqdm(rgbs, desc="Merging patches into hd normal maps"):
        image_name = os.path.basename(image_path)
        image_name = image_name.split(".")[0]
        # normal
        normals_row = []
        # align normal maps from left to right row by row
        for j in range(y):
            normals = []
            # load 0 -> x normal patches
            for i in range(x):
                normal_path = os.path.join(
                    normal_patches_folder, "%06s_%02d_%02d.npy" % (image_name, j, i)
                )
                normal = np.load(normal_path)
                normal = normal * 2.0 - 1.0
                normal = normal / (np.linalg.norm(normal, axis=0) + 1e-15)[None]
                normals.append(normal)
            # load last column normal patches
            normal_path = os.path.join(
                normal_patches_folder, "%06s_%02d_%02d.npy" % (image_name, j, x)
            )
            normal = np.load(normal_path)
            normal = normal * 2.0 - 1.0
            normal = normal / (np.linalg.norm(normal, axis=0) + 1e-15)[None]
            normals.append(normal)

            # align from left to right
            normal_left = normals[0]

            for i, normal_right in enumerate(normals[1:-1]):
                s1 = int(
                    step * (i + 1)
                )  # start index of normal_left of the overlapped part
                s2 = 0  # start index of normal_right of the overlapped part
                e2 = normal_left.shape[2] - int(
                    step * (i + 1)
                )  # end index of normal_right of the overlapped part
                normal_left = align_normal_x(
                    normal_left, normal_right, s1, normal_left.shape[2], s2, e2
                )
            # process the last column
            normal_left = align_normal_x(
                normal_left,
                normals[-1],
                W - image_size,
                normal_left.shape[2],
                0,
                image_size - W + normal_left.shape[2],
            )
            normals_row.append(normal_left)

        # load last row normal patches, need to process separately
        normals = []
        for i in range(x):
            normal_path = os.path.join(
                normal_patches_folder, "%06s_%02d_%02d.npy" % (image_name, y, i)
            )
            normal = np.load(normal_path)
            normal = normal * 2.0 - 1.0
            normal = normal / (np.linalg.norm(normal, axis=0) + 1e-15)[None]
            normals.append(normal)
        normal_path = os.path.join(
            normal_patches_folder, "%06s_%02d_%02d.npy" % (image_name, y, x)
        )
        normal = np.load(normal_path)
        normal = normal * 2.0 - 1.0
        normal = normal / (np.linalg.norm(normal, axis=0) + 1e-15)[None]
        normals.append(normal)

        # align from left to right
        normal_left = normals[0]
        for i, normal_right in enumerate(normals[1:-1]):
            s1 = int(step * (i + 1))
            s2 = 0
            e2 = normal_left.shape[2] - int(step * (i + 1))
            normal_left = align_normal_x(
                normal_left, normal_right, s1, normal_left.shape[2], s2, e2
            )
        # process the last column
        normal_left = align_normal_x(
            normal_left,
            normals[-1],
            W - image_size,
            normal_left.shape[2],
            0,
            image_size - W + normal_left.shape[2],
        )
        normals_row.append(normal_left)

        normal_top = normals_row[0]
        # align normal maps from top to down
        for i, normal_bottom in enumerate(normals_row[1:-1]):
            s1 = int(step * (i + 1))
            s2 = 0
            e2 = normal_top.shape[1] - int(step * (i + 1))
            normal_top = align_normal_y(
                normal_top, normal_bottom, s1, normal_top.shape[1], s2, e2
            )
        # align the last row

        normal_top = align_normal_y(
            normal_top,
            normals_row[-1],
            H - image_size,
            normal_top.shape[1],
            0,
            image_size - H + normal_top.shape[1],
        )

        # align to middle part
        mid_file = os.path.join(normal_patches_folder, "%06s_mid.npy" % (image_name))
        mid_normal = np.load(mid_file)
        mid_normal = mid_normal * 2.0 - 1.0
        mid_normal = mid_normal / (np.linalg.norm(mid_normal, axis=0) + 1e-15)[None]

        # align the whole normal image middle part to the direction of the middle part of the cropped image
        R = best_fit_transform(
            normal_top[
                :,
                H // 2 - int(image_size / 2) : H // 2 + int(image_size / 2),
                W // 2 - int(image_size / 2) : W // 2 + int(image_size / 2),
            ]
            .reshape(3, -1)
            .T,
            mid_normal.reshape(3, -1).T,
        )
        normal_top = (R @ normal_top.reshape(3, -1)).reshape(normal_top.shape)

        normal_top = (normal_top + 1.0) / 2.0
        save_normal(
            normal_top.transpose(1, 2, 0),
            os.path.join(out_path_for_training, "%06s.png" % (image_name)),
            verbose=False,
            format="png",
        )

        save_normal(
            normal_top.transpose(1, 2, 0),
            os.path.join(out_path_for_training, "%06s.npy" % (image_name)),
            verbose=False,
            format="npy",
        )


if __name__ == "__main__":
    tyro.cli(NormalsFromPretrained).main()
