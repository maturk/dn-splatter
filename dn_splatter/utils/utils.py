"""Some useful util functions"""

import os
import random
from pathlib import Path
from typing import List, Literal, Optional, Union

import cv2
import numpy as np
import open3d as o3d
import torch
from dn_splatter.utils.camera_utils import OPENGL_TO_OPENCV, get_means3d_backproj
from natsort import natsorted
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import resize
from tqdm import tqdm

from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.models.base_model import Model
from nerfstudio.process_data.process_data_utils import (
    convert_video_to_images,
    get_num_frames_in_video,
)
from nerfstudio.utils import colormaps
from nerfstudio.utils.rich_utils import CONSOLE

# Depth Scale Factor m to mm
SCALE_FACTOR = 0.001


def video_to_frames(
    video_path: Path, image_dir: Path("./data/frames"), force: bool = False
):
    """Extract frames from video, requires nerfstudio install"""
    is_empty = False

    if not image_dir.exists():
        is_empty = True
    else:
        is_empty = not any(image_dir.iterdir())

    if is_empty or force:
        num_frames_target = get_num_frames_in_video(video=video_path)
        summary_log, num_extracted_frames = convert_video_to_images(
            video_path,
            image_dir=image_dir,
            num_frames_target=num_frames_target,
            num_downscales=0,
            verbose=True,
            image_prefix="frame_",
            keep_image_dir=False,
        )
        assert num_extracted_frames == num_frames_target


def get_filename_list(image_dir: Path, ends_with: Optional[str] = None) -> List:
    """List directory and save filenames

    Returns:
        image_filenames
    """
    image_filenames = os.listdir(image_dir)
    if ends_with is not None:
        image_filenames = [
            image_dir / name
            for name in image_filenames
            if name.lower().endswith(ends_with)
        ]
    else:
        image_filenames = [image_dir / name for name in image_filenames]
    image_filenames = natsorted(image_filenames)
    return image_filenames


def image_path_to_tensor(
    image_path: Path, size: Optional[tuple] = None, black_and_white=False
) -> Tensor:
    """Convert image from path to tensor

    Returns:
        image: Tensor
    """
    img = Image.open(image_path)
    if black_and_white:
        img = img.convert("1")
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    if size:
        img_tensor = resize(
            img_tensor.permute(2, 0, 1), size=size, antialias=None
        ).permute(1, 2, 0)
    return img_tensor


def depth_path_to_tensor(
    depth_path: Path, scale_factor: float = SCALE_FACTOR, return_color=False
) -> Tensor:
    """Load depth image in either .npy or .png format and return tensor

    Args:
        depth_path: Path
        scale_factor: float
        return_color: bool
    Returns:
        depth tensor and optionally colored depth tensor
    """
    if depth_path.suffix == ".png":
        depth = cv2.imread(str(depth_path.absolute()), cv2.IMREAD_ANYDEPTH)
    elif depth_path.suffix == ".npy":
        depth = np.load(depth_path, allow_pickle=True)
        if len(depth.shape) == 3:
            depth = depth[..., 0]
    else:
        raise Exception(f"Format is not supported {depth_path.suffix}")
    depth = depth * scale_factor
    depth = depth.astype(np.float32)
    depth = torch.from_numpy(depth).unsqueeze(-1)
    if not return_color:
        return depth
    else:
        depth_color = colormaps.apply_depth_colormap(depth)
        return depth, depth_color  # type: ignore


def save_img(image, image_path, verbose=True) -> None:
    """helper to save images

    Args:
        image: image to save (numpy, Tensor)
        image_path: path to save
        verbose: whether to print save path

    Returns:
        None
    """
    if image.shape[-1] == 1 and torch.is_tensor(image):
        image = image.repeat(1, 1, 3)
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy() * 255
        image = image.astype(np.uint8)
    if not Path(os.path.dirname(image_path)).exists():
        Path(os.path.dirname(image_path)).mkdir(parents=True)
    im = Image.fromarray(image)
    if verbose:
        print("saving to: ", os.getcwd() + "/" + image_path)
    im.save(image_path)


def save_depth(depth, depth_path, verbose=True, scale_factor=SCALE_FACTOR) -> None:
    """helper to save metric depths

    Args:
        depth: image to save (numpy, Tensor)
        depth_path: path to save
        verbose: whether to print save path
        scale_factor: depth metric scaling factor

    Returns:
        None
    """
    if torch.is_tensor(depth):
        depth = depth.float() / scale_factor
        depth = depth.detach().cpu().numpy()
    else:
        depth = depth / scale_factor
    if not Path(os.path.dirname(depth_path)).exists():
        Path(os.path.dirname(depth_path)).mkdir(parents=True)
    if verbose:
        print("saving to: ", depth_path)
    np.save(depth_path, depth)


def save_normal(
    normal: Union[np.array, Tensor],
    normal_path: Path,
    verbose: bool = True,
    format: Literal["png", "npy"] = "png",
) -> None:
    """helper to save normal

    Args:
        normal: image to save (numpy, Tensor)
        normal_path: path to save
        verbose: whether to print save path

    Returns:
        None
    """
    if torch.is_tensor(normal):
        normal = normal.float()
        normal = normal.detach().cpu().numpy()
    else:
        normal = normal
    if not Path(os.path.dirname(normal_path)).exists():
        Path(os.path.dirname(normal_path)).mkdir(parents=True)
    if verbose:
        print("saving to: ", normal_path)
    if format == "npy":
        np.save(normal_path, normal)
    elif format == "png":
        normal = normal * 255
        normal = normal.astype(np.uint8)
        nm = Image.fromarray(normal)
        nm.save(normal_path)


def gs_get_point_clouds(
    eval_data: Optional[InputDataset],
    train_data: Optional[InputDataset],
    model: Model,
    render_output_path: Path,
    num_points: int = 1_000_000,
) -> None:
    """Saves pointcloud rendered from a model using predicted eval/train depths

    Args:
        eval_data: eval input dataset
        train_data: train input dataset
        model: model object
        render_output_path: path to render results to
        num_points: number of points to extract in pd

    Returns:
        None
    """
    CONSOLE.print("[bold green] Generating pointcloud ...")
    H, W = (
        int(train_data.cameras[0].height.item()),
        int(train_data.cameras[0].width.item()),
    )
    pixels_per_frame = W * H
    samples_per_frame = (num_points + (len(train_data) + len(eval_data))) // (
        len(train_data) + len(eval_data)
    )
    points = []
    colors = []
    if len(train_data) > 0:
        for image_idx in tqdm(range(len(train_data)), leave=False):
            camera = train_data.cameras[image_idx : image_idx + 1].to(model.device)
            outputs = model.get_outputs(camera)
            rgb_out, depth_out = outputs["rgb"], outputs["depth"]

            c2w = torch.concatenate(
                [
                    camera.camera_to_worlds,
                    torch.tensor([[[0, 0, 0, 1]]]).to(model.device),
                ],
                dim=1,
            )
            # convert from opengl to opencv
            c2w = torch.matmul(
                c2w, torch.from_numpy(OPENGL_TO_OPENCV).float().to(model.device)
            )
            # backproject
            point, _ = get_means3d_backproj(
                depths=depth_out.float(),
                fx=camera.fx,
                fy=camera.fy,
                cx=camera.cx,
                cy=camera.cy,
                img_size=(W, H),
                c2w=c2w.float(),
                device=model.device,
            )
            point = point.squeeze(0)

            # sample pixels for this frame
            indices = random.sample(range(pixels_per_frame), samples_per_frame)
            mask = torch.tensor(indices, device=model.device)

            color = rgb_out.view(-1, 3)[mask].detach().cpu().numpy()
            point = point[mask].detach().cpu().numpy()
            points.append(point)
            colors.append(color)

    if len(eval_data) > 0:
        for image_idx in tqdm(range(len(eval_data)), leave=False):
            camera = eval_data.cameras[image_idx : image_idx + 1].to(model.device)
            outputs = model.get_outputs(camera)
            rgb_out, depth_out = outputs["rgb"], outputs["depth"]

            c2w = torch.concatenate(
                [
                    camera.camera_to_worlds,
                    torch.tensor([[[0, 0, 0, 1]]]).to(model.device),
                ],
                dim=1,
            )
            # convert from opengl to opencv
            c2w = torch.matmul(
                c2w, torch.from_numpy(OPENGL_TO_OPENCV).float().to(model.device)
            )
            # backproject
            point, _ = get_means3d_backproj(
                depths=depth_out.float(),
                fx=camera.fx,
                fy=camera.fy,
                cx=camera.cx,
                cy=camera.cy,
                img_size=(W, H),
                c2w=c2w.float(),
                device=model.device,
            )
            point = point.squeeze(0)

            # sample pixels for this frame
            indices = random.sample(range(pixels_per_frame), samples_per_frame)
            mask = torch.tensor(indices, device=model.device)

            color = rgb_out.view(-1, 3)[mask].detach().cpu().numpy()
            point = point[mask].detach().cpu().numpy()
            points.append(point)
            colors.append(color)

    points = np.vstack(points)
    colors = np.vstack(colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    CONSOLE.print(
        f"[bold yellow]Saved pointcloud to {os.getcwd() + render_output_path}'/pointcloud.ply'"
    )
    o3d.io.write_point_cloud(os.getcwd() + f"{render_output_path}/pointcloud.ply", pcd)
    return (points, colors)


def gs_render_dataset_images(
    train_cache: List,
    eval_cache: List,
    train_dataset,
    eval_dataset,
    model,
    render_output_path: Path,
    mushroom=False,
    save_train_images=False,
) -> None:
    """Render and save all train/eval images of gs model to directory

    Args:
        train_cache: list of cached train images
        eval_cache: list of cached eval images
        eval_data: eval input dataset
        train_data: train input dataset
        model: model object
        render_output_path: path to render results to
        mushroom: if dataset is Mushroom dataset or not
        save_train_images: whether to save train images or not

    Returns:
        None
    """
    CONSOLE.print(f"[bold yellow]Saving results to {render_output_path}")
    if len(eval_cache) > 0:
        for i, _ in tqdm(enumerate(range(len(eval_cache))), leave=False):
            image_idx = i
            data = eval_cache[image_idx]
            # ground truth data
            gt_img = data["image"]
            if "sensor_depth" in data:
                depth_gt = data["sensor_depth"]
                depth_gt_color = colormaps.apply_depth_colormap(data["sensor_depth"])
            else:
                depth_gt = None
                depth_gt_color = None
            normal_gt = data["normal"] if "normal" in data else None
            camera = eval_dataset.cameras[image_idx : image_idx + 1].to(model.device)

            # save the image with its original name for easy comparison
            if mushroom:
                seq_name = Path(eval_dataset.image_filenames[image_idx])
                image_name = f"{seq_name.parts[-3]}_{seq_name.stem}"
            else:
                image_name = Path(eval_dataset.image_filenames[image_idx]).stem
            outputs = model.get_outputs(camera)
            rgb_out, depth_out, normal_out, surface_normal = (
                outputs["rgb"],
                outputs["depth"],
                outputs["normal"],
                outputs["surface_normal"],
            )

            depth_color = colormaps.apply_depth_colormap(depth_out)
            depth = depth_out.detach().cpu().numpy()
            save_outputs_helper(
                rgb_out,
                gt_img,
                depth_color,
                depth_gt_color,
                depth_gt,
                depth,
                normal_gt if normal_gt is not None else None,
                surface_normal if surface_normal is not None else None,
                render_output_path,
                image_name,
            )

    if save_train_images and len(train_cache) > 0:
        for i, _ in tqdm(enumerate(range(len(train_cache))), leave=False):
            image_idx = i
            data = train_cache[image_idx]
            # ground truth data
            gt_img = data["image"]
            if "sensor_depth" in data:
                depth_gt = data["sensor_depth"]
                depth_gt_color = colormaps.apply_depth_colormap(data["sensor_depth"])
            else:
                depth_gt = None
                depth_gt_color = None
            normal_gt = data["normal"] if "normal" in data else None
            camera = train_dataset.cameras[image_idx : image_idx + 1].to(model.device)

            # save the image with its original name for easy comparison
            if mushroom:
                seq_name = Path(train_dataset.image_filenames[image_idx])
                image_name = f"{seq_name.parts[-3]}_{seq_name.stem}"
            else:
                image_name = Path(train_dataset.image_filenames[image_idx]).stem
            outputs = model.get_outputs(camera)
            rgb_out, depth_out, normal_out, surface_normal = (
                outputs["rgb"],
                outputs["depth"],
                outputs["normal"],
                outputs["surface_normal"],
            )

            depth_color = colormaps.apply_depth_colormap(depth_out)
            depth = depth_out.detach().cpu().numpy()
            save_outputs_helper(
                rgb_out,
                gt_img,
                depth_color,
                depth_gt_color,
                depth_gt,
                depth,
                normal_gt if normal_gt is not None else None,
                surface_normal if surface_normal is not None else None,
                render_output_path,
                image_name,
            )


def ns_render_dataset_images(
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    train_dataset: InputDataset,
    eval_dataset: InputDataset,
    model: Model,
    render_output_path: Path,
    mushroom=False,
    save_train_images=False,
) -> None:
    """render and save all train/eval images of nerfstudio model to directory

    Args:
        train_dataloader: train dataloader
        eval_dataloader: eval dataloader
        eval_data: eval input dataset
        train_data: train input dataset
        model: model object
        render_output_path: path to render results to
        mushroom: whether the dataset is Mushroom dataset or not
        save_train_images:  whether to save train images or not

    Returns:
        None
    """
    CONSOLE.print(f"[bold yellow]Saving results to {render_output_path}")
    if len(eval_dataloader) > 0:
        for image_idx, (camera, batch) in tqdm(enumerate(eval_dataloader)):
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(camera)
            # ground truth data
            data = batch.copy()
            gt_img = data["image"]
            if "sensor_depth" in data:
                depth_gt = data["sensor_depth"]
                depth_gt_color = colormaps.apply_depth_colormap(data["sensor_depth"])
            else:
                depth_gt = None
                depth_gt_color = None
            normal_gt = data["normal"] if "normal" in data else None
            # save the image with its original name for easy comparison
            if mushroom:
                seq_name = Path(eval_dataset.image_filenames[image_idx])
                image_name = f"{seq_name.parts[-3]}_{seq_name.stem}"
            else:
                image_name = Path(eval_dataset.image_filenames[image_idx]).stem

            rgb_out, depth_out, normal_out, surface_normal = (
                outputs["rgb"],
                outputs["depth"],
                outputs["normal"] if "normal" in outputs else None,
                outputs["surface_normal"] if "surface_normal" in outputs else None,
            )
            depth_color = colormaps.apply_depth_colormap(depth_out)
            depth = depth_out.detach().cpu().numpy()
            save_outputs_helper(
                rgb_out,
                gt_img,
                depth_color,
                depth_gt_color,
                depth_gt,
                depth,
                normal_gt,
                surface_normal,
                render_output_path,
                image_name,
            )

    if save_train_images and len(train_dataloader) > 0:
        for image_idx, (camera, batch) in tqdm(enumerate(train_dataloader)):
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(camera)
            # ground truth data
            data = batch.copy()
            gt_img = data["image"]
            if "sensor_depth" in data:
                depth_gt = data["sensor_depth"]
                depth_gt_color = colormaps.apply_depth_colormap(data["sensor_depth"])
            else:
                depth_gt = None
                depth_gt_color = None
            normal_gt = data["normal"] if "normal" in data else None
            # save the image with its original name for easy comparison
            if mushroom:
                seq_name = Path(train_dataset.image_filenames[image_idx])
                image_name = f"{seq_name.parts[-3]}_{seq_name.stem}"
            else:
                image_name = Path(train_dataset.image_filenames[image_idx]).stem

            rgb_out, depth_out, normal_out = (
                outputs["rgb"],
                outputs["depth"],
                outputs["normal"] if "normal" in outputs else None,
            )
            depth_color = colormaps.apply_depth_colormap(depth_out)
            depth = depth_out.detach().cpu().numpy()
            save_outputs_helper(
                rgb_out,
                gt_img,
                depth_color,
                depth_gt_color,
                depth_gt,
                depth,
                normal_gt,
                normal_out,
                render_output_path,
                image_name,
            )


def save_outputs_helper(
    rgb_out: Optional[Tensor] = None,
    gt_img: Optional[Tensor] = None,
    depth_color: Optional[Tensor] = None,
    depth_gt_color: Optional[Tensor] = None,
    depth_gt: Optional[Tensor] = None,
    depth: Optional[Tensor] = None,
    normal_gt: Optional[Tensor] = None,
    normal: Optional[Tensor] = None,
    render_output_path: Optional[Path] = None,
    image_name: Optional[str] = None,
) -> None:
    """Helper to save model rgb/depth/gt outputs to disk

    Args:
        rgb_out: rgb image
        gt_img: gt rgb image
        depth_color: colored depth image
        depth_gt_color: gt colored depth image
        depth_gt: gt depth map
        depth: depth map
        render_output_path: save directory path
        image_name: stem of save name

    Returns:
        None
    """
    if image_name is None:
        image_name = ""

    if rgb_out is not None and gt_img is not None:
        save_img(
            rgb_out,
            os.getcwd() + f"/{render_output_path}/pred/rgb/{image_name}.png",
            False,
        )
        save_img(
            gt_img,
            os.getcwd() + f"/{render_output_path}/gt/rgb/{image_name}.png",
            False,
        )
    if depth_color is not None:
        save_img(
            depth_color,
            os.getcwd()
            + f"/{render_output_path}/pred/depth/colorised/{image_name}.png",
            False,
        )
    if depth_gt_color is not None:
        save_img(
            depth_gt_color,
            os.getcwd() + f"/{render_output_path}/gt/depth/colorised/{image_name}.png",
            False,
        )
    if depth_gt is not None:
        # save metric depths
        save_depth(
            depth_gt,
            os.getcwd() + f"/{render_output_path}/gt/depth/raw/{image_name}.npy",
            False,
        )
    if depth is not None:
        save_depth(
            depth,
            os.getcwd() + f"/{render_output_path}/pred/depth/raw/{image_name}.npy",
            False,
        )

    if normal is not None:
        save_normal(
            normal,
            os.getcwd() + f"/{render_output_path}/pred/normal/{image_name}.png",
            verbose=False,
        )

    if normal_gt is not None:
        save_normal(
            normal_gt,
            os.getcwd() + f"/{render_output_path}/gt/normal/{image_name}.png",
            verbose=False,
        )
