#!/usr/bin/env python
"""eval.py

run with: python python dn_splatter/eval.py --data [PATH_TO_DATA]

options : 
        --no-eval-rgb
        --no-eval-depth

eval-faro option is used for reference faro scanner projected depth maps
"""
import json
import os
from pathlib import Path

import cv2
import torch
import torchvision.transforms.functional as F
import tyro
from rich.console import Console
from rich.progress import track
from torchmetrics.functional import mean_squared_error
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torch import nn
import numpy as np
from torch import Tensor


# from dn_splatter.metrics import DepthMetrics
# from dn_splatter.utils.utils import depth_path_to_tensor

CONSOLE = Console(width=120)
BATCH_SIZE = 20


def depth_path_to_tensor(
    depth_path: Path, scale_factor: float = 1, return_color=False
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


class DepthMetrics(nn.Module):
    """Computation of error metrics between predicted and ground truth depths

    from:
        https://arxiv.org/abs/1806.01260

    Returns:
        abs_rel: normalized avg absolute realtive error
        sqrt_rel: normalized square-root of absolute error
        rmse: root mean square error
        rmse_log: root mean square error in log space
        a1, a2, a3: metrics
    """

    def __init__(self, tolerance: float = 0.1, **kwargs):
        self.tolerance = tolerance
        super().__init__()

    @torch.no_grad()
    def forward(self, pred, gt):
        mask = gt > self.tolerance

        thresh = torch.max((gt[mask] / pred[mask]), (pred[mask] / gt[mask]))
        a1 = (thresh < 1.25).float().mean()
        a2 = (thresh < 1.25**2).float().mean()
        a3 = (thresh < 1.25**3).float().mean()
        rmse = (gt[mask] - pred[mask]) ** 2
        rmse = torch.sqrt(rmse.mean())

        rmse_log = (torch.log(gt[mask]) - torch.log(pred[mask])) ** 2
        rmse_log[rmse_log == float("inf")] = float("nan")
        rmse_log = torch.sqrt(rmse_log).nanmean()

        abs_rel = torch.abs(gt - pred)[mask] / gt[mask]
        abs_rel = abs_rel.mean()
        sq_rel = (gt - pred)[mask] ** 2 / gt[mask]
        sq_rel = sq_rel.mean()

        return (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3)


@torch.no_grad()
def rgb_eval(data: Path):
    render_path = data / Path("rgb")  # os.path.join(args.data, "/rgb")
    gt_path = data / Path("gt")  # os.path.join(args.data, "gt", "rgb")

    # classify f with begin fix: long_frame
    image_list = [f for f in os.listdir(render_path) if (f.endswith(".png"))]
    long_image_list = [f for f in os.listdir(gt_path) if "long" in f]
    short_image_list = [f for f in os.listdir(gt_path) if "short" in f]
    long_image_list = sorted(
        long_image_list, key=lambda x: int((x.split(".")[0]).split("_")[-1])
    )
    short_image_list = sorted(
        short_image_list, key=lambda x: int((x.split(".")[0]).split("_")[-1])
    )

    mse = mean_squared_error
    psnr = PeakSignalNoiseRatio(data_range=1.0)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=11)
    lpips = LearnedPerceptualImagePatchSimilarity()

    num_frames = len(long_image_list)

    psnr_score_batch = []
    ssim_score_batch = []
    mse_score_batch = []
    lpips_score_batch = []

    CONSOLE.print(
        f"[bold green]Batchifying and evaluating a total of {num_frames} rgb frames"
    )

    for batch_index in track(range(0, num_frames, BATCH_SIZE)):
        CONSOLE.print(
            f"[bold yellow]Evaluating batch {batch_index // BATCH_SIZE} / {num_frames//BATCH_SIZE}"
        )
        batch_frames = long_image_list[batch_index : batch_index + BATCH_SIZE]
        predicted_rgb = []
        gt_rgb = []

        for i in batch_frames:
            render_img = cv2.imread(os.path.join(render_path, i)) / 255
            origin_img = cv2.imread(os.path.join(gt_path, i)) / 255
            origin_img = F.to_tensor(origin_img).to(torch.float32)
            render_img = F.to_tensor(render_img).to(torch.float32)

            predicted_rgb.append(render_img)
            gt_rgb.append(origin_img)

        predicted_image = torch.stack(predicted_rgb, 0)
        gt_image = torch.stack(gt_rgb, 0)

        mse_score = mse(predicted_image, gt_image)
        mse_score_batch.append(mse_score)
        psnr_score = psnr(predicted_image, gt_image)
        psnr_score_batch.append(psnr_score)
        ssim_score = ssim(predicted_image, gt_image)
        ssim_score_batch.append(ssim_score)
        lpips_score = lpips(predicted_image, gt_image)
        lpips_score_batch.append(lpips_score)

    mean_scores = {
        "long_mse": float(torch.stack(mse_score_batch).mean().item()),
        "long_psnr": float(torch.stack(psnr_score_batch).mean().item()),
        "long_ssim": float(torch.stack(ssim_score_batch).mean().item()),
        "long_lpips": float(torch.stack(lpips_score_batch).mean().item()),
    }

    num_frames = len(short_image_list)
    psnr_score_batch = []
    ssim_score_batch = []
    mse_score_batch = []
    lpips_score_batch = []
    for batch_index in track(range(0, num_frames, BATCH_SIZE)):
        CONSOLE.print(
            f"[bold yellow]Evaluating batch {batch_index // BATCH_SIZE} / {num_frames//BATCH_SIZE}"
        )
        batch_frames = short_image_list[batch_index : batch_index + BATCH_SIZE]
        predicted_rgb = []
        gt_rgb = []

        for i in batch_frames:
            render_img = cv2.imread(os.path.join(render_path, i)) / 255
            origin_img = cv2.imread(os.path.join(gt_path, i)) / 255
            origin_img = F.to_tensor(origin_img).to(torch.float32)
            render_img = F.to_tensor(render_img).to(torch.float32)

            predicted_rgb.append(render_img)
            gt_rgb.append(origin_img)

        predicted_image = torch.stack(predicted_rgb, 0)
        gt_image = torch.stack(gt_rgb, 0)

        mse_score = mse(predicted_image, gt_image)
        mse_score_batch.append(mse_score)
        psnr_score = psnr(predicted_image, gt_image)
        psnr_score_batch.append(psnr_score)
        ssim_score = ssim(predicted_image, gt_image)
        ssim_score_batch.append(ssim_score)
        lpips_score = lpips(predicted_image, gt_image)
        lpips_score_batch.append(lpips_score)

    mean_scores.update(
        {
            "short_mse": float(torch.stack(mse_score_batch).mean().item()),
            "short_psnr": float(torch.stack(psnr_score_batch).mean().item()),
            "short_ssim": float(torch.stack(ssim_score_batch).mean().item()),
            "short_lpips": float(torch.stack(lpips_score_batch).mean().item()),
        }
    )

    with open(os.path.join(data, "metrics.json"), "w") as outFile:
        print(f"Saving results to {os.path.join(data, 'metrics.json')}")
        json.dump(mean_scores, outFile, indent=2)


@torch.no_grad()
def depth_eval(data: Path):
    depth_metrics = DepthMetrics()

    render_path = data / Path("depth")  # os.path.join(args.data, "/rgb")
    gt_path = data / Path("gt_depth")  # os.path.join(args.data, "gt", "rgb")

    depth_list = [f for f in os.listdir(render_path) if f.endswith(".npy")]
    long_depth_list = [f for f in os.listdir(gt_path) if "long" in f]
    short_depth_list = [f for f in os.listdir(gt_path) if "short" in f]
    long_depth_list = sorted(
        long_depth_list, key=lambda x: int(x.split(".")[0].split("_")[-1])
    )
    short_depth_list = sorted(
        short_depth_list, key=lambda x: int(x.split(".")[0].split("_")[-1])
    )

    mse = mean_squared_error

    num_frames = len(long_depth_list)

    mse_score_batch = []
    abs_rel_score_batch = []
    sq_rel_score_batch = []
    rmse_score_batch = []
    rmse_log_score_batch = []
    a1_score_batch = []
    a2_score_batch = []
    a3_score_batch = []
    CONSOLE.print(
        f"[bold green]Batchifying and evaluating a total of {num_frames} depth frames"
    )
    for batch_index in track(range(0, num_frames, BATCH_SIZE)):
        CONSOLE.print(
            f"[bold yellow]Evaluating batch {batch_index // BATCH_SIZE} / {num_frames//BATCH_SIZE}"
        )
        batch_frames = long_depth_list[batch_index : batch_index + BATCH_SIZE]
        predicted_depth = []
        gt_depth = []

        for i in batch_frames:
            render_img = np.load(Path(os.path.join(render_path, i)), allow_pickle=True)
            render_img = render_img.astype(np.float32)
            render_img = torch.from_numpy(render_img)

            origin_img = np.load(Path(os.path.join(gt_path, i)), allow_pickle=True)
            origin_img = origin_img.astype(np.float32)
            origin_img = torch.from_numpy(origin_img)

            if origin_img.shape[-2:] != render_img.shape[-2:]:
                render_img = F.resize(
                    render_img, size=origin_img.shape[-2:], antialias=None
                )
            predicted_depth.append(render_img)
            gt_depth.append(origin_img)

        predicted_depth = torch.stack(predicted_depth, 0)
        gt_depth = torch.stack(gt_depth, 0)

        mse_score = mse(predicted_depth, gt_depth)
        mse_score_batch.append(mse_score)
        (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3) = depth_metrics(
            predicted_depth, gt_depth
        )
        abs_rel_score_batch.append(abs_rel)
        sq_rel_score_batch.append(sq_rel)
        rmse_score_batch.append(rmse)
        rmse_log_score_batch.append(rmse_log)
        a1_score_batch.append(a1)
        a2_score_batch.append(a2)
        a3_score_batch.append(a3)

    mean_scores = {
        "long_mse": float(torch.stack(mse_score_batch).mean().item()),
        "long_abs_rel": float(torch.stack(abs_rel_score_batch).mean().item()),
        "long_sq_rel": float(torch.stack(sq_rel_score_batch).mean().item()),
        "long_rmse": float(torch.stack(rmse_score_batch).mean().item()),
        "long_rmse_log": float(torch.stack(rmse_log_score_batch).mean().item()),
        "long_a1": float(torch.stack(a1_score_batch).mean().item()),
        "long_a2": float(torch.stack(a2_score_batch).mean().item()),
        "long_a3": float(torch.stack(a3_score_batch).mean().item()),
    }

    num_frames = len(short_depth_list)

    mse_score_batch = []
    abs_rel_score_batch = []
    sq_rel_score_batch = []
    rmse_score_batch = []
    rmse_log_score_batch = []
    a1_score_batch = []
    a2_score_batch = []
    a3_score_batch = []

    for batch_index in track(range(0, num_frames, BATCH_SIZE)):
        CONSOLE.print(
            f"[bold yellow]Evaluating batch {batch_index // BATCH_SIZE} / {num_frames//BATCH_SIZE}"
        )
        batch_frames = short_depth_list[batch_index : batch_index + BATCH_SIZE]
        predicted_depth = []
        gt_depth = []

        for i in batch_frames:
            render_img = np.load(Path(os.path.join(render_path, i)), allow_pickle=True)
            render_img = render_img.astype(np.float32)
            render_img = torch.from_numpy(render_img)

            origin_img = np.load(Path(os.path.join(gt_path, i)), allow_pickle=True)
            origin_img = origin_img.astype(np.float32)
            origin_img = torch.from_numpy(origin_img)

            # if origin_img.shape[-2:] != render_img.shape[-2:]:
            #     render_img = F.resize(
            #         render_img, size=origin_img.shape[-2:], antialias=None
            #     )
            predicted_depth.append(render_img)
            gt_depth.append(origin_img)

        predicted_depth = torch.stack(predicted_depth, 0)
        gt_depth = torch.stack(gt_depth, 0)

        mse_score = mse(predicted_depth, gt_depth)
        mse_score_batch.append(mse_score)
        (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3) = depth_metrics(
            predicted_depth, gt_depth
        )
        abs_rel_score_batch.append(abs_rel)
        sq_rel_score_batch.append(sq_rel)
        rmse_score_batch.append(rmse)
        rmse_log_score_batch.append(rmse_log)
        a1_score_batch.append(a1)
        a2_score_batch.append(a2)
        a3_score_batch.append(a3)

    mean_scores.update(
        {
            "short_mse": float(torch.stack(mse_score_batch).mean().item()),
            "short_abs_rel": float(torch.stack(abs_rel_score_batch).mean().item()),
            "short_sq_rel": float(torch.stack(sq_rel_score_batch).mean().item()),
            "short_rmse": float(torch.stack(rmse_score_batch).mean().item()),
            "short_rmse_log": float(torch.stack(rmse_log_score_batch).mean().item()),
            "short_a1": float(torch.stack(a1_score_batch).mean().item()),
            "short_a2": float(torch.stack(a2_score_batch).mean().item()),
            "short_a3": float(torch.stack(a3_score_batch).mean().item()),
        }
    )

    # print(list(mean_scores.keys()))
    # print(list(mean_scores.values()))
    with open(data / "metrics.json", "w") as outFile:
        print(f"Saving results to {os.path.join(render_path, 'metrics.json')}")
        json.dump(mean_scores, outFile, indent=2)


def depth_eval_faro(data: Path, path_to_faro: Path):
    depth_metrics = DepthMetrics()

    render_path = data / Path("depth/raw/")
    gt_path = path_to_faro

    depth_list = [f for f in os.listdir(render_path) if f.endswith(".png")]
    depth_list = sorted(depth_list, key=lambda x: int(x.split(".")[0].split("_")[-1]))

    mse = mean_squared_error

    num_frames = len(depth_list)

    mse_score_batch = []
    abs_rel_score_batch = []
    sq_rel_score_batch = []
    rmse_score_batch = []
    rmse_log_score_batch = []
    a1_score_batch = []
    a2_score_batch = []
    a3_score_batch = []
    CONSOLE.print(
        f"[bold green]Batchifying and evaluating a total of {num_frames} depth frames"
    )

    for batch_index in track(range(0, num_frames, BATCH_SIZE)):
        CONSOLE.print(
            f"[bold yellow]Evaluating batch {batch_index // BATCH_SIZE} / {num_frames//BATCH_SIZE}"
        )
        batch_frames = depth_list[batch_index : batch_index + BATCH_SIZE]
        predicted_depth = []
        gt_depth = []
        for i in batch_frames:
            render_img = depth_path_to_tensor(
                Path(os.path.join(render_path, i))
            ).permute(2, 0, 1)

            if not Path(os.path.join(gt_path, i)).exists():
                print("could not find frame ", i, " skipping it...")
                continue
            origin_img = depth_path_to_tensor(Path(os.path.join(gt_path, i))).permute(
                2, 0, 1
            )
            if origin_img.shape[-2:] != render_img.shape[-2:]:
                render_img = F.resize(
                    render_img, size=origin_img.shape[-2:], antialias=None
                )
            predicted_depth.append(render_img)
            gt_depth.append(origin_img)

        predicted_depth = torch.stack(predicted_depth, 0)
        gt_depth = torch.stack(gt_depth, 0)

        mse_score = mse(predicted_depth, gt_depth)
        mse_score_batch.append(mse_score)

        (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3) = depth_metrics(
            predicted_depth, gt_depth
        )
        abs_rel_score_batch.append(abs_rel)
        sq_rel_score_batch.append(sq_rel)
        rmse_score_batch.append(rmse)
        rmse_log_score_batch.append(rmse_log)
        a1_score_batch.append(a1)
        a2_score_batch.append(a2)
        a3_score_batch.append(a3)

    mean_scores = {
        "mse": float(torch.stack(mse_score_batch).mean().item()),
        "abs_rel": float(torch.stack(abs_rel_score_batch).mean().item()),
        "sq_rel": float(torch.stack(sq_rel_score_batch).mean().item()),
        "rmse": float(torch.stack(rmse_score_batch).mean().item()),
        "rmse_log": float(torch.stack(rmse_log_score_batch).mean().item()),
        "a1": float(torch.stack(a1_score_batch).mean().item()),
        "a2": float(torch.stack(a2_score_batch).mean().item()),
        "a3": float(torch.stack(a3_score_batch).mean().item()),
    }
    print("faro scanner metrics")
    print(list(mean_scores.keys()))
    print(list(mean_scores.values()))


def main(
    data: Path,
    eval_rgb: bool = True,
    eval_depth: bool = False,
):
    if eval_rgb:
        rgb_eval(data)
    if eval_depth:
        depth_eval(data)


if __name__ == "__main__":
    tyro.cli(main)
