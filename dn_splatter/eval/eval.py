#!/usr/bin/env python
"""eval.py

run with: python python dn_splatter/eval.py --data [PATH_TO_DATA]

options : 
        --no-eval-rgb
        --no-eval-depth
        --eval-faro / --no-eval-faro 

eval-faro option is used for reference faro scanner projected depth maps
"""
import json
import os
from pathlib import Path
from typing import Optional

import cv2
import torch
import torchvision.transforms.functional as F
import tyro
from rich.console import Console
from rich.progress import track
from torchmetrics.functional import mean_squared_error
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from dn_splatter.metrics import DepthMetrics
from dn_splatter.utils.utils import depth_path_to_tensor

CONSOLE = Console(width=120)
BATCH_SIZE = 20


@torch.no_grad()
def rgb_eval(data: Path):
    render_path = data / Path("rgb")  # os.path.join(args.data, "/rgb")
    gt_path = data / Path("gt/rgb/")  # os.path.join(args.data, "gt", "rgb")

    image_list = [f for f in os.listdir(render_path) if f.endswith(".png")]
    image_list = sorted(image_list, key=lambda x: int(x.split(".")[0].split("_")[-1]))

    mse = mean_squared_error
    psnr = PeakSignalNoiseRatio(data_range=1.0)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=11)
    lpips = LearnedPerceptualImagePatchSimilarity()

    num_frames = len(image_list)

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
        batch_frames = image_list[batch_index : batch_index + BATCH_SIZE]
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
        "mse": float(torch.stack(mse_score_batch).mean().item()),
        "psnr": float(torch.stack(psnr_score_batch).mean().item()),
        "ssim": float(torch.stack(ssim_score_batch).mean().item()),
        "lpips": float(torch.stack(lpips_score_batch).mean().item()),
    }
    print(list(mean_scores.keys()))
    print(list(mean_scores.values()))

    with open(os.path.join(render_path, "metrics.json"), "w") as outFile:
        print(f"Saving results to {os.path.join(render_path, 'metrics.json')}")
        json.dump(mean_scores, outFile, indent=2)


@torch.no_grad()
def depth_eval(data: Path):
    depth_metrics = DepthMetrics()

    render_path = data / Path("depth/raw/")  # os.path.join(args.data, "/rgb")
    gt_path = data / Path("gt/depth/raw")  # os.path.join(args.data, "gt", "rgb")

    depth_list = [f for f in os.listdir(render_path) if f.endswith(".npy")]
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
    print(list(mean_scores.keys()))
    print(list(mean_scores.values()))
    with open(os.path.join(render_path, "metrics.json"), "w") as outFile:
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
    eval_depth: bool = True,
    eval_faro: bool = False,
    path_to_faro: Optional[Path] = None,
):
    if eval_rgb:
        rgb_eval(data)
    if eval_depth:
        depth_eval(data)
    if eval_faro:
        assert path_to_faro is not None, "need to specify faro path"
        depth_eval_faro(data, path_to_faro=path_to_faro)


if __name__ == "__main__":
    tyro.cli(main)
