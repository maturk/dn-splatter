#!/usr/bin/env python
"""eval.py

run with: python python dn_splatter/eval.py --data [PATH_TO_DATA]

options : 
        --eval-faro / --no-eval-faro 

eval-faro option is used for reference faro scanner projected depth maps
"""
import json
import os
from pathlib import Path
from typing import Optional

import torch
import torchvision.transforms.functional as F
import tyro
from dn_splatter.metrics import DepthMetrics
from dn_splatter.utils.utils import depth_path_to_tensor
from rich.console import Console
from rich.progress import track
from torchmetrics.functional import mean_squared_error

CONSOLE = Console(width=120)
BATCH_SIZE = 40


def depth_eval_faro(data: Path, path_to_faro: Path):
    transform_meta = data / "dataparser_transforms.json"
    meta = json.load(open(transform_meta, "r"))
    scale = meta["scale"]
    depth_metrics = DepthMetrics()

    render_path = data / Path("final_renders/pred/depth/raw/")
    gt_path = path_to_faro

    long_depth_list = [
        f for f in os.listdir(render_path) if f.endswith(".npy") and "long_capture" in f
    ]
    short_depth_list = [
        f
        for f in os.listdir(render_path)
        if f.endswith(".npy") and "short_capture" in f
    ]
    long_depth_list = sorted(
        long_depth_list, key=lambda x: int(x.split(".")[0].split("_")[-1])
    )
    short_depth_list = sorted(
        short_depth_list, key=lambda x: int(x.split(".")[0].split("_")[-1])
    )

    test_id_within = path_to_faro / Path("long_capture/test.txt")
    with open(test_id_within) as f:
        lines = f.readlines()
    i_eval_within = [num.split("\n")[0] for num in lines]

    mse = mean_squared_error

    long_num_frames = len(long_depth_list)
    short_num_frames = len(short_depth_list)

    mse_score_batch = []
    abs_rel_score_batch = []
    sq_rel_score_batch = []
    rmse_score_batch = []
    rmse_log_score_batch = []
    a1_score_batch = []
    a2_score_batch = []
    a3_score_batch = []

    CONSOLE.print(
        f"[bold green]Batchifying and evaluating a total of {long_num_frames + short_num_frames} depth frames"
    )

    def calculate_metrics(num_frames, depth_list, capture_mode):
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

                gt_folder = gt_path / Path(capture_mode) / Path("reference_depth")

                if "iphone" in str(gt_folder.parent):
                    name_list = i.split(".")[0].split("_")
                    depth_name = name_list[-2] + "_" + name_list[-1]
                    gt_name = gt_folder / Path(depth_name + ".png")
                elif "kinect" in str(gt_folder.parent):
                    depth_name = i.split(".")[0].split("_")[-1]
                    gt_name = gt_folder / Path(depth_name + ".png")
                if not gt_name.exists():
                    print("could not find frame ", gt_name, " skipping it...")
                    continue
                if capture_mode == "long_capture":
                    if depth_name not in i_eval_within:
                        continue

                origin_img = depth_path_to_tensor(gt_name).permute(2, 0, 1)

                if origin_img.shape[-2:] != render_img.shape[-2:]:
                    render_img = F.resize(
                        render_img, size=origin_img.shape[-2:], antialias=None
                    )

                render_img = render_img / scale
                predicted_depth.append(render_img)
                gt_depth.append(origin_img)

            predicted_depth = torch.stack(predicted_depth, 0)
            gt_depth = torch.stack(gt_depth, 0)

            mse_score = mse(predicted_depth, gt_depth)
            (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3) = depth_metrics(
                predicted_depth, gt_depth
            )

            mse_score_batch.append(mse_score)
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
        return mean_scores

    long_means_scores = calculate_metrics(
        long_num_frames, long_depth_list, "long_capture"
    )
    short_means_scores = calculate_metrics(
        short_num_frames, short_depth_list, "short_capture"
    )

    metrics_dict = {}
    for key in long_means_scores.keys():
        metrics_dict["within_faro_" + key] = long_means_scores[key]

    for key in short_means_scores.keys():
        metrics_dict["with_faro_" + key] = short_means_scores[key]

    return metrics_dict


def main(data: Path, eval_faro: bool = False, path_to_faro: Optional[Path] = None):
    if eval_faro:
        assert path_to_faro is not None, "need to specify faro path"
        depth_eval_faro(data, path_to_faro=path_to_faro)


if __name__ == "__main__":
    tyro.cli(main)
