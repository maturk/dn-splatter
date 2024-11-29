#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from operator import gt
import os
from re import L

import torch
from random import randint
from utils.loss_utils import (
    l1_loss,
    local_pearson_loss,
    pearson_depth_loss,
    ssim,
    mean_angular_error,
)
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import colormap
from utils.image_utils import gradient_map, dilate_edge, find_edges
from utils.image_utils import normal2curv

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    depth_supervision,
    normal_supervision,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_curvature_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        gt_image = viewpoint_cam.original_image.cuda()
        gt_depth = viewpoint_cam.gt_depth.cuda()
        h, w = gt_depth.shape[-2:]
        # translate the normal into world space
        gt_normal = viewpoint_cam.gt_normal.cuda()

        ssim_loss = ssim(image, gt_image)
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_loss)

        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg["rend_normal"]
        surf_normal = render_pkg["surf_normal"]
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # depth and normal l1
        depth = render_pkg["surf_depth"]
        confidence_map = viewpoint_cam.depth_confidence.cuda()
        normal_diff = mean_angular_error(surf_normal, gt_normal)
        normal_confidence = 1 - (normal_diff > 0.1).float()

        if depth_supervision:

            # confidence loss
            depth_mask = gt_depth > 0.1

            # l1 loss
            if iteration < opt.depth_mask_steps:
                Ldepth = (
                    l1_loss(depth[depth_mask], gt_depth[depth_mask]) * opt.lambda_depth
                )
            elif iteration >= opt.depth_mask_steps:

                gt_depth = torch.where(confidence_map > 0, gt_depth, 0).cuda()
                depth_mask = gt_depth > 0.1
                Ldepth = (
                    l1_loss(depth[depth_mask], gt_depth[depth_mask]) * opt.lambda_depth
                )

        else:
            Ldepth = 0.0

        lambda_normal_l1 = opt.lambda_normal_l1 if iteration > 7000 else 0.0
        if normal_supervision:
            normal_dilated_edges = find_edges(gt_normal)
            if iteration < opt.normal_mask_steps:

                normal_l1 = (
                    l1_loss(
                        surf_normal[~normal_dilated_edges],
                        gt_normal[~normal_dilated_edges],
                    )
                    * lambda_normal_l1
                )

            else:
                normal_confidence = (normal_confidence > 0).squeeze(0)
                normal_l1 = (
                    l1_loss(
                        surf_normal[:, normal_confidence],
                        gt_normal[:, normal_confidence],
                    )
                    * lambda_normal_l1
                )

        else:
            normal_l1 = 0.0

        # loss
        total_loss = loss + dist_loss + normal_loss + normal_l1 + Ldepth

        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}",
                    "curvature": f"{ema_curvature_for_log:.{5}f}",
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train_loss_patches/dist_loss", ema_dist_for_log, iteration
                )
                tb_writer.add_scalar(
                    "train_loss_patches/normal_loss", ema_normal_for_log, iteration
                )

            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.opacity_cull,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/reg_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)
        tb_writer.add_scalar(
            "total_points", scene.gaussians.get_xyz.shape[0], iteration
        )

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )
                    # depth
                    gt_depth = viewpoint.gt_depth.to("cuda")
                    depth_dilated_edges = find_edges(gt_depth)
                    gt_depth = torch.where(
                        depth_dilated_edges,
                        torch.tensor(0.0).cuda(),
                        gt_depth,
                    ).cuda()

                    norm = gt_depth.max()
                    gt_depth = gt_depth / norm
                    gt_depth = colormap(gt_depth.cpu().numpy()[0], cmap="turbo")

                    # normal
                    gt_normal = viewpoint.gt_normal.to("cuda") * 0.5 + 0.5

                    # curvature
                    curvature = render_pkg["curvature"]
                    curvature = colormap(curvature.cpu().numpy()[0], cmap="turbo")

                    # edge
                    edge = render_pkg["edge"]
                    edge = colormap(edge.cpu().numpy()[0], cmap="turbo")

                    gt_edge = gradient_map(gt_image)
                    gt_edge = torch.where(
                        gt_edge > 0.1, torch.tensor(1).cuda(), torch.tensor(0).cuda()
                    ).float()

                    gt_edge = colormap(gt_edge.cpu().numpy()[0], cmap="turbo")

                    if tb_writer and (idx < 5):

                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap="turbo")
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/depth".format(viewpoint.image_name),
                            depth[None],
                            global_step=iteration,
                        )
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )

                        rend_alpha = render_pkg["rend_alpha"]
                        rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                        surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5

                        normal_diff = mean_angular_error(
                            surf_normal * 2 - 1, gt_normal * 2 - 1
                        )
                        normal_confidence = 1 - (normal_diff > 0.1).float()
                        normal_confidence = colormap(
                            normal_confidence.cpu().numpy(), cmap="turbo"
                        )

                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/normal_confidence".format(viewpoint.image_name),
                            normal_confidence[None],
                            global_step=iteration,
                        )

                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/rend_normal".format(viewpoint.image_name),
                            rend_normal[None],
                            global_step=iteration,
                        )
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/surf_normal".format(viewpoint.image_name),
                            surf_normal[None],
                            global_step=iteration,
                        )
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/rend_alpha".format(viewpoint.image_name),
                            rend_alpha[None],
                            global_step=iteration,
                        )
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render_curvature".format(viewpoint.image_name),
                            curvature[None],
                            global_step=iteration,
                        )
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render_edge".format(viewpoint.image_name),
                            edge[None],
                            global_step=iteration,
                        )

                        rend_dist = render_pkg["rend_dist"]
                        rend_dist = colormap(rend_dist.cpu().numpy()[0])
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/rend_dist".format(viewpoint.image_name),
                            rend_dist[None],
                            global_step=iteration,
                        )

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_depth".format(viewpoint.image_name),
                                gt_depth[None],
                                global_step=iteration,
                            )
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_normal".format(viewpoint.image_name),
                                gt_normal[None],
                                global_step=iteration,
                            )
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_edge".format(viewpoint.image_name),
                                gt_edge[None],
                                global_step=iteration,
                            )

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )

        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--depth_supervision", action="store_true", default=None)
    parser.add_argument("--normal_supervision", action="store_true", default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.depth_supervision,
        args.normal_supervision,
    )

    # All done
    print("\nTraining complete.")
