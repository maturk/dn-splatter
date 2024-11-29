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

from networkx import weakly_connected_components
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import math


def l1_loss(network_output, gt, mean=True):
    if mean:
        return torch.abs((network_output - gt)).mean()
    else:
        return torch.abs((network_output - gt))


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def edgeaware_l1_loss(network_output, gt, rgb, mask):
    l1 = l1_loss(network_output, gt, mean=False)
    grad_img_x = torch.mean(torch.abs(rgb[:, :-1, :] - rgb[:, 1:, :]), -1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(rgb[:, :, :-1] - rgb[:, :, 1:]), -1, keepdim=True)
    lambda_x = torch.exp(-grad_img_x)
    lambda_y = torch.exp(-grad_img_y)
    loss_x = lambda_x * l1[..., :, :-1, :]
    loss_y = lambda_y * l1[..., :, :, :-1]
    return (
        loss_x[mask[..., :, :-1, :].repeat(3, 1, 1)].mean()
        + loss_y[mask[..., :, :, :-1].repeat(3, 1, 1)].mean()
    )


def confidence_l1_loss(network_output, gt, confidence):
    l1 = l1_loss(network_output, gt, mean=False)
    l1 = l1 * torch.exp(-confidence)
    return l1.mean()


def weight_l1_loss(network_output, gt, dist):
    l1 = l1_loss(network_output, gt, mean=False)
    l1 = l1 / (dist + 1)
    return l1


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(
        disp[:, 1:-1, :-2] + disp[:, 1:-1, 2:] - 2 * disp[:, 1:-1, 1:-1]
    )
    grad_disp_y = torch.abs(
        disp[:, :-2, 1:-1] + disp[:, 2:, 1:-1] - 2 * disp[:, 1:-1, 1:-1]
    )
    grad_img_x = (
        torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True)
        * 0.5
    )
    grad_img_y = (
        torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True)
        * 0.5
    )
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def normalize_vectors(vectors: torch.Tensor) -> torch.Tensor:
    norms = torch.norm(vectors, dim=1, keepdim=True)
    return vectors / norms


def curvature_loss(normal, rgb):
    """
    smooth loss borrowed from dn-splatter
    """
    normal = (normal + 1) / 2

    grad_depth_x = torch.abs(normal[:, :-1, :] - normal[:, 1:, :])
    grad_depth_y = torch.abs(normal[:, :, :-1] - normal[:, :, 1:])

    grad_img_x = torch.mean(torch.abs(rgb[:, :-1, :] - rgb[:, 1:, :]), -1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(rgb[:, :, :-1] - rgb[:, :, 1:]), -1, keepdim=True)

    grad_depth_x *= torch.exp(-grad_img_x)
    grad_depth_y *= torch.exp(-grad_img_y)

    return grad_depth_x.mean() + grad_depth_y.mean()


# borrowed from sparseGS: https://github.com/ForMyCat/SparseGS/blob/95e7aef29c5562400d3b2b38cc7e90436a432b7c/utils/loss_utils.py#L80
def pearson_depth_loss(depth_src, depth_target):
    # co = pearson(depth_src.reshape(-1), depth_target.reshape(-1))
    """
    depth_src: [n]
    """
    src = depth_src - depth_src.mean()
    target = depth_target - depth_target.mean()

    src = src / (src.std() + 1e-6)
    target = target / (target.std() + 1e-6)

    co = ((src * target)).mean()
    assert not torch.any(torch.isnan(co))
    return 1 - co


def local_pearson_loss(depth_src, depth_target, mask, box_p=128, p_corr=0.5):
    # Randomly select patch, top left corner of the patch (x_0,y_0) has to be 0 <= x_0 <= max_h, 0 <= y_0 <= max_w
    """
    depth_src: [B, H, W]
    depth_target: [B, H, W]
    """
    num_box_h = math.floor(depth_src.shape[1] / box_p)
    num_box_w = math.floor(depth_src.shape[2] / box_p)
    max_h = depth_src.shape[1] - box_p
    max_w = depth_src.shape[2] - box_p

    n_corr = int(p_corr * num_box_h * num_box_w)
    x_0 = torch.randint(0, max_h, size=(n_corr,), device="cuda")
    y_0 = torch.randint(0, max_w, size=(n_corr,), device="cuda")
    x_1 = x_0 + box_p
    y_1 = y_0 + box_p
    _loss = torch.tensor(0.0, device="cuda")
    for i in range(len(x_0)):
        _loss += pearson_depth_loss(
            depth_src[:, x_0[i] : x_1[i], y_0[i] : y_1[i]][
                mask[:, x_0[i] : x_1[i], y_0[i] : y_1[i]]
            ].reshape(-1),
            depth_target[:, x_0[i] : x_1[i], y_0[i] : y_1[i]][
                mask[:, x_0[i] : x_1[i], y_0[i] : y_1[i]]
            ].reshape(-1),
        )
    return _loss / n_corr


def mean_angular_error(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Compute the mean angular error between predicted and reference normals

    Args:
        predicted_normals: [C, H, W] tensor of predicted normals
        reference_normals : [C, H, W] tensor of gt normals

    Returns:
        mae: [H, W] mean angular error
    """
    pred = torch.nn.functional.normalize(pred, dim=0)
    gt = torch.nn.functional.normalize(gt, dim=0)

    dot_products = torch.sum(gt * pred, axis=0)  # over the C dimension
    # Clamp the dot product to ensure valid cosine values (to avoid nans)
    dot_products = torch.clip(dot_products, -1.0, 1.0)
    # Calculate the angle between the vectors (in radians)
    mae = torch.arccos(dot_products)
    return mae
