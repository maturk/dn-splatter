"""Metrics"""

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class PDMetrics(nn.Module):
    """Computation of error metrics between predicted and ground truth point clouds

    Input:
        pred: predicted pd
        gt: ground truth pd

    Returns:
        acc
        comp
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.acc = calculate_accuracy
        self.cmp = calculate_completeness

    @torch.no_grad()
    def forward(self, pred, gt):
        pred_points = np.asarray(pred.points)
        gt_points = np.asarray(gt.points)
        acc_score = self.acc(pred_points, gt_points)
        cmp_score = self.cmp(pred_points, gt_points)

        return (acc_score, cmp_score)


def calculate_accuracy(reconstructed_points, reference_points, percentile=90):
    """
    Calculat accuracy: How far away 90% of the reconstructed point clouds are from the reference point cloud.
    """
    tree = cKDTree(reference_points)
    distances, _ = tree.query(reconstructed_points)
    return np.percentile(distances, percentile)


def calculate_completeness(reconstructed_points, reference_points, threshold=0.05):
    """
    calucate completeness: What percentage of the reference point cloud is within
    a specific distance of the reconstructed point cloud.
    """
    tree = cKDTree(reconstructed_points)
    distances, _ = tree.query(reference_points)
    within_threshold = np.sum(distances < threshold) / len(distances)
    return within_threshold * 100


def mean_angular_error(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Compute the mean angular error between predicted and reference normals

    Args:
        predicted_normals: [B, C, H, W] tensor of predicted normals
        reference_normals : [B, C, H, W] tensor of gt normals

    Returns:
        mae: [B, H, W] mean angular error
    """
    dot_products = torch.sum(gt * pred, dim=1)  # over the C dimension
    # Clamp the dot product to ensure valid cosine values (to avoid nans)
    dot_products = torch.clamp(dot_products, -1.0, 1.0)
    # Calculate the angle between the vectors (in radians)
    mae = torch.acos(dot_products)
    return mae


class RGBMetrics(nn.Module):
    """Computation of error metrics between predicted and ground truth images

    Input:
        pred: predicted image [B, C, H, W]
        gt: ground truth image [B, C, H, W]

    Returns:
        PSNR
        SSIM
        LPIPS
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=11)
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    @torch.no_grad()
    def forward(self, pred, gt):
        self.device = pred.device
        self.psnr.to(self.device)
        self.ssim.to(self.device)
        self.lpips.to(self.device)

        psnr_score = self.psnr(pred, gt)
        ssim_score = self.ssim(pred, gt)
        lpips_score = self.lpips(pred, gt)

        return (psnr_score, ssim_score, lpips_score)


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
        # rmse_log[rmse_log == float("inf")] = float("nan")
        rmse_log = torch.sqrt(rmse_log).nanmean()

        abs_rel = torch.abs(gt - pred)[mask] / gt[mask]
        abs_rel = abs_rel.mean()
        sq_rel = (gt - pred)[mask] ** 2 / gt[mask]
        sq_rel = sq_rel.mean()

        return (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3)


class NormalMetrics(nn.Module):
    """Computation of error metrics between predicted and ground truth normal maps.

    Args:
        predicted_normals: [B, C, H, W] tensor of predicted normals
        reference_normals : [B, C, H, W] tensor of gt normals

    Returns:
        All metrics are averaged over the batch
        mae: mean angular error
        rmse: root mean squared error
        mean: mean error
        med: median error
    """

    def __init__(self, **kwargs):
        super().__init__()

    @torch.no_grad()
    def forward(self, pred, gt):
        b, c, _, _ = gt.shape
        # calculate MAE
        mae = mean_angular_error(pred, gt).mean()
        # calculate RMSE
        rmse = torch.sqrt(torch.mean(torch.square(gt - pred), dim=[1, 2, 3])).mean()
        # calculate Mean
        mean_err = torch.mean(torch.abs(gt - pred), dim=[1, 2, 3]).mean()
        # calculate Median
        med_err = torch.median(
            torch.abs(gt.view(b, c, -1) - pred.view(b, c, -1))
        ).mean()
        return mae, rmse, mean_err, med_err
