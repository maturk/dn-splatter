"""Loss functions"""

import abc
import math
from enum import Enum
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
    StructuralSimilarityIndexMeasure,
)

from nerfstudio.field_components.field_heads import FieldHeadNames


class DepthLossType(Enum):
    """Enum for specifying depth loss"""

    MSE = "mse"
    L1 = "L1"
    LogL1 = "LogL1"
    HuberL1 = "HuberL1"
    TV = "TV"
    EdgeAwareLogL1 = "EdgeAwareLogL1"
    EdgeAwareTV = "EdgeAwareTV"
    PearsonDepth = "PearsonDepth"
    LocalPearsonDepthLoss = "LocalPearsonDepthLoss"


class DepthLoss(nn.Module):
    """Factory method class for various depth losses"""

    def __init__(self, depth_loss_type: DepthLossType, **kwargs):
        super().__init__()
        self.depth_loss_type = depth_loss_type
        self.kwargs = kwargs
        self.loss = self._get_loss_instance()

    @abc.abstractmethod
    def forward(self, *args) -> Tensor:
        return self.loss(*args)

    def _get_loss_instance(self) -> nn.Module:
        if self.depth_loss_type == DepthLossType.MSE:
            return torch.nn.MSELoss()
        if self.depth_loss_type == DepthLossType.L1:
            return L1(**self.kwargs)
        elif self.depth_loss_type == DepthLossType.LogL1:
            return LogL1(**self.kwargs)
        elif self.depth_loss_type == DepthLossType.HuberL1:
            return HuberL1(**self.kwargs)
        elif self.depth_loss_type == DepthLossType.EdgeAwareLogL1:
            return EdgeAwareLogL1(**self.kwargs)
        elif self.depth_loss_type == DepthLossType.EdgeAwareTV:
            return EdgeAwareTV(**self.kwargs)
        elif self.depth_loss_type == DepthLossType.TV:
            return TVLoss(**self.kwargs)
        elif self.depth_loss_type == DepthLossType.PearsonDepth:
            return PearsonDepthLoss(**self.kwargs)
        elif self.depth_loss_type == DepthLossType.LocalPearsonDepthLoss:
            return LocalPearsonDepthLoss(**self.kwargs)
        else:
            raise ValueError(f"Unsupported loss type: {self.depth_loss_type}")


class DSSIML1(nn.Module):
    """Implementation of DSSIM+L1 loss

    Args:
        implementation: use 'scalar' to return scalar value, use 'per-pixel' to return per-pixel loss

    reference:
        https://arxiv.org/abs/1909.09051 and
        https://arxiv.org/abs/1609.03677

    original implementation uses 3x3 kernel size and single resolution SSIM
    """

    def __init__(
        self,
        kernel_size: int = 3,
        alpha: float = 0.85,
        single_resolution: bool = True,
        implementation: Literal["scalar", "per-pixel"] = "per-pixel",
        **kwargs,
    ):
        super().__init__()
        self.implementation = implementation

        # torchvision SSIM returns a scalar value for SSIM, not per pixel tensor
        if single_resolution and implementation == "scalar":
            self.ssim = StructuralSimilarityIndexMeasure(
                gaussian_kernel=True,
                kernel_size=kernel_size,
                reduction="elementwise_mean",
            )
        elif implementation == "scalar":
            self.ssim = MultiScaleStructuralSimilarityIndexMeasure(
                gaussian_kernel=True,
                kernel_size=kernel_size,
                reduction="elementwise_mean",
            )
        else:
            self.mu_x_pool = nn.AvgPool2d(kernel_size, 1)
            self.mu_y_pool = nn.AvgPool2d(kernel_size, 1)
            self.sig_x_pool = nn.AvgPool2d(kernel_size, 1)
            self.sig_y_pool = nn.AvgPool2d(kernel_size, 1)
            self.sig_xy_pool = nn.AvgPool2d(kernel_size, 1)
            self.refl = nn.ReflectionPad2d(int((kernel_size - 1) / 2))
            self.C1 = 0.01**2
            self.C2 = 0.03**2

        self.alpha = alpha

    def ssim_per_pixel(self, pred, gt):
        x = self.refl(pred)
        y = self.refl(gt)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x**2) - mu_x**2
        sigma_y = self.sig_y_pool(y**2) - mu_y**2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

    def forward(self, pred: Tensor, gt: Tensor):
        """Compute DSSIM+L1 loss"""
        if (pred.shape[-1] == 1 or pred.shape[-1] == 3) and pred.dim() == 3:
            pred = pred.permute(2, 0, 1).unsqueeze(0)

        if (gt.shape[-1] == 1 or pred.shape[-1] == 3) and gt.dim() == 3:
            gt = gt.permute(2, 0, 1).unsqueeze(0)

        if self.implementation == "scalar":
            abs_diff = torch.abs(pred - gt)
            l1_loss = abs_diff.mean()
            ssim_loss = self.ssim(pred, gt)
            return self.alpha * (1 - ssim_loss) / 2 + (1 - self.alpha) * l1_loss
        else:
            abs_diff = torch.abs(pred - gt)
            l1_loss = abs_diff.mean(1, True)
            ssim_loss = self.ssim_per_pixel(pred, gt).mean(1, True)
            return self.alpha * ssim_loss + (1 - self.alpha) * l1_loss


class L1(nn.Module):
    """L1 loss"""

    def __init__(
        self, implementation: Literal["scalar", "per-pixel"] = "scalar", **kwargs
    ):
        super().__init__()
        self.implementation = implementation

    def forward(self, pred, gt):
        if self.implementation == "scalar":
            return torch.abs(pred - gt).mean()
        else:
            return torch.abs(pred - gt)


class LogL1(nn.Module):
    """Log-L1 loss"""

    def __init__(
        self, implementation: Literal["scalar", "per-pixel"] = "scalar", **kwargs
    ):
        super().__init__()
        self.implementation = implementation

    def forward(self, pred, gt):
        if self.implementation == "scalar":
            return torch.log(1 + torch.abs(pred - gt)).mean()
        else:
            return torch.log(1 + torch.abs(pred - gt))


class EdgeAwareLogL1(nn.Module):
    """Gradient aware Log-L1 loss"""

    def __init__(
        self, implementation: Literal["scalar", "per-pixel"] = "scalar", **kwargs
    ):
        super().__init__()
        self.implementation = implementation
        self.logl1 = LogL1(implementation="per-pixel")

    def forward(self, pred: Tensor, gt: Tensor, rgb: Tensor, mask: Optional[Tensor]):
        logl1 = self.logl1(pred, gt)

        grad_img_x = torch.mean(
            torch.abs(rgb[..., :, :-1, :] - rgb[..., :, 1:, :]), -1, keepdim=True
        )
        grad_img_y = torch.mean(
            torch.abs(rgb[..., :-1, :, :] - rgb[..., 1:, :, :]), -1, keepdim=True
        )
        lambda_x = torch.exp(-grad_img_x)
        lambda_y = torch.exp(-grad_img_y)

        loss_x = lambda_x * logl1[..., :, :-1, :]
        loss_y = lambda_y * logl1[..., :-1, :, :]

        if self.implementation == "per-pixel":
            if mask is not None:
                loss_x[~mask[..., :, :-1, :]] = 0
                loss_y[~mask[..., :-1, :, :]] = 0
            return loss_x[..., :-1, :, :] + loss_y[..., :, :-1, :]

        if mask is not None:
            assert mask.shape[:2] == pred.shape[:2]
            loss_x = loss_x[mask[..., :, :-1, :]]
            loss_y = loss_y[mask[..., :-1, :, :]]

        if self.implementation == "scalar":
            return loss_x.mean() + loss_y.mean()


class HuberL1(nn.Module):
    """L1+huber loss"""

    def __init__(
        self,
        tresh=0.2,
        implementation: Literal["scalar", "per-pixel"] = "scalar",
        **kwargs,
    ):
        super().__init__()
        self.tresh = tresh
        self.implementation = implementation

    def forward(self, pred, gt):
        mask = gt != 0
        l1 = torch.abs(pred[mask] - gt[mask])
        d = self.tresh * torch.max(l1)
        loss = torch.where(l1 < d, ((pred - gt) ** 2 + d**2) / (2 * d), l1)
        if self.implementation == "scalar":
            return loss.mean()
        else:
            return loss


class EdgeAwareTV(nn.Module):
    """Edge Aware Smooth Loss"""

    def __init__(self):
        super().__init__()

    def forward(self, depth: Tensor, rgb: Tensor):
        """
        Args:
            depth: [batch, H, W, 1]
            rgb: [batch, H, W, 3]
        """
        grad_depth_x = torch.abs(depth[..., :, :-1, :] - depth[..., :, 1:, :])
        grad_depth_y = torch.abs(depth[..., :-1, :, :] - depth[..., 1:, :, :])

        grad_img_x = torch.mean(
            torch.abs(rgb[..., :, :-1, :] - rgb[..., :, 1:, :]), -1, keepdim=True
        )
        grad_img_y = torch.mean(
            torch.abs(rgb[..., :-1, :, :] - rgb[..., 1:, :, :]), -1, keepdim=True
        )

        grad_depth_x *= torch.exp(-grad_img_x)
        grad_depth_y *= torch.exp(-grad_img_y)

        return grad_depth_x.mean() + grad_depth_y.mean()


class TVLoss(nn.Module):
    """TV loss"""

    def __init__(self):
        super().__init__()

    def forward(self, pred):
        """
        Args:
            pred: [batch, H, W, 3]

        Returns:
            tv_loss: [batch]
        """
        h_diff = pred[..., :, :-1, :] - pred[..., :, 1:, :]
        w_diff = pred[..., :-1, :, :] - pred[..., 1:, :, :]
        return torch.mean(torch.abs(h_diff)) + torch.mean(torch.abs(w_diff))


# sensor depth loss, adapted from https://github.com/dazinovic/neural-rgbd-surface-reconstruction/blob/main/losses.py
class SensorDepthLoss(nn.Module):
    """Sensor Depth loss"""

    def __init__(self, truncation: float):
        super(SensorDepthLoss, self).__init__()
        self.truncation = truncation  #  0.05 * 0.3 5cm scaled

    def forward(self, batch, outputs):
        """take the mim

        Args:
            batch (Dict): inputs
            outputs (Dict): outputs data from surface model

        Returns:
            l1_loss: l1 loss
            freespace_loss: free space loss
            sdf_loss: sdf loss
        """
        depth_pred = outputs["depth"]
        depth_gt = batch["sensor_depth"].to(depth_pred.device)[..., None]
        valid_gt_mask = depth_gt > 0.0

        l1_loss = torch.sum(valid_gt_mask * torch.abs(depth_gt - depth_pred)) / (
            valid_gt_mask.sum() + 1e-6
        )

        # free space loss and sdf loss
        ray_samples = outputs["ray_samples"]
        filed_outputs = outputs["field_outputs"]
        pred_sdf = filed_outputs[FieldHeadNames.SDF][..., 0]
        directions_norm = outputs["directions_norm"]

        z_vals = ray_samples.frustums.starts[..., 0] / directions_norm

        truncation = self.truncation
        front_mask = valid_gt_mask & (z_vals < (depth_gt - truncation))
        back_mask = valid_gt_mask & (z_vals > (depth_gt + truncation))
        sdf_mask = valid_gt_mask & (~front_mask) & (~back_mask)

        num_fs_samples = front_mask.sum()
        num_sdf_samples = sdf_mask.sum()
        num_samples = num_fs_samples + num_sdf_samples + 1e-6
        fs_weight = 1.0 - num_fs_samples / num_samples
        sdf_weight = 1.0 - num_sdf_samples / num_samples

        free_space_loss = (
            torch.mean((F.relu(truncation - pred_sdf) * front_mask) ** 2) * fs_weight
        )

        sdf_loss = (
            torch.mean(((z_vals + pred_sdf) - depth_gt) ** 2 * sdf_mask) * sdf_weight
        )
        return l1_loss, free_space_loss, sdf_loss


class NormalLossType(Enum):
    """Enum for specifying depth loss"""

    L1 = "L1"
    Smooth = "Smooth"


class NormalLoss(nn.Module):
    """Factory method class for various depth losses"""

    def __init__(self, normal_loss_type: NormalLossType, **kwargs):
        super().__init__()
        self.normal_loss_type = normal_loss_type
        self.kwargs = kwargs
        self.loss = self._get_loss_instance()

    @abc.abstractmethod
    def forward(self, *args) -> Tensor:
        return self.loss(*args)

    def _get_loss_instance(self) -> nn.Module:
        if self.normal_loss_type == NormalLossType.L1:
            return L1(**self.kwargs)
        elif self.normal_loss_type == NormalLossType.Smooth:
            return TVLoss(**self.kwargs)
        else:
            raise ValueError(f"Unsupported loss type: {self.normal_loss_type}")


# pearson depth loss, adapted from https://github.com/ForMyCat/SparseGS/blob/95e7aef29c5562400d3b2b38cc7e90436a432b7c/utils/loss_utils.py#L80
class PearsonDepthLoss(nn.Module):
    """PearsonDepthLoss"""

    def __init__(self):
        super(PearsonDepthLoss, self).__init__()

    def forward(self, depth_pred, depth_gt):
        """take the mim
        Args:
            batch (Dict): inputs render depth, target depth
            outputs (Dict): outputs data
        Returns:
            p_loss: pearson depth loss
        """
        src = depth_pred - depth_pred.mean()
        target = depth_gt - depth_gt.mean()

        src = src / (src.std() + 1e-6)
        target = target / (target.std() + 1e-6)

        co = (src * target).mean()
        assert not torch.any(torch.isnan(co))
        return 1 - co


# pearson local depth loss, adapted from https://github.com/ForMyCat/SparseGS/blob/95e7aef29c5562400d3b2b38cc7e90436a432b7c/utils/loss_utils.py#L94
class LocalPearsonDepthLoss(nn.Module):
    """LocalPearsonDepthLoss"""

    def __init__(self):
        super(LocalPearsonDepthLoss, self).__init__()
        self.pearson_depth_loss = PearsonDepthLoss()

    def forward(self, depth_pred, depth_gt, box_p=128, p_corr=0.5):
        """take the mim
        Args:
            batch (Dict): inputs render depth, target depth
            outputs (Dict): outputs data
        Returns:
            p_loss: pearson depth loss
        """
        num_box_h = math.floor(depth_pred.shape[0] / box_p)
        num_box_w = math.floor(depth_pred.shape[1] / box_p)
        max_h = depth_pred.shape[0] - box_p
        max_w = depth_pred.shape[1] - box_p
        _loss = torch.tensor(0.0, device="cuda")
        n_corr = int(p_corr * num_box_h * num_box_w)
        x_0 = torch.randint(0, max_h, size=(n_corr,), device="cuda")
        y_0 = torch.randint(0, max_w, size=(n_corr,), device="cuda")
        x_1 = x_0 + box_p
        y_1 = y_0 + box_p
        _loss = torch.tensor(0.0, device="cuda")
        for i in range(len(x_0)):
            _loss += self.pearson_depth_loss(
                depth_pred[x_0[i] : x_1[i], y_0[i] : y_1[i]].reshape(-1),
                depth_gt[x_0[i] : x_1[i], y_0[i] : y_1[i]].reshape(-1),
            )
        return _loss / n_corr  #
