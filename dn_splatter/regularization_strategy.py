from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dn_splatter.losses import DepthLoss, DepthLossType, NormalLoss, NormalLossType


def mean_angular_error(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Compute the mean angular error between predicted and reference normals

    Args:
        predicted_normals: [C, H, W] tensor of predicted normals
        reference_normals : [C, H, W] tensor of gt normals

    Returns:
        mae: [H, W] mean angular error
    """
    dot_products = torch.sum(gt * pred, axis=0)  # over the C dimension
    # Clamp the dot product to ensure valid cosine values (to avoid nans)
    dot_products = torch.clip(dot_products, -1.0, 1.0)
    # Calculate the angle between the vectors (in radians)
    mae = torch.arccos(dot_products)
    return mae


def dilate_edge(edge, dilation_size=1):

    kernel_size = 2 * dilation_size + 1
    dilation_kernel = torch.ones((1, 1, kernel_size, kernel_size)).cuda()

    edge_dilated = F.conv2d(edge, dilation_kernel, padding=dilation_size)
    edge_dilated = torch.clamp(edge_dilated, 0, 1)

    return edge_dilated


def find_edges(im, threshold=0.01, dilation_itr=1):
    laplacian_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=im.dtype, device=im.device
    ).float()
    laplacian_kernel = laplacian_kernel.unsqueeze(0).unsqueeze(0)
    # Apply the Laplacian kernel to the image
    if im.shape[0] == 1:
        laplacian = F.conv2d(
            (1.0 / (im + 1e-6)).unsqueeze(
                0
            ),  # Add batch dimension, shape: (1, 1, h, w)
            laplacian_kernel,
            padding=1,
        ).squeeze(
            0
        )  # shape: (1, h, w)

        edges = (laplacian > threshold) * 1.0
        structure_el = laplacian_kernel * 0.0 + 1.0

        dilated_edges = edges
        for i in range(dilation_itr):
            dilated_edges = F.conv2d(
                dilated_edges.unsqueeze(0),
                structure_el,
                padding=1,
            ).squeeze(0)
    elif im.shape[0] == 3:
        laplacian = []
        for i in range(3):
            channel_laplacian = F.conv2d(
                (1.0 / (im[i : i + 1] + 1e-6)).unsqueeze(0),  # Shape: (1, 1, h, w)
                laplacian_kernel,
                padding=1,
            ).squeeze(
                0
            )  # Shape: (1, h, w)
            laplacian.append(channel_laplacian)
        laplacian = torch.cat(laplacian, dim=0)  # Shape: (3, h, w)
        edges = (laplacian > threshold) * 1.0
        structure_el = laplacian_kernel * 0.0 + 1.0

        for i in range(dilation_itr):
            dilated_edges = []
            for j in range(3):
                channel_dilated = F.conv2d(
                    edges[j : j + 1].unsqueeze(0),  # Shape: (1, 1, h, w)
                    structure_el,
                    padding=1,
                ).squeeze(
                    0
                )  # Shape: (1, h, w)
                dilated_edges.append(channel_dilated)
            dilated_edges = torch.cat(dilated_edges, dim=0)  # Shape: (3, h, w)

    dilated_edges = dilated_edges > 0.0
    return dilated_edges


class RegularizationStrategy(nn.Module):
    """Depth and normal regularization super class"""

    def __init__(self, **kwargs):
        super().__init__()
        # to keep track of which device the nn.Module is on
        self.device_indicator_param = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_param.device

    @abstractmethod
    def get_loss(self, **kwargs):
        """Loss"""

    def forward(self, **kwargs):
        """"""
        return self.get_loss(**kwargs)


class DNRegularization(RegularizationStrategy):
    """Regularization strategy as proposed in DN-Splatter

    This consists of an EdgeAware Depth loss, a Normal loss, normal smoothing loss, and a scale loss.
    """

    def __init__(
        self,
        depth_tolerance: float = 0.1,
        depth_loss_type: Optional[DepthLossType] = DepthLossType.EdgeAwareLogL1,
        depth_lambda: float = 0.2,
        normal_lambda: float = 0.1,
    ):
        super().__init__()
        self.depth_tolerance = depth_tolerance
        self.depth_loss_type = depth_loss_type
        self.depth_loss = DepthLoss(self.depth_loss_type)
        self.depth_lambda = depth_lambda

        self.normal_loss_type: NormalLossType = NormalLossType.L1
        self.normal_loss = NormalLoss(self.normal_loss_type)
        self.normal_smooth_loss_type: NormalLossType = NormalLossType.Smooth
        self.normal_smooth_loss = NormalLoss(self.normal_smooth_loss_type)
        self.normal_lambda = normal_lambda

    def get_loss(self, pred_depth, gt_depth, pred_normal, gt_normal, **kwargs):
        """Regularization loss"""

        depth_loss, normal_loss = 0.0, 0.0
        if self.depth_loss is not None:
            depth_loss = self.get_depth_loss(pred_depth, gt_depth, **kwargs)
        if self.normal_loss is not None:
            normal_loss = self.get_normal_loss(pred_normal, gt_normal, **kwargs)
        scales = kwargs["scales"]
        scale_loss = self.get_scale_loss(scales=scales)
        return depth_loss + normal_loss + scale_loss

    def get_depth_loss(self, pred_depth, gt_depth, **kwargs):
        """Depth loss"""

        valid_gt_mask = gt_depth > self.depth_tolerance
        if self.depth_loss_type == DepthLossType.EdgeAwareLogL1:
            gt_img = kwargs["gt_img"]
            depth_loss = self.depth_loss(
                pred_depth, gt_depth.float(), gt_img, valid_gt_mask
            )
        elif self.config.depth_loss_type == DepthLossType.PearsonDepth:
            mono_depth_loss_pearson = (
                self.depth_loss(pred_depth, gt_depth.float()) * valid_gt_mask.sum()
            ) / valid_gt_mask.sum()
            local_depth_loss = DepthLoss(DepthLossType.LocalPearsonDepthLoss)
            mono_depth_loss_local = (
                local_depth_loss(pred_depth, gt_depth.float()) * valid_gt_mask.sum()
            ) / valid_gt_mask.sum()
            depth_loss = (
                mono_depth_loss_pearson + self.pearson_lambda * mono_depth_loss_local
            )

        else:
            depth_loss = self.depth_loss(
                pred_depth[valid_gt_mask], gt_depth[valid_gt_mask].float()
            )

        depth_loss += self.depth_lambda * depth_loss

        return depth_loss

    def get_normal_loss(self, pred_normal, gt_normal, **kwargs):
        """Normal loss and normal smoothing"""
        normal_loss = self.normal_loss(pred_normal, gt_normal)
        normal_loss += self.normal_smooth_loss(pred_normal)

        return normal_loss

    def get_scale_loss(self, scales):
        """Scale loss"""
        # loss to minimise gaussian scale corresponding to normal direction
        scale_loss = torch.min(torch.exp(scales), dim=1, keepdim=True)[0].mean()
        return scale_loss
