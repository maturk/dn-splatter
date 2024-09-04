"""Our version of Nerfacto for evaluation"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

import torch
import torchvision.transforms.functional as TF
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from dn_splatter.metrics import DepthMetrics
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig


@dataclass
class GNerfactoModelConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: GNerfactoModel)
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    is_euclidean_depth: bool = False
    """Whether input depth maps are Euclidean distances (or z-distances)."""
    far_plane: float = 2.0
    """How far along the ray to stop sampling."""
    predict_normals: bool = True
    """Whether to predict normals or not."""


class GNerfactoModel(NerfactoModel):
    config: GNerfactoModelConfig

    def populate_modules(self):
        super().populate_modules()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=11)
        self.lpips = LearnedPerceptualImagePatchSimilarity()
        self.depth_metrics = DepthMetrics()
        self.collider = NearFarCollider(
            near_plane=self.config.near_plane, far_plane=self.config.far_plane
        )

    def get_outputs(self, ray_bundle: RayBundle):
        outputs = super().get_outputs(ray_bundle)
        if ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata:
            outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]
        return outputs

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs[
            "rgb"
        ]  # Blended with background (black if random background)
        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        if "mask" in batch:
            mask = batch["mask"].to(self.device)
            gt_rgb = gt_rgb * mask
            predicted_rgb = predicted_rgb * mask

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {
            "rgb_psnr": float(psnr.item()),
            "rgb_ssim": float(ssim),
        }  # type: ignore
        metrics_dict["rgb_lpips"] = float(lpips)

        if "sensor_depth" in batch:
            gt_depth = batch["sensor_depth"].to(self.device)

            gt_depth = gt_depth * outputs["directions_norm"]

            predicted_depth = outputs["depth"]
            if predicted_depth.shape[:2] != gt_depth.shape[:2]:
                predicted_depth = TF.resize(
                    predicted_depth.permute(2, 0, 1), gt_depth.shape[:2], antialias=None
                ).permute(1, 2, 0)

            gt_depth = gt_depth.to(torch.float32)  # it is in float64 previous
            if "mask" in batch:
                gt_depth = gt_depth * mask
                predicted_depth = predicted_depth * mask

            # add depth eval metrics
            (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3) = self.depth_metrics(
                predicted_depth.permute(2, 0, 1), gt_depth.permute(2, 0, 1)
            )

            depth_metrics = {
                "depth_abs_rel": float(abs_rel.item()),
                "depth_sq_rel": float(sq_rel.item()),
                "depth_rmse": float(rmse.item()),
                "depth_rmse_log": float(rmse_log.item()),
                "depth_a1": float(a1.item()),
                "depth_a2": float(a2.item()),
                "depth_a3": float(a3.item()),
            }
            metrics_dict.update(depth_metrics)

        images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict
