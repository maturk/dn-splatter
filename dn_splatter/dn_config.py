from __future__ import annotations

from dn_splatter.data.normal_nerfstudio import NormalNerfstudioConfig
from dn_splatter.dn_datamanager import DNSplatterManagerConfig
from dn_splatter.dn_model import DNSplatterModelConfig
from dn_splatter.dn_pipeline import DNSplatterPipelineConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

dn_splatter = MethodSpecification(
    config=TrainerConfig(
        method_name="dn-splatter",
        steps_per_eval_image=500,
        steps_per_eval_batch=500,
        steps_per_save=1000000,
        steps_per_eval_all_images=1000000,
        max_num_iterations=30000,
        mixed_precision=False,
        gradient_accumulation_steps={"camera_opt": 100, "color": 10, "shs": 10},
        pipeline=DNSplatterPipelineConfig(
            datamanager=DNSplatterManagerConfig(
                dataparser=NormalNerfstudioConfig(load_3D_points=True)
            ),
            model=DNSplatterModelConfig(),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6, max_steps=30000
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5, max_steps=30000
                ),
            },
            "normals": {
                "optimizer": AdamOptimizerConfig(
                    lr=1e-3, eps=1e-15
                ),  # this does nothing, its just here to make the trainer happy
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="DN-Splatter: depth and normal priors for 3DGS",
)

dn_splatter_big = MethodSpecification(
    config=TrainerConfig(
        method_name="dn-splatter-big",
        steps_per_eval_image=500,
        steps_per_eval_batch=500,
        steps_per_save=1000000,
        steps_per_eval_all_images=1000000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=DNSplatterPipelineConfig(
            datamanager=DNSplatterManagerConfig(
                dataparser=NormalNerfstudioConfig(load_3D_points=True)
            ),
            model=DNSplatterModelConfig(
                cull_alpha_thresh=0.005,
                continue_cull_post_densification=False,
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5, max_steps=30000
                ),
            },
            "normals": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="DN-Splatter Big variant",
)
