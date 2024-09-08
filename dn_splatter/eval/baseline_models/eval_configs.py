"""
Eval configs
"""

from __future__ import annotations

from dn_splatter.data.dn_dataset import GDataset
from dn_splatter.dn_pipeline import DNSplatterPipelineConfig
from dn_splatter.eval.baseline_models.g_depthnerfacto import GDepthNerfactoModelConfig
from dn_splatter.eval.baseline_models.g_nerfacto import GNerfactoModelConfig
from dn_splatter.eval.baseline_models.g_neusfacto import DNeuSFactoModelConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.pixel_samplers import PairPixelSamplerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
    MultiStepSchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.plugins.types import MethodSpecification

gnerfacto = MethodSpecification(
    config=TrainerConfig(
        method_name="gnerfacto",
        steps_per_eval_batch=500,
        steps_per_save=500,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=DNSplatterPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                _target=VanillaDataManager[GDataset],
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=GNerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.0001, max_steps=200000
                ),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.0001, max_steps=200000
                ),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=5000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Our version of nerfacto for experimentation.",
)

gdepthfacto = MethodSpecification(
    config=TrainerConfig(
        method_name="gdepthfacto",
        steps_per_eval_batch=500,
        steps_per_save=500,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=DNSplatterPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                _target=VanillaDataManager[GDataset],
                pixel_sampler=PairPixelSamplerConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=GDepthNerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.0001, max_steps=200000
                ),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.0001, max_steps=200000
                ),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=5000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Our version of depth-nerfacto for experimentation.",
)

gneusfacto = MethodSpecification(
    config=TrainerConfig(
        method_name="gneusfacto",
        steps_per_eval_image=500,
        steps_per_eval_batch=500,
        steps_per_save=500,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=DNSplatterPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                _target=VanillaDataManager[GDataset],
                train_num_rays_per_batch=2048,
                eval_num_rays_per_batch=1024,
            ),
            model=DNeuSFactoModelConfig(
                # proposal network allows for significantly smaller sdf/color network
                sdf_field=SDFFieldConfig(
                    use_grid_feature=True,
                    num_layers=2,
                    num_layers_color=2,
                    hidden_dim=256,
                    bias=0.5,
                    beta_init=0.8,
                    use_appearance_embedding=False,
                    inside_outside=False,
                ),
                background_model="none",
                eval_num_rays_per_chunk=1024,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": MultiStepSchedulerConfig(
                    max_steps=20001, milestones=(10000, 1500, 18000)
                ),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(
                    warm_up_end=500, learning_rate_alpha=0.05, max_steps=20001
                ),
            },
            "field_background": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(
                    warm_up_end=500, learning_rate_alpha=0.05, max_steps=20001
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Our version of neus-facto for experimentation.",
)
