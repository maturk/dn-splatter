"""
Depth + normal splatter
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import Tensor
from torch.nn import Parameter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from dn_splatter.losses import DepthLoss, DepthLossType, TVLoss
from dn_splatter.metrics import DepthMetrics, RGBMetrics
from dn_splatter.utils.camera_utils import get_colored_points_from_depth, project_pix
from dn_splatter.utils.knn import knn_sk
from dn_splatter.utils.normal_utils import normal_from_depth_image
from gsplat._torch_impl import quat_to_rotmat
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from gsplat.sh import num_sh_bases, spherical_harmonics
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.model_components import renderers
from nerfstudio.models.splatfacto import (
    RGB2SH,
    SplatfactoModel,
    SplatfactoModelConfig,
)
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class DNSplatterModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: DNSplatterModel)

    ### DNSplatter configs ###
    use_depth_loss: bool = False
    """Enable depth loss while training"""
    depth_loss_type: DepthLossType = DepthLossType.LogL1
    """Choose which depth loss to train with Literal["MSE", "LogL1", "HuberL1", "L1", "EdgeAwareLogL1")"""
    depth_tolerance: float = 0.1
    """Min depth value for depth loss"""
    smooth_loss_type: DepthLossType = DepthLossType.TV
    """Choose which smooth loss to train with Literal["TV", "EdgeAwareTV")"""
    sensor_depth_lambda: float = 0.0
    """Regularizer for sensor depth loss"""
    mono_depth_lambda: float = 0.0
    """Regularizer for mono depth loss"""
    use_depth_smooth_loss: bool = False
    """Whether to enable depth smooth loss or not"""
    smooth_loss_lambda: float = 0.1
    """Regularizer for smooth loss"""
    predict_normals: bool = True
    """Whether to extract and render normals or skip this"""
    use_normal_loss: bool = False
    """Enables normal loss('s)"""
    use_normal_cosine_loss: bool = False
    """Cosine similarity loss"""
    use_normal_tv_loss: bool = True
    """Use TV loss on predicted normals."""
    normal_supervision: Literal["mono", "depth"] = "depth"
    """Type of supervision for normals. Mono for monocular normals and depth for pseudo normals from depth maps."""
    normal_lambda: float = 0.1
    """Regularizer for normal loss"""
    normal_direction_warmup_steps = 5000
    """Warmup length for trying to align correct normal directions based on camera view vector"""
    use_sparse_loss: bool = False
    """Encourage opacities to be 0 or 1. From 'Neural volumes: Learning dynamic renderable volumes from images'."""
    sparse_lambda: float = 0.1
    """Regularizer for sparse loss"""
    sparse_loss_steps: int = 10
    """Enable sparse loss at steps"""
    use_binary_opacities: bool = False
    """Enable binary opacities"""
    binary_opacities_threshold: float = 0.9
    """Threshold for clipping opacities"""
    two_d_gaussians: bool = True
    """Encourage 2D Gaussians"""

    ### SuGaR style sdf loss settings ###
    use_sdf_loss: bool = False
    """Enable sdf loss during training"""
    sdf_loss_lambda: float = 0.1
    """Regularizer for sdf loss"""
    apply_sdf_loss_after_iters: int = 200
    """Start applying sdf loss after n training iterations"""
    apply_sdf_loss_iters: int = 10
    """Iterations to apply sdf loss"""
    knn_to_track: int = 16
    """How many nearest neighbours per gaussian to track"""
    num_sdf_samples: int = 100
    """Number of sdf samples to take"""

    ### Splatfacto configs ###
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    num_downscales: int = 0
    """at the beginning, resolution is 1/2^d, where d is this number"""
    use_scale_regularization: bool = True
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 5.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    stop_split_at: int = 15000
    """stop splitting at this step"""


class DNSplatterModel(SplatfactoModel):
    """Depth + Normal splatter"""

    config: DNSplatterModelConfig

    def populate_modules(self):
        if self.seed_points is not None and not self.config.random_init:
            means = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
        else:
            means = torch.nn.Parameter((torch.rand((500000, 3)) - 0.5) * 10)
        self.xys_grad_norm = None
        self.max_2Dsize = None
        dim_sh = num_sh_bases(self.config.sh_degree)
        num_points = means.shape[0]

        if self.seed_points is not None and not self.config.random_init:
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().cuda()
            if self.config.sh_degree > 0:
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)
                shs[:, 1:, 3:] = 0.0
            else:
                CONSOLE.log("use color only optimization with sigmoid activation")
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
            features_dc = torch.nn.Parameter(shs[:, 0, :])
            features_rest = torch.nn.Parameter(shs[:, 1:, :])
        else:
            features_dc = torch.nn.Parameter(torch.rand(num_points, 3))
            features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))

        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))

        self.step = 0
        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)

        self.mse_loss = torch.nn.MSELoss()

        # Depth Losses
        if self.config.use_depth_loss:
            self.depth_loss = DepthLoss(self.config.depth_loss_type)
        if self.config.use_depth_smooth_loss:
            if self.config.smooth_loss_type == DepthLossType.EdgeAwareTV:
                self.smooth_loss = DepthLoss(depth_loss_type=DepthLossType.EdgeAwareTV)
            else:
                self.smooth_loss = DepthLoss(depth_loss_type=DepthLossType.TV)

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=11)
        self.lpips = LearnedPerceptualImagePatchSimilarity()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.rgb_metrics = RGBMetrics()
        self.depth_metrics = DepthMetrics()
        distances, indices = self.k_nearest_sklearn(means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)

        # init normals if present
        if (
            self.seed_points is not None
            and len(self.seed_points) == 3
            and self.training
        ):  # type: ignore
            self.normals_seed = self.seed_points[-1].float()  # type: ignore
            self.normals_seed = self.normals_seed / torch.norm(
                self.normals_seed, dim=-1, keepdim=True
            )
            normals = torch.nn.Parameter(self.normals_seed.detach())
            scales = torch.log(avg_dist.repeat(1, 3))
            scales[:, 2] = 0  # torch.log((avg_dist / 10)[:, 0])
            scales = torch.nn.Parameter(scales)
            quats = torch.zeros(len(self.normals_seed), 4)
            z_vector = torch.tensor(
                [0, 0, 1], dtype=torch.float, device=self.normals_seed.device
            )
            for i in tqdm(
                range(len(self.normals_seed)),
                desc="Initialising normals... (slow) this operation is not batched yet",
            ):
                rotation_matrix = rotate_vector_to_vector(
                    z_vector, self.normals_seed[i]
                )
                quaternion = matrix_to_quaternion(rotation_matrix)
                quats[i, :] = quaternion
            quats = torch.nn.Parameter(quats)
        else:
            scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
            quats = torch.nn.Parameter(random_quat_tensor(num_points))

            # init random normals based on the above scales and quats
            normals = F.one_hot(torch.argmin(scales, dim=-1), num_classes=3).float()
            rots = quat_to_rotmat(quats)
            normals = torch.bmm(rots, normals[:, :, None]).squeeze(-1)
            normals = F.normalize(normals, dim=1)
            normals = torch.nn.Parameter(normals.detach())

        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
                "normals": normals,
            }
        )

        if self.config.use_sdf_loss:
            self._knn = knn_sk(
                x=self.means.data.to("cuda"),
                y=self.means.data.to("cuda"),
                k=self.config.knn_to_track,
            )

        self.camera_idx = 0
        self.camera = None
        if self.config.use_normal_tv_loss:
            self.tv_loss = TVLoss()

    @property
    def normals(self):
        return self.gauss_params["normals"]

    @property
    def colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.gauss_params["features_dc"])
        else:
            return torch.sigmoid(self.gauss_params["features_dc"])

    @property
    def num_points(self):
        return self.gauss_params["means"].shape[0]

    def refinement_after(self, optimizers: Optimizers, step):
        assert step == self.step
        if self.step <= self.config.warmup_length:
            return
        with torch.no_grad():
            # Offset all the opacity reset logic by refine_every so that we don't
            # save checkpoints right when the opacity is reset (saves every 2k)
            # then cull
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.config.reset_alpha_every * self.config.refine_every
            do_densification = (
                self.step < self.config.stop_split_at
                and self.step % reset_interval
                > self.num_train_data + self.config.refine_every
            )
            if do_densification:
                # then we densify
                assert (
                    self.xys_grad_norm is not None
                    and self.vis_counts is not None
                    and self.max_2Dsize is not None
                )
                avg_grad_norm = (
                    (self.xys_grad_norm / self.vis_counts)
                    * 0.5
                    * max(self.last_size[0], self.last_size[1])
                )
                high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()
                splits = (
                    self.scales.exp().max(dim=-1).values
                    > self.config.densify_size_thresh
                ).squeeze()
                if self.step < self.config.stop_screen_size_at:
                    splits |= (
                        self.max_2Dsize > self.config.split_screen_size
                    ).squeeze()
                splits &= high_grads
                nsamps = self.config.n_split_samples
                split_params = self.split_gaussians(splits, nsamps)

                dups = (
                    self.scales.exp().max(dim=-1).values
                    <= self.config.densify_size_thresh
                ).squeeze()
                dups &= high_grads
                dup_params = self.dup_gaussians(dups)
                for name, param in self.gauss_params.items():
                    self.gauss_params[name] = torch.nn.Parameter(
                        torch.cat(
                            [param.detach(), split_params[name], dup_params[name]],
                            dim=0,
                        )
                    )

                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_params["scales"][:, 0]),
                        torch.zeros_like(dup_params["scales"][:, 0]),
                    ],
                    dim=0,
                )

                split_idcs = torch.where(splits)[0]
                self.dup_in_all_optim(optimizers, split_idcs, nsamps)

                dup_idcs = torch.where(dups)[0]
                self.dup_in_all_optim(optimizers, dup_idcs, 1)

                # After a guassian is split into two new gaussians, the original one should also be pruned.
                splits_mask = torch.cat(
                    (
                        splits,
                        torch.zeros(
                            nsamps * splits.sum() + dups.sum(),
                            device=self.device,
                            dtype=torch.bool,
                        ),
                    )
                )

                deleted_mask = self.cull_gaussians(splits_mask)
            elif (
                self.step >= self.config.stop_split_at
                and self.config.continue_cull_post_densification
            ):
                deleted_mask = self.cull_gaussians()
            else:
                # if we donot allow culling post refinement, no more gaussians will be pruned.
                deleted_mask = None

            if deleted_mask is not None:
                self.remove_from_all_optim(optimizers, deleted_mask)

            if (
                self.step < self.config.stop_split_at
                and self.step % reset_interval == self.config.refine_every
            ):
                # Reset value is set to be twice of the cull_alpha_thresh
                reset_value = self.config.cull_alpha_thresh * 2.0
                self.opacities.data = torch.clamp(
                    self.opacities.data,
                    max=torch.logit(
                        torch.tensor(reset_value, device=self.device)
                    ).item(),
                )
                # reset the exp of optimizer
                optim = optimizers.optimizers["opacities"]
                param = optim.param_groups[0]["params"][0]
                param_state = optim.state[param]
                param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])

            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

            if (
                self.config.use_sdf_loss
                and self.step >= self.config.apply_sdf_loss_after_iters
                and deleted_mask is not None
            ):
                # BUG: it is possible to have NaNs
                means = torch.nan_to_num(self.means.data.detach().to("cuda"))
                start = time.time()
                self._knn = knn_sk(x=means, y=means, k=self.config.knn_to_track)
                CONSOLE.log(
                    f"Recomputing KNN took: {time.time() - start} seconds for {self.num_points} points"
                )

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in [
                "means",
                "scales",
                "quats",
                "features_dc",
                "features_rest",
                "opacities",
                "normals",
            ]
        }

    def get_outputs(
        self, camera: Cameras
    ) -> Dict[str, Union[torch.Tensor, List[Tensor]]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        # get the background color
        if self.training:
            if self.config.background_color == "random":
                background = torch.rand(3, device=self.device)
            elif self.config.background_color == "white":
                background = torch.ones(3, device=self.device)
            elif self.config.background_color == "black":
                background = torch.zeros(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        else:
            if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
                background = renderers.BACKGROUND_COLOR_OVERRIDE.to(self.device)
            else:
                background = self.background_color.to(self.device)

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return {
                    "rgb": background.repeat(
                        int(camera.height.item()), int(camera.width.item()), 1
                    )
                }
        else:
            crop_ids = None
        camera_downscale = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_downscale)
        # shift the camera to center of scene looking at center
        R = camera.camera_to_worlds[0, :3, :3]  # 3 x 3
        T = camera.camera_to_worlds[0, :3, 3:4]  # 3 x 1
        # flip the z and y axes to align with gsplat conventions
        R_edit = torch.diag(
            torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype)
        )
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        # calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        fovx = 2 * math.atan(camera.width / (2 * camera.fx))
        fovy = 2 * math.atan(camera.height / (2 * camera.fy))
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat(
            (features_dc_crop[:, None, :], features_rest_crop), dim=1
        )

        BLOCK_WIDTH = (
            16  # this controls the tile size of rasterization, 16 is a good default
        )
        self.xys, self.depths, self.radii, self.conics, self.comp, self.num_tiles_hit, cov3d = project_gaussians(  # type: ignore
            means_crop,
            torch.exp(scales_crop),
            1,
            quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            viewmat.squeeze()[:3, :],
            camera.fx.item(),
            camera.fy.item(),
            cx,
            cy,
            H,
            W,
            BLOCK_WIDTH,
        )  # type: ignore
        if (self.radii).sum() == 0:
            return {
                "rgb": background.repeat(
                    int(camera.height.item()), int(camera.width.item()), 1
                )
            }

        # rescale the camera back to original dimensions before returning
        camera.rescale_output_resolution(camera_downscale)

        if (self.radii).sum() == 0:
            rgb = background.repeat(H, W, 1)
            depth = background.new_ones(*rgb.shape[:2], 1) * 10
            accumulation = background.new_zeros(*rgb.shape[:2], 1)

            return {
                "rgb": rgb,
                "depth": depth,
                "accumulation": accumulation,
                "background": background,
            }

        # Important to allow xys grads to populate properly
        if self.training:
            self.xys.retain_grad()

        if self.config.sh_degree > 0:
            viewdirs = (
                means_crop.detach() - camera.camera_to_worlds.detach()[..., :3, 3]
            )  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors_crop)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
        else:
            rgbs = torch.sigmoid(colors_crop[:, 0, :])

        assert (self.num_tiles_hit > 0).any()  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        opacities = None
        if self.config.rasterize_mode == "antialiased":
            opacities = torch.sigmoid(opacities_crop) * self.comp[:, None]
        elif self.config.rasterize_mode == "classic":
            opacities = torch.sigmoid(opacities_crop)
        else:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.use_binary_opacities and self.step > self.config.warmup_length:
            skip_steps = self.config.reset_alpha_every * self.config.refine_every
            margin = 200
            if not self.step % skip_steps == 0 and self.step % skip_steps not in range(
                1, margin + 1
            ):
                opacities = torch.where(
                    opacities >= self.config.binary_opacities_threshold,
                    torch.ones_like(opacities),
                    torch.zeros_like(opacities),
                )

        rgb, alpha = rasterize_gaussians(  # type: ignore
            self.xys,
            self.depths,
            self.radii,
            self.conics,
            self.num_tiles_hit,  # type: ignore
            rgbs,
            opacities,
            H,
            W,
            BLOCK_WIDTH,
            background=background,
            return_alpha=True,
        )  # type: ignore
        alpha = alpha[..., None]
        rgb = torch.clamp(rgb, max=1.0)  # type: ignore

        # depth image
        depth_im = rasterize_gaussians(  # type: ignore
            self.xys,
            self.depths,
            self.radii,
            self.conics,
            self.num_tiles_hit,
            self.depths[:, None].repeat(1, 3),
            opacities,
            H,
            W,
            BLOCK_WIDTH,
            background=torch.zeros(3, device=self.device),
        )[..., 0:1]
        depth_im = torch.where(alpha > 0, depth_im / alpha, depth_im.detach().max())

        # visible gaussians
        self.vis_indices = torch.where(self.radii > 0)[0]

        normals_im = torch.full(rgb.shape, 0.0)
        if self.config.predict_normals:
            quats_crop = quats_crop / quats_crop.norm(dim=-1, keepdim=True)
            normals = F.one_hot(
                torch.argmin(scales_crop, dim=-1), num_classes=3
            ).float()
            rots = quat_to_rotmat(quats_crop)
            normals = torch.bmm(rots, normals[:, :, None]).squeeze(-1)
            normals = F.normalize(normals, dim=1)
            if True:
                viewdirs = (
                    -means_crop.detach() + camera.camera_to_worlds.detach()[..., :3, 3]
                )
                viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
                dots = (normals * viewdirs).sum(-1)
                negative_dot_indices = dots < 0
                normals[negative_dot_indices] = -normals[negative_dot_indices]
            # update parameter group normals
            self.gauss_params["normals"] = normals
            # convert normals from world space to camera space
            normals = normals @ camera.camera_to_worlds.squeeze(0)[:3, :3]
            normals_im: Tensor = rasterize_gaussians(  # type: ignore
                self.xys,
                self.depths,
                self.radii,
                self.conics,
                self.num_tiles_hit,
                normals,
                torch.sigmoid(opacities_crop),
                H,
                W,
                BLOCK_WIDTH,
                # TODO: what should the background for normals be
            )
            # convert normals from [-1,1] to [0,1]
            normals_im = normals_im / normals_im.norm(dim=-1, keepdim=True)
            normals_im = (normals_im + 1) / 2

        if hasattr(camera, "metadata"):
            if camera.metadata is not None and "cam_idx" in camera.metadata:
                self.camera_idx = camera.metadata["cam_idx"]  # type: ignore
        self.camera = camera

        return {
            "rgb": rgb,
            "depth": depth_im,
            "normal": normals_im,
            "accumulation": alpha,
            "background": background,
        }

    def get_loss_dict(
        self, outputs, batch, metrics_dict=None
    ) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        loss_dict = super().get_loss_dict(
            outputs=outputs, batch=batch, metrics_dict=metrics_dict
        )
        main_loss = loss_dict["main_loss"]
        scale_reg = loss_dict["scale_reg"]

        gt_img = self.get_gt_img(batch["image"])

        # minimum to reasonable level
        gt_img = self.get_gt_img(batch["image"]).clamp(min=10 / 255.0)
        pred_img = outputs["rgb"]
        depth_out = outputs["depth"]

        if "sensor_depth" in batch:
            sensor_depth_gt = self.get_gt_img(batch["sensor_depth"])
        if "mono_depth" in batch:
            mono_depth_gt = self.get_gt_img(batch["mono_depth"])
        if "normal" in batch:
            batch["normal"] = self.get_gt_img(batch["normal"])

        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            assert batch["mask"].shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            mask = batch["mask"].to(self.device)
            depth_out = depth_out * mask
            if "sensor_depth" in batch:
                sensor_depth_gt = sensor_depth_gt * mask
            if "mono_depth" in batch:
                mono_depth_gt = mono_depth_gt * mask
            if "normal" in batch:
                batch["normal"] = batch["normal"] * mask
            if "normal" in outputs:
                outputs["normal"] = outputs["normal"] * mask

        # RGB loss
        rgb_loss = main_loss

        # Depth Loss
        depth_loss = 0
        if self.config.use_depth_loss:
            if "sensor_depth" in batch and self.config.sensor_depth_lambda > 0.0:
                valid_gt_mask = sensor_depth_gt > self.config.depth_tolerance
                if self.config.depth_loss_type == DepthLossType.EdgeAwareLogL1:
                    sensor_depth_loss = self.depth_loss(
                        depth_out, sensor_depth_gt.float(), gt_img, valid_gt_mask
                    )
                    depth_loss += self.config.sensor_depth_lambda * sensor_depth_loss

                else:
                    sensor_depth_loss = self.depth_loss(
                        depth_out[valid_gt_mask], sensor_depth_gt[valid_gt_mask].float()
                    )

                    depth_loss += self.config.sensor_depth_lambda * sensor_depth_loss

            if "mono_depth" in batch and self.config.mono_depth_lambda > 0.0:
                valid_gt_mask = mono_depth_gt > 0.0
                if self.config.depth_loss_type == DepthLossType.EdgeAwareLogL1:
                    valid_gt_mask = mono_depth_gt > self.config.depth_tolerance
                    mono_depth_loss = self.depth_loss(
                        depth_out, mono_depth_gt.float(), gt_img, valid_gt_mask
                    )
                    depth_loss += self.config.mono_depth_lambda * mono_depth_loss
                else:
                    mono_depth_loss = self.depth_loss(
                        depth_out[valid_gt_mask], mono_depth_gt[valid_gt_mask].float()
                    )
                    depth_loss += self.config.mono_depth_lambda * mono_depth_loss

        # Smooth loss
        if self.config.use_depth_smooth_loss:
            if self.config.smooth_loss_type == DepthLossType.TV:
                smooth_loss = self.smooth_loss(depth_out)
                depth_loss += self.config.smooth_loss_lambda * smooth_loss
            elif self.config.smooth_loss_type == DepthLossType.EdgeAwareTV:
                assert depth_out.shape[:2] == outputs["rgb"].shape[:2]
                smooth_loss = self.smooth_loss(depth_out, gt_img)
                depth_loss += self.config.smooth_loss_lambda * smooth_loss

        if self.config.use_depth_loss and depth_loss == 0 and self.step % 100 == 0:
            CONSOLE.log(
                "WARNING: you have enabled depth loss but depth loss is still ZERO. Remember to set --pipeline.model.sensor-depth-lambda and/or --pipeleine.model.mono-depth-lambda > 0"
            )

        # Normal loss
        normal_loss = 0
        if self.config.use_normal_loss:
            pred_normal = outputs["normal"]

            if "normal" in batch and self.config.normal_supervision == "mono":
                gt_normal = batch["normal"]
            elif self.config.normal_supervision == "depth":
                c2w = self.camera.camera_to_worlds.squeeze(0).detach()
                c2w = c2w @ torch.diag(
                    torch.tensor([1, -1, -1, 1], device=c2w.device, dtype=c2w.dtype)
                )
                gt_normal = normal_from_depth_image(
                    depths=depth_out.detach(),
                    fx=self.camera.fx.item(),
                    fy=self.camera.fy.item(),
                    cx=self.camera.cx.item(),
                    cy=self.camera.cy.item(),
                    img_size=(self.camera.width.item(), self.camera.height.item()),
                    c2w=torch.eye(4, dtype=torch.float, device=depth_out.device),
                    device=self.device,
                    smooth=False,
                )
                gt_normal = gt_normal @ torch.diag(
                    torch.tensor(
                        [1, -1, -1], device=depth_out.device, dtype=depth_out.dtype
                    )
                )
                gt_normal = (1 + gt_normal) / 2
            else:
                CONSOLE.log(
                    "WARNING: You have enabled normal supervision with monocular normals but none were found."
                )
                CONSOLE.log(
                    "WARNING: Remember to first generate normal maps for your dataset using the normals_from_pretrain.py script."
                )
                quit()
            if gt_normal is not None:
                # normal map loss
                normal_loss += torch.abs(gt_normal - pred_normal).mean()
                if self.config.use_normal_cosine_loss:
                    from dn_splatter.metrics import mean_angular_error

                    normal_loss += mean_angular_error(
                        pred=(pred_normal.permute(2, 0, 1) - 1) / 2,
                        gt=(gt_normal.permute(2, 0, 1) - 1) / 2,
                    ).mean()
            if self.config.use_normal_tv_loss:
                normal_loss += self.tv_loss(pred_normal)

        if self.config.two_d_gaussians:
            # loss to minimise gaussian scale corresponding to normal direction
            normal_loss += torch.min(torch.exp(self.scales), dim=1, keepdim=True)[
                0
            ].mean()

        sparse_loss = 0
        if (
            self.config.use_sparse_loss
            and self.step % self.config.sparse_loss_steps == 0
        ):  # type: ignore
            skip_steps = self.config.reset_alpha_every * self.config.refine_every
            margin = 100
            if not self.step % skip_steps == 0 and self.step % skip_steps not in range(
                1, margin + 1
            ):
                opacities = torch.sigmoid(self.opacities[self.vis_indices])
                sparse_loss = (
                    -opacities * torch.log(opacities + 1e-10)
                    - (1 - opacities) * torch.log(1 - opacities + 1e-10)
                ).mean()
                sparse_loss *= self.config.sparse_lambda

        sdf_loss = 0
        if (
            self.config.use_sdf_loss
            and self.step > self.config.apply_sdf_loss_after_iters
            and self.step % self.config.apply_sdf_loss_iters == 0
        ):
            if self.num_points > self.config.num_sdf_samples:
                num_samples = self.num_points
            else:
                num_samples = self.config.num_sdf_samples

            start = time.time()
            # sample points according to gaussian distribution on surface
            samples, indices = self.sample_points_in_gaussians(
                num_samples=num_samples, vis_indices=self.vis_indices
            )
            # print(f"sample_points_in_gaussians took { time.time() - start} s")
            start = time.time()
            # query closest gaussians to sampled points
            with torch.no_grad():
                closest_gaussians = self._knn[indices]
            # print(f"sampling _knn took { time.time() - start} s")
            start = time.time()
            # compute current sdf estimates of samples
            current_sdfs = self.get_sdf(
                sdf_samples=samples,
                closest_gaussians=closest_gaussians,
                vis_indices=self.vis_indices,
            )
            # print(f"get_sdf took { time.time() - start} s")
            start = time.time()
            # estimate ideal sdfs
            ideal_sdfs, valid_indices = self.get_ideal_sdf(
                sdf_samples=samples.clone().detach(),
                depth=depth_out.clone().detach(),
                camera=self.camera,  # type: ignore
                mask=batch["mask"] if "mask" in batch else None,
            )
            ideal_sdfs = torch.abs(ideal_sdfs)
            # print(f"get_ideal_sdf took { time.time() - start} s")
            current_sdfs = current_sdfs[valid_indices]

            weight = self.get_sdf_loss_weight(valid_indices)

            sdf_loss = (torch.abs(ideal_sdfs - current_sdfs) / (weight + 1e-5)).mean()

        main_loss = (
            rgb_loss
            + depth_loss
            + self.config.normal_lambda * normal_loss
            + sparse_loss
            + self.config.sdf_loss_lambda * sdf_loss
        )

        return {"main_loss": main_loss, "scale_reg": scale_reg}

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        d = self._get_downscale_factor()
        if d > 1:
            # use torchvision to resize
            newsize = (batch["image"].shape[0] // d, batch["image"].shape[1] // d)
            gt_img = TF.resize(
                batch["image"].permute(2, 0, 1), newsize, antialias=None
            ).permute(1, 2, 0)

            if "sensor_depth" in batch:
                depth_size = (
                    batch["sensor_depth"].shape[0] // d,
                    batch["sensor_depth"].shape[1] // d,
                )
                sensor_depth_gt = TF.resize(
                    batch["sensor_depth"].permute(2, 0, 1), depth_size, antialias=None
                ).permute(1, 2, 0)
        else:
            gt_img = batch["image"]
            if "sensor_depth" in batch:
                sensor_depth_gt = batch["sensor_depth"]

        metrics_dict = {}
        gt_rgb = gt_img.to(self.device)  # RGB or RGBA image
        predicted_rgb = outputs["rgb"]

        # comment out for now, as it will slow down the training speed.
        (psnr, ssim, lpips) = self.rgb_metrics(
            gt_rgb.permute(2, 0, 1).unsqueeze(0),
            predicted_rgb.permute(2, 0, 1).unsqueeze(0).to(self.device),
        )
        rgb_mse = self.mse_loss(gt_rgb.permute(2, 0, 1), predicted_rgb.permute(2, 0, 1))
        metrics_dict = {
            "rgb_mse": float(rgb_mse),
            "rgb_psnr": float(psnr.item()),
            "rgb_ssim": float(ssim),
            "rgb_lpips": float(lpips),
        }

        metrics_dict["gaussian_count"] = self.num_points

        if self.config.use_depth_loss and "sensor_depth" in batch:
            depth_out = outputs["depth"]
            (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3) = self.depth_metrics(
                depth_out.permute(2, 0, 1), sensor_depth_gt.permute(2, 0, 1)
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

        # track scales
        metrics_dict.update(
            {"avg_min_scale": torch.nanmean(torch.exp(self.scales[..., -1]))}
        )

        return metrics_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Main function for eval/test images

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """

        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs[
            "rgb"
        ]  # Blended with background (black if random background)
        predicted_depth = outputs["depth"]

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

        predicted_depth = outputs["depth"]
        if "sensor_depth" in batch:
            gt_depth = batch["sensor_depth"].to(self.device)

            # TODO: remove this once the OpenCV bug is fixed, need gt_depth align with predicted depth
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

        images_dict = {"img": combined_rgb, "depth": predicted_depth}

        return metrics_dict, images_dict

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb
            )
        )
        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION], self.after_train
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.config.refine_every,
                args=[training_callback_attributes.optimizers],
            )
        )

        return cbs

    def sample_points_in_gaussians(
        self,
        num_samples: int,
        vis_indices: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Sample points in world space based on gaussian distributions

        Args:
            num samples
            visible indices

        Returns:
            random points and their gaussian indices
        """

        if vis_indices is not None:
            vis_scales = torch.exp(self.scales[vis_indices])
        else:
            vis_scales = torch.exp(self.scales)

        areas = vis_scales[..., 0] * vis_scales[..., 1] * vis_scales[..., 2]

        areas = areas.abs()
        cum_probs = areas.cumsum(dim=-1) / areas.sum(dim=-1, keepdim=True)

        # This picks which gaussians to sample based on their extent/volume in 3d space
        random_indices = torch.multinomial(
            cum_probs, num_samples=num_samples, replacement=True
        )

        # random indices from vis_indices
        if vis_indices is not None:
            random_indices = vis_indices[random_indices]

        centered_samples = torch.randn(
            size=(len(random_indices), 3), device=self.device, dtype=torch.float
        )  # (N_samples, 3)

        scaled_samples = (
            torch.exp(self.scales[random_indices]) * centered_samples
        )  # scale based on extents
        quats = self.quats[random_indices] / self.quats[random_indices].norm(
            dim=-1, keepdim=True
        )
        rots = quat_to_rotmat(quats)
        # rotate random points from gaussian frame to world frame based on current rotation matrices
        random_points = (
            self.means[random_indices]
            + torch.bmm(rots, scaled_samples[..., None]).squeeze()
        )
        return random_points, random_indices

    def get_ideal_sdf(
        self,
        sdf_samples: Tensor,
        depth: Tensor,
        camera: Cameras,
        mask: Optional[Tensor] = None,
        min_depth: float = 0.01,
    ) -> Tuple[Tensor, Tensor]:
        """Project sampled points into camera frame and compute ideal sdf estimate

        Args:
            sdf_samples: current point samples
            depth: current rendered depth map
            camera: current camera frame
            tolerance: minimum depth

        Returns:
            ideal_sdf, valid indices
        """
        c2w = camera.camera_to_worlds.squeeze(0)
        c2w = c2w @ torch.diag(
            torch.tensor([1, -1, -1, 1], device=c2w.device, dtype=c2w.dtype)
        )

        projections = project_pix(
            sdf_samples,
            fx=camera.fx.item(),
            fy=camera.fx.item(),
            cx=camera.cx.item(),
            cy=camera.cy.item(),
            c2w=c2w,
            device=self.device,
            return_z_depths=True,
        )

        projections[:, :2] = uv = torch.floor(projections[:, :2]).long()

        valid_indices = valid_uv_indices = (
            (uv[:, 0] > 0)
            & (uv[:, 0] < camera.width.item())
            & (uv[:, 1] > 0)
            & (uv[:, 1] < camera.height.item())
        )

        if mask is not None:
            valid_indices = valid_uv_indices.detach().clone()
            valid_indices[valid_uv_indices] = mask[
                uv[valid_uv_indices, 1], uv[valid_uv_indices, 0]
            ][..., 0]

        z_depth_points = projections[valid_indices][..., -1]
        z_depth_ideal = depth[uv[valid_indices, 1], uv[valid_indices, 0], 0]

        return z_depth_ideal - z_depth_points, valid_indices

    def get_closest_gaussians(self, samples) -> torch.Tensor:
        """Get closest gaussians to samples

        Args:
            samples: tensor of 3d point samples

        Returns:
            knn gaussians
        """
        closest_gaussians = knn_sk(
            x=self.means.data.to("cuda"),
            y=samples.to("cuda"),
            k=self.config.knn_to_track,
        )
        return closest_gaussians

    def get_density(
        self,
        sdf_samples: Tensor,
        closest_gaussians: Optional[Tensor] = None,
        vis_indices: Optional[Tensor] = None,
    ):
        """Estimate current density at sample points based on current gaussian distributions

        Args:
            sdf_samples: current point samples
            closest_gaussians: closest knn gaussians per current point sample
            vis_indices: visibility mask

        Returns:
            densities
        """
        if closest_gaussians is None:
            closest_gaussians = self.get_closest_gaussians(samples=sdf_samples)
        closest_gaussians_idx = closest_gaussians
        closest_gaussian_centers = self.means[closest_gaussians]

        closest_gaussian_inv_scaled_rotation = scale_rot_to_inv_cov3d(
            scale=torch.exp(self.scales[closest_gaussians_idx]),
            quat=self.quats[closest_gaussians_idx],
            return_sqrt=True,
        )  # sigma^-1
        closest_gaussian_opacities = torch.sigmoid(
            self.opacities[closest_gaussians_idx]
        )

        # Compute the density field as a sum of local gaussian opacities
        # (num_samples, knn, 3)
        dist = sdf_samples[:, None, :] - closest_gaussian_centers
        # (num_samples, knn, 3, 1)
        man_distance = (
            closest_gaussian_inv_scaled_rotation.transpose(-1, -2) @ dist[..., None]
        )
        # Mahalanobis distance
        # (num_samples, knn)
        neighbor_opacities = (
            (man_distance[..., 0] * man_distance[..., 0])
            .sum(dim=-1)
            .clamp(min=0.0, max=1e8)
        )
        # (num_samples, knn)
        neighbor_opacities = closest_gaussian_opacities[..., 0] * torch.exp(
            -1.0 / 2 * neighbor_opacities
        )
        densities = neighbor_opacities.sum(dim=-1)  # (num_samples,)

        # BUG: this seems to be quite sensitive to the EPS
        density_mask = densities >= 1.0
        densities[density_mask] = densities[density_mask] / (
            densities[density_mask].detach() + 1e-5
        )
        opacity_min_clamp = 1e-4
        clamped_densities = densities.clamp(min=opacity_min_clamp)

        return clamped_densities

    def get_sdf(
        self,
        sdf_samples: Tensor,
        closest_gaussians: Optional[Tensor] = None,
        vis_indices: Optional[Tensor] = None,
    ) -> Tensor:
        """Estimate current sdf values at sample points based on current gaussian distributions

        Args:
            sdf_samples: current point samples
            closest_gaussians: closest knn gaussians per current point sample
            vis_indices: visibility mask

        Returns:
            sdf values
        """
        densities = self.get_density(
            sdf_samples=sdf_samples,
            closest_gaussians=closest_gaussians,
            vis_indices=vis_indices,
        )
        sdf_values = 1 * torch.sqrt(-2.0 * torch.log(densities))
        return sdf_values

    def get_sdf_weight(
        self,
        closest_gaussians_idx: Tensor,
    ):
        # weight by scale
        return torch.exp(self.scales).min(dim=-1)[0][closest_gaussians_idx].mean(dim=1)

    @torch.no_grad()
    def get_sdf_loss_weight(
        self, valid_indices: Tensor, mode: Literal["area", "std"] = "std"
    ):
        """Regularizer for the sdf loss

        Args:
            valid_indices: valid indices
            mode: compute weight as the area of the gaussians or as the standard deviation

        Returns:
            sdf_loss_weight
        """
        if mode == "area":
            # use areas as a weight
            vis_scales = torch.exp(self.scales[valid_indices]).clone().detach()
            max_indices = torch.topk(vis_scales, k=2, dim=-1)[1]
            max_values = torch.gather(vis_scales, dim=-1, index=max_indices)
            areas = torch.prod(max_values, dim=-1)
            return areas

        if mode == "std":
            # use gaussian standard deviations as a weight
            viewdirs = (
                -self.means[valid_indices].detach()
                + self.camera.camera_to_worlds.detach()[..., :3, 3]
            )
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            quats = self.quats[valid_indices] / self.quats[valid_indices].norm(
                dim=-1, keepdim=True
            )
            inv_rots = quat_to_rotmat(invert_quaternion(quat=quats))
            gaussian_standard_deviations = (
                torch.exp(self.scales[valid_indices])
                * torch.bmm(inv_rots, viewdirs[..., None])[..., 0]
            ).norm(dim=-1)
            return gaussian_standard_deviations

    @torch.no_grad()
    def compute_level_surface_points(
        self,
        camera: Cameras,
        num_samples: int,
        mask: Optional[Tensor] = None,
        surface_levels: Tuple[float, float, float] = (0.1, 0.3, 0.5),
        return_normal: Literal[
            "analytical", "closest_gaussian", "average"
        ] = "closest_gaussian",
    ) -> Tensor:
        """Compute level surface intersections and their normals

        Args:
            camera: current camera object to find surface intersections
            num_samples: number of samples per camera to target
            mask: optional mask per camera
            surface_levels: surface levels to compute
            return_normal: normal return mode

        Returns:
            level surface intersection points, normals
        """
        c2w = camera.camera_to_worlds.squeeze(0)
        c2w = c2w @ torch.diag(
            torch.tensor([1, -1, -1, 1], device=c2w.device, dtype=c2w.dtype)
        )
        outputs = self.get_outputs(camera=camera)
        assert "depth" in outputs
        depth: Tensor = outputs["depth"]  # type: ignore
        rgb: Tensor = outputs["rgb"]  # type: ignore
        W, H = camera.width.item(), camera.height.item()

        # backproject from depth map
        points, colors = get_colored_points_from_depth(
            depths=depth,
            rgbs=rgb,
            fx=camera.fx.item(),
            fy=camera.fy.item(),
            cx=camera.cx.item(),
            cy=camera.cy.item(),
            img_size=(W, H),  # img_size = (w,h)
            c2w=c2w,
        )
        points = points.view(H, W, -1)  # type: ignore
        colors = colors.view(H, W, 3)

        if mask is not None:
            mask = mask.to(points.device)
            points = points * mask
            depth = depth * mask

        no_depth_mask = (depth <= 0.0)[..., 0]
        points = points[~no_depth_mask]
        colors = colors[~no_depth_mask]

        # get closest gaussians
        closest_gaussians_idx = knn_sk(
            self.means.data, points, k=self.config.knn_to_track
        )

        # compute gaussian stds along ray direction
        viewdirs = -self.means.detach() + camera.camera_to_worlds.detach()[..., :3, 3]
        viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
        quats = self.quats / self.quats.norm(dim=-1, keepdim=True)
        inv_rots = quat_to_rotmat(invert_quaternion(quat=quats))
        gaussian_standard_deviations = (
            torch.exp(self.scales) * torch.bmm(inv_rots, viewdirs[..., None])[..., 0]
        ).norm(dim=-1)
        points_stds = gaussian_standard_deviations[closest_gaussians_idx][
            ..., 0
        ]  # get first closest gaussian std

        range_size = 3
        n_points_in_range = 21
        n_points_per_pass = 2_000_000

        # sampling on ray
        points_range = (
            torch.linspace(-range_size, range_size, n_points_in_range)
            .to(self.device)
            .view(1, -1, 1)
        )  # (1, n_points_in_range, 1)
        points_range = points_range * points_stds[..., None, None].expand(
            -1, n_points_in_range, 1
        )  # (n_points, n_points_in_range, 1)
        camera_to_samples = torch.nn.functional.normalize(
            points - camera.camera_to_worlds.detach()[..., :3, 3], dim=-1
        )  # (n_points, 3)
        samples = (
            points[:, None, :] + points_range * camera_to_samples[:, None, :]
        ).view(
            -1, 3
        )  # (n_points * n_points_in_range, 3)
        samples_closest_gaussians_idx = (
            closest_gaussians_idx[:, None, :]
            .expand(-1, n_points_in_range, -1)
            .reshape(-1, self.config.knn_to_track)
        )

        densities = torch.zeros(len(samples), dtype=torch.float, device=self.device)
        gaussian_strengths = torch.sigmoid(self.opacities)
        gaussian_centers = self.means
        gaussian_inv_scaled_rotation = scale_rot_to_inv_cov3d(
            scale=torch.exp(self.scales), quat=self.quats, return_sqrt=True
        )

        # compute densities along rays
        for i in range(0, len(samples), n_points_per_pass):
            i_start = i
            i_end = min(len(samples), i + n_points_per_pass)

            pass_closest_gaussians_idx = samples_closest_gaussians_idx[i_start:i_end]

            closest_gaussian_centers = gaussian_centers[pass_closest_gaussians_idx]
            closest_gaussian_inv_scaled_rotation = gaussian_inv_scaled_rotation[
                pass_closest_gaussians_idx
            ]

            closest_gaussian_strengths = gaussian_strengths[pass_closest_gaussians_idx]
            shift = samples[i_start:i_end, None] - closest_gaussian_centers
            man_distance = (
                closest_gaussian_inv_scaled_rotation.transpose(-1, -2)
                @ shift[..., None]
            )
            neighbor_opacities = (
                (man_distance[..., 0] * man_distance[..., 0])
                .sum(dim=-1)
                .clamp(min=0.0, max=1e8)
            )
            neighbor_opacities = closest_gaussian_strengths[..., 0] * torch.exp(
                -1.0 / 2 * neighbor_opacities
            )
            pass_densities = neighbor_opacities.sum(dim=-1)

            pass_density_mask = pass_densities >= 1.0
            pass_densities[pass_density_mask] = pass_densities[pass_density_mask] / (
                pass_densities[pass_density_mask].detach() + 1e-5
            )
            densities[i_start:i_end] = pass_densities

        densities = densities.reshape(
            -1, n_points_in_range
        )  # (num_samples, n_points_in_range (21))

        all_outputs = {}
        for surface_level in surface_levels:
            outputs = {}

            under_level = densities - surface_level < 0
            above_level = densities - surface_level > 0

            _, first_point_above_level = above_level.max(dim=-1, keepdim=True)
            empty_pixels = ~under_level[..., 0] + (first_point_above_level[..., 0] == 0)

            # depth as level point
            valid_densities = densities[~empty_pixels]
            valid_range = points_range[~empty_pixels][..., 0]
            valid_first_point_above_level = first_point_above_level[~empty_pixels]

            first_value_above_level = valid_densities.gather(
                dim=-1, index=valid_first_point_above_level
            ).view(-1)
            value_before_level = valid_densities.gather(
                dim=-1, index=valid_first_point_above_level - 1
            ).view(-1)

            first_t_above_level = valid_range.gather(
                dim=-1, index=valid_first_point_above_level
            ).view(-1)
            t_before_level = valid_range.gather(
                dim=-1, index=valid_first_point_above_level - 1
            ).view(-1)

            intersection_t = (surface_level - value_before_level) / (
                first_value_above_level - value_before_level
            ) * (first_t_above_level - t_before_level) + t_before_level
            intersection_points = (
                points[~empty_pixels]
                + intersection_t[:, None] * camera_to_samples[~empty_pixels]
            )
            intersection_colors = colors[~empty_pixels]

            # normal
            if return_normal == "analytical":
                points_closest_gaussians_idx = closest_gaussians_idx[~empty_pixels]
                closest_gaussian_centers = gaussian_centers[
                    points_closest_gaussians_idx
                ]
                closest_gaussian_inv_scaled_rotation = gaussian_inv_scaled_rotation[
                    points_closest_gaussians_idx
                ]
                closest_gaussian_strengths = gaussian_strengths[
                    points_closest_gaussians_idx
                ]
                shift = intersection_points[:, None] - closest_gaussian_centers
                man_distance = (
                    closest_gaussian_inv_scaled_rotation.transpose(-1, -2)
                    @ shift[..., None]
                )
                neighbor_opacities = (
                    (man_distance[..., 0] * man_distance[..., 0])
                    .sum(dim=-1)
                    .clamp(min=0.0, max=1e8)
                )
                neighbor_opacities = closest_gaussian_strengths[..., 0] * torch.exp(
                    -1.0 / 2 * neighbor_opacities
                )
                density_grad = (
                    neighbor_opacities[..., None]
                    * (closest_gaussian_inv_scaled_rotation @ man_distance)[..., 0]
                ).sum(dim=-2)
                intersection_normals = -torch.nn.functional.normalize(
                    density_grad, dim=-1
                )
            elif return_normal == "closest_gaussian":
                points_closest_gaussians_idx = closest_gaussians_idx[~empty_pixels]
                intersection_normals = self.normals[
                    points_closest_gaussians_idx[..., 0]
                ]
            else:
                raise NotImplementedError

            # sample pixels for this frame
            assert intersection_points.shape[0] == intersection_normals.shape[0]
            indices = random.sample(
                range(intersection_points.shape[0]),
                num_samples
                if num_samples < intersection_points.shape[0]
                else intersection_points.shape[0],
            )
            samples_mask = torch.tensor(indices, device=points.device)
            intersection_points = intersection_points[samples_mask]
            intersection_normals = intersection_normals[samples_mask]
            intersection_colors = intersection_colors[samples_mask]

            outputs["points"] = intersection_points
            outputs["normals"] = intersection_normals
            outputs["colors"] = intersection_colors
            all_outputs[surface_level] = outputs

        return all_outputs

    def get_density_grad(
        self,
        samples: Tensor,
        num_closest_gaussians: Optional[int] = None,
        closest_gaussians: Optional[Tensor] = None,
    ) -> Tensor:
        """Estimate analytical normal from the gradient of the density

        Args:
            samples: point samples to query density and compute grad density

        Returns:
            grad_density
        """
        if closest_gaussians is None:
            closest_gaussians = self.get_closest_gaussians(samples=samples)
        if num_closest_gaussians is not None:
            assert num_closest_gaussians >= 1
            closest_gaussians = closest_gaussians[..., :num_closest_gaussians]

        closest_gaussians_idx = closest_gaussians
        closest_gaussian_centers = self.means[closest_gaussians]
        closest_gaussian_inv_scaled_rotation = scale_rot_to_inv_cov3d(
            scale=torch.exp(self.scales[closest_gaussians_idx]),
            quat=self.quats[closest_gaussians_idx],
            return_sqrt=True,
        )
        dist = samples[:, None, :] - closest_gaussian_centers
        # (num_samples, knn, 3, 1)
        man_distance = (
            closest_gaussian_inv_scaled_rotation.transpose(-1, -2) @ dist[..., None]
        )
        # Mahalanobis distance
        # (num_samples, knn)
        neighbor_opacities = (
            (man_distance[..., 0] * man_distance[..., 0])
            .sum(dim=-1)
            .clamp(min=0.0, max=1e8)
        )
        density_grad = (
            neighbor_opacities[..., None]
            * (closest_gaussian_inv_scaled_rotation @ man_distance)[..., 0]
        ).sum(dim=-2)
        # normal is the negative of the grad
        density_grad = -torch.nn.functional.normalize(density_grad, dim=-1)
        return density_grad


def random_quat_tensor(N, **kwargs):
    u = torch.rand(N, **kwargs)
    v = torch.rand(N, **kwargs)
    w = torch.rand(N, **kwargs)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def rotate_vector_to_vector(v1: Tensor, v2: Tensor):
    """
    Returns a rotation matrix that rotates v1 to align with v2.
    """
    assert v1.dim() == v2.dim()
    assert v1.dim() == 1
    u = v1 / torch.norm(v1)
    Ru = v2 / torch.norm(v2)
    I: Tensor = torch.eye(3, device=v1.device)  # noqa: E741
    # the cos angle between the vectors
    c = torch.dot(u, Ru)
    eps = 1.0e-10
    if torch.abs(c - 1.0) < eps:
        # same direction
        return I
    elif torch.abs(c + 1.0) < eps:
        # opposite direction
        return -I
    else:
        # the cross product matrix of a vector to rotate around
        K = torch.outer(Ru, u) - torch.outer(u, Ru)
        # Rodrigues' formula
        return I + K + (K @ K) / (1 + c)


def matrix_to_quaternion(matrix: Tensor):
    """
    Convert a 3x3 rotation matrix to a unit quaternion.
    """
    assert matrix.shape == (3, 3)
    trace = torch.trace(matrix)

    if trace > 0:
        S = torch.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (matrix[2, 1] - matrix[1, 2]) / S
        y = (matrix[0, 2] - matrix[2, 0]) / S
        z = (matrix[1, 0] - matrix[0, 1]) / S
    elif (matrix[0, 0] > matrix[1, 1]) and (matrix[0, 0] > matrix[2, 2]):
        S = torch.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
        w = (matrix[2, 1] - matrix[1, 2]) / S
        x = 0.25 * S
        y = (matrix[0, 1] + matrix[1, 0]) / S
        z = (matrix[0, 2] + matrix[2, 0]) / S
    elif matrix[1, 1] > matrix[2, 2]:
        S = torch.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
        w = (matrix[0, 2] - matrix[2, 0]) / S
        x = (matrix[0, 1] + matrix[1, 0]) / S
        y = 0.25 * S
        z = (matrix[1, 2] + matrix[2, 1]) / S
    else:
        S = torch.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
        w = (matrix[1, 0] - matrix[0, 1]) / S
        x = (matrix[0, 2] + matrix[2, 0]) / S
        y = (matrix[1, 2] + matrix[2, 1]) / S
        z = 0.25 * S

    return torch.tensor([w, x, y, z], dtype=matrix.dtype, device=matrix.device)


def scale_rot_to_inv_cov3d(scale, quat, return_sqrt=False):
    assert scale.shape[-1] == 3, scale.shape
    assert quat.shape[-1] == 4, quat.shape
    assert scale.shape[:-1] == quat.shape[:-1], (scale.shape, quat.shape)
    scale = 1.0 / scale.clamp(min=1e-3)
    R = quat_to_rotmat(quat)  # (..., 3, 3)
    M = R * scale[..., None, :]  # (..., 3, 3)
    if return_sqrt:
        return M
    return M @ M.transpose(-1, -2)  # (..., 3, 3)


def invert_quaternion(quat: Tensor):
    """Invert quaternion in wxyz convention

    Args:
        quaternion: quat shape (..., 4), with real part first

    Returns:
        inverse quat, shape (..., 4).
    """
    scaling = torch.tensor([1, -1, -1, -1], device=quat.device)
    return quat * scaling
