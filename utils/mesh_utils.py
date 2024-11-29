#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import torch
import numpy as np
import os
import math
from tqdm import tqdm
from utils.general_utils import build_rotation, colormap
from utils.render_utils import save_img_f32, save_img_u8
from utils.image_utils import find_edges
from functools import partial
import open3d as o3d


def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy

    print(
        "post processing the mesh to have {} clusterscluster_to_kep".format(
            cluster_to_keep
        )
    )
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        (
            triangle_clusters,
            cluster_n_triangles,
            cluster_area,
        ) = mesh_0.cluster_connected_triangles()

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50)  # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=viewpoint_cam.image_width,
            height=viewpoint_cam.image_height,
            cx=viewpoint_cam.image_width / 2,
            cy=viewpoint_cam.image_height / 2,
            fx=viewpoint_cam.image_width / (2 * math.tan(viewpoint_cam.FoVx / 2.0)),
            fy=viewpoint_cam.image_height / (2 * math.tan(viewpoint_cam.FoVy / 2.0)),
        )

        extrinsic = np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj


def get_camera_coords(img_size: tuple, pixel_offset: float = 0.5):
    """Generates camera pixel coordinates [W,H]

    Returns:
        stacked coords [H*W,2] where [:,0] corresponds to W and [:,1] corresponds to H
    """

    # img size is (w,h)
    image_coords = torch.meshgrid(
        torch.arange(img_size[0]),
        torch.arange(img_size[1]),
        indexing="xy",  # W = u by H = v
    )
    image_coords = (
        torch.stack(image_coords, dim=-1) + pixel_offset
    )  # stored as (x, y) coordinates
    image_coords = image_coords.view(-1, 2)
    image_coords = image_coords.float()

    return image_coords


def get_means3d_backproj(
    depths,
    fx: float,
    fy: float,
    cx: int,
    cy: int,
    img_size: tuple,
    c2w,
    device: torch.device,
):
    """Backprojection using camera intrinsics and extrinsics

    image_coords -> (x,y,depth) -> (X, Y, depth)

    Returns:
        Tuple of (means: Tensor, image_coords: Tensor)
    """

    if depths.dim() == 3:
        depths = depths.view(-1, 1)
    elif depths.shape[-1] != 1:
        depths = depths.unsqueeze(-1).contiguous()
        depths = depths.view(-1, 1)
    if depths.dtype != torch.float:
        depths = depths.float()
        c2w = c2w.float()
    if c2w.device != device:
        c2w = c2w.to(device)

    image_coords = get_camera_coords(img_size)
    image_coords = image_coords.to(device)  # note image_coords is (H,W)

    # TODO: account for skew / radial distortion
    means3d = torch.empty(
        size=(img_size[0], img_size[1], 3), dtype=torch.float32, device=device
    ).view(-1, 3)
    means3d[:, 0] = (image_coords[:, 0] - cx) * depths[:, 0] / fx  # x
    means3d[:, 1] = (image_coords[:, 1] - cy) * depths[:, 0] / fy  # y
    means3d[:, 2] = depths[:, 0]  # z

    if c2w is None:
        c2w = torch.eye((means3d.shape[0], 4, 4), device=device)

    # to world coords
    means3d = means3d @ torch.linalg.inv(c2w[..., :3, :3]) + c2w[..., :3, 3]  # [N, 3]
    return means3d, image_coords


def get_colored_points_from_depth(
    depths,
    rgbs,
    c2w,
    fx: float,
    fy: float,
    cx: int,
    cy: int,
    img_size: tuple,
    mask,
):
    """Return colored pointclouds from depth and rgb frame and c2w. Optional masking.

    Returns:
        Tuple of (points, colors)
    """
    points, _ = get_means3d_backproj(
        depths=depths.float(),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        img_size=img_size,
        c2w=c2w.float(),
        device=depths.device,
    )
    points = points.squeeze(0)
    if mask is not None:
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, device=depths.device)
        colors = rgbs.view(-1, 3)[mask]
        points = points[mask]
    else:
        colors = rgbs.view(-1, 3)
        points = points
    return (points, colors)


def pick_indices_at_random(valid_mask, samples_per_frame):
    indices = torch.nonzero(torch.ravel(valid_mask))
    if samples_per_frame < len(indices):
        which = torch.randperm(len(indices))[:samples_per_frame]
        indices = indices[which]
    return torch.ravel(indices)


class GaussianExtractor(object):
    def __init__(self, gaussians, render, pipe, bg_color=None):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        if bg_color is None:
            bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussians = gaussians
        self.render = partial(render, pipe=pipe, bg_color=background)
        self.clean()

    @torch.no_grad()
    def clean(self):
        self.depthmaps = []
        self.alphamaps = []
        self.rgbmaps = []
        self.normals = []
        self.depth_normals = []
        self.viewpoint_stack = []

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack):
        """
        reconstruct radiance field given cameras
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        for i, viewpoint_cam in tqdm(
            enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"
        ):
            render_pkg = self.render(viewpoint_cam, self.gaussians)
            rgb = render_pkg["render"]
            alpha = render_pkg["rend_alpha"]
            normal = torch.nn.functional.normalize(render_pkg["rend_normal"], dim=0)
            depth = render_pkg["surf_depth"]
            depth_normal = render_pkg["surf_normal"]
            self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(depth.cpu())
            self.alphamaps.append(alpha.cpu())
            self.normals.append(normal.cpu())
            self.depth_normals.append(depth_normal.cpu())

        self.rgbmaps = torch.stack(self.rgbmaps, dim=0)
        self.depthmaps = torch.stack(self.depthmaps, dim=0)
        self.alphamaps = torch.stack(self.alphamaps, dim=0)
        self.depth_normals = torch.stack(self.depth_normals, dim=0)
        self.estimate_bounding_sphere()

    def estimate_bounding_sphere(self):
        """
        Estimate the bounding sphere given camera pose
        """
        from utils.render_utils import transform_poses_pca, focus_point_fn

        torch.cuda.empty_cache()
        c2ws = np.array(
            [
                np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy()))
                for cam in self.viewpoint_stack
            ]
        )
        poses = c2ws[:, :3, :] @ np.diag([1, -1, -1, 1])
        center = focus_point_fn(poses)
        self.radius = np.linalg.norm(c2ws[:, :3, 3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()
        print(f"The estimated bounding radius is {self.radius:.2f}")
        print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")

    @torch.no_grad()
    def extract_mesh_bounded(
        self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True
    ):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.

        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        print("Running tsdf volume integration ...")
        print(f"voxel_size: {voxel_size}")
        print(f"sdf_trunc: {sdf_trunc}")
        print(f"depth_truc: {depth_trunc}")

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        for i, cam_o3d in tqdm(
            enumerate(to_cam_open3d(self.viewpoint_stack)),
            desc="TSDF integration progress",
        ):
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]

            # if we have mask provided, use it
            if mask_backgrond and (self.viewpoint_stack[i].gt_alpha_mask is not None):
                depth[(self.viewpoint_stack[i].gt_alpha_mask < 0.5)] = 0

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(
                    np.asarray(
                        rgb.permute(1, 2, 0).cpu().numpy() * 255,
                        order="C",
                        dtype=np.uint8,
                    )
                ),
                o3d.geometry.Image(
                    np.asarray(depth.permute(1, 2, 0).cpu().numpy(), order="C")
                ),
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
                depth_scale=1.0,
            )

            volume.integrate(
                rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic
            )

        mesh = volume.extract_triangle_mesh()

        return mesh

    @torch.no_grad()
    def extract_mesh_unbounded(self, resolution=1024):
        """
        Experimental features, extracting meshes from unbounded scenes, not fully test across datasets.
        return o3d.mesh
        """

        def contract(x):
            mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))

        def uncontract(y):
            mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, y, (1 / (2 - mag) * (y / mag)))

        def compute_sdf_perframe(i, points, depthmap, rgbmap, normalmap, viewpoint_cam):
            """
            compute per frame sdf
            """
            new_points = (
                torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
                @ viewpoint_cam.full_proj_transform
            )
            z = new_points[..., -1:]
            pix_coords = new_points[..., :2] / new_points[..., -1:]
            mask_proj = ((pix_coords > -1.0) & (pix_coords < 1.0) & (z > 0)).all(dim=-1)
            sampled_depth = torch.nn.functional.grid_sample(
                depthmap.cuda()[None],
                pix_coords[None, None],
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ).reshape(-1, 1)
            sampled_rgb = (
                torch.nn.functional.grid_sample(
                    rgbmap.cuda()[None],
                    pix_coords[None, None],
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
                .reshape(3, -1)
                .T
            )
            sampled_normal = (
                torch.nn.functional.grid_sample(
                    normalmap.cuda()[None],
                    pix_coords[None, None],
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
                .reshape(3, -1)
                .T
            )
            sdf = sampled_depth - z
            return sdf, sampled_rgb, sampled_normal, mask_proj

        def compute_unbounded_tsdf(
            samples, inv_contraction, voxel_size, return_rgb=False
        ):
            """
            Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
            """
            if inv_contraction is not None:
                samples = inv_contraction(samples)
                mask = torch.linalg.norm(samples, dim=-1) > 1
                # adaptive sdf_truncation
                sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                sdf_trunc[mask] *= 1 / (
                    2 - torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9)
                )
            else:
                sdf_trunc = 5 * voxel_size

            tsdfs = torch.ones_like(samples[:, 0]) * 1
            rgbs = torch.zeros((samples.shape[0], 3)).cuda()

            weights = torch.ones_like(samples[:, 0])
            for i, viewpoint_cam in tqdm(
                enumerate(self.viewpoint_stack), desc="TSDF integration progress"
            ):
                sdf, rgb, normal, mask_proj = compute_sdf_perframe(
                    i,
                    samples,
                    depthmap=self.depthmaps[i],
                    rgbmap=self.rgbmaps[i],
                    normalmap=self.depth_normals[i],
                    viewpoint_cam=self.viewpoint_stack[i],
                )

                # volume integration
                sdf = sdf.flatten()
                mask_proj = mask_proj & (sdf > -sdf_trunc)
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
                w = weights[mask_proj]
                wp = w + 1
                tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
                rgbs[mask_proj] = (rgbs[mask_proj] * w[:, None] + rgb[mask_proj]) / wp[
                    :, None
                ]
                # update weight
                weights[mask_proj] = wp

            if return_rgb:
                return tsdfs, rgbs

            return tsdfs

        normalize = lambda x: (x - self.center) / self.radius
        unnormalize = lambda x: (x * self.radius) + self.center
        inv_contraction = lambda x: unnormalize(uncontract(x))

        N = resolution
        voxel_size = self.radius * 2 / N
        print(f"Computing sdf gird resolution {N} x {N} x {N}")
        print(f"Define the voxel_size as {voxel_size}")
        sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)
        from utils.mcube_utils import marching_cubes_with_contraction

        R = contract(normalize(self.gaussians.get_xyz)).norm(dim=-1).cpu().numpy()
        R = np.quantile(R, q=0.95)
        R = min(R + 0.01, 1.9)

        mesh = marching_cubes_with_contraction(
            sdf=sdf_function,
            bounding_box_min=(-R, -R, -R),
            bounding_box_max=(R, R, R),
            level=0,
            resolution=N,
            inv_contraction=inv_contraction,
        )

        # coloring the mesh
        torch.cuda.empty_cache()
        mesh = mesh.as_open3d
        print("texturing mesh ... ")
        _, rgbs = compute_unbounded_tsdf(
            torch.tensor(np.asarray(mesh.vertices)).float().cuda(),
            inv_contraction=None,
            voxel_size=voxel_size,
            return_rgb=True,
        )
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
        return mesh

    @torch.no_grad()
    def export_image(self, path):
        render_path = os.path.join(path, "rgb")
        gts_path = os.path.join(path, "gt")
        vis_path = os.path.join(path, "vis")
        render_depth_path = os.path.join(path, "depth")
        render_normal_path = os.path.join(path, "normal")
        gt_depth_path = os.path.join(path, "gt_depth")

        os.makedirs(render_path, exist_ok=True)
        os.makedirs(vis_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)
        os.makedirs(render_depth_path, exist_ok=True)
        os.makedirs(render_normal_path, exist_ok=True)
        os.makedirs(gt_depth_path, exist_ok=True)

        for idx, viewpoint_cam in tqdm(
            enumerate(self.viewpoint_stack), desc="export images"
        ):
            name = viewpoint_cam.image_name
            gt = viewpoint_cam.original_image[0:3, :, :]
            gt_depth = viewpoint_cam.gt_depth
            render_depth = self.depthmaps[idx]
            normal = self.depth_normals[idx]
            save_img_u8(
                gt.permute(1, 2, 0).cpu().numpy(),
                os.path.join(gts_path, name + ".png"),
            )
            save_img_u8(
                self.rgbmaps[idx].permute(1, 2, 0).cpu().numpy(),
                os.path.join(render_path, name + ".png"),
            )
            save_img_u8(
                colormap((self.depthmaps[idx][0].cpu().numpy()), cmap="turbo").permute(
                    1, 2, 0
                ),
                os.path.join(vis_path, "depth_{0:05d}".format(idx) + ".png"),
            )
            save_img_u8(
                self.normals[idx].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5,
                os.path.join(vis_path, "normal_{0:05d}".format(idx) + ".png"),
            )
            save_img_u8(
                self.depth_normals[idx].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5,
                os.path.join(vis_path, "depth_normal_{0:05d}".format(idx) + ".png"),
            )
            np.save(os.path.join(gt_depth_path, name + ".npy"), gt_depth.cpu().numpy())
            np.save(
                os.path.join(render_depth_path, name + ".npy"),
                render_depth.permute(1, 2, 0).cpu().numpy() * 1000,
            )
            np.save(
                os.path.join(render_normal_path, name + ".npy"),
                normal.permute(1, 2, 0).cpu().numpy(),
            )

    @torch.no_grad()
    def extract_gaussian_to_points(self, poisson_depth=11):
        points = self.gaussians.get_xyz
        colors = self.gaussians.get_features
        rotation = self.gaussians.get_rotation
        rotation = build_rotation(rotation)
        normals = rotation[..., 2, ...]
        colors = colors[:, 0, :]

        colors = torch.clamp(colors, 0, 1).cpu().numpy()
        points = points.cpu().numpy()
        normals = normals.cpu().numpy()

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=poisson_depth
        )

        return mesh

    @torch.no_grad()
    def extract_rgbd_to_mesh(self, sample_num=10_000_000):
        points = []
        colors = []
        normals = []
        num_frames = len(self.viewpoint_stack)  # type: ignore
        samples_per_frame = (sample_num + num_frames) // (num_frames)

        for i, cam_o3d in tqdm(
            enumerate(to_cam_open3d(self.viewpoint_stack)),
            desc="TSDF integration progress",
        ):
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]  # [1, H, W]

            c2w = torch.from_numpy(np.linalg.inv(cam_o3d.extrinsic)).float()

            valid_depth = (
                find_edges(
                    depth,
                )
                < 0.2
            ) * 1.0
            valid_mask = valid_depth
            indices = pick_indices_at_random(valid_mask, samples_per_frame)
            xyzs, rgbs = get_colored_points_from_depth(
                depths=depth.permute(1, 2, 0),
                rgbs=rgb.permute(1, 2, 0),
                fx=cam_o3d.intrinsic.intrinsic_matrix[0, 0],
                fy=cam_o3d.intrinsic.intrinsic_matrix[1, 1],
                cx=cam_o3d.intrinsic.intrinsic_matrix[0, 2],  # type: ignore
                cy=cam_o3d.intrinsic.intrinsic_matrix[1, 2],  # type: ignore
                img_size=(cam_o3d.intrinsic.width, cam_o3d.intrinsic.height),
                c2w=c2w,
                mask=indices,
            )

        points = torch.cat(points, dim=0)
        colors = torch.cat(colors, dim=0)
        colors = torch.clamp(colors, 0, 1)
        normals = torch.cat(normals, dim=0)

        points = points.cpu().numpy()
        normals = normals.cpu().numpy()
        colors = colors.cpu().numpy()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.select_by_index(ind)
        pcd = pcd.voxel_down_sample(voxel_size=0.02)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=11
        )
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)

        return mesh
