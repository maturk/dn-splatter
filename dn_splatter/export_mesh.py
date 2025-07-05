"""Various GS mesh exporters"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import tyro
from tqdm import tqdm
from typing_extensions import Annotated

from dn_splatter.utils.camera_utils import (
    get_colored_points_from_depth,
    get_means3d_backproj,
    project_pix,
)
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE

"""
Methods for extracting meshes from GS:

1) GaussiansToPoisson:
    - takes Gaussian means and predicted normals -> poisson
2) DepthAndNormalMapsPoisson
    - backproject rendered depth and normal maps -> poisson
3) LevelSetExtractor (SuGaR)
    - cast rays into scene from cameras
    - extract level sets based on gaussian density function
    - estimate normals (analytically or nearest gaussians)
    - poisson
4) Marching Cubes
    - voxelize scene and
    - compute isosurface level sets based on gaussian densities
    - run marching cubes algorithm
5) TSDF
    - voxelize scene and
    - backproject depths and integrate points into voxels for tsdf fusion
    - run marching cubes algorithm
"""


def pick_indices_at_random(valid_mask, samples_per_frame):
    indices = torch.nonzero(torch.ravel(valid_mask))
    if samples_per_frame < len(indices):
        which = torch.randperm(len(indices))[:samples_per_frame]
        indices = indices[which]
    return torch.ravel(indices)


def find_depth_edges(depth_im, threshold=0.01, dilation_itr=3):
    laplacian_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=depth_im.dtype, device=depth_im.device
    )
    laplacian_kernel = laplacian_kernel.unsqueeze(0).unsqueeze(0)
    depth_laplacian = (
        F.conv2d(
            (1.0 / (depth_im + 1e-6)).unsqueeze(0).unsqueeze(0).squeeze(-1),
            laplacian_kernel,
            padding=1,
        )
        .squeeze(0)
        .squeeze(0)
        .unsqueeze(-1)
    )

    edges = (depth_laplacian > threshold) * 1.0
    structure_el = laplacian_kernel * 0.0 + 1.0

    dilated_edges = edges
    for i in range(dilation_itr):
        dilated_edges = (
            F.conv2d(
                dilated_edges.unsqueeze(0).unsqueeze(0).squeeze(-1),
                structure_el,
                padding=1,
            )
            .squeeze(0)
            .squeeze(0)
            .unsqueeze(-1)
        )
    dilated_edges = (dilated_edges > 0.0) * 1.0
    return dilated_edges


@dataclass
class GSMeshExporter:
    """Base class for GS mesh exporters"""

    load_config: Path
    """Path to the trained config YAML file."""
    output_dir: Path = Path("./mesh_exports/")
    """Path to the output directory."""

    cropbox_pos: Optional[Annotated[Tuple[float, float, float], "x, y, z"]] = None
    """Cropbox position for the mesh."""
    cropbox_rpy: Optional[Annotated[Tuple[float, float, float], "rx, ry, rz"]] = None
    """Cropbox rotation for the mesh."""
    cropbox_scale: Optional[Annotated[Tuple[float, float, float], "sx, sy, sz"]] = None
    """Cropbox scale for the mesh."""

    def cropbox(self) -> Optional[OrientedBox]:
        """Returns the cropbox for the mesh."""
        if self.cropbox_pos is None and self.cropbox_rpy is None and self.cropbox_scale is None:
            return None

        if self.cropbox_pos is None:
            self.cropbox_pos = (0.0, 0.0, 0.0)
        if self.cropbox_rpy is None:
            self.cropbox_rpy = (0.0, 0.0, 0.0)
        if self.cropbox_scale is None:
            self.cropbox_scale = (1.0, 1.0, 1.0)

        return OrientedBox.from_params(
            pos=self.cropbox_pos,
            rpy=self.cropbox_rpy,
            scale=self.cropbox_scale,
        )


@dataclass
class GaussiansToPoisson(GSMeshExporter):
    """
    Extract Gaussian positions and normals -> Poisson
    """

    densify_gaussians: Optional[int] = None
    """Densify gaussians based on gaussian distribution"""
    use_masks: bool = False
    """If dataset has masks, use these to auto crop gaussians within masked regions."""
    min_opacity: Optional[float] = None
    """Remove opacities less than min_opacity"""
    mask_color: Optional[tuple] = None
    """Remove gaussians with this color from the computation"""
    down_sample_voxel: Optional[float] = None
    """pcd down sample voxel size. Recommended value around 0.005"""
    outlier_removal: bool = False
    """Remove outliers"""
    std_ratio: float = 2.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""
    poisson_depth: int = 9
    """Poisson Octree max depth, higher values increase mesh detail"""

    def main(self):
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        assert isinstance(pipeline.model, SplatfactoModel)

        model: SplatfactoModel = pipeline.model
        crop_box = self.cropbox()

        with torch.no_grad():
            positions = model.means.cpu()
            normals = model.normals.cpu()
            opacities = model.opacities.cpu()
            assert model.config.sh_degree > 0

            colors = torch.clamp(model.colors.clone(), 0.0, 1.0).cpu().data.float()

            assert positions.shape[0] == normals.shape[0]  # type: ignore

            if self.use_masks:
                outs = pipeline.datamanager.dataparser.get_dataparser_outputs(  # type: ignore
                    split="train"
                ).mask_filenames  # type: ignore
                assert outs is not None
                # apply depth consistency check
                cameras: Cameras = pipeline.datamanager.train_dataset.cameras  # type: ignore
                for image_idx, data in enumerate(
                    pipeline.datamanager.train_dataset  # type: ignore
                ):  # type: ignore
                    mask = data["mask"]
                    camera = cameras[image_idx : image_idx + 1].to("cpu")
                    c2w = torch.eye(4, dtype=torch.float)
                    c2w[:3, :4] = camera.camera_to_worlds.squeeze(0)
                    c2w = c2w @ torch.diag(
                        torch.tensor([1, -1, -1, 1], device="cpu", dtype=torch.float)
                    )
                    c2w = c2w[:3, :4]
                    H, W = camera.height.item(), camera.width.item()

                    uv_depth = project_pix(
                        p=positions,
                        fx=camera.fx.item(),
                        fy=camera.fy.item(),
                        cx=camera.cx.item(),  # type: ignore
                        cy=camera.cy.item(),  # type: ignore
                        c2w=c2w,
                        device="cpu",  # type: ignore
                        return_z_depths=True,
                    )
                    uv_depth[:, :2] = uv = torch.floor(uv_depth[:, :2] - 0.5).long()
                    indices_to_remove = []
                    for i in range(uv_depth.shape[0]):
                        if not (
                            (uv[i, 0] > 0)
                            & (uv[i, 0] < W)
                            & (uv[i, 1] > 0)
                            & (uv[i, 1] < H)
                        ):
                            # filter out invalid projections
                            continue
                        if mask is not None:
                            # filter out masked regions
                            if not mask[uv[i, 1], uv[i, 0]]:
                                indices_to_remove.append(i)
                            continue

                    print("total filtered by mask ", len(indices_to_remove))
                    mask = torch.ones(positions.shape[0], dtype=torch.bool)
                    mask[indices_to_remove] = 0
                    positions = positions[mask]
                    normals = normals[mask]  # type: ignore
                    opacities = opacities[mask]
                    colors = colors[mask]

            assert positions.shape[0] == normals.shape[0]  # type: ignore

            if self.min_opacity is not None:
                mask = (opacities > self.min_opacity)[..., 0]
                mask = torch.BoolTensor(mask)
                CONSOLE.print(
                    f"Removing {torch.count_nonzero(~mask)} / {opacities.shape[0]} gaussians with too small opacity"
                )
                positions = positions[mask]
                normals = normals[mask]  # type: ignore
                colors = colors[mask]

            if self.mask_color is not None:
                mask = torch.all(
                    colors
                    != torch.tensor(
                        [self.mask_color], dtype=colors.dtype, device=colors.device
                    ),
                    dim=-1,
                )
                CONSOLE.print(
                    f"Removing {torch.count_nonzero(~mask)} gaussians with mask color"
                )
                positions = positions[mask]
                normals = normals[mask]  # type: ignore
                colors = colors[mask]

            if self.densify_gaussians is not None:
                extra_positions, gs_indices = model.sample_points_in_gaussians(
                    num_samples=self.densify_gaussians
                )
                positions = torch.cat([positions, extra_positions.cpu()], dim=0)
                extra_normals = model.normals[gs_indices]
                normals = torch.cat([normals, extra_normals.cpu()], dim=0)
                extra_colors = (
                    torch.clamp(model.colors[gs_indices], 0.0, 1.0).cpu().data.float()
                )
                colors = torch.cat([colors, extra_colors.cpu()], dim=0)

            positions = positions.cpu().numpy()
            normals = normals.cpu().numpy()
            colors = colors.cpu().numpy()

            if crop_box is not None:
                pts = torch.from_numpy(positions).float().to(crop_box.T.device)
                inside_crop = crop_box.within(pts).cpu().numpy()
                if inside_crop.sum() == 0:
                    CONSOLE.print("[yellow]Warning: No points within crop box[/yellow]")
                positions = positions[inside_crop]
                normals = normals[inside_crop]
                colors = colors[inside_crop]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(positions)
            pcd.normals = o3d.utility.Vector3dVector(normals)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            if self.down_sample_voxel is not None:
                pcd = pcd.voxel_down_sample(voxel_size=self.down_sample_voxel)

            if self.outlier_removal:
                cl, ind = pcd.remove_statistical_outlier(
                    nb_neighbors=20, std_ratio=self.std_ratio
                )
                pcd = pcd.select_by_index(ind)

            CONSOLE.print("Computing Mesh... this may take a while.")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=self.poisson_depth
            )
            vertices_to_remove = densities < np.quantile(densities, 0.01)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            CONSOLE.print("[bold green]:white_check_mark: Computing Mesh")

            CONSOLE.print("Saving Mesh...")
            o3d.io.write_triangle_mesh(
                str(self.output_dir / "GaussiansToPoisson_poisson_mesh.ply"), mesh
            )
            o3d.io.write_point_cloud(
                str(self.output_dir / "GaussiansToPoisson_pcd.ply"), pcd
            )
            CONSOLE.print(
                f"[bold green]:white_check_mark: Saving Mesh to {self.output_dir / 'GaussiansToPoisson_poisson_mesh.ply'}"
            )


@dataclass
class DepthAndNormalMapsPoisson(GSMeshExporter):
    """
    Idea: backproject depth and normal maps into 3D oriented point cloud -> Poisson
    """

    total_points: int = 2_000_000
    """Total target surface samples"""
    normal_method: Literal["density_grad", "normal_maps"] = "normal_maps"
    """Normal estimation method"""
    use_masks: bool = True
    """If dataset has masks, use these to auto crop gaussians within masked regions."""
    filter_edges_from_depth_maps: bool = False
    """Filter out edges when backprojecting from depth maps"""
    down_sample_voxel: Optional[float] = None
    """pcd down sample voxel size. Recommended value around 0.005"""
    outlier_removal: bool = False
    """Remove outliers"""
    std_ratio: float = 2.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""
    edge_threshold: float = 0.004
    """Threshold for edge detection in depth maps (inverse depth Laplacian, resolution sensitive)"""
    edge_dilation_iterations: int = 10
    """Number of morphological dilation iterations for edge detection (swells edges)"""
    poisson_depth: int = 9
    """Poisson Octree max depth, higher values increase mesh detail"""

    def main(self):
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        assert isinstance(pipeline.model, SplatfactoModel)

        model: SplatfactoModel = pipeline.model
        crop_box = self.cropbox()

        with torch.no_grad():
            cameras: Cameras = pipeline.datamanager.train_dataset.cameras  # type: ignore
            # TODO: do eval dataset as well
            num_frames = len(pipeline.datamanager.train_dataset)  # type: ignore
            samples_per_frame = (self.total_points + num_frames) // (num_frames)
            print("samples per frame: ", samples_per_frame)
            points = []
            normals = []
            colors = []
            for image_idx, data in enumerate(
                pipeline.datamanager.train_dataset
            ):  # type: ignore
                mask = None
                if "mask" in data:
                    mask = data["mask"]
                camera = cameras[image_idx : image_idx + 1]
                outputs = model.get_outputs_for_camera(camera=camera)
                assert "depth" in outputs
                depth_map = outputs["depth"]
                c2w = torch.eye(4, dtype=torch.float, device=depth_map.device)
                c2w[:3, :4] = camera.camera_to_worlds.squeeze(0)
                c2w = c2w @ torch.diag(
                    torch.tensor([1, -1, -1, 1], device=c2w.device, dtype=torch.float)
                )
                c2w = c2w[:3, :4]
                H, W = camera.height.item(), camera.width.item()

                if self.filter_edges_from_depth_maps:
                    valid_depth = (
                        find_depth_edges(
                            depth_map,
                            threshold=self.edge_threshold,
                            dilation_itr=self.edge_dilation_iterations,
                        )
                        < 0.2
                    )
                else:
                    valid_depth = depth_map
                valid_mask = valid_depth

                indices = pick_indices_at_random(valid_mask, samples_per_frame)
                if len(indices) == 0:
                    continue

                if mask is not None and self.use_masks:
                    depth_map[~mask] = 0
                xyzs, rgbs = get_colored_points_from_depth(
                    depths=depth_map,
                    rgbs=outputs["rgb"],
                    fx=camera.fx.item(),
                    fy=camera.fy.item(),
                    cx=camera.cx.item(),  # type: ignore
                    cy=camera.cy.item(),  # type: ignore
                    img_size=(W, H),
                    c2w=c2w,
                    mask=indices,
                )
                if self.normal_method == "normal_maps":
                    # normals to OPENGL
                    assert "normal" in outputs
                    normal_map = outputs["surface_normal"]
                    h, w, _ = normal_map.shape
                    normal_map = normal_map.view(-1, 3)
                    normal_map = 2 * normal_map - 1
                    normal_map = normal_map @ torch.diag(
                        torch.tensor(
                            [1, -1, -1], device=normal_map.device, dtype=torch.float
                        )
                    )
                    normal_map = normal_map.view(h, w, 3)
                    # normals to World
                    rot = c2w[:3, :3]
                    normal_map = normal_map.permute(2, 0, 1).reshape(3, -1)
                    normal_map = torch.nn.functional.normalize(normal_map, p=2, dim=0)
                    normal_map = rot @ normal_map
                    normal_map = normal_map.permute(1, 0).reshape(h, w, 3)

                    normal_map = normal_map.view(-1, 3)[indices]
                else:
                    # grad of density
                    xyzs, _ = get_means3d_backproj(
                        depths=depth_map * 0.99,
                        fx=camera.fx.item(),
                        fy=camera.fy.item(),
                        cx=camera.cx.item(),  # type: ignore
                        cy=camera.cy.item(),  # type: ignore
                        img_size=(W, H),
                        c2w=c2w,
                        device=c2w.device,
                        # mask=indices,
                    )
                    normals = model.get_density_grad(
                        samples=xyzs.cuda(), num_closest_gaussians=1
                    )
                    viewdirs = -xyzs + c2w[..., :3, 3]
                    viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
                    dots = (normals * viewdirs).sum(-1)
                    negative_dot_indices = dots < 0
                    normals[negative_dot_indices] = -normals[negative_dot_indices]
                    normals = normals @ c2w[:3, :3]
                    normals = normals @ torch.diag(
                        torch.tensor(
                            [1, -1, -1], device=normals.device, dtype=torch.float
                        )
                    )
                    normal_map = normals / normals.norm(dim=-1, keepdim=True)
                    normal_map = (normal_map + 1) / 2

                    normal_map = outputs["surface_normal"].cpu()
                    normal_map = normal_map.view(-1, 3)[indices]

                if crop_box is not None:
                    inside_crop = crop_box.within(xyzs).squeeze()
                    if inside_crop.sum() == 0:
                        continue
                    xyzs = xyzs[inside_crop]
                    rgbs = rgbs[inside_crop]
                    normal_map = normal_map[inside_crop]

                points.append(xyzs)
                colors.append(rgbs)
                normals.append(normal_map)

            points = torch.cat(points, dim=0)
            colors = torch.cat(colors, dim=0)
            normals = torch.cat(normals, dim=0)

            points = points.cpu().numpy()
            normals = normals.cpu().numpy()
            colors = colors.cpu().numpy()

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.normals = o3d.utility.Vector3dVector(normals)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            if self.outlier_removal:
                cl, ind = pcd.remove_statistical_outlier(
                    nb_neighbors=20, std_ratio=self.std_ratio
                )
                pcd = pcd.select_by_index(ind)

            o3d.io.write_point_cloud(
                str(self.output_dir / "DepthAndNormalMapsPoisson_pcd.ply"), pcd
            )
            CONSOLE.print("Computing Mesh... this may take a while.")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=self.poisson_depth
            )
            vertices_to_remove = densities < np.quantile(densities, 0.01)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            CONSOLE.print("[bold green]:white_check_mark: Computing Mesh")

            CONSOLE.print(
                f"Saving Mesh to {str(self.output_dir / 'DepthAndNormalMapsPoisson_poisson_mesh.ply')}"
            )
            o3d.io.write_triangle_mesh(
                str(self.output_dir / "DepthAndNormalMapsPoisson_poisson_mesh.ply"),
                mesh,
            )


@dataclass
class LevelSetExtractor(GSMeshExporter):
    """Extract level sets based on gaussian density from training views

    Inspired by SuGaR
    """

    total_points: int = 2_000_000
    """Total target surface samples"""
    use_masks: bool = False
    """If dataset has masks, use these to limit surface sampling regions."""
    surface_levels: Tuple[float, float, float] = (0.1, 0.3, 0.5)
    """Surface levels to extract"""
    return_normal: Literal[
        "analytical", "closest_gaussian", "average"
    ] = "closest_gaussian"
    """Normal mode"""
    poisson_depth: int = 9
    """Poisson Octree max depth, higher values increase mesh detail"""

    def main(self):
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        assert isinstance(pipeline.model, SplatfactoModel)

        model: SplatfactoModel = pipeline.model
        crop_box = self.cropbox()

        # assert hasattr(pipeline.model,"compute_level_surface_points_from_camera_fast")

        with torch.no_grad():
            cameras: Cameras = pipeline.datamanager.train_dataset.cameras  # type: ignore
            num_frames = len(pipeline.datamanager.train_dataset)  # type: ignore
            samples_per_frame = (self.total_points + num_frames) // (num_frames)
            surface_levels_outputs = {}
            for surface_level in self.surface_levels:
                surface_levels_outputs[surface_level] = {
                    "points": torch.zeros(0, 3, device="cuda"),
                    "colors": torch.zeros(0, 3, device="cuda"),
                    "normals": torch.zeros(0, 3, device="cuda"),
                }

            # TODO: do eval dataset as well maybe
            for image_idx, data in tqdm(
                enumerate(pipeline.datamanager.train_dataset),
                desc="Computing surface levels for train images",
            ):  # type: ignore
                print(
                    "image:",
                    image_idx,
                    f"out of {len(pipeline.datamanager.train_dataset)}",
                )
                camera = cameras[image_idx : image_idx + 1].to("cuda")
                mask = None
                if "mask" in data and self.use_masks:
                    mask = data["mask"]
                frame_outputs = model.compute_level_surface_points(
                    camera=camera,
                    mask=mask,
                    num_samples=samples_per_frame,
                    surface_levels=self.surface_levels,
                    return_normal=self.return_normal,
                )  # type: ignore

                for surface_level in self.surface_levels:
                    img_surface_points = frame_outputs[surface_level]["points"]
                    img_surface_colors = frame_outputs[surface_level]["colors"]
                    img_surface_normals = frame_outputs[surface_level]["normals"]

                    if crop_box is not None:
                        inside_crop = crop_box.within(img_surface_points).squeeze()
                        if inside_crop.sum() == 0:
                            continue
                        img_surface_points = img_surface_points[inside_crop]
                        img_surface_colors = img_surface_colors[inside_crop]
                        img_surface_normals = img_surface_normals[inside_crop]

                    surface_levels_outputs[surface_level]["points"] = torch.cat(
                        [
                            surface_levels_outputs[surface_level]["points"],
                            img_surface_points,
                        ],
                        dim=0,
                    )
                    surface_levels_outputs[surface_level]["colors"] = torch.cat(
                        [
                            surface_levels_outputs[surface_level]["colors"],
                            img_surface_colors,
                        ],
                        dim=0,
                    )
                    surface_levels_outputs[surface_level]["normals"] = torch.cat(
                        [
                            surface_levels_outputs[surface_level]["normals"],
                            img_surface_normals,
                        ],
                        dim=0,
                    )

            for surface_level in self.surface_levels:
                outs = surface_levels_outputs[surface_level]
                points = outs["points"].cpu().numpy()
                colors = outs["colors"].cpu().numpy()
                normals = outs["normals"].cpu().numpy()

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                pcd.normals = o3d.utility.Vector3dVector(normals)

                CONSOLE.print(
                    "Saving unclean points to ",
                    str(
                        self.output_dir
                        / f"before_clean_points_surface_level_{surface_level}_{self.return_normal}.ply"
                    ),
                )
                o3d.io.write_point_cloud(
                    str(
                        self.output_dir
                        / f"before_clean_points_surface_level_{surface_level}_{self.return_normal}.ply"
                    ),
                    pcd,
                )
                cl, ind = pcd.remove_statistical_outlier(
                    nb_neighbors=20, std_ratio=20.0
                )
                pcd_clean = pcd.select_by_index(ind)
                CONSOLE.print(
                    "Saving cleaned points to ",
                    str(
                        self.output_dir
                        / f"after_clean_points_surface_level_{surface_level}_{self.return_normal}.ply"
                    ),
                )
                o3d.io.write_point_cloud(
                    str(
                        self.output_dir
                        / f"after_clean_points_surface_level_{surface_level}_{self.return_normal}.ply"
                    ),
                    pcd_clean,
                )
                CONSOLE.print("Computing Mesh... this may take a while.")
                (
                    mesh,
                    densities,
                ) = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd_clean, depth=self.poisson_depth
                )

                vertices_to_remove = densities < np.quantile(densities, 0.01)
                mesh.remove_vertices_by_mask(vertices_to_remove)
                CONSOLE.print("[bold green]:white_check_mark: Computing Mesh")

                CONSOLE.print("Saving Mesh...")
                o3d.io.write_triangle_mesh(
                    str(
                        self.output_dir
                        / f"poisson_mesh_surface_level_{surface_level}_{self.return_normal}.ply"
                    ),
                    mesh,
                )
                CONSOLE.print(
                    f"[bold green]:white_check_mark: Saving Mesh to {self.output_dir / f'poisson_mesh_surface_level_{surface_level}_{self.return_normal}.ply'}"
                )
                mesh = mesh.filter_smooth_laplacian()
                o3d.io.write_triangle_mesh(
                    str(
                        self.output_dir
                        / f"smoothed_1_poisson_mesh_surface_level_{surface_level}_{self.return_normal}.ply"
                    ),
                    mesh,
                )
                mesh = mesh.filter_smooth_laplacian()
                o3d.io.write_triangle_mesh(
                    str(
                        self.output_dir
                        / f"smoothed_2_poisson_mesh_surface_level_{surface_level}_{self.return_normal}.ply"
                    ),
                    mesh,
                )


@dataclass
class MarchingCubesMesh(GSMeshExporter):
    """Export a GS mesh using marching cubes."""

    isosurface_threshold: float = 0.5
    """The isosurface threshold for extraction."""
    camera_radius_multiplier: int = 2
    """Depending on your scene, multiplier for average camera radius."""
    resolution: int = 512
    """Marching cubes resolution."""
    target_triangles: Optional[int] = 1_000_000
    """Target number of triangles to simplify mesh to."""
    batch_size: int = 2_000_000
    """Batch size for querying level sets."""

    def main(self) -> None:
        """Main function."""
        import mcubes

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        assert isinstance(pipeline.model, SplatfactoModel)

        model: SplatfactoModel = pipeline.model
        crop_box = self.cropbox()

        CONSOLE.print("Extracting mesh with marching cubes... which may take a while")
        with torch.no_grad():
            cameras: Cameras = pipeline.datamanager.train_dataset.cameras  # type: ignore

            # compute scene radius
            centers = cameras.camera_to_worlds[..., :, 3]
            avg_center = cameras.camera_to_worlds[..., :, 3].mean(dim=0, keepdim=True)
            radius = (
                self.camera_radius_multiplier
                * torch.norm(centers - avg_center, dim=-1).max().item()
            )
            # voxel grid to sample (in model/world coordinates)
            X = torch.linspace(-1, 1, self.resolution) * radius
            Y = torch.linspace(-1, 1, self.resolution) * radius
            Z = torch.linspace(-1, 1, self.resolution) * radius
            xx, yy, zz = torch.meshgrid(X, Y, Z, indexing="ij")
            grid_coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

            # Mask out-of-cropbox points in model/world coordinates
            if crop_box is not None:
                mask = crop_box.within(grid_coords)
            else:
                mask = torch.ones(grid_coords.shape[0], dtype=torch.bool, device=grid_coords.device)

            # Only query densities for points inside cropbox
            samples = grid_coords[mask].to(model.device)
            total_samples = len(samples)
            densities_flat = torch.zeros(grid_coords.shape[0], device=model.device)

            CONSOLE.print("Computing voxel grid densities...")
            with torch.no_grad():
                densities_inside = []
                for batch_index in range(0, total_samples, self.batch_size):
                    CONSOLE.print(
                        f"[bold green]Processing batch {batch_index // self.batch_size} / {total_samples//self.batch_size}"
                    )
                    batch_samples = samples[batch_index : batch_index + self.batch_size]
                    batch_densities = model.get_density(batch_samples)
                    densities_inside.append(batch_densities)
                densities_inside = torch.cat(densities_inside, dim=0)
                densities_flat[mask] = densities_inside
                densities = densities_flat.reshape(self.resolution, self.resolution, self.resolution)

            # Optionally, mask out-of-cropbox voxels to a low value so marching cubes ignores them
            if crop_box is not None:
                densities[~mask.reshape(self.resolution, self.resolution, self.resolution)] = -1e6

            CONSOLE.print(
                f"Computing mesh for surface level {self.isosurface_threshold}"
            )
            vertices, triangles = mcubes.marching_cubes(
                densities.cpu().numpy(), self.isosurface_threshold
            )
            # Map vertices back to world coordinates
            # marching cubes returns vertices in voxel index coordinates, so map to world/model coordinates

            # Convert X, Y, Z to numpy for numpy operations
            X_np = X.cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
            Y_np = Y.cpu().numpy() if torch.is_tensor(Y) else np.asarray(Y)
            Z_np = Z.cpu().numpy() if torch.is_tensor(Z) else np.asarray(Z)

            v_x = vertices[:, 0] / (self.resolution - 1)
            v_y = vertices[:, 1] / (self.resolution - 1)
            v_z = vertices[:, 2] / (self.resolution - 1)
            world_x = X_np[0] + v_x * (X_np[-1] - X_np[0])
            world_y = Y_np[0] + v_y * (Y_np[-1] - Y_np[0])
            world_z = Z_np[0] + v_z * (Z_np[-1] - Z_np[0])
            vertices_world = np.stack([world_x, world_y, world_z], axis=-1)

            closest_gaussians = model.get_closest_gaussians(
                torch.from_numpy(vertices_world).float().to(model.device)
            )[..., 0]
            verts_colors = model.colors[closest_gaussians].cpu().numpy()

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices_world)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            mesh.vertex_colors = o3d.utility.Vector3dVector(verts_colors)

            o3d.io.write_triangle_mesh(
                str(self.output_dir / f"marching_cubes_raw_{self.resolution}.ply"),
                mesh,
            )
            # simplify mesh
            if self.target_triangles is not None:
                mesh = mesh.simplify_quadric_decimation(self.target_triangles)

            CONSOLE.print(
                f"Finished computing mesh: {str(self.output_dir / f'marching_cubes_{self.resolution}.ply')}"
            )
            o3d.io.write_triangle_mesh(
                str(self.output_dir / f"marching_cubes_{self.resolution}.ply"), mesh
            )


@dataclass
class TSDFFusion(GSMeshExporter):
    """
    Backproject depths and run TSDF fusion
    """

    voxel_size: float = 0.01
    """tsdf voxel size"""
    sdf_truc: float = 0.03
    """TSDF truncation"""
    total_points: int = 2_000_000
    """Total target surface samples"""
    target_triangles: Optional[int] = None
    """Target number of triangles to simplify mesh to."""

    def main(self):
        import vdbfusion

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        assert isinstance(pipeline.model, SplatfactoModel)

        model: SplatfactoModel = pipeline.model
        crop_box = self.cropbox()

        TSDFvolume = vdbfusion.VDBVolume(
            voxel_size=self.voxel_size, sdf_trunc=self.sdf_truc, space_carving=True
        )

        with torch.no_grad():
            cameras: Cameras = pipeline.datamanager.train_dataset.cameras  # type: ignore
            # TODO: do eval dataset as well
            num_frames = len(pipeline.datamanager.train_dataset)  # type: ignore
            samples_per_frame = (self.total_points + num_frames) // (num_frames)
            print("samples per frame: ", samples_per_frame)
            points = []
            colors = []
            for image_idx, data in enumerate(
                pipeline.datamanager.train_dataset
            ):  # type: ignore
                mask = None
                if "mask" in data:
                    mask = data["mask"]
                camera = cameras[image_idx : image_idx + 1]
                outputs = model.get_outputs_for_camera(
                    camera=camera,
                    obb_box=crop_box
                )
                assert "depth" in outputs
                depth_map = outputs["depth"]
                c2w = torch.eye(4, dtype=torch.float, device=depth_map.device)
                c2w[:3, :4] = camera.camera_to_worlds.squeeze(0)
                c2w = c2w @ torch.diag(
                    torch.tensor([1, -1, -1, 1], device=c2w.device, dtype=torch.float)
                )
                c2w = c2w[:3, :4]
                H, W = camera.height.item(), camera.width.item()

                indices = random.sample(range(H * W), samples_per_frame)

                if mask is not None:
                    depth_map[~mask] = 0
                xyzs, rgbs = get_colored_points_from_depth(
                    depths=depth_map,
                    rgbs=outputs["rgb"],
                    fx=camera.fx.item(),
                    fy=camera.fy.item(),
                    cx=camera.cx.item(),  # type: ignore
                    cy=camera.cy.item(),  # type: ignore
                    img_size=(W, H),
                    c2w=c2w,
                    # mask=indices,
                )
                # xyzs = xyzs[mask.view(-1,1)[...,0]]
                points.append(xyzs)
                colors.append(rgbs)
                TSDFvolume.integrate(
                    xyzs.double().cpu().numpy(),
                    extrinsic=c2w[:3, 3].double().cpu().numpy(),
                )

            vertices, faces = TSDFvolume.extract_triangle_mesh(min_weight=5)

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.compute_vertex_normals()
            colors = torch.cat(colors, dim=0)
            colors = colors.cpu().numpy()
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

            # simplify mesh
            if self.target_triangles is not None:
                mesh = mesh.simplify_quadric_decimation(self.target_triangles)

            o3d.io.write_triangle_mesh(
                str(self.output_dir / "TSDFfusion_mesh.ply"),
                mesh,
            )
            CONSOLE.print(
                f"Finished computing mesh: {str(self.output_dir / 'TSDFfusion.ply')}"
            )


@dataclass
class Open3DTSDFFusion(GSMeshExporter):
    """
    Backproject depths and run TSDF fusion
    """

    voxel_size: float = 0.01
    """tsdf voxel size"""
    sdf_truc: float = 0.03
    """TSDF truncation"""
    depth_trunc: float = 20

    def main(self):
        import open3d as o3d

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        assert isinstance(pipeline.model, SplatfactoModel)

        model: SplatfactoModel = pipeline.model
        crop_box = self.cropbox()

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_size,
            sdf_trunc=self.sdf_truc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        with torch.no_grad():
            cameras: Cameras = pipeline.datamanager.train_dataset.cameras  # type: ignore
            # TODO: do eval dataset as well

            for image_idx, data in enumerate(
                pipeline.datamanager.train_dataset
            ):  # type: ignore
                mask = None
                if "mask" in data:
                    mask = data["mask"]
                camera = cameras[image_idx : image_idx + 1]
                outputs = model.get_outputs_for_camera(
                    camera=camera,
                    obb_box=crop_box,
                )
                assert "depth" in outputs
                depth_map = outputs["depth"]
                c2w = torch.eye(4, dtype=torch.float, device=depth_map.device)
                c2w[:3, :4] = camera.camera_to_worlds.squeeze(0)
                c2w = c2w @ torch.diag(
                    torch.tensor([1, -1, -1, 1], device=c2w.device, dtype=torch.float)
                )

                H, W = camera.height.item(), camera.width.item()
                intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    width=W,
                    height=H,
                    fx=camera.fx.item(),
                    fy=camera.fy.item(),
                    cx=camera.cx.item(),
                    cy=camera.cy.item(),
                )
                rgb_map = outputs["rgb"]
                if mask is not None:
                    depth_map[~mask] = 0

                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(
                        np.asarray(
                            rgb_map.cpu().numpy() * 255,
                            order="C",
                            dtype=np.uint8,
                        )
                    ),
                    o3d.geometry.Image(
                        np.asarray(depth_map.squeeze(-1).cpu().numpy(), order="C")
                    ),
                    depth_trunc=self.depth_trunc,
                    convert_rgb_to_intensity=False,
                    depth_scale=1.0,
                )

                volume.integrate(
                    rgbd,
                    intrinsic=intrinsic,
                    extrinsic=np.linalg.inv(c2w.cpu().numpy()),
                )

            mesh = volume.extract_triangle_mesh()

            mesh_0 = mesh
            with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug
            ) as cm:
                (
                    triangle_clusters,
                    cluster_n_triangles,
                    cluster_area,
                ) = mesh_0.cluster_connected_triangles()

            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)
            cluster_area = np.asarray(cluster_area)
            n_cluster = np.sort(cluster_n_triangles.copy())[-50]
            n_cluster = max(n_cluster, 50)  # filter meshes smaller than 50
            triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
            mesh_0.remove_triangles_by_mask(triangles_to_remove)
            mesh_0.remove_unreferenced_vertices()
            mesh_0.remove_degenerate_triangles()

            o3d.io.write_triangle_mesh(
                str(self.output_dir / "Open3dTSDFfusion_mesh.ply"),
                mesh,
            )
            CONSOLE.print(
                f"Finished computing mesh: {str(self.output_dir / 'Open3dTSDFfusion.ply')}"
            )


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[TSDFFusion, tyro.conf.subcommand(name="tsdf")],
        Annotated[Open3DTSDFFusion, tyro.conf.subcommand(name="o3dtsdf")],
        Annotated[DepthAndNormalMapsPoisson, tyro.conf.subcommand(name="dn")],
        Annotated[LevelSetExtractor, tyro.conf.subcommand(name="sugar-coarse")],
        Annotated[GaussiansToPoisson, tyro.conf.subcommand(name="gaussians")],
        Annotated[MarchingCubesMesh, tyro.conf.subcommand(name="marching")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    # tyro.cli(GaussiansToPoisson).main()
    # tyro.cli(DepthAndNormalMapsPoisson).main()
    # tyro.cli(LevelSetExtractor).main()
    # tyro.cli(MarchingCubesMesh).main()
    # tyro.cli(TSDFFusion).main()
    tyro.cli(Open3DTSDFFusion).main()
