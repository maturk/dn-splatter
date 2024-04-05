"""Renders normal maps from mesh and camera pose trajectory

Note: 
    If you are running this in Headless mode, e.g. in a server with no monitor,
    you need to compile Open3D in headless mode: http://www.open3d.org/docs/release/tutorial/visualization/headless_rendering.html?highlight=headless

    - Tested with Open3D 0.17.0 and 0.16.1. Some versions of Open3D will not work.

Important:
    Normal maps are rendered in OpenCV camera coordinate system (default Open3D conventions)
"""

from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from dn_splatter.utils.utils import save_img
from torch import Tensor
from tqdm import tqdm


def render_normals_gt(
    mesh_path,
    poses: Tensor,
    save_dir: Path,
    w=1200,
    h=680,
    fx=600.0,
    fy=600.0,
    cx=599.5,
    cy=339.6,
):
    """Render normal maps given a mesh ply file, a trajectory of poses, and camera intrinsics"""
    np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    mesh = trimesh.load_mesh(mesh_path).as_open3d
    mesh.compute_vertex_normals()
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=w, height=h)
    vis.add_geometry(mesh)
    vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Normal
    vis.update_geometry(mesh)

    for i, c2w in tqdm(
        enumerate(poses), desc="Generating normals for each input pose ..."
    ):
        w2c = np.linalg.inv(c2w)
        camera = vis.get_view_control().convert_to_pinhole_camera_parameters()
        camera.extrinsic = w2c
        vis.get_view_control().convert_from_pinhole_camera_parameters(camera)
        vis.poll_events()
        vis.update_renderer()
        color_image = vis.capture_screen_float_buffer(True)
        image = np.asarray(color_image) * 255
        image = image.astype(np.uint8)
        save_img(image, f"{str(save_dir)}/normal_{i:05d}.png", verbose=False)
    vis.destroy_window()
