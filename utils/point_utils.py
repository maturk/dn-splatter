import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, cv2
import matplotlib.pyplot as plt
import math


def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.0))
    fy = H / (2 * math.tan(view.FoVy / 2.0))
    intrins = (
        torch.tensor([[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]])
        .float()
        .cuda()
    )
    grid_x, grid_y = torch.meshgrid(
        torch.arange(W, device="cuda").float(),
        torch.arange(H, device="cuda").float(),
        indexing="xy",
    )
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(
        -1, 3
    )
    rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points


def depth_to_normal(view, depth):
    """
    view: view camera
    depth: depthmap
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output


def backproject(
    depths: np.ndarray,
    fx: float,
    fy: float,
    cx: int,
    cy: int,
    img_size: tuple,
    c2w: np.ndarray,
):
    if depths.ndim == 3:
        depths = depths.reshape(-1, 1)
    elif depths.shape[-1] != 1:
        depths = depths[..., np.newaxis]
        depths = depths.reshape(-1, 1)

    image_coords = get_camera_coords(img_size)

    means3d = np.zeros([img_size[0], img_size[1], 3], dtype=np.float32).reshape(-1, 3)
    means3d[:, 0] = (image_coords[:, 0] - cx) * depths[:, 0] / fx  # x
    means3d[:, 1] = (image_coords[:, 1] - cy) * depths[:, 0] / fy  # y
    means3d[:, 2] = depths[:, 0]  # z

    # to world coords
    means3d = means3d @ np.linalg.inv(c2w[..., :3, :3]) + c2w[..., :3, 3]
    return means3d, image_coords


def get_camera_coords(img_size: tuple, pixel_offset: float = 0.5) -> np.ndarray:
    image_coords = np.meshgrid(
        np.arange(img_size[0]),
        np.arange(img_size[1]),
        indexing="xy",  # W = u by H = v
    )
    image_coords = (
        np.stack(image_coords, axis=-1) + pixel_offset
    )  # stored as (x, y) coordinates
    image_coords = image_coords.reshape(-1, 2)
    image_coords = image_coords.astype(np.float32)
    return image_coords


def backproject(
    depths: np.ndarray,
    fx: float,
    fy: float,
    cx: int,
    cy: int,
    img_size: tuple,
    c2w: np.ndarray,
):
    if depths.ndim == 3:
        depths = depths.reshape(-1, 1)
    elif depths.shape[-1] != 1:
        depths = depths[..., np.newaxis]
        depths = depths.reshape(-1, 1)

    image_coords = get_camera_coords(img_size)

    means3d = np.zeros([img_size[0], img_size[1], 3], dtype=np.float32).reshape(-1, 3)
    means3d[:, 0] = (image_coords[:, 0] - cx) * depths[:, 0] / fx  # x
    means3d[:, 1] = (image_coords[:, 1] - cy) * depths[:, 0] / fy  # y
    means3d[:, 2] = depths[:, 0]  # z

    # to world coords
    means3d = means3d @ np.linalg.inv(c2w[..., :3, :3]) + c2w[..., :3, 3]
    return means3d, image_coords


def compute_angle_between_normals(normal_map1, normal_map2):
    norm1 = np.linalg.norm(normal_map1, axis=2, keepdims=True)
    norm2 = np.linalg.norm(normal_map2, axis=2, keepdims=True)
    normal_map1_normalized = normal_map1 / norm1
    normal_map2_normalized = normal_map2 / norm2

    dot_product = np.sum(normal_map1_normalized * normal_map2_normalized, axis=2)

    dot_product = np.clip(dot_product, -1.0, 1.0)

    angles = np.arccos(dot_product)

    angles_degrees = np.degrees(angles)

    return angles_degrees
