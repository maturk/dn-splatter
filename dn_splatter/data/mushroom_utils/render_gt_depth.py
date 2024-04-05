import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
import tyro
from pytorch3d import transforms as py3d_transform
from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
from pytorch3d.structures import Meshes

# from dn_splatter.utils.camera_utils import OPENGL_TO_OPENCV

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OPENGL_TO_OPENCV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


def main(
    input_dir: Path,
    gt_mesh_path: Path,
    transformation_path: Path,
):
    mesh = trimesh.load(os.path.join(gt_mesh_path), force="mesh", process=False)

    initial_transformation = np.array(
        json.load(open(os.path.join(transformation_path)))["gt_transformation"]
    ).reshape(4, 4)
    initial_transformation = np.linalg.inv(initial_transformation)

    mesh = mesh.apply_transform(initial_transformation)

    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.int64)
    mesh = Meshes(verts=[vertices], faces=[faces]).to(device)

    Rz_rot = py3d_transform.euler_angles_to_matrix(
        torch.tensor([0.0, 0.0, math.pi]), convention="XYZ"
    ).cuda()
    output_path = os.path.join(input_dir, "reference_depth")

    os.makedirs(output_path, exist_ok=True)

    transformation_info = json.load(
        open(os.path.join(input_dir, "transformations_colmap.json"))
    )
    frames = transformation_info["frames"]

    if "fl_x" in transformation_info:
        intrinsic_matrix_base = (
            np.array(
                [
                    transformation_info["fl_x"],
                    0,
                    transformation_info["cx"],
                    0,
                    0,
                    transformation_info["fl_y"],
                    transformation_info["cy"],
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                ]
            )
            .reshape(4, 4)
            .astype(np.float32)
        )
        H = transformation_info["h"]
        W = transformation_info["w"]

    for frame in frames:
        image_path = frame["file_path"]
        image_name = image_path.split("/")[-1]

        if "fl_x" in frame:
            intrinsic_matrix_base = (
                np.array(
                    [
                        frame["fl_x"],
                        0,
                        frame["cx"],
                        0,
                        0,
                        frame["fl_y"],
                        frame["cy"],
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        1,
                    ]
                )
                .reshape(4, 4)
                .astype(np.float32)
            )
            H = frame["h"]
            W = frame["w"]

        intrinsic_matrix = torch.from_numpy(intrinsic_matrix_base).unsqueeze(0).cuda()

        focal_length = torch.stack(
            [intrinsic_matrix[:, 0, 0], intrinsic_matrix[:, 1, 1]], dim=-1
        )
        principal_point = intrinsic_matrix[:, :2, 2]

        image_size = torch.tensor([[H, W]]).cuda()

        image_size_wh = image_size.flip(dims=(1,))

        s = image_size.min(dim=1, keepdim=True)[0] / 2

        s.expand(-1, 2)
        image_size_wh / 2.0

        c2w = frame["transform_matrix"]

        c2w = np.matmul(np.array(c2w), OPENGL_TO_OPENCV).astype(np.float32)
        c2w = np.linalg.inv(c2w)
        c2w = torch.from_numpy(c2w).cuda()

        R = c2w[:3, :3]
        T = c2w[:3, 3]

        R2 = (Rz_rot @ R).permute(-1, -2)
        T2 = Rz_rot @ T

        cameras = PerspectiveCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            R=R2.unsqueeze(0),
            T=T2.unsqueeze(0),
            image_size=image_size,
            in_ndc=False,
            # K = intrinsic_matrix,
            device=device,
        )

        raster_settings = RasterizationSettings(
            image_size=(H, W), blur_radius=0.0, faces_per_pixel=1
        )

        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

        print("start rendering")
        render_img = rasterizer(mesh.to(device)).zbuf[0, ..., 0]
        render_img = render_img.cpu().numpy()

        render_img[render_img < 0] = 0

        render_img = (render_img * 1000).astype(np.uint16)
        print(render_img.max())
        if image_name.endswith("jpg"):
            image_name = image_name.replace("jpg", "png")
        cv2.imwrite(os.path.join(output_path, image_name), render_img)
        # import matplotlib.pyplot as plt
        # plt.imsave(os.path.join(output_path, image_name), render_img, cmap="viridis")


if __name__ == "__main__":
    tyro.cli(main)
