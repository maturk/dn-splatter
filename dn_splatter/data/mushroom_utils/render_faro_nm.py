import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import trimesh
import tyro
from PIL import Image
from pytorch3d import transforms as py3d_transform
from pytorch3d.ops.interp_face_attrs import interpolate_face_attributes
from pytorch3d.renderer import (
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    RasterizationSettings,
)
from pytorch3d.renderer.blending import BlendParams, softmax_rgb_blend
from pytorch3d.structures import Meshes

# from dn_splatter.utils.camera_utils import OPENGL_TO_OPENCV

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def interpolate_vertex_normals(
    fragments, vertex_textures, faces_packed
) -> torch.Tensor:
    """
    Detemine the normal color for each rasterized face. Interpolate the normal colors for
    vertices which form the face using the barycentric coordinates.
    Args:
        meshes: A Meshes class representing a batch of meshes.
        fragments:
            The outputs of rasterization. From this we use

            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image.
            - barycentric_coords: FloatTensor of shape (N, H, W, K, 3) specifying
              the barycentric coordianates of each pixel
              relative to the faces (in the packed
              representation) which overlap the pixel.

    Returns:
        texels: An normal color per pixel of shape (N, H, W, K, C).
        There will be one C dimensional value for each element in
        fragments.pix_to_face.
    """
    # vertex_textures = meshes.textures.verts_rgb_padded().reshape(-1, 3)  # (V, C)
    # vertex_textures = vertex_textures[meshes.verts_padded_to_packed_idx(), :]

    # X: -1 to + 1: Red: 0 to  255
    # Y: -1 to + 1: Green: 0 to  255
    # Z: 0 to  -1: Blue: 128 to  255

    vertex_textures[:, :2] += 1
    vertex_textures[:, :2] /= 2
    vertex_textures[:, 2] += 3
    vertex_textures[:, 2] /= 4
    vertex_textures /= torch.norm(vertex_textures, p=2, dim=-1).view(
        vertex_textures.shape[0], 1
    )

    faces_textures = vertex_textures[faces_packed]  # (F, 3, C)
    texels = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_textures
    )
    return texels


class NormalShader(nn.Module):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function returns the
    soft aggregated color using all the faces per pixel.
    To use the default values, simply initialize the shader with the desired
    device e.g.
    .. code-block::
        shader = SoftPhongShader(device=torch.device("cuda:0"))
    """

    def __init__(
        self,
        device="cpu",
        cameras=None,
        blend_params=None,
        vertex_textures=None,
        faces_packed=None,
    ):
        super().__init__()

        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()
        self.vertex_textures = vertex_textures
        self.faces_packed = faces_packed

    def forward(self, fragments, mesh, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of SoftPhongShader"
            raise ValueError(msg)

        texels = interpolate_vertex_normals(
            fragments, self.vertex_textures, self.faces_packed
        )
        images = softmax_rgb_blend(texels, fragments, self.blend_params)
        return images


def main(
    input_dir: Path = Path(
        "/home/nvme/kqxure/code/updating_models/nerfstudio/room_datasets/activity/kinect/long_capture/"
    ),
    gt_mesh_path: Path = Path(
        "/home/nvme/kqxure/code/updating_models/nerfstudio/room_datasets/activity/gt_mesh_clean.ply"
    ),
    transformation_path: Path = Path(
        "/home/nvme/kqxure/code/updating_models/nerfstudio/room_datasets/activity/icp_kinect.json"
    ),
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
    output_path = os.path.join(input_dir, "reference_normal")

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

    mesh = mesh.to(device)
    vertex_textures = mesh.verts_normals_packed()  # .to(device)
    faces_packed = mesh.faces_packed()
    OPENGL_TO_OPENCV = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )

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
            image_size=(H, W),
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True,
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=NormalShader(
                device=device,
                cameras=cameras,
                vertex_textures=vertex_textures.clone(),
                faces_packed=faces_packed.clone(),
            ),
        )

        print("start rendering")
        render_img = renderer(mesh)[0, ..., :3] * 255
        render_img = render_img.squeeze().cpu().numpy().astype(np.uint8)

        if image_name.endswith("jpg"):
            image_name = image_name.replace("jpg", "png")

        Image.fromarray(render_img).save(os.path.join(output_path, image_name))


if __name__ == "__main__":
    tyro.cli(main)
