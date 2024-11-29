import open3d as o3d
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import os
import cv2
from rich.console import Console
from rich.progress import track
from natsort import natsorted
from copy import deepcopy
from utils.general_utils import load_from_json
import tyro
from utils.point_utils import backproject
from utils.point_utils import compute_angle_between_normals
from PIL import Image

CONSOLE = Console(width=120)
OPENGL_TO_OPENCV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
SCALE_FACTOR = 0.001


def depth_path_to_array(
    depth_path: Path, scale_factor: float = SCALE_FACTOR, return_color=False
) -> np.ndarray:
    if depth_path.suffix == ".png":
        depth = cv2.imread(str(depth_path.absolute()), cv2.IMREAD_ANYDEPTH)
    elif depth_path.suffix == ".npy":
        depth = np.load(depth_path, allow_pickle=True)
        if len(depth.shape) == 3:
            depth = depth[..., 0]
    else:
        raise Exception(f"Format is not supported {depth_path.suffix}")
    depth = depth * scale_factor
    depth = depth.astype(np.float32)
    return depth


@dataclass
class DepthNormalConsistency:
    """
    Check the normal consistency between the normals from the pre-trained model and the normals from the depth map,
    generate depth confidence mask based on the normal consistency.
    """

    data_dir: Path = Path("dataset/room_datasets/vr_room/iphone/long_capture")
    """Path to data root"""
    transforms_name: str = "transformations_colmap.json"
    """transforms file name"""

    def main(self):
        if os.path.exists(os.path.join(self.data_dir, self.transforms_name)):
            CONSOLE.log(f"Found path to {self.transforms_name}")
        else:
            raise Exception(f"Could not find {self.transforms_name}")

        output_normal_path = os.path.join(self.data_dir, "depth_normals")
        output_mask_path = os.path.join(self.data_dir, "depth_normals_mask")

        os.makedirs(output_normal_path, exist_ok=True)
        os.makedirs(output_mask_path, exist_ok=True)

        mono_normal_path = os.path.join(self.data_dir, "normals_from_pretrain")

        transforms = load_from_json(self.data_dir / Path(self.transforms_name))
        assert "frames" in transforms
        sorted_frames = natsorted(transforms["frames"], key=lambda x: x["file_path"])
        converted_json = deepcopy(transforms)
        converted_json["frames"] = deepcopy(sorted_frames)
        num_frames = len(sorted_frames)
        CONSOLE.log(f"{num_frames} frames to process ...")
        if "fl_x" in transforms:
            fx = transforms["fl_x"]
            fy = transforms["fl_y"]
            cx = transforms["cx"]
            cy = transforms["cy"]
            h = transforms["h"]
            w = transforms["w"]
        else:
            fx = transforms["frames"][0]["fl_x"]
            fy = transforms["frames"][0]["fl_y"]
            cx = transforms["frames"][0]["cx"]
            cy = transforms["frames"][0]["cy"]
            h = transforms["frames"][0]["h"]
            w = transforms["frames"][0]["w"]
            # raise NotImplementedError("TODO per frame intrinsics")

        for i in track(range(num_frames), description="Processing frames..."):
            c2w_ref = np.array(sorted_frames[i]["transform_matrix"])
            if c2w_ref.shape[0] != 4:
                c2w_ref = np.concatenate([c2w_ref, np.array([[0, 0, 0, 1]])], axis=0)
            c2w_ref = c2w_ref @ OPENGL_TO_OPENCV
            depth_i = depth_path_to_array(
                self.data_dir / Path(sorted_frames[i]["depth_file_path"])
            )
            depth_i = cv2.resize(depth_i, (w, h), interpolation=cv2.INTER_NEAREST)
            means3d, image_coords = backproject(
                depths=depth_i,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                img_size=(w, h),
                c2w=c2w_ref,
            )
            cam_center = c2w_ref[:3, 3]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(means3d)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=200)
            )
            normals = np.array(pcd.normals)

            # check normal direction: if ray dir and normal angle is smaller than 90, reverse normal
            ray_dir = means3d - cam_center.reshape(1, 3)
            normal_dir_not_correct = (ray_dir * normals).sum(axis=-1) > 0
            normals[normal_dir_not_correct] = -normals[normal_dir_not_correct]

            normals = normals.reshape(h, w, 3)
            # color normal
            normals = (normals + 1) / 2
            saved_normals = (normals * 255).astype(np.uint8)
            name = sorted_frames[i]["file_path"].split("/")[-1]

            cv2.imwrite(
                os.path.join(output_normal_path, name),
                saved_normals,
            )

            mono_normal = Image.open(
                os.path.join(mono_normal_path, name.replace("jpg", "png"))
            )
            mono_normal = np.array(mono_normal) / 255.0
            h, w, _ = mono_normal.shape
            w2c = np.linalg.inv(c2w_ref)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            normal = mono_normal.reshape(-1, 3).transpose(1, 0)
            normal = (normal - 0.5) * 2
            normal = (R @ normal).T
            normal = normal / np.linalg.norm(normal, axis=1, keepdims=True)
            mono_normal = normal.reshape(h, w, 3) * 0.5 + 0.5

            degree_map = compute_angle_between_normals(normals, mono_normal)
            mask = (degree_map > 10).astype(np.uint8)
            cv2.imwrite(
                os.path.join(output_mask_path, name),
                mask * 255.0,
            )


if __name__ == "__main__":
    tyro.cli(DepthNormalConsistency).main()
