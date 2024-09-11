from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from dn_splatter.scripts.depth_from_pretrain import depth_from_pretrain
from dn_splatter.utils.camera_utils import project_pix
from dn_splatter.utils.utils import depth_path_to_tensor, get_filename_list, save_depth
from rich.console import Console
from rich.progress import track
from torch import Tensor

from nerfstudio.data.utils.colmap_parsing_utils import (
    qvec2rotmat,
    read_cameras_binary,
    read_cameras_text,
    read_images_binary,
    read_images_text,
    read_points3D_binary,
    read_points3D_text,
)
from nerfstudio.utils import colormaps
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE

CONSOLE = Console(width=120)
BATCH_SIZE = 100
device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ColmapToAlignedMonoDepths:
    """Converts COLMAP dataset SfM points to scale aligned mono-depth estimates

    COLMAP dataset is expected to be in the following form:
    <data>
    |---image_path
    |   |---<image 0>
    |   |---<image 1>
    |   |---...
    |---colmap
        |---sparse
            |---0
                |---cameras.bin
                |---images.bin
                |---points3D.bin

    This function provides the following directories in that <data> root
    |---sfm_depths
    |   |---<sfm_depth 0>
    |   |---<sfm_depth 1>
    |   |---...
    |---mono_depth
    |   |---<mono_depth 0>.png
    |   |---<mono_depth 0>_aligned.npy
    """

    data: Path
    """Input dataset path"""
    sparse_path: Path = Path("colmap/sparse/0")
    """Default path of colmap sparse dir"""
    img_dir_name: str = "images"
    """Directory name of where input images are stored. Default is '/images', but you can remap it to something else. """
    mono_depth_network: Literal["zoe"] = "zoe"
    """What mono depth network to use"""
    skip_colmap_to_depths: bool = False
    """Skip colmap to sfm step"""
    skip_mono_depth_creation: bool = False
    """Skip mono depth creation"""
    skip_alignment: bool = False
    """Skip alignment"""
    iterations: int = 1000
    """Number of grad descent iterations to align depths"""
    align_method: Literal["closed_form", "grad_descent"] = "closed_form"
    """Use closed form solution for depth alignment or graident descent"""

    def main(self) -> None:
        sfm_depth_path = self.data / Path("sfm_depths")

        if not self.skip_colmap_to_depths:
            CONSOLE.print("Generating sfm depth maps from sparse colmap reconstruction")
            colmap_sfm_points_to_depths(
                recon_dir=self.data / self.sparse_path,
                output_dir=sfm_depth_path,
                include_depth_debug=True,
                input_images_dir=self.data / self.img_dir_name,
            )

        if not self.skip_mono_depth_creation:
            CONSOLE.print("Computing mono depth estimates")
            if not (self.data / Path("mono_depth")).exists() or True:
                depth_from_pretrain(
                    input_folder=self.data,
                    img_dir_name=self.img_dir_name,
                    path_to_transforms=None,
                    create_new_transforms=False,
                    is_euclidean_depth=False,
                    pretrain_model=self.mono_depth_network,
                )
            else:
                CONSOLE.print("Found previous /mono_depth path")
        if not self.skip_alignment:
            CONSOLE.print("Aligning sparse depth maps with mono estimates")
            # Align sparse sfm depth maps with mono depth maps
            batch_size = BATCH_SIZE
            sfm_depth_filenames = get_filename_list(
                image_dir=self.data / Path("sfm_depths"), ends_with=".npy"
            )
            mono_depth_filenames = get_filename_list(
                image_dir=self.data / Path("mono_depth"), ends_with=".npy"
            )
            # filter out aligned depth and frames not have pose
            sfm_name = [item.name for item in sfm_depth_filenames]
            mono_depth_filenames = [
                item
                for item in mono_depth_filenames
                if "_aligned.npy" not in item.name and str(item.stem) in str(sfm_name)
            ]
            assert len(sfm_depth_filenames) == len(mono_depth_filenames)

            H, W = depth_path_to_tensor(sfm_depth_filenames[0]).shape[:2]

            num_frames = len(sfm_depth_filenames)

            for batch_index in range(0, num_frames, batch_size):
                batch_sfm_frames = sfm_depth_filenames[
                    batch_index : batch_index + batch_size
                ]
                batch_mono_frames = mono_depth_filenames[
                    batch_index : batch_index + batch_size
                ]

                with torch.no_grad():
                    mono_depth_tensors = []
                    sparse_depths = []

                    for frame_index in range(len(batch_sfm_frames)):
                        sfm_frame = batch_sfm_frames[frame_index]
                        mono_frame = batch_mono_frames[frame_index]
                        mono_depth = depth_path_to_tensor(
                            mono_frame,
                            return_color=False,
                            scale_factor=0.001 if mono_frame.suffix == ".png" else 1,
                        )  # note that npy depth maps are in meters
                        mono_depth_tensors.append(mono_depth)

                        sfm_depth = depth_path_to_tensor(
                            sfm_frame, return_color=False, scale_factor=1
                        )
                        sparse_depths.append(sfm_depth)

                    mono_depth_tensors = torch.stack(mono_depth_tensors, dim=0)
                    sparse_depths = torch.stack(sparse_depths, dim=0)

                if self.align_method == "closed_form":
                    mask = (sparse_depths > 0.1) & (sparse_depths < 10.0)
                    scale, shift = compute_scale_and_shift(
                        mono_depth_tensors, sparse_depths, mask=mask
                    )
                    scale = scale.unsqueeze(1).unsqueeze(2)
                    shift = shift.unsqueeze(1).unsqueeze(2)
                    depth_aligned = scale * mono_depth_tensors + shift
                    mse_loss = torch.nn.MSELoss()
                    avg = mse_loss(depth_aligned[mask], sparse_depths[mask])
                    CONSOLE.print(
                        f"[bold yellow]Average depth alignment error for batch depths is: {avg:3f} which is {'good' if avg<0.2 else 'bad'}"
                    )

                elif self.align_method == "grad_descent":
                    depth_aligned = grad_descent(
                        mono_depth_tensors, sparse_depths, iterations=self.iterations
                    )

                # save depths
                for idx in track(
                    range(depth_aligned.shape[0]),
                    description="saving aligned depth images...",
                ):
                    depth_aligned_numpy = depth_aligned[idx, ...].detach().cpu().numpy()
                    file_name = str(Path(batch_mono_frames[idx]).with_suffix(""))
                    # save only npy
                    np.save(Path(file_name + "_aligned.npy"), depth_aligned_numpy)


# copy from monosdf
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def grad_descent(
    mono_depth_tensors: torch.Tensor,
    sparse_depths: torch.Tensor,
    iterations: int = 1000,
    lr: float = 0.1,
    threshold: float = 0.0,
) -> Tensor:
    """Align mono depth estimates with sparse depths.

    Args:
        mono_depth_tensors: mono depths
        sparse_depths: sparse sfm points
        H: height
        W: width
        iterations: number of gradient descent iterations
        lr: learning rate
        threshold: masking threshold of invalid depths. Default 0.

    Returns:
        aligned_depths: tensor of scale aligned mono depths
    """
    aligned_mono_depths = []
    for idx in track(
        range(mono_depth_tensors.shape[0]),
        description="Alignment with grad descent ...",
    ):
        scale = torch.nn.Parameter(
            torch.tensor([1.0], device=device, dtype=torch.float)
        )
        shift = torch.nn.Parameter(
            torch.tensor([0.0], device=device, dtype=torch.float)
        )

        estimated_mono_depth = mono_depth_tensors[idx, ...].float().to(device)
        sparse_depth = sparse_depths[idx].float().to(device)

        mask = sparse_depth > threshold
        estimated_mono_depth_map_masked = estimated_mono_depth[mask]
        sparse_depth_masked = sparse_depth[mask]

        mse_loss = torch.nn.MSELoss()
        optimizer = torch.optim.Adam([scale, shift], lr=lr)

        avg_err = []
        for step in range(iterations):
            optimizer.zero_grad()
            loss = mse_loss(
                scale * estimated_mono_depth_map_masked + shift, sparse_depth_masked
            )
            loss.backward()
            optimizer.step()
        avg_err.append(loss.item())
        aligned_mono_depths.append(scale * estimated_mono_depth + shift)

    avg = sum(avg_err) / len(avg_err)
    CONSOLE.print(
        f"[bold yellow]Average depth alignment error for batch depths is: {avg:3f} which is {'good' if avg<0.2 else 'bad'}"
    )
    return torch.stack(aligned_mono_depths, dim=0)


def colmap_sfm_points_to_depths(
    recon_dir: Path,
    output_dir: Path,
    min_depth: float = 0.001,
    max_depth: float = 1000,
    max_repoj_err: float = 2.5,
    min_n_visible: int = 5,
    include_depth_debug: bool = True,
    input_images_dir: Optional[Path] = Path(),
) -> Dict[int, Path]:
    """Converts COLMAP's points3d.bin to sparse depth maps

    Args:
        recon_dir: Path to the reconstruction directory, e.g. "sparse/0"
        output_dir: Path to the output directory.
        verbose: If True, logs progress of depth image creation.
        min_depth: Discard points closer than this to the camera.
        max_depth: Discard points farther than this from the camera.
        max_repoj_err: Discard points with reprojection error greater than this
          amount (in pixels).
        min_n_visible: Discard 3D points that have been triangulated with fewer
          than this many frames.
        include_depth_debug: Also include debug images showing depth overlaid
          upon RGB.

    Returns:
        Depth file paths indexed by COLMAP image id
    """
    depth_scale_to_integer_factor = 1

    if (recon_dir / "points3D.bin").exists():
        ptid_to_info = read_points3D_binary(recon_dir / "points3D.bin")
        cam_id_to_camera = read_cameras_binary(recon_dir / "cameras.bin")
        im_id_to_image = read_images_binary(recon_dir / "images.bin")
    elif (recon_dir / "points3D.txt").exists():
        ptid_to_info = read_points3D_text(recon_dir / "points3D.txt")
        cam_id_to_camera = read_cameras_text(recon_dir / "cameras.txt")
        im_id_to_image = read_images_text(recon_dir / "images.txt")
    # Only support first camera
    CAMERA_ID = 1
    W = cam_id_to_camera[CAMERA_ID].width
    H = cam_id_to_camera[CAMERA_ID].height

    iter_images = iter(im_id_to_image.items())
    image_id_to_depth_path = {}

    for im_id, im_data in track(iter_images, description="..."):
        # TODO(1480) BEGIN delete when abandoning colmap_parsing_utils
        pids = [pid for pid in im_data.point3D_ids if pid != -1]
        xyz_world = np.array([ptid_to_info[pid].xyz for pid in pids])
        # delete
        # xyz_world = np.array([p.xyz for p in ptid_to_info.values()])
        rotation = qvec2rotmat(im_data.qvec)

        z = (rotation @ xyz_world.T)[-1] + im_data.tvec[-1]
        errors = np.array([ptid_to_info[pid].error for pid in pids])
        n_visible = np.array([len(ptid_to_info[pid].image_ids) for pid in pids])
        uv = np.array(
            [
                im_data.xys[i]
                for i in range(len(im_data.xys))
                if im_data.point3D_ids[i] != -1
            ]
        )

        idx = np.where(
            (z >= min_depth)
            & (z <= max_depth)
            & (errors <= max_repoj_err)
            & (n_visible >= min_n_visible)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < W)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < H)
        )
        z = z[idx]
        uv = uv[idx]

        uu, vv = uv[:, 0].astype(int), uv[:, 1].astype(int)
        depth = np.zeros((H, W), dtype=np.float32)
        depth[vv, uu] = z

        depth_img = depth_scale_to_integer_factor * depth

        out_name = Path(str(im_data.name)).stem
        depth_path = output_dir / out_name

        save_depth(
            depth=depth_img, depth_path=depth_path, scale_factor=1, verbose=False
        )

        image_id_to_depth_path[im_id] = depth_path
        if include_depth_debug:
            assert (
                input_images_dir is not None
            ), "Need explicit input_images_dir for debug images"
            assert input_images_dir.exists(), input_images_dir

            depth_flat = depth.flatten()[:, None]
            overlay = (
                255.0
                * colormaps.apply_depth_colormap(torch.from_numpy(depth_flat)).numpy()
            )
            overlay = overlay.reshape([H, W, 3])
            input_image_path = input_images_dir / im_data.name
            input_image = cv2.imread(str(input_image_path))  # type: ignore

            # BUG: why is input image not == overlay image shape?
            if input_image.shape[:2] != overlay.shape[:2]:
                print("images are not the right size!")
                quit()
            debug = 0.3 * input_image + 0.7 + overlay
            out_name = out_name + ".debug.jpg"
            output_path = output_dir / "debug_depth" / out_name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), debug.astype(np.uint8))  # type: ignore

    return image_id_to_depth_path


def sdfstudio_grad_descent(
    mono_depth_tensors: torch.Tensor,
    sparse_depths: torch.Tensor,
    H=int,
    W=int,
    iterations: int = 1000,
    lr: float = 0.1,
    threshold: float = 0.0,
) -> Tensor:
    """Align mono depth estimates with sparse depths.

    Args:
        mono_depth_tensors: mono depths
        sparse_depths: sparse sfm points
        H: height
        W: width
        iterations: number of gradient descent iterations
        lr: learning rate
        threshold: masking threshold of invalid depths. Default 0.

    Returns:
        aligned_depths: tensor of scale aligned mono depths
    """
    aligned_mono_depths = []
    for idx in track(
        range(mono_depth_tensors.shape[0]),
        description="Depth align with sparse point cloud ...",
    ):
        scale = torch.nn.Parameter(torch.tensor([1], device=device, dtype=torch.float))
        shift = torch.nn.Parameter(torch.tensor([0], device=device, dtype=torch.float))

        estimated_mono_depth = mono_depth_tensors[idx, ...].float().to(device)
        sparse_points = sparse_depths[idx].float().to(device)

        sparse_points[:, :2] = uv = torch.floor(sparse_points[:, :2] - 0.5).long()
        valid_indices = (
            (uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 0) & (uv[:, 1] < H)
        )

        sparse_points = sparse_points[valid_indices]
        sparse_depth_map = 0.0 * torch.ones((H, W, 1), dtype=torch.float32).to(
            device
        )  # type: ignore
        for i in range(sparse_points.shape[0]):
            sparse_depth_map[
                sparse_points[i, 1].long(), sparse_points[i, 0].long(), 0
            ] = sparse_points[i, 2]

        mask = sparse_depth_map > threshold
        estimated_mono_depth_map_masked = estimated_mono_depth[mask]
        sparse_depth_masked = sparse_depth_map[mask]

        mse_loss = torch.nn.MSELoss()
        optimizer = torch.optim.Adam([scale, shift], lr=lr)

        avg_err = []
        for step in range(iterations):
            optimizer.zero_grad()
            loss = mse_loss(
                scale * estimated_mono_depth_map_masked + shift, sparse_depth_masked
            )
            loss.backward()
            optimizer.step()
        avg_err.append(loss.item())
        aligned_mono_depths.append(scale * estimated_mono_depth + shift)
    avg = sum(avg_err) / len(avg_err)
    CONSOLE.print(
        f"[bold yellow]Average depth alignment error for batch depths is: {avg:3f} which is {'good' if avg<0.2 else 'bad'}"
    )
    return torch.stack(aligned_mono_depths, dim=0)


def sdfstudio_alignment(input_dir: Path, iterations: int = 1000):
    """
    Align SDFStudio formatted data with mono depth estimates

    Args:
        input_dir: path to dataset scan root directory
        save_depth: whether to save depths
        iterations: number of gradient descent alignment iterations

    Returns:
        None
    """
    meta = load_from_json(input_dir / "meta_data.json")
    num_frames = len(meta["frames"])
    batch_size = BATCH_SIZE
    H, W = int(meta["height"]), int(meta["width"])
    for batch_index in range(0, len(meta["frames"]), batch_size):
        CONSOLE.print(
            f"[bold green]Processing batch {batch_index // batch_size} / {num_frames//batch_size}"
        )

        batch_frames = meta["frames"][batch_index : batch_index + batch_size]
        with torch.no_grad():
            mono_depth_tensors = []
            sparse_depths = []
            for frame_index in range(len(batch_frames)):
                frame = meta["frames"][batch_index : batch_index + batch_size][
                    frame_index
                ]
                # load depth
                mono_depth = depth_path_to_tensor(
                    input_dir / frame["mono_depth_path"],
                    return_color=False,
                    scale_factor=1,
                )
                mono_depth_tensors.append(mono_depth)

                # load intrinsics and extrinsics
                intrinsic = np.array(frame["intrinsics"]).reshape(4, 4)
                c2w = torch.from_numpy(
                    np.array(frame["camtoworld"]).reshape(4, 4)
                ).float()

                # load sparse points
                points = torch.from_numpy(
                    np.loadtxt(input_dir / frame["sfm_sparse_points_view"])
                ).float()
                # project sparse points to image plane
                uv_depth = project_pix(
                    p=points,
                    fx=intrinsic[0, 0],
                    fy=intrinsic[1, 1],
                    cx=intrinsic[0, 2],
                    cy=intrinsic[1, 2],
                    c2w=c2w,
                    device="cpu",  # type: ignore
                    return_z_depths=True,
                )
                sparse_depths.append(uv_depth)

            mono_depth_tensors = torch.stack(mono_depth_tensors, dim=0)

        depth_aligned = sdfstudio_grad_descent(
            mono_depth_tensors, sparse_depths, H, W, iterations=iterations
        )

        # save depths
        for idx in track(
            range(depth_aligned.shape[0]), description="saving aligned depth images..."
        ):
            depth_aligned_numpy = depth_aligned[idx, ...].detach().cpu().numpy()
            file_name = meta["frames"][batch_index : batch_index + batch_size][idx][
                "mono_depth_path"
            ].split(".")[0]
            np.save(input_dir / Path(file_name + "_aligned.npy"), depth_aligned_numpy)
            image_save_name = Path(file_name + "_aligned.png")
            plt.imsave(
                str(input_dir / image_save_name),
                depth_aligned_numpy[..., 0],
                cmap="viridis",
            )


if __name__ == "__main__":
    tyro.cli(ColmapToAlignedMonoDepths).main()
