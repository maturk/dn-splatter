"""
nerfstudio format to sdfstudio format
borrowed from sdfstudio https://github.com/autonomousvision/sdfstudio/blob/master/scripts/datasets/process_nerfstudio_to_sdfstudio.py
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


def main(args):
    """
    Given data that follows the nerfstudio format such as the output from colmap or polycam,
    convert to a format that sdfstudio will ingest
    """
    output_dir = Path(args.output_dir)
    input_dir = Path(args.input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cam_params = json.load(open(input_dir / "transforms.json"))

    # === load camera intrinsics and poses ===
    cam_intrinsics = []
    if args.data_type == "colmap":
        cam_intrinsics.append(
            np.array(
                [
                    [cam_params["fl_x"], 0, cam_params["cx"]],
                    [0, cam_params["fl_y"], cam_params["cy"]],
                    [0, 0, 1],
                ]
            )
        )

    frames = cam_params["frames"]
    poses = []
    image_paths = []
    depth_paths = []
    mono_depth_paths = []
    # only load images with corresponding pose info
    # currently in random order??, probably need to sort
    for frame in frames:
        # load intrinsics from polycam
        if args.data_type == "polycam":
            cam_intrinsics.append(
                np.array(
                    [
                        [frame["fl_x"], 0, frame["cx"]],
                        [0, frame["fl_y"], frame["cy"]],
                        [0, 0, 1],
                    ]
                )
            )

        # load poses
        # OpenGL/Blender convention, needs to change to COLMAP/OpenCV convention
        # https://docs.nerf.studio/en/latest/quickstart/data_conventions.html
        # IGNORED for now
        c2w = np.array(frame["transform_matrix"])
        if c2w.shape == (3, 4):
            c2w = np.vstack([c2w, np.array([0, 0, 0, 1])])
        c2w = c2w.reshape(4, 4)
        c2w[0:3, 1:3] *= -1
        poses.append(c2w)

        # load images
        img_path = input_dir / frame["file_path"]
        assert img_path.exists()
        image_paths.append(img_path)

        # load sensor depths
        if "depth_file_path" in frame:
            depth_path = input_dir / frame["depth_file_path"]
            assert depth_path.exists()
            depth_paths.append(depth_path)
        if "mono_depth_file_path" in frame:
            mono_depth_path = input_dir / frame["mono_depth_file_path"]
            assert mono_depth_path.exists()
            mono_depth_paths.append(mono_depth_paths)

    # Check correctness
    assert len(poses) == len(image_paths)
    assert len(poses) == len(cam_intrinsics) or len(cam_intrinsics) == 1

    # Filter invalid poses
    poses = np.array(poses)
    valid_poses = np.isfinite(poses).all(axis=2).all(axis=1)
    min_vertices = poses[:, :3, 3][valid_poses].min(axis=0)
    max_vertices = poses[:, :3, 3][valid_poses].max(axis=0)

    # === Normalize the scene ===
    if args.scene_type in ["indoor", "object"]:
        # Enlarge bbox by 1.05 for object scene and by 5.0 for indoor scene
        # TODO: Adaptively estimate `scene_scale_mult` based on depth-map or point-cloud prior
        if not args.scene_scale_mult:
            args.scene_scale_mult = 1.05 if args.scene_type == "object" else 5.0
        scene_scale = 2.0 / (
            np.max(max_vertices - min_vertices) * args.scene_scale_mult
        )
        scene_center = (min_vertices + max_vertices) / 2.0
        # normalize pose to unit cube
        poses[:, :3, 3] -= scene_center
        poses[:, :3, 3] *= scene_scale
        # calculate scale matrix
        scale_mat = np.eye(4).astype(np.float32)
        scale_mat[:3, 3] -= scene_center
        scale_mat[:3] *= scene_scale
        scale_mat = np.linalg.inv(scale_mat)
    else:
        scene_scale = 1.0
        scale_mat = np.eye(4).astype(np.float32)

    # === Construct the scene box ===
    if args.scene_type == "indoor":
        scene_box = {
            "aabb": [[-1, -1, -1], [1, 1, 1]],
            "near": 0.05,
            "far": 2.5,
            "radius": 1.0,
            "collider_type": "box",
        }
    elif args.scene_type == "object":
        scene_box = {
            "aabb": [[-1, -1, -1], [1, 1, 1]],
            "near": 0.05,
            "far": 2.0,
            "radius": 1.0,
            "collider_type": "near_far",
        }
    elif args.scene_type == "unbound":
        # TODO: case-by-case near far based on depth prior
        #  such as colmap sparse points or sensor depths
        scene_box = {
            "aabb": [min_vertices.tolist(), max_vertices.tolist()],
            "near": 0.05,
            "far": 2.5 * np.max(max_vertices - min_vertices),
            "radius": np.min(max_vertices - min_vertices) / 2.0,
            "collider_type": "box",
        }

    # === Resize the images and intrinsics ===
    # Only resize the images when we want to use mono prior
    if args.data_type == "colmap":
        h, w = cam_params["h"], cam_params["w"]
    else:
        h, w = frames[0]["h"], frames[0]["w"]

    # === Construct the frames in the meta_data.json ===
    frames = []
    out_index = 0
    mono_depth, sensor_depth = False, False
    if len(mono_depth_paths) > 0:
        mono_depth = True
    else:
        sensor_depth = True

    for idx, (valid, pose, image_path) in enumerate(
        tqdm(zip(valid_poses, poses, image_paths))
    ):
        if not valid:
            continue

        # save rgb image
        out_img_path = output_dir / f"{out_index:06d}_rgb.png"
        if args.save_imgs:
            img = Image.open(image_path)
            img.save(out_img_path)
        rgb_path = str(out_img_path.relative_to(output_dir))

        frame = {
            "rgb_path": rgb_path,
            "camtoworld": pose.tolist(),
            "intrinsics": cam_intrinsics[0].tolist()
            if args.data_type == "colmap"
            else cam_intrinsics[idx].tolist(),
        }

        if sensor_depth:
            # load depth
            if args.save_imgs:
                depth_path = depth_paths[idx]
                out_depth_path = output_dir / f"{out_index:06d}_sensor_depth.png"
                depth = cv2.imread(str(depth_path), -1).astype(np.float32) / 1000.0
                # scale depth as we normalize the scene to unit box
                new_depth = np.copy(depth) * scene_scale
                plt.imsave(out_depth_path, new_depth, cmap="viridis")
                np.save(str(out_depth_path).replace(".png", ".npy"), new_depth)
            frame["mono_depth_path"] = rgb_path.replace("_rgb.png", "_sensor_depth.npy")
        if mono_depth:
            # load mono depth
            if args.save_imgs:
                mono_depth_path = mono_depth_paths[idx]
                out_mono_depth_path = output_dir / f"{out_index:06d}_mono_depth.png"
                mono_depth = (
                    cv2.imread(str(mono_depth_path), -1).astype(np.float32) / 1000.0
                )
                # scale depth as we normalize the scene to unit box
                new_mono_depth = np.copy(mono_depth) * scene_scale
                plt.imsave(out_mono_depth_path, new_mono_depth, cmap="viridis")
                np.save(
                    str(out_mono_depth_path).replace(".png", ".npy"), new_mono_depth
                )
            frame["mono_depth_path"] = rgb_path.replace("_rgb.png", "_mono_depth.npy")

        # load normal
        out_normal_path = output_dir / f"{out_index:06d}_normal.png"
        if args.normal_from_pretrain:
            # load normal
            normal_path = input_dir / "normals_from_pretrain" / f"{image_path.stem}.png"
        elif args.normal_from_depth:
            # load normal from sensor depth
            normal_path = input_dir / "normals_from_depth" / f"{image_path.stem}.png"
        if args.normal_from_pretrain or args.normal_from_depth:
            if args.save_imgs:
                normal = Image.open(normal_path)
                normal.save(out_normal_path)
                normal = np.array(normal)
                np.save(str(out_normal_path).replace(".png", ".npy"), normal / 255.0)
            frame["mono_normal_path"] = rgb_path.replace("_rgb.png", "_normal.npy")

        frames.append(frame)
        out_index += 1

    # === Construct and export the metadata ===
    meta_data = {
        "camera_model": "OPENCV",
        "height": h,
        "width": w,
        "has_foreground_mask": False,
        "pairs": None,
        "worldtogt": scale_mat.tolist(),
        "has_mono_prior": True,
        "scene_box": scene_box,
        "frames": frames,
    }
    with open(output_dir / "meta_data.json", "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=4)

    print(f"Done! The processed data has been saved in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="preprocess nerfstudio dataset to sdfstudio dataset, "
        "currently support colmap and polycam"
    )

    parser.add_argument(
        "--input_dir", required=True, help="path to nerfstudio data directory"
    )
    parser.add_argument(
        "--output_dir", required=True, help="path to output data directory"
    )
    parser.add_argument(
        "--data-type", dest="data_type", required=True, choices=["colmap", "polycam"]
    )
    parser.add_argument(
        "--scene-type",
        dest="scene_type",
        required=True,
        choices=["indoor", "object", "unbound"],
        help="The scene will be normalized into a unit sphere when selecting indoor or object.",
    )
    parser.add_argument(
        "--scene-scale-mult",
        dest="scene_scale_mult",
        type=float,
        default=None,
        help="The bounding box of the scene is firstly calculated by the camera positions, "
        "then multiply with scene_scale_mult",
    )
    parser.add_argument(
        "--normal_from_pretrain",
        action="store_true",
        help="Use normal from pretrain model",
    )
    parser.add_argument(
        "--normal_from_depth", action="store_true", help="Use normal from sensor depth"
    )
    parser.add_argument(
        "--save-imgs",
        action="store_true",
        required=True,
        help="Use normal from mono depth",
    )

    args = parser.parse_args()

    main(args)
