import open3d as o3d
import cv2
import numpy as np
import os


def generate_iPhone_pointcloud(
    input_folder, meta, i_train, num_points: int = 1_000_000
):
    print("Generating pointcloud from iPhone data...")
    frames = meta["frames"]
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.04,
        sdf_trunc=0.2,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    index = i_train
    samples_per_frame = (num_points + len(index)) // (len(index))

    points_list = []
    colors_list = []

    for frame in frames:
        H, W, fx, fy, cx, cy = (
            frame["h"],
            frame["w"],
            frame["fl_x"],
            frame["fl_y"],
            frame["cx"],
            frame["cy"],
        )
        name = frame["file_path"].split("/")[-1]
        color = cv2.imread(str(input_folder / "rgb" / name))
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color = o3d.geometry.Image(color)

        # pose
        pose = frame["transform_matrix"]
        pose = np.matmul(np.array(pose), OPENGL_TO_OPENCV)
        depth = cv2.imread(
            str(input_folder / "depth" / name.replace("jpg", "png")),
            cv2.IMREAD_ANYDEPTH,
        )
        depth = cv2.resize(depth, (W, H))
        depth = o3d.geometry.Image(depth)

        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False
        )

        volume.integrate(
            rgbd,
            camera_intrinsics,  # type: ignore
            np.linalg.inv(pose),
        )

        pcd = volume.extract_point_cloud()

        # randomly select samples_per_frame points from points
        samples_per_frame = min(samples_per_frame, len(pcd.points))
        mask = random.sample(range(len(pcd.points)), samples_per_frame)
        mask = np.asarray(mask)
        color = np.asarray(pcd.colors)[mask]
        point = np.asarray(pcd.points)[mask]

        points_list.append(np.asarray(point))
        colors_list.append(np.asarray(color))

    points = np.vstack(points_list)
    colors = np.vstack(colors_list)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(os.path.join(input_folder / "point_cloud.ply"), pcd)

    mesh = volume.extract_triangle_mesh()
    o3d.io.write_triangle_mesh(os.path.join(input_folder / "TSDFVolume.ply"), mesh)
