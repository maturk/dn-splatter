import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d
import trimesh
import json
import tyro
from pathlib import Path
import os


def open3d_mesh_from_trimesh(tri_mesh):
    vertices = np.asarray(tri_mesh.vertices)
    faces = np.asarray(tri_mesh.faces)

    # Create open3d TriangleMesh object
    o3d_mesh = o3d.geometry.TriangleMesh()

    # Assign vertices and faces to open3d mesh
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    return o3d_mesh


def calculate_accuracy(
    reconstructed_points, reference_points, percentile=90
):  # Calculat accuracy: How far away 90% of the reconstructed point clouds are from the reference point cloud.
    tree = cKDTree(reference_points)
    distances, _ = tree.query(reconstructed_points)
    return np.percentile(distances, percentile)


def calculate_completeness(
    reconstructed_points, reference_points, threshold=0.05
):  # calucate completeness: What percentage of the reference point cloud is within a specific distance of the reconstructed point cloud.
    tree = cKDTree(reconstructed_points)
    distances, _ = tree.query(reference_points)
    within_threshold = np.sum(distances < threshold) / len(distances)
    return within_threshold * 100


def main(
    export_pd: Path = Path(
        "room_datasets/activity/iphone/long_capture/pointcloud_train_downsample.ply"
    ),
    path_to_room: Path = Path("room_datasets/activity"),
    device_type: Path = Path("iphone"),
    evaluate_protocol: str = "within",
):
    # import predicted pd
    reconstructed_pd = o3d.io.read_point_cloud(str(export_pd))

    # load training pose
    if evaluate_protocol == "within":
        within_pose = json.load(
            open(
                os.path.join(
                    path_to_room, device_type, "long_capture", "transformations.json"
                )
            )
        )
        ref_pose = within_pose["frames"][0]["transform_matrix"]
        with_diff_pose = json.load(
            open(
                os.path.join(
                    path_to_room,
                    device_type,
                    "long_capture",
                    "transformations_colmap.json",
                )
            )
        )
        diff_pose = with_diff_pose["frames"][0]["transform_matrix"]
        align_transformation = np.matmul(np.linalg.inv(ref_pose), diff_pose)
        print(align_transformation)
        reconstructed_pd = reconstructed_pd.transform(align_transformation)

    # load the transformation matrix to convert from colmap pose to reference mesh
    initial_transformation = np.array(
        json.load(
            open(os.path.join(path_to_room, "icp_{}.json".format(str(device_type))))
        )["gt_transformation"]
    ).reshape(4, 4)
    reconstructed_pd = reconstructed_pd.transform(initial_transformation)
    reconstructed_pd = reconstructed_pd.voxel_down_sample(voxel_size=0.01)
    # import reference pd
    reference_pd = o3d.io.read_point_cloud(os.path.join(path_to_room, "gt_pd.ply"))

    reconstructed_points = np.asarray(reconstructed_pd.points)
    reference_points = np.asarray(reference_pd.points)
    accuracy = calculate_accuracy(reconstructed_points, reference_points)
    completeness = calculate_completeness(reconstructed_points, reference_points)
    print(accuracy, completeness)


if __name__ == "__main__":
    tyro.cli(main)
