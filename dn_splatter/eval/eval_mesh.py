"""
adapted from go_surf scripts:
https://github.com/JingwenWang95/go-surf/blob/master/tools/mesh_metrics.py#L33
"""

from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
import tyro
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree

transform = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]
)


def get_align_transformation(rec_meshfile, gt_meshfile):
    """
    Get the transformation matrix to align the reconstructed mesh to the ground truth mesh.
    """
    o3d_rec_mesh = o3d.io.read_triangle_mesh(rec_meshfile)
    o3d_gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)
    o3d_rec_pc = o3d.geometry.PointCloud(points=o3d_rec_mesh.vertices)
    o3d_gt_pc = o3d.geometry.PointCloud(points=o3d_gt_mesh.vertices)
    trans_init = np.eye(4)
    threshold = 0.1
    reg_p2p = o3d.pipelines.registration.registration_icp(
        o3d_rec_pc,
        o3d_gt_pc,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    transformation = reg_p2p.transformation
    return transformation


def get_threshold_percentage(dist, thresholds):
    """Evaluates a point cloud.
    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    """
    in_threshold = [(dist <= t).astype(np.float32).mean() for t in thresholds]
    return in_threshold


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    """Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    """
    kdtree = cKDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array([np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def get_colored_pcd(pcd, metric):
    cmap = plt.cm.get_cmap("jet")
    color = cmap(metric / 0.10)[..., :3]
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    pcd_o3d.colors = o3d.utility.Vector3dVector(color)
    return pcd_o3d


def compute_metrics(mesh_pred, mesh_target):
    area_pred = int(mesh_pred.area * 1e4)
    area_tgt = int(mesh_target.area * 1e4)
    print("pred: {}, target: {}".format(area_pred, area_tgt))

    # iou, v_gt, v_pred = compute_iou(mesh_pred, mesh_target)

    pointcloud_pred, idx = mesh_pred.sample(area_pred, return_index=True)
    pointcloud_pred = pointcloud_pred.astype(np.float32)
    normals_pred = mesh_pred.face_normals[idx]

    pointcloud_tgt, idx = mesh_target.sample(area_tgt, return_index=True)
    pointcloud_tgt = pointcloud_tgt.astype(np.float32)
    normals_tgt = mesh_target.face_normals[idx]

    thresholds = np.array([0.05])

    # for every point in gt compute the min distance to points in pred
    completeness, completeness_normals = distance_p2p(
        pointcloud_tgt, normals_tgt, pointcloud_pred, normals_pred
    )
    recall = get_threshold_percentage(completeness, thresholds)
    completeness2 = completeness**2

    # color gt_point_cloud using completion
    com_mesh = get_colored_pcd(pointcloud_tgt, completeness)

    completeness = completeness.mean()
    completeness2 = completeness2.mean()
    completeness_normals = completeness_normals.mean()

    # Accuracy: how far are th points of the predicted pointcloud
    # from the target pointcloud
    accuracy, accuracy_normals = distance_p2p(
        pointcloud_pred, normals_pred, pointcloud_tgt, normals_tgt
    )
    precision = get_threshold_percentage(accuracy, thresholds)
    accuracy2 = accuracy**2

    # color pred_point_cloud using completion
    acc_mesh = get_colored_pcd(pointcloud_pred, accuracy)

    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()
    accuracy_normals = accuracy_normals.mean()

    # Chamfer distance
    chamferL2 = 0.5 * (completeness2 + accuracy2)
    normals_correctness = 0.5 * completeness_normals + 0.5 * accuracy_normals
    chamferL1 = 0.5 * (completeness + accuracy)

    # F-Score
    F = [
        2 * precision[i] * recall[i] / (precision[i] + recall[i])
        for i in range(len(precision))
    ]
    rst = {
        "Acc": accuracy,  # lower better
        "Comp": completeness,  # lower better
        "C-L1": chamferL1,  # lower better
        "NC": normals_correctness,  # higher better
        "F-score": F[0],  # higher better
    }

    return rst


def main(
    gt_mesh: Path, pred_mesh: Path, align: bool = False, dataset: str = "scannetpp"
):
    gt_mesh_ply = trimesh.load(gt_mesh, process=False)
    pd_mesh_ply = trimesh.load(pred_mesh, process=False)

    if dataset == "scannetpp":
        pd_mesh_ply = pd_mesh_ply.apply_transform(transform)
    if align:
        transformation = get_align_transformation(str(pred_mesh), str(gt_mesh))
        pd_mesh_ply = pd_mesh_ply.apply_transform(transformation)

    o3d.visualization.draw_geometries([pd_mesh_ply.as_open3d, gt_mesh_ply.as_open3d])
    to_align, _ = trimesh.bounds.oriented_bounds(gt_mesh_ply)
    gt_mesh_ply.vertices = (
        to_align[:3, :3] @ gt_mesh_ply.vertices.T + to_align[:3, 3:]
    ).T
    pd_mesh_ply.vertices = (
        to_align[:3, :3] @ pd_mesh_ply.vertices.T + to_align[:3, 3:]
    ).T
    print(
        f"saving pred mesh after tranforms to {str(pred_mesh.parent / pred_mesh.stem) + '_pred_mesh_culled.ply'}"
    )
    pd_mesh_ply.export(str(pred_mesh.parent / pred_mesh.stem) + "_pred_mesh_culled.ply")

    min_points = gt_mesh_ply.vertices.min(axis=0) * 1.01  # 1.01
    max_points = gt_mesh_ply.vertices.max(axis=0) * 1.01  # 1.01

    mask_min = (pd_mesh_ply.vertices - min_points[None]) > 0
    mask_max = (pd_mesh_ply.vertices - max_points[None]) < 0
    mask = np.concatenate((mask_min, mask_max), axis=1).all(axis=1)
    face_mask = mask[pd_mesh_ply.faces].all(axis=1)
    pd_mesh_ply.update_vertices(mask)
    pd_mesh_ply.update_faces(face_mask)

    # pd_mesh_ply.export(str(pred_mesh.parent / pred_mesh.stem) + "_pred_mesh_culled.ply")
    o3d.visualization.draw_geometries([pd_mesh_ply.as_open3d, gt_mesh_ply.as_open3d])

    gt_mesh_ply.export(str(gt_mesh.parent / gt_mesh.stem) + "_pred_mesh_culled.ply")

    rst = compute_metrics(pd_mesh_ply, gt_mesh_ply)

    rst = compute_metrics(pd_mesh_ply, gt_mesh_ply)
    # print(f"Savisng results to: {output / 'metrics.json'}")
    json.dump(rst, open(pred_mesh.parent / "metrics.json", "w"))
    # mesh metrics:
    #    "Acc": accuracy,  # lower better
    #    "Comp": completeness,  # lower better
    #    "C-L1": chamferL1,  # lower better
    #    "NC": normals_correctness,  # higher better
    #    "F-score": F[0],  # higher better
    print(rst)
    print(rst.values())


if __name__ == "__main__":
    tyro.cli(main)
