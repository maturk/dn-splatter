"""
adapted from go_surf scripts:
https://github.com/JingwenWang95/go-surf/blob/master/tools/mesh_metrics.py#L33
"""

import json
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import cv2
import numpy as np
import open3d as o3d
import torch
import trimesh
import tyro
import pyrender
from matplotlib import pyplot as plt
from PIL import Image
from scipy.spatial import cKDTree

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trimesh_to_open3d(tri_mesh):
    vertices = np.asarray(tri_mesh.vertices).astype(np.float32)
    faces = np.asarray(tri_mesh.faces).astype(np.int32)
    vertex_tensor = o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32)
    triangle_tensor = o3d.core.Tensor(faces, dtype=o3d.core.Dtype.Int32)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh())
    mesh.vertex.positions = vertex_tensor
    mesh.triangle.indices = triangle_tensor
    return mesh


def render_depth_maps(mesh, poses, H, W, K, far=10.0, debug=False):
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=0.01, zfar=far)
    camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(camera_node)
    renderer = pyrender.OffscreenRenderer(W, H)
    render_flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.DEPTH_ONLY

    depth_maps = []
    for i, pose in enumerate(tqdm(poses, desc="Rendering depth maps")):
        scene.set_pose(camera_node, pose)
        depth = renderer.render(scene, render_flags)

        if debug:
            global_max = np.max(depth)
            normalized_images = np.uint8((depth / global_max) * 255)
            colormapped_images = cv2.applyColorMap(normalized_images, cv2.COLORMAP_INFERNO)
            cv2.imwrite("depth_map_" + str(i) + ".png", colormapped_images)
        depth_maps.append(depth)

    return depth_maps


def cull_from_one_pose(
    points,
    pose,
    H,
    W,
    K,
    rendered_depth=None,
    depth_gt=None,
    remove_missing_depth=True,
    remove_occlusion=True,
):
    c2w = deepcopy(pose)
    # to OpenCV
    c2w[:3, 1] *= -1
    c2w[:3, 2] *= -1
    w2c = np.linalg.inv(c2w)
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]

    # pts under camera frame
    camera_space = rotation @ points.transpose() + translation[:, None]  # [3, N]
    uvz = (K @ camera_space).transpose()  # [N, 3]
    pz = uvz[:, 2] + 1e-8
    px = uvz[:, 0] / pz
    py = uvz[:, 1] / pz

    # step 1: inside frustum
    in_frustum = (0 <= px) & (px <= W - 1) & (0 <= py) & (py <= H - 1) & (pz > 0)
    u = np.clip(px, 0, W - 1).astype(np.int32)
    v = np.clip(py, 0, H - 1).astype(np.int32)
    eps = 0.02
    obs_mask = in_frustum
    # step 2: not occluded
    if remove_occlusion:
        obs_mask = in_frustum & (
            pz < (rendered_depth[v, u] + eps)
        )  # & (depth_gt[v, u] > 0.)

    # step 3: valid depth in gt
    if remove_missing_depth:
        invalid_mask = in_frustum & (depth_gt[v, u] <= 0.0)
    else:
        invalid_mask = np.zeros_like(obs_mask)

    return obs_mask.astype(np.int32), invalid_mask.astype(np.int32)


def get_grid_culling_pattern(
    points,
    poses,
    H,
    W,
    K,
    rendered_depth_list=None,
    depth_gt_list=None,
    remove_missing_depth=True,
    remove_occlusion=True,
    verbose=False,
):

    obs_mask = np.zeros(points.shape[0])
    invalid_mask = np.zeros(points.shape[0])
    for i, pose in enumerate(tqdm(poses, desc="Getting grid culling pattern")):
        rendered_depth = (
            rendered_depth_list[i] if rendered_depth_list is not None else None
        )
        depth_gt = depth_gt_list[i] if depth_gt_list is not None else None
        obs, invalid = cull_from_one_pose(
            points,
            pose,
            H,
            W,
            K,
            rendered_depth=rendered_depth,
            depth_gt=depth_gt,
            remove_missing_depth=remove_missing_depth,
            remove_occlusion=remove_occlusion,
        )
        obs_mask = obs_mask + obs
        invalid_mask = invalid_mask + invalid

    return obs_mask, invalid_mask


# For meshes with backward-facing faces. For some reason the no_culling flag in pyrender doesn't work for depth maps
def render_depth_maps_doublesided(mesh, poses, H, W, K, far=10.0):
    K = torch.tensor(K).cuda().float()
    depth_maps_1 = render_depth_maps(mesh, poses, H, W, K, far=far)
    mesh.faces[:, [1, 2]] = mesh.faces[:, [2, 1]]
    depth_maps_2 = render_depth_maps(mesh, poses, H, W, K, far=far)
    mesh.faces[:, [1, 2]] = mesh.faces[
        :, [2, 1]
    ]  # it's a pass by reference, so I restore the original order

    depth_maps = []
    for i in range(len(depth_maps_1)):
        depth_map = np.where(depth_maps_1[i] > 0, depth_maps_1[i], depth_maps_2[i])
        depth_map = np.where(
            (depth_maps_2[i] > 0) & (depth_maps_2[i] < depth_map),
            depth_maps_2[i],
            depth_map,
        )
        depth_maps.append(depth_map)

    return depth_maps


# borrowed from go-surf https://github.com/JingwenWang95/go-surf/blob/71bb12549abe86207b4f5bb799ac828014dcaad4/tools/frustum_culling.py#L194
def cull_mesh(
    dataset_path,
    dataset,
    mesh,
    transformation_files,
    remove_missing_depth=True,
    remove_occlusion=True,
    subdivide=True,
    max_edge=0.015,
):
    mesh.remove_unreferenced_vertices()
    vertices = mesh.vertices
    triangles = mesh.faces

    if subdivide:
        vertices, triangles = trimesh.remesh.subdivide_to_size(
            vertices, triangles, max_edge=max_edge, max_iter=10
        )

    print("Processed culling by bound")
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    # load dataset
    transformation_files = json.load(open(transformation_files, "r"))
    H, W = transformation_files["h"], transformation_files["w"]
    fl_x, fl_y = transformation_files["fl_x"], transformation_files["fl_y"]
    cx, cy = transformation_files["cx"], transformation_files["cy"]
    K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]]).astype(np.float32)

    frames = transformation_files["frames"]
    c2w_list = []
    depth_gt_list = []

    for i, frame in enumerate(frames):

        if dataset == "scannetpp":
            name = frame["file_path"].split("/")[-1].split(".")[0]
            depth_path = dataset_path / "depth" / (name + ".png")
            depth_gt = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            c2w = np.array(frame["transform_matrix"]).astype(np.float32)
            c2w = np.concatenate([c2w, np.array([[0, 0, 0, 1]])], axis=0)
            # c2w[0:3, 1:3] *= -1
            depth_gt = np.array(depth_gt) / 1000.0
            depth_gt = cv2.resize(depth_gt, (W, H), interpolation=cv2.INTER_NEAREST)
        elif dataset == "replica":
            depth_path = dataset_path / Path(frame["depth_file_path"])
            depth_gt = Image.open(depth_path)
            c2w = np.array(frame["transform_matrix"]).astype(np.float32)
            c2w[0:3, 1:3] *= -1
            depth_gt = (np.array(depth_gt) / 6553.5).astype(np.float32)
        c2w_list.append(c2w)
        depth_gt_list.append(depth_gt)

    rendered_depth_maps = render_depth_maps_doublesided(
        mesh, c2w_list, H, W, K, far=10.0
    )

    # we don't need subdivided mesh to render depth
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    mesh.remove_unreferenced_vertices()

    # Cull faces
    points = vertices[:, :3]
    obs_mask, invalid_mask = get_grid_culling_pattern(
        points,
        c2w_list,
        H,
        W,
        K,
        rendered_depth_list=rendered_depth_maps,
        depth_gt_list=depth_gt_list,
        remove_missing_depth=remove_missing_depth,
        remove_occlusion=remove_occlusion,
        verbose=True,
    )
    obs1 = obs_mask[triangles[:, 0]]
    obs2 = obs_mask[triangles[:, 1]]
    obs3 = obs_mask[triangles[:, 2]]
    th1 = 3
    obs_mask = (obs1 > th1) | (obs2 > th1) | (obs3 > th1)
    inv1 = invalid_mask[triangles[:, 0]]
    inv2 = invalid_mask[triangles[:, 1]]
    inv3 = invalid_mask[triangles[:, 2]]
    invalid_mask = (inv1 > 0.7 * obs1) & (inv2 > 0.7 * obs2) & (inv3 > 0.7 * obs3)
    valid_mask = obs_mask & (~invalid_mask)
    triangles_in_frustum = triangles[valid_mask, :]

    mesh = trimesh.Trimesh(vertices, triangles_in_frustum, process=False)
    mesh.remove_unreferenced_vertices()

    return mesh


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


transform = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ]
)


def main(
    gt_mesh_path: Path,  # path to gt mesh folder
    pred_mesh_path: Path,  # path to the pred mesh ply
    output: Path = Path("."),  # output path
    align: bool = False,
    dataset: str = "scannetpp",
    transformation_file: Optional[Path] = None,
    dataset_path: Optional[Path] = None,
    output_same_as_pred_mesh: Optional[bool] = True,
    rename_output_file: Optional[str] = None,
):
    gt_mesh = trimesh.load(str(gt_mesh_path), process=False)
    pred_mesh = trimesh.load(str(pred_mesh_path), process=False)

    if dataset == "scannetpp":
        gt_mesh = gt_mesh.apply_transform(transform)
        # pred_mesh = pred_mesh.apply_transform(transform)
    if align:
        transformation = get_align_transformation(
            str(pred_mesh_path), str(gt_mesh_path)
        )
        pred_mesh = pred_mesh.apply_transform(transformation)

    if output_same_as_pred_mesh:
        output = pred_mesh_path.parent

    gt_mesh = cull_mesh(
        dataset_path,
        dataset,
        gt_mesh,
        str(transformation_file),
        remove_missing_depth=True,
        remove_occlusion=True,
        subdivide=True,
        max_edge=0.015,
    )

    pred_mesh = cull_mesh(
        dataset_path,
        dataset,
        pred_mesh,
        transformation_file,
        remove_missing_depth=True,
        remove_occlusion=True,
        subdivide=True,
        max_edge=0.015,
    )

    print(str(pred_mesh_path.parent / pred_mesh_path.stem) + "_mesh_culled.ply")
    pred_mesh.export(str(output / "mesh_cull.ply"))
    # gt_mesh.export(str(output / "gt_mesh_cull.ply"))
    rst = compute_metrics(pred_mesh, gt_mesh)
    if rename_output_file is None:
        print(f"Saving results to: {output / 'mesh_metrics.json'}")
        json.dump(rst, open(output / "mesh_metrics.json", "w"))
    else:
        print(f"Saving results to: {output / Path(rename_output_file)}")
        json.dump(rst, open(output / Path(rename_output_file), "w"))
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
