import argparse
import json
import os
import numpy as np
import IsoOctree

EPS = 1e-6


def to_homog(p):
    return np.hstack([p, np.ones((p.shape[0], 1))])


CAM_CONVENTION_CHANGE = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
)


def compute_depth_validity_mask(depth_img, max_valid_depth_rel_delta):
    depth_dx = np.diff(depth_img, axis=1)
    depth_dy = np.diff(depth_img, axis=0)
    depth_dx_ok = (
        np.abs(depth_dx)
        < np.minimum(depth_img[:, 1:], depth_img[:, :-1]) * max_valid_depth_rel_delta
    )
    depth_dy_ok = (
        np.abs(depth_dy)
        < np.minimum(depth_img[1:, :], depth_img[:-1, :]) * max_valid_depth_rel_delta
    )

    valid = np.ones_like(depth_img, dtype=bool)
    valid[:, 1:] &= depth_dx_ok
    valid[:, :-1] &= depth_dx_ok
    valid[1:, :] &= depth_dy_ok
    valid[:-1, :] &= depth_dy_ok

    MARGIN = 0
    if MARGIN > 0:
        valid[:MARGIN, :] = False
        valid[-MARGIN:] = False
        valid[:, :MARGIN] = False
        valid[:, -MARGIN:] = False

    return valid


class CameraModel:
    def __init__(self, data):
        self.resolution = data["w"], data["h"]

        self.camera_matrix = np.array(
            [[data["fl_x"], 0, data["cx"]], [0, data["fl_y"], data["cy"]], [0, 0, 1]]
        )

        self.inverse_camera_matrix = np.linalg.inv(self.camera_matrix)

    def unproject(self, pixel_coordinates, normalize=False):
        for i in range(2):
            p = pixel_coordinates[:, i].astype(int)
            assert np.all(p >= 0) and np.all(p < self.resolution[i])

        homog = to_homog(pixel_coordinates) @ self.inverse_camera_matrix.T
        if normalize:
            return homog / np.linalg.norm(homog, axis=1)[:, None]

        return homog

    def project(self, points):
        projections = np.zeros((points.shape[0], 2))
        valid_mask = points[:, 2] > EPS
        projections_homog = points[valid_mask, :] @ self.camera_matrix.T
        projections[valid_mask] = (
            projections_homog[:, :2] / projections_homog[:, -1][:, None]
        )
        for i in range(2):
            proj_i = projections[:, i].astype(int)
            valid_mask = valid_mask & (proj_i >= 0) & (proj_i < self.resolution[i])

        return projections, valid_mask

    def all_pixels(self, pixel_origin_center=True, stride=1):
        x = np.arange(0, self.resolution[0])[::stride].astype(float)
        y = np.arange(0, self.resolution[1])[::stride].astype(float)
        xv, yv = np.meshgrid(x, y)
        if pixel_origin_center:
            xv += 0.5
            yv += 0.5
        return np.hstack([np.ravel(xv)[:, None], np.ravel(yv)[:, None]])


class Frame:
    def __init__(self, camera_model, data, root_folder):
        self.camera = camera_model

        self.pose_c2w = np.array(data["transform_matrix"])
        if self.pose_c2w.shape[0] == 3:
            self.pose_c2w = np.vstack([self.pose_c2w, [0, 0, 0, 1]])
        self.pose_c2w = self.pose_c2w @ CAM_CONVENTION_CHANGE
        self.pose_w2c = np.linalg.inv(self.pose_c2w)

        self.root_folder = root_folder
        self.image_id = data["file_path"].split("/")[-1].split("_")[1].split(".")[0]
        self.cache = {}

        self.depth_is_z = True
        self.cam_coordinate_normals = False
        self.auto_flip_normals = False

        self.depth_scale = 1 / 1000.0
        self.max_valid_depth_rel_delta = 0.005  # 0.005

        self.color_normal_data = False

    def image_path(self, type):
        if self.color_normal_data:
            if type == "normal":
                return os.path.join(
                    self.root_folder, type, f"frame_{self.image_id}.png"
                )
            else:
                return os.path.join(
                    self.root_folder,
                    type,
                    "raw",
                    f"frame_{self.image_id}.npy",
                )
        else:
            ext = {"rgb": "png"}.get(type, "npy")
            return os.path.join(self.root_folder, type, f"frame_{self.image_id}.{ext}")

    def image_data_exists(self):
        return os.path.exists(self.image_path("depth")) and os.path.exists(
            self.image_path("normal")
        )

    def load_image(self, type, cache=False, stride=1):
        p = self.image_path(type)
        if type in self.cache:
            data, prev_stride = self.cache[type]
            if stride == prev_stride:
                return data

        if p.endswith(".png"):
            from PIL import Image

            data = np.array(Image.open(p))
        elif p.endswith(".npy"):
            data = np.load(p)
        else:
            raise RuntimeError("Unknown image type")

        assert data.shape[0] == self.camera.resolution[1]
        assert data.shape[1] == self.camera.resolution[0]

        data = data[::stride, ::stride, ...]

        if type == "normal" and self.color_normal_data:
            data = data.astype(float) / 255.0 * 2 - 1

        if type == "depth":
            data = data.astype(float)[..., 0] * self.depth_scale

        if cache:
            self.cache[type] = (data, stride)

        return data

    @property
    def position(self):
        return self.pose_c2w[:3, 3]

    def unproject_to_rays(self, pixel_coordinates, **kwargs):
        return (
            self.camera.unproject(pixel_coordinates, **kwargs) @ self.pose_c2w[:3, :3].T
        )

    def world_to_camera(self, points):
        return (to_homog(points) @ self.pose_w2c.T)[:, :3]

    def project(self, points):
        return self.camera.project(self.world_to_camera(points))

    def get_depth_values(
        self, points, cache=False, return_normals=False, interpolate=True
    ):
        pixel_coords, valid_mask = self.project(points)

        valid_pixels = pixel_coords[valid_mask, :]
        depths = np.zeros((points.shape[0],))
        di = self.load_image("depth", cache=cache)
        depth_mask = compute_depth_validity_mask(di, self.max_valid_depth_rel_delta)

        if return_normals:
            normals = np.zeros((points.shape[0], 3))
            normal_image = self.load_image("normal", cache=cache)

        MIN_DEPTH = 1e-3

        if any(valid_mask):
            ix = valid_pixels[:, 0].astype(int)
            iy = valid_pixels[:, 1].astype(int)
            if interpolate:
                tx = valid_pixels[:, 0] - ix
                ty = valid_pixels[:, 1] - iy
                ix1 = np.minimum(ix + 1, di.shape[1] - 1)
                iy1 = np.minimum(iy + 1, di.shape[0] - 1)
                dd = (
                    di[iy, ix] * (1 - tx) * (1 - ty)
                    + di[iy, ix1] * tx * (1 - ty)
                    + di[iy1, ix] * (1 - tx) * ty
                    + di[iy1, ix1] * tx * ty
                )
            else:
                dd = di[iy, ix]

            new_valid_mask = depth_mask[iy, ix] & (dd > MIN_DEPTH)

            if return_normals:
                normals[valid_mask, :] = normal_image[iy, ix, ...]

            depths[valid_mask] = dd
            valid_mask[valid_mask] = new_valid_mask

        computed_depths = self.world_to_camera(points)[:, 2]
        if return_normals:
            rays = points - self.position[None, :]
            valid_mask &= np.sum(normals * rays, axis=1) < 0
            return depths, computed_depths, normals, valid_mask
        return depths, computed_depths, valid_mask

    def get_samples(self, cache=False, stride=1):
        all_pixels = self.camera.all_pixels(stride=stride)
        all_rays = self.unproject_to_rays(all_pixels, normalize=not self.depth_is_z)
        depth_img = self.load_image("depth", cache=cache, stride=stride)
        depth_img = np.where(depth_img <= 4, depth_img, 0)
        depth_mask_img = compute_depth_validity_mask(
            depth_img, self.max_valid_depth_rel_delta * stride
        )

        depths = depth_img.ravel()
        depth_mask = depth_mask_img.ravel()

        normals = (
            self.load_image("normal", cache=cache, stride=stride)
            .reshape(-1, 3)
            .astype(float)
        )

        if self.cam_coordinate_normals:
            R = self.pose_c2w[:3, :3] @ CAM_CONVENTION_CHANGE[:3, :3]
            normals = normals @ R.T
            normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        if self.auto_flip_normals:
            signs = np.sum(normals * all_rays, axis=1)
            normals[signs > 0, :] *= -1

        signs = np.sum(normals * all_rays, axis=1)
        valid_mask = (signs < 0) & depth_mask

        all_positions = self.position[None, :] + all_rays * depths[:, None]

        # normalized_rays = all_rays / np.linalg.norm(all_rays, axis=1)[:, None]
        return (
            all_positions[valid_mask, :],
            normals[valid_mask, :],
            depths[valid_mask],
            all_rays[valid_mask, :],
        )


def write_debug_ply(ply_file_path, points, normals=None, colors=None):
    import open3d as o3d

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        point_cloud.normals = o3d.utility.Vector3dVector(normals)
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Write the point cloud to a PLY file
    o3d.io.write_point_cloud(ply_file_path, point_cloud)


def writeMeshAsObj(mesh, filename):
    print("writing", filename)
    with open(filename, "wt") as f:
        for v in mesh.vertices:
            f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
        for t in mesh.triangles:
            f.write("f %d %d %d\n" % (t[0] + 1, t[1] + 1, t[2] + 1))


def load_frame_metadata(
    root_dir,
    json_file_path,
    max_frames=None,
    frame_stride=1,
    camera_coordinate_normals=False,
):
    with open(json_file_path, "r") as file:
        data = json.load(file)

    # assert data["camera_model"] == "OPENCV"

    camera = CameraModel(data)

    frames = []
    for frame_i, json_frame in enumerate(data["frames"]):
        if frame_i % frame_stride != 0:
            continue

        frame = Frame(camera, json_frame, root_dir)
        frame.cam_coordinate_normals = camera_coordinate_normals
        if camera_coordinate_normals:
            # note: a bit hacky to couple these two
            frame.color_normal_data = True
        if not frame.image_data_exists():
            continue

        frames.append(frame)

        if max_frames is not None and len(frames) >= max_frames:
            break

    return frames


def build_mesh_projection(
    frames,
    subdivision_threshold,
    pixel_stride,
    max_depth,
    max_tsdf_rel=0.05,
    max_angle_to_max_weight_normal_deg=60,
    cache=False,
    max_tsdf_abs=None,
    choose_best_frame=False,
    two_pass=True,
    debug_ply_file=None,
    use_normals=True,
):
    min_dot_max_weight_normal = np.cos(max_angle_to_max_weight_normal_deg / 180 * np.pi)
    BACK_MASK_COEFF = 0.25  # relative to depth

    if not use_normals:
        choose_best_frame = False
        two_pass = False

    def isoFunc(points):
        max_weight_normals = np.zeros((points.shape[0], 3))

        if choose_best_frame:
            # alternative 1: choose the "best" frame for each point and use that as the only depth source
            passes = [True]
        else:
            if two_pass:
                # alternative 2: first find the "best" frame and use its normal as a reference:
                # then ignore (negative-ish) TSDF values that are not consistent with the normal
                passes = [True, False]
            else:
                # alternative 3: more typical TSDF fusion without the reference normal thing
                passes = [False]

        for normal_pass in passes:
            valid_mask = np.zeros((points.shape[0],), dtype=bool)
            back_mask = (
                valid_mask & False
            )  # a vector full of False, same shape as valid_mask
            values = np.zeros((points.shape[0],))
            weights = values * 0

            for frame in frames:
                r = frame.get_depth_values(
                    points, cache=cache, return_normals=use_normals
                )
                if use_normals:
                    projected_depth, point_depth, normals, valid = r
                else:
                    projected_depth, point_depth, valid = r

                max_tsdf = max_tsdf_rel * projected_depth[valid]
                if max_tsdf_abs is not None:
                    max_tsdf = np.minimum(max_tsdf, max_tsdf_abs)
                valid_tsdf_values = (
                    projected_depth[valid] - point_depth[valid]
                ) / max_tsdf

                back_mask[valid] |= (
                    valid_tsdf_values * max_tsdf_rel > -BACK_MASK_COEFF
                ) & (valid_tsdf_values < 0)
                valid1 = valid_tsdf_values > -1

                valid_tsdf_values = valid_tsdf_values[valid1]
                valid[valid] = valid1
                valid_tsdf_values = np.minimum(valid_tsdf_values, 1)

                rays = points[valid] - frame.position[None, :]
                rays = rays / np.maximum(EPS, np.linalg.norm(rays, axis=1))[:, None]

                if normal_pass or not use_normals:
                    dir_weight = 1
                else:
                    dir_weight = -np.sum(normals[valid, :] * rays, axis=1)

                valid_tsdf_values *= dir_weight
                w = dir_weight / np.maximum(EPS, projected_depth[valid])

                if normal_pass:
                    w *= np.maximum(0, np.minimum(valid_tsdf_values + 0.5, 1))
                    valid2 = w > weights[valid]
                    valid[valid] = valid2
                    weights[valid] = w[valid2]
                    max_weight_normals[valid, :] = normals[valid, :]

                    valid_tsdf_values = valid_tsdf_values[valid2]
                    values[valid] = valid_tsdf_values
                else:
                    if use_normals:
                        has_max_normal = (
                            np.sum(max_weight_normals[valid, :] ** 2, axis=1) > 0
                        )
                        md = min_dot_max_weight_normal
                        max_normal_weight = np.maximum(
                            np.sum(
                                max_weight_normals[valid, :] * normals[valid, :], axis=1
                            )
                            - md,
                            0,
                        ) / (1 - md)

                        neg_weight = 1 - np.maximum(
                            0, np.minimum(valid_tsdf_values[has_max_normal] + 0.5, 1)
                        )
                        w[has_max_normal] *= max_normal_weight[
                            has_max_normal
                        ] * neg_weight + (1 - neg_weight)

                    values[valid] += valid_tsdf_values * w
                    weights[valid] += w

                valid_mask |= valid

            if normal_pass:
                valid_mask &= np.sum(max_weight_normals**2, axis=1) > 0
                max_weight_normals[valid_mask, :] /= np.linalg.norm(
                    max_weight_normals[valid_mask, :], axis=1
                )[:, None]

        valid_mask &= weights > 0
        values[valid_mask] /= weights[valid_mask]
        values[~valid_mask] = 1
        # avoid holes in the reconstruction, caused by large negative TSDF
        # values begin ignored in large voxels
        values[~valid_mask & back_mask] = -1

        return values

    point_cloud_hint = []
    point_cloud_hint_normals_debug = []
    for frame in frames:
        cur_points, cur_normals = frame.get_samples(stride=pixel_stride)[:2]
        point_cloud_hint.append(cur_points)
        if debug_ply_file:
            point_cloud_hint_normals_debug.append(cur_normals)

    point_cloud_hint = np.vstack(point_cloud_hint)

    if debug_ply_file is not None:
        point_cloud_hint_normals_debug = np.vstack(point_cloud_hint_normals_debug)
        write_debug_ply(
            debug_ply_file, point_cloud_hint, normals=point_cloud_hint_normals_debug
        )

    return IsoOctree.buildMeshWithPointCloudHint(
        isoFunc,
        point_cloud_hint,
        maxDepth=max_depth,
        subdivisionThreshold=subdivision_threshold,
    )


def main():
    parser = argparse.ArgumentParser(description="Parse transformations.json file")
    parser.add_argument(
        "root_folder",
        type=str,
        help="path to the render folder containing depth and normal images",
    )
    parser.add_argument(
        "--transformation_path",
        type=str,
        help="Root folder path containing transformations.json",
    )
    parser.add_argument(
        "-cam",
        "--camera_coordinate_normals",
        action="store_true",
        help="Whether normals are in camera coordinates",
    )
    parser.add_argument(
        "--max_frames", type=int, default=None, help="Maximum number of frames to parse"
    )
    parser.add_argument(
        "--frame_stride", type=int, default=1, help="Use every Nth frame"
    )
    parser.add_argument(
        "--pixel_stride", type=int, default=6, help="Use every Nth pixel row & col"
    )
    parser.add_argument(
        "--max_depth", type=int, default=10, help="Max meshing octree depth"
    )
    parser.add_argument(
        "--subdivision_threshold",
        type=int,
        default=50,
        help="Subdivision threshold (surface samples per node)",
    )
    parser.add_argument(
        "--tsdf_rel", default=0.05, type=float, help="TSDF fusion tuning param"
    )
    parser.add_argument(
        "--tsdf_abs", type=float, default=None, help="Max TSDF distance (absolute)"
    )
    parser.add_argument(
        "--disable_normals",
        action="store_true",
        help="do not use normals in TSDF fusion",
    )
    parser.add_argument(
        "--cache", action="store_true", help="cache images to mem in projection mode"
    )
    parser.add_argument(
        "-o",
        "--output_mesh_file",
        default="",
        type=str,
        help="defaults to input_folder/mesh.obj",
    )
    parser.add_argument(
        "--debug_ply_file",
        type=str,
        default=None,
        help="Path to an output PLY file with intermediary sample data for debug/visualization",
    )

    args = parser.parse_args()

    if args.transformation_path is None:
        args.transformation_path = os.path.join(
            args.root_folder, "transformations_colmap.json"
        )

    frames = load_frame_metadata(
        args.root_folder,
        args.transformation_path,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        camera_coordinate_normals=args.camera_coordinate_normals,
    )

    mesh = build_mesh_projection(
        frames,
        cache=args.cache,
        max_tsdf_rel=args.tsdf_rel,
        pixel_stride=args.pixel_stride,
        max_depth=args.max_depth,
        max_tsdf_abs=args.tsdf_abs,
        subdivision_threshold=args.subdivision_threshold,
        use_normals=not args.disable_normals,
        debug_ply_file=args.debug_ply_file,
    )

    if len(args.output_mesh_file) == 0:
        out_path = os.path.join(args.root_folder, "mesh.obj")
    else:
        out_path = args.output_mesh_file

    writeMeshAsObj(mesh, out_path)


if __name__ == "__main__":
    main()
