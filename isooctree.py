import argparse
import json
import os
import numpy as np
import IsoOctree
import open3d as o3d

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
        self.max_valid_depth_rel_delta = 0.02

    def image_path(self, type):
        ext = {"rgb": "png"}.get(type, "npy")
        if "MuSHRoom" in self.root_folder:
            return os.path.join(
                self.root_folder, type, f"long_frame_{self.image_id}.{ext}"
            )
        else:
            return os.path.join(self.root_folder, type, f"frame_{self.image_id}.{ext}")
        # return os.path.join(self.root_folder, type, f"frame_{self.image_id}.{ext}")

    def image_data_exists(self):
        return os.path.exists(self.image_path("depth")) and os.path.exists(
            self.image_path("normal")
        )

    def load_image(self, type, cache=False, stride=1):
        p = self.image_path(type)
        if type in self.cache:
            return self.cache[type]

        if p.endswith(".png"):
            raise RuntimeError("TODO: not implemented")
        elif p.endswith(".npy"):
            data = np.load(p)
        else:
            raise RuntimeError("Unknown image type")

        assert data.shape[0] == self.camera.resolution[1]
        assert data.shape[1] == self.camera.resolution[0]

        data = data[::stride, ::stride, ...]

        if type == "depth":
            data = data.astype(float)[..., 0] * self.depth_scale

        if cache:
            self.cache[type] = data

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

    def get_depth_values(self, points, cache=False, return_normals=False):
        pixel_coords, valid_mask = self.project(points)
        pixel_coords = pixel_coords.astype(int)

        valid_pixels = pixel_coords[valid_mask, :]
        depths = np.zeros((points.shape[0],))
        di = self.load_image("depth", cache=cache)
        depth_mask = compute_depth_validity_mask(di, self.max_valid_depth_rel_delta)

        if return_normals:
            normals = np.zeros((points.shape[0], 3))
            normal_image = self.load_image("normal", cache=cache)

        MIN_DEPTH = 1e-3

        if any(valid_mask):
            # TODO: fix this indexing!
            dd = depths[valid_mask]
            new_valid_mask = np.ones_like(dd, dtype=bool)
            if return_normals:
                nn = normals[valid_mask, :]

            for j in range(valid_pixels.shape[0]):
                # print(depths[valid_mask][j])
                # print(di[valid_pixels[j, 1], valid_pixels[j, 0]])
                dd[j] = di[valid_pixels[j, 1], valid_pixels[j, 0]]
                new_valid_mask[j] = depth_mask[
                    valid_pixels[j, 1], valid_pixels[j, 0]
                ] & (dd[j] > MIN_DEPTH)
                if return_normals:
                    nn[j, :] = normal_image[valid_pixels[j, 1], valid_pixels[j, 0], ...]

            depths[valid_mask] = dd
            if return_normals:
                normals[valid_mask, :] = nn

            valid_mask[valid_mask] = new_valid_mask

            # depths[valid_mask] = di[valid_pixels[:, 1], valid_pixels[:, 0]]
            # valid_mask &= depth_mask[valid_pixels[:, 1], valid_pixels[:, 0]]

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
            normals = normals @ self.pose_c2w[:3, :3].T

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
    root_dir, json_file_path, max_frames=None, frame_stride=1, pixel_stride=1
):
    with open(json_file_path, "r") as file:
        data = json.load(file)

    # root_dir = os.path.dirname(json_file_path)

    # assert data["camera_model"] == "OPENCV"

    camera = CameraModel(data)

    frames = []
    for frame_i, json_frame in enumerate(data["frames"]):
        if frame_i % frame_stride != 0:
            continue
        frame = Frame(camera, json_frame, root_dir)
        if not frame.image_data_exists():
            continue

        frames.append(frame)

        if max_frames is not None and len(frames) >= max_frames:
            break

    return frames


def generate_and_write_debug_point_cloud(path, frames, pixel_stride=1):
    all_points = []
    all_normals = []
    all_depths = []
    for frame in frames:
        points, normals, depths = frame.get_samples(stride=pixel_stride)[:3]
        all_points.append(points)
        all_normals.append(normals)
        all_depths.append(depths)

    all_points = np.vstack(all_points)
    all_normals = np.vstack(all_normals)
    all_depths = np.hstack(all_depths)
    if len(all_depths) > 0:
        all_depths = np.minimum(1, all_depths / max(1e-6, np.quantile(all_depths, 0.9)))

    colors = np.hstack([all_depths[:, None]] * 3)
    write_debug_ply(path, all_points, normals=all_normals, colors=colors)


def build_mesh_closest_surface_sample(
    frames,
    pixel_stride=1,
    max_depth=9,
    max_search_distance=0.3,
    k_nearest=10,
    subdivision_threshold=50,
    debug_ply_file=None,
):

    points = []
    normals = []

    for frame in frames:
        cur_points, cur_normals = frame.get_samples(stride=pixel_stride)[:2]
        points.append(cur_points)
        normals.append(cur_normals)

    points = np.vstack(points)
    normals = np.vstack(normals)

    if debug_ply_file is not None:
        write_debug_ply(debug_ply_file, points, normals=normals)

    from scipy.spatial import KDTree

    tree = KDTree(points)

    def isoValue(p0):
        # TODO: vectorize / make faster
        _, ii = tree.query(p0, k=k_nearest, distance_upper_bound=max_search_distance)
        ii = [i for i in ii if i < len(points)]

        if len(ii) == 0:
            return 1.0

        return np.sum(
            [np.dot(normals[ii[i], :], p0 - points[ii[i], :]) for i in range(len(ii))]
        )

    def isoValues(points):
        return [isoValue(points[i, :]) for i in range(points.shape[0])]

    return IsoOctree.buildMeshWithPointCloudHint(
        isoValues,
        points,
        maxDepth=max_depth,
        subdivisionThreshold=subdivision_threshold,
    )


def build_mesh_tsdf(
    frames,
    pixel_stride=1,
    max_depth=9,
    max_search_distance=0.3,
    k_nearest=10,
    subdivision_threshold=50,
    carve_min_rel=0.1,
    use_carving=True,
    tsdf_rel=0.01,
    debug_ply_file=None,
):

    all_samples = []
    refine_surface_points = []

    POS, NEG, CARVE = range(3)

    for frame in frames:
        points, normals, depths, all_rays = frame.get_samples(stride=pixel_stride)
        refine_surface_points.append(points)

        unirand = lambda: np.random.uniform(0, 1, size=(points.shape[0],))

        cam_pos = frame.position[None, :]

        if use_carving:
            carve_sample = (
                cam_pos
                + (
                    (unirand() * (1 - carve_min_rel - tsdf_rel) + carve_min_rel)
                    * depths
                )[:, None]
                * all_rays
            )
            all_samples.append((carve_sample, normals, points, CARVE))

        def sample_along_normal(sign, max_dist):
            rel = unirand()
            # value = rel * tsdf_max_value * sign
            sample_point = points + sign * normals * (max_dist * depths * rel)[:, None]
            return sample_point, normals, points, POS if sign > 0 else NEG

        all_samples.append(sample_along_normal(1, tsdf_rel))
        all_samples.append(sample_along_normal(-1, tsdf_rel))

    points = np.vstack([s[0] for s in all_samples])
    normals = np.vstack([s[1] for s in all_samples])
    surface_points = np.vstack([s[2] for s in all_samples])

    refine_surface_points = np.vstack(refine_surface_points)

    if debug_ply_file is not None:
        sample_types = np.hstack(
            [np.ones((s[0].shape[0],)) * s[3] for s in all_samples]
        )
        colors = np.zeros((points.shape[0], 3))
        colors[sample_types == CARVE, :] = 0.5  # gray
        colors[sample_types == POS, 1] = 1  # green
        colors[sample_types == NEG, 0] = 1  # red
        write_debug_ply(debug_ply_file, points, normals=normals, colors=colors)

    from scipy.spatial import KDTree

    tree = KDTree(points)

    def isoValue(p0):
        # TODO: vectorize / make faster
        _, ii = tree.query(p0, k=k_nearest, distance_upper_bound=max_search_distance)
        ii = [i for i in ii if i < len(points)]

        if len(ii) == 0:
            return 1.0

        cur_n = normals[ii, :]
        cur_diff = p0[None, :] - surface_points[ii, :]
        return np.max(np.minimum(np.sum(cur_n * cur_diff, axis=1), max_search_distance))

    def isoValues(points):
        return [isoValue(points[i, :]) for i in range(points.shape[0])]

    return IsoOctree.buildMeshWithPointCloudHint(
        isoValues,
        refine_surface_points,
        maxDepth=max_depth,
        subdivisionThreshold=subdivision_threshold,
    )


def build_mesh_projection(
    frames,
    subdivision_threshold,
    pixel_stride,
    max_depth,
    max_search_distance,
    max_tsdf_rel=0.05,
    uncertain_weight=0.1,
    uncertain_region=0.1,
    cache=False,
    debug_ply_file=None,
):

    # def uncertainty_func(x, uncertainty_rel=0.2, uncertainty_weight=0.01):
    #    y = x*1
    #    uncertain = np.abs(x) < uncertainty_rel
    #    y[uncertain] *= uncertainty_weight
    #    y[~uncertain] -= np.sign(y[~uncertain]) * uncertainty_rel
    #    return y

    def isoFunc(points):
        valid_mask = np.zeros((points.shape[0],), dtype=bool)
        valid_no_carve_mask = np.zeros((points.shape[0],), dtype=bool)
        values = np.zeros((points.shape[0],))
        weights = values * 0

        for frame in frames:
            projected_depth, point_depth, normals, valid = frame.get_depth_values(
                points, cache=cache, return_normals=True
            )
            max_tsdf = max_tsdf_rel * projected_depth[valid]
            valid_tsdf_values = (projected_depth[valid] - point_depth[valid]) / max_tsdf
            valid_no_carve = valid

            valid1 = valid_tsdf_values > -1
            valid_tsdf_values = valid_tsdf_values[valid1]
            valid[valid] = valid1
            valid_no_carve = valid_tsdf_values < 1
            valid_tsdf_values = np.minimum(valid_tsdf_values, 1)

            rays = points[valid] - frame.position[None, :]
            rays = rays / np.maximum(EPS, np.linalg.norm(rays, axis=1))[:, None]

            dir_weight = -np.sum(normals[valid, :] * rays, axis=1)
            valid_tsdf_values *= dir_weight

            w = dir_weight / np.maximum(max_search_distance, projected_depth[valid])
            # certainty = np.maximum(0, np.minimum(valid_tsdf_values / uncertain_region, 1))
            # print(certainty)
            # w *= (1 - certainty) * uncertain_weight + certainty
            values[valid] += valid_tsdf_values * w
            weights[valid] += w
            valid_mask |= valid
            valid_no_carve_mask[valid] |= valid_no_carve

        valid_mask = valid_no_carve_mask
        values[valid_mask] /= weights[valid_mask]
        values[~valid_mask] = 1

        return values

    point_cloud_hint = []
    for frame in frames:
        cur_points = frame.get_samples(stride=pixel_stride)[0]
        point_cloud_hint.append(cur_points)

    point_cloud_hint = np.vstack(point_cloud_hint)

    if debug_ply_file is not None:
        # import matplotlib.pyplot as plt
        # t = np.linspace(-2, 2, num=100)
        # plt.plot(t, uncertainty_func(t))
        # plt.show()

        write_debug_ply(debug_ply_file, point_cloud_hint)

    return IsoOctree.buildMeshWithPointCloudHint(
        isoFunc,
        point_cloud_hint,
        maxDepth=max_depth,
        subdivisionThreshold=subdivision_threshold,
    )


def writeMeshAsPly(mesh, filename):
    print("writing", filename)
    with open(filename, "wt") as f:
        # Write the PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(mesh.vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(mesh.triangles)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        # Write vertices
        for v in mesh.vertices:
            f.write("%f %f %f\n" % (v[0], v[1], v[2]))

        # Write faces
        for t in mesh.triangles:
            f.write("3 %d %d %d\n" % (t[0], t[1], t[2]))


def main():
    parser = argparse.ArgumentParser(description="Parse transformations.json file")
    parser.add_argument(
        "root_folder",
        type=str,
        help="Root folder path containing transformations.json",
    )
    parser.add_argument(
        "--transformation_path",
        type=str,
        help="Root folder path containing transformations.json",
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
        "--max_distance", type=float, default=0.3, help="Max distance to surface points"
    )
    parser.add_argument(
        "--subdivision_threshold",
        type=int,
        default=50,
        help="Subdivision threshold (surface samples per node)",
    )
    parser.add_argument("--k_nearest", type=int, default=10, help="K nearest samples")
    parser.add_argument(
        "--tsdf_rel", default=0.05, type=float, help="TSDF fusion tuning param"
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
        "--method",
        type=str,
        default="projection",
        choices=["tsdf", "closest_surface_sample", "projection"],
        help="Meshing method",
    )
    parser.add_argument(
        "--debug_ply",
        action="store_true",
        help="Write a PLY file with intermediary sample data for debug/visualization",
    )

    args = parser.parse_args()

    frames = load_frame_metadata(
        args.root_folder,
        args.transformation_path,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        pixel_stride=args.pixel_stride,
    )

    common_params = dict(
        pixel_stride=args.pixel_stride,
        max_depth=args.max_depth,
        max_search_distance=args.max_distance,
        subdivision_threshold=args.subdivision_threshold,
    )

    if args.debug_ply:
        common_params["debug_ply_file"] = os.path.join(args.root_folder, "debug.ply")

    if args.method == "closest_surface_sample":
        mesh = build_mesh_closest_surface_sample(
            frames, k_nearest=args.k_nearest, **common_params
        )
    elif args.method == "tsdf":
        mesh = build_mesh_tsdf(frames, k_nearest=args.k_nearest, **common_params)
    elif args.method == "projection":
        mesh = build_mesh_projection(
            frames, cache=args.cache, max_tsdf_rel=args.tsdf_rel, **common_params
        )
    else:
        assert False

    if len(args.output_mesh_file) == 0:
        out_path = os.path.join(args.root_folder, "mesh.ply")
    else:
        out_path = args.output_mesh_file

    writeMeshAsPly(mesh, out_path)


if __name__ == "__main__":
    main()
