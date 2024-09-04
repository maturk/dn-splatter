"""Convert poses from transforms.json to colmap sfm"""

import json
import os
from pathlib import Path

import numpy as np

from nerfstudio.data.utils.colmap_parsing_utils import rotmat2qvec


class PosesToColmap:
    def __init__(
        self,
        transforms_path: str = "datasets/room_datasets/vr_room/iphone/long_capture/transformations_colmap.json",
        run_colmap: bool = False,
        assume_colmap_world_coordinate_convention: bool = True,
    ):
        self.transforms_path = transforms_path
        self.run_colmap_cmd = run_colmap
        self.assume_colmap_world_coordinate_convention = (
            assume_colmap_world_coordinate_convention
        )
        self.camera = "OPENCV"
        self.image_path = "images"
        self.use_gpu = "False"

    def run_colmap(self):
        """
        colmap feature_extractor \
            --database_path $PROJECT_PATH/database.db \
            --image_path $PROJECT_PATH/images

        colmap exhaustive_matcher \ # or alternatively any other matcher
            --database_path $PROJECT_PATH/database.db

        colmap point_triangulator \
            --database_path $PROJECT_PATH/database.db \
            --image_path $PROJECT_PATH/images
            --input_path path/to/manually/created/sparse/model \
            --output_path path/to/triangulated/sparse/model
        
        """
        output_db = self.base_dir / "database.db"
        print("output db: ", output_db)
        use_gpu = 1 if self.use_gpu else 0

        feature_cmd = (
            "colmap",
            "feature_extractor",
            "--database_path",
            str(output_db),
            "--image_path",
            str(self.base_dir / self.image_path),
            "--ImageReader.single_camera",
            "0",
            "--ImageReader.camera_model",
            self.camera,
            "--SiftExtraction.use_gpu",
            str(use_gpu),
        )
        feature_cmd = (" ").join(feature_cmd)
        print(feature_cmd)
        os.system(feature_cmd)

        match_cmd = (
            "colmap",
            "exhaustive_matcher",
            "--database_path",
            str(output_db),
        )
        match_cmd = (" ").join(match_cmd)
        print(match_cmd)
        os.system(match_cmd)

        triangulate_cmd = (
            "colmap",
            "point_triangulator",
            "--database_path",
            str(output_db),
            "--image_path",
            str(self.base_dir / self.image_path),
            "--input_path",
            str(self.sparse_dir),
            "--output_path",
            str(self.sparse_dir),
        )
        triangulate_cmd = (" ").join(triangulate_cmd)
        print(triangulate_cmd)
        os.system(triangulate_cmd)

    def manual_sparse(self):
        print("Creating sparse model manually")
        with open(self.transforms_path) as f:
            data = json.load(f)

        self.base_dir = Path(self.transforms_path).parent.absolute()
        self.sparse_dir = self.base_dir / "sparse" / "0"
        self.sparse_dir.mkdir(parents=True, exist_ok=True)

        points_txt = self.sparse_dir / "points3D.txt"
        with open(str(points_txt), "w") as f:
            f.write("")
        images_txt = self.sparse_dir / "images.txt"
        cameras_txt = self.sparse_dir / "cameras.txt"

        camera_model = data["camera_model"]

        assert data["fl_x"]
        fx = data["fl_x"]
        fy = data["fl_y"]
        cx = data["cx"]
        cy = data["cy"]
        height = data["h"]
        width = data["w"]
        with open(str(cameras_txt), "w") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write("# Number of cameras: 1\n")
            f.write(f"1 {camera_model} {width} {height} {fx} {fy} {cx} {cy} 0 0 0 0\n")

        with open(str(images_txt), "w") as imgs_txt:
            for id, frame in enumerate(data["frames"]):
                c2w = np.array(frame["transform_matrix"])
                if self.assume_colmap_world_coordinate_convention:
                    c2w[2, :] *= -1
                    c2w = c2w[np.array([0, 2, 1, 3]), :]
                c2w[0:3, 1:3] *= -1
                w2c = np.linalg.inv(c2w)
                rotation = w2c[:3, :3]
                translation = w2c[:3, 3]
                qvec = rotmat2qvec(rotation)
                name = Path(frame["file_path"]).name
                camera_id = 1
                image_id = id + 1

                qvec_str = ", ".join(map(str, qvec.tolist()))
                translation_str = ", ".join(map(str, translation.tolist()))
                imgs_txt.write(
                    f"{image_id} {qvec_str} {translation_str} {camera_id} {name}"
                )
                imgs_txt.write("\n")
                imgs_txt.write("\n")

    def main(self):
        self.manual_sparse()
        if self.run_colmap_cmd:
            self.run_colmap()


if __name__ == "__main__":
    import tyro

    tyro.cli(PosesToColmap).main()
