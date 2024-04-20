import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from nerfstudio.process_data import colmap_utils, hloc_utils

import tyro

"""
<base_dir>
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
"""


@dataclass
class ConvertColmap:
    """Convert images to COLMAP format"""

    image_path: Path
    """Input to images folder"""
    use_gpu: bool = True
    """Whether to use_gpu with colmap"""
    skip_matching: bool = False
    """Skip matching"""
    skip_undistortion: bool = True
    """Skip undistorting images"""
    camera: Literal["OPENCV"] = "OPENCV"
    """Camera type"""
    resize: bool = False
    """Resize images"""

    def main(self):
        image_path = str(self.image_path.resolve())
        use_gpu = 1 if self.use_gpu else 0
        colmap_command = "colmap"

        base_dir = str(Path(image_path).parent)

        if not self.skip_matching:
            os.makedirs(base_dir + "/colmap/sparse", exist_ok=True)
            ## Feature extraction
            feat_extracton_cmd = (
                colmap_command + " feature_extractor "
                "--database_path "
                + base_dir
                + "/colmap/database.db \
                --image_path "
                + image_path
                + " \
                --ImageReader.single_camera 1 \
                --ImageReader.camera_model "
                + self.camera
                + " \
                --SiftExtraction.use_gpu "
                + str(use_gpu)
            )
            exit_code = os.system(feat_extracton_cmd)
            if exit_code != 0:
                logging.error(
                    f"Feature extraction failed with code {exit_code}. Exiting."
                )
                exit(exit_code)

            ## Feature matching
            feat_matching_cmd = (
                colmap_command
                + " exhaustive_matcher \
                --database_path "
                + base_dir
                + "/colmap/database.db \
                --SiftMatching.use_gpu "
                + str(use_gpu)
            )
            exit_code = os.system(feat_matching_cmd)
            if exit_code != 0:
                logging.error(
                    f"Feature matching failed with code {exit_code}. Exiting."
                )
                exit(exit_code)

            ### Bundle adjustment
            # The default Mapper tolerance is unnecessarily large,
            # decreasing it speeds up bundle adjustment steps.
            mapper_cmd = (
                colmap_command
                + " mapper \
                --database_path "
                + base_dir
                + "/colmap/database.db \
                --image_path "
                + image_path
                + " \
                --output_path "
                + base_dir
                + "/colmap/sparse \
                --Mapper.ba_global_function_tolerance=0.000001"
            )
            exit_code = os.system(mapper_cmd)
            if exit_code != 0:
                logging.error(f"Mapper failed with code {exit_code}. Exiting.")
                exit(exit_code)

        ### Image undistortion
        if not self.skip_undistortion:
            img_undist_cmd = (
                colmap_command
                + " image_undistorter \
                --image_path "
                + image_path
                + " \
                --input_path "
                + base_dir
                + "/colmap/sparse/0 \
                --output_path "
                + base_dir
                + "\
                --output_type COLMAP"
            )
            exit_code = os.system(img_undist_cmd)
            if exit_code != 0:
                logging.error(f"Mapper failed with code {exit_code}. Exiting.")
                exit(exit_code)

        # TODO: move files to new destination folder
        # files = os.listdir(base_dir + "/sparse")
        # os.makedirs(base_dir + "/sparse/0", exist_ok=True)
        ## Copy each file from the source directory to the destination directory
        # for file in files:
        #    if file == "0":
        #        continue
        #    source_file = os.path.join(base_dir, "sparse", file)
        #    destination_file = os.path.join(base_dir, "sparse", "0", file)
        #    shutil.move(source_file, destination_file)

        if self.resize:
            raise NotImplementedError

        print("creating transforms.json and sparse_pc.ply")
        recon_dir = Path(base_dir, "colmap/sparse/0")
        output_dir = Path(base_dir)
        colmap_utils.colmap_to_json(recon_dir, output_dir)
        print("DONE, Congratulation")


if __name__ == "__main__":
    tyro.cli(ConvertColmap).main()
