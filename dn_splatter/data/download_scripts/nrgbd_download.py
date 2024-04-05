"""Script to download pre-processed Replica dataset. Total size of dataset is 12.4 gb."""
import os
import subprocess
from pathlib import Path

import tyro


def download_nrgbd(
    save_dir: Path = Path(os.getcwd() + "/datasets"),
    all: bool = True,
    test: bool = False,
):
    save_zip_dir = save_dir / "NRGBD"
    save_dir.mkdir(parents=True, exist_ok=True)

    url = "http://kaldir.vc.in.tum.de/neural_rgbd/neural_rgbd_data.zip"
    mesh_url = "http://kaldir.vc.in.tum.de/neural_rgbd/meshes.zip"
    commands = []
    commands.append(["wget", "-P", str(save_zip_dir), url])
    commands.append(["wget", "-P", str(save_zip_dir), mesh_url])
    file_name = "neural_rgbd_data.zip"
    mesh_name = "meshes.zip"
    extract_commands = []
    extract_commands.append(
        ["unzip", save_zip_dir / file_name, "-d", save_dir / "NRGBD"]
    )
    extract_commands.append(
        ["unzip", save_zip_dir / mesh_name, "-d", save_dir / "NRGBD"]
    )

    try:
        for command in commands:
            subprocess.run(command, check=True)
            print("File file downloaded succesfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e}")
    try:
        for e_command in extract_commands:
            subprocess.run(e_command, check=True)
            print("Extraction complete.")
    except subprocess.CalledProcessError as e:
        print(f"Extraction failed: {e}")


if __name__ == "__main__":
    tyro.cli(download_nrgbd)
