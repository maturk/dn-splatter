import os
import subprocess
from pathlib import Path

import tyro


def download_reference_depth(
    save_dir: Path = Path(os.getcwd() + "/datasets"),
):
    save_dir.mkdir(parents=True, exist_ok=True)

    depth_url = "https://zenodo.org/records/10438963/files/reference_depth.tar.gz"

    wget_command = ["wget", "-P", str(save_dir), depth_url]
    file_name = "reference_depth.tar.gz"
    extract_command = ["tar", "-xvzf", save_dir / file_name, "-C", save_dir]

    try:
        subprocess.run(wget_command, check=True)
        print("File file downloaded succesfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e}")
    try:
        subprocess.run(extract_command, check=True)
        print("Extraction complete.")
    except subprocess.CalledProcessError as e:
        print(f"Extraction failed: {e}")


if __name__ == "__main__":
    tyro.cli(download_reference_depth)
