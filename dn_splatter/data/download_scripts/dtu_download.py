import os
import subprocess
from pathlib import Path

import tyro


def download_dtu(
    save_dir: Path = Path(os.getcwd() + "/datasets"),
):
    save_zip_dir = save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    url = "https://s3.eu-central-1.amazonaws.com/avg-projects/monosdf/data/DTU.tar"
    wget_command = ["wget", "-P", str(save_zip_dir), url]
    file_name = "DTU.tar"
    extract_command = ["tar", "-xvf", save_dir / file_name, "-C", save_dir]

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
    tyro.cli(download_dtu)
