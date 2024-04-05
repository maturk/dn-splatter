"""Script to download pre-processed Replica dataset. Total size of dataset is 12.4 gb."""
import os
import subprocess
from pathlib import Path

import tyro


def download_replica(
    save_dir: Path = Path(os.getcwd() + "/datasets"),
):
    save_zip_dir = save_dir / "Replica"
    save_dir.mkdir(parents=True, exist_ok=True)

    url = "https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip"
    wget_command = ["wget", "-P", str(save_zip_dir), url]
    file_name = "Replica.zip"
    extract_command = ["unzip", save_zip_dir / file_name, "-d", save_dir]

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
    tyro.cli(download_replica)
