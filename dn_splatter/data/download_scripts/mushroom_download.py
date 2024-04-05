"""Script to download example mushroom dataset to /datasets folder"""

import os
import subprocess
from pathlib import Path
from typing import Literal

import tyro

room_names = Literal[
    "coffee_room",
    "computer",
    "classroom",
    "honka",
    "koivu",
    "vr_room",
    "kokko",
    "sauna",
    "activity",
    "olohuone",
]


def download_mushroom(
    room_name: room_names = "activity",  # type: ignore
    save_dir: Path = Path(os.getcwd() + "/datasets"),
    sequence: Literal["iphone", "kinect", "faro", "all"] = "all",
):
    save_dir.mkdir(parents=True, exist_ok=True)

    iphone_url = (
        "https://zenodo.org/records/10230733/files/" + room_name + "_iphone.tar.gz"
    )
    kinect_url = (
        "https://zenodo.org/records/10209072/files/" + room_name + "_kinect.tar.gz"
    )
    mesh_pd_url = (
        "https://zenodo.org/records/10222321/files/" + room_name + "_mesh_pd.tar.gz"
    )

    commands = []
    extract_commands = []

    if sequence in ["iphone", "all"]:
        wget_command = ["wget", "-P", str(save_dir), iphone_url]
        commands.append(wget_command)
        file_name = room_name + "_iphone.tar.gz"
        extract_commands.append(["tar", "-xvzf", save_dir / file_name, "-C", save_dir])

    if sequence in ["kinect", "all"]:
        wget_command = ["wget", "-P", str(save_dir), kinect_url]
        commands.append(wget_command)
        file_name = room_name + "_kinect.tar.gz"
        extract_commands.append(["tar", "-xvzf", save_dir / file_name, "-C", save_dir])

    if sequence in ["faro", "all"]:
        wget_command = ["wget", "-P", str(save_dir), mesh_pd_url]
        commands.append(wget_command)
        file_name = room_name + "_mesh_pd.tar.gz"
        extract_commands.append(["tar", "-xvzf", save_dir / file_name, "-C", save_dir])

    for command in commands:
        try:
            subprocess.run(command, check=True)
            print("File file downloaded succesfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading file: {e}")
    for extract_command in extract_commands:
        try:
            subprocess.run(extract_command, check=True)
            print("Extraction complete.")
        except subprocess.CalledProcessError as e:
            print(f"Extraction failed: {e}")


if __name__ == "__main__":
    tyro.cli(download_mushroom)
