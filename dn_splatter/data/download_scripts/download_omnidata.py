import os
import subprocess
from pathlib import Path

import tyro


def download_omnidata(
    save_dir: Path = Path(os.getcwd() + "/omnidata_ckpt"),
):
    save_dir.mkdir(parents=True, exist_ok=True)
    # Download the pretrained model weights using wget
    try:
        subprocess.run(
            [
                "wget",
                "-P",
                save_dir,
                "https://zenodo.org/records/10447888/files/omnidata_dpt_normal_v2.ckpt",
            ]
        )
        print("Pretrained model weights downloaded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e}")


if __name__ == "__main__":
    tyro.cli(download_omnidata)
