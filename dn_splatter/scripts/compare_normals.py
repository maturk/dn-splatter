import rerun as rr
import numpy as np
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser


def main(first_normal_dir: Path, second_normal_dir: Path):
    assert first_normal_dir.is_dir(), f"{first_normal_dir} is not a directory"
    assert second_normal_dir.is_dir(), f"{second_normal_dir} is not a directory"

    first_normal_glob = sorted(first_normal_dir.glob("*.png"))
    second_normal_glob = sorted(second_normal_dir.glob("*.png"))
    assert len(first_normal_glob) > 0, f"No normal images found in {first_normal_dir}"
    assert len(second_normal_glob) > 0, f"No normal images found in {second_normal_dir}"
    for idx, (first_path, second_path) in enumerate(
        zip(first_normal_glob, second_normal_glob)
    ):
        rr.set_time_sequence("idx", idx)
        first_normal = np.array(Image.open(first_path), dtype="uint8")[..., :3]
        second_normal = np.array(Image.open(second_path), dtype="uint8")[..., :3]
        rr.log("first_normal", rr.Image(first_normal))
        rr.log("second_normal", rr.Image(second_normal))


if __name__ == "__main__":
    parser = ArgumentParser("Surface Normal Comparison")
    parser.add_argument(
        "first_normal_dir",
        type=Path,
        help="Path to the first directory containing surface normal predictions",
    )
    parser.add_argument(
        "second_normal_dir",
        type=Path,
        help="Path to the second directory containing surface normal predictions",
    )
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, "compare_normals")
    main(
        first_normal_dir=args.first_normal_dir, second_normal_dir=args.second_normal_dir
    )
    rr.script_teardown(args)
