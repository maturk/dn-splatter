"""Load DNSplatter model and render all outputs to disk"""

from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from dn_splatter.utils.utils import save_outputs_helper

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.utils import colormaps
from nerfstudio.utils.eval_utils import eval_setup


@dataclass
class RenderModel:
    """Render outputs of a GS model."""

    load_config: Path = Path("")
    """Path to the config YAML file."""
    output_dir: Path = Path("./renders/")
    """Path to the output directory."""
    render_rgb: bool = False
    render_depth: bool = False
    render_normal: bool = True

    def main(self):
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        assert isinstance(pipeline.model, SplatfactoModel)

        model: SplatfactoModel = pipeline.model
        train_dataset: InputDataset = pipeline.datamanager.train_dataset

        with torch.no_grad():
            cameras: Cameras = pipeline.datamanager.train_dataset.cameras  # type: ignore
            for image_idx, data in enumerate(
                pipeline.datamanager.cached_train  # Undistorted images
            ):  # type: ignore

                # process batch gt data
                mask = None
                if "mask" in data:
                    mask = data["mask"]

                gt_img = 256 - data["image"]  # TODO: not sure why negative
                if "sensor_depth" in data:
                    depth_gt = data["sensor_depth"]
                    depth_gt_color = colormaps.apply_depth_colormap(
                        data["sensor_depth"]
                    )
                else:
                    depth_gt = None
                    depth_gt_color = None
                if "normal" in data:
                    normal_gt = data["normal"]

                # process pred outputs
                camera = cameras[image_idx : image_idx + 1].to("cpu")
                outputs = model.get_outputs_for_camera(camera=camera)

                rgb_out, depth_out = outputs["rgb"], outputs["depth"]

                normal = None
                if "normal" in outputs:
                    normal = outputs["normal"]

                seq_name = Path(train_dataset.image_filenames[image_idx])
                image_name = f"{seq_name.stem}"

                depth_color = colormaps.apply_depth_colormap(depth_out)
                depth = depth_out.detach().cpu().numpy()

                if mask is not None:
                    rgb_out = rgb_out * mask
                    gt_img = gt_img * mask
                    if depth_color is not None:
                        depth_color = depth_color * mask
                    if depth_gt_color is not None:
                        depth_gt_color = depth_gt_color * mask
                    if depth_gt is not None:
                        depth_gt = depth_gt * mask
                    if depth is not None:
                        depth = depth * mask
                    if normal_gt is not None:
                        normal_gt = normal_gt * mask
                    if normal is not None:
                        normal = normal * mask

                # save all outputs
                save_outputs_helper(
                    self.output_dir,
                    rgb_out if self.render_rgb else None,
                    gt_img if self.render_rgb else None,
                    depth_color if self.render_depth else None,
                    depth_gt_color if self.render_depth else None,
                    depth_gt if self.render_depth else None,
                    depth if self.render_depth else None,
                    normal_gt if self.render_normal else None,
                    normal if self.render_normal else None,
                    image_name,
                )


if __name__ == "__main__":
    tyro.cli(RenderModel).main()
