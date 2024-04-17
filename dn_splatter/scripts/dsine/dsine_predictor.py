import os
import torch
import numpy as np
from typing import Literal, Optional
from torchvision import transforms
from jaxtyping import UInt8, Float
import torch.nn.functional as F

from dn_splatter.scripts.dsine.dsine import DSINE


def pad_input(original_height: int, original_width: int):
    if original_width % 32 == 0:
        left = 0
        right = 0
    else:
        new_width = 32 * ((original_width // 32) + 1)
        left = (new_width - original_width) // 2
        right = (new_width - original_width) - left

    if original_height % 32 == 0:
        top = 0
        bottom = 0
    else:
        new_height = 32 * ((original_height // 32) + 1)
        top = (new_height - original_height) // 2
        bottom = (new_height - original_height) - top
    return left, right, top, bottom


def get_intrins_from_fov(new_fov, height, width, device):
    # NOTE: top-left pixel should be (0,0)
    if width >= height:
        new_fu = (width / 2.0) / np.tan(np.deg2rad(new_fov / 2.0))
        new_fv = (width / 2.0) / np.tan(np.deg2rad(new_fov / 2.0))
    else:
        new_fu = (height / 2.0) / np.tan(np.deg2rad(new_fov / 2.0))
        new_fv = (height / 2.0) / np.tan(np.deg2rad(new_fov / 2.0))

    new_cu = (width / 2.0) - 0.5
    new_cv = (height / 2.0) - 0.5

    new_intrins = torch.tensor(
        [[new_fu, 0, new_cu], [0, new_fv, new_cv], [0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )

    return new_intrins


def _load_state_dict(local_file_path: Optional[str] = None):
    if local_file_path is not None and os.path.exists(local_file_path):
        # Load state_dict from local file
        state_dict = torch.load(local_file_path, map_location=torch.device("cpu"))
    else:
        # Load state_dict from the default URL
        file_name = "dsine.pt"
        url = f"https://huggingface.co/camenduru/DSINE/resolve/main/dsine.pt"
        state_dict = torch.hub.load_state_dict_from_url(
            url, file_name=file_name, map_location=torch.device("cpu")
        )

    return state_dict["model"]


class DSinePredictor:
    def __init__(self, device: Literal["cpu", "cuda"]):
        self.device = device
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.model = self.load_model()

    def load_model(self):
        state_dict = _load_state_dict(None)
        model = DSINE()
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model = model.to(self.device)
        model.pixel_coords = model.pixel_coords.to(self.device)

        return model

    def __call__(
        self,
        rgb: UInt8[np.ndarray, "h w 3"],
        K_33: Optional[Float[np.ndarray, "3 3"]] = None,
    ) -> Float[torch.Tensor, "b 3 h w"]:
        rgb = rgb.astype(np.float32) / 255.0
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        _, _, h, w = rgb.shape

        # zero-pad the input image so that both the width and height are multiples of 32
        left, right, top, bottom = pad_input(h, w)
        rgb = F.pad(rgb, (left, right, top, bottom), mode="constant", value=0.0)
        rgb = self.transform(rgb)

        if K_33 is None:
            K_b33: Float[torch.Tensor, "b 3 3"] = get_intrins_from_fov(
                new_fov=60.0, height=h, width=w, device=self.device
            ).unsqueeze(0)
        else:
            K_b33 = torch.from_numpy(K_33).unsqueeze(0).to(self.device)

        # add padding to intrinsics
        K_b33[:, 0, 2] += left
        K_b33[:, 1, 2] += top

        with torch.no_grad():
            normal_b3hw: Float[torch.Tensor, "b 3 h-t w-l"] = self.model(
                rgb, intrins=K_b33
            )[-1]
            normal_b3hw: Float[torch.Tensor, "b 3 h w"] = normal_b3hw[
                :, :, top : top + h, left : left + w
            ]

        return normal_b3hw
