from pathlib import Path

import numpy as np
import tyro
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from dn_splatter.metrics import NormalMetrics
from dn_splatter.utils.utils import image_path_to_tensor


def tensor_to_normals(img_tensor: Tensor) -> Tensor:
    """
    Convert image tensor [0, 1] to normal vectors [-1, 1].
    """
    normals = img_tensor * 2 - 1  # Normalize to [-1, 1]
    return normals


class NormalsDataset(Dataset):
    def __init__(
        self, gt_folder_path: Path, prediction_folder_path: Path, transform=None
    ):
        self.gt_rgb_norm_fnames = sorted(
            [f for f in gt_folder_path.rglob("*") if f.is_file() and f.suffix == ".png"]
        )
        self.pred_rgb_norm_fnames = sorted(
            [
                f
                for f in prediction_folder_path.rglob("*")
                if f.is_file() and f.suffix == ".png"
            ]
        )
        self.transform = transform

    def __len__(self):
        return len(self.gt_rgb_norm_fnames)

    def __getitem__(self, idx):
        gt_tensor = image_path_to_tensor(self.gt_rgb_norm_fnames[idx])
        pred_tensor = image_path_to_tensor(self.pred_rgb_norm_fnames[idx])
        if self.transform:
            gt_tensor = self.transform(gt_tensor)
            pred_tensor = self.transform(pred_tensor)

        return gt_tensor, pred_tensor


def main(render_path: Path):
    gt_folder_path = render_path / Path("gt/normal")
    prediction_folder_path = render_path / Path("pred/normal")
    data_transform = tensor_to_normals

    # Create the dataset and DataLoader
    batch_size = 5
    dataset = NormalsDataset(
        gt_folder_path, prediction_folder_path, transform=data_transform
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    metrics = NormalMetrics()
    mae_acc, rmse_acc, mean_acc, med_acc = 0.0, 0.0, 0.0, 0.0
    # Access batches in a loop
    for batch_idx, (gt_images, prediction_images) in enumerate(dataloader):
        mae_run, rmse_run, mean_err_run, med_err_run = metrics(
            prediction_images.permute(0, 3, 1, 2), gt_images.permute(0, 3, 1, 2)
        )
        mae_acc += mae_run
        rmse_acc += rmse_run
        mean_acc += mean_err_run
        med_acc += med_err_run

    print("Performance report (normals estimation):")
    print("Error (Lower is better):")
    print(
        f"MAE (rad): {mae_acc / len(dataloader)}; MAE (deg): {np.rad2deg(mae_acc / len(dataloader))}"
    )
    print(f"RMSE: {rmse_acc / len(dataloader)}")
    print(f"Mean: {mean_acc / len(dataloader)}")
    print(f"Med: {med_acc / len(dataloader)}")
    print(
        f"{mae_acc / len(dataloader)}, {np.rad2deg(mae_acc / len(dataloader))},"
        f" {rmse_acc / len(dataloader)},"
        f" {mean_acc / len(dataloader)}"
    )


if __name__ == "__main__":
    tyro.cli(main)
