"""KNN implementations"""

import torch
from torch import Tensor

device = torch.device("cuda:0")


def fast_knn(x: Tensor, y: Tensor, k: int = 2):
    """Wrapper for torch_cluster.knn

    Args:
        x: input data
        y: query data
        k: k-nearest neighbours
    """
    from torch_cluster import knn

    assert x.is_cuda
    assert y.is_cuda
    assert x.dim() == y.dim() == 2
    with torch.no_grad():
        k = k + 1
        outs = knn(x.clone(), y.clone(), k, None, None)[1, :]
        outs = outs.reshape(y.shape[0], k)[:, 1:]
    return outs


def knn_sk(x: torch.Tensor, y: torch.Tensor, k: int):
    import numpy as np

    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()

    from sklearn.neighbors import NearestNeighbors

    nn_model = NearestNeighbors(
        n_neighbors=k + 1, algorithm="auto", metric="euclidean"
    ).fit(x_np)

    distances, indices = nn_model.kneighbors(y_np)

    return torch.from_numpy(indices[:, 1:].astype(np.int64)).long().to(x.device)
