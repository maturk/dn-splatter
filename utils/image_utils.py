#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def gradient_map(image):
    sobel_x = (
        torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .cuda()
        / 4
    )
    sobel_y = (
        torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .cuda()
        / 4
    )

    grad_x = torch.cat(
        [
            F.conv2d(image[i].unsqueeze(0), sobel_x, padding=1)
            for i in range(image.shape[0])
        ]
    )
    grad_y = torch.cat(
        [
            F.conv2d(image[i].unsqueeze(0), sobel_y, padding=1)
            for i in range(image.shape[0])
        ]
    )
    magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    magnitude = magnitude.norm(dim=0, keepdim=True)

    return magnitude


def dilate_edge(edge, dilation_size=1):

    kernel_size = 2 * dilation_size + 1
    dilation_kernel = torch.ones((1, 1, kernel_size, kernel_size)).cuda()

    edge_dilated = F.conv2d(edge, dilation_kernel, padding=dilation_size)
    edge_dilated = torch.clamp(edge_dilated, 0, 1)

    return edge_dilated


def normal2curv(normal):
    # normal = normal.detach()
    n = normal.permute([1, 2, 0])
    n = torch.nn.functional.pad(n[None], [0, 0, 1, 1, 1, 1], mode="replicate")
    m = torch.nn.functional.pad(
        m[None].to(torch.float32), [0, 0, 1, 1, 1, 1], mode="replicate"
    ).to(torch.bool)
    n_c = (n[:, 1:-1, 1:-1, :]) * m[:, 1:-1, 1:-1, :]
    n_u = (n[:, :-2, 1:-1, :] - n_c) * m[:, :-2, 1:-1, :]
    n_l = (n[:, 1:-1, :-2, :] - n_c) * m[:, 1:-1, :-2, :]
    n_b = (n[:, 2:, 1:-1, :] - n_c) * m[:, 2:, 1:-1, :]
    n_r = (n[:, 1:-1, 2:, :] - n_c) * m[:, 1:-1, 2:, :]
    curv = (n_u + n_l + n_b + n_r)[0]
    curv = curv.permute([2, 0, 1])
    curv = curv.norm(1, 0, True)
    return curv


def find_edges(im, threshold=0.01, dilation_itr=1):
    laplacian_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=im.dtype, device=im.device
    ).float()
    laplacian_kernel = laplacian_kernel.unsqueeze(0).unsqueeze(0)
    # Apply the Laplacian kernel to the image
    if im.shape[0] == 1:
        laplacian = F.conv2d(
            (1.0 / (im + 1e-6)).unsqueeze(
                0
            ),  # Add batch dimension, shape: (1, 1, h, w)
            laplacian_kernel,
            padding=1,
        ).squeeze(
            0
        )  # shape: (1, h, w)

        edges = (laplacian > threshold) * 1.0
        structure_el = laplacian_kernel * 0.0 + 1.0

        dilated_edges = edges
        for i in range(dilation_itr):
            dilated_edges = F.conv2d(
                dilated_edges.unsqueeze(0),
                structure_el,
                padding=1,
            ).squeeze(0)
    elif im.shape[0] == 3:
        laplacian = []
        for i in range(3):
            channel_laplacian = F.conv2d(
                (1.0 / (im[i : i + 1] + 1e-6)).unsqueeze(0),  # Shape: (1, 1, h, w)
                laplacian_kernel,
                padding=1,
            ).squeeze(
                0
            )  # Shape: (1, h, w)
            laplacian.append(channel_laplacian)
        laplacian = torch.cat(laplacian, dim=0)  # Shape: (3, h, w)
        edges = (laplacian > threshold) * 1.0
        structure_el = laplacian_kernel * 0.0 + 1.0

        for i in range(dilation_itr):
            dilated_edges = []
            for j in range(3):
                channel_dilated = F.conv2d(
                    edges[j : j + 1].unsqueeze(0),  # Shape: (1, 1, h, w)
                    structure_el,
                    padding=1,
                ).squeeze(
                    0
                )  # Shape: (1, h, w)
                dilated_edges.append(channel_dilated)
            dilated_edges = torch.cat(dilated_edges, dim=0)  # Shape: (3, h, w)

    dilated_edges = dilated_edges > 0.0
    return dilated_edges
