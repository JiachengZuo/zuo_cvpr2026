# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import Adam
from gsplat.rendering import rasterization
from tqdm import tqdm
import os
from IPython import embed
from torch.utils.data import Dataset, DataLoader
import random
import open3d as o3d
from PIL import Image
from torchvision import transforms as TF


palette_10 = {
    0: (128, 64, 128),
    1: (244, 35, 232),
    2: (70, 70, 70),
    3: (102, 102, 156),
    4: (190, 153, 153),
    5: (153, 153, 153),
    6: (250, 170, 30),
    7: (220, 220, 0),
    8: (107, 142, 35),
    9: (70, 130, 180),
}

# 0:Road
# 1:Building
# 2:Vegetation
# 3:Vehicle
# 4:Person
# 5:Cyclist
# 6:Traffic Sign
# 7:truck
# 8:Sidewalk
# 9:Sky

def concat_list(list_1, list_2):
    if list_2[0].shape == torch.Size([0]):
        return list_1
    concated_list = []
    for i in range(len(list_1)):
        item = torch.concat((list_1[i],list_2[i]),dim=0)
        concated_list.append(item)
    return concated_list


def get_masked_gs(point_map, gs_map, mask, idx=None):
    # point_map: B,S,H,W,3 (no K dim)
    # gs_map: B,S,H,W,K,C when K>1, or B,S,H,W,C when K=1
    # mask: B,S,H,W
    if idx is not None:
        point_map = point_map[:,idx,...]
        gs_map = gs_map[:,idx,...]
        mask = mask[:,idx,...]
    world_points = point_map[mask].reshape(-1, 3)  # (N, 3)
    selected = gs_map[mask]  # (N, K, C) when K>1, else (N, C)
    if selected.dim() == 3:
        N, K, C = selected.shape
        selected = selected.reshape(N * K, C)  # flatten K into batch dim
        world_points = world_points.repeat_interleave(K, dim=0)  # (N*K, 3)
    rgbs = selected[..., :3]
    opacity = selected[..., 3:4].reshape(-1)
    scales = selected[..., 4:7].reshape(-1, 3)
    rotation = selected[..., 7:11].reshape(-1, 4)
    return world_points, rgbs, opacity, scales, rotation


def get_split_gs(gs_map, mask):
    """
    Extract Gaussian parameters from gs_map using a boolean mask.

    Args:
        gs_map: (..., H, W, K, C) when K>1, or (..., H, W, C) when K=1
        mask: (..., H, W) boolean mask

    Returns:
        rgbs: (N*K, 3)
        opacity: (N*K,)
        scales: (N*K, 3)
        rotation: (N*K, 4)
    """
    selected = gs_map[mask]  # (N, K, C) when K>1, else (N, C)
    if selected.dim() == 3:
        N, K, C = selected.shape
        selected = selected.reshape(N * K, C)  # flatten K into batch dim
    rgbs = selected[..., :3]
    opacity = selected[..., 3:4].reshape(-1)
    scales = selected[..., 4:7].reshape(-1, 3)
    rotation = selected[..., 7:11].reshape(-1, 4)
    return rgbs, opacity, scales, rotation


def gs_dict(points, rgbs, opacity, scales, rotation):
    gs_dict = {}
    gs_dict['means'] = points
    gs_dict['quats'] = rotation
    if opacity.dim() == 1:
        opacity = opacity[...,None]
    gs_dict['opacities'] = opacity
    gs_dict['scales'] = scales
    gs_dict['features_dc'] = rgbs
    return gs_dict

def get_gs_items(gs_dict):
    points = gs_dict['means'] 
    rotation = gs_dict['quats'] 
    opacity = gs_dict['opacities'][...,0] 
    scales = gs_dict['scales'] 
    rgbs = gs_dict['features_dc'] 
    return points, rgbs, opacity, scales, rotation


import numpy as np

import numpy as np

def downsample_3dgs(points, rgbs, opacity, scales, rotation, num_points=200000):
    N = points.shape[0]
    if num_points >= N:
        return points, rgbs, opacity, scales, rotation

    # Compute importance weights: opacity * volume
    volume = scales.prod(dim=1)                    # (N,)
    weights = opacity * volume                     # (N,)
    weights = weights / weights.sum()              # Normalize to sum to 1

    # Sample indices with probability proportional to weights
    indices = torch.multinomial(weights, num_points, replacement=False)  # (num_points,)

    return (
        points[indices],
        rgbs[indices],
        opacity[indices],
        scales[indices],
        rotation[indices]
    )


def get_gs_conf_flat(gs_conf, mask):
    """
    Extract and flatten gs_conf values using a boolean mask.

    Args:
        gs_conf: (..., H, W, K) when K>1, or (..., H, W) when K=1
        mask: (..., H, W) boolean mask

    Returns:
        conf_flat: (N*K,) flat confidence values
    """
    selected = gs_conf[mask]  # (N, K) when K>1, else (N,)
    if selected.dim() == 2:
        selected = selected.reshape(-1)  # (N*K,)
    return selected


def repeat_mask_derived_tensor(tensor_from_mask, gs_map, mask):
    """
    Repeat a tensor derived from a spatial mask to match K-expanded Gaussian counts.

    When K>1, Gaussian params are K-expanded (N*K rows), but mask-derived values
    like gs_dynamic_list, gs_timestamps, frame_idx are only N rows.
    This helper repeats them K times to match.

    Args:
        tensor_from_mask: (N, ...) tensor derived from spatial mask indexing
        gs_map: the gs_map tensor (to check for K dimension)
        mask: (..., H, W) the spatial mask used to derive tensor_from_mask

    Returns:
        Repeated tensor with shape (N*K, ...) if K>1, else unchanged.
    """
    if gs_map.dim() >= 6:  # has K dimension: (B, S, H, W, K, C)
        K = gs_map.shape[-2]
        return tensor_from_mask.repeat_interleave(K, dim=0)
    return tensor_from_mask


def get_k_from_gs_map(gs_map):
    """Get K (number of Gaussians per pixel) from gs_map shape."""
    if gs_map.dim() >= 6:  # (B, S, H, W, K, C)
        return gs_map.shape[-2]
    # Old format (B, S, H, W, C) — one Gaussian per pixel
    return 1


def apply_k_activation_mask(gs_map, gs_conf, active_count, active_k_mask=None):
    """
    Apply per-pixel K-dim activation mask to gs_map by zeroing opacity.

    This implements the "activation based on redundancy/demand principles":
    - Redundant pixels (high repeat, low contribution) → fewer active Gaussians
    - High-demand pixels (low repeat, high error) → more active Gaussians

    If active_k_mask is provided, it's used directly. Otherwise, active_count
    determines how many of the first K slots are active per pixel.

    Args:
        gs_map: (..., H, W, K, 11) Gaussian parameters
        gs_conf: (..., H, W, K) per-slot confidence values
        active_count: (..., H, W) number of Gaussians to activate per pixel
                      (values 0..K, typically computed from demand scoring)
        active_k_mask: (..., H, W, K) optional pre-computed boolean mask
                      (overrides active_count if provided)

    Returns:
        active_mask: (..., H, W, K) boolean mask (True = active)
        gs_map_masked: gs_map with opacity zeroed for inactive K slots
    """
    K = gs_map.shape[-2]

    if active_k_mask is not None:
        active_mask = active_k_mask.bool()
    else:
        # Simple slot activation: first active_count slots are active
        # (model learns to put best Gaussians in earlier slots)
        active_mask = torch.zeros(gs_map.shape[:-1], dtype=torch.bool,
                                  device=gs_map.device)
        for k in range(K):
            active_mask[..., k] = (k < active_count)

    gs_map_masked = gs_map.clone()
    # Zero opacity for inactive slots (channel 3 is opacity within the 11 params)
    gs_map_masked[..., 3] = gs_map_masked[..., 3] * active_mask.float()
    return active_mask, gs_map_masked
