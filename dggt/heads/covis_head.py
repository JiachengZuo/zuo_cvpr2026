# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
CoVisHead — Lightweight DPT-style head for co-visibility prediction.

Predicts per-pixel visible frame count (0~S) from backbone features
(aggregated_tokens_list). The head shares the DPT multi-scale fusion
architecture but with reduced feature channels for efficiency.

Also produces a shared feature map (cov_feat) that can be used as
conditioning input for DPTHead and GaussianHead.
"""

import torch
import torch.nn as nn
from typing import List, Tuple

from .dpt_head import (
    _make_scratch,
    _make_fusion_block,
    custom_interpolate,
)
from .utils import create_uv_grid, position_grid_to_embed


class CoVisHead(nn.Module):
    """
    Lightweight DPT head for co-visibility prediction.

    Args:
        dim_in (int): Input feature dimension (2048 for aggregated tokens).
        patch_size (int): Patch size (default 14).
        features (int): Base feature channels for fusion (default 128).
        out_channels (List[int]): Projection output channels per layer.
        intermediate_layer_idx (List[int]): Which aggregator layers to use.
        pos_embed (bool): Whether to apply positional embedding.
        down_ratio (int): Downscaling factor for output resolution.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        patch_size: int = 14,
        features: int = 128,
        out_channels: List[int] = None,
        intermediate_layer_idx: List[int] = None,
        pos_embed: bool = True,
        down_ratio: int = 1,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = [128, 256, 512, 512]
        if intermediate_layer_idx is None:
            intermediate_layer_idx = [4, 11, 17, 23]

        self.patch_size = patch_size
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx

        self.norm = nn.LayerNorm(dim_in)

        # Projection layers for each selected layer.
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=oc,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for oc in out_channels
            ]
        )

        # Resize layers for upsampling feature maps.
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        self.scratch = _make_scratch(out_channels, features, expand=False)

        # Attach additional modules to scratch.
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        head_features_1 = features  # 128
        head_features_2 = 16
        output_dim = 2  # 1 for covisibility score + 1 for confidence

        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1,
            head_features_1 // 2,  # 128 -> 64
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(
                head_features_1 // 2,  # 64
                head_features_2,  # 16
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                head_features_2,
                output_dim,  # 2
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with chunked frame processing.

        Returns:
            cov_score: [B, S, H, W] co-visibility score (visible frame count)
            cov_conf:  [B, S, H, W] prediction confidence
            cov_feat:  [B*S, 64, H, W] shared features for conditioning
        """
        B, S, _, H, W = images.shape

        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(
                aggregated_tokens_list, images, patch_start_idx
            )

        assert frames_chunk_size > 0
        all_scores, all_confs, all_feats = [], [], []

        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(
                frames_start_idx + frames_chunk_size, S
            )
            chunk_score, chunk_conf, chunk_feat = self._forward_impl(
                aggregated_tokens_list,
                images,
                patch_start_idx,
                frames_start_idx,
                frames_end_idx,
            )
            all_scores.append(chunk_score)
            all_confs.append(chunk_conf)
            all_feats.append(chunk_feat)

        return (
            torch.cat(all_scores, dim=1),
            torch.cat(all_confs, dim=1),
            torch.cat(all_feats, dim=0),
        )

    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Implementation of the CoVisHead forward pass.

        Returns:
            cov_score: [B, S_chunk, H, W]
            cov_conf:  [B, S_chunk, H, W]
            cov_feat:  [B*S_chunk, 64, H, W]
        """
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out = []
        dpt_idx = 0

        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]

            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            x = x.view(B * S, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape(
                (x.shape[0], x.shape[-1], patch_h, patch_w)
            )
            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)

            out.append(x)
            dpt_idx += 1

        # Fuse features from multiple layers.
        out = self.scratch_forward(out)

        # Interpolate to target resolution.
        out = custom_interpolate(
            out,
            (
                int(patch_h * self.patch_size / self.down_ratio),
                int(patch_w * self.patch_size / self.down_ratio),
            ),
            mode="bilinear",
            align_corners=True,
        )

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        # Capture shared features BEFORE output_conv2
        out_conv1 = self.scratch.output_conv1(out)  # [B*S, 64, H, W]
        cov_feat = out_conv1  # shared features for conditioning

        out_conv2 = self.scratch.output_conv2(out_conv1)  # [B*S, 2, H, W]

        # Activate: score via sigmoid (GT is [0,1] normalized count), conf via sigmoid
        score = torch.sigmoid(out_conv2[:, 0:1, :, :])
        conf = torch.sigmoid(out_conv2[:, 1:2, :, :])

        score = score.view(B, S, H, W)
        conf = conf.view(B, S, H, W)

        return score, conf, cov_feat

    def scratch_forward(
        self, features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass through the fusion blocks.
        """
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4

        out = self.scratch.refinenet3(
            out, layer_3_rn, size=layer_2_rn.shape[2:]
        )
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(
            out, layer_2_rn, size=layer_1_rn.shape[2:]
        )
        del layer_2_rn, layer_2

        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1

        out = self.scratch.output_conv1(out)
        return out

    def _apply_pos_embed(
        self,
        x: torch.Tensor,
        W: int,
        H: int,
        ratio: float = 0.1,
    ) -> torch.Tensor:
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(
            patch_w,
            patch_h,
            aspect_ratio=W / H,
            dtype=x.dtype,
            device=x.device,
        )
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(
            x.shape[0], -1, -1, -1
        )
        return x + pos_embed


# Aliases for consistency
CovisibilityHead = CoVisHead
