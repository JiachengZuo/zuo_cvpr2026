#!/usr/bin/env python3
"""
Deactivate Redundant Gaussians by Visibility & Contribution Analysis.

Based on: test_method.md — "Repeated Visibility + Gradient Importance +
           Rendering Contribution" method.

Algorithm:
  1. Run DGGT inference, collect all static Gaussians into a pool.
  2. For each Gaussian, compute visible_count: how many frames it projects into.
  3. Compute contribution proxy: opacity × projected scale area.
  4. Compute redundancy_score = repeat_score × (1 - contribution_norm).
  5. Deactivate (set opacity=0) Gaussians that are high-repeat AND low-contribution.
  6. Re-render and compare PSNR before vs after deactivation.

Usage:
    source /home/djhuai/anaconda3/bin/activate dggt
    python deactivate_redundant_gaussians.py \
        --image_dir data/nuscenes/processed_2Hz/mini/mini \
        --scene_names 003 \
        --ckpt_path /path/to/model_latest_waymo.pt \
        --output_path output_redundancy_test \
        --mode 2 --sequence_length 4 --input_views 1 \
        --repeat_thresh 0.5 --contrib_thresh 0.2
"""

import argparse
import json
import os
import time
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips

from dggt.models.vggt import VGGT
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from dggt.utils.geometry import unproject_depth_map_to_point_map
from dggt.utils.gs import concat_list, get_split_gs
from gsplat.rendering import rasterization
from datasets.dataset import WaymoOpenDataset


# ============================================================================
#  Utilities (mirror inference.py implementations exactly)
# ============================================================================

def alpha_t(t, t0, alpha, gamma0=1, gamma1=0.1):
    sigma = torch.log(torch.tensor(gamma1)).to(gamma0.device) / ((gamma0) ** 2 + 1e-6)
    conf = torch.exp(sigma * (t0 - t) ** 2)
    alpha_ = alpha * conf
    return alpha_.float()


def compute_metrics(img1, img2, loss_fn):
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)
    psnr_list, ssim_list, lpips_list = [], [], []
    for i in range(img1.shape[0]):
        im1 = img1[i].cpu().permute(1, 2, 0).numpy()
        im2 = img2[i].cpu().permute(1, 2, 0).numpy()
        psnr = peak_signal_noise_ratio(im1, im2, data_range=1.0)
        ssim = structural_similarity(im1, im2, channel_axis=2, data_range=1.0)
        lpips_val = loss_fn(img1[i].unsqueeze(0) * 2 - 1,
                             img2[i].unsqueeze(0) * 2 - 1)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_list.append(lpips_val.item())
    return (sum(psnr_list) / len(psnr_list),
            sum(ssim_list) / len(ssim_list),
            sum(lpips_list) / len(lpips_list))


def project_world_to_2d(pts, w2c_3x4, K_3x3, H, W, thresh=0.001):
    """Project 3D world pts (N,3) -> 2D pixel coords.

    Returns:
        px: (N, 2) float tensor of pixel coordinates
        visible: (N,) bool tensor (in front of camera AND within bounds)
    """
    if isinstance(pts, np.ndarray):
        pts = torch.from_numpy(pts)
    device = w2c_3x4.device if isinstance(w2c_3x4, torch.Tensor) else pts.device
    pts = pts.to(device).float()
    # w2c may be 3x4 or 4x4 — slice works for both
    if isinstance(w2c_3x4, torch.Tensor):
        w2c = w2c_3x4.to(device)
        R = w2c[:3, :3]
        t = w2c[:3, 3]
    else:
        R = w2c_3x4[:3, :3]
        t = w2c_3x4[:3, 3]
    K = K_3x3.to(device) if isinstance(K_3x3, torch.Tensor) else K_3x3

    cam = pts @ R.T + t
    in_front = cam[:, 2] > thresh
    homo = cam @ K.T
    px = homo[:, :2] / (homo[:, 2:3] + 1e-8)
    in_bounds = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
    visible = in_front & in_bounds
    return px, visible


def parse_scene_names(scene_names_str):
    scene_names_str = scene_names_str.strip()
    if scene_names_str.startswith("(") and scene_names_str.endswith(")"):
        start, end = scene_names_str[1:-1].split(",")
        return [str(i).zfill(3) for i in range(int(start), int(end) + 1)]
    else:
        return [str(int(x)).zfill(3) for x in scene_names_str.split()]


# ============================================================================
#  Redundancy Computation
# ============================================================================

def compute_visible_count(means, scales, extrinsics, intrinsics, H, W,
                           n_frames):
    """Count how many frames each Gaussian is visible in.

    Visibility criteria (per test_method.md):
      - projection position within image bounds
      - depth > 0  (in front of camera)

    Args:
        means:   (N, 3) world positions
        scales:  (N, 3) scale parameters
        extrinsics: (S, 4, 4)  camera-from-world 4x4 matrices
        intrinsics: (S, 3, 3)
        H, W: image resolution
        n_frames: number of frames

    Returns:
        visible_count: (N,) int tensor
        avg_proj_area: (N,) float — average projected radius in pixel units
    """
    N = means.shape[0]
    device = means.device
    visible_count = torch.zeros(N, device=device, dtype=torch.int32)
    total_area = torch.zeros(N, device=device)

    for t in range(n_frames):
        px, vis = project_world_to_2d(means, extrinsics[t], intrinsics[t], H, W)
        visible_count += vis.int()

        # Compute projected area proxy for visible Gaussians
        if vis.any():
            # Gaussian 3D scale magnitude (radius in world units)
            scale_mag = torch.norm(scales[vis], dim=-1).clamp(max=2.0)
            # Distance from camera to Gaussian center
            pt_t = means[vis].float()
            R = extrinsics[t, :3, :3]
            tvec = extrinsics[t, :3, 3]
            z_cam = (pt_t @ R.T + tvec)[:, 2].clamp(min=0.1)
            # Approximate projected radius in pixels: scale / z_cam * focal
            # Using a rough approximation: scale_mag / z_cam
            proj_area = scale_mag / z_cam
            total_area[vis] += proj_area

    # Average projected area per visible frame
    denom = visible_count.float().clamp(min=1)
    avg_area = total_area / denom

    return visible_count, avg_area


def compute_redundancy_score(visible_count, opacity, avg_proj_area,
                              repeat_thresh=0.5,
                              contrib_thresh=0.2,
                              n_frames=8):
    """Compute redundancy score based on method from test_method.md.

    For each Gaussian i:
      - repeat_score[i] = visible_count[i] / n_frames         ∈ [0, 1]
      - contribution[i]  = opacity[i] × avg_proj_area[i]      (un-normalized)
      - contribution_norm[i] = contribution[i] / max_j(contribution[j])
      - redundancy_score[i] = repeat_score[i] × (1 - contribution_norm[i])

    A Gaussian is marked redundant if BOTH:
      (a) repeat_score >= repeat_thresh   (high repeat — seen in many frames)
      (b) contribution_norm <= contrib_thresh  (low contribution)

    This follows test_method.md step 5-6.

    Returns:
        redundancy_score: (N,) float  [0, 1]
        repeat_score: (N,) float      [0, 1]
        contribution_norm: (N,) float [0, 1]
        redundant_mask: (N,) bool
    """
    device = visible_count.device

    # 1. Repeat score
    repeat_score = visible_count.float() / float(n_frames)

    # 2. Contribution proxy: opacity × projected area
    contrib_raw = opacity.float() * avg_proj_area
    contrib_max = contrib_raw.max() + 1e-6
    contribution_norm = contrib_raw / contrib_max

    # 3. Redundancy score
    redundancy_score = repeat_score * (1.0 - contribution_norm)

    # 4. Redundant mask: high repeat AND low contribution
    redundant_mask = (
        (repeat_score >= repeat_thresh) &
        (contribution_norm <= contrib_thresh)
    )

    return redundancy_score, repeat_score, contribution_norm, redundant_mask


# ============================================================================
#  Rendering — one pass for all frames (exact mirror of inference.py)
# ============================================================================

def render_all_frames(static_points, static_rgbs, static_opacity_base,
                      static_scales, static_rotation,
                      dynamic_points, dynamic_rgbs, dynamic_opacitys,
                      dynamic_scales, dynamic_rotations,
                      gs_timestamps, gs_conf, timestamps,
                      extrinsic, intrinsic,
                      sky_model, images,
                      H, W, S, active_mask=None):
    """Render S frames of Gaussians via gsplat rasterization.

    This mirrors inference.py lines 273-331 exactly.

    Args:
        ... (Gaussian parameters)
        active_mask: (N_static,) bool or None.
            If provided, static opacity is zeroed for inactive Gaussians.

    Returns:
        rendered_image: (S, 3, H, W) composed renders
        renders_raw:    (S, H, W, 3) RGB renders (before sky composition)
        alphas:         (S, 1, H, W) alpha channel
    """
    chunked_renders = []
    chunked_alphas = []

    for idx in range(S):
        t0 = timestamps[idx]
        static_opacity_ = alpha_t(gs_timestamps, t0, static_opacity_base,
                                   gamma0=gs_conf)

        # Apply active mask: zero opacity for redundant Gaussians
        if active_mask is not None:
            static_opacity_ = static_opacity_.clone()
            static_opacity_[~active_mask] = 0.0

        static_gs_list = [static_points, static_rgbs,
                          static_opacity_, static_scales, static_rotation]

        if dynamic_points and dynamic_points[0].shape[0] > 0:
            world_points, rgbs, opacity, scales, rotation = concat_list(
                static_gs_list,
                [dynamic_points[idx], dynamic_rgbs[idx],
                 dynamic_opacitys[idx], dynamic_scales[idx],
                 dynamic_rotations[idx]]
            )
        else:
            world_points, rgbs, opacity, scales, rotation = static_gs_list

        renders_chunk, alphas_chunk, _ = rasterization(
            means=world_points,
            quats=rotation,
            scales=scales,
            opacities=opacity,
            colors=rgbs,
            viewmats=extrinsic[idx:idx + 1],
            Ks=intrinsic[idx:idx + 1],
            width=W,
            height=H,
            render_mode='RGB+ED',
        )
        chunked_renders.append(renders_chunk)
        chunked_alphas.append(alphas_chunk)

    # Assemble frame stack — mirror inference.py lines 321-331
    renders = torch.cat(chunked_renders, dim=0)       # (S, H, W, 4) [R,G,B,D]
    renders_rgb = renders[..., :-1]                    # (S, H, W, 3)
    alphas = torch.cat(chunked_alphas, dim=0)          # (S, H, W, 1)

    # Sky background
    bg_render = sky_model(images, extrinsic, intrinsic)
    bg_render = (bg_render - bg_render.min()) / \
                (bg_render.max() - bg_render.min() + 1e-8)

    renders_composed = alphas * renders_rgb + (1 - alphas) * bg_render
    # Permute to (S, 3, H, W) for metric computation
    rendered_image = renders_composed.permute(0, 3, 1, 2)

    return rendered_image, renders_rgb, alphas.permute(0, 3, 1, 2)


# ============================================================================
#  Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Deactivate Redundant Gaussians by Visibility & Contribution")
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--scene_names', type=str, nargs='+', required=True)
    parser.add_argument('--input_views', type=int, default=1)
    parser.add_argument('--sequence_length', type=int, default=4)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--mode', type=int, choices=[2, 3], default=2)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--intervals', type=int, default=2)

    # Redundancy thresholds (from test_method.md recommendations)
    parser.add_argument('--repeat_thresh', type=float, default=0.5,
                        help='Min repeat_score to be considered redundant '
                             '(visible_count/n_frames >= thresh)')
    parser.add_argument('--contrib_thresh', type=float, default=0.2,
                        help='Max contribution_norm to be considered redundant '
                             '(contribution <= thresh means low contribution)')

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    loss_fn = lpips.LPIPS(net='alex').to(device)

    # ---- Data ----
    scene_names_str = ' '.join(args.scene_names)
    scene_names = parse_scene_names(scene_names_str)
    dataset = WaymoOpenDataset(
        args.image_dir,
        scene_names=scene_names,
        sequence_length=args.sequence_length,
        start_idx=args.start_idx,
        mode=args.mode,
        views=args.input_views,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # ---- Model ----
    model = VGGT().to(device)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['images'].to(device)
            sky_mask = batch['masks'].to(device).permute(0, 1, 3, 4, 2)
            timestamps = batch['timestamps'][0].to(device)

            bg_mask = (sky_mask == 0).any(dim=-1)

            print(f"\n{'='*70}")
            print(f"  Scene {scene_names[batch_idx]}: Redundancy Analysis")
            print(f"{'='*70}")

            # ---- DGGT Inference ----
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
                H, W = images.shape[-2:]
                extrinsics, intrinsics = pose_encoding_to_extri_intri(
                    predictions['pose_enc'], (H, W))
                extrinsic = extrinsics[0]  # (S, 3, 4)
                bottom = torch.tensor(
                    [0.0, 0.0, 0.0, 1.0],
                    device=extrinsic.device
                ).view(1, 1, 4).expand(extrinsic.shape[0], 1, 4)
                extrinsic = torch.cat([extrinsic, bottom], dim=1)  # (S, 4, 4)
                intrinsic = intrinsics[0]  # (S, 3, 3)

                depth_map = predictions["depth"][0]
                point_map = unproject_depth_map_to_point_map(
                    depth_map, extrinsics[0], intrinsics[0])
                point_map = point_map[None, ...]  # (1, S, H, W, 3)
                point_map = torch.from_numpy(point_map).to(device).float()

                gs_map = predictions["gs_map"]
                gs_conf = predictions["gs_conf"]
                dy_map = predictions["dynamic_conf"].squeeze(-1)

            S = dy_map.shape[1]  # number of frames

            # ---- Collect Static Gaussians ----
            # Mirror inference.py lines 192-200
            static_mask = (bg_mask & (dy_map < 0.5))
            static_points = point_map[static_mask].reshape(-1, 3)
            static_rgbs, static_opacity, static_scales, static_rotation = \
                get_split_gs(gs_map, static_mask)
            gs_dynamic_list = dy_map[static_mask].sigmoid()
            static_opacity = static_opacity * (1 - gs_dynamic_list)
            static_gs_conf = gs_conf[static_mask]
            frame_idx = torch.nonzero(static_mask, as_tuple=False)[:, 1]
            gs_timestamps = timestamps[frame_idx]

            N_static = static_points.shape[0]
            print(f"  Static Gaussians: {N_static:,}")

            # ---- Collect Dynamic Gaussians per frame ----
            # Mirror inference.py lines 229-244
            dynamic_points_l, dynamic_rgbs_l, dynamic_opacitys_l = [], [], []
            dynamic_scales_l, dynamic_rotations_l = [], []
            for i in range(S):
                point_map_i = point_map[:, i]
                bg_mask_i = bg_mask[:, i]
                dyn_pt = point_map_i[bg_mask_i].reshape(-1, 3)
                dyn_rgb, dyn_op, dyn_sc, dyn_rot = get_split_gs(
                    gs_map[:, i], bg_mask_i)
                dyn_list_i = dy_map[:, i][bg_mask_i].sigmoid()
                dyn_op = dyn_op * dyn_list_i
                dynamic_points_l.append(dyn_pt)
                dynamic_rgbs_l.append(dyn_rgb)
                dynamic_opacitys_l.append(dyn_op)
                dynamic_scales_l.append(dyn_sc)
                dynamic_rotations_l.append(dyn_rot)

            n_dynamic = [dp.shape[0] for dp in dynamic_points_l]
            print(f"  Dynamic per frame: {n_dynamic}")

            # ================================================================
            # [Redundancy Analysis] — test_method.md steps 2-5
            # ================================================================
            print(f"\n  --- Computing visible_count for {N_static:,} static "
                  f"Gaussians ---")
            t0 = time.time()

            visible_count, avg_proj_area = compute_visible_count(
                static_points, static_scales,
                extrinsic, intrinsic, H, W, S)

            redundancy_score, repeat_score, contrib_norm, redundant_mask = \
                compute_redundancy_score(
                    visible_count, static_opacity, avg_proj_area,
                    repeat_thresh=args.repeat_thresh,
                    contrib_thresh=args.contrib_thresh,
                    n_frames=S)

            n_redundant = redundant_mask.sum().item()
            elapsed = time.time() - t0
            print(f"  Computed in {elapsed:.1f}s")

            # ---- Stats ----
            print(f"\n  --- Redundancy Statistics ---")
            print(f"  Visible count distribution:")
            for v in range(S + 1):
                n_at_v = (visible_count == v).sum().item()
                if n_at_v > 0:
                    print(f"    seen_by_{v}_frames: {n_at_v:>8,} "
                          f"({100 * n_at_v / N_static:.1f}%)")

            print(f"\n  Repeat score (visible_count / {S}):")
            print(f"    mean={repeat_score.float().mean():.3f}  "
                  f"median={repeat_score.float().median():.3f}")
            print(f"    >= {args.repeat_thresh}: "
                  f"{(repeat_score >= args.repeat_thresh).sum().item():,}")

            print(f"\n  Contribution norm:")
            print(f"    mean={contrib_norm.float().mean():.4f}  "
                  f"median={contrib_norm.float().median():.4f}")
            print(f"    <= {args.contrib_thresh}: "
                  f"{(contrib_norm <= args.contrib_thresh).sum().item():,}")

            print(f"\n  Redundancy score:")
            print(f"    mean={redundancy_score.float().mean():.4f}  "
                  f"max={redundancy_score.float().max():.4f}")

            print(f"\n  >>> REDUNDANT Gaussians (repeat>={args.repeat_thresh} "
                  f"& contrib<={args.contrib_thresh}): "
                  f"{n_redundant:,} ({100 * n_redundant / max(N_static, 1):.1f}%) "
                  f"<<<")

            # Active mask: True = active, False = redundant (deactivated)
            active_mask = torch.ones(N_static, dtype=torch.bool, device=device)
            active_mask[redundant_mask] = False

            # ================================================================
            # Render: BASELINE (all Gaussians active)
            # ================================================================
            print(f"\n  --- Rendering BASELINE (all {N_static:,} static) ---")
            baseline_final, _, _ = render_all_frames(
                static_points, static_rgbs, static_opacity,
                static_scales, static_rotation,
                dynamic_points_l, dynamic_rgbs_l, dynamic_opacitys_l,
                dynamic_scales_l, dynamic_rotations_l,
                gs_timestamps, static_gs_conf, timestamps,
                extrinsic, intrinsic,
                model.sky_model, images,
                H, W, S,
                active_mask=None  # no masking — baseline
            )

            # ================================================================
            # Render: MASKED (redundant Gaussians deactivated)
            # ================================================================
            print(f"  --- Rendering MASKED ({n_redundant:,} redundant "
                  f"deactivated) ---")
            masked_final, _, _ = render_all_frames(
                static_points, static_rgbs, static_opacity,
                static_scales, static_rotation,
                dynamic_points_l, dynamic_rgbs_l, dynamic_opacitys_l,
                dynamic_scales_l, dynamic_rotations_l,
                gs_timestamps, static_gs_conf, timestamps,
                extrinsic, intrinsic,
                model.sky_model, images,
                H, W, S,
                active_mask=active_mask  # redundant Gaussians → opacity = 0
            )

            # ================================================================
            # Compare Metrics
            # ================================================================
            target = images[0]  # (S, 3, H, W)

            psnr_base, ssim_base, lpips_base = compute_metrics(
                baseline_final, target, loss_fn)
            psnr_mask, ssim_mask, lpips_mask = compute_metrics(
                masked_final, target, loss_fn)

            print(f"\n  {'='*60}")
            print(f"  PSNR / SSIM / LPIPS Comparison")
            print(f"  {'='*60}")
            print(f"  Baseline (all {N_static:,}):  "
                  f"PSNR={psnr_base:.4f}  SSIM={ssim_base:.4f}  "
                  f"LPIPS={lpips_base:.4f}")
            print(f"  Masked  (-{n_redundant:,} off): "
                  f"PSNR={psnr_mask:.4f}  SSIM={ssim_mask:.4f}  "
                  f"LPIPS={lpips_mask:.4f}")
            print(f"  Delta:                     "
                  f"PSNR={psnr_mask - psnr_base:+.4f}  "
                  f"SSIM={ssim_mask - ssim_base:+.4f}  "
                  f"LPIPS={lpips_mask - lpips_base:+.4f}")

            delta_psnr = psnr_mask - psnr_base
            if delta_psnr > -0.5:
                print(f"  ✓ PSNR drop < 0.5 dB — these {n_redundant:,} "
                      f"Gaussians are SAFE to deactivate!")
            else:
                print(f"  ✗ PSNR drop >= 0.5 dB — threshold may be "
                      f"too aggressive, try lowering")

            # Per-frame metrics
            print(f"\n  Per-frame PSNR comparison:")
            for idx in range(S):
                psnr_f_base, _, _ = compute_metrics(
                    baseline_final[idx:idx + 1], target[idx:idx + 1], loss_fn)
                psnr_f_mask, _, _ = compute_metrics(
                    masked_final[idx:idx + 1], target[idx:idx + 1], loss_fn)
                delta_f = psnr_f_mask - psnr_f_base
                marker = "✓" if delta_f > -0.5 else "✗"
                print(f"    Frame {idx}: baseline={psnr_f_base:.4f}  "
                      f"masked={psnr_f_mask:.4f}  delta={delta_f:+.4f}  "
                      f"{marker}")

            # ---- Save Results ----
            result = {
                'scene': scene_names[batch_idx],
                'n_static': int(N_static),
                'n_redundant': int(n_redundant),
                'redundant_ratio_pct': round(
                    100 * n_redundant / max(N_static, 1), 1),
                'repeat_thresh': float(args.repeat_thresh),
                'contrib_thresh': float(args.contrib_thresh),
                'n_frames': int(S),
                'visible_count_distribution': {
                    str(v): int((visible_count == v).sum().cpu().item())
                    for v in range(S + 1) if (visible_count == v).sum() > 0
                },
                'repeat_score_mean': round(
                    float(repeat_score.float().mean().cpu().item()), 4),
                'contribution_norm_mean': round(
                    float(contrib_norm.float().mean().cpu().item()), 4),
                'redundancy_score_mean': round(
                    float(redundancy_score.float().mean().cpu().item()), 4),
                'redundancy_score_max': round(
                    float(redundancy_score.float().max().cpu().item()), 4),
                'baseline_psnr': round(float(psnr_base), 4),
                'baseline_ssim': round(float(ssim_base), 4),
                'baseline_lpips': round(float(lpips_base), 4),
                'masked_psnr': round(float(psnr_mask), 4),
                'masked_ssim': round(float(ssim_mask), 4),
                'masked_lpips': round(float(lpips_mask), 4),
                'delta_psnr': round(float(delta_psnr), 4),
            }
            results.append(result)

            # Save per-frame comparison images
            scene_out = os.path.join(args.output_path,
                                      scene_names[batch_idx])
            os.makedirs(scene_out, exist_ok=True)

            for idx in range(S):
                base_img = baseline_final[idx].cpu().clamp(0, 1)
                mask_img = masked_final[idx].cpu().clamp(0, 1)
                tgt_img = target[idx].cpu().clamp(0, 1)
                # Stack: baseline | masked | ground truth
                combined = torch.cat([base_img, mask_img, tgt_img], dim=-1)
                T.ToPILImage()(combined).save(
                    os.path.join(scene_out, f"compare_frame_{idx}.png"))

            # ---- Deactivation Heatmap: where were Gaussians removed? ----
            print(f"\n  --- Generating deactivation heatmaps ---")
            for idx in range(S):
                # Recompute per-frame images (in case S > 4)
                base_img_f = baseline_final[idx].cpu().clamp(0, 1)
                tgt_img_f = target[idx].cpu().clamp(0, 1)

                # Project ALL static Gaussians to this frame
                _, vis_all = project_world_to_2d(
                    static_points, extrinsic[idx], intrinsic[idx], H, W)

                if vis_all.any():
                    # Total Gaussians landing on each pixel
                    px_all, _ = project_world_to_2d(
                        static_points[vis_all], extrinsic[idx],
                        intrinsic[idx], H, W)
                    xi_all = px_all[:, 0].long().clamp(0, W - 1).cpu().numpy()
                    yi_all = px_all[:, 1].long().clamp(0, H - 1).cpu().numpy()

                    total_map = np.zeros((H, W), dtype=np.int32)
                    np.add.at(total_map, (yi_all, xi_all), 1)

                    # Redundant Gaussians landing on each pixel
                    vis_red = vis_all & redundant_mask
                    if vis_red.any():
                        px_red, _ = project_world_to_2d(
                            static_points[vis_red], extrinsic[idx],
                            intrinsic[idx], H, W)
                        xi_red = px_red[:, 0].long().clamp(0, W - 1).cpu().numpy()
                        yi_red = px_red[:, 1].long().clamp(0, H - 1).cpu().numpy()

                        deact_map = np.zeros((H, W), dtype=np.int32)
                        np.add.at(deact_map, (yi_red, xi_red), 1)

                        # Deactivation ratio: 0=no removal, 1=all removed
                        ratio_map = np.divide(
                            deact_map.astype(np.float32),
                            total_map.astype(np.float32).clip(min=1),
                        )
                    else:
                        deact_map = np.zeros((H, W), dtype=np.int32)
                        ratio_map = np.zeros((H, W), dtype=np.float32)
                else:
                    total_map = np.zeros((H, W), dtype=np.int32)
                    deact_map = np.zeros((H, W), dtype=np.int32)
                    ratio_map = np.zeros((H, W), dtype=np.float32)

                # All images in BGR for cv2.imwrite
                # Baseline render → BGR
                base_bgr = (base_img_f.permute(1, 2, 0).cpu().numpy()
                            .clip(0, 1) * 255).astype(np.uint8)[:, :, ::-1]
                # GT → BGR
                gt_bgr = (tgt_img_f.permute(1, 2, 0).cpu().numpy()
                          .clip(0, 1) * 255).astype(np.uint8)[:, :, ::-1]

                # Heatmap: red=deactivated, green=kept → BGR
                heat_bgr = np.zeros((H, W, 3), dtype=np.float32)
                heat_bgr[..., 2] = ratio_map          # R channel (BGR→R=2)
                heat_bgr[..., 1] = 1.0 - ratio_map    # G channel
                heat_bgr[..., 0] = 0.0                 # B channel
                heat_bgr = (heat_bgr.clip(0, 1) * 255).astype(np.uint8)

                # Overlay heatmap on baseline
                overlay = cv2.addWeighted(base_bgr, 0.55, heat_bgr, 0.45, 0)

                # Deactivated count heatmap (HOT colormap)
                deact_viz = (np.clip(deact_map / max(deact_map.max(), 1),
                                     0, 1) * 255).astype(np.uint8)
                deact_hot = cv2.applyColorMap(deact_viz, cv2.COLORMAP_HOT)

                # 4-panel grid: top=[baseline | overlay], bot=[deact_hot | GT]
                top = np.concatenate([base_bgr, overlay], axis=1)
                bot = np.concatenate([deact_hot, gt_bgr], axis=1)
                panel = np.concatenate([top, bot], axis=0)

                # Add text labels in each quadrant
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(panel, 'Baseline', (10, 25), font,
                            0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(panel, 'Deact Overlay', (W + 10, 25), font,
                            0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(panel, 'Deact Density', (10, H + 25), font,
                            0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(panel, 'Ground Truth', (W + 10, H + 25), font,
                            0.7, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imwrite(
                    os.path.join(scene_out,
                                 f"deactivation_heatmap_{idx}.png"),
                    panel)

            print(f"  Comparison + heatmap images saved to {scene_out}/")

    # ========================================================================
    # Summary
    # ========================================================================
    summary_path = os.path.join(args.output_path, "redundancy_results.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"  Results saved to: {summary_path}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"  {'Scene':<8} {'Static':>10} {'Redundant':>10} {'Ratio':>8}  "
          f"{'Base PSNR':>10} {'Mask PSNR':>10} {'Delta':>8}")
    print(f"  {'-'*68}")
    for r in results:
        print(f"  {r['scene']:<8} {r['n_static']:>10,} {r['n_redundant']:>10,} "
              f"{r['redundant_ratio_pct']:>7.1f}%  "
              f"{r['baseline_psnr']:>10.4f} {r['masked_psnr']:>10.4f} "
              f"{r['delta_psnr']:>+8.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
