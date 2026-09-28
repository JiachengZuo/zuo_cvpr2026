#!/usr/bin/env python3
# Copyright (C) 2026-present <user>. All rights reserved.
#
# -------------------------------------------------------------------
# Covisibility Map generation with time windows, following CoMapGS
# (arXiv:2503.20998), Eq.1.
#
# Given a set of training view images T = {I0, I1, ..., I_{n-1}} and a
# window size S (e.g. S=4), frames are grouped into temporal windows:
#   Window 0: I_0 ... I_{S-1}
#   Window 1: I_S ... I_{2S-1}
#   ...
#
# Within each window, for every pair (Ii, Ij) with j != i we predict
# dense correspondences C^j_i = f_cor(Ii, Ij) using MASt3R as the
# dense correspondence prediction function f_cor.
#
# The covisibility map M_i of view Ii within its window is (paper Eq.1):
#     M_i(x, y) = sum_{j != i, j in same window} delta_{(x,y) in P(C^j_i)}
# where P(C^j_i) is the set of pixel coordinates of Ii that have a
# correspondence match in Ij. The map value at (x,y) therefore counts in
# how many other views within the same time window the same 3D point is
# visible: 0 .. (window_size - 1).
#
# Each frame is output as a grayscale image: brighter pixels = higher
# repeat visibility across the window.
#
# Maps are refined with morphological operations (erosion/dilation) exactly
# as described in the paper.
# -------------------------------------------------------------------
import argparse
import os
import os.path as path
import json
import re

import numpy as np
import torch

import mast3r.utils.path_to_dust3r  # noqa
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import extract_correspondences_nonsym

from dust3r.inference import inference
from dust3r.utils.image import load_images


def get_argparser():
    parser = argparse.ArgumentParser(
        description="Covisibility map generation with time windows, following CoMapGS (arXiv:2503.20998).")
    parser_weights = parser.add_mutually_exclusive_group(required=True)
    parser_weights.add_argument("--weights", type=str, default=None, help="path to the model weights")
    parser_weights.add_argument("--model_name", type=str, default=None,
                                choices=["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"],
                                help="name of the model weights (downloaded from HF/naver)")
    parser.add_argument("--images", required=True, type=str,
                        help="path to the image folder, or a comma-separated list of image files")
    parser.add_argument("--output", required=True, type=str, help="output directory for the covisibility maps")
    parser.add_argument("--device", type=str, default="cuda", help="pytorch device")
    parser.add_argument("--image_size", type=int, default=512, help="preprocess size for MASt3R")
    parser.add_argument("--subsample", type=int, default=8,
                        help="subsampling step for 2D-2D matching (matches every 8th pixel; 1 = fully dense)")
    parser.add_argument("--conf_thr", type=float, default=0.0,
                        help="keep only matched points with confidence >= conf_thr "
                             "(CoMapGS Eq.1 counts any correspondence; 0.0 keeps all reciprocal matches)")
    parser.add_argument("--no_morpho", action="store_true",
                        help="disable the morphological refinement of the maps (paper keeps it enabled)")
    parser.add_argument("--dilate_k", type=int, default=3, help="kernel size for morphological ops")
    parser.add_argument("--overlay", action="store_true",
                        help="also save a jet heatmap overlay of each covisibility map on the source image")
    parser.add_argument("--visibility", action="store_true",
                        help="also record, per matched pixel, how many other views see it and WHICH views "
                             "(saves visibility_XXX.json per view + pairwise_overlap.{json,txt})")
    parser.add_argument("--camera", type=int, default=None,
                        help="only use images from this camera ID (e.g., 0 for front-view). "
                             "Matches filenames ending with '_{camera_id}.jpg' or '_{camera_id}.png' "
                             "(supports jpg/jpeg/png/bmp/webp/heic).")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="limit to the first N images (e.g., --max_frames 8 to use only first 8 frames)")
    parser.add_argument("--window_size", "-S", type=int, default=None,
                        help="Time window size: group every S consecutive frames into a window. "
                             "Covisibility is computed only among frames within the same window. "
                             "Example: -S 4 groups frames 0-3, 4-7, 8-11, etc. "
                             "If not set, computes across all frames (original behavior).")
    return parser


def get_image_list(images_arg, camera=None, max_frames=None):
    """Return an explicit list of image files from either a folder or a comma-separated list.

    Args:
        images_arg: folder path or comma-separated file list
        camera: if set (int), only keep images matching '_<camera>.<ext>' (e.g. '_0.jpg')
        max_frames: if set (int), limit to the first N images after filtering
    """
    if path.isdir(images_arg):
        file_list = [path.join(dirpath, filename)
                     for dirpath, dirs, filenames in os.walk(images_arg)
                     for filename in filenames]
        file_list = sorted(file_list)
    else:
        file_list = [p.strip() for p in images_arg.split(",") if p.strip()]
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic")
    file_list = [f for f in file_list if f.lower().endswith(valid_ext)]

    # Filter by camera ID (e.g., filename pattern: xxx_0.jpg for front camera)
    if camera is not None:
        before = len(file_list)
        file_list = [f for f in file_list
                     if re.search(rf"_{camera}\.(jpg|jpeg|png|bmp|webp|heic)$",
                                  path.basename(f), re.IGNORECASE)]
        print(f">> camera={camera}: kept {len(file_list)}/{before} images "
              f"(filtered by '_{camera}.<ext>')")

    # Limit to first N frames
    if max_frames is not None and len(file_list) > max_frames:
        print(f">> max_frames={max_frames}: keeping first {max_frames} of {len(file_list)} images")
        file_list = file_list[:max_frames]

    return file_list


def matches_to_pixel(matches_g, grid_shape, true_shape):
    """Scale 2D-2D matches from descriptor-grid resolution to full image pixel coordinates."""
    Hg, Wg = grid_shape
    H, W = true_shape
    xy = matches_g.astype(np.float32)
    xy *= np.float32([W / Wg, H / Hg])
    return np.floor(xy).astype(np.int32)


def morphology_cleanup(mask, ksize):
    """Erosion then dilation (opening) to remove stray single-pixel matches, as in the paper."""
    import cv2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return cleaned > 0


def colorize(cov_map):
    """Normalize covisibility counts 0..(window_size-1) to 0..255 grayscale and jet color.

    Returns:
        viz: uint8 grayscale (0..255), brighter = higher covisibility
        jet_rgb: uint8 RGB jet colormap visualization
    """
    import cv2
    counts = cov_map.astype(np.float32)
    if counts.max() > 0:
        counts = (counts - counts.min()) / (counts.max() - counts.min())
    viz = (counts * 255).astype(np.uint8)
    jet_rgb = cv2.applyColorMap(viz, cv2.COLORMAP_JET)[:, :, ::-1]  # BGR -> RGB
    return viz, jet_rgb


def process_window(window_indices, all_images, all_image_paths,
                   model, device, args, window_out, window_id):
    """Compute covisibility maps for one time window of images.

    Args:
        window_indices: list of int indices into all_images / all_image_paths
        all_images: list of preprocessed image dicts (all frames)
        all_image_paths: list of file paths (all frames)
        model: MASt3R model
        device: torch device
        args: parsed arguments
        window_out: output directory path for this window
        window_id: integer window id (0-based)
    """
    nw = len(window_indices)
    window_images = [all_images[i] for i in window_indices]
    window_paths = [all_image_paths[i] for i in window_indices]

    # per-view covisibility maps, sized as the preprocessed (true) image shape
    cov_maps = []
    img_for_viz = []
    for im in window_images:
        H, W = tuple(im["true_shape"][0])
        cov_maps.append(np.zeros((H, W), dtype=np.int32))
        img_t = im["img"].squeeze(0).permute(1, 2, 0).cpu().numpy()  # [-1, 1]
        img_for_viz.append(((img_t + 1) / 2 * 255).astype(np.uint8))

    # ---- visibility bookkeeping (if requested) ----
    if args.visibility:
        # per-view pixel -> set of other view indices (local: 0..nw-1)
        vis_viewids = [dict() for _ in range(nw)]
        # pairwise overlap nw x nw: unique pixels of view i seen in view j
        pair_overlap = np.zeros((nw, nw), dtype=np.int64)

    # ---- dense correspondences on all ordered pairs (i,j), i != j, within window --
    pairs = [(window_images[i], window_images[j]) for i in range(nw) for j in range(nw) if i != j]
    print(f"  >> dense correspondences on {len(pairs)} ordered pairs")
    output = inference(pairs, model, device, batch_size=1, verbose=True)
    pred1, pred2 = output["pred1"], output["pred2"]
    all_desc1 = pred1["desc"]           # (B, Hg, Wg, D)
    all_desc2 = pred2["desc"]
    all_conf1 = pred1["desc_conf"]      # (B, Hg, Wg)
    all_conf2 = pred2["desc_conf"]

    pair_idx = 0
    for i in range(nw):
        for j in range(nw):
            if i == j:
                continue
            desc1 = all_desc1[pair_idx].detach()  # (Hg, Wg, D)
            desc2 = all_desc2[pair_idx].detach()
            conf1 = all_conf1[pair_idx].cpu().numpy()  # (Hg, Wg)
            conf2 = all_conf2[pair_idx].cpu().numpy()
            pair_idx += 1

            # dense 2D-2D correspondences: returns (xy1 @view i, xy2 @view j, conf)
            corres = extract_correspondences_nonsym(
                desc1, desc2, conf1, conf2,
                subsample=args.subsample, device=device, pixel_tol=0)
            xy1, xy2, conf = corres
            xy1 = xy1.cpu().numpy()
            conf = conf.cpu().numpy()

            # keep confident matches
            keep = conf >= args.conf_thr
            xy1 = xy1[keep]
            if len(xy1) == 0:
                continue

            # P(C^j_i): set of pixel coords in view i that matched view j
            xy1 = np.unique(xy1, axis=0)
            H_i, W_i = cov_maps[i].shape
            valid = (xy1[:, 0] >= 0) & (xy1[:, 0] < W_i) & (xy1[:, 1] >= 0) & (xy1[:, 1] < H_i)
            xy1 = xy1[valid]
            cov_maps[i][xy1[:, 1], xy1[:, 0]] += 1

            if args.visibility:
                pair_overlap[i, j] = len(xy1)
                vids_i = vis_viewids[i]
                for x, y in zip(xy1[:, 0].tolist(), xy1[:, 1].tolist()):
                    key = (x, y)
                    if key in vids_i:
                        vids_i[key].add(j)
                    else:
                        vids_i[key] = {j}

    # ---- morphological refinement ----------------------------------
    if not args.no_morpho:
        for i in range(nw):
            mask = cov_maps[i] >= 1
            cleaned = morphology_cleanup(mask, args.dilate_k)
            cov_maps[i] = np.where(cleaned, cov_maps[i], 0)

    # ---- save ------------------------------------------------------
    import cv2
    os.makedirs(window_out, exist_ok=True)

    for k, (cov_map, img) in enumerate(zip(cov_maps, img_for_viz)):
        global_idx = window_indices[k]  # original frame index in the full sequence
        out_path = path.join(window_out, f"covisibility_map_{global_idx:03d}.png")
        viz, jet_rgb = colorize(cov_map)
        cv2.imwrite(out_path, viz)

        if args.overlay:
            H, W = cov_map.shape
            src = cv2.resize(img, (W, H))
            combined = (src.astype(np.float32) * 0.6 + jet_rgb.astype(np.float32) * 0.4).astype(np.uint8)
            cv2.imwrite(path.splitext(out_path)[0] + "_overlay.png", combined[:, :, ::-1])

    # ---- per-window stats ------------------------------------------
    stats = {
        "window_id": window_id,
        "window_range": [window_indices[0], window_indices[-1]],
        "n_views": nw,
        "min_count": [int(c.min()) for c in cov_maps],
        "max_count": [int(c.max()) for c in cov_maps],
        "image_size": [tuple(int(s) for s in im["true_shape"][0]) for im in window_images],
        "images": window_paths,
    }
    with open(path.join(window_out, "covisibility_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # ---- pixel-level visibility + pairwise overlap (--visibility) ---
    if args.visibility:
        for k in range(nw):
            global_idx = window_indices[k]
            vids = vis_viewids[k]
            entries = [
                {"x": x, "y": y, "count": len(s), "views_local": sorted(s)}
                for (x, y), s in vids.items()
            ]
            entries.sort(key=lambda e: e["count"], reverse=True)
            with open(path.join(window_out, f"visibility_{global_idx:03d}.json"), "w") as f:
                json.dump({"view": global_idx, "window_view": k,
                           "n_matched_pixels": len(entries),
                           "pixels": entries}, f, indent=2)

        pair_json = {"window_id": window_id, "n_views": nw,
                     "images": window_paths,
                     "overlap": pair_overlap.tolist()}
        with open(path.join(window_out, "pairwise_overlap.json"), "w") as f:
            json.dump(pair_json, f, indent=2)
        with open(path.join(window_out, "pairwise_overlap.txt"), "w") as f:
            header = " ".join(f"{'v'+str(k):>8}" for k in range(nw))
            f.write("pairwise unique-pixel overlap (view i rows, view j cols)\n")
            f.write(f"            {header}\n")
            for i in range(nw):
                row = " ".join(f"{v:>8}" for v in pair_overlap[i])
                f.write(f"view {i:<6} {row}\n")


def main():
    parser = get_argparser()
    args = parser.parse_args()

    device = "cuda" if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
    print(f">> device = {device}")

    # ---- model -----------------------------------------------------
    weights = args.weights if args.weights is not None else "naver/" + args.model_name
    print(f">> loading MASt3R model from {weights}")
    model = AsymmetricMASt3R.from_pretrained(weights).to(device)
    model.eval()

    # ---- images ----------------------------------------------------
    images_list = get_image_list(args.images,
                                  camera=args.camera,
                                  max_frames=args.max_frames)
    print(f">> found {len(images_list)} images")
    assert len(images_list) > 1, "need at least 2 images to build a covisibility map"
    images = load_images(images_list, size=args.image_size)

    n = len(images)
    os.makedirs(args.output, exist_ok=True)

    # ---- windowing -------------------------------------------------
    if args.window_size is not None:
        S = args.window_size
        num_windows = (n + S - 1) // S
        print(f">> window_size={S}: splitting {n} frames into {num_windows} window(s)")

        for w in range(num_windows):
            start = w * S
            end = min(start + S, n)
            window_indices = list(range(start, end))
            nw = len(window_indices)

            if nw < 2:
                print(f"\n  >> Window {w}: only {nw} image(s) [frames {start}..{end-1}], "
                      f"skipping (need >= 2 for covisibility)")
                continue

            print(f"\n{'=' * 60}")
            print(f"  Window {w}: frames [{start} .. {end-1}]  ({nw} images)")
            print(f"{'=' * 60}")

            window_out = path.join(args.output, f"window_{w:03d}")
            process_window(window_indices, images, images_list,
                           model, device, args, window_out, w)
            print(f"  >> saved {nw} covisibility maps to {window_out}")

        # ---- global summary across all windows ---------------------
        global_stats = {
            "n_total_frames": n,
            "window_size": S,
            "num_windows": num_windows,
            "images": images_list,
        }
        with open(path.join(args.output, "covisibility_stats.json"), "w") as f:
            json.dump(global_stats, f, indent=2)

        print(f"\n>> all {num_windows} windows saved to {args.output}/")

    else:
        # ---- original behavior: all frames in one group ------------
        print(f"\n{'=' * 60}")
        print(f"  All frames: [0 .. {n-1}]  ({n} images)")
        print(f"{'=' * 60}")

        process_window(list(range(n)), images, images_list,
                       model, device, args, args.output, 0)
        print(f"\n>> saved {n} covisibility maps to {args.output}")


if __name__ == "__main__":
    main()
