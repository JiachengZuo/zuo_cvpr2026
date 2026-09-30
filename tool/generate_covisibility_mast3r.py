#!/usr/bin/env python3
"""Generate co-visibility GT using MASt3R dense correspondences.
Based on mast3r/mast3r/covisibility.py (CoMapGS Eq.1).

For each frame i, visibility = count of frames j in [i, i+S) that see
each pixel of frame i. Self-visibility = 1, others determined by
MASt3R pairwise dense matching.

Usage:
    python tool/generate_covisibility_mast3r.py \
        --data_dir data/nuscenes/processed_10Hz/mini --sequence_length 4
"""

import argparse, os, re, sys
import numpy as np
import torch, cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_MAST3R_DIR = "/home/djhuai/zuo/cvpr/mast3r"
if _MAST3R_DIR not in sys.path:
    sys.path.insert(0, _MAST3R_DIR)

import mast3r.utils.path_to_dust3r  # noqa
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import extract_correspondences_nonsym
from dust3r.inference import inference
from dust3r.utils.image import load_images


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--scenes", type=str, nargs="*", default=None)
    p.add_argument("--sequence_length", type=int, default=4)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--model_name", type=str,
                   default="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--subsample", type=int, default=8)
    p.add_argument("--conf_thr", type=float, default=0.0)
    p.add_argument("--dilate_k", type=int, default=3)
    p.add_argument("--chunk_size", type=int, default=20)
    p.add_argument("--max_frames", type=int, default=None)
    return p.parse_args()


def get_images(scene_dir, camera):
    d = os.path.join(scene_dir, "images")
    if not os.path.isdir(d):
        return []
    pat = re.compile(rf"(\d{{3}})_{camera}\.(jpg|jpeg|png)$", re.IGNORECASE)
    return sorted([os.path.join(d, f) for f in os.listdir(d) if pat.search(f)])


def scale_matches(xy_g, Hg, Wg, H, W):
    """Scale descriptor-grid matches to true_shape pixel coords."""
    xy = xy_g.astype(np.float32) * np.float32([W/Wg, H/Hg])
    return np.floor(xy).astype(np.int32)


def morpho(mask, k):
    krn = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, krn) > 0


def resize_map(arr, th, tw):
    if arr.shape == (th, tw):
        return arr
    im = Image.fromarray(arr.astype(np.float32))
    return np.array(im.resize((tw, th), Image.Resampling.NEAREST), dtype=np.float32)


def train_res(oh, ow, ps=14):
    tw, th = 518, round(oh * 518 / ow / ps) * ps
    return th, tw


def extract_matched_pixels(desc1, desc2, conf1, conf2, Hg1, Wg1, mh, mw,
                             subsample, device, conf_thr):
    """Extract unique pixel coordinates in view1 that match view2."""
    cor = extract_correspondences_nonsym(
        desc1, desc2, conf1, conf2,
        subsample=subsample, device=device, pixel_tol=0)
    xy, _, cf = cor
    xy = xy.cpu().numpy(); cf = cf.cpu().numpy()
    xy = xy[cf >= conf_thr]
    if not len(xy):
        return None
    xy = scale_matches(xy, Hg1, Wg1, mh, mw)
    ok = (xy[:,0]>=0) & (xy[:,0]<mw) & (xy[:,1]>=0) & (xy[:,1]<mh)
    xy = xy[ok]
    if not len(xy):
        return None
    return np.unique(xy, axis=0)


def process_scene(scene_dir, scene_name, model, device, args):
    """Process one scene: runs mast3r, builds per-frame covis maps.

    Groups frames into non-overlapping windows of size S.
    All frames in a group share the same reference window.
    E.g. S=4: group0=[0,1,2,3], group1=[4,5,6,7], ...
    Within group0, every frame computes co-visibility against {0,1,2,3}.
    """
    S = args.sequence_length
    cam = args.camera

    paths = get_images(scene_dir, cam)
    if args.max_frames:
        paths = paths[:args.max_frames]
    N = len(paths)
    print(f"  {N} images")
    if N < S:
        return

    # Resolutions
    im0 = Image.open(paths[0])
    ow, oh = im0.size
    th, tw = train_res(oh, ow)
    s = load_images([paths[0]], size=args.image_size, verbose=False)
    mh, mw = tuple(int(x) for x in s[0]["true_shape"][0])
    del s
    print(f"  orig={ow}x{oh}  mast3r={mw}x{mh}  train={tw}x{th}")

    outd = os.path.join(scene_dir, "visibility_counts")
    os.makedirs(outd, exist_ok=True)

    # ---- Non-overlapping groups ----
    num_groups = N // S  # full groups only (drop tail < S)
    group_batch = max(1, args.chunk_size // S)  # groups per inference batch
    print(f"  Groups: {num_groups} (S={S}, batch={group_batch} groups, tail={N - num_groups*S} frames skipped)")

    all_covs = [np.ones((mh, mw), dtype=np.float32) for _ in range(N)]

    groups_done = 0
    while groups_done < num_groups:
        batch_start_g = groups_done
        batch_end_g = min(groups_done + group_batch, num_groups)
        num_batch_groups = batch_end_g - batch_start_g

        # Load all images for this batch of groups
        batch_start_f = batch_start_g * S
        batch_end_f = batch_end_g * S
        B = batch_end_f - batch_start_f  # total frames in batch
        batch_paths = paths[batch_start_f:batch_end_f]
        images = load_images(batch_paths, size=args.image_size, verbose=False)

        # Build pairs: all pairs WITHIN each group (no cross-group pairs)
        pairs = []
        pair_ij = []  # (local_i, local_j) within the batch
        for g in range(num_batch_groups):
            g_off = g * S  # local frame offset of this group within batch
            for i in range(S):
                for j in range(i + 1, S):
                    pairs.append((images[g_off + i], images[g_off + j]))
                    pair_ij.append((g_off + i, g_off + j))

        print(f"  Groups[{batch_start_g}..{batch_end_g-1}]: "
              f"frames[{batch_start_f}..{batch_end_f-1}], {len(pairs)} pairs")

        if not pairs:
            groups_done += num_batch_groups
            continue

        # Inference
        output = inference(pairs, model, device, batch_size=1, verbose=False)
        pd1 = output["pred1"]; pd2 = output["pred2"]

        # Per-frame match pixels (local indices within batch)
        match_px = [{} for _ in range(B)]

        for pid, (li, lj) in enumerate(pair_ij):
            desc1 = pd1["desc"][pid].detach()
            desc2 = pd2["desc"][pid].detach()
            c1 = pd1["desc_conf"][pid].cpu().numpy()
            c2 = pd2["desc_conf"][pid].cpu().numpy()
            Hg1, Wg1 = desc1.shape[:2]
            Hg2, Wg2 = desc2.shape[:2]

            gi = batch_start_f + li  # global index
            gj = batch_start_f + lj

            # li -> lj
            xy1 = extract_matched_pixels(
                desc1, desc2, c1, c2, Hg1, Wg1, mh, mw,
                args.subsample, device, args.conf_thr)
            if xy1 is not None:
                for x, y in zip(xy1[:,0].tolist(), xy1[:,1].tolist()):
                    k = y*mw + x
                    s = match_px[li].get(k)
                    if s is None:
                        match_px[li][k] = {gj}
                    else:
                        s.add(gj)

            # lj -> li
            xy2 = extract_matched_pixels(
                desc2, desc1, c2, c1, Hg2, Wg2, mh, mw,
                args.subsample, device, args.conf_thr)
            if xy2 is not None:
                for x, y in zip(xy2[:,0].tolist(), xy2[:,1].tolist()):
                    k = y*mw + x
                    s = match_px[lj].get(k)
                    if s is None:
                        match_px[lj][k] = {gi}
                    else:
                        s.add(gj)

        del output, pd1, pd2

        # Build covis maps: each frame's window = its OWN group's indices
        for g in range(num_batch_groups):
            g_start_global = batch_start_f + g * S
            group_set = set(range(g_start_global, g_start_global + S))
            for li_rel in range(S):
                li = g * S + li_rel
                gi = g_start_global + li_rel
                cov = np.ones((mh, mw), dtype=np.float32)
                mp = match_px[li]
                for k, vis_j in mp.items():
                    count = len(vis_j & group_set)
                    if count > 0:
                        y, x = divmod(k, mw)
                        cov[y, x] += count
                all_covs[gi] = cov

        torch.cuda.empty_cache()
        groups_done += num_batch_groups

    # ---- Post-processing: morpho, resize, save ----
    print(f"  Post-processing {N} frames...")
    for fi in range(N):
        cov = all_covs[fi]

        # Morphological refinement
        if args.dilate_k > 0:
            msk = cov > 1
            if msk.any():
                cl = morpho(msk, args.dilate_k)
                cov = np.where(cl, cov, 1.0)

        # Resize to training resolution
        if mh != th or mw != tw:
            cov = resize_map(cov, th, tw)

        out_path = os.path.join(outd, f"{fi:03d}_{cam}.npy")
        np.save(out_path, cov.astype(np.float32))

        if fi % 20 == 0 or fi == N-1:
            m = cov
            mt = (m>1).sum()
            print(f"    Frame {fi:03d}: max={m.max():.0f} mean={m[m>0].mean():.2f} "
                  f"matched={mt}/{m.size} ({100*mt/m.size:.1f}%)")

    nf = len([f for f in os.listdir(outd) if f.endswith(f"_{cam}.npy")])
    print(f"  Saved {nf} files to {outd}/")

    # Visualization
    visualize_covisibility(scene_dir, scene_name, cam, S, th, tw)


def visualize_covisibility(scene_dir, scene_name, cam, S, th, tw):
    """Generate visualization images: original | heatmap | overlay for sampled frames."""
    counts_dir = os.path.join(scene_dir, "visibility_counts")
    imgs_dir = os.path.join(scene_dir, "images")
    viz_dir = os.path.join(scene_dir, "visibility_viz")
    os.makedirs(viz_dir, exist_ok=True)

    # Collect all .npy files
    npy_files = sorted([f for f in os.listdir(counts_dir)
                        if f.endswith(f"_{cam}.npy")])
    if not npy_files:
        print("  [viz] No .npy files found, skip.")
        return

    # Sample frames: evenly spaced, up to 8
    N = len(npy_files)
    sample_idx = np.linspace(0, N-1, min(8, N), dtype=int).tolist()

    for fi in sample_idx:
        fname = npy_files[fi]
        cov = np.load(os.path.join(counts_dir, fname))  # [th, tw]

        # Find original image
        img_path = os.path.join(imgs_dir, f"{fi:03d}_{cam}.jpg")
        if not os.path.exists(img_path):
            # try other extensions
            for ext in ['png', 'jpeg']:
                alt = os.path.join(imgs_dir, f"{fi:03d}_{cam}.{ext}")
                if os.path.exists(alt):
                    img_path = alt
                    break
            else:
                continue

        img = cv2.imread(img_path)[:, :, ::-1]  # BGR -> RGB
        img = cv2.resize(img, (tw, th))

        # --- 1. Heatmap ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(img)
        axes[0].set_title(f"Frame {fi:03d} (Original)", fontsize=11)
        axes[0].axis('off')

        # heatmap: blue(1) -> cyan -> green -> red(S), fixed scale across group
        im_h = axes[1].imshow(cov, cmap='jet', vmin=1, vmax=S)
        axes[1].set_title(f"Co-visibility Map (1~{S})", fontsize=11)
        axes[1].axis('off')
        plt.colorbar(im_h, ax=axes[1], fraction=0.046, pad=0.04)

        # --- 3. Overlay ---
        cov_norm = (cov - 1) / max(S - 1, 1)
        heat = plt.cm.jet(cov_norm)[..., :3]
        overlay = (img.astype(np.float32) * 0.5 + heat * 255 * 0.5).astype(np.uint8)
        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay", fontsize=11)
        axes[2].axis('off')

        plt.tight_layout()
        out_path = os.path.join(viz_dir, f"{fi:03d}_{cam}.png")
        plt.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close()

    print(f"  [viz] Saved {len(sample_idx)} images to {viz_dir}/")


def main():
    args = parse_args()
    device = "cuda" if (args.device=="cuda" and torch.cuda.is_available()) else "cpu"
    print(f">> device={device}")

    # Model
    if args.model_name.startswith("naver/") or args.model_name.startswith("/"):
        wt = args.model_name
    else:
        wt = "naver/" + args.model_name
    print(f">> loading {wt}")
    model = AsymmetricMASt3R.from_pretrained(wt).to(device)
    model.eval()

    # Scenes
    scenes = args.scenes or sorted(d for d in os.listdir(args.data_dir)
                                    if os.path.isdir(os.path.join(args.data_dir, d)))
    print(f"Scenes={scenes} S={args.sequence_length} chunk={args.chunk_size}")

    for sn in scenes:
        sd = os.path.join(args.data_dir, sn)
        if not os.path.isdir(sd):
            continue
        print(f"\n=== Scene {sn} ===")
        process_scene(sd, sn, model, device, args)

    print(f"\nDone!")


if __name__ == "__main__":
    main()
