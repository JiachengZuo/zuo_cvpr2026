"""
Generate co-visibility GT data for training.

Pipeline:
  1. For each frame (front view only), load LiDAR point cloud
  2. Project LiDAR to camera, interpolate to dense depth map
  3. Backproject to world coordinates → dense 3D position map per pixel
  4. For each sliding window of S consecutive frames, compute cross-frame visibility
  5. Each pixel's visibility = how many of the S frames can see it
  6. Save per-frame .npy files in <scene>/visibility_counts/

Usage:
    # Generate for all scenes in mini dataset (S=4 consecutive frames)
    python tool/generate_covisibility_gt.py \
        --data_dir data/nuscenes/processed_10Hz/mini \
        --sequence_length 4 \
        --views 1

    # Single scene (for testing)
    python tool/generate_covisibility_gt.py \
        --data_dir data/nuscenes/processed_10Hz/mini \
        --scenes 009
"""

import argparse
import numpy as np
import os
import sys
from scipy.interpolate import griddata


def parse_args():
    parser = argparse.ArgumentParser(description="Generate co-visibility GT")
    parser.add_argument(
        "--data_dir", type=str,
        default="data/nuscenes/processed_10Hz/mini",
        help="Root directory containing scene folders"
    )
    parser.add_argument(
        "--scenes", type=str, nargs="*", default=None,
        help="Scene names to process (e.g. 007 008 009). Default: all scenes in data_dir."
    )
    parser.add_argument(
        "--sequence_length", type=int, default=4,
        help="Window size S for co-visibility computation"
    )
    parser.add_argument(
        "--views", type=int, default=1,
        help="Number of views (1 = front view only)"
    )
    parser.add_argument(
        "--cache_position_maps", action="store_true", default=True,
        help="Cache dense position maps to position_maps/ directory"
    )
    parser.add_argument(
        "--force_recompute", action="store_true", default=False,
        help="Recompute position maps even if cached"
    )
    parser.add_argument(
        "--min_lidar_points", type=int, default=100,
        help="Minimum LiDAR points required for valid interpolation"
    )
    return parser.parse_args()


def load_intrinsics(intr_path):
    """Load camera intrinsics. Returns K matrix (3x3)."""
    intr_raw = np.loadtxt(intr_path)
    fx, fy, cx, cy = intr_raw[0], intr_raw[1], intr_raw[2], intr_raw[3]
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])
    return K


def load_extrinsic(extr_path):
    """Load camera extrinsic (T_cam_world, 4x4)."""
    return np.loadtxt(extr_path).reshape(4, 4)


def get_world_to_cam(T_cam_world):
    """Convert T_cam_world to T_world_cam."""
    R = T_cam_world[:3, :3]
    t = T_cam_world[:3, 3]
    T_world_cam = np.eye(4)
    T_world_cam[:3, :3] = R.T
    T_world_cam[:3, 3] = -R.T @ t
    return T_world_cam


def load_lidar_points(lidar_path, lidar_pose_path):
    """
    Load LiDAR points and transform to world coordinates.
    Returns: world_xyz (N,3), intensities (N,)
    """
    if not os.path.exists(lidar_path) or not os.path.exists(lidar_pose_path):
        return None, None
    points_raw = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    T_lidar_world = np.loadtxt(lidar_pose_path).reshape(4, 4)
    xyz_lidar = points_raw[:, :3].T  # (3, N)
    xyz_lidar_h = np.vstack([xyz_lidar, np.ones((1, xyz_lidar.shape[1]))])
    xyz_world = (T_lidar_world @ xyz_lidar_h)[:3, :].T  # (N, 3)
    return xyz_world, points_raw[:, 3]


def project_points(points_3d, P, h, w):
    """
    Project 3D world points to image plane.
    Returns: visible (N,), x_px (N,), y_px (N,), depth (N,)
    """
    N = points_3d.shape[0]
    X_h = np.hstack([points_3d, np.ones((N, 1))])
    x_proj = (P @ X_h.T).T  # (N, 3)
    depth = x_proj[:, 2]
    x_px = x_proj[:, 0] / np.maximum(depth, 1e-10)
    y_px = x_proj[:, 1] / np.maximum(depth, 1e-10)
    visible = (depth > 0) & (x_px >= 0) & (x_px < w) & (y_px >= 0) & (y_px < h)
    return visible, x_px, y_px, depth


def create_dense_depth(lidar_points_world, P, h, w, min_points=100):
    """
    Project LiDAR points to camera, interpolate to dense depth map.
    Returns: depth_dense (h, w), n_sparse_points, or (zeros, 0) if too few points.
    """
    visible, xs, ys, depths = project_points(lidar_points_world, P, h, w)
    idx = np.where(visible)[0]

    if len(idx) < min_points:
        return np.zeros((h, w), dtype=np.float32), 0

    # Sparse depth samples at LiDAR hit locations
    px = xs[idx].astype(np.float32)
    py = ys[idx].astype(np.float32)
    pz = depths[idx].astype(np.float32)

    # Grid for all pixels
    yi, xi = np.mgrid[:h, :w]

    # Linear interpolation first
    depth_dense = griddata(
        (py, px), pz, (yi, xi),
        method='linear', fill_value=np.nan
    )

    # Fill NaN regions with nearest-neighbor
    nan_mask = np.isnan(depth_dense)
    if nan_mask.any():
        depth_dense[nan_mask] = griddata(
            (py, px), pz, (yi[nan_mask], xi[nan_mask]),
            method='nearest'
        )

    return depth_dense, len(idx)


def depth_to_world_positions(depth_map, K, T_cam_world):
    """
    Backproject dense depth map to world coordinates.
    Returns: positions (h, w, 3) — world xyz per pixel, valid_mask (h, w)
    """
    h, w = depth_map.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Pixel grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u_f = u.astype(np.float32)
    v_f = v.astype(np.float32)
    d_f = depth_map.astype(np.float32)

    valid = d_f > 0

    # Camera coordinates
    x_cam = np.zeros((h, w), dtype=np.float32)
    y_cam = np.zeros((h, w), dtype=np.float32)
    z_cam = np.zeros((h, w), dtype=np.float32)

    x_cam[valid] = (u_f[valid] - cx) * d_f[valid] / fx
    y_cam[valid] = (v_f[valid] - cy) * d_f[valid] / fy
    z_cam[valid] = d_f[valid]

    # To world coordinates
    world_positions = np.zeros((h, w, 3), dtype=np.float32)
    cam_pts = np.stack([x_cam[valid], y_cam[valid], z_cam[valid],
                         np.ones(valid.sum())], axis=0)
    world_pts = (T_cam_world @ cam_pts).T[:, :3]
    world_positions[valid, :] = world_pts.astype(np.float32)

    return world_positions, valid


def compute_cross_frame_visibility(position_maps, extrinsics, K,
                                    img_h, img_w, sequence_length,
                                    target_h=None, target_w=None,
                                    batch_size=50000):
    """
    Compute per-pixel visibility counts for a sliding window of consecutive frames.

    For each frame i, counts how many frames j in [i, i+1, ..., i+S-1] can see
    each pixel of frame i.

    Args:
        position_maps: list of (h, w, 3) arrays — world positions per frame
        extrinsics: list of T_cam_world (4x4) per frame
        K: (3x3) — camera intrinsic matrix
        img_h, img_w: original image dimensions
        sequence_length: S, window size
        target_h, target_w: output resolution (if None, use img_h, img_w)
        batch_size: number of points to project at once

    Returns:
        visibility_counts: (num_frames, target_h, target_w) float32 array
    """
    n_frames = len(position_maps)
    out_h = target_h if target_h is not None else img_h
    out_w = target_w if target_w is not None else img_w

    visibility_counts = np.zeros((n_frames, out_h, out_w), dtype=np.float32)

    # Precompute projection matrices for each frame
    P_matrices = []
    for i in range(n_frames):
        T_world_cam = get_world_to_cam(extrinsics[i])
        P_matrices.append(K @ T_world_cam[:3, :4])

    from PIL import Image

    for ref_idx in range(n_frames):
        pos_map = position_maps[ref_idx]
        if pos_map is None:
            if ref_idx % 20 == 0:
                print(f"    Frame {ref_idx:03d}: skipped (no position map)")
            continue

        # Resize position map to output resolution if needed
        if out_h != img_h or out_w != img_w:
            pos_h, pos_w = pos_map.shape[:2]
            pos_flat = pos_map.reshape(-1, 3)
            # Use PIL for fast resize of each channel
            pos_resized = np.zeros((out_h, out_w, 3), dtype=np.float32)
            for c in range(3):
                ch = Image.fromarray(pos_map[:, :, c])
                ch = ch.resize((out_w, out_h), Image.Resampling.BILINEAR)
                pos_resized[:, :, c] = np.array(ch, dtype=np.float32)
            pos_map = pos_resized

        # Valid pixels in reference frame
        valid_mask = (pos_map[:, :, 0] != 0) | (pos_map[:, :, 1] != 0) | (pos_map[:, :, 2] != 0)
        valid_indices = np.where(valid_mask)
        valid_pts = pos_map[valid_mask]  # (N_valid, 3)
        n_valid = valid_pts.shape[0]

        if n_valid == 0:
            if ref_idx % 20 == 0:
                print(f"    Frame {ref_idx:03d}: no valid points")
            continue

        # Count visibility for each pixel across frames [ref_idx, ..., ref_idx+S-1]
        frame_count = np.zeros(n_valid, dtype=np.float32)

        end_idx = min(ref_idx + sequence_length, n_frames)

        # Process points in batches to avoid large memory allocations
        n_batches = (n_valid + batch_size - 1) // batch_size
        for bi in range(n_batches):
            b_start = bi * batch_size
            b_end = min(b_start + batch_size, n_valid)
            batch_pts = valid_pts[b_start:b_end]

            for j in range(ref_idx, end_idx):
                visible, _, _, _ = project_points(batch_pts, P_matrices[j], out_h, out_w)
                frame_count[b_start:b_end] += visible.astype(np.float32)

        # Write back to pixel positions
        vis_2d = np.zeros((out_h, out_w), dtype=np.float32)
        vis_2d[valid_indices[0], valid_indices[1]] = frame_count
        visibility_counts[ref_idx] = vis_2d

        if ref_idx % 20 == 0:
            vis_nonzero = vis_2d[vis_2d > 0]
            avg_vis = vis_nonzero.mean() if len(vis_nonzero) > 0 else 0
            print(f"    Frame {ref_idx:03d}/{n_frames}: avg_vis={avg_vis:.2f}, "
                  f"n_valid={n_valid}, batch_count={n_batches}")

    return visibility_counts


def compute_position_map_for_frame(frame_idx, scene_dir, cam_idx, img_h, img_w):
    """
    Compute dense world position map for a single frame.
    Returns: positions (h, w, 3), valid count, or (None, 0) on failure.
    """
    lidar_path = os.path.join(scene_dir, "lidar", f"{frame_idx:03d}.bin")
    lidar_pose_path = os.path.join(scene_dir, "lidar_pose", f"{frame_idx:03d}.txt")
    extr_path = os.path.join(scene_dir, "extrinsics", f"{frame_idx:03d}_{cam_idx}.txt")
    intr_path = os.path.join(scene_dir, "intrinsics", f"{cam_idx}.txt")

    if not all(os.path.exists(p) for p in [lidar_path, lidar_pose_path, extr_path, intr_path]):
        return None, 0

    # Load data
    lidar_pts_world, _ = load_lidar_points(lidar_path, lidar_pose_path)
    if lidar_pts_world is None or len(lidar_pts_world) < 50:
        return None, 0

    T_cam_world = load_extrinsic(extr_path)
    K = load_intrinsics(intr_path)

    # Project LiDAR → interpolate dense depth
    T_world_cam = get_world_to_cam(T_cam_world)
    P = K @ T_world_cam[:3, :4]
    depth_dense, n_sparse = create_dense_depth(lidar_pts_world, P, img_h, img_w)

    if n_sparse < 50:
        return None, 0

    coverage = (depth_dense > 0).sum() / (img_h * img_w) * 100

    # Backproject to world
    world_positions, valid = depth_to_world_positions(depth_dense, K, T_cam_world)

    return world_positions, coverage


def main():
    args = parse_args()

    data_dir = args.data_dir
    sequence_length = args.sequence_length
    cam_idx = 0  # Front view only

    # Discover scenes
    if args.scenes:
        scene_names = args.scenes
    else:
        scene_names = sorted(
            [d for d in os.listdir(data_dir)
             if os.path.isdir(os.path.join(data_dir, d))]
        )

    print(f"{'='*60}")
    print(f"Co-Visibility GT Generation")
    print(f"Data dir: {data_dir}")
    print(f"Scenes: {scene_names}")
    print(f"Sequence length: {sequence_length}")
    print(f"Camera view: {cam_idx} (front)")
    print(f"{'='*60}")

    for scene_name in scene_names:
        scene_dir = os.path.join(data_dir, scene_name)
        if not os.path.isdir(scene_dir):
            print(f"\n⚠ Scene {scene_name}: not found, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Processing scene: {scene_name}")
        print(f"{'='*60}")

        # ---- Discover frames ----
        image_dir = os.path.join(scene_dir, "images")
        if not os.path.isdir(image_dir):
            print(f"  No images directory, skipping")
            continue

        image_files = sorted([f for f in os.listdir(image_dir)
                              if f.endswith(f"_{cam_idx}.jpg") or f.endswith(f"_{cam_idx}.png")])
        if not image_files:
            print(f"  No front-view images found, skipping")
            continue

        # Parse frame indices
        frame_indices = []
        for f in image_files:
            try:
                fid = int(f.split('_')[0])
                frame_indices.append(fid)
            except ValueError:
                continue
        frame_indices = sorted(set(frame_indices))
        n_frames = len(frame_indices)
        print(f"  Found {n_frames} frames (front view)")

        # ---- Get image dimensions from first image ----
        from PIL import Image
        first_img_path = os.path.join(image_dir, f"{frame_indices[0]:03d}_{cam_idx}.jpg")
        if not os.path.exists(first_img_path):
            first_img_path = os.path.join(image_dir, f"{frame_indices[0]:03d}_{cam_idx}.png")
        img_sample = Image.open(first_img_path)
        img_w, img_h = img_sample.size
        print(f"  Image size: {img_w}x{img_h}")

        # ---- Cache directory for position maps ----
        cache_dir = os.path.join(scene_dir, "position_maps")
        if args.cache_position_maps:
            os.makedirs(cache_dir, exist_ok=True)

        # ---- Step 1: Compute/locate dense position maps ----
        print(f"\n  Step 1: Computing dense position maps...")
        position_maps = []
        extrinsics = []
        K = None
        n_valid_frames = 0

        for fi in frame_indices:
            cache_path = os.path.join(cache_dir, f"{fi:03d}_{cam_idx}_pos.npy")
            extr_path = os.path.join(scene_dir, "extrinsics", f"{fi:03d}_{cam_idx}.txt")

            if os.path.exists(cache_path) and not args.force_recompute:
                pos_map = np.load(cache_path)
                # Verify shape matches current resolution
                if pos_map.shape[:2] == (img_h, img_w):
                    position_maps.append(pos_map)
                    if os.path.exists(extr_path):
                        extrinsics.append(load_extrinsic(extr_path))
                    else:
                        extrinsics.append(np.eye(4))
                    if K is None:
                        intr_path = os.path.join(scene_dir, "intrinsics", f"{cam_idx}.txt")
                        if os.path.exists(intr_path):
                            K = load_intrinsics(intr_path)
                    n_valid_frames += 1
                    continue
                else:
                    print(f"    Frame {fi:03d}: cached shape mismatch, recomputing")

            pos_map, coverage = compute_position_map_for_frame(
                fi, scene_dir, cam_idx, img_h, img_w
            )

            if pos_map is not None:
                position_maps.append(pos_map)
                extrinsics.append(load_extrinsic(extr_path))
                if K is None:
                    intr_path = os.path.join(scene_dir, "intrinsics", f"{cam_idx}.txt")
                    K = load_intrinsics(intr_path)
                n_valid_frames += 1

                if args.cache_position_maps:
                    np.save(cache_path, pos_map)

                if fi % 50 == 0 or fi == frame_indices[0]:
                    print(f"    Frame {fi:03d}: {coverage:.1f}% coverage")
            else:
                position_maps.append(None)
                extrinsics.append(np.eye(4))
                print(f"    Frame {fi:03d}: FAILED (insufficient LiDAR points)")

        print(f"  Valid position maps: {n_valid_frames}/{n_frames}")

        if K is None:
            print(f"  ERROR: No intrinsics found, skipping scene")
            continue

        # ---- Step 2: Compute cross-frame visibility ----
        print(f"\n  Step 2: Computing cross-frame visibility...")
        # Use target resolution matching what the model sees (~294×518 for pad mode)
        # This reduces memory from 1.44M to ~152K pixels per frame
        target_h = round(img_h * 518 / max(img_w, img_h) / 14) * 14
        target_w = 518
        print(f"  Output resolution: {target_w}x{target_h}")

        visibility_counts = compute_cross_frame_visibility(
            position_maps, extrinsics, K,
            img_h, img_w, sequence_length,
            target_h=target_h, target_w=target_w,
            batch_size=50000
        )

        # ---- Step 3: Save results ----
        output_dir = os.path.join(scene_dir, "visibility_counts")
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n  Step 3: Saving visibility per frame...")
        for i, fi in enumerate(frame_indices):
            vis_map = visibility_counts[i]  # (H, W)
            out_path = os.path.join(output_dir, f"{fi:03d}_{cam_idx}.npy")
            np.save(out_path, vis_map.astype(np.float32))

            if fi % 50 == 0 or fi == frame_indices[0]:
                avg_vis = vis_map[vis_map > 0].mean() if (vis_map > 0).any() else 0
                max_vis = vis_map.max()
                print(f"    Frame {fi:03d}: avg_vis={avg_vis:.2f}, max_vis={max_vis:.0f}")

        # ---- Summary ----
        total_valid = sum(1 for vm in visibility_counts if vm.max() > 0)
        print(f"\n  ✅ Scene {scene_name} complete!")
        print(f"    Saved {total_valid} visibility maps to {output_dir}/")
        print(f"    Files: {frame_indices[0]:03d}_{cam_idx}.npy ~ {frame_indices[-1]:03d}_{cam_idx}.npy")

    print(f"\n{'='*60}")
    print(f"All scenes processed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
