"""3D稠密可见性分析 — 方法一：LiDAR点云插值生成稠密深度图
流程：
  1. 将每帧的 LiDAR 点云投影到该帧图像 → 稀疏深度采样
  2. 插值 → 稠密深度图 (每像素都有深度)
  3. 对每像素用深度+内参反投影到3D世界坐标
  4. 投影到所有帧统计可见性
  5. 生成热力图
"""
import cv2
import numpy as np
import os
from scipy.interpolate import griddata

# ========== 路径 ==========
base_dir = "/home/djhuai/zuo/mobicom/dggt_2/dggt/data/nuscenes/processed_10Hz/mini/009/"
image_dir = os.path.join(base_dir, "images/")
intr_dir = os.path.join(base_dir, "intrinsics/")
extr_dir = os.path.join(base_dir, "extrinsics/")
lidar_dir = os.path.join(base_dir, "lidar/")
lidar_pose_dir = os.path.join(base_dir, "lidar_pose/")
output_dir = os.path.join(base_dir, "visibility_3d_dense/")
os.makedirs(output_dir, exist_ok=True)

N_FRAMES = 10
IMG_H, IMG_W = 900, 1600

# ========== 1. 加载内参 ==========
intr_raw = np.loadtxt(os.path.join(intr_dir, "0.txt"))
fx, fy, cx, cy = intr_raw[0], intr_raw[1], intr_raw[2], intr_raw[3]
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])
print(f"内参: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

# ========== 2. 加载外参 ==========
def load_extr(scene_id, frame=0):
    path = os.path.join(extr_dir, f"{scene_id:03d}_{frame}.txt")
    return np.loadtxt(path).reshape(4, 4)  # T_cam_world

def get_world_to_cam(T_cam_world):
    R = T_cam_world[:3, :3]
    t = T_cam_world[:3, 3]
    T_world_cam = np.eye(4)
    T_world_cam[:3, :3] = R.T
    T_world_cam[:3, 3] = -R.T @ t
    return T_world_cam

frame_info = []
for i in range(N_FRAMES):
    T_cam_world = load_extr(i, 0)
    T_world_cam = get_world_to_cam(T_cam_world)
    P = K @ T_world_cam[:3, :4]
    frame_info.append({
        "id": i,
        "name": f"{i:03d}_0",
        "T_cam_world": T_cam_world,
        "T_world_cam": T_world_cam,
        "P": P,
    })

# ========== 3. 加载 LiDAR 点云 ==========
def load_lidar_points(scene_id):
    lidar_file = os.path.join(lidar_dir, f"{scene_id:03d}.bin")
    lidar_pose_file = os.path.join(lidar_pose_dir, f"{scene_id:03d}.txt")
    points_raw = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
    T_lidar_world = np.loadtxt(lidar_pose_file).reshape(4, 4)
    xyz_lidar = points_raw[:, :3].T
    xyz_lidar_h = np.vstack([xyz_lidar, np.ones((1, xyz_lidar.shape[1]))])
    xyz_world = (T_lidar_world @ xyz_lidar_h)[:3, :].T
    return xyz_world, points_raw[:, 3]

# ========== 4. 投影函数 ==========
def project_points(points_3d, P, h, w):
    N = points_3d.shape[0]
    X_h = np.hstack([points_3d, np.ones((N, 1))])
    x_proj = (P @ X_h.T).T
    depth = x_proj[:, 2]
    x_px = x_proj[:, 0] / np.maximum(depth, 1e-10)
    y_px = x_proj[:, 1] / np.maximum(depth, 1e-10)
    visible = (depth > 0) & (x_px >= 0) & (x_px < w) & (y_px >= 0) & (y_px < h)
    return visible, x_px, y_px, depth

# ========== 5. 插值生成稠密深度图 ==========
def create_dense_depth(lidar_points_world, P, h, w):
    """将世界坐标系下的LiDAR点云投影到图像，插值生成稠密深度图"""
    visible, xs, ys, depths = project_points(lidar_points_world, P, h, w)
    idx = np.where(visible)[0]

    if len(idx) < 50:
        print(f"    警告: 只有 {len(idx)} 个可见LiDAR点，无法有效插值")
        return np.zeros((h, w), dtype=np.float32), 0

    # 稀疏深度采样：取可见点的像素坐标和深度值
    px = xs[idx].astype(np.float32)
    py = ys[idx].astype(np.float32)
    pz = depths[idx].astype(np.float32)

    # 用 griddata 插值到所有像素
    yi, xi = np.mgrid[:h, :w]

    # 先尝试线性插值，对超出边界的点用最近邻
    depth_dense = griddata(
        (py, px), pz, (yi, xi),
        method='linear', fill_value=np.nan
    )

    # 对 NaN 区域用最近邻填充
    nan_mask = np.isnan(depth_dense)
    if nan_mask.any():
        depth_dense[nan_mask] = griddata(
            (py, px), pz, (yi[nan_mask], xi[nan_mask]),
            method='nearest'
        )

    return depth_dense, len(idx)

def pixel_to_world(depth_map, K, T_cam_world):
    """将稠密深度图反投影到世界坐标系
    返回: Nx3 array of world coordinates, (h*w, 3)
    以及每个点对应的像素坐标 (h*w, 2)
    """
    h, w = depth_map.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 生成像素网格
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u_flat = u.ravel().astype(np.float32)
    v_flat = v.ravel().astype(np.float32)
    d_flat = depth_map.ravel().astype(np.float32)

    # 只处理有有效深度的像素
    valid = d_flat > 0
    u_v = u_flat[valid]
    v_v = v_flat[valid]
    d_v = d_flat[valid]

    # 像素→相机坐标系 (z forward, x right, y down)
    x_cam = (u_v - cx) * d_v / fx
    y_cam = (v_v - cy) * d_v / fy
    z_cam = d_v

    # 相机坐标系下的点 (N, 3)
    cam_pts = np.stack([x_cam, y_cam, z_cam], axis=1)

    # 添加齐次坐标 → 世界坐标系
    cam_pts_h = np.hstack([cam_pts, np.ones((cam_pts.shape[0], 1))])
    world_pts = (T_cam_world @ cam_pts_h.T).T[:, :3]

    # 返回世界坐标和对应的像素坐标
    pixel_coords = np.stack([u_v, v_v], axis=1)

    return world_pts, pixel_coords, valid

# ========== 6. 主循环：对每帧做稠密可见性分析 ==========
print(f"\n{'='*60}")
print(f"稠密 LiDAR 插值 + 跨帧可见性分析")
print(f"{'='*60}")

for ref_idx in range(N_FRAMES):
    ref_name = f"{ref_idx:03d}_0"
    print(f"\n▶ 处理 {ref_name}...")

    # 加载 LiDAR
    lidar_pts_world, _ = load_lidar_points(ref_idx)
    print(f"   加载 LiDAR: {len(lidar_pts_world)} 点")

    # 投影到参考帧并插值生成稠密深度图
    P_ref = frame_info[ref_idx]["P"]
    depth_dense, n_sparse = create_dense_depth(lidar_pts_world, P_ref, IMG_H, IMG_W)
    coverage = (depth_dense > 0).sum() / (IMG_H * IMG_W) * 100
    print(f"   稀疏采样: {n_sparse} 点 → 稠密覆盖: {coverage:.1f}%")

    # 保存深度图
    depth_vis = (depth_dense / depth_dense.max() * 255).astype(np.uint8) if depth_dense.max() > 0 else np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
    cv2.imwrite(os.path.join(output_dir, f"depth_{ref_name}.png"), depth_color)

    # 反投影到世界坐标系
    T_cam_world_ref = frame_info[ref_idx]["T_cam_world"]
    world_pts, pixel_coords, valid_mask = pixel_to_world(depth_dense, K, T_cam_world_ref)

    n_valid = world_pts.shape[0]
    print(f"   有效3D点: {n_valid}")

    # 投影到所有帧统计可见性
    visibility_counts = np.zeros(n_valid, dtype=np.int32)

    for fi, finfo in enumerate(frame_info):
        P = finfo["P"]
        visible, xs, ys, depths = project_points(world_pts, P, IMG_H, IMG_W)
        visibility_counts += visible.astype(np.int32)

    # 统计
    avg_seen = visibility_counts.mean()
    all_seen = (visibility_counts == N_FRAMES).sum()
    zero_seen = (visibility_counts == 0).sum()
    print(f"   平均被看到帧数: {avg_seen:.2f}")
    print(f"   被所有{N_FRAMES}帧看到: {all_seen} ({all_seen/n_valid*100:.1f}%)")
    print(f"   被0帧看到: {zero_seen} ({zero_seen/n_valid*100:.1f}%)")

    # ---- 生成热力图叠加 ----
    img_ref = cv2.imread(os.path.join(image_dir, f"{ref_name}.jpg"))
    if img_ref is None:
        img_ref = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

    # 将可见性计数映射回像素位置
    pixel_vis = np.zeros((IMG_H, IMG_W), dtype=np.int32)
    for k in range(n_valid):
        u, v = int(pixel_coords[k, 0]), int(pixel_coords[k, 1])
        if 0 <= u < IMG_W and 0 <= v < IMG_H:
            if visibility_counts[k] > pixel_vis[v, u]:
                pixel_vis[v, u] = visibility_counts[k]

    # 有效像素掩码
    mask_valid = pixel_vis > 0

    # ---- 高/低重复像素统计 ----
    HIGH_THRESH = int(N_FRAMES * 0.7)  # ≥70% 帧数视为高重复
    LOW_THRESH = int(N_FRAMES * 0.3)   # ≤30% 帧数视为低重复

    mask_high = pixel_vis >= HIGH_THRESH
    mask_low = (pixel_vis > 0) & (pixel_vis <= LOW_THRESH)

    high_count = mask_high.sum()
    low_count = mask_low.sum()
    total_valid = mask_valid.sum()
    high_pct = high_count / total_valid * 100 if total_valid > 0 else 0
    low_pct = low_count / total_valid * 100 if total_valid > 0 else 0

    print(f"   高重复(≥{HIGH_THRESH}帧): {high_count} 点 ({high_pct:.1f}%)")
    print(f"   低重复(≤{LOW_THRESH}帧): {low_count} 点 ({low_pct:.1f}%)")

    # 归一化
    vis_norm = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    max_val = 1
    if mask_valid.any():
        max_val = pixel_vis.max()
        vis_norm[mask_valid] = (pixel_vis[mask_valid] / max_val * 255).astype(np.uint8)

    # 热力图叠加
    heatmap = cv2.applyColorMap(vis_norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_ref, 0.5, heatmap, 0.5, 0)

    # 文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(overlay, f"{ref_name} | LiDAR插值稠密3D可见性",
                (10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(overlay, f"有效3D点: {n_valid}({coverage:.0f}%) | 平均被见: {avg_seen:.1f}帧",
                (10, 60), font, 0.6, (255, 255, 255), 2)
    cv2.putText(overlay, f"RED=高可见(最多{max_val}帧) BLUE=低可见",
                (10, 85), font, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, f"高重复≥{HIGH_THRESH}帧: {high_count}({high_pct:.1f}%)  低重复≤{LOW_THRESH}帧: {low_count}({low_pct:.1f}%)",
                (10, 110), font, 0.5, (255, 255, 255), 1)

    cv2.imwrite(os.path.join(output_dir, f"heatmap_{ref_name}.png"), overlay)
    print(f"   已保存: heatmap_{ref_name}.png")

    # 也保存纯可见性图（无原图叠加，更容易看清）
    heatmap_only = heatmap.copy()
    heatmap_only[~mask_valid] = 0
    cv2.imwrite(os.path.join(output_dir, f"heatmap_only_{ref_name}.png"), heatmap_only)

# ========== 7. 组合大图 ==========
print(f"\n生成组合大图...")
scale = 500 / IMG_W
disp_w, disp_h = 500, int(IMG_H * scale)

rows = []
for row_idx in range(2):
    row_imgs = []
    for col_idx in range(5):
        idx = row_idx * 5 + col_idx
        path = os.path.join(output_dir, f"heatmap_{idx:03d}_0.png")
        overlay = cv2.imread(path)
        if overlay is not None:
            disp = cv2.resize(overlay, (disp_w, disp_h))
        else:
            disp = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
        row_imgs.append(disp)
    rows.append(np.hstack(row_imgs))

composite = np.vstack(rows)
cv2.imwrite(os.path.join(output_dir, "all_heatmaps.png"), composite)
print(f"  已保存: all_heatmaps.png")

# 也保存深度图组合
rows_d = []
for row_idx in range(2):
    row_imgs = []
    for col_idx in range(5):
        idx = row_idx * 5 + col_idx
        path = os.path.join(output_dir, f"depth_{idx:03d}_0.png")
        d_img = cv2.imread(path)
        if d_img is not None:
            disp = cv2.resize(d_img, (disp_w, disp_h))
        else:
            disp = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
        row_imgs.append(disp)
    rows_d.append(np.hstack(row_imgs))

composite_d = np.vstack(rows_d)
cv2.imwrite(os.path.join(output_dir, "all_depths.png"), composite_d)

print(f"\n{'='*60}")
print(f"✅ 所有结果已保存至: {output_dir}/")
print(f"  每帧热力图: heatmap_000_0.png ~ heatmap_009_0.png")
print(f"  每帧深度图: depth_000_0.png ~ depth_009_0.png")
print(f"  组合大图: all_heatmaps.png, all_depths.png")
print(f"{'='*60}")