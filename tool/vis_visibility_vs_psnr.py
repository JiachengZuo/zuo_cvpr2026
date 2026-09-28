"""可见性 vs PSNR 相关性分析 + 线图
左侧y轴: 高重复可见区域比例 (≥7帧/70%)
右侧y轴: PSNR (dB)
横轴: Frame Number

数据来源:
  可见性: 从 visibility_3d_dense/ 的每帧输出统计
  PSNR:  从 output_error_analysis_007/error_metrics.csv 读取
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, csv

# ========== 路径 ==========
base_dir = "/home/djhuai/zuo/mobicom/dggt_2/dggt/data/nuscenes/processed_10Hz/mini/009/"
vis_dir = os.path.join(base_dir, "visibility_3d_dense/")
err_csv = "/home/djhuai/zuo/mobicom/dggt_2/dggt/analyze/output_error_analysis_009/error_metrics.csv"
output_dir = os.path.join(base_dir, "visibility_3d_dense/")

N_FRAMES = 10
HIGH_THRESH = 7   # ≥70%

# ========== 1. 读取可见性统计数据 ==========
# 需要从每帧的输出中提取高重复可见比例
# 有两种方式: (a) 重新跑统计 (b) 从 vis_3d_dense.py 的输出日志提取
# 这里用稳妥方式: 直接从 heatmap_only 提取

def extract_visibility_ratio(ref_idx):
    """从保存的 heatmap_only 中提取可见性统计"""
    heatmap_path = os.path.join(vis_dir, f"heatmap_only_{ref_idx:03d}_0.png")
    if not os.path.exists(heatmap_path):
        return 0.0, 0.0

    heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
    if heatmap is None:
        return 0.0, 0.0

    # heatmap_only 中 0=无可见性, 非0值对应可见性计数
    # 但 heatmap_only 存的是归一化后的值 (0-255)，无法还原精确计数
    # 所以改用重新统计的方式

    return None

# 改用重新运行统计的方式: 在 vis_3d_dense.py 中已有统计逻辑
# 直接读取之前脚本输出的数据不可行，我们在 vis_3d_dense.py 的输出中有每个像素的可见性计数吗？
# 没有保存计数原始值。所以这里需要重新计算各帧的高重复比例。

import cv2

def compute_visibility_stats(ref_idx):
    """重新计算第ref_idx帧的可见性统计"""
    # 读取depth map (保存了深度，可以用来推断可见性计数)
    depth_path = os.path.join(vis_dir, f"depth_{ref_idx:03d}_0.png")
    if not os.path.exists(depth_path):
        return 0.0, 0.0, 0.0

    # 用重计算的方式: 从heatmap_only图的像素值映射回计数
    # heatmap_only 使用 normalize: vis_norm = pixel_vis / max_val * 255
    # 因此 pixel_vis ≈ heatmap_val / 255 * max_val
    # 但max_val是每帧独立的，无法直接从heatmap_only反推

    # 最佳方式: 重新运行投影统计
    # 直接复用 vis_3d_dense.py 的逻辑来计算
    return None

# ========== 改用: 从 vis_3d_dense.py 运行时打印的日志提取 ==========
# 从上一轮运行的结果中已有统计:
# 000_0: 高重复(≥7帧): 738651 点 (51.3%)
# 001_0: 高重复(≥7帧): ...
# 需要从日志提取，或者重新计算一份

# ========== 重新计算 ==========
# 复用 vis_3d_dense.py 中的逻辑来计算每帧的高重复比例
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_extr(scene_id, frame=0):
    extr_dir = os.path.join(base_dir, "extrinsics/")
    path = os.path.join(extr_dir, f"{scene_id:03d}_{frame}.txt")
    return np.loadtxt(path).reshape(4, 4)

def get_world_to_cam(T_cam_world):
    R = T_cam_world[:3, :3]
    t = T_cam_world[:3, 3]
    T_world_cam = np.eye(4)
    T_world_cam[:3, :3] = R.T
    T_world_cam[:3, 3] = -R.T @ t
    return T_world_cam

def load_lidar_points(scene_id):
    lidar_dir_ = os.path.join(base_dir, "lidar/")
    lidar_pose_dir_ = os.path.join(base_dir, "lidar_pose/")
    lidar_file = os.path.join(lidar_dir_, f"{scene_id:03d}.bin")
    lidar_pose_file = os.path.join(lidar_pose_dir_, f"{scene_id:03d}.txt")
    points_raw = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
    T_lidar_world = np.loadtxt(lidar_pose_file).reshape(4, 4)
    xyz_lidar = points_raw[:, :3].T
    xyz_lidar_h = np.vstack([xyz_lidar, np.ones((1, xyz_lidar.shape[1]))])
    xyz_world = (T_lidar_world @ xyz_lidar_h)[:3, :].T
    return xyz_world, points_raw[:, 3]

def project_points(points_3d, P, h, w):
    N = points_3d.shape[0]
    X_h = np.hstack([points_3d, np.ones((N, 1))])
    x_proj = (P @ X_h.T).T
    depth = x_proj[:, 2]
    x_px = x_proj[:, 0] / np.maximum(depth, 1e-10)
    y_px = x_proj[:, 1] / np.maximum(depth, 1e-10)
    visible = (depth > 0) & (x_px >= 0) & (x_px < w) & (y_px >= 0) & (y_px < h)
    return visible, x_px, y_px, depth

def create_dense_depth(lidar_points_world, P, h, w):
    from scipy.interpolate import griddata
    visible, xs, ys, depths = project_points(lidar_points_world, P, h, w)
    idx = np.where(visible)[0]
    if len(idx) < 50:
        return np.zeros((h, w), dtype=np.float32), 0
    px = xs[idx].astype(np.float32)
    py = ys[idx].astype(np.float32)
    pz = depths[idx].astype(np.float32)
    yi, xi = np.mgrid[:h, :w]
    depth_dense = griddata((py, px), pz, (yi, xi), method='linear', fill_value=np.nan)
    nan_mask = np.isnan(depth_dense)
    if nan_mask.any():
        depth_dense[nan_mask] = griddata(
            (py, px), pz, (yi[nan_mask], xi[nan_mask]), method='nearest')
    return depth_dense, len(idx)

def pixel_to_world(depth_map, K, T_cam_world):
    h, w = depth_map.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u_flat = u.ravel().astype(np.float32)
    v_flat = v.ravel().astype(np.float32)
    d_flat = depth_map.ravel().astype(np.float32)
    valid = d_flat > 0
    u_v, v_v, d_v = u_flat[valid], v_flat[valid], d_flat[valid]
    x_cam = (u_v - cx) * d_v / fx
    y_cam = (v_v - cy) * d_v / fy
    z_cam = d_v
    cam_pts = np.stack([x_cam, y_cam, z_cam], axis=1)
    cam_pts_h = np.hstack([cam_pts, np.ones((cam_pts.shape[0], 1))])
    world_pts = (T_cam_world @ cam_pts_h.T).T[:, :3]
    pixel_coords = np.stack([u_v, v_v], axis=1)
    return world_pts, pixel_coords, valid

# ========== 加载内参和外参 ==========
intr_raw = np.loadtxt(os.path.join(base_dir, "intrinsics", "0.txt"))
fx, fy, cx, cy = intr_raw[0], intr_raw[1], intr_raw[2], intr_raw[3]
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

frame_info = []
for i in range(N_FRAMES):
    T_cam_world = load_extr(i, 0)
    T_world_cam = get_world_to_cam(T_cam_world)
    P = K @ T_world_cam[:3, :4]
    frame_info.append({"id": i, "T_cam_world": T_cam_world, "P": P})

# ========== 对每帧计算高重复可见比例 ==========
high_ratios = []  # 高重复(≥7帧)像素占总有效像素比

for ref_idx in range(N_FRAMES):
    print(f"  计算帧 {ref_idx:03d}_0 的可见性...")

    # 加载LiDAR并插值
    lidar_pts, _ = load_lidar_points(ref_idx)
    P_ref = frame_info[ref_idx]["P"]
    depth_dense, _ = create_dense_depth(lidar_pts, P_ref, 900, 1600)

    # 反投影到世界
    T_cam_world_ref = frame_info[ref_idx]["T_cam_world"]
    world_pts, pixel_coords, valid_mask = pixel_to_world(depth_dense, K, T_cam_world_ref)
    n_valid = world_pts.shape[0]

    if n_valid == 0:
        high_ratios.append(0.0)
        continue

    # 统计被所有帧看到的次数
    visibility_counts = np.zeros(n_valid, dtype=np.int32)
    for fi, finfo in enumerate(frame_info):
        visible, _, _, _ = project_points(world_pts, finfo["P"], 900, 1600)
        visibility_counts += visible.astype(np.int32)

    # 高重复像素比
    high_count = (visibility_counts >= 7).sum()
    ratio = high_count / n_valid * 100
    high_ratios.append(ratio)
    print(f"    高重复(≥7帧): {high_count}/{n_valid} = {ratio:.1f}%")

# ========== 2. 读取 PSNR 数据 ==========
psnr_values = {}
with open(err_csv, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        pair_id = int(row[0])
        psnr = float(row[1])
        psnr_values[pair_id] = psnr

# PSNR pair_id 到帧号的映射: pair 0 → frame_0, pair 2 → frame_2, ..., pair 9 → frame_9
# 缺 pair_id=1 (对应frame_001_0)
frame_psnr = []
for i in range(N_FRAMES):
    if i in psnr_values:
        frame_psnr.append(psnr_values[i])
    else:
        # pair_id=1 不存在，用前后帧插值
        if i == 1 and 0 in psnr_values and 2 in psnr_values:
            frame_psnr.append((psnr_values[0] + psnr_values[2]) / 2)
        else:
            frame_psnr.append(np.nan)

print(f"\n高重复比例: {high_ratios}")
print(f"PSNR 值:    {frame_psnr}")

# ========== 3. 画图 ==========
fig, ax1 = plt.subplots(figsize=(10, 6))

frames = list(range(N_FRAMES))
x_labels = [f"{i:03d}_0" for i in frames]

# 左侧y轴: 高重复可见比例
color_high = '#E74C3C'
ax1.set_xlabel('Frame', fontsize=13)
ax1.set_ylabel('High Visibility Ratio (≥7/10 frames, %)', color=color_high, fontsize=12)
line1 = ax1.plot(frames, high_ratios, 'o-', color=color_high, linewidth=2.5,
                 markersize=8, label='High Visibility Ratio', zorder=5)
ax1.tick_params(axis='y', labelcolor=color_high)
ax1.set_ylim(40, 100)
ax1.grid(True, alpha=0.3, linestyle='--')

# 在每个数据点上标注数值
for i, v in enumerate(high_ratios):
    ax1.annotate(f'{v:.1f}%', (frames[i], v), textcoords="offset points",
                xytext=(0, 12), ha='center', fontsize=9, color=color_high, fontweight='bold')

# 右侧y轴: PSNR
ax2 = ax1.twinx()
color_psnr = '#2980B9'
ax2.set_ylabel('PSNR (dB)', color=color_psnr, fontsize=12)
line2 = ax2.plot(frames, frame_psnr, 's--', color=color_psnr, linewidth=2.5,
                 markersize=8, label='PSNR', zorder=4)
ax2.tick_params(axis='y', labelcolor=color_psnr)
ax2.set_ylim(24, 30)

for i, v in enumerate(frame_psnr):
    if not np.isnan(v):
        ax2.annotate(f'{v:.2f}dB', (frames[i], v), textcoords="offset points",
                    xytext=(0, -15), ha='center', fontsize=9, color=color_psnr, fontweight='bold')

# 标题和x轴标签
plt.title('Correlation: High Visibility Ratio vs PSNR', fontsize=14, fontweight='bold')
ax1.set_xticks(frames)
ax1.set_xticklabels(x_labels, rotation=30)

# 合并图例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower left', fontsize=11)

# 添加背景色区域标注: 缺了pair_id=1的帧
ax1.axvspan(0.5, 1.5, alpha=0.08, color='gray', label='PSNR interpolated')
ax1.annotate('PSNR interpolated\n(no pair_001)', xy=(1, 27.5),
            xytext=(1.5, 28), fontsize=8, color='gray',
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

fig.tight_layout()
plt.savefig(os.path.join(output_dir, "visibility_vs_psnr.png"), dpi=150, bbox_inches='tight')
print(f"\n✅ 已保存: {output_dir}/visibility_vs_psnr.png")

# ========== 4. 计算相关系数 ==========
valid_idx = [i for i in range(N_FRAMES) if not np.isnan(frame_psnr[i])]
valid_high = [high_ratios[i] for i in valid_idx]
valid_psnr = [frame_psnr[i] for i in valid_idx]

corr_coef = np.corrcoef(valid_high, valid_psnr)[0, 1]
print(f"\n{'='*50}")
print(f"相关性分析结果:")
print(f"  有效样本数: {len(valid_idx)} 帧")
print(f"  高重复比例范围: {min(valid_high):.1f}% ~ {max(valid_high):.1f}%")
print(f"  PSNR 范围:      {min(valid_psnr):.2f}dB ~ {max(valid_psnr):.2f}dB")
print(f"  Pearson相关系数: {corr_coef:.4f}")
if corr_coef > 0:
    print(f"  结论: 正相关 — 高可见性区域比例越大，PSNR 越高 (重建质量越好)")
else:
    print(f"  结论: 负相关 — 高可见性区域比例越大，PSNR 反而降低")
if abs(corr_coef) > 0.7:
    print(f"  相关强度: 强相关")
elif abs(corr_coef) > 0.4:
    print(f"  相关强度: 中等相关")
else:
    print(f"  相关强度: 弱相关")

# ========== 5. 也画一个散点图 ==========
fig2, ax = plt.subplots(figsize=(7, 6))
ax.scatter(valid_high, valid_psnr, c='#E74C3C', s=80, zorder=5, alpha=0.8)

# 添加帧号标注
for i, idx in enumerate(valid_idx):
    ax.annotate(f'{idx:03d}_0', (valid_high[i], valid_psnr[i]),
               textcoords="offset points", xytext=(8, 5), fontsize=9)

# 线性拟合
z = np.polyfit(valid_high, valid_psnr, 1)
p = np.poly1d(z)
x_fit = np.linspace(min(valid_high)-2, max(valid_high)+2, 100)
ax.plot(x_fit, p(x_fit), '--', color='#3498DB', linewidth=2, alpha=0.7,
        label=f'Linear fit: PSNR = {z[0]:.3f}×ratio + {z[1]:.1f}')

ax.set_xlabel('High Visibility Ratio (≥7/10 frames, %)', fontsize=12)
ax.set_ylabel('PSNR (dB)', fontsize=12)
ax.set_title(f'Scatter: High Visibility vs PSNR (r={corr_coef:.3f})', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig2.tight_layout()
plt.savefig(os.path.join(output_dir, "visibility_vs_psnr_scatter.png"), dpi=150, bbox_inches='tight')
print(f"✅ 已保存: {output_dir}/visibility_vs_psnr_scatter.png")

print(f"{'='*50}")