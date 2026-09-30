"""可视化：前10帧前视图（*_0.jpg）多帧叠加热力图
对前10帧（000_0.jpg ~ 009_0.jpg），计算每帧到其他所有帧的像素重叠，
累加生成"被其他帧看到次数"的热力图，叠加在原图上显示。
"""
import cv2
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute import compute_overlap_mask

image_dir = "/home/djhuai/zuo/mobicom/dggt_2/dggt/data/nuscenes/processed_10Hz/mini/007/images/"
output_dir = "/home/djhuai/zuo/mobicom/dggt_2/dggt/data/nuscenes/processed_10Hz/mini/007/overlap_heatmap/"
os.makedirs(output_dir, exist_ok=True)

# 前十帧: 000_0.jpg ~ 009_0.jpg
n_frames = 10
img_names = [f"{i:03d}_0.jpg" for i in range(n_frames)]
img_paths = [os.path.join(image_dir, name) for name in img_names]

# 加载图片
imgs = []
for i, path in enumerate(img_paths):
    img = cv2.imread(path)
    if img is None:
        print(f"⚠ 无法加载: {path}")
        img = np.zeros((900, 1600, 3), dtype=np.uint8)
    imgs.append(img)
    print(f"  已加载: {img_names[i]}  shape={img.shape}")

print(f"\n共加载 {len(imgs)} 张图片")

h, w = imgs[0].shape[:2]

# ========== 1. 计算每帧到其他所有帧的 overlap ==========
# overlap_counts[i] 记录第 i 帧每个像素被其他帧看到的次数
overlap_counts = [np.zeros((h, w), dtype=np.int32) for _ in range(n_frames)]
# pairwise_overlaps[i][j] = 第 i 帧到第 j 帧的 overlap mask (i != j)
pairwise_overlaps = [[None for _ in range(n_frames)] for _ in range(n_frames)]

for i in range(n_frames):
    for j in range(n_frames):
        if i == j:
            continue
        mask = compute_overlap_mask(imgs[i], imgs[j], flow_thresh=1.5)
        mask_bin = (mask > 0).astype(np.int32)
        pairwise_overlaps[i][j] = mask_bin
        overlap_counts[i] += mask_bin
    overlap_ratio_i = overlap_counts[i].mean() / (n_frames - 1) * 100
    print(f"  帧 {img_names[i]}: 平均被其他帧看到的比例 = {overlap_ratio_i:.2f}%")

# ========== 2. 生成叠加热力图 ==========
for i in range(n_frames):
    # 归一化到 [0, 255]
    count = overlap_counts[i]
    max_count = n_frames - 1
    # heatmap: 0~max_count -> 0~255
    heatmap_raw = (count / max_count * 255).astype(np.uint8)
    # 应用颜色映射: 0=黑色, 低=蓝色/紫色, 中=黄色, 高=红色
    heatmap_color = cv2.applyColorMap(heatmap_raw, cv2.COLORMAP_JET)

    # 叠加到原图: 将热力图半透明叠加
    img_orig = imgs[i].copy()
    overlay = cv2.addWeighted(img_orig, 0.6, heatmap_color, 0.4, 0)

    # 添加信息标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    seen_ratio = count.mean() / max_count * 100
    cv2.putText(overlay, f"{img_names[i]}  Seen ratio: {seen_ratio:.1f}%",
                (10, 40), font, 1.0, (255, 255, 255), 2)

    # 保存
    cv2.imwrite(os.path.join(output_dir, f"heatmap_{img_names[i]}.png"), overlay)
    print(f"  已保存: heatmap_{img_names[i]}.png")

# ========== 3. 生成组合大图 ==========
# 2行5列展示所有帧的叠加热力图
scale = 500 / w
disp_w, disp_h = 500, int(h * scale)

rows = []
for row_idx in range(2):
    row_imgs = []
    for col_idx in range(5):
        idx = row_idx * 5 + col_idx
        overlay = cv2.imread(os.path.join(output_dir, f"heatmap_{img_names[idx]}.png"))
        if overlay is not None:
            disp = cv2.resize(overlay, (disp_w, disp_h))
        else:
            disp = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
        row_imgs.append(disp)
    row = np.hstack(row_imgs)
    rows.append(row)

composite = np.vstack(rows)
composite_path = os.path.join(output_dir, "all_heatmaps.png")
cv2.imwrite(composite_path, composite)
print(f"\n✅ 组合热力图已保存: {composite_path}")

# ========== 4. 保存每帧的"被看到次数"原始数据 ==========
counts_dir = os.path.join(output_dir, "counts")
os.makedirs(counts_dir, exist_ok=True)
for i in range(n_frames):
    count_img = (overlap_counts[i] / max_count * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(counts_dir, f"count_{img_names[i]}.png"), count_img)

print(f"✅ 所有数据已保存至: {output_dir}/")
print("完成!")