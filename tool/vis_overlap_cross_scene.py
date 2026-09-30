"""可视化：不同场景第0帧之间的逐帧像素重叠（前10帧）
输入图片命名: 000_0.jpg, 001_0.jpg, ..., 009_0.jpg (不同场景ID, 帧索引0)
"""
import cv2
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute import compute_overlap_mask

# 路径设置
image_dir = "/home/djhuai/zuo/mobicom/dggt_2/dggt/data/nuscenes/processed_10Hz/mini/007/images/"
output_dir = "/home/djhuai/zuo/mobicom/dggt_2/dggt/data/nuscenes/processed_10Hz/mini/007/overlap_cross_scene/"
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
        # 用空白占位
        img = np.zeros((900, 1600, 3), dtype=np.uint8)
    imgs.append(img)
    print(f"  已加载: {img_names[i]}  shape={img.shape}")

print(f"\n共加载 {len(imgs)} 张图片")

# 逐对计算重叠掩码 (t -> t+1)
overlap_masks = []
for i in range(len(imgs) - 1):
    mask = compute_overlap_mask(imgs[i], imgs[i+1], flow_thresh=1.5)
    overlap_masks.append(mask)
    overlap_ratio = mask.mean() / 255.0 * 100
    print(f"  重叠 {img_names[i]} → {img_names[i+1]}: Overlap ratio = {overlap_ratio:.2f}%")

# ========== 生成综合可视化 ==========
# 布局: 每行3列 (img_t, overlap_mask, img_t1)，共 (n_frames-1) 行
rows = []
for i in range(len(imgs) - 1):
    h, w = imgs[i].shape[:2]
    # 调整尺寸便于显示（缩放到宽度400）
    scale = 400 / w
    disp_w, disp_h = 400, int(h * scale)

    img_t_disp = cv2.resize(imgs[i], (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
    img_t1_disp = cv2.resize(imgs[i+1], (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
    mask_disp = cv2.resize(overlap_masks[i], (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)

    # mask转彩色: 白色=重叠, 黑色=不重叠
    mask_color = cv2.cvtColor(mask_disp, cv2.COLOR_GRAY2BGR)

    # 添加标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_t_disp, f"Scene {img_names[i]}", (10, 30), font, 0.7, (0, 255, 0), 2)
    cv2.putText(img_t1_disp, f"Scene {img_names[i+1]}", (10, 30), font, 0.7, (0, 255, 0), 2)
    overlap_ratio = overlap_masks[i].mean() / 255.0 * 100
    cv2.putText(mask_color, f"Overlap: {overlap_ratio:.1f}%", (10, 30), font, 0.7, (0, 0, 255), 2)

    row = np.hstack([img_t_disp, mask_color, img_t1_disp])
    rows.append(row)

# 垂直拼接所有行
composite = np.vstack(rows)

# 保存可视化
composite_path = os.path.join(output_dir, "overlap_cross_scene_vis.png")
cv2.imwrite(composite_path, composite)
print(f"\n✅ 综合可视化已保存: {composite_path}")

# 也保存每对的重叠掩码（全分辨率）
masks_dir = os.path.join(output_dir, "masks")
os.makedirs(masks_dir, exist_ok=True)
for i in range(len(imgs) - 1):
    mask_path = os.path.join(masks_dir, f"overlap_{img_names[i]}_to_{img_names[i+1]}.png")
    cv2.imwrite(mask_path, overlap_masks[i])

print(f"✅ 所有掩码已保存至: {masks_dir}/")
print("完成!")