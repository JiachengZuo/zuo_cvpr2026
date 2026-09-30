import cv2
import numpy as np

def compute_overlap_mask(img_t, img_t1, flow_thresh=1.5):
    h, w = img_t.shape[:2]
    # 1. 前向光流 t -> t+1
    flow_forward = cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(img_t1, cv2.COLOR_BGR2GRAY),
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    # 2. 反向光流 t+1 -> t
    flow_backward = cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(img_t1, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY),
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    # 生成像素网格
    y, x = np.mgrid[:h, :w]
    x2 = x + flow_forward[..., 0]
    y2 = y + flow_forward[..., 1]
    # 边界mask：映射后不超出图像
    bound_mask = (x2 >= 0) & (x2 < w) & (y2 >= 0) & (y2 < h)
    # 反向采样回流
    x_back = np.round(x2).astype(np.int32)
    y_back = np.round(y2).astype(np.int32)
    x_back = np.clip(x_back, 0, w-1)
    y_back = np.clip(y_back, 0, h-1)
    dx_back = flow_backward[y_back, x_back, 0]
    dy_back = flow_backward[y_back, x_back, 1]
    # 回流误差
    err = np.sqrt((dx_back + flow_forward[...,0])**2 + (dy_back + flow_forward[...,1])**2)
    consistency_mask = err < flow_thresh
    # 最终重复可见掩码
    overlap_mask = bound_mask & consistency_mask
    return overlap_mask.astype(np.uint8) * 255