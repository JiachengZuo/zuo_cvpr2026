# MASt3R 共视图生成 (CoMapGS)

本文档说明如何用本仓库的 `mast3r/covisibility.py` 脚本,按照 CoMapGS 论文
(arXiv:2503.20998) Eq.1 的逻辑,为一组训练视角图像生成**共视图 / covisibility map**。

## 1. 基本原理(论文 Eq.1)

给定训练视图 `T = {I1, I2, ..., In}`,对每一对视图 `(Ii, Ij)`(j ≠ i)用 MASt3R
作为稠密对应预测函数 `f_cor`,得到稠密对应 `C^j_i = f_cor(Ii, Ij)`。
令 `P(C^j_i)` 为视图 `Ii` 中"和 `Ij` 存在对应匹配"的像素坐标集合,则视图 `Ii` 的共视图为:

```
M_i(x, y) = Σ_{j ≠ i} δ_{(x,y) ∈ P(C^j_i)}
```

也就是:像素 `(x,y)` 处的值 = **这个 3D 点同时在多少个其他视图里可见**(范围 0 到 n-1)。
最后按论文做法用形态学操作(开运算:先腐蚀再膨胀)去掉孤立的单像素匹配点,
并归一化到 0..255 灰度(或 jet 彩色)用于可视化。

## 2. 环境

conda 环境 **`mast3r`** 已创建好,激活方式:

```bash
source /home/djhuai/anaconda3/bin/activate mast3r
# 或者,如果 conda init 过:
# conda activate mast3r
```

环境里已装好 PyTorch(CUDA)、MASt3R 依赖,权重文件在 `checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`。

## 3. 运行命令

必须在仓库根目录下、以模块方式运行(否则 `mast3r/dust3r` 的导入路径会找不到):

```bash
cd /home/djhuai/zuo/cvpr/mast3r
source /home/djhuai/anaconda3/bin/activate mast3r

python -m mast3r.covisibility \
    --weights checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
    --images assets/003_120 \
    --output out/covis_120 \
    --overlay \
    --subsample 2
```

完整版(再输出像素级重复明细和视图配对重叠矩阵):

```bash
python -m mast3r.covisibility \
    --weights checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
    --images assets/NLE_tower \
    --output out/covis \
    --overlay --visibility \
    --subsample 2
```

> 图片数量为 n 时,脚本会对所有**有序**图像对跑稠密对应,共 n×(n−1) 对。
> 每对大约几十毫秒(GPU 上约 6–7 it/s),7 张图即 42 对,约 6 秒。

### 用自己的图片数据

- **一个文件夹**:`--images <文件夹路径>`,自动递归收集 `.jpg/.jpeg/.png/.bmp/.webp`。
- **指定文件**:`--images a.jpg,b.jpg,c.jpg`,逗号分隔。

## 4. 参数说明

| 参数 | 默认 | 说明 |
|---|---|---|
| `--weights` | 无(必选其一) | 模型权重 `.pth` 路径 |
| `--model_name` | 无 | 或指定 HF 模型名,与 `--weights` 二选一 |
| `--images` | 必选 | 图片文件夹或逗号分隔的文件列表(`--images folder` 或 `--images a,b`) |
| `--output` | 必选 | 输出目录 |
| `--image_size` | 512 | MASt3R 预处理尺寸(长边),不必改 |
| `--subsample` | 8 | 2D-2D 匹配的网格步长。**越小匹配越密**:`1`=全稠密。**推荐 `2`**(见下) |
| `--conf_thr` | 0.0 | 匹配置信度阈值。论文 Eq.1 统计任意对应,0.0 即保留全部互反匹配 |
| `--no_morpho` | 关 | 加上则跳过形态学开运算(论文默认启用) |
| `--dilate_k` | 3 | 形态学核大小 |
| `--overlay` | 关 | 额外输出"原图 + jet 热力图"半透明叠加图 |
| `--visibility` | 关 | 额外输出"像素重复见到明细 + 视图配对重叠矩阵"(见下) |

## 5. 输出文件

`--output` 目录下生成:

- `covisibility_map_000.png` ~ `covisibility_map_00N.png` — 每张视图的共视图,
  灰度值 = 归一化后的共视数(纯白 = 最高共视,黑色 = 无对应)。
- `covisibility_map_000_overlay.png` … — 叠加了 jet 热力图的预览图(启用 `--overlay` 时)。
- `covisibility_stats.json` — 统计信息:`n_views`、每张图的最大共视数 `max_count`、
  预处理尺寸 `image_size`、用到的图片列表。

启用 `--visibility` 时额外生成:

- `visibility_000.json` ~ `visibility_00N.json` — **逐像素重复明细**。每个被记录的像素
  有坐标 `(x, y)`、`count`(该像素的点被几个其他视图同时见到)和 `views`(具体是哪几个
  视图索引)。文件里 `pixels` 按 count 从高到低排序,第一行就能看到"最被反复看到的像素在哪、被谁看到"。
- `pairwise_overlap.json` / `.txt` — **视图配对重叠矩阵**(n×n)。`overlap[i][j]` = 视图 i
  中被视图 j 也见到的唯一像素个数,数值越大说明这两个视图看到的重叠区域越多。

## 6. 常见问题

### 为什么生成的图大部分是黑的、白色匹配点很少?

这是**模型 + 匹配过滤的固有行为**,不是脚本出错:

1. **采样太稀**:`--subsample 8` 时只在每 8×8 网格取一个查询点,一对图像只有约
   1000 个互反匹配点;
2. **互反最近邻过滤**:只有"双向最近邻"(我匹配到你、你也匹配到我)的点才保留;
3. **形态学开运算**:把孤立单像素点清理掉一层;
4. **颜色归一化**:按全图最大值拉伸,比如某像素只在 1 个视角可见(max=5 时),
   映射后灰度只有约 51/255,视觉上接近黑——只有共视数最高的像素才是纯白;
5. **本来就没有对应**:天空、空旷背景等区域没有可区分的纹理,任何参数下都不会有匹配。

### 怎么让白色更多、更密?

用 `--subsample 2`。实测(单对图像)互反匹配点:

| subsample | 匹配点数 | 效果 |
|---|---|---|
| 8 | 1,088 | 稀疏,白区仅 ~0.1% |
| 4 | 2,568 | 中等 |
| **2** | **5,188** | 推荐:白区提升到 1–20%,代价小 |
| 1 | 6,972 | 只比 2 多 34%,耗时明显上升,性价比低 |

实测 7 视图整景:`subsample 8` 每张图白区 0.03%–0.23%,`subsample 2` 提升到
1.2%–20%(最大共视数也从 2–5 提高到 4–5)。**日常使用建议 `--subsample 2`。**

### 报错 "can't import mast3r / dust3r"?

一定在仓库根目录用 `python -m mast3r.covisibility ...` 运行,不要直接
`python mast3r/covisibility.py`。

### 卡在 `preparing RoPE2D...` ?

日志开头的 `Warning, cannot find cuda-compiled version of RoPE2D, using a slow
pytorch version instead` 是无害提示(cuda 版本 RoPE2D 内核未编译,用 PyTorch
回退实现),不影响结果,只是稍慢。



共视图计算
```
python -m mast3r.covisibility \
    --weights checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
    --images /home/djhuai/zuo/mobicom/dggt_2/dggt/data/nuscenes/processed_10Hz/mini/003/images \
    --output out/covis_003_cam0_f8 \
    --camera 0 \
    --max_frames 8 \
    --visibility \
    --image_size 518 \
    --subsample 2
```

## 7. 时间窗口模式 (`covisibility_window.py`)

新增 `mast3r/covisibility_window.py`，支持按时间窗口分组计算共视图。

给定 `--window_size S`（例如 `-S 4`），将帧序列按 `S` 帧一组划窗：
- Window 0: 帧 0~3
- Window 1: 帧 4~7
- ...

每个窗口**内部**独立计算共视图：只在该窗口的其他帧之间做稠密匹配，输出该窗口内每个像素被"同窗口内几个其他帧同时看到"的计数图。

### 用法示例

```bash
cd /home/djhuai/zuo/cvpr/mast3r
source /home/djhuai/anaconda3/bin/activate mast3r

# 每 4 帧一个窗口计算共视图
python -m mast3r.covisibility_window \
    --weights checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
    --images assets/003_120 \
    --output out/covis_window \
    --window_size 4 \
    --overlay \
    --subsample 2
```

### 输出结构

```
out/covis_window/
├── covisibility_stats.json          # 全局统计（窗口数、帧数等）
├── window_000/                      # 第一个窗口（帧 0~3）
│   ├── covisibility_map_000.png     # 帧 0 的灰度共视图
│   ├── covisibility_map_001.png     # 帧 1 的灰度共视图
│   ├── covisibility_map_002.png     # 帧 2 的灰度共视图
│   ├── covisibility_map_003.png     # 帧 3 的灰度共视图
│   ├── covisibility_stats.json      # 该窗口统计
│   └── ...                          # overlay / visibility (若启用)
├── window_001/                      # 第二个窗口（帧 4~7）
│   └── ...
```

每个 `covisibility_map_XXX.png` 是**灰度图**：像素越亮（越白）= 该像素在窗口内被越多次重复看到；纯黑 = 无共视对应。

### 新增参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--window_size` / `-S` | 无（不分组） | 时间窗口大小。例：`-S 4` 每 4 帧一组。不设则退化为原版全局计算 |

其余参数与原版 `covisibility.py` 完全一致。


--camera 0 就是你要的前视图筛选，两个脚本都已经内置了。用法：

python -m mast3r.covisibility_window \
    --weights checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
    --images /path/to/nuscenes/images \
    --output out/covis_window \
    --camera 0 \
    --window_size 4 \
    --subsample 1