理解。你的约束是：

N每张图=N像素
N每张图​=N像素​

因此你不是单纯增加或删除高斯，而是要进行：

高斯重分配：从冗余区域移除 M 个，再向其他区域补回 M 个
高斯重分配：从冗余区域移除 M 个，再向其他区域补回 M 个​

总高斯数保持不变。
1. 先计算可释放的高斯数量

假设筛选出的冗余高斯集合为：

Gred
Gred​

其数量为：

M=∣Gred∣
M=∣Gred​∣

将它们标记为 inactive，暂时不参与渲染，但不要改变总数量：
python复制代码
收起

redundant_mask = redundancy_score > threshold
num_free = redundant_mask.sum()

复制

这些高斯可以理解为“空闲高斯槽位”。
2. 计算每个区域的需求分数

对图像划分区域 k=1,…,K
k=1,…,K，定义：

    RkRk​：区域重复可见率；
    EkEk​：区域重建误差；
    NkNk​：区域当前有效高斯数量；
    TkTk​：纹理或结构复杂度。

推荐使用：

Dk=(1−Rk)⋅Ek⋅1Nk+1
Dk​=(1−Rk​)⋅Ek​⋅Nk​+11​

这表示：

    重复越少，需求越高；
    误差越大，需求越高；
    已有高斯越少，需求越高。

如果想加入纹理复杂度：

Dk=(1−Rk)(0.6Ek+0.2Tk)1Nk+1
Dk​=(1−Rk​)(0.6Ek​+0.2Tk​)Nk​+11​
3. 计算每个区域应该接收多少个高斯

将需求分数归一化：

Pk=Dk∑jDj
Pk​=∑j​Dj​Dk​​

然后将释放的 M
M 个高斯分配出去：

M~k=MPk
M~k​=MPk​

但是需要处理整数问题。推荐使用最大余数法。
python复制代码
收起

raw_num = num_free * demand / (demand.sum() + 1e-8)

num_add = torch.floor(raw_num).long()

remain = num_free - num_add.sum()

fraction = raw_num - num_add.float()
selected = torch.topk(fraction, remain).indices
num_add[selected] += 1

复制

这样可以保证：

∑kMk=M
k∑​Mk​=M

也就是释放多少个，就补回多少个。
4. 设置每个区域的容量上限

为了避免所有高斯都被分配到一个区域，需要设置区域上限：

Mk≤Mkmax
Mk​≤Mkmax​

例如：
python复制代码
收起

max_add = 2 * average_region_count
num_add = torch.clamp(num_add, max=max_add)

复制

如果某些区域达到上限，将剩余高斯继续分给下一个高需求区域。

更合理的限制方式是根据区域面积和复杂度设置：

Mkmax=η⋅∣Ωk∣(1+λTk)
Mkmax​=η⋅∣Ωk​∣(1+λTk​)
5. 不要新建高斯，直接重用被释放的高斯

因为你的总数量必须固定，所以最简单的方法是：

    找出冗余高斯；
    保留它们的参数槽位；
    把它们的位置、颜色、尺度、旋转、opacity 重新初始化到目标区域。

假设一个高斯参数为：
python复制代码
收起

xyz
scaling
rotation
opacity
features

复制

则：
python复制代码
收起

free_ids = torch.where(redundant_mask)[0]

for region_id in target_regions:
    ids = free_ids[:num_add[region_id]]
    free_ids = free_ids[num_add[region_id]:]

    pixels = select_high_error_pixels(
        region_id,
        num_add[region_id]
    )

    xyz_new = backproject_to_world(
        pixels,
        depth,
        camera
    )

    xyz[ids] = xyz_new
    scaling[ids] = initial_scale
    rotation[ids] = initial_rotation
    opacity[ids] = initial_opacity
    features[ids] = initialize_color(target[pixels])

复制

这里不是删除高斯，而是把“低利用率高斯”重定位到高需求区域。
6. 新高斯应该放在区域中的什么位置

不要在目标区域内随机放置，优先选择高误差像素。

对每个像素定义：

e(p)=∥Irender(p)−Igt(p)∥1
e(p)=
​Irender(p)−Igt(p)
​1​

在区域 k
k 内按照 e(p)
e(p) 从大到小选择像素。
python复制代码
收起

error_map = torch.abs(rendered - target).mean(dim=-1)

selected_pixels = topk_pixels(
    error_map[region_mask],
    num=num_new
)

复制

如果有深度：

X=K−1p~d(p)
X=K−1p~​d(p)

再转换到世界坐标：

Xworld=Tcw−1X
Xworld​=Tcw−1​X

这样重新分配的高斯会落在当前最需要优化的位置。
7. 建议加入“边际收益”判断

区域误差高不一定代表增加高斯一定有效。因此每轮分配后，重新训练少量 iteration，计算区域损失下降：

Gk=Ekbefore−EkafterMk+ϵ
Gk​=Mk​+ϵEkbefore​−Ekafter​​

如果某区域增加高斯后误差下降明显，说明分配有效；如果几乎没有下降，下一轮降低它的分配权重。

更新需求：

Dknew=Dk⋅1Gk+ϵ
Dknew​=Dk​⋅Gk​+ϵ1​

或者使用：

Dknew=Dk⋅exp⁡(−λGk)
Dknew​=Dk​⋅exp(−λGk​)

不过第一版可以先不加，等基本流程稳定后再引入。
最简单的完整策略

你可以采用下面这个固定流程：
python复制代码
收起

# 1. 找到冗余高斯
redundant_mask = redundancy_score > threshold
free_ids = torch.where(redundant_mask)[0]
num_free = len(free_ids)

# 2. 计算每个区域的指标
low_repeat = 1.0 - repeat_score
sparsity = 1.0 / (gaussian_count + 1.0)

demand = low_repeat * error_score * sparsity

# 3. 归一化分配
raw_num = num_free * demand / (demand.sum() + 1e-8)

num_add = torch.floor(raw_num).long()

remain = num_free - num_add.sum()
fraction = raw_num - num_add.float()

if remain > 0:
    ids = torch.topk(fraction, remain).indices
    num_add[ids] += 1

# 4. 重用空闲高斯
for region_id in range(num_regions):
    n = num_add[region_id]

    if n == 0:
        continue

    ids = free_ids[:n]
    free_ids = free_ids[n:]

    pixels = select_high_error_pixels(region_id, n)
    xyz_new = backproject_to_world(pixels, depth, camera)

    reinitialize_gaussians(ids, xyz_new, target, pixels)

复制

并且始终检查：
python复制代码
收起

assert num_add.sum() == num_free

复制
最推荐的分配公式

在你的固定数量约束下，建议首先使用：

Dk=(1−Rk)⋅EkNk+1
Dk​=(1−Rk​)⋅Nk​+1Ek​​​

然后：

Mk=IntegerAllocate⁡(MDk∑jDj)
Mk​=IntegerAllocate(M∑j​Dj​Dk​​)​

满足：

∑kMk=M
k∑​Mk​=M​

这样就能同时满足：

    每张图高斯总数不变；
    冗余高斯被重用；
    高斯优先分配到低重复、高误差、低密度区域；
    分配前后保持严格数量平衡。