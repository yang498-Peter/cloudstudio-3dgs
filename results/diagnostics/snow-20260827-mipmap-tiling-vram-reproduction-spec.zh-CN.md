# MipMap 分块训练与显存管理复刻规范（snow-20260827）

更新时间：2026-08-28（UTC+08:00）
任务：`D:\mipmap-lite\1fad9647-f717-4cd4-9391-11f219d7e5d1\snow\snow-20260827`
适用二进制：

- `divide_engine.exe` SHA-256：`9DBBCB7059363B8460D643D73553432C035D5A0073962A17988CB53A3D0A747D`
- `mipmap_gaussian_splat.dll` SHA-256：`A910B39DBAD956DC35E9E436ACFD0FB8D92364E03BB44B0401CBEF6BCB8D492E`

## 1. 结论

MipMap 的 LiDAR/GS 分块不是固定网格，也不是八叉树。`divide_mode=2` 对应一棵**按照片投影像素负载平衡的递归平面 KD 树**：本任务 `pipeline_mode=1` 只允许 X/Y 轴，每个候选块对可见锚点重新投影，按每张图的有效裁剪矩形估算图像负载，再选择切轴与切线。各块保留完整 Z 范围。

显存管理不是“把 `tiles.json.max_memory` 直接分配成同样大的 CUDA buffer”。它分三层：

1. 任务预检得到可用训练预算：`min(GPU0 可用显存 GiB, 12, 可用系统内存 GiB)`；
2. 分块器按图像裁剪像素数估算每块原始工作量，并对比较值乘 `0.8` 后决定是否继续切；
3. GS 训练只串行驻留一个 Tile，释放上一块对象，并在训练循环的偶数 step 调用 CUDA caching allocator 的缓存释放虚函数。模型/优化器活跃张量不会因此消失，只有未使用缓存块可归还给驱动。

目前已能实现一个可运行的兼容分块器和显存审计器。在 snow 上，以启动时 `7.013 GiB` 预算运行、不强制深度和切轴，会自动得到 4 个叶块以及 `X → 左右各 Y` 拓扑；三条切线和厂商结果在浮点误差内一致，四个叶块中心误差为 `3.47e-9–3.97e-9 m`。这是 snow 的**数值级闭环复刻**，但仍不能承诺任意新数据集位级一致，因为原始畸变相机的内部投影器和两阶段数据绑定还没有完全静态还原。

## 2. 两类空间框：core 与导出 ROI

厂商节点内部实际同时维护两类框：

- `Node +0x00..+0x14`：连续的 ownership/partition box，子块由父框与切平面相交得到；
- `Node +0x18..+0x2c`：当前节点候选点的离散包围盒，只负责生成 64 个候选坐标。

这两类框不能混用。子节点会重算离散点包围盒，但不会把 ownership box 缩到点云边界。`tiles.json` 保存的是 ownership box 再加 halo 后的导出/训练 ROI。对每一维：

```text
halo = 0.002 * core_extent
export_min = core_min - halo
export_max = core_max + halo
```

反解为：

```text
core_padding = export_extent * 0.002 / 1.004
core_min = export_min + core_padding
core_max = export_max - core_padding
```

snow 反解出的共同根 core：

```text
X [-54.280109411, 62.156955725]
Y [-53.151596075, 48.701530461]
Z [-14.547475526, 22.072318355]
```

真实树：

```text
root: X = 0.142307
├─ left:  Y = 3.243397   -> Tile_0 / Tile_1
└─ right: Y = -0.492836  -> Tile_2 / Tile_3
```

相邻块各自向边界外扩 0.2%，所以两侧合计 overlap 约为局部跨度的 0.4%。这解释了已实测的约 20–23 cm overlap；不是一个固定米数的 halo。

## 3. 根 ROI 的生成

`divide_engine.exe` RVA `0x3FA80` 的根框构造使用相机/视图几何和支持度至少 3 的 SfM 点，先对范围每侧扩 20%，再与场景/用户 ROI 相交。

在 `mvs_undistort.pb.bin` 上，支持点与相机并集扩 20% 得到：

```text
Y [-53.1515976, 48.7015320]
```

与运行时根 core Y 在微米级一致。X/Z 被场景 ROI 进一步裁剪为上节数值。这说明根框不是原始 LAS 全包围盒，也不是单纯相机轨迹包围盒。

兼容实现必须允许调用方传入最终场景 ROI；若只有 MVS，可把“支持点框扩 20%”作为 fallback，但不能承诺与厂商 X/Z 完全一致。

## 4. 递归停止与分裂条件

递归函数位于 `divide_engine.exe` RVA `0x3C690`。直接控制流为：

```text
if anchor_count < 100 and pixel_load < 100000:
    stop/discard

raw_memory = pixel_load * bytes_per_pixel(resolution_level)
comparison_memory = 0.8 * raw_memory

if comparison_memory > memory_budget and depth < 10:
    split_and_recurse()
else:
    save_leaf(raw_memory)
```

注意低支持门是 `AND`：只有锚点少于 100 且像素负载少于 100,000 时才停止。不要误写成任一条件成立就停止。

分辨率级别对应的原始字节/像素系数来自 RVA `0x3C7B9..0x3C87A`：

```text
F(level) = pow(0.5, 2*max(level-1, 0))
         + 4.5 * pow(0.5, 2*min(max(level-1, 0), 1))

level 0/1: 5.5 B/px
level 2:   1.375 B/px
level 3:   1.1875 B/px
```

`0.8` 只出现在分裂比较值中；`tiles.json.max_memory` 保存未乘 `0.8` 的原始估算 GiB。这一点已用运行产物校准，取代此前把叶值写成 4.4 B/px 的旧解释。

## 5. `min_avali_memory_size` 与分块预算

高层参数构造 RVA `0x4A7F1..0x4A8C6` 在启用相关 3D/GS 输出开关时执行：

```text
gpu_budget_gib = min(GetGPUAvailMemoryBytes(0) / GiB, 12.0)
min_avali_memory_size = min(gpu_budget_gib,
                            GetAvailMemoryBytes() / GiB)
```

该字段由任务参数解析器以 double 写入结构体 `+0x90`。分块递归内部使用字节预算与上节的 `comparison_memory` 比较。若高层没有提供有效预算，另一路 fallback 会按系统可用内存和 64/128 GiB 上限构造预算；普通 GS 任务应优先使用 GPU/RAM 联合预算。

工程实现建议：

```text
budget_gib = min(gpu_free_gib, 12.0, system_available_gib)
if gpu_free_gib is unavailable:
    budget_gib = min(system_available_gib, 64.0)      if RAM <= 128 GiB
                 min(0.5*system_available_gib,128.0) otherwise
```

不要把 `max_memory` 当作 CUDA 峰值，也不要拿 `nvidia-smi used_memory` 直接验证它；前者是照片工作集估算，后者还取决于 Gaussian 数、optimizer state、rasterizer workspace、临时梯度和 allocator 缓存。

## 6. 如何计算单块照片负载

切分器的核心不是点数，而是**每块在每张相关照片上的裁剪面积**。

对候选区域内的每个 SfM 锚点：

1. 遍历该点关联的可见 image；
2. 用 image 的 3×4 world-to-camera 矩阵计算 `q = P*[X,Y,Z,1]`；
3. 用相机模型投影并四舍五入到像素；
4. 对每张 image 累积 `min_x/min_y/max_x/max_y`；
5. 原始矩形宽和高都至少 256 px 才计入；
6. 四周各扩 128 px，然后裁到图像边界；
7. 把所有有效 image 的裁剪面积求和得到 `pixel_load`。

本任务去畸变 MVS 的 pinhole 投影已经由观测点精确验证：

```text
u = fx * qx/qz + cx
v = fy * qy/qz + cy
```

厂商函数 RVA `0x3E1F0` 明确做重新投影、C `round()` 和边界检查；观测记录在此只用于确定 image 关联。对 `mvs_undistort.pb.bin` 重新投影 150,029 条观测时，与 PB 保存像素只有 15 条发生 1 px 差异，吻合率为 `99.9900019%`，因此去畸变 pinhole 路径已经可独立复刻。

原始 `mvs.pb.bin` 使用 2 个畸变类型 2 相机、342 个 image；本任务的 `mvs_raw.pb.bin` 与它 SHA-256 完全相同。兼容回放目前对这一条路径复用 PB 保存的 observation xy，作为父照片负载聚合尚未还原时的透明 surrogate；不能把它表述为厂商 division 入口实际加载了 raw 文件，或厂商实现本身不重新投影。

## 7. 切轴、切线和 64 个候选

切分入口 RVA `0x3D830`，候选评价 RVA `0x3E7C0`：

- `pipeline_mode != 0`：只评估 X/Y；本任务为此分支；
- `pipeline_mode == 0`：允许 X/Y/Z；
- 轴长度低于最长候选轴的 20% 时不作为正常候选；
- 每个轴构造 64 个均匀空间候选；
- 用 prefix/suffix 累积快速得到每条切线左右两侧的 per-image 矩形；
- 选择 `left_pixel_load - right_pixel_load` 穿过 0 的位置；零平台取中间，落到端点则回退中心；
- 每个轴的代价是 `max(left_load, right_load)`；
- 若两个最佳轴代价差至少 10%，选择更低代价轴；若差小于 10%，选择空间跨度更长的轴；
- poor split 的 0.9 比例还会触发 fallback/retry，它不是主轴选择公式。

snow 的数值闭环需要区分三个计算语义，但新静态证据表明不能把它们直接等同为三个同时传入的文件：

1. 342 个父照片的 observation graph 语义决定候选切线的左右照片负载与切轴；当前兼容工具用 `mvs.pb.bin` 保存坐标重放这一语义；
2. `mvs_undistort.pb.bin` 的点坐标包围盒决定 64 个候选坐标；
3. `mvs_undistort.pb.bin` 的 1368-view 投影/crop 负载决定叶块 `max_memory`。

`divide_engine.exe` RVA `0x2FFF7..0x30079` 已直接证明：division 入口先检查 `mvs_undistort.pb.bin`，存在则加载它，否则回退 `mvs.pb.bin`。因此厂商本次运行不是把 raw/undistort 两个 PB 同时传给递归切分器。另一个上层参数构造函数 RVA `0x49DE0` 会加载 `mvs.pb.bin`，主要用于缺省 ROI/场景参数初始化；它与 division 入口选择的 MVSBlock 是不同用途。

为什么 raw 342-view 回放仍能精确解释切轴？父图审计给出了结构答案：1368 个 undistort view 都能用同一 PB 内保留的“原相机 ID + timestamp”精确映射到 342 张父照片，每组 4 个；1368/1368 的 metadata 映射与相机中心映射一致，最大父中心距离仅 `7.26e-15 m`；虚拟 view 的 `image_rect` 虽为 1456×2912，其 metadata 仍全部保留 2912×2912 原图尺寸。18,847 个 undistort 点全部能按 float32 坐标回到原 MVS；将虚拟 view 折叠到父照片后，137,468 个唯一 `(point,parent-image)` 对全部存在于原 153,316 对中，precision=100%，覆盖率 89.663%。因此 raw 回放是“父照片负载聚合”的高质量 surrogate，不是实际文件双路传参证据。厂商怎样在已加载的 undistort block 内部利用这些 metadata 完成父 view 聚合/选择，仍需继续静态追踪。

根节点的 undistort 点 X 范围为：

```text
[-53.4560470581, 48.8680877686]
step = (max-min)/63 = 1.6241926163
candidate[33] = 0.1423092797
float32 实现 = 0.1423072815
厂商切线      = 0.1423072866
```

左子节点点 Y 范围的 `candidate[39]` 同样精确得到 `3.2433967590`。右子节点在候选 envelope 使用 observation support 至少 2 的点后，得到 `-0.4928359985`。这个“子节点 support≥2”规则已被 snow 数值精确验证，但尚未在静态代码中绑定到一个明确条件分支，仍属于高置信运行推断。

snow 根节点使用父照片 observation 负载 surrogate 回放时：

```text
X max-child load = 1,075,712,554 px
Y max-child load = 1,048,496,233 px
relative difference ≈ 2.5%
```

二者差异小于 10%，因此 tie-break 选择空间跨度更长的 X。若把 1368 个 undistort view 全部作为独立切分负载，Y 会比 X 低约 11.9% 并错误选 Y；这说明实际切分负载存在父 view 聚合/筛选语义，而不是证明厂商同时加载 raw PB。

使用 `7.013 GiB` 预算，不强制深度、不强制轴，完整回放自动产生：

```text
root  X =  0.14230728149414062
left  Y =  3.243396759033203
right Y = -0.49283599853515625
leaf count = 4
```

## 8. snow 叶块内存公式的运行校准

用厂商真实 core ROI 在 `mvs_undistort.pb.bin` 上重算有效裁剪面积，再乘 5.5 B/px：

| Tile | 重放 pixel load | 重放 GiB | 厂商 `max_memory` GiB | 相对误差 |
|---|---:|---:|---:|---:|
| Tile_0 | 1,306,820,686 | 6.6939 | 6.5911 | +1.560% |
| Tile_1 | 1,179,953,745 | 6.0440 | 6.0416 | +0.041% |
| Tile_2 | 992,946,852 | 5.0861 | 5.0818 | +0.086% |
| Tile_3 | 832,279,267 | 4.2632 | 4.2639 | -0.017% |

Tile_1–3 均在 0.2% 内；Tile_0 的 1.6% 差异符合观测坐标取整、边界点和厂商重新投影/裁剪细节尚未完全同构。这个结果强力确认 `5.5 B/px` 是叶值公式。

根 core 在去畸变 MVS 上重放为 3,825,202,955 px：

```text
raw = 19.5937 GiB
comparison = 15.6750 GiB
```

显著高于约 7 GiB 的训练可用量，需要继续切；实际四个叶的 comparison 值为约 3.41–5.36 GiB，均可接受。

## 9. 每个 Tile 如何获得照片

每个空间块不是简单复制全部照片。需要区分“训练上下文 view 集”和“内存计费 view 集”：

- 块 MVS 只保留与块内锚点/ROI 有关联的 view；
- 每个 view 保存自己的 crop rectangle；
- crop 至少 256×256，外围再保留 128 px 上下文；
- 本任务原始 342 张照片经鱼眼展开形成最多 1368 个派生 view；各 Tile 实际块 MVS 包含 656/644/607/595 个 view；
- 由导出 ROI 内点的 observation 图闭包预测 view 集，与实际集合的 Jaccard 为 `0.9924/0.9787/0.9918/0.9691`，且实际 view 全部在预测集合内；
- 真正通过 256 px 最小矩形、参与 `max_memory` 计费的 view 只有 541/545/476/484；因此“保留为训练上下文”不等于“计入计划像素负载”；
- 相邻块共享大量照片，这是独立 Tile 能优化到相近视觉表面的主要原因之一。

块点记录还出现可解释的重复：Tile_0–3 分别有 6,333/3,933/4,659/3,271 个 unique source point 被重复写入，绝大多数位于导出 ROI 内。这支持“上下文图点 + 本地 ROI 点追加”的构造模型，但重复记录的精确语义尚未静态闭合。

兼容训练器应让相邻块共享 halo 内可见照片，并在坐标、相机曝光参数和语义/depth mask 上保持同一约定；不能按“照片中心落在哪块”做硬归属。

## 10. 训练调度和显存释放

运行时文件时间戳证明 4 块严格串行：

```text
Tile_0 train + LOD
  88.27 s gap
Tile_1 train + LOD
 118.13 s gap
Tile_2 train + LOD
  59.83 s gap
Tile_3 train + LOD
```

没有两块训练窗口重叠。所有块在同一 `reconstruct_full_engine.exe` 进程内执行，所以相机模型、网络 engine 和部分长期上下文可以复用，但块点云、Gaussian 参数、Adam state、渲染临时张量按块创建和销毁。

`mipmap_gaussian_splat.dll` RVA `0xA6B13..0xA6B2B`：

```text
if ((global_step & 1) == 0):
    CUDACachingAllocator::allocator->virtual_0x60()
global_step++
```

同一 `virtual_0x60` 还在对象清理路径 RVA `0x59862..0x59870`、MCMC 后处理和颜色协调结束路径出现。结合 PyTorch CUDA allocator ABI 和无额外显式参数的调用形态，高置信对应 `emptyCache()`；`recordStream` 在 DLL 中是另一条带 allocation/stream 实参的虚调用。

复刻时建议：

```python
for tile in tiles:                         # 串行
    model, optimizers = build_tile(tile)
    for step in range(total_steps):
        train_one_view(model, optimizers)
        if step % 2 == 0:
            torch.cuda.empty_cache()       # 对齐厂商策略，性能需自行基准
    export_level0_and_lods(model)
    del optimizers, model, tile_tensors
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
```

需要强调：`empty_cache()` 只归还未使用缓存段，不会释放仍被 model/optimizer 引用的 tensor。自己实现时每两步调用可能牺牲吞吐；若目标是结果兼容而非显存曲线兼容，可改为每个 densification 周期/每块结束调用，并用峰值基准决定。

## 11. snow 的真实 RAM/VRAM 曲线

日志共有 220 条 MemoryProfile 采样。以 point-cloud PB 写完到 level-0 PB 写完作为训练窗口：

| Tile | 训练秒 | 计划 GiB | VRAM median / p95 / peak GiB | 进程 RAM peak GiB | LOD 后 VRAM GiB |
|---|---:|---:|---:|---:|---:|
| Tile_0 | 1551.50 | 6.591 | 3.16 / 4.25 / 4.53 | 4.75 | 0.91 |
| Tile_1 | 1124.89 | 6.042 | 2.96 / 4.31 / 4.56 | 3.62 | 0.90 |
| Tile_2 | 928.03 | 5.082 | 3.03 / 4.16 / 4.41 | 2.50 | 0.92 |
| Tile_3 | 581.51 | 4.264 | 2.72 / 3.22 / 3.44 | 2.30 | 无后续样本 |

前三块结束后显存均回到约 0.9 GiB 基线，直接支持“上一块训练对象已释放、长期 runtime 仍驻留”的模型。计划最大值和 CUDA 峰值不是一一对应关系；例如 Tile_1 计划值低于 Tile_0，但实测峰值略高。

## 12. 可直接运行的复刻/审计工具

### 12.1 分块回放

文件：`.tools/replay_mipmap_adaptive_tiling.py`

snow 验证命令：

```powershell
$env:PYTHONPATH='G:\cloudstudio-3dgs\.tools\python'
$task='D:\mipmap-lite\1fad9647-f717-4cd4-9391-11f219d7e5d1\snow\snow-20260827'
python .tools\replay_mipmap_adaptive_tiling.py `
  'D:\Program Files\MipMap\MipMapLite\resources\resources\catch3d\divide_engine.exe' `
  "$task\result\milestones\mvs.pb.bin" `
  --candidate-mvs "$task\result\milestones\mvs_undistort.pb.bin" `
  --memory-mvs "$task\result\milestones\mvs_undistort.pb.bin" `
  --split-use-stored-observations `
  --tiles "$task\result\task\tiles.json" `
  --resolution-level 1 `
  --budget-gib 7.013 `
  --child-envelope-min-observations 2 `
  --output 'results\diagnostics\snow-20260827-mipmap-adaptive-tiling-exact-budget-replay.json'
```

该命令不使用 `--force-depth` 或 `--force-axis-by-depth`，4 个叶块完全由预算自动得到。`--split-use-stored-observations` 明确表示工具用原图 observation xy 模拟“342 个父照片负载”；厂商本次 division 入口实际优先加载 undistort PB。该组合是 snow 的精确兼容模型，不是厂商双 PB 传参模型。

父 view 审计器：`.tools/audit_mipmap_undistort_parent_views.py`。它只读比较 `mvs.pb.bin` 与 `mvs_undistort.pb.bin`，输出父相机中心、点坐标和 observation graph 的映射质量。

### 12.2 显存时序审计

文件：`.tools/audit_mipmap_tile_memory.py`

```powershell
python .tools\audit_mipmap_tile_memory.py `
  'D:\mipmap-lite\1fad9647-f717-4cd4-9391-11f219d7e5d1\snow\snow-20260827' `
  --output 'results\diagnostics\snow-20260827-mipmap-tile-memory-audit.json'
```

脚本只读任务日志、Tile 计划和产物时间戳，不会启动、停止或修改 MipMap。

## 13. 兼容实现的推荐模块边界

```text
SceneRootBuilder
  supported SfM points + cameras -> expanded/clipped root ROI

ParentPhotoLoadBalancer
  1368 virtual views -> 342 parent-photo groups
  -> per-image prefix/suffix rectangles
  -> cost-balanced axis/cut load

UndistortedCandidateEnvelope
  1368-view undistorted MVS points
  -> root all relevant points
  -> child support>=2 point envelope
  -> 64 float32 candidate coordinates

AdaptiveViewTiler
  recursive ownership box + separate candidate envelope
  -> intersect parent ownership box at selected cut
  -> 0.2% halo

TileViewBuilder
  core/halo + point/view association
  -> cropped per-tile MVS and images

MemoryPlanner
  undistorted fresh projection/crop rectangles
  min(GPU free, 12 GiB, RAM free)
  -> 0.8 * pixel_load * F(level) split comparison
  -> raw pixel_load * F(level) diagnostic value

SerialTileTrainer
  one active Tile
  -> LAS ROI init
  -> photo/depth/normal GS training
  -> level0
  -> LOD/SOG
  -> explicit object/cache release

GlobalPublisher
  concatenate PLY in Tile order
  index independent SOG blocks
```

最终 snow PLY 是 Tile_0→1→2→3 的完整串接，包含 halo 中的两套 Gaussian；没有发现后处理去重或 core Cut。SOG 同样保持 4 个独立 block。因而复刻分块训练时，不能把“训练 halo”与“发布时必然裁 core”混为一谈。若产品要消除边界双层，需要在自有发布/渲染层另行定义 half-open owner 或 blend 策略。

## 14. 当前复刻程度与剩余差距

| 子系统 | 当前掌握度 | 状态 |
|---|---:|---|
| divide mode、XY/XYZ 轴规则、递归停止 | 95% | 可直接实现 |
| 64 候选、矩形门限、128 px halo、轴 tie-break | 97% | snow 数值闭合，可直接实现 |
| 根 ROI 20% 扩张与场景裁剪 | 85% | 需要调用方提供最终 SceneROI 才能严格一致 |
| 0.2% 空间 halo | 99% | snow 精确反解验证 |
| snow 拓扑、块数和切线坐标 | 99.9% | 自动预算回放，叶中心误差 <4e-9 m |
| 叶块内存估值 | 98% | Tile_1–3 误差 <0.1%，Tile_0 仍差 1.56% |
| Tile view 选择 | 96% | observation 图闭包 Jaccard 0.969–0.992 |
| 启动预算和串行显存管理 | 90% | 静态调用链 + 运行曲线共同支持 |
| 父照片负载聚合/筛选 | 75% | 4:1 中心映射和 observation 子图已闭合，内部聚合分支未闭合 |
| raw 畸变相机重新投影 | 60% | 兼容工具现用保存 observation xy 作 surrogate |
| MVS 文件选择调用链 | 90% | division 优先 undistort、fallback mvs 已直接确认 |
| 通用新场景的块数/边界严格一致 | 85% | snow 已闭合，仍需第二数据集验证 |

离通用可复刻最近的四个缺口：

1. 继续追 division 内部如何把 1368 个虚拟 view 聚合/筛选成父照片负载，并直接绑定子节点 support≥2 的条件来源；
2. 还原畸变类型 2 的原始相机投影器，或用 undistort-to-parent 映射替代 raw observation surrogate；
3. 找到 poor-split 的 `config+0x14` 实际值、reject/retry 分支，以及 `allow_low_resolution_optimize` 真正触发条件；
4. 用至少一个大范围、一个狭长或低纹理数据集交叉验证预算到块数、切轴和边界的泛化。

因此当前结论是：**已经足以实现并运行一个结果兼容的分块训练/显存规划器，并在 snow 上复现到数值级；尚不足以承诺任意数据上 Tile 边界与 MipMap 浮点级完全相同。**

## 15. 本轮闭环、证据等级与可实现伪代码

### 15.1 已经闭环的 snow 数值

| 项目 | 回放结果 | 厂商结果 | 结论 |
|---|---:|---:|---|
| 叶块数量 | 4 | 4 | 自动预算一致 |
| root X | 0.142307281494 | 0.142307286643 | 浮点误差内一致 |
| left Y | 3.243396759033 | 3.243396754732/764369 | 浮点误差内一致 |
| right Y | -0.492835998535 | -0.492836003190/-0.492835993553 | 浮点误差内一致 |
| Tile_0–3 叶中心误差 | 3.47e-9–3.97e-9 m | — | ownership 分区闭合 |
| Tile_0–3 pixel load | 1,306,820,686 / 1,179,953,745 / 992,946,852 / 832,279,267 | 同一 runtime ROI 重放 | 完全一致 |

### 15.2 证据分层

**直接静态证据：** 两套 Node box 字段；64 个 float32 候选；X/Y 或 X/Y/Z 轴限制；per-image prefix/suffix 矩形；256 px 门限；128 px 图像 halo；10% 轴 tie-break；20% 根框扩张；0.2% 空间 halo；`0.8 * raw_memory` 分裂比较；深度/低支持停止条件；CUDA allocator 缓存释放调用。

**运行测量闭环：** 7.013 GiB 自动得到 4 块；三条切线精确；undistort 投影 99.9900% 像素一致；四块 crop pixel load 精确；view 集合 Jaccard 0.969–0.992；四块串行以及块间显存回到约 0.9 GiB。

**高置信兼容推断：** undistort 内的 4:1 父照片图用于切分负载聚合、undistort 点框用于候选坐标、1368-view fresh projection 用于内存计费；子节点候选 envelope 使用 support≥2 点。兼容工具暂以原 MVS observation xy 模拟父照片负载。

### 15.3 可复刻伪代码

```python
budget = min(gpu_free_gib, 12.0, ram_available_gib)
root_owner = intersect(scene_roi, expand(union(cameras, sfm_support_ge_3), 20%))
root_candidates = bbox(all_relevant_undistorted_points_in(root_owner))

def recurse(owner_box, candidate_box, depth):
    memory_pixels = undistorted_crop_pixel_load(owner_box)
    if anchor_count < 100 and memory_pixels < 100_000:
        return
    if 0.8 * memory_pixels * bytes_per_pixel(level) <= budget_bytes or depth >= 10:
        save_leaf(expand(owner_box, 0.2%), memory_pixels)
        return

    candidates = 64_float32_uniform_positions(candidate_box)
    axis, cut = choose_by_parent_photo_load(candidates, tie_break=10%)
    left_owner, right_owner = intersect_parent_at_cut(owner_box, axis, cut)
    left_points, right_points = partition_points_at_cut(points, axis, cut)
    left_candidates = bbox(points_with_support_ge_2(left_points))
    right_candidates = bbox(points_with_support_ge_2(right_points))
    recurse(left_owner, left_candidates, depth + 1)
    recurse(right_owner, right_candidates, depth + 1)
```

这段伪代码是当前最小可用兼容方案。正式实现必须保留 `owner_box` 与 `candidate_box` 两套状态，否则子块 ROI 会错误缩到点包围盒，无法复现连续分区和厂商 halo。
