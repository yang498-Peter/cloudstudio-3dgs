# snow-20260827 MipMap Gaussian 初始化与训练静态实现审计

> 审计对象：`snow-20260827` 已完成任务及其实际调用的 MipMap Lite 二进制。
> 审计日期：2026-08-28（Asia/Singapore）。
> 方法：只读文件取证、PE 导入/导出表、常量恢复、函数级反汇编、产物格式解析、公开上游实现交叉核对。
> 安全边界：未启动、停止、修改、注入、调试附加或转储 MipMap 进程；未修改客户照片、LAS、任务目录和软件安装目录。

## 1. 结论先行

这次已经能把原报告 6.8 从“DLL 能力推测”推进到“实际任务分支与主要调用链”级别。

1. MipMap 的 Gaussian 训练器不是外部 Python 脚本，而是 `mipmap_gaussian_splat.dll` 内的原生 C++/CUDA + LibTorch 2.7.1/CUDA 12.8 实现。主程序调用链为：

   `reconstruct_full_engine.exe → mipmap_engine.dll → mipmap_gaussian_splat.dll → LibTorch/CUDA/OpenCV/Open3D/mipmap_classify.dll`

2. 初始化骨架与 Graphdeco 原始 3DGS 高度同源，但 MipMap 的 LiDAR 初始化已经恢复出明确改造：xyz 做 offset/scale 归一化；尺度取 7 邻域中除自身外 6 个欧氏距离的均值，线性三轴为 `[d,d,0.5d]`；法向来自 30 邻域局部协方差/特征向量式估计，再用 Z 轴到法向的最短弧生成 wxyz 四元数；RGB/255 后转 SH0；opacity 精确初始化为 `logit(0.1)`。二进制中同时存在 `simple_knn.cu` 风格内核和原始 rasterizer 风格的 `renderCUDA/preprocessCUDA/computeCov2DCUDA`。

3. MipMap 在这一骨架上增加了 LiDAR/平面几何分支：点云 normal/orientation 输入、`PlaneRasterizeGaussians`、mesh/depth/normal 监督、单目深度、语义分割、SIFT 相机细化、颜色协调、分块 Cut/Merge 和 SOG LOD。

4. 六个可训练张量的模型内存顺序已恢复为：

   `xyz → log-scale → quaternion → SH-DC → SH-rest → opacity-logit`

5. 本次 `reconstruct type 2 + High` 实际分支的六组 Adam 学习率已恢复：

   | 参数 | 初始学习率 | 末值/备注 |
   |---|---:|---|
   | xyz | `1.6e-5` | 指数调度末值 `1.6e-6` |
   | log-scale | `0.005` | 独立 Adam |
   | quaternion | `0.001` | 独立 Adam |
   | SH-DC | `0.0025` | 独立 Adam |
   | SH-rest | `0.000125` | 独立 Adam |
   | opacity-logit | `0.05` | 独立 Adam |

6. 最重要的新结论：本任务对应的 type 1/2 参数分支明确把 MCMC 开关写成 `0`，redundancy-cull 开关也为 `0`。训练循环因此调用 `AfterTrain` 而不是 `AfterTrainMCMC`，实际生命周期为梯度驱动的 `SplitGS → CloneGS → CullGS → opacity reset`；`RelocateGS/AddNewGS/CullGSRedundancy` 虽然编译在 DLL 中，但都不是本次执行分支。

7. 这意味着先前把 Tile 间局部密度变化解释为“可能由 MCMC relocation 主导”并不适用于本次运行。对本任务，更可靠的解释是：连续 mean/scale/SH/opacity 梯度优化，加上 gradient-driven split/clone 与 opacity/尺度/screen cull，最终形成每块不同的净 Gaussian 数量。

8. 基础照片 loss 的通式已恢复为 `(1-λ)×L1 + λ×(1-SSIM)`；本任务 `resolution_level=1` 对应 High，实际 `λ=0.4`，所以是 **`0.6×mean L1 + 0.4×(1-SSIM)`**。主训练调用给 `GetLoss` 的显式 RGB mask 是 undefined，故 snow 走 unmasked/mean-L1 分支；SegFormer mask 进入 renderer `forward`，并不是直接乘在 RGB loss 上。几何监督不是一个泛化 depth loss，而是 `DA2 mono-depth 0.5 + mesh/LiDAR depth 0.5→0.25 + mesh-normal 0.05 + 后期自洽 normal 0.01`。opacity-mean 为 `0.01`，sky-opacity 在后半程条件启用；single-view 仅在 planar 分支，本任务未执行，三条 scale loss 与另一条 opacity loss 权重为 0。

9. 标准 lifecycle 的主要判定也已恢复：从 step 500 起每 100 step 进入一次；gradient threshold 随 `resolution_level` preset 改变，High/level 1 本任务为 `1.5e-4`，level 2/3 为 `2e-4`，与 Tile 数量无关。大/小 Gaussian 按线性最大轴与 `0.2` 比较后分别 split/clone；候选还要求 `sigmoid(opacity)>0.15`。Split 每个父点生成 2 个带旋转随机偏移的子点，并把线性尺度除以 `1.6`；Cull 基准阈值为 opacity `0.05`、空间尺度 `0.2`、screen radius `0.15`。但任务没有保存中间 checkpoint，因此仍不能恢复每次事件数量和单粒子身份轨迹。

10. 训练内颜色/曝光 nuisance model 也已恢复：后半程按相机应用 `exp(a)*rgb+b`；另一条启用的 BilateralGrid 实际张量为 `[N_camera,12,8,16,16]`，自定义 Adam 的 LR/β/ε 为 `0.002/0.9/0.999/1e-15`，带 1/30 训练期 warm-up、随后指数衰减到 1% LR，并在每步以权重 `5.0` 累加 TV backward。12 通道高置信对应每格点 `3×4` RGB affine。SIFT pose refine 在中后段受质量门限控制。

11. High 的 `[5,10,5]` 是三阶段完整 view epoch 数，不是三个任意倍率。每个 epoch 对 `0…V-1` 用 `std::_Random_device → MT19937 → Fisher–Yates` 重新置换，因此每台相机每 epoch 恰好使用一次；最终 surface 产品实测为 DC-only，sky 为 degree-1，不能把结构体 `SH degree=1` 直接解释成 surface 终态一定使用一阶视角相关颜色。

## 2. 证据等级

- **A：直接运行证据**：任务 JSON、明文日志、文件时间戳、实际 PB/PLY/SOG、实际加载/生成的模型文件。
- **B：直接静态证据**：PE 导入/导出、函数反汇编、常量、调用地址、内嵌源文件路径和 CUDA kernel 名。
- **C：强交叉推断**：MipMap 二进制与公开 Graphdeco/Depth Anything V2/3DGS-MCMC 代码结构逐项一致，但没有厂商源码可逐行比较。
- **D：未证实**：仅存在符号但没有本任务分支调用，或需要运行时 checkpoint/trace 才能回答。

下文会明确区分这些等级。

## 3. 实际参与的文件

### 3.1 主二进制

| 文件 | 作用 | SHA-256 |
|---|---|---|
| `reconstruct_full_engine.exe` | 读取 task JSON、调度 AT/3D/后处理 | `B29CF993DB1E8FB8C14927F139256EEA0DBB8022468828C5744B7B453EA4C6C9` |
| `mipmap_engine.dll` | 分块、任务编排、MVS/点云/后处理桥接 | `46341599D23A5B39C5138B9BBD96375E10FF93D6186462B80939ED2D4F973815` |
| `mipmap_gaussian_splat.dll` | Gaussian 初始化、渲染、loss、反向优化、生命周期、Cut/Merge/输出 | `A910B39DBAD956DC35E9E436ACFD0FB8D92364E03BB44B0401CBEF6BCB8D492E` |
| `mipmap_classify.dll` | SegFormer 语义分割、TensorRT 单目深度 | `A480CC5C5620555C834785DE7A57F8FEECEEE75FC746BA5C766C0FA6FC063468` |

安装路径：`D:\Program Files\MipMap\MipMapLite\resources\resources\catch3d`。

`reconstruct_full_engine.exe` 不直接导入 GS DLL，而是从 `mipmap_engine.dll` 导入 `ReconstructBlockPartion/ReconstructBlockTask/PostProcess3D` 等接口，再由 engine 进入 GS 实现。这解释了为什么只看 EXE 导入表会漏掉训练器。

### 3.2 本次训练的直接输入与阶段产物

任务配置 `result\task\block_reconstruction_Tile_0.json` 直接证明：

- `pipeline_mode=1`；
- `resolution_level=1`，UI 对应 High；
- LiDAR 输入为 `G:/S1/USA/2026-02-24_16-21-11snow/process/2026-02-24_16-21-11snow_2/2026-02-24_16-21-11snow_colorized.las`；
- `remove_moving_object=true`；
- 生成 Gaussian PLY 和 SOG tiles；
- 四块各自有独立 ROI。

主要中间文件：

| 阶段 | 文件 | 内容/用途 |
|---|---|---|
| AT | `result\AT\mvs_undistort.xml` | 相机内外参、去畸变/投影相关数据 |
| 语义 | `result\milestones\classify\*.tif` | 342 张图的分类/动态对象掩膜证据 |
| MVS/视图 | `result\milestones\block_mvs\Tile_*.pb.bin` | 每块相机/视图和重建输入描述 |
| 初始化点云 | `result\milestones\point_cloud\Tile_*_point_cloud.pb.bin` | 每块进入 GS 的点云；层级 PNTS 点数与 LAS ROI 几乎 1:1 |
| 背景 | `gaussian_splat_background.pb.bin` | 100,000 个背景/天空 Gaussian |
| 块终态 | `result\milestones\splats\Tile_*\level_0.pb.bin` | 每块带 halo 的块级终态 Gaussian，56 B/vertex；snow 最终全部保留 |
| 最终产品 | PLY + SOG tiles | Cut/Merge/背景追加/LOD 后的产品 |

四块点数关系：

| Tile | 输入点数 | level-0 GS | GS/输入 |
|---|---:|---:|---:|
| 0 | 2,700,801 | 1,773,436 | 65.6633% |
| 1 | 1,520,716 | 1,988,450 | 130.7575% |
| 2 | 1,802,271 | 1,506,518 | 83.5900% |
| 3 | 1,221,675 | 750,498 | 61.4319% |

因此“固定比例抽稀 LiDAR”已经被运行结果排除。

### 3.3 AI 模型文件

| 文件 | 大小 | 任务中的证据 |
|---|---:|---|
| `da2_v1.onx` | 391,390,944 B | 单目深度加密模型 |
| `da2_v1_NVIDIA GeForce RTX 5070 Laptop GPU.ege` | 202,207,360 B | 12:31:58 在 Tile_0 训练期生成的 TensorRT engine |
| `seg_model_v1.onx` | 141,733,488 B | 语义模型候选 |
| `seg_v1.onx` | 110,344,128 B | 语义模型 |
| `seg_v1_...ege` | 77,023,296 B | 12:21:22–12:21:24 生成；随后出现 342 个 TIF |

模型实际位于 `C:\ProgramData\MipMap\MipMapLite\gdal_data`。其中本次使用的 `da2_v1.onx` SHA-256 为 `220ECB854F60710876FD19FAEAFB10E16F5CD359451577A0BCE081080DDF71AD`；任务期生成的 RTX 5070 Laptop GPU engine SHA-256 为 `91251D0466284AF9C35558987482EF0EA16DFE67E615C117C377393C6A109419`。engine 的 12:31:58 时间戳与本任务进入 3D/Gaussian 输入准备阶段一致，是本次实际编译/使用 DA2 模型的额外运行证据。

这些 `.onx` 不是标准明文 ONNX：头部近似高熵随机数据，ONNX Runtime 报 `InvalidProtobuf`；DLL 明文包含 `Failed to build engine from encrypted onnx`。因此可以确认模型被厂商加密/封装，不能从文件直接恢复网络图和 encoder 型号。

## 4. 内嵌实现来源与算法血缘

### 4.1 厂商构建路径

DLL 中保留了以下编译路径（B 级证据）：

- `...\src\mipmap_3dgs_c\bindings.cu`
- `...\src\mipmap_3dgs_c\plane_rasterizer\cuda_rasterizer\rasterizer_impl.cu`
- `...\src\mipmap_3dgs_c\spz\load-spz.cc`
- `...\mipmap_engine_common\proto\mvs.pb.cc`
- `...\mipmap_engine_common\proto\point_cloud.pb.cc`
- `...\mipmap_feature2d\proto\image_features.pb.cc`

这证明核心不是 Python 包装层，而是厂商自己的 `mipmap_3dgs_c` C++/CUDA 模块。

### 4.2 Graphdeco 3DGS 代码血缘

二进制含有：

- `simple_knn_cu::boxKnn/boxKnn2/boxMinMax/boxMeanDist/coord2Morton`；
- rasterizer 的 `renderCUDA/preprocessCUDA/computeCov2DCUDA`；
- SH 投影、协方差投影、tile rasterization；
- 与原始 3DGS 一致的 `RGB2SH + KNN scale + inverse sigmoid opacity + 六组 Adam + clone/split/cull` 结构。

这与 [Graphdeco Gaussian Splatting 官方实现](https://github.com/graphdeco-inria/gaussian-splatting) 以及其 [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) 逐项一致。应表述为“明确同源/基于同一代码家族”，但不能在没有源代码许可信息的情况下断言厂商逐字复制了某个具体 commit。

### 4.3 MipMap 自有扩展

MipMap 增加了原始 3DGS 没有的模块：

- `PlaneRasterizeGaussians` 和 `forward_planar`；
- LiDAR point normal/orientation 初始化分支；
- mesh/depth/normal/normal-gradient/scale-ratio 等监督接口；
- SegFormer 动态对象语义掩膜；
- TensorRT 单目深度；
- `RefineCameraPoseWithSIFT`；
- `ApplyColorHarmonization`；
- `Cut/SaveROI/MergeGSData`；
- `CreateLoD/CreateSogLOD`。

所以更准确的架构定义是：**Graphdeco 风格 3DGS 优化内核 + LiDAR/平面几何先验 + 厂商分块工程管线**。

## 5. Gaussian 初始化：已经恢复到什么程度

### 5.1 参数布局

`InitialParameters(PointCloud...)` 导出 RVA 为 `0xDE040`。结合初始化返回 tuple、`InitialOptimizer` 对每个模型偏移创建 Adam、以及 opacity/scale loss 对模型偏移的直接访问，可恢复：

| 模型偏移 | 张量 | PB/PLY 终态含义 |
|---:|---|---|
| `+0x00` | xyz | mean x/y/z |
| `+0x08` | scaling | 三轴 log-scale |
| `+0x10` | rotation | wxyz quaternion |
| `+0x18` | features_dc | SH0/DC 颜色 |
| `+0x20` | features_rest | 高阶 SH 系数 |
| `+0x28` | opacity | opacity logit |

`GetScaleLoss/GetScaleMeanLoss/GetScaleRatioLoss` 均直接访问 `model+0x08`；`GetOpacityLoss` 直接访问 `model+0x28`，这两项不是根据命名猜测，而是反汇编直接证据。

### 5.2 初始化伪代码

综合函数 `InitialParameters`（RVA `0xDE040`）及其三个内部 helper 的逐指令证据，可写成：

```text
input = Tile ROI point cloud derived from LiDAR

xyz = (input.xyz - supplied_offset) * supplied_scale

neighbors = KDTreeKNN(input.xyz, K=7)                # 包含 distance(self)=0
d = mean(sqrt(dist2[1:7]))                           # 排除自身，平均另外 6 个邻点
linear_scale = [d, d, 0.5*d] * supplied_scale
log_scale = log(linear_scale)

normal = LocalNormal(input.xyz, K=30)                # 邻域协方差/特征向量式估计
rotation = shortest_arc_quaternion(z_axis, normal)   # wxyz，特殊处理 ±Z/退化情形

features_dc = RGB_to_SH0(float(input.rgb) / 255.0)
features_rest = zeros(N, ((clamp(degree,0,4)+1)^2 - 1), 3)
opacity_logit = logit(full(N, 0.1))
```

这里的直接证据包括：

- `RVA 0xDE344–0xDE358` 把 K 常量 `7` 传给内部 KNN helper `0x13D050`；helper 对平方距离开根号，并排除第一个零距离后除以 6；
- `RVA 0xDE6F8–0xDE7A3` 将同一个 `d` 复制两次、第三轴乘 `0.5`，拼接后取 log；
- normal 分支在 `RVA 0xDE413` 写入 K=`30` 后调用 `0xD84C0`；
- `0xFA2A0` 明确调用 normalize、`linalg_cross` 和 dot，构造从 `[0,0,1]` 到 normal 的最短弧四元数，等价主式为 `q ∝ [1+dot(z,n), cross(z,n)]`，最后归一化并按 wxyz 排列；
- 对不可用/退化 normal 存在 epsilon `1e-6` 检查和确定性伪随机单位球 fallback；因此不能假定所有点都无条件沿同一输入 normal；
- `RVA 0xDEB32` 出现精确除数 `255.0`，随后 `0xF43B0` 执行 RGB→SH0；
- SH degree 被 clamp 到 `0..4`；初始 opacity 概率精确为 `0.1`，存储为 `logit(0.1)≈-2.1972246`；六个张量均进入 requires-grad 路径。

证据边界：

- “30 邻域 + 对称局部矩阵 + 特征向量求法向”是直接控制流和数值结构；将其简称为“PCA normal”是算法层强归纳，厂商没有保留源码变量名；
- fallback 的触发条件和生成公式可见，但真实 snow 每个点是否触发，静态分析不能统计；
- 初始点来自本任务的 LiDAR-derived Tile point cloud 是任务链直接证据；不能把它扩展成“永远直接读取原 LAS、没有任何上游 ROI/过滤”。

因此“LiDAR-derived 点云直接给出 mean/RGB，并通过局部密度与 normal 给出尺度/方向先验”已经从强推断升级为 B 级直接静态结论。

## 6. 优化器与本任务参数

### 6.1 六个独立 Adam

`InitialOptimizer` RVA `0xDDAB0` 依次为六个张量各创建一个 `torch::optim::Adam`。不是一个 Adam 的六个 param group，而是六个独立 Adam 对象。xyz 另外挂接 initial→final 学习率调度器。

反汇编中的模型 LR 偏移与张量映射：

| 模型 LR 偏移 | 张量 | 本任务值 |
|---:|---|---:|
| `+0x198` | xyz initial | `0.000016` |
| `+0x19C` | xyz final | `0.0000016` |
| `+0x1A0` | scale | `0.005` |
| `+0x1A4` | rotation | `0.001` |
| `+0x1A8` | SH-DC | `0.0025` |
| `+0x1AC` | SH-rest | `0.000125` |
| `+0x1B0` | opacity | `0.05` |

这里修正了只按原始 3DGS param-group 顺序猜名字可能造成的错配；真实模型张量顺序由 `GetScaleLoss` 和 `GetOpacityLoss` 的直接内存访问确认。

### 6.2 type 2 / High 参数构造

高层 Tile 训练函数在约 `RVA 0x65760` 构造 `GaussianSplatTrainingParams`，随后根据 reconstruction type 覆盖默认值。本任务日志明确为 `reconstruct type 2`，task JSON 为 High；对应 type 1/2 分支可确认：

#### 6.2.1 训练长度不是固定 30k

type 2 先读取 `Reconstruct3DParams+0x198` 并构造三段整数 schedule。启动器对同一个 `resolution_level` 字段的检查和显示字符串已经直接证明 `1=high、2=medium、3=low`；`CreateLevels` 又以 `+0x198` 的 1/2/3 值生成对应的 LoD 配置，因此这里不是“块数”分支，而是 reconstruction resolution/quality preset：

- `resolution_level=1`（High）：`[5,10,5]`；
- `resolution_level=2/3`（Medium/Low）：`[5,6,5]`。

内部 helper `0x5A360` 只是复制这三个整数。令三段为 `s0,s1,s2`，令 `V` 为该 Tile 训练视图容器的元素数，则：

```text
total_steps = (s0 + s1 + s2) * V
```

也就是 High 为 `20×V`，Medium/Low 为 `16×V`，不是固定 30k，也与一个任务切成多少 Tile 无关。结构体通用默认的 `30000` 会在 type 2 分支被这个动态 total 覆盖。

`V` 的来源也已从对象布局锁定：`GSTraining` 持有 `shared_ptr<MVSBlock>`，代码计算 `V=(MVSBlock.field_20-MVSBlock.field_18)/16`，即其中一个 16-byte 元素 view/image 容器的长度，而不是整项任务的照片总数。本任务四块 `block_mvs` 中可直接数到的唯一 `result/.temp/undistort/<id>.jpg` view path 分别是 656、644、607、595；MVSBlock protobuf schema 与后续消费者均支持它们一一对应训练 view。以这一高置信映射计算，High 的候选总步数分别为 **13,120、12,880、12,140、11,900**。目前缺少运行时内存 trace 来把“唯一 path 数=该 16-byte vector.size()”提升成逐字节实测，所以四个数仍标为高置信候选；`20×V` 公式和 High 分支本身是直接确定。

#### 6.2.2 视图调度不是有放回随机采样，而是逐 epoch 随机置换

`GSTraining::Run` 把 schedule vector 的当前元素作为 `BatchTraning` 的 epoch count 传入。`BatchTraning` 每个 epoch 都按当前 camera vector 长度分配 int64 索引数组并顺序填入 `0…V-1`，随后调用 `MSVCP140!std::_Random_device` 取 seed、按标准常量 `1812433253` 初始化 624-word MT19937 state，再执行 Fisher–Yates 等价交换，最后按这份 permutation 逐 view 训练。

因此 High 的 `[5,10,5]` 应准确解释为三阶段 **5、10、5 个完整随机视图 epoch**：每个阶段内每个相机每 epoch 恰好出现一次，只是顺序随机；总量自然是 `20×V`。这排除了“每 step 从所有相机有放回随机抽一张”以及“某些相机因随机性长期少采样”的实现。空间点的有效监督次数仍会因相机 frustum、遮挡、renderer/mono/mesh mask 不同而变化，但相机级抽样基数是均衡的。

#### 6.2.3 Gaussian 数量上限

type 2 的 cap 公式已从整数控制流恢复：

```text
max_gaussians = max(3,000,000,
                    min(500,000 * C,
                        10 * N_input))
```

`N_input` 是初始点云点数；`C` 来自高层配置的一个 double 倍率字段（`config+0x90`），其厂商字段名尚未恢复。其他 reconstruction type 可见 `700,000*C`，不能套到本次 type 2。本公式也解释了为什么 cap 不等于“输入点数固定乘某个小比例”。

继续向上追 `C` 时遇到的是明确的二进制边界，而不是尚未搜索到普通字符串：`reconstruct_full_engine.exe` 在 RVA `0x1AD25` 调入 `mipmap_engine.dll!Reconstruct3D`，但当前 `mipmap_engine.dll` 的常规 `.text/.rdata` raw size 为 0，实际载荷位于 `.ac1`、`.<^5` 与虚拟化/保护段 `.Tce`，导出入口不能按磁盘 RVA 直接反汇编。也就是说 `C` 的下游公式已直接恢复，上游字段名/preset 若不做合法的运行时去虚拟化或受控 A/B，就不能再由普通静态 xref 诚实推出；本文保留 `C`，不凭终态点数反算并冒充厂商配置。

#### 6.2.4 已命名的 type 2 字段

以下偏移相对 `GaussianSplatTrainingParams` 起始地址；名称由其消费者函数而不是常量外观确定：

| 偏移 | type 2 值 | 已确认用途 |
|---:|---:|---|
| `+0x04` | `(s0+s1+s2)*V` | total steps |
| `+0x0C` | 上述 cap 公式 | 最大 Gaussian 数 |
| `+0x14` | `1` | 传给 `InitialParameters` 的 SH degree；所以本任务实际为 degree 1 |
| `+0x20` | `0.4` | High/level-1 的 `lambda_dssim`；level 2/3 为 `0.2` |
| `+0x24` | `100` | densification interval |
| `+0x28` | `500` | densification start |
| `+0x2C` | true | opacity-reset enable |
| `+0x30` | `30` | reset interval |
| `+0x34` | `1.5e-4` | High/level-1 gradient threshold；level 2/3 为 `2e-4` |
| `+0x38` | `0.2` | split/clone 的线性最大轴分界 |
| `+0x40` | `0.2` | split 的 screen-space/max-radii 额外候选阈值 |
| `+0x48` | `0.2` | CullGS 空间尺度阈值 |
| `+0x4C` | `0.05` | CullGS opacity 阈值 |
| `+0x50` | `0.2` | opacity reset 的概率上限 |
| `+0x54` | `0.15` | CullGS screen-radius 阈值 |
| `+0x5B` | false | 选择普通 `forward`；true 才走 `forward_planar` |
| `+0x5C` | `0.015` | single-view loss weight |
| `+0x60` | `0` | scheduled scale loss weight，本任务关闭 |
| `+0x64` | `3000` | single-view loss 起始 step |
| `+0x68` | `1500` | scheduled scale loss 起始 step；但 weight=0 |
| `+0x6C` | `0` | scale-mean weight，本任务关闭 |
| `+0x70` | `0` | scale-ratio weight，本任务关闭 |
| `+0x74` | `0.5` | mesh/LiDAR depth 初始权重；中期后半减半为 `0.25` |
| `+0x78` | `0.5` | DA2 mono-depth 权重 |
| `+0x7C/+0x84/+0x80` | `s0V/(s0+s1/2)V/(s0+s1)V` | depth 开始/减半/结束 |
| `+0x88` | `0.01` | normal 分支 1 权重 |
| `+0x8C/+0x90` | `(s0+s1)V/total` | normal 分支 1 起止 |
| `+0x94` | `0.05` | normal 分支 2 权重 |
| `+0x98/+0x9C` | `0/total` | normal 分支 2 起止 |
| `+0xA0/+0xA4` | `0.04/total÷2` | sky-opacity 权重/开始 |
| `+0xA8` | `0` | 外部 normal 参考选择 mesh depth；值 1 才选择 mono depth |
| `+0xAC/+0xB0` | true / `total÷2` | 后半程在 render forward 中应用 color harmonization；尚无 `a,b` optimizer 证据 |
| `+0xB4` | `0.01` | opacity-mean regularizer |
| `+0xB8` | `0` | 另一条直接 opacity loss，本任务关闭 |
| `+0xBC…+0xD4` | 见 6.1 | 六组 Adam LR |
| `+0xD8` | true | normal-aware initialization；此 flag 直接传入 `InitialParameters` |
| `+0xD9/+0xDC/+0xE0/+0xE4` | true / `(s0+s1/2)V` / `(s0+s1)V` / `1` | SIFT pose refinement 开关、窗口和调用间隔；另有运行质量门限 |
| `+0xE8` | false | MCMC flag |
| `+0xE9` | false | redundancy cull flag；本任务关闭 `CullGSRedundancy` |
| `+0xEC/+0xF0/+0xF4` | block AABB min xyz | 从当前 `MVSBlock+0x88/+0x8C/+0x90` 逐 float 复制 |
| `+0xF8/+0xFC/+0x100` | block AABB max xyz | 从当前 `MVSBlock+0x94/+0x98/+0x9C` 逐 float 复制；同一 AABB 后续传给 `Cut` |
| `+0x104` | `0.1` | type-2 覆盖值；尚未把消费者唯一绑定到业务名 |
| `+0x108` | true | `remove_moving_object`；控制 renderer 额外排除 SegFormer label 33 |
| `+0x109` | true | 启用每相机 BilateralGrid 颜色/曝光校正路径 |
| `+0x10C/+0x110/+0x114` | `16/16/8` | 直接作为 BilateralGrid 三个网格维度传入构造器 |
| `+0x118` | `0.002` | BilateralGrid 自定义 Adam 更新的初始学习率 |
| `+0x11C` | `5.0` | 每步在 Adam update 前传给 bilateral-grid TV backward 的权重 |

`+0xEC…+0x100` 已不再是 opaque preset：构造代码从 `MVSBlock+0x88…+0x9C` 连续复制六个 float，布局正是 `AlignedBox<float,3>` 的 min/max；后处理又把相同 `MVSBlock+0x88` 作为 `Cut` 的第一个实参。`+0x108` 也已由 renderer mask 消费者绑定为 `remove_moving_object`。当前这段结构体真正尚未命名的只剩 `+0x104=0.1` 等少数值，以及高层 cap 倍率 `C`。另需明确纠正：`0x40A00000` 解码为 **5.0**，不是 10.0。

两项重要修正：

- `block_reconstruction_Tile_*.json` 确实没有这些核心字段；它们由 EXE/DLL 中的 type preset 和高层对象动态构造；
- 本任务 SH degree 不是“最多 4 所以可能 4”，而是 call-site 直接从 params `+0x14` 传入，type 2 实值为 `1`；normal-aware 初始化 flag 也为 true；
- `+0x5B=false` 排除了 snow/type-2 的 planar/single-view 分支；`+0xE9=false` 排除了 redundancy cull；`+0xE8=false` 排除了 MCMC relocation。这三个“功能存在但本次未启用”的边界已经从不确定项升级为直接静态结论。

## 7. 实际训练循环

### 7.1 主干顺序

函数级调用关系可简化为：

```text
for step in training_schedule:
    UpdateLearningRate(step)
    OptimizersZeroGrad()
    camera = select_training_view()

    rendered_rgb, depth, alpha, ... = forward()/forward_planar()
    base_loss = GetLoss(rendered_rgb, gt_rgb, ...)

    + optional GetSingleViewLoss(...)       # 仅 forward_planar；snow 未走
    + 0.5 * DA2 mono-depth loss
    + scheduled mesh/LiDAR depth loss(0.5 → 0.25)
    + scheduled GetNormalLoss(...)
    + optional opacity / opacity-mean loss
    + optional scale / scale-mean / scale-ratio loss
    + optional sky-opacity loss

    total_loss.backward()

    if mcmc_flag:
        AfterTrainMCMC()   # RelocateGS + AddNewGS
    else:
        AfterTrain()       # 本任务实际路径

    OptimizersStep(step)
    ShrinkBigScaleGS(...)
    optional RefineCameraPoseWithSIFT(...)

final CullGS(...)
```

注意：生命周期操作与 Adam step 的相对顺序以反汇编主循环为准；它不是一个仅在训练结束后统一 densify 的离线步骤。

### 7.2 基础照片 loss

`GetLoss`（RVA `0xD9D90`）已经恢复到算子级公式，不再只是与 Graphdeco“相似”：

```text
if mask is defined:
    L1 = sum(abs(pred - gt) * mask) / clamp_min(sum(mask), 1)
else:
    L1 = mean(abs(pred - gt))

L_base = (1 - lambda_dssim) * L1
       + lambda_dssim * (1 - SSIM(pred, gt, ...))
```

type 2 内部再按 resolution preset 覆盖 `lambda_dssim`：level 2/3 为 `0.2`，而本任务 High/level 1 为 `0.4`，所以 snow 基础项精确为：

```text
L_base = 0.6 * mean(abs(pred-gt))
       + 0.4 * (1 - SSIM(pred,gt))
```

本任务普通与 planar 两个主循环 call-site 都为 `GetLoss` 的显式 mask 参数构造默认/undefined Tensor；snow 实际走上面 `mean(abs(...))` 分支。SegFormer 生成的 mask 作为第二个 Tensor 输入传给 `GaussianSplatModel::forward`/`forward_planar`，影响 rasterization/forward，而不是作为这里的 RGB loss mask。`GetLoss` 返回的第二个 tensor 还用于记录/后续组合，但在没有变量名的情况下本文不强行给它命名；这不影响上面 total base loss 公式的确定性。

### 7.3 深度与法向

令三段 schedule 为 `s0,s1,s2`、训练视图数为 `V`，主循环中的本次 type 2 阶段可恢复为：

| 项 | 权重 | 生效窗口 | 直接证据 |
|---|---:|---|---|
| DA2 mono-depth | `0.5` | `MonoDepthInfo.valid=true` 且像素有效时 | `GetMonoDepth` + 独立 `GetDepthRegularizerLoss` call-site |
| mesh/LiDAR depth | `0.5` | `step>s0V` 到 `step≤(s0+s1/2)V` | `GetGTDepthFromMesh` + weight helper |
| mesh/LiDAR depth | `0.25` | 随后到 `step≤(s0+s1)V` | weight helper 在半程减半 |
| normal 分支 1：渲染自洽 | `0.01` | `step>(s0+s1)V` 到 total 前 | rendered depth→normal 对比直接渲染的 Gaussian normal |
| normal 分支 2：外部参考 | `0.05` | 近乎全程：`step>0` 到 total 前 | 本任务 `+0xA8=0`，参考来自 mesh depth→normal |

这表明训练阶段不是“所有几何 loss 从头固定加权”，而是大致：

```text
有效 DA2 全程：RGB/SSIM + mono-depth(0.5)
早期：再加 mesh-normal(0.05)
中期：再加 mesh-depth(0.5 → 0.25)
后期：再加 rendered-depth/Gaussian-normal 自洽(0.01)
```

#### 两路 depth 的真实数据流

`BatchTraning` 的导出签名直接接收：

- `shared_ptr<MeshRasterizerGPU>`；
- `map<unsigned int, MonoDepthInfo>`；
- `shared_ptr<BilateralGrid>`。

训练器不是先把两路 depth 融成一张图再做一个 loss，而是保留两条独立监督：

```text
DA2 encrypted model
  → TensorRTMonoDepth::InferRaw
  → per-view MonoDepthInfo(cv::Mat + valid/aligned flag + scale state)
  → GetMonoDepth(view_id,H,W)
  → resize/clone/device tensor
  → mono * model_scene_scale（仅 valid flag 为真）
  → GetDepthRegularizerLoss(rendered_depth, mono_depth, mono_mask, 0.5)

LiDAR-derived Tile point cloud / surface
  → MeshRasterizerGPU
  → GetGTDepthFromMesh(view)
  → mesh_depth * model_scene_scale + (depth != -1) mask
  → GetDepthRegularizerLoss(rendered_depth, mesh_depth, mesh_mask, 0.5→0.25)
```

因此本次参数不是“LiDAR 权重 1、DA2 仅弱 fallback”；可直接恢复的 scalar weight 是两路初始都为 `0.5`。但两路 mask、可用窗口、尺度对齐和像素覆盖不同，不能把“相同 scalar”误读成“相同有效梯度”。

在 mesh 路径进入后段后，代码进一步执行近似：

```text
mesh_mask &= rendered_depth < mesh_depth + 0.1 * scene_scale
```

也就是剔除渲染面明显落在 mesh 后方超过 `0.1*scene_scale` 的不一致/遮挡像素。mono 路径首先要求 `mono>0`，后段还把有效值约束到 `[0.1,1.0]` 的归一化范围。这里的阈值属于模型/归一化坐标，不应直接标成米。

两路 mask 的交并顺序也已恢复：

```text
mono_base_mask = float(mono_depth > 0)
if step < s0*V:
    mono_loss_mask = mono_base_mask
else:
    mono_loss_mask = mono_base_mask
                   * float(mono_depth >= 0.1)
                   * float(mono_depth <= 1.0)

mesh_loss_mask = mesh_valid_mask
if step > (s0+s1/2)*V:
    mesh_loss_mask *= float(rendered_depth
                            < mesh_depth + 0.1*scene_scale)
```

DA2 与 mesh 各自用自己的 mask 调两次 `GetDepthRegularizerLoss`；训练主循环没有 `mono_mask &= mesh_mask` 这种像素级求交。mesh 的角色一是在训练前为 DA2 做 affine metric alignment，二是训练中作为独立 depth/normal 监督。

`GetMonoDepth` 读取的是已经标定好的 cv::Mat 和 valid flag；本轮继续追到其上游 helper `RVA 0x62E50`，确认 snow 调用点 `0x61610` 把该 helper 的第五个布尔参数设为 false，从而选择“scale+shift”分支。完整标定公式见 9.1。这里已经不再是未知 P0。

#### 两条 normal 的真实语义

- 分支 1：`ComputeNormalFromDepth(rendered_depth)` 与 rasterizer 直接输出的 Gaussian normal 做 `GetNormalLoss`，是渲染深度几何与 Gaussian 椭球方向的自洽约束，权重 `0.01`，只在后期启用；
- 分支 2：把外部参考 depth 先转 normal，再和 Gaussian normal 比较。selector `params+0xA8` 为 1 时选 mono depth，为 0 时选 mesh depth；snow/type-2 实值为 0，因此本任务是 mesh-normal supervision，权重 `0.05`，近乎全程。

这意味着本任务实际几何约束不只是“mean 靠近 LiDAR”：mesh-depth 约束前后位置，mesh-normal 约束最短轴/表面方向，DA2 补充稠密深度，而较低 xyz LR 防止照片把 LiDAR 初始化整体拉飞。

#### depth regularizer 的像素公式

`GetDepthRegularizerLoss`（RVA `0xD8A50`）先对预测/参考 depth 使用同一个分段压缩，再做 masked L1。内部 helper `0xCAB50` 在本函数中使用固定分界 `M=1`：

```text
T(d) = 0.5 * d                  , d < 1
     = 1 - 0.5 / d              , d >= 1

L_depth = weight
        * sum(abs(T(d_pred)-T(d_ref)) * mask)
        / clamp_min(sum(mask), 1)
```

这不是直接在米制 depth 上做普通 L1：近处线性映射到 `[0,0.5)`，远处用 reciprocal 压缩到 `[0.5,1)`，会降低极远深度的绝对误差支配。

#### normal loss 的方向处理

`GetNormalLoss`（RVA `0xDB1C0`）先计算每像素 `dot(n_pred,n_ref)`；若 dot<0，就把参考 normal 取反，再计算 component-wise absolute difference。随后通过 `nan_to_num/isfinite` 和有效 mask 做归一化求和，最后乘阶段权重。核心可写为：

```text
n_ref_aligned = where(dot(n_pred,n_ref) < 0, -n_ref, n_ref)
delta = abs(n_pred - n_ref_aligned)
L_normal = weight * finite_masked_mean(delta)
```

因此它是对 normal 正负号不敏感的 hemisphere-aligned L1，而不是简单的 `1-dot`。这对点云/PCA normal 的朝向可能不一致非常实用。

`GetNormalGradientLoss` 与 `GetLossWithGradientWeight` 虽然导出，但主训练路径未找到直接 call-site；它们仍是编译能力/其他模式候选，不属于本任务已证实 loss。

### 7.4 其他约束

“存在 call-site”还不等于“本 preset 权重非零”。按本任务参数和主循环共同过滤后：

| 项 | 本任务值/阶段 | 判定 |
|---|---|---|
| `GetSingleViewLoss` | 参数保留 `0.015`/step≥3000 | **本任务未执行**：唯一 call-site 位于 `forward_planar` 分支，而 type-2 `+0x5B=false` 选择普通 `forward` |
| `GetOpacityMeanRegularizerLoss` | `0.01` | 实际启用 |
| `GetSkyOpacityLoss` | 后半程权重 `0.04`；直接输入为 `float(mono_depth>0)` | 条件启用，使用无单目有效深度区域作背景/sky 证据 |
| scheduled `GetScaleLoss` | step≥1500，但权重 `0` | 本任务关闭 |
| `GetScaleMeanLoss` | 权重 `0` | 本任务关闭 |
| `GetScaleRatioLoss` | 权重 `0` | 本任务关闭 |
| 另一条直接 `GetOpacityLoss` | 权重 `0` | 本任务关闭 |
| `ShrinkBigScaleGS` | 训练器有直接调用 | 后处理/约束操作，不应和 additive loss 混为一谈 |

`GetOpacityMeanRegularizerLoss` 的实现不是抽象的“透明度约束”，而是：

```text
L_opacity_mean = 0.01 * mean(sigmoid(opacity_logit)[active_slice])
```

它持续推动低贡献 Gaussian 变透明，再由 opacity cull 删除，是一种容量稀疏化机制。另一条关闭的 `GetOpacityLoss` 近似在 `p=0.5` 处惩罚最大、向 0/1 两端减小，用于鼓励二值化；本任务权重为 0。

所以更准确的结论是：DLL 具备 single-view、opacity、绝对尺度、轴比、sky 等完整正则能力；snow/type-2 实际 additive regularizer 以 opacity-mean、条件 sky-opacity、双 depth 和双 normal 为主，single-view、三条 scale loss 和直接 opacity-binarization 在本 preset 中关闭。

`GetSkyOpacityLoss` 的输入链也已闭合：call-site 传入的不是 SegFormer 原始类别图，而是 `mono_base_mask=float(mono_depth>0)`。函数先统计 `<0.5` 的无效/空洞像素比例，少于 1% 时直接返回 0；否则把 Gaussian 投影位置取整、裁到图像范围，在这些坐标采样 mask，并对 `sigmoid(opacity)` 构造 BCE 形态的正负对数项。因此可直接说它把“单目深度无有效值的图像区域”作为 sky/background opacity 监督代理；类别 33 是否正好是 sky 仍没有类别表证据，不能把二者等同。

### 7.5 SIFT 与颜色协调的位置

本轮把此前“颜色协调只是输出后处理”的判断修正为：**snow/type-2 有训练/render-forward 内颜色建模；没有找到最终 Merge 后再次校色的静态证据。**

#### 训练内 per-camera color harmonization

`params+0xAC=true`，从 `step≥total/2` 起把 model 的 color-harmonization flag 打开；普通 `forward` 和 `forward_planar` 都直接调用 `ApplyColorHarmonization`。其逐相机公式已经恢复为：

```text
rendered_rgb' = exp(a_camera) * rendered_rgb + b_camera
```

其中 `a,b` 来自模型保存的 per-camera tensor；无有效相机索引时保持 identity。`exp(a)` 保证增益为正。这条变换位于 render/loss 前向路径中，因此后半程照片残差会基于校色后的 render 计算。当前六个已恢复 Adam 都属于 Gaussian 参数，尚未找到 `a,b` 自己进入独立 optimizer 的直接证据，所以只能确认“训练前向内应用”，不能把 `a,b` 宣称为本任务中联合梯度学习；真正有单独 backward/update 证据的是下一条 BilateralGrid。

#### BilateralGrid

`params+0x109=true` 还启用另一条 per-camera `BilateralGrid` 路径。训练器把当前渲染送入自定义 CUDA bilateral-grid slice kernel，再把校正后图像用于 loss；backward 后把 slice 梯度传回原渲染，并单独更新 grid。它是可训练的空间/强度相关颜色校正，而非最终一次性 LUT。

这条支线的关键超参和每步顺序也已恢复：

- grid 构造器分配的主张量 shape 由五个 int64 直接组成：**`[N_camera, 12, 8, 16, 16]`**。`N_camera` 来自当前 block 的 camera vector 长度，`8/16/16` 来自 `+0x114/+0x110/+0x10C`，通道 `12` 是硬编码；因此它不是一个跨相机共享的小 LUT；
- `+0x118=0.002` 被转换成 double 写入该 grid 自己的 Adam options；同一 options 还直接写入 `beta1=0.9`、`beta2=0.999`、`epsilon=1e-15`。自定义 CUDA 符号含 `bilateral_grid_adam_update_kernel`；
- LR schedule 的总步数就是该 Tile 的 `total_steps`，warm-up 步数为 `max(1, floor(total_steps/30))`。warm-up multiplier 从 `0.01` 线性升至 `1.0`；其后按指数曲线在训练结束降到 `0.01`，故实际 grid LR 是 `0.002×multiplier`；
- `+0x11C=5.0` 每步先传给 RVA `0xFE080`。该函数的实参序列与嵌入符号 `bilateral_grid_tv_backward_kernel(const float*, float, float*, int,int,int,int)` 完全一致，随后从 grid 对象取四个维度并调用 CUDA launch wrapper；因此可直接确认这是 **TV backward 权重 5.0**；
- 之后依次执行自定义 Adam update、梯度清零/缓冲处理和 step/LR schedule 更新。嵌入 fatbin 同时含 `tv_forward_stage1/stage2`、`tv_backward`、`slice_forward/backward`、`init_identity` 与 `adam_update` kernels。

12 通道 + identity-init + slice forward/backward 与常见 bilateral color grid 的“每格点 `3×4` RGB 仿射矩阵”布局完全吻合：以空间 `(x,y)` 和强度/guide 轴做切片，再把 `[r,g,b,1]` 乘以 3×4 系数。这一 **3×4 解释为高置信结构推断**，不是导出符号里的字段名；guide 的精确公式和插值边界仍封装在 CUDA kernel。兼容实现可以先落实为“每相机 `[12,8,16,16]` grid、identity 初始化、自定义 Adam、LR 0.002 带 1/30 warm-up/指数衰减、每步 TV backward 权重 5.0”，而不是只写成笼统的颜色协调开关。

#### SIFT pose refinement

`params+0xD9=true`，允许在 `(s0+s1/2)V … (s0+s1)V` 窗口内、间隔 `1` 调用 `RefineCameraPoseWithSIFT`；主循环还有一个 `≥0.4` 的运行质量门限。因此“代码具备且 preset 开启”已确定，但本次每个视图实际成功 refine 的次数仍需运行日志/trace 才能统计。

对 `ApplyColorHarmonization`（RVA `0xD08B0`）做全 `.text` 直接 call xref 后，只发现 `GaussianSplatModel::forward` 与 `forward_planar` 两处调用；`MergeGSData`、`CreateLevels` 和 engine 的后处理导入链都没有该调用。因而不能再宣称“训练完成后块级再次执行 ApplyColorHarmonization”。复刻时应先实现已证实的训练内 per-camera photometric nuisance model；如果另做 Tile seam color solver，那属于兼容实现增强，不能标成已恢复的厂商步骤。

### 7.6 SH degree：参数为 1，但 surface 终态实际为 DC-only

type-2 的 `params+0x14=1` 确实传给 `InitialParameters`，表示模型为 degree 1 分配/初始化 SH-rest 容量；但最终文件提供了更严格的运行事实：

- `result\3D\model-gs-ply\gs.ply` 为 6,018,902 个 surface Gaussian，只含 xyz、f_dc、opacity、scale、rotation；
- `ue\gs_full.ply` 虽固定预留 45 个 `f_rest_*` float（degree-4 形状），对 100,000 个确定性样本检查，45 项全部精确为 0；
- `sky_full.ply` 的前 9 个 rest float 在约 99.365% sky Gaussian 上非零，剩余 36 个全零，恰好对应 RGB×3 个 degree-1 非 DC 基函数；
- 每块 `level_0.pb.bin` 均满足 `(size-4)/56 = vertex_count`，56 B 即 14 float：xyz3 + DC3 + opacity1 + scale3 + rotation4，不保存 SH-rest。

因此不能仅凭“degree=1”声称 surface 实际学到了 view-dependent SH。当前最强结论是：**本任务持久化的 surface 表示为 DC-only，sky 使用 degree-1；UE 的 degree-4 header 只是固定导出 schema。** 静态分析也没有找到 active-degree 递增调用。尚不能仅凭终态区分“surface SH-rest 从未训练”和“训练后导出时清零/丢弃”，兼容实现应先以 surface degree 0 为默认，再做受控 A/B 决定是否启用 degree 1。

## 8. Gaussian 生命周期：本任务不是 MCMC

### 8.1 标准分支

`AfterTrain` 导出 RVA `0xCC8B0`。type 2 的外层门控为：

```text
step >= 500
and step % 100 == 0
```

只有满足门控才进入本段 lifecycle。其候选构造可写为：

```text
grad = accumulated_xy_gradient / observation_count
eligible = norm(grad) > grad_threshold
           and sigmoid(opacity_logit) > 0.15

large = max(exp(log_scale), axis=1) > 0.2
small = max(exp(log_scale), axis=1) <= 0.2
screen_large = max_radii2D > 0.2

split_mask = eligible and large     # 另传 screen_large 作为附加候选/约束 mask
clone_mask = eligible and small
```

High/`resolution_level=1` 的 `grad_threshold=1.5e-4`；level 2/3 为 `2e-4`。只要当前 Gaussian 数未到 cap，标准顺序为 `SplitGS → CloneGS → CullGS → optional CullGSRedundancy`；但 snow/type-2 的 redundancy flag `params+0xE9=false`，所以本任务实际停在 `CullGS`，没有执行 redundancy cull。

#### Split 的实际变换

`SplitGS`（RVA `0xE2830`）不是简单复制：

```text
for each selected parent:
    create 2 children
    local_offset ~ Normal(0, exp(parent_log_scale))
    child_xyz = parent_xyz + rotate(parent_quaternion, local_offset)
    child_log_scale = log(exp(parent_log_scale) / 1.6)
    child_rotation/SH/opacity = repeated parent values
    remove parent
```

因此一次净数量变化是“删 1、加 2”，且子点围绕父点在父椭球方向中随机采样。常量 `2` 与尺度除数 `1.6` 均为直接反汇编证据。

#### Clone 的实际变换

`CloneGS`（RVA `0xD0A10`）对 small+high-gradient 候选复制同一 mean/scale/rotation/SH/opacity；复制时没有 Split 的随机 offset 和 `/1.6` 缩尺度。新旧副本随后再由 Adam 独立演化。

#### Cull 与 cap 模式

`CullGS`（RVA `0xD4880`）组合三类条件：

```text
sigmoid(opacity) < opacity_threshold
or max(exp(log_scale)) > space_threshold
or max_radii2D > screen_threshold
```

type 2 基准阈值为：

| 阶段 | opacity | space | screen |
|---|---:|---:|---:|
| total 前半程 | `0.10`（基准 0.05×2） | `0.2` | `0.15` |
| total 后半程 | `0.05` | `0.2` | `0.15` |
| 已到 cap/禁止继续 densify 的宽松模式 | `0.0125`（0.05×0.25） | `1.0`（0.2×5） | `0.75`（0.15×5） |

这里 cap 模式反而放宽删除阈值，避免在不能补充新 Gaussian 时进行过强 cull。可选 redundancy 路径存在并接收 cameras、两个动态阈值、常量 `3.0` 和邻域/采样常量 `30`；但 snow/type-2 明确关闭它，不能把 DLL 的能力写成本任务实际执行。

#### Opacity reset 不是把所有值设成 0.2

reset enable=true、内部 interval=30，但它位于外层 100-step lifecycle 门控内。因此自然 step 编号下两者交集是 300 的倍数，且要晚于 step 500（典型候选为 600、900、1200……）。实际算子是：

```text
opacity_logit = clamp_max(opacity_logit, logit(0.2))
```

也就是把 opacity 概率上限压到 `0.2`，低于 0.2 的 Gaussian 不会被抬高到 0.2。替换参数张量后代码还同步处理 opacity Adam state，避免 optimizer 继续引用旧张量。

DLL 日志字符串与上述控制流一一对应：

- `GS training clone point by grad`；
- `GS training split point by grad`；
- `GS training cull point by opacity`；
- `Reset Opacity step`。

因此本任务的数量变化机制已经不是抽象的“有 densification”，而是可以复刻为 high-gradient small→clone、high-gradient large→two-child split、三条件 cull 和周期 opacity cap-reset。

### 8.2 MCMC 分支

`AfterTrainMCMC`（RVA `0xCF580`）直接调用：

- `RelocateGS`；
- `AddNewGS`；
- 内嵌 CUDA `mipmap::mcmc_refine::compute_dead_mask_kernel` 与 `relocation_kernel`。

该实现与 [3DGS-MCMC 官方项目](https://ubc-vision.github.io/3dgs-mcmc/) 及 gsplat relocation 算子属于同一算法家族：低 opacity/dead Gaussian 可被重定位到高价值区域，并调整 opacity/scale 以维持渲染贡献。

但是训练主循环在 `mcmc_flag` 上二选一，本任务 type 2 参数明确写 `0`，所以不能用这个编译能力解释本次 Tile_0–3 的实际结果。

### 8.3 对已有几何位移观察的解释更新

本任务中，厘米级 mean 位移仍可由以下机制产生：

- xyz 的连续 photometric/depth/normal 梯度；
- clone/split 后子 Gaussian 在局部继续优化；
- cull 删除原位置低贡献 Gaussian，使终态空间分布看起来像“从 A 消失、在 B 出现”；
- Tile 独立训练与 Cut ownership 改变 overlap 中最终保留者。

但“单个 Gaussian 通过 MCMC 从 A 瞬移到 B”不是本任务所选分支。没有 checkpoint 时，也不能从终态最近邻关系反推出单粒子身份轨迹。

## 9. 单目深度与语义算法

### 9.1 单目深度

`mipmap_classify.dll` 导出：

- `TensorRTMonoDepth::MakeEngine`；
- `WarmUp`；
- `Infer/InferRaw`。

反汇编恢复的预处理：

- resize 到 `518 × 518`；
- OpenCV 颜色转换/通道拆分；
- 每通道减 `123.675, 116.28, 103.53`；
- 除 `58.395, 57.12, 57.375`；
- TensorRT 推理；
- 正输出执行 `1/output`，非正值写 0。

这些均等于 ImageNet `mean=[0.485,0.456,0.406]`、`std=[0.229,0.224,0.225]` 乘 255，并与 [Depth Anything V2 官方实现](https://github.com/DepthAnything/Depth-Anything-V2) 的默认 518 输入和归一化完全一致。结合文件名 `da2_v1`，可高置信判断为 Depth Anything V2 家族。由于 `.onx` 加密，不能确认具体 ViT-S/B/L/G encoder、输出 head 和厂商后处理改动。

GS DLL 直接链接/调用 `TensorRTMonoDepth`，不是只看见一个无关模型文件；本次生成的 `.ege` 进一步把模型、GPU 和任务时间绑定起来。训练数据通过 `map<view_id,MonoDepthInfo>` 留在内存中，未在 task result 里发现独立持久化的 depth/normal 图。342 张 `milestones\classify\*.tif` 实测为 `728×728`、8-bit、仅 0/255 的二值 mask，不能把它们误认作 DA2 depth。

#### DA2→metric mesh depth 的精确 RANSAC 标定

上游 helper `RVA 0x62E50` 在 `InferRaw` 后接收同视图的 mesh raster depth，并执行以下实际算法：

```text
pairs = {(x=mono_depth[p], y=mesh_depth[p]) | x>0 and y>0}
if len(pairs) <= 1000:
    valid = false
    return raw_mono

best_inliers = empty
repeat 2000 times:
    choose two distinct random pairs (x1,y1),(x2,y2)
    if abs(x2-x1) < 1e-6: continue

    a = (y2-y1)/(x2-x1)
    b = y1-a*x1
    if a <= 0.01 or a >= 100: continue

    inlier[p] = abs(y[p]-(a*x[p]+b)) / y[p] < 0.01
    keep the hypothesis with the most inliers

if best_inlier_count <= 0.05 * len(pairs):
    valid = false
    return raw_mono

# 对最佳内点做普通最小二乘重估，而不是直接采用两点模型
a = (N*sum(x*y)-sum(x)*sum(y)) / (N*sum(x*x)-sum(x)^2)
b = (sum(x*x)*sum(y)-sum(x*y)*sum(x)) / (N*sum(x*x)-sum(x)^2)

aligned[p] = a*mono[p]+b  if mono[p]>0 else 0
valid = true
return aligned
```

直接常量和控制流包括：有效 pair 必须超过 `1000`；RANSAC `2000` 次；斜率范围 `(0.01,100)`；相对误差 `<1%`；最佳内点必须超过总 pair 的 `5%`；最终对内点做 affine OLS refit。函数还编译了纯 scale 分支，但 snow 调用明确选择 affine `a*x+b`。

因此 DA2 在 MipMap 中不是直接相信的绝对深度，也不是只在无 LiDAR 区域盲填：它先用当前视图 rasterized mesh depth 做 robust metric alignment，只有标定通过的视图才把 `MonoDepthInfo.valid=true`，训练器才施加权重 `0.5` 的 mono-depth loss。这正是“LiDAR 锚定、视觉先验补密”的具体实现。

### 9.2 语义分割

DLL 直接导出 `SegFormerSeg`，任务期先生成对应 TensorRT engine，随后生成 342 个 TIF；task JSON 又启用了 `remove_moving_object`。因此该任务确实执行了 SegFormer 家族的图像语义/动态对象掩膜阶段。

训练迭代中实际读取 `Catalog::GetUndistortSegMapPath(image_id)` 返回的 TIF，并使用 OpenCV 构造：

```text
render_mask = (seg != 255)
if remove_moving_object:              # snow=true
    render_mask &= (seg != 33)
```

该 H×W mask 转为 GPU Tensor 后，作为第二个 Tensor 参数传入 `GaussianSplatModel::forward`/`forward_planar`。它不是 `GetLoss` 的显式 RGB mask；后者在本任务 call-site 是 undefined。由此可以确定：255 是 ignore/void 类，snow 额外排除 label ID 33，并在 rasterization/forward 层抑制这些像素；但没有类别表，不能把 33 武断命名为人、车或 sky。主循环也没有在这里做额外形态学膨胀/腐蚀调用。

另一个内部 helper（RVA `0xEB170`）把 GT RGB 用 `0.299R+0.587G+0.114B` 转灰度，通过两个轴向 roll 差分并计算 `sqrt(dx²+dy²+1e-6)`，随后以 `<0.01` 生成低图像梯度 mask。该 mask 被传入一条 normal loss，用于降低纹理/边缘处不稳定 normal 的影响。这是图像 gradient 实际进入几何监督的直接路径，但它不是公开导出的 `GetNormalGradientLoss`。

仍未知的是 SegFormer 的完整类别表、置信度阈值和 label 33 的语义。模型加密且任务配置无明文类别映射，所以本文只保留整数 ID 和可见的布尔组合。

## 10. 分块训练、Cut 与 Merge（静态能力与 snow 实际结果分开）

每个 Tile 先使用带 overlap/halo 的 ROI 独立构造点云和相机集合，生成独立 level-0 GS。后处理导出：

- `GaussianSplatData::Cut(AlignedBox<float,3> const&, SceneROI const&, double)`；
- `MergeGSData`；
- 背景/天空追加；
- `SavePly`；
- `CreateSogLOD`。

静态调用链能证明这些函数存在并被 `CreateLevels` 使用，但后续完整终态审计否定了“snow 用 core Cut 消除 halo”的早期解释。实际结果是：

```text
训练 halo + core
        ↓
Cut 按当前 block AABB / SceneROI 筛选
（snow 的有效 block AABB 仍覆盖导出 halo）
        ↓
MergeGSData 连接四块完整 level-0 数据
        ↓
背景追加
        ↓
最终 PLY
        ↓
CreateSogLOD / SOG tiles
```

这里“调用存在、块独立、Cut/Merge 顺序”是 A/B 级证据；训练内 per-camera `exp(a)*x+b` 已恢复，但没有最终 merge seam 校色 call-site。更强的运行证据是：最终 PLY vertex 数精确等于四个块 level-0 数量之和，并能按 Tile_0→1→2→3 分段逐属性精确匹配；因此 snow 的 Cut 没有把训练 halo 裁成互斥 core。

### 10.1 `Cut()` 的实际筛选机制

`GaussianSplatData::Cut`（`RVA 0x7D200`）和并行 mask lambda（`RVA 0x7CFB0`）进一步证明，它不是邻块 Gaussian 的加权融合器，而是 geometry-based ownership selection：

1. 若 `SceneROI` 的 XY 边界有 3 个以上顶点，先构造成 polygon、执行 geometry correct，再按第三个 double 参数做 buffer；圆弧离散参数为 `30`；
2. 计算 primary/secondary 3D AABB；
3. 对每个 Gaussian mean 做逐轴 inclusive 判断：`min_x≤x≤max_x`、`min_y≤y≤max_y`、`min_z≤z≤max_z`；
4. 根据 `SceneROI` 内部模式/有效性，继续要求 mean 位于 secondary AABB，或通过 buffered polygon 的 geometry predicate；这里没有额外 bool 实参；
5. 若提供有效 Z 范围，还要求 `z∈[z_min-margin, z_max+margin]`；
6. 对 mask=true 的 Gaussian，整条复制 xyz、颜色特征、opacity、scale、rotation 等数组到新 `GaussianSplatData`。

`CreateLevels` 中两个直接 call-site（level-0 训练后及后续 LoD 路径）都传入相同来源：`MVSBlock+0x88` 的 block AABB、`Reconstruct3DParams+0x1D8` 的 `SceneROI`，第三个 double 明确为 **`0.0`**。因此本任务没有调用方额外扩张的 polygon/Z margin；调用参数里也没有 owner ID 或相邻块索引。snow 的运行结果进一步说明这个 block AABB 是带 0.2% halo 的有效块范围，而不是互斥 ownership core。兼容实现若要逐产物复现，应保留 halo；若产品另行要求无双层边界，必须在自有发布/渲染层定义 half-open owner 或 blend，不能归因于本次厂商 Cut。

### 10.2 `MergeGSData()` 是纯 append，不是融合

`MergeGSData`（RVA `0x84800`）对两份 `GaussianSplatData` 的六组对应向量依次执行尾部插入；反汇编中可见 24/24/4/4/16/24-byte 元素步长的连续 append。函数内没有空间索引、距离计算、nearest-neighbor、颜色重估、scale/opacity averaging 或 seam solver 调用。

因此相邻 Tile 的 overlap 不是在终态“融合两层 Gaussian”；`MergeGSData` 机械连接被 Cut 选中的记录。本次被选记录仍是四块完整 level-0，halo 两层都进入最终 PLY，SOG 也保持四个独立 block。最终 seam 表现主要依赖共享照片监督、训练内 per-camera photometric 校正、halo 较窄和渲染器的块级加载行为，而不是 merge 后的 Gaussian 参数融合或已证实的 core ownership 裁剪。

## 11. 可以复刻的工程级算法框架

下面分为“已恢复的 MipMap-compatible preset”和“仍需替代实现的缺口”，避免把工程建议冒充厂商逐 bit 代码。

### 11.1 可直接照写的 High/type-2 核心 preset

```yaml
training:
  resolution_level: 1                # launcher 直接显示为 high
  stage_units: [5, 10, 5]
  total_steps: 20 * V
  max_gaussians: max(3000000, min(500000*C, 10*N_input))

representation:
  xyz: trainable
  log_scale: trainable               # linear scale = exp(log_scale)
  quaternion: trainable_wxyz
  color_surface: SH_DC_only_in_persisted_output
  opacity: trainable_logit

initialization:
  xyz: (point_xyz - tile_offset) * scene_scale
  scale_knn_k: 7                      # self + 6 neighbors
  scale_distance: mean(sqrt(d2[1:7]))
  linear_scale_axes: [d, d, 0.5*d]
  normal_knn_k: 30
  rotation: shortest_arc_quaternion(+Z, local_normal)
  rgb: RGB_to_SH0(rgb/255)
  opacity_probability: 0.1

optimizer_adam:
  xyz_lr_initial: 0.000016
  xyz_lr_final: 0.0000016
  scale_lr: 0.005
  rotation_lr: 0.001
  sh_dc_lr: 0.0025
  sh_rest_lr: 0.000125
  opacity_lr: 0.05

loss:
  rgb_l1: 0.6
  dssim: 0.4
  rgb_explicit_loss_mask: undefined  # snow call-site，因此 L1 走 mean 分支
  mono_depth: 0.5
  mesh_depth: 0.5_then_0.25_in_middle_window
  mesh_normal: 0.05_nearly_full_training
  rendered_depth_vs_gaussian_normal: 0.01_late
  opacity_mean: 0.01
  sky_opacity: 0.04_after_half_if_mask_available
  single_view: disabled_by_forward_mode
  scale: 0
  scale_mean: 0
  scale_ratio: 0
  opacity_binarization: 0

da2_metric_alignment:
  model_input: 518x518_ImageNet_normalized
  relation: mesh_depth = a*mono_depth+b
  minimum_positive_pairs: 1001
  ransac_iterations: 2000
  slope_range_open: [0.01, 100]
  inlier_relative_error: 0.01
  minimum_inlier_ratio_open: 0.05
  final_refit: affine_ordinary_least_squares_on_best_inliers
  invalid_view_behavior: disable_mono_depth_loss

lifecycle:
  densify_from_step: 500
  densify_interval: 100
  grad_threshold: 0.00015            # High/level 1；level 2/3 为 0.0002
  opacity_eligible_min: 0.15
  clone_if_max_linear_scale_le: 0.2
  split_if_max_linear_scale_gt: 0.2
  split_children: 2
  split_scale_divisor: 1.6
  cull_opacity: 0.10_first_half_then_0.05
  cull_max_linear_scale: 0.2
  cull_max_screen_radius: 0.15
  opacity_reset_outer_equivalent_interval: 300
  opacity_reset_probability_cap: 0.2
  mcmc_relocation: false
  redundancy_cull: false

photometric_nuisance:
  color_harmonization: exp(a_camera)*rgb+b_camera_after_half
  bilateral_grid: enabled
  bilateral_grid_tensor_shape: [N_camera, 12, 8, 16, 16]
  bilateral_grid_interpretation: per_voxel_3x4_RGB_affine_high_confidence
  bilateral_grid_adam: {lr: 0.002, beta1: 0.9, beta2: 0.999, eps: 1.0e-15}
  bilateral_grid_lr_warmup_steps: max(1, floor(total_steps/30))
  bilateral_grid_lr_multiplier: linear_0.01_to_1_then_exponential_to_0.01
  bilateral_grid_tv_weight: 5.0
  bilateral_grid_update: tv_backward_then_custom_adam_then_zero_grad_then_step_schedule
  sift_pose_refine: enabled_in_middle_late_window_with_quality_gate

tile_spatial_control:
  params_block_aabb: copied_from_MVSBlock_min_xyz_max_xyz
  cut_aabb: same_MVSBlock_aabb
  cut_scene_roi: project_SceneROI
  cut_margin: 0.0
  merge: append_only_no_dedup_no_parameter_blending

masks:
  renderer: (seg_id != 255) and (seg_id != 33_if_remove_moving_object)
  mono_early: mono_depth > 0
  mono_late: (mono_depth > 0) and (0.1 <= mono_depth <= 1.0)
  mesh_late_occlusion: rendered_depth < mesh_depth + 0.1*scene_scale
  normal_low_gradient: image_gradient < 0.01
  sky_proxy: mono_depth > 0             # loss 内使用其无效区域
```

这里所有数值均来自本任务 type-2 构造和实际消费者。关键修正是：该分支由 `resolution_level` 而非 Tile 数量选择；snow 的 `resolution_level=1` 明确是 High，所以使用 `[5,10,5]`、`20V`、DSSIM `0.4` 和 gradient threshold `1.5e-4`。

### 11.2 训练循环兼容伪代码

```text
G = InitialParameters(tile_point_cloud)
optim = six_independent_Adam(G)

for step in 1..total_steps:
    cam = next(random_permutation(0..V-1))  # 每 epoch 重建 permutation；无放回
    rgb, rendered_depth, alpha, rendered_normal, radii = render(G, cam)

    if step >= total_steps/2:
        rgb = exp(camera_gain_log[cam]) * rgb + camera_bias[cam]
    rgb = bilateral_grid[cam](rgb)                         # enabled preset

    loss = 0.6*mean_L1(rgb, image) + 0.4*(1-SSIM(rgb,image))

    mono, mono_valid = GetMonoDepth(cam)
    if mono_valid:
        loss += DepthLoss(rendered_depth, mono, mono_mask, 0.5)

    mesh_depth, mesh_mask = RasterizeLiDARMesh(cam)
    if mesh_depth_window(step):
        mesh_mask &= rendered_depth < mesh_depth + 0.1*scene_scale
        loss += DepthLoss(rendered_depth, mesh_depth, mesh_mask,
                          0.5 before midpoint else 0.25)

    mesh_normal = ComputeNormalFromDepth(mesh_depth)
    loss += 0.05 * HemisphereAlignedNormalL1(rendered_normal, mesh_normal)

    if late_window(step):
        rendered_depth_normal = ComputeNormalFromDepth(rendered_depth)
        loss += 0.01 * HemisphereAlignedNormalL1(rendered_normal,
                                                  rendered_depth_normal)

    loss += 0.01 * mean(sigmoid(opacity_logit))
    loss += optional_sky_opacity_loss(step, sky_mask)
    backward(loss)

    if step >= 500 and step % 100 == 0:
        grad = accumulated_xy_grad / observation_count
        eligible = norm(grad)>threshold and sigmoid(opacity)>0.15
        split(eligible and max(exp(scale))>0.2, children=2, divisor=1.6)
        clone(eligible and max(exp(scale))<=0.2)
        cull(opacity/scale/screen thresholds)
        if reset_enabled and step % 300 == 0:
            opacity_logit = min(opacity_logit, logit(0.2))

    six_Adam_step()
    update_Gaussian_learning_rates(step)
    if bilateral_grid_enabled:
        bilateral_grid_TV_backward(weight=5.0)   # 累加到 slice backward 得到的 grid grad
        bilateral_grid_custom_Adam_step()
        bilateral_grid_grad.zero_()
        update_grid_lr(linear_warmup_then_exponential_decay)
    optional_SIFT_pose_refine_in_window()

Cut(tile_export_aabb_and_scene_roi)  # snow 仍覆盖 0.2% halo，不是互斥 core
MergeGSData(all_full_tile_outputs)
append_sky()
export_PLY_and_CreateSogLOD()
```

### 11.3 仍不能照抄、必须自行补齐的参数块

1. **发布层边界策略**：Cut 实参已绑定到 `MVSBlock` block AABB、project `SceneROI` 和 margin `0.0`，但没有 owner ID/tie-break；snow 最终还保留了完整 halo。逐产物复刻应保留它，产品若需要无双层边界则必须另外定义唯一 owner/blend，不能假设 `MergeGSData` 会去重。
2. **训练长度中的精确运行 V 与高层倍率 C**：照 `High total=20V`、`cap=max(3M,min(500k*C,10*N))` 实现。V 的代码来源已锁定为 MVSBlock 的 16-byte view vector；PB 唯一路径数给出四块高置信 V，但仍应通过自建小任务/trace 做一次一一对应验证。C 的厂商字段名和 preset 来源仍未知，不能猜成固定 1。
3. **语义类别与 sky 代理含义**：mask 组合顺序已恢复；剩余的是 SegFormer label 33 的真实类别、阈值，以及“mono 无效区”与真实 sky/动态对象之间的误差。兼容实现应保存原始 seg、mono-valid 与最终各 loss mask，避免把不同来源混成一个万能 mask。

复刻器应额外保存每 100 step 的 checkpoint 摘要：GS 总数、clone/split/cull 数、opacity 分位数、scale 分位数、mean 位移、depth/normal/RGB 各 loss、相机校正量。MipMap 本次没有保留这些中间状态；新实现保留它们，才能真正验证 grow→cull 和几何修正，而不是只比较最终 PLY。

## 12. 尚未破解和下一步可做的工作

### 12.1 已经不需要继续猜的事项

- 训练器语言/运行时：原生 C++/CUDA + LibTorch；
- 初始化 xyz 归一化、K=7 scale、`[d,d,0.5d]` 三轴、K=30 normal、Z→normal 四元数、RGB/255→SH0；
- 初始化 opacity：0.1 概率、logit 存储；本任务 SH degree=1、normal-aware flag=true；
- 六个参数张量及 Adam LR；
- 基础照片项通式，以及 High 本任务的 unmasked mean-L1、0.6/0.4 配比；
- type 2 的动态总步数、Gaussian cap、双 depth/双 normal/opacity-mean/sky schedule；
- split/clone/cull 的候选公式、Split 的 2 子点与 `/1.6`、Cull 阈值、opacity clamp-reset；
- 有独立 MCMC relocation 分支；
- 本任务 MCMC flag=0；
- 本任务 single-view/planar 分支关闭、redundancy cull 关闭；
- 本任务 color harmonization 与 BilateralGrid 进入训练前向，SIFT refine preset 开启；
- 本任务 surface 终态为 DC-only，sky 为 degree-1；
- DA2 为 Depth Anything V2 家族；
- DA2 与 mesh/LiDAR depth 是两条独立 loss，初始 scalar weight 均为 0.5；mesh 另提供 normal；
- DA2 用 mesh 正值像素做 2000 次 RANSAC affine 标定，1% 内点阈值、5% 最低内点率，再 OLS refit；失败视图关闭 mono-depth loss；
- 语义为 SegFormer 家族；renderer mask 精确为 `(seg!=255)&(seg!=33)`；
- mono/mesh/sky/低图像梯度 normal mask 的组合顺序；
- Cut 实参来源与 margin=0；MergeGSData 为纯 append；snow Cut 后仍保留完整 halo，SOG 为四块独立索引。

### 12.2 仍需更深 SSA/受控实验才能进一步确认

- params `+0x104=0.1` 等少数剩余字段的厂商名和唯一消费者；`+0xEC…+0x100` 已确定为 block AABB，`+0x108` 已确定为 remove-moving-object，BilateralGrid 的尺寸/LR/TV backward 权重已确定；
- cap 公式中高层 double `C` 的业务字段名，以及 PB path 数与 MVSBlock view vector 的一次运行级一一对应验证；
- SegFormer label 33 的类别名、分类阈值和模型 head；
- sky-opacity 的 mono-invalid 代理对真实 sky/遮挡/无深度区的误分类率；
- surface SH-rest 是从未训练还是导出时清零；当前直接确认 params degree=1，但 surface 最终 DC-only；
- 每次 split/clone/cull/reset 的真实数量；本次无中间 checkpoint，不能从终态净点数反推；
- block builder 如何让相邻 inclusive AABB 获得唯一 owner；当前 Cut 无 owner ID，Merge 无去重，最终 merge 后也未找到校色调用；
- 加密 ONNX 的具体 encoder/head。

如果继续做，优先级应为：

1. 对一个极小自建任务做受控 quality/cap A/B，或在合法调试条件下观察已解包的 `Reconstruct3DParams+0x90`，恢复 cap 倍率 `C`；当前磁盘版 `mipmap_engine.dll` 的保护/虚拟化段阻断普通静态追踪；
2. 用极小自建点云做 High/type-2 A/B：改变图片数验证 V，改变 quality 验证 `[5,10,5]↔[5,6,5]`、DSSIM `0.4↔0.2` 和 grad threshold `1.5e-4↔2e-4`，同时保存每个 lifecycle 的 N/opacity/scale/mean；
3. 逆向 block builder 的 AABB 端点生成规则，确认共享边界是否半开；再做 surface SH、DA2、mesh-normal、BilateralGrid（含 TV=5.0）的同视角消融；
4. 若软件允许合法调试，再记录厂商自己的事件日志；不要对客户运行实例做注入或内存转储。

本轮为纯只读分析在工作区 `.tools/python` 安装了本地 Capstone 5.0.6，并使用自建 PE/RVA 反汇编脚本标注 LibTorch import；没有修改或加载 MipMap 安装目录中的任何文件。继续安装 Ghidra/IDA 的主要价值在剩余结构体字段、Cut/颜色协调和复杂主循环局部变量的可视化 SSA，不是推翻当前主结论。

## 13. 关键静态地址索引

以下 RVA 绑定到本报告开头给出的 DLL SHA-256；软件升级后不能直接复用：

| RVA | 函数 |
|---:|---|
| `0xDE040` | `InitialParameters(PointCloud...)` |
| `0x13D050` | 初始化 K=7 邻域距离 helper |
| `0xD84C0` | K=30 局部 normal helper |
| `0xFA2A0` | normal→wxyz shortest-arc quaternion |
| `0xF43B0` | RGB→SH0 |
| `0xDDAB0` | `InitialOptimizer()` |
| `0xDFB20` | `OptimizersStep(int)` |
| `0xE4360` | `UpdateLearningRate(int)` |
| `0xEC190` | `forward(...)` |
| `0xED9E0` | `forward_planar(...)` |
| `0xD9D90` | `GetLoss(...)` |
| `0xD8A50` | `GetDepthRegularizerLoss(...)` |
| `0xD8C10` | `GetDepthRegularizerLossWeight(int)` |
| `0xD8C40` | `GetGTDepthFromMesh(...)` |
| `0xDA710` | `GetMonoDepth(...)` |
| `0xD1ED0` | `ComputeNormalFromDepth(...)` |
| `0xDB1C0` | `GetNormalLoss(...)` |
| `0xDB6D0` | `GetOpacityLoss(...)` |
| `0xDBA00` | `GetOpacityMeanRegularizerLoss(...)` |
| `0xDC730` | `GetSingleViewLoss(...)` |
| `0xDC880` | `GetSingleViewLossWeight(int)` |
| `0xDBF80` | `GetScaleLossWeight(int)` |
| `0xCC8B0` | `AfterTrain(...)` |
| `0xCF580` | `AfterTrainMCMC(...)` |
| `0xE2830` | `SplitGS(...)` |
| `0xD0A10` | `CloneGS(...)` |
| `0xD4880` | `CullGS(...)` |
| `0xD5880` | `CullGSRedundancy(...)` |
| `0xDFDF0` | `RelocateGS(...)` |
| `0xD08B0` | `ApplyColorHarmonization(...)`，`exp(a)*x+b` |
| `0xA6F30` | `BatchTraning(...)`，接收 mesh rasterizer/MonoDepthInfo/BilateralGrid |
| `0xA72A1…0xA74AF` | 每 epoch 生成 `0…V-1`、RandomDevice/MT19937 播种并做无放回随机置换 |
| `0x62E50` | DA2 mono-depth↔mesh-depth RANSAC affine alignment helper |
| `0x65760` | type preset/`GaussianSplatTrainingParams` 构造；High 选择 `[5,10,5]`、DSSIM 0.4、grad 1.5e-4 |
| `0x65AAE…0x65B24` | `MVSBlock+0x88…+0x9C` → params `+0xEC…+0x100` 的 block AABB min/max 复制 |
| `0xEB170` | GT RGB 灰度化与轴向差分图像 gradient helper |
| `0xAB864…0xAB9E8` | type-2 BilateralGrid 启用、Adam options、相机数/总步数、`16/16/8` 构造实参 |
| `0xFC0B0` | BilateralGrid 构造；分配 `[N_camera,12,8,16,16]` 并 identity-init |
| `0xFE080` | BilateralGrid TV backward；权重从 params `+0x11C=5.0` 传入 |
| `0xFD790` | BilateralGrid 自定义 Adam update |
| `0xFE330` | BilateralGrid gradient `zero_()` |
| `0xFDDC0` | BilateralGrid LR warm-up/指数衰减 schedule |
| `0x7CFB0` | `Cut` 的并行 inclusive AABB/polygon/Z-range mask lambda |
| `0x7D200` | `GaussianSplatData::Cut(...)` |
| `0x5CE6C/0x5D588` | `CreateLevels` 中两处 Cut call-site；实参为 block AABB、SceneROI、margin 0.0 |
| `0x84800` | `MergeGSData(...)` |
| `0xB3910` | `CreateSogLOD(...)` |

## 14. 最终证据边界

本报告已经把“函数名存在”与“本任务实际分支调用”分开：

- **直接确定**：文件链、初始化主要参数、六个 Adam、LR、High/level-1 的 `20V` schedule 与 `0.6 L1+0.4 DSSIM` 照片 loss、标准 lifecycle、MCMC 二选一、本任务 MCMC=0、Depth Anything V2/SegFormer 家族、Cut/Merge 输出链，以及 snow 最终完整保留四块 halo。
- **本轮新增的直接确定**：DA2 mono depth 与 mesh/LiDAR depth 是两条独立监督，初始权重均为 0.5；normal 分别为后期自洽 normal 0.01 和近全程 mesh-normal 0.05；single-view/planar 与 redundancy cull 本次关闭；SegFormer renderer mask、mono/mesh/sky/低梯度 normal mask 的数据流；训练内 `exp(a)*x+b` color harmonization；BilateralGrid 的 `[N_camera,12,8,16,16]` 张量、LR `0.002`、TV backward 权重 `5.0` 与自定义 CUDA Adam；带门限的 SIFT refine；type-2 params 内 block AABB 与 Cut AABB 同源、Cut 的 SceneROI/margin=0 参数及 MergeGSData 纯 append；surface 最终 DC-only、sky degree-1。
- **强推断/兼容方案**：K=30 helper 属于通常所称的 PCA/local-covariance normal；高层 cap 倍率 `C` 和 `+0x104` 的业务含义仍按消费者约束保留。
- **仍未知**：每次 lifecycle 事件数量、`+0x104` 的业务名与 cap 倍率 C、SegFormer label 33/加密模型精确网络、每粒 Gaussian 身份轨迹、block builder 的共享边界唯一-owner 规则。最终 Merge 后没有发现 seam 校色调用，不能再把“seam 校色逐式实现”列成一个已知存在但尚未恢复的阶段。

因此可以把原 6.8 更新为：MipMap 的核心不是“把 LAS 固定转换为 surfel”，也不是本次任务中的 MCMC relocation；它以 LAS 点云作为高质量初始化，使用照片 + DA2 稠密深度 + LiDAR/mesh 深度/法向共同约束，在每相机颜色/曝光 nuisance model 下连续优化参数，并通过 gradient split/clone 和 opacity/尺度/screen cull 在每个 Tile 内重新分配 Gaussian 容量。后续运行产物证明 snow 的最终 PLY 是四块完整 level-0 的直接串接，SOG 也保持四块独立索引；没有执行可观察到的 core Cut/去重，不能再说它已用 Cut/Merge 隐藏 overlap 双层。

对工程复刻而言，训练骨架、High 的 schedule/照片权重、学习率、初始化、DA2 metric alignment、各 mask、BilateralGrid 的主要超参、lifecycle 阈值和 Cut/Merge 方式已经足够实现一个高相似 preset；离“严格复刻”最近的剩余障碍不再是 densification/depth/mask/Cut 是否存在，而是受保护 engine 内的 cap 倍率 C、`+0x104`、分块重投影/端点细节和没有中间 checkpoint 的事件计数。

## 15. 分块训练与显存管理专项补充（2026-08-28）

本节由 `divide_engine.exe` 新的静态反汇编、snow 全部 MVS/Tile 产物回放和 220 条 MemoryProfile 联合得到。详细可执行规范见 `snow-20260827-mipmap-tiling-vram-reproduction-spec.zh-CN.md`。

### 15.1 分块器已经从“推断”提升为直接定位

`divide_mode=2` 的主递归位于 `divide_engine.exe` RVA `0x3C690`。本任务 `pipeline_mode=1` 只允许 X/Y 轴，因此它是一棵平面递归 KD 树；每个节点根据可见 SfM 锚点在照片上的投影裁剪面积衡量负载，而不是按 LAS 点数或固定米数衡量负载。

直接恢复的常量和条件：

- 每个候选轴 64 条均匀候选切线；
- per-image 矩形原始宽、高必须分别至少 256 px；
- 有效矩形四周各扩 128 px 后裁到图像边界；
- 节点仅在 `anchor_count<100 AND pixel_load<100000` 时作为低支持节点停止；
- 最大递归深度 10；
- 候选轴至少为最长轴的 20%；
- 切线取左右像素负载差穿零处，零平台取中间，端点回退中心；
- 两个轴的最大子负载差达到 10% 时选择低负载轴；不足 10% 时选择空间跨度更长的轴。

snow 根节点 raw MVS 回放的 X/Y 最大子负载仅差 0.1% 以内，因此较长 X 轴胜出；实际结构为根 X，再对左右列分别沿不同 Y 切分。运行时 core 切线为 X `0.142307`、左 Y `3.243397`、右 Y `-0.492836`。

### 15.2 内存估值的 5.5 与 0.8 已经分清

RVA `0x3C7B9..0x3C87A` 计算：

```text
raw_memory = pixel_load * F(resolution_level)
comparison_memory = 0.8 * raw_memory
```

其中 level 0/1 的 `F=5.5 B/px`，level 2 为 1.375，level 3 为 1.1875。递归用 `comparison_memory` 与预算比较，叶记录的 `max_memory` 则是未乘 0.8 的 raw 值。此前把叶值解释为 4.4 B/px 不准确，本节正式更正。

用真实 core ROI 在 `mvs_undistort.pb.bin` 上回放，Tile_1/2/3 的 5.5 B/px 估值相对厂商 `max_memory` 误差分别只有 0.101%、0.150%、0.064%；Tile_0 误差 1.619%。这已足以把 5.5 公式视为运行级确认。

### 15.3 启动预算不是固定“8 GB 卡就分 8 GB”

RVA `0x4A7F1..0x4A8C6` 在启用 3D/GS 相关产品时调用 `mipmap_hardware.dll`：

```text
min_avali_memory_size = min(GetGPUAvailMemoryBytes(0)/GiB,
                            12.0,
                            GetAvailMemoryBytes()/GiB)
```

所以分块预算以启动时 GPU0 的**可用**显存为核心，同时受 12 GiB 和系统可用内存限制。`tiles.json.max_memory` 是照片裁剪工作集估值，不是 CUDA allocator 的硬分配量。

### 15.4 训练只串行驻留一个 Tile

按 point-cloud、level-0 和 levels_info 时间戳，Tile_0→1、1→2、2→3 之间分别有 88.27、118.13、59.83 秒非重叠间隔，没有并行块训练。训练期实测 CUDA 峰值分别为 4.53、4.56、4.41、3.44 GiB；前三块 LOD 完成后均回到约 0.9 GiB 基线。

GS DLL RVA `0xA6B13..0xA6B2B` 在每个偶数 global step 通过 `c10::cuda::CUDACachingAllocator::allocator` 调用同一个无显式附加实参的虚函数 `+0x60`；该调用也在对象销毁和若干后处理结束路径出现。结合 PyTorch allocator ABI，高置信对应 `emptyCache()`。这释放的是未使用缓存段，不会释放仍被 Gaussian/optimizer 引用的活跃 tensor。

### 15.5 当前可复刻边界

已新增两个只读工具：

- `.tools/replay_mipmap_adaptive_tiling.py`：解析厂商内嵌 protobuf descriptor，回放 KD 切分、crop 负载、halo 和内存估值；
- `.tools/audit_mipmap_tile_memory.py`：把日志采样和 Tile 产物时间戳对齐，输出训练/LOD 窗口及 RAM/VRAM 分位数。

兼容回放能得到正确的 `X→Y/Y` 拓扑和 4 块，但三条切线仍相差约 0.69–1.81 m，约为一个 64-bin 候选间隔。主要剩余差异是厂商 RVA `0x3E1F0` 会从所有相机模型重新投影并 round，工具目前复用 PB 已有 observation xy；另有 root pre-clip 和 poor-split retry 的少量端点/浮点细节。因此现在足以做可用兼容实现，但不能宣称任意场景边界与厂商位级一致。

## 16. CloudStudio 对位实现状态（2026-08-28）

### 16.1 门禁结论

本轮没有启动 CUDA/GPU 训练。历史 `TRAINING_READY` 已拆为：

- `UPSTREAM_DATA_READY`：仅证明 AT、Face4、mask、LiDAR depth、DA2、天空证据和 Tile 输入完成，固定 `training_allowed=false`；
- `TRAINING_IMPLEMENTATION_READY`：还必须证明完整算法消费、CPU 合同测试、短 GPU smoke 和零 unresolved blockers。

Snow 新门为 `outputs/snow-20260224-full-20260825/mipmap_upstream_data_ready_lidar_tiles_gate_v23y.json`，签名 `b5a660cc7e67225d11b2e8e254eb6900d0f519494c940569db4792cf30dd6731`。旧字面 `TRAINING_READY` 会被 Trainer 拒绝。

### 16.2 已落地并由真实数据证明的消费路径

- K=7/K=30 Tile 初始化几何由 Trainer 按 Tile ID、PLY SHA、NPZ SHA、点数、尺度比例和四元数归一性加载；
- Face4 Tile crop 同步作用于 RGB、renderer mask、LiDAR depth、DA2 depth、range scale 和 sensor coordinates，并执行 `cx-=x, cy-=y`；
- DA2 只在 RANSAC affine 有效视图以线性 range L1、权重 0.5 消费；
- 视图采样已提供每 epoch Fisher–Yates 无放回排列；
- snow 的经典 gradient lifecycle 已实现 opacity/gradient 候选门、0.2 m clone/split 分界、2 子点 `/1.6`、两段 opacity cull、world/screen cull 和 300-step reset，MCMC/冗余 cull 关闭。

新的真实联合消费审计为 `mipmap_tile_face4_renderer_da2_consumption_audit_v23w.json`，覆盖 5 Tile、2432 个重叠视图实例、1127 个唯一 Face4、约 51.93 亿 crop 像素；签名 `e7bccc65112c700df109d8733fa6b00d6f08f955791fc0294f74bc32da95c9f0`。

### 16.3 阶段调度与剩余阻塞

`mipmap_high_type2_loss_schedule_audit_v23x.json` 已把 5 Tile 的 `[5,10,5]` 阶段换算为确定性边界：

| Tile | V | 5V | 10V | 15V | 20V |
|---:|---:|---:|---:|---:|---:|
| 0 | 476 | 2380 | 4760 | 7140 | 9520 |
| 1 | 374 | 1870 | 3740 | 5610 | 7480 |
| 2 | 470 | 2350 | 4700 | 7050 | 9400 |
| 3 | 607 | 3035 | 6070 | 9105 | 12140 |
| 4 | 505 | 2525 | 5050 | 7575 | 10100 |

该工件签名为 `8a927d8070d8b51ad0c8f83c5b834c3a04db87ceec3e21ee66299a186c25aeb9`，但只是 schedule oracle，固定 `training_allowed=false`。仍未解除的核心阻塞为：

1. renderer mask 只证明可追溯布尔兼容 mask 被消费，尚无 label 33 已解析的 SegFormer 等价分割；
2. 缺 LiDAR-derived mesh rasterizer、稠密 mesh depth/normal、遮挡过滤和竞品分段压缩 depth L1；
3. 缺后期 rendered-depth/Gaussian-normal 自洽 loss 的同算法实现；
4. 缺 `[N_camera,12,8,16,16]` BilateralGrid 的精确 guide/边界；现有 PPISP 不是同一算法；
5. 缺质量门控 SIFT refine 的成功门限和运行消费；
6. Gaussian cap 高层倍率 `C`、halo 唯一 ownership、Cut/Merge 和最终原鱼眼评估/PLY/SOG/LOD 仍未闭环；
7. 100k SH1 目前只有初始化/暖启动附加能力，没有独立天空训练、验证选择和合并。

所以当前正式判定仍是 **NO-GO**。下一步优先实现并验签 mesh depth/normal cache 与 Trainer 逐像素消费；现有 sparse LiDAR range 和 nearest-normal anchor 不能冒充该路径。
