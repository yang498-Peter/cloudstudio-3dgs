# 竞品二进制与交付件审计（2026-08-29）

对象：`mipmap_gaussian_splat.dll`（36.3 MB，x64，链接时间戳 2026-07-03，静态链接 LibTorch 2.7.1 / CUDA 12.8，**C++ 符号未 strip**）与交付 PLY（`USAgs.ply`、`sky.ply`）。

授权范围：用户明确授权对该 DLL 做算法研究。全部为只读静态分析（导出表、导入表、RTTI、字符串、嵌入 CUDA fatbin 符号，以及针对生命周期的 capstone 反汇编），未运行、未修改、未分发该二进制。

方法学边界贯穿全文：凡是编译期常量或字符串可证的，标数值与证据；凡是宿主运行时填入的，一律标 `NOT RECOVERABLE`，不用推测数字冒充实测。逐项细节见同目录的 `report_A_lifecycle.md`、`report_C_loss_sh_sky.md`、`report_D_pipeline_lod.md`（副本在 `C:\Peter\3dgs-runs\house0305_dll_audit\`）。

## 1. 被推翻的既有假设

本节优先级最高：以下四条写在 `MIPMAP_ALIGNED_FACE4_PIPELINE_SOP.zh-CN.md` 里的判断，与交付件实测不符，以本节为准。

| 项 | SOP 旧表述 | 实测 | 影响 |
|---|---|---|---|
| 竞品表面高斯数 | `6,018,902` | **`22,452,075`**（`USAgs.ply` 头部 `element vertex`） | 容量基线差 **3.7×**；旧数字来自某个 level-0 分块而非完整交付件 |
| 竞品天空 | "独立训练 100k **SH1**" | **SH0**，`f_rest` 属性数 = 0，点数恰为 `100,000` | 我方"必须先实现 SH1 天空训练器"的阻塞项前提不成立 |
| 表面 SH | "参数结构 degree 1，终态 DC-only，训练期是否活跃未知" | 交付 PLY **无任何 `f_rest` 属性** | 见 §2 的完整裁决 |
| clone/split 尺度分界 | `0.2` | **`0.15`**（硬编码 f64 @0x349940，asm 0xcea95） | 我方分界偏大，同等梯度下更少点被 split |

`USAgs.ply` 与 `sky.ply` 均为 14 个 float 属性：`x y z f_dc_0..2 opacity scale_0..2 rot_0..3`（无法线）。我方导出为 17 属性（多 `nx ny nz`）。**跨文件统计务必按实际属性数解析，不能写死 stride**——本次审计初版即因写死 14 而读错列，得出"我方最长轴 1.59 m"的伪结论，改正后为 12.8 mm。

## 2. SH degree 裁决

证据两侧：

- 支持"训练期存在 SH-rest"：`InitialParameters(...)` 返回 **6 元张量**（xyz、features_dc、**features_rest**、scaling、rotation、opacity）；PLY writer 有 `property float f_rest_` 分支；光栅器同时暴露 `degreesToUse` 与 `degree`（活跃 vs 最大）；SPZ 编解码断言 `0 ≤ shDegree ≤ 3`。
- 支持"实际交付为 degree 0"：**两个交付 PLY 的 f_rest 属性数都是 0**。PLY writer 在有 SH-rest 时会写出对应属性，交付件里没有。

**裁决**：DLL **具备** SH0–SH3 的完整机制，但**本次交付的表面与天空模型都是 degree 0**。`shDeg` 是 `InitialParameters(PointCloud, float, Tensor, bool, int shDeg)` 的运行时实参，其取值 `NOT RECOVERABLE`。

因此：我方沿用 SH0 **与竞品本次交付一致，不是差距来源**。SH 提升可作为独立 A/B，但不能声称"竞品用了 SH3 所以我们必须跟"。

## 3. 形态与容量差距（"不锐利"的量化根因）

同一解析器、同一 log-scale 约定下的实测：

| 指标 | 竞品 `USAgs.ply` | 我方 D1 交付 | 差距 |
|---|---:|---:|---|
| 高斯数 | 22,452,075 | 7,453,193 | **3.01×** |
| 最长轴 P50 | **4.407 mm** | 12.818 mm | 我方粗 **2.91×** |
| 最长轴 P95 | 55.594 mm | 73.923 mm | 我方粗 1.33× |
| 最短轴 P50 | **0.428 mm** | 5.006 mm | 我方厚 **11.70×** |
| 轴比 P50 | **10.235** | 2.297 | 我方扁平度差 **4.46×** |
| 轴比 P95 | 59.627 | 12.080 | — |
| opacity P50 | 0.1974 | 0.5724 | — |
| opacity < 0.1 占比 | 21.38% | 33.84% | 我方死质量更多 |
| 包围盒 (m) | 113.4 × 117.9 × 33.8 | 761.7 × 665.6 × 638.7 | 我方有大量离群/漂浮 |

还有**第四个因素，且是我方自己引入的**：`experiments/runs.csv` 中 house0305 的 43 条记录里，41 条训练全部是 `f2_fisheye`；D1 与 W3 的 `run_manifest.json` 直接写着 `"factor": 2`。**我方每一次 house0305 训练都在半分辨率上进行**，从未在全分辨率上验证过。这单独就是 2× 的线性细节损失，且与上述三项相乘。

结论：用户反复反馈的"细节看不清、不锐利、高斯没有按细节自适应变小"是**四个乘性因素叠加**——训练分辨率减半、数量少 3×、单个粗 2.9×、厚 11.7×。竞品做的是**亚毫米厚的真薄片**（最短轴中位 0.43 mm），我方是 5 mm 胖团。我方 V48b 的"变薄"实验把轴比从 2.16 推到 2.60，方向正确但幅度远不够，目标是 **10.2**。

新 SOP 路线的 Tile 训练规格本身就是 `factor=1`（§9.9 的全分辨率 smoke 已验证可行，峰值显存 `1,137,484,800 bytes`），因此该项由路线切换自动修复。**但必须显式记住：不得再以"提速"为由退回 factor 2。** 全分辨率是 4× 像素量、约 4× 训练时间，这是必要成本而非可选项。

我方包围盒比竞品大一个量级，说明交付件里混入了远处漂浮物与天空壳；竞品表面模型本身是紧致的。

## 4. 竞品天空的精确配方（可直接复刻）

`sky.ply`：**100,000** 个高斯，**SH0**，包围盒 403.5 × 404.0 × 283.6 m，最长轴 P50 `2.056 m` / P95 `6.301 m`，最短轴 P50 `1.262 m`，轴比 P50 `1.521`（近各向同性），opacity P50 `0.1000`、P05 `0.0039`、低于 0.1 占比 44.7%。

训练侧：`GaussianSplatLoD::TrainBackground(shared_ptr<MVSBlock>, Reconstruct3DParams const&)` 是**独立于 `GSTraining::Run` 的静态入口**，配 `GetSkyOpacityLoss(Tensor& img, int, int, float)`；产物 `sky.ply` / `sky.sog` / `sky.splat` / `ue/sky_full.ply`。天空是**独立产品，不并入表面 PLY**——这一条与我方 V31 复盘结论一致（SH0 照片球壳并入主体会串色）。

## 5. 生命周期真实语义

`GaussianSplatTrainingParams` 是无 RTTI、无 JSON 键的朴素结构体，宿主填充后被拷入 `GaussianSplatModel`，致密化器读结构体字段（`movss xmm,[this+off]`）。因此**参数默认值不在 DLL 内**。

已恢复的字段映射与硬编码常量：

- `model+0xe8`（i32）= **绝对最大高斯数上限**，门控 `if (count >= cap) skip`（asm 0xce47b）。**不存在 `cap = C × initial_count` 的倍率形式**——我方文档里的"cap 倍率 C"是个不存在的概念，对齐时应匹配绝对上限。
- `model+0x100`（i32）= 致密化间隔（`iter % interval`）
- `model+0x11c`（f32）= 致密化梯度阈值 —— **`NOT RECOVERABLE`，宿主提供**。这反过来证实我方"必须现场按梯度分位定标"的做法是唯一正确路径（V42a 实测 P50/P95/P99 = 2.11e-5 / 7.71e-5 / 1.50e-4，据此定标 7.5e-5）。
- clone/split 分界 **0.15**（硬编码），split 子点 scale **÷1.6**，子点数推断为 2
- 透明度裁剪阈值为字段 `[0x124/0x128/0x130]`；纯裁剪步上分别缩放 **×0.25 / ×5.0 / ×5.0**；阈值 A 在**训练前半段 ×2**
- 冗余裁剪分位 0.9999；梯度 epsilon 1e-8；初始 scale 0.1；SH degree 钳位 0–3
- MCMC 路径独立存在（`AddNewGS` 里 `mulsd 1.05` = 每次 refine +5%）

**每个间隔步的执行顺序**（`AfterTrain` 控制流）：

```
iter % interval != 0            -> return
平均梯度
若启用 且 count < cap:
    Split   (scale > 0.15, 子点 ÷1.6)
    Clone   (scale <= 0.15)
OpacityCull
可选 RedundancyCull
```

> 注意：**Split 在 Clone 之前**。我方实现是 clone 先行，顺序不同会改变同一批父点的处理归属。此外这与我方 V37 复盘发现的 `backward -> Split/Clone/Cull -> Adam`（生命周期在优化器之前）是两个独立的顺序问题，都需对齐。

`SplitOrientationKernel(float4*, int, int*, int)` 取父点四元数，子点**继承父朝向**并沿父点旋转后的主轴位移。

## 6. 竞品实际消费的监督信号

与我方 §9.1 的"LiDAR 主导、DA2 与 mesh 权重固定 0"决策形成对照——竞品**两者都用**：

- **单目深度**：DepthAnything-V2 经 TensorRT（`da2_v1.onx`、`[MonoDepthEstimate]`、`Mono depth estimation completed. Cached N images`），经 `GetMonoDepth` 与 `MonoDepthInfo` 贯穿 `Run`/`BatchTraning`
- **Mesh 深度与法线**：`GetGTDepthFromMesh(MVSView*, MeshRasterizerGPU, map<uint,MonoDepthInfo>*)` → `tuple<depth,normal>`
- 两者共同喂入 `GetDepthRegularizerLoss` + `GetDepthRegularizerLossWeight(int iter)`（按迭代调度）

**逐项权重全部 `NOT RECOVERABLE`**（运行时 float 实参或 `Get*LossWeight(int)` 的返回值，非字符串、非立即数）。可确定的是**损失项集合**：

```
L =  w_rgb·[(1−λ_dssim)·L1 + λ_dssim·(1−SSIM)]      # GetLoss，带 mask 实参
   + w_depthReg(iter)·L_depthReg                    # 单目深度 + mesh 深度
   + w_singleView(iter)·L_singleView
   + w_normal·L_normal + w_normalGrad·L_normalGrad
   + w_opacity·L_opacity + w_opacityMean·L_opacityMeanReg
   + w_scale(iter)·L_scale + w_scaleMean·L_scaleMean + w_scaleRatio·L_scaleRatio
   + w_skyOpacity·L_skyOpacity
   + w_tv·L_bilateralGridTV
```

其中 **`GetScaleRatioLoss` 的存在直接印证我方 V48b 的"比例压平"方向正确**——竞品有专门的轴比损失项，这正是其轴比 10.2 的来源之一。

## 7. 外观模型：BilateralGrid 是真实实现

`mipmap::bilateral_grid` 命名空间下的完整 CUDA kernel 组：`slice_forward` / `slice_forward_chw` / `slice_backward` / `slice_backward_chw` / `tv_forward_stage1` / `tv_forward_stage2` / `tv_backward` / **`adam_update`** / `init_identity` / `accumulate_grad`。`BatchTraning` 直接接收一个 `BilateralGrid` 参数。

即：竞品的逐相机外观模型是**带 TV 正则、自带 Adam 优化器的可学习双边网格**。我方 PPISP 是受控光度模型，**不是等价实现**——这一点 SOP §9.17 已经写对，本次审计给出了正面证据。

**精确形状已证实为 `[N_cameras, 12, 8, 16, 16]`**，取自 `BilateralGrid` 构造函数（RVA `0xfc0b0`）分配的 rank-5 张量：`shape[1]` 是硬编码字面量 `12`（3×4 仿射颜色变换），其余维度取自 `GaussianSplatTrainingParams` 字段 `0x10c/0x110/0x114`，其**内建默认值 `16, 16, 8`** 从参数构造函数 `0xa37ba` 恢复（guidance/深度分箱 8，网格 H×W 16×16）。维度可被调用方覆盖，但这组默认值与 gsplat 的经典双边网格一致。

其余已恢复要点：

- 每格 3×4 仿射，逐像素**三线性切片**（HWC 与 CHW 两种输出布局都已接线），`init_identity` 单位初始化
- 逐相机索引直接用**训练图像下标**（守卫串 `BilateralGrid::apply/backward: image_idx out of range`），无学习式查表
- TV 平滑损失真实接线（两阶段前向归约 + 反向，损失例程 `0xfe19d`）；TV 权重数值 `NOT RECOVERABLE`（候选默认 `params[+0x11c]=5.0`，仅为相邻配置位推断）
- **自带融合 CUDA Adam**（非 `torch::optim::Adam`），含 m/v 与梯度累加器，**LR 默认 0.002**（`params[+0x118]` = f32 `0x3b03126f`，与 gsplat 2e-3 一致）；betas/eps `NOT RECOVERABLE`

**重要更正**：字符串 `"Exposure … Range -4.0 to +4.0"` **不是厂商模型**，而是 exiv2 内嵌的 Adobe Camera Raw / XMP `crs:` 元数据样板（周围是 `CropUnits`、`AutoExposure`、`Exposure2012` 等）。竞品**没有独立的逐图曝光标量**，全局亮度被吸收进仿射网格的偏置列。我方若要对齐，需要的是网格本身，而不是再加一个曝光参数。

## 8. 端到端流程（用于流程固化对照）

`GaussianSplatLoD(PointCloud, Mesh, Scene, Reconstruct3DParams, outDir)` 编排：

0. 场景过滤 `CreateFilteredScene`
1. 单目深度（DA2 ONNX）
2. 天空/语义分割（`[CreateSegModel]`，逐图去畸变 seg map 经 `Catalog::GetUndistortSegMapPath(imageId)`，分类器在 `mipmap_classify.dll`）
3. 图像去畸变
4. 逐块高斯初始化（`[InitialGS]`，`Initialize GS scene with scale:`）
5. **逐块训练**（`GSTraining(MVSBlock)` → `Run` → `BatchTraning`），含 §5 生命周期与 §6 损失
6. 逐块 splat 导出（`Catalog::GetBlockSplatsPath(name, index)`）
7. **独立天空训练**（`TrainBackground`）
8. 合并并构建 LoD 层级（`CreateLevels`、`levels_info.json`）
9. 层级切片成瓦片（`CutGltfLoD`/`CutSogLoD`，`Cut(AlignedBox, SceneROI, margin)`）
10. 写 3D Tiles（`tileset.json`、`KHR_gaussian_splatting[_compression_spz_2]`、refine=REPLACE）+ `lod-meta.json`

**分块规则**：`MVSBlock` 是 protobuf 消息，自包含相机、图像、稀疏点、观测，以及 `bounding_box` 与 `tight_bounding_box` —— **两者之差即 halo/重叠余量**，与我方 core+halo 设计同构。每个 `MVSBlock` 起一个 `GSTraining`，**块间串行独立**。块数与切分几何由上游 `mipmap_engine_util.dll` 决定，本 DLL 内不可恢复——因此**我方按自身 LiDAR 可见性得到 5 块、竞品 4 块，是数据差异而非实现差异**，不得为凑数强改。

训练期还有 `GSTraining::RefineCameraPoseWithSIFT(SIFTDetector, SIFTMatcher, Camera&, cv::Mat, Tensor, Tensor)`（private），即在训练循环内用 SIFT 精修相机位姿。

## 9. label 33 结案

本 DLL **不含任何 class-id→name 表、类别枚举或数据集名**（cityscapes / ade20k / mapillary / segformer / mask2former / upernet / num_class / class_names 全部无命中）。`mipmap.engine.message` protobuf 是摄影测量元数据，不是语义分类体系。

seg map 由外部预计算，经 `Catalog::GetUndistortSegMapPath(unsigned int imageId)` 按图索引读入；分类模型在 `mipmap_classify.dll`。255 是惯例的 ignore id，**33 的语义在本二进制之外**。

结论：**label 33 在此对象上已穷尽，不再作为未闭合项挂账**。我方继续使用可追溯的 `圆形/FoV 有效 & ~人物动态` 布尔掩膜，不伪造未知多类别语义——该做法不变。

## 10. 仍未闭合（需要更强手段或不同对象）

- 全部 loss 权重与 `Get*LossWeight(int)` 调度曲线（宿主/编译期，需在权重加载点反汇编）
- `GaussianSplatTrainingParams` 的字段名与默认值（宿主侧）
- opacity reset 的数值与间隔（经指针表加载，邻近无立即数）
- 实际使用的 `shDeg` 实参、迭代总数、LR 调度、致密化起止步
- BilateralGrid 的 Adam betas/eps、精确 TV 权重、设备端切片数学（在 SASS 中）
- 分块数/切分几何（上游 DLL）
- `RunScaffold` 的确切角色

## 11. 对我方路线的直接影响

按可执行性排序，全部需以受控 A/B 验证，不得直接并入基线：

0. **全分辨率训练（`factor=1`）不可再退让**。这是四个乘性因素中唯一完全由我方自设、且修复代价明确（约 4× 训练时间）的一项。新 SOP 路线默认如此，只需守住。
1. **容量目标从"匹配 6.0M"改为"匹配 22.4M"**。这是最大的单点差距，且直接对应用户的画质反馈。
2. **薄片形态是第一优先级**：最短轴中位 5.0 mm → 目标 0.43 mm，轴比 2.3 → 10.2。竞品有专门的 `GetScaleRatioLoss`，我方 V48b 已验证该方向安全（覆盖不降）但强度远不够。
3. **生命周期顺序对齐两处**：Split 先于 Clone；生命周期先于 Adam。
4. **clone/split 分界 0.2 → 0.15**。
5. **透明度裁剪的分步缩放语义**（纯裁剪步 ×0.25/×5/×5、前半段阈值 A ×2）是我方从未实现的机制，可能正是 V33a/V40 系列"裁剪要么崩塌要么无效"的缺失环节。
6. **天空按 100k / SH0 / 各向同性 / opacity P50 0.10 复刻**，独立导出。
7. DA2 与 mesh 深度在竞品是**在用**的；我方权重 0 的决策基于"LiDAR 是更强真值"这一自有理由，可保留，但不能再写成"竞品也不用"。
8. 我方交付件的离群漂浮物需在导出前按场景 ROI 裁剪（竞品包围盒紧致，且流程里有显式 `Cut(AlignedBox, SceneROI, margin)`）。
9. **外观模型要么补齐要么明说不补**：若要对齐，需实现 `[N,12,8,16,16]` 逐相机网格 + 单位初始化 + 三线性切片 + TV 损失 + 独立 Adam(lr 2e-3)；PPISP 无法表达 guidance 相关的空间变化校正。这一项影响的是跨相机色彩一致性，不直接影响锐度，优先级低于 §11.1–2。
