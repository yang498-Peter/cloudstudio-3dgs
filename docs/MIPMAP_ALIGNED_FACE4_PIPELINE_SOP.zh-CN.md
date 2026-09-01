# MipMap 对位 Face4 训练流水线强制 SOP

更新日期：2026-08-28
状态：`MANDATORY / FAIL-CLOSED`

## 1. 结论

当前不能宣称 CloudStudio 的雪堆结果整体优于竞品。

- 我们的最新 v23 前半段已经做到：时间同步检查、原始鱼眼有效圆与人物掩膜前置、342/342 原图 AT、两组共享单焦距 KB4 内参、独立逐图位姿、签名训练数据集和 1368 张 Face4。
- 竞品当前完整结果仍领先：SegFormer renderer 有效性、DA2 单目深度、LiDAR/mesh 深度与法线约束、4 个空间 Tile、自适应高斯 split/clone/cull、独立天空、PLY/SOG/LOD 全部完成。
- 我们 v23 新路线尚未训练。当前可比较的我方最终 PLY 是旧 v22j，因此不能把“新 AT 更好”直接等同于“最终高斯更好”。
- 竞品未提供与我方同一组 36 个原始鱼眼验证视角的 PSNR/SSIM/LPIPS，而且两份 PLY 坐标系尚未对齐；禁止用不同口径数字宣布胜负。

## 2. 当前实测对比

| 项目 | CloudStudio | MipMap | 判断 |
|---|---:|---:|---|
| AT 重投影 RMSE | `1.315863 px` | `1.251524 px` | 我方约高 `5.14%`；特征/轨迹口径不同，属于近似对比 |
| POS 修正 P50 / P95 / 最大 | `1.008 / 2.316 / 3.240 cm` | `2.099 / 4.546 / 5.443 cm` | 我方改动更小不自动等于更准，可能也是欠修正 |
| 最终相机中心差异（双方 AT）P50 / P95 | `1.340 / 3.038 cm` | 基准 | 同量级但并不相同 |
| 最终旋转差异 P50 / P95 | `0.362 / 0.678°` | 基准 | 仍需改善/解释 |
| 表面高斯数 | `6,158,096`（旧 v22j） | `6,018,902` | 数量接近，不能代表质量接近 |
| 长轴尺度 P50 / P95 | `8.242 / 26.820 mm` | `10.465 / 89.649 mm` | 我方中位更薄，竞品允许更强多尺度表达 |
| 最大长轴 | `31.356 m` | `1.996 m` | 我方存在必须裁掉的超大异常高斯 |
| opacity P05 / P50 | `0.0213 / 0.2381` | `0.0547 / 0.1149` | 我方低分位低于 `0.05`，更容易形成雾/透明空洞 |
| Tile / 自适应拓扑 | 已完成 LiDAR 可见性驱动的 5 Tile 签名计划，尚未训练 | 4 Tile，合计 `6,018,902` | 竞品已有成品；我方计划链已补齐 |
| 天空 | 已完成独立证据掩膜与 SH1 100k 初始化，尚未训练 | 独立训练 100k | 竞品已有成品；我方输入链已补齐 |
| 最终交付 | PLY 与我方质量报告 | PLY + SOG + LOD + Tile | 竞品领先 |

我方可保留的优势是输入身份签名、时间同步证据、人物掩膜在匹配前明确生效，以及更细的中位表面高斯；这些优势必须继续进入后半段，不能代替后半段。

## 3. 不可跳过的固定顺序

以下顺序是唯一允许进入正式训练的产品路线。每一步必须产生签名清单、统计和上游 SHA256 绑定；任一项 `FAIL`、`SKIPPED`、`NOT_RUN` 或身份不一致都必须停止。

1. `input_preflight`：原始鱼眼、POS、LiDAR、时间戳、双相机 ID 和文件哈希完整。
2. `time_sync_audit`：先做相机时间偏移扫描；若最优偏移不是 0，先修正时间戳并从第 1 步重跑。
3. `raw_circle_mask`：在原始 `2912×2912` 鱼眼上按主点绘制半径 `1200 px` 有效圆，圆外不得参与 AT 或训练。
4. `raw_person_mask`：人物/移动目标掩膜与有效圆组合，公式固定为 `circle_valid & ~person_dynamic`。
5. `masked_feature_matching`：在遮罩后提取 ALIKED 特征并用 LightGlue 匹配；CUDA、保留/删除特征数和三份清单身份必须写入运行清单。
6. `known_pose_triangulation`：用 POS 初值做已知位姿三角化；注册图数必须等于输入图数。
7. `shared_single_focal_kb4_at`：只在原始鱼眼上联合优化逐图独立 pose 与左右两组共享 KB4 内参；Face4 不是自由 AT 相机。
8. `accepted_training_manifest`：只有收敛且通过门限的 `candidate_model` 才能发布签名训练 Manifest。
9. `rebased_masks_and_split`：AT 改变数据签名后，圆形掩膜、人物掩膜、train/val split 必须重新绑定；旧清单禁止混用。
10. `face4_rgb_and_person_mask`：AT 完成后再生成 `N×4` 个固定虚拟针孔面；每个面必须携带虚拟内外参及组合掩膜。
11. `renderer_dynamic_mask`：生成并验签 Face4 renderer 有效性掩膜。竞品最终规则已恢复为 `(seg!=255)&(seg!=33)`，但 label 33 类别名仍未知；我方使用可追溯的 `圆形/FoV有效 & ~人物动态` 等价布尔掩膜，不伪造未知多类别标签。
12. `new_at_lidar_depth`：按已接受的新 AT 重新投影 LiDAR 深度；旧 AT 深度清单禁止复用。
13. `da2_monocular_depth`：生成 DA2 相对深度，并签名记录尺度对齐和有效区。
14. `independent_sky_background`：天空/远景独立建模，不得让表面高斯用超大尺度代替天空。
15. `spatial_tile_plan`：按空间拆分 Tile，并为每个 Tile 固化 LAS 子集、相机子集及重叠区。
16. `tile_gaussian_training`：各 Tile 按 High/type-2 执行 `20×Tile视图数` 步；参数结构为 SH degree 1，但表面终态为 DC-only，现有证据不能区分 surface SH-rest 从未激活还是导出时丢弃；RGB 为 `0.6 mean-L1 + 0.4(1-SSIM)`，DA2 `0.5`、LiDAR/mesh depth `0.5→0.25`、mesh normal `0.05`、后期自洽 normal `0.01` 和 opacity-mean `0.01`。snow 分支固定使用 gradient split/clone/cull/reset，禁止启用 MCMC relocation 或 redundancy cull 冒充竞品。
17. `raw_fisheye_evaluation`：回到原始鱼眼验证集评估 PSNR、SSIM、LPIPS、深度、覆盖率、空洞、漂浮点和尺度异常。
18. `ply_sog_lod_export`：只有评估门通过后才导出表面 PLY、天空、Tile、SOG/LOD 和最终质量报告。

## 4. 强制门禁

`cloudstudio_3dgs/pipeline/mipmap_gate.py` 定义上述有序阶段。门禁使用规范化 JSON 的 SHA256 签名，阶段只能是固定序列的连续前缀，不能交换或漏项。

- `FACE4_BASE_READY`：只表示第 1–10 步完成，`training_allowed=false`。
- `RENDERER_MASK_READY`：第 11 步完成，renderer mask 的 train/val Manifest 与 Face4 SHA256 完全绑定，仍不允许训练。
- `LIDAR_DEPTH_READY`：第 12 步完成，完整新 AT 深度与训练 Dataset、圆形 mask 和 LAS SHA256 完全绑定，仍不允许训练。
- `DA2_DEPTH_READY`：第 13 步完成，train/val DA2、Face4、LiDAR 深度、模型和 RANSAC 规则全部验签一致，仍不允许训练。
- `SKY_BACKGROUND_READY`：第 14 步完成，train/val 天空证据与独立 SH1 初始化全部绑定，仍不允许训练。
- `UPSTREAM_DATA_READY`：第 1–15 步全部完成且上游身份匹配；仍固定 `training_allowed=false`，只说明数据准备完成。
- `TRAINING_IMPLEMENTATION_READY`：除上游身份外，还必须验签完整 High/type-2 实现合同、CPU 合同测试和短 GPU smoke，且 unresolved blockers 为空；只有此状态才允许 Trainer 启动。
- `TrainerConfig.validate()`：当数据来自 `independent_pos_prior_shared_single_focal_kb4_at_v2` 时，缺门禁、门禁未就绪、缺 Face4 或身份不一致都会直接报错。
- 训练后第 17–18 步仍是正式产品门；训练跑完不等于质量通过。

当前雪堆签名门禁：

- 文件：`outputs/snow-20260224-full-20260825/mipmap_upstream_data_ready_lidar_tiles_gate_v23y.json`
- 状态：`UPSTREAM_DATA_READY`
- `training_allowed=false`
- 签名：`b5a660cc7e67225d11b2e8e254eb6900d0f519494c940569db4792cf30dd6731`
- 下一步：补齐并验签训练实现合同；当前禁止启动长训练。
- 历史文件 `mipmap_aligned_training_ready_lidar_tiles_gate_v23q.json` 的字面 `TRAINING_READY` 已废止，新的 `verify_training_gate()` 会明确拒绝它。

## 5. 参数基线与证据边界

当前 v23e AT 的可复现参数/结果包括：

- 双鱼眼，两组共享 `single_focal_kb4`；逐图独立 pose；
- Huber 尺度 `2.0 px`；最多 15 轮内参/pose 交替并已收敛；
- POS 先验 sigma 为 `[0.03, 0.03, 0.06] m`，不是已经验证通过的“统一 2 cm”；
- 2 cm 先验只能作为独立 Stage 2 对照实验，不能无对照替换产品基线；
- 竞品 High/type-2 已直接确认参数结构 degree 1，而表面终态为 DC-only、天空为 degree 1；不能把“最终表面 PLY 无 SH rest”解释成训练全程必然只用了 SH0，也不能反向宣称 surface SH1 在训练中一定活跃。
- High/type-2 的 densification 从 step 500 开始、每 100 步运行；gradient threshold `1.5e-4`，候选 opacity `>0.15`，尺度分界 `0.2`，split 两子点并 `/1.6`，cull opacity/scale/screen 阈值为 `0.05/0.2/0.15`，每 300 步把 opacity 上限 reset 到 `0.2`。

竞品已经确认运行的部分是 SegFormer renderer mask、DA2、四 Tile、gradient split/clone/cull/reset、独立天空和最终多格式交付。保留下来的 342 张 `milestones/classify/*.tif` 只有两种 SHA256、各 171 张，且只有 `0/255`，所以不能把它们写成逐图多类别结果；未知 label 33 语义继续标为未知。

## 6. 明令禁止的捷径

- 禁止先生成 Face4 再把 4N 个面当独立自由相机做 AT。
- 禁止只在训练时遮人物，而让人物特征进入 AT。
- 禁止跳过时间同步或把非零最佳偏移当成普通噪声。
- 禁止把旧 pose 下的 LiDAR 深度接到新 AT 数据集。
- 禁止用全局固定 615.8 万拓扑冒充分 Tile 自适应训练。
- 禁止用固定大球壳或超大表面高斯冒充独立天空训练。
- 禁止只看平均 PSNR；P10、LPIPS、深度、空洞、低 opacity、超大尺度和 LiDAR 漂浮距离必须一起检查。
- 禁止在竞品与我方坐标未对齐、验证视角不一致时宣布“我们更好”。

## 7. 本轮问题记录

问题现象：过去曾在 Face4 语义、新 AT 深度、DA2、Tile 和天空未完成时直接试训，导致透明空洞、雾状低 opacity 高斯和超大尺度异常，且无法与竞品完整路线公平比较。

修改文件：

- `cloudstudio_3dgs/pipeline/mipmap_gate.py`
- `cloudstudio_3dgs/data/depth_cache.py`
- `cloudstudio_3dgs/data/mono_depth.py`
- `tools/build_mipmap_frontend_gate.py`
- `tools/advance_mipmap_lidar_depth_gate.py`
- `tools/build_da2_face_cache.py`
- `cloudstudio_3dgs/training/trainer.py`
- `tests/test_mipmap_gate.py`
- 本 SOP

修改内容：新增签名有序阶段门禁、真实前半段验签工具、Trainer fail-closed 接口、完整 LAS 紧凑深度缓存、LiDAR 深度门、DA2 Small + 竞品 RANSAC affine 对齐缓存、独立天空证据/SH1 初始化、LiDAR 可见性自适应 Tile 计划及严格串行 CUDA 缓存策略。

验证方式：真实雪堆前半段 342 原图、342/342 注册、306/36 split、1224/144 Face4 和 renderer mask 全部逐文件验签；新 AT 深度覆盖 342/342，使用完整 `7,036,347` 点 LAS，342 个 NPZ 哈希全部通过。门禁已推进至 `LIDAR_DEPTH_READY`。DA2 双面探针复现了一个通过、一个因 `4.78%<5%` 内点率而 fail-closed 的视图。

当前状态：第 15 步已完成。DA2 train 为 `1123/1224` 标定通过，val 为 `80/144` 通过；天空证据 train 为 `438/1224`、val 为 `17/144` 视图通过；LiDAR 可见性重建 `1,710,000` 个规划锚点和 `1,607,923` 个 Face4 观测，按 6.5 GiB 保守预算生成 5 个串行 Tile。下一步是补齐训练实现，不是正式 Tile 长训练。

## 8. 天空与 LiDAR 分块实现纠正（2026-08-28）

问题现象：最初兼容回放直接使用照片 AT 的 `75,112` 个稀疏点做空间锚点。它能复现竞品纯视觉分块器的控制流，但雪堆已有 703 万点 LiDAR；稀疏点外包围盒受离群点影响扩大到约 300 m，且不能完整表示真实可见表面。

修改文件：

- `cloudstudio_3dgs/data/sky_background.py`
- `cloudstudio_3dgs/pipeline/adaptive_tiling.py`
- `cloudstudio_3dgs/pipeline/face4_observations.py`
- `cloudstudio_3dgs/pipeline/lidar_face4_observations.py`
- `cloudstudio_3dgs/training/tile_scheduler.py`
- `cloudstudio_3dgs/training/trainer.py`
- `tools/build_independent_sky_background.py`
- `tools/build_lidar_face4_adaptive_tile_plan.py`
- `tools/advance_mipmap_sky_gate.py`
- `tools/advance_mipmap_tile_gate.py`
- `tests/test_adaptive_tiling_and_sky.py`

修改内容：天空不再把固定球壳追加到表面检查点，而是从 Face4、已标定 DA2 远距和世界方向生成独立证据掩膜，并准备 degree-1 SH 的 100k 独立初始化。分块采用混合路线：完整 LAS 头确定场景 ROI；每张原鱼眼的新 AT LiDAR z-buffer 均匀抽取 5000 个可见深度样本，反投影到世界坐标后再投影到四个 Face4，由这些 LiDAR 可见性观测计算每块照片裁剪与像素负载。AT 仅提供已优化相机/内参，不再把视觉稀疏点当最终空间锚点。

验证方式：天空 train `438/1224`、val `17/144` 视图通过，独立初始化含 `100,000` 个 SH1 高斯；LiDAR 分块使用 `1,710,000` 个深度锚点，形成 all/train `1,607,923/1,435,928` 个 Face4 观测，最终为 `X→Y` 主拓扑下 5 个叶块。相关 Python 语法检查通过，`tests/test_adaptive_tiling_and_sky.py`、`tests/test_mipmap_gate.py` 等通过。

当前状态：LiDAR 版 Tile 门现已重新签为 `UPSTREAM_DATA_READY` 且 `training_allowed=false`。竞品 4 Tile 是其纯视觉 MVS 观测、运行时预算和 ROI 的结果；我方更密的 LiDAR 可见性在相同保守预算下得到 5 Tile 是预期差异，禁止为凑数强制改成 4 Tile。

### 8.0.1 四块兼容计划重新验收（2026-08-30）

问题现象：后续 V71/V72 证明五块全 halo 拼接会新增一条边界，并在重叠区留下几乎完全重合但颜色、opacity 冲突的双层 Gaussian。跨块 0.1 mm 内近邻比例达到 90.255%，而竞品仅为 0.0082%。因此“禁止四块”只适用于当时未经显存实测的保守判断，不能继续作为质量主线结论。

修改文件：

- `tools/build_lidar_face4_adaptive_tile_plan.py`
- `tools/build_v73_four_tile_raster_smoke.py`
- 本 SOP

修改内容：LiDAR 分块器新增显式 `--force-depth` 兼容模式，与预算模式互斥。`--force-depth 2` 重新使用已签名 Face4/LAS 可见性计算切线、裁窗和负载，形成 `X→Y/Y` 四叶树；没有直接合并旧 Tile 文件。默认生产调用仍按实测预算自适应，不受影响。另新增最大叶块的 factor=1 fail-closed raster smoke 构建器。

验证方式：新计划签名为 `192ca762757d6b74ea142bb1882fe645a91f5bb2c4ae6af70279f9ef8d2a3b11`，四块视图数为 624/470/607/505，halo-inclusive LiDAR 数为 2,851,911/1,477,056/1,850,945/1,076,290。四块 K7/K30 初始化几何已生成并签名为 `cd4505b57a7a919810f2eac69226e6bdcc16709abdd749a613e22b8ac9cdd466`。最大 Tile_0 在 2,851,911 个初始 Gaussian、624 个全分辨率 Face4 视图下完成 1-step factor=1 raster smoke，峰值 VRAM 1,943,262,720 bytes，无 OOM/NaN。该 smoke 还发现孤立 LiDAR 邻域令 K7 最大切向尺度达到 7.537 m，第一步发生 281 次 0.2 m clamp；V73 配置因此必须在初始化阶段先应用 0.2 m 上限，禁止巨型初始 Gaussian 污染前 500 步梯度。

当前状态：四块路线已通过输入、初始化与最大块显存可行性门；初始化即应用 0.2 m 上限的复测中，初始/最终最大轴均为 0.200000003 m、world clamp 为 0，run manifest 签名为 `ef037a3731fa5080d802d80eb8c017eb34bf3d53ca30f52989c3f524617fd223`。但该结果尚未授权质量长训。必须先补齐四块各自的严格 mesh/DA2 sidecar，再完成单 Tile step 502 生命周期门和相邻双 Tile 接缝门；不得从 smoke 直接跳到四块全量。

### 8.0.2 Tile_0 mesh/DA2 生产顺序修复（2026-08-30）

问题现象：新的 raw mesh raster 完整包含 624 个 Face，但它的 `source_type=1` 只表示“未执行跨视图分类”。若直接送入只允许 source 2/3 的严格 admission，输出会全部为空，即使部分记录仍显示 `mesh_depth_enabled=true`。

修改文件：`tools/build_v73_competitor_boundary.py`、`results/diagnostics/snow-v73-competitor-plus-roadmap.zh-CN.md`；生产工具继续使用 `tools/filter_mesh_geometry_cross_view.py` 和 `tools/build_strict_mesh_admission.py`。

修改内容：固定顺序为 `raw mesh(1) → cross-view classification(2/3/4) → reject 4 → whole-view P95≤0.10 m → DA2 mesh-native RANSAC`。V73 动态门禁同时按 Tile 实际视图数生成 10V/15V/20V 步数，并按初始 Gaussian 数生成边界 cap，禁止复用旧 374 视图常数。

验证方式：Tile_0 跨视图统计为 source 2 共 109,059,467 像素、source 3 共 324,397,484 像素、source 4 共 643,786,409 像素；严格输出启用 416/624 个视图并保留 330,093,921 个有效像素；DA2 RANSAC 成功 388/624。签名 step-502 配置通过 `TrainerConfig.validate()`。普通 Python 入口首先在首帧前暴露当前 gsplat `csrc.pyd` 缺少 EWA 算子；没有生成 checkpoint。符号探针确认 `csrc.3dgs-only.backup.pyd` 同时包含 EWA projection 与 raster 算子，使用预加载入口重启后已越过 step 60，DA2/mesh-normal loss 非零，峰值显存 3,826,780,672 bytes。

当前状态：`TILE0_COMPETITOR_BOUNDARY502_RUNNING`。正式四块长训仍被 step-502、5V 和相邻 Tile 接缝门禁阻塞。

## 8. 参数对位暂停门（2026-08-28）

问题现象：原 `TRAINING_READY` 只证明 AT、Face4、renderer mask、LiDAR depth、DA2、天空证据和空间 Tile 计划已经完成并通过身份验签；它不能证明 Trainer 已按竞品 High/type-2 消费这些输入。若直接把该状态解释为允许长训练，会遗漏逐 Tile K=7/K=30 初始化、Face4 crop/内参平移、DA2 与 mesh depth/normal、三阶段 loss、无放回视图采样、gradient split/clone/cull/reset、BilateralGrid、SIFT refine 和独立天空训练。

修改文件：

- `cloudstudio_3dgs/training/mipmap_type2_contract.py`
- `tools/build_mipmap_type2_parameter_spec.py`
- `tests/test_mipmap_type2_contract.py`

修改内容：新增签名 `mipmap_high_type2_parameter_spec_v1`。它把已恢复的 High/type-2 数值与实际 5 个 LiDAR Tile 绑定，但固定输出 `training_allowed=false`；13 项实现或消费证据全部补齐前，禁止晋级长训练。原 `TRAINING_READY` 从本节起仅解释为 `UPSTREAM_INPUTS_READY_ONLY`，不能单独作为 GPU 长训练授权。

验证方式：对参数规范执行签名验签、Tile plan/Tile inputs/upstream gate 三方 SHA256 绑定检查、逐 Tile `20×V` 与 `[5,10,5]` 边界检查，并用篡改/错绑负例证明 fail-closed。

当前状态：参数研究规范已准备，正式训练仍为 `BLOCKED`。逐 Tile 精确 K=7/K=30 初始化及 Trainer 消费、Tile Face4 crop/主点平移消费已经完成；下一项是 renderer semantic mask 与 mesh depth/normal 的实际 Trainer 消费，不能因为上游文件存在就视为已经接入损失。

### 8.1 逐 Tile 初始化几何准备

问题现象：原逐 Tile PLY 只含 xyz/RGB，无法证明训练会采用竞品的 K=7 邻距尺度、K=30 局部法向、`[d,d,0.5d]` 三轴和 `+Z→normal` 四元数；旧 K=3/RMS 与过薄短轴参数禁止复用。

修改文件：

- `cloudstudio_3dgs/training/mipmap_tile_geometry.py`
- `tools/build_mipmap_tile_geometry.py`
- `tests/test_mipmap_tile_geometry.py`

修改内容：按每块 halo-inclusive LiDAR PLY 的原始点序计算 K=7（含自身，平均第 1–6 邻点欧氏距离）尺度、K=30 PCA normal/eigenvalues、`[d,d,0.5d]` 线性尺度和 wxyz 最短弧四元数；逐块保存 NPZ，并绑定 Tile input 与初始化 PLY SHA256。PCA normal 无方向，兼容实现固定到 +Z 半球；由于厂商退化 normal fallback 的逐点随机状态无法位级恢复，证据边界明确标为算法兼容而非厂商 bit-exact。

验证方式：平面合成点检查三轴比例、normal 和批处理稳定性；Manifest 执行签名篡改负例与实际 PLY SHA256/点数绑定检查。

当前状态：真实 5 Tile CPU 计算与全量 artifact SHA256 验签已经完成，输出为 `outputs/snow-20260224-full-20260825/tile_initialization_geometry_k7_k30_v23t/tile_geometry_manifest.json`，Manifest SHA256 为 `eeaef9a33410547d2e9ee7052afc8ddde04a6a7fc839efadb252dd6c87fea334`。5 块共 7,271,982 行、约 378 MB，没有零邻距替换或退化 normal fallback。各 Tile 切向尺度 P50 为 6.453–9.409 mm、P95 为 14.749–38.123 mm，短轴严格为切向尺度的一半。Trainer 实际加载器又按 Tile ID、初始化 PLY SHA256、行数、NPZ SHA256、三轴比例和四元数归一性复验；5 Tile 全量消费审计为 `outputs/snow-20260224-full-20260825/mipmap_tile_geometry_consumption_audit_v23u.json`，签名 `17e68a2c592fcd505a27f205cd64b6f7b57ccd0935a53b1f1f07a92a67c60365`。该结果仍不授权 GPU 训练。

### 8.2 Tile Face4 crop、主点与 DA2 消费

问题现象：Tile 计划中的公开 sample ID 使用 `base::face`，而 DA2 缓存记录使用文件名式 `base__face`。未验收的 crop 代码曾错误使用 DA2 命名匹配 Tile；真实雪堆会将全部 Tile 视图判为未知，若绕过该检查则会退化成不分块的整 Face4 训练。

修改文件：

- `cloudstudio_3dgs/training/face_dataset.py`
- `cloudstudio_3dgs/training/trainer.py`
- `cloudstudio_3dgs/training/tile_face4_consumption.py`
- `tools/audit_mipmap_tile_face4_consumption.py`
- `tests/test_face_dataset.py`
- `tests/test_training.py`

修改内容：Tile 选择严格使用 `base::face`，DA2 查表仍使用其已签名的 `base__face`；对 RGB、组合 mask、LiDAR depth/confidence/mask、DA2 metric depth/mask、z-depth→range scale 和 sensor pixel coordinates 使用同一矩形裁剪，并对 pinhole `K` 执行 `cx-=x, cy-=y`。Trainer 从签名 Tile input 中选唯一 Tile 并把其 views 传入 Face4 dataset；配置同时绑定 Tile input、K7/K30 geometry、Face4 和 DA2 Manifest。

验证方式：合成 Face4 回归覆盖公开 ID、错误 DA2 式 ID 拒绝、像素精确裁剪、深度同步裁剪与主点平移。真实雪堆 5 Tile 结构检查覆盖全部 2,432 个含重叠视图实例、1,127 个唯一 Face4、约 51.93 亿 crop 像素负载，并对每 Tile 各加载一个 SHA256 已验证的真实 RGB/mask/DA2 样本；各 Tile DA2 有效视图为 464/476、368/374、466/470、591/607、499/505。

当前状态：新的联合消费审计 `outputs/snow-20260224-full-20260825/mipmap_tile_face4_renderer_da2_consumption_audit_v23w.json` 状态为 `CONSUMPTION_READY`，签名 `e7bccc65112c700df109d8733fa6b00d6f08f955791fc0294f74bc32da95c9f0`。它证明兼容 renderer 布尔 mask Manifest、Tile crop 和 DA2 被同一 Dataset 消费；但该 mask 仍不是已恢复 label 33 语义的 SegFormer 等价物。正式训练继续保持 `training_allowed=false`。

### 8.3 实现门强化与阶段调度固化

问题现象：第一版训练实现合同只覆盖初始化、crop、DA2、采样、经典生命周期、loss 数值和天空目标，遗漏 mesh consumer、SegFormer 等价语义、BilateralGrid、SIFT、容量公式及 Cut/Merge/最终交付时，仍可能被伪造为“完整实现”。此外，参数表虽记录 `[5,10,5]`，但没有逐 Tile 的确定性 step 边界工件。

修改文件：

- `cloudstudio_3dgs/pipeline/mipmap_gate.py`
- `cloudstudio_3dgs/training/face_dataset.py`
- `cloudstudio_3dgs/training/trainer.py`
- `cloudstudio_3dgs/training/tile_face4_consumption.py`
- `cloudstudio_3dgs/training/mipmap_loss_schedule.py`
- `tools/audit_mipmap_tile_face4_consumption.py`
- `tools/build_mipmap_loss_schedule_audit.py`
- `tests/test_face_dataset.py`
- `tests/test_mipmap_gate.py`
- `tests/test_mipmap_loss_schedule.py`

修改内容：实现合同新增 renderer 语义、LiDAR mesh depth/normal、BilateralGrid/SIFT、容量公式、halo ownership/Cut/Merge、原鱼眼评估和 PLY/SOG/LOD 强制字段；缺任何字段即 fail-closed。Face4 Dataset 现在读取 renderer-mask Manifest、验证其签名和 Face4 绑定，并以 Manifest 指定的 mask 路径/SHA 作为实际监督 mask。阶段调度器固定 `0–5V / 5–10V / 10–15V / 15–20V` 四段，以及 DA2、mesh depth、mesh normal、自洽 normal、opacity mean 和 sky-opacity 权重。

验证方式：CPU 定向回归 `126 passed`；真实 5 Tile renderer/crop/DA2 联合消费审计签名为 `e7bccc65112c700df109d8733fa6b00d6f08f955791fc0294f74bc32da95c9f0`。逐 Tile loss schedule 审计为 `outputs/snow-20260224-full-20260825/mipmap_high_type2_loss_schedule_audit_v23x.json`，签名 `8a927d8070d8b51ad0c8f83c5b834c3a04db87ceec3e21ee66299a186c25aeb9`，5 Tile 总步数为 `9520 / 7480 / 9400 / 12140 / 10100`。

当前状态：调度器只是 CPU schedule oracle，不会把缺失的 mesh rasterizer、rendered-normal consistency 或独立天空优化器伪装成已实现；其报告固定 `training_allowed=false`。下一优先项是构建并接入 LiDAR-derived mesh depth/normal cache 与逐像素遮挡过滤。

## 9. LiDAR 主导路线决策与真实 Face4 几何接入（2026-08-28）

本节是当前有效决策，覆盖第 3、4、8.3 节中把 DA2、完整 mesh depth/normal、SegFormer 等价语义、BilateralGrid 和 SIFT refinement 一并视为正式训练强制阻塞项的旧表述。竞品参数仍作为调研证据保留，但不再定义我方 LiDAR 数据的必要算法集合。

### 9.1 第一性原理决策

- 几何真值权威为新 AT 下的真实 LiDAR，不是 DA2。
- DA2 是相对深度模型预测；当前 train/val 对齐成功率分别为 `91.75%/55.56%`，暂不消费，权重固定为 `0`。
- mesh 只是在 LiDAR 点之间插值，没有增加新测量；完整 mesh 和局部 mesh 均暂缓，权重固定为 `0`。
- 如果 LiDAR 主导正式基线仍存在无法接受的空洞或漂浮，再分别做低权重 DA2、置信度局部 mesh 的独立 A/B；不得直接并入基线后归因不清。
- 不能把“703 万 LiDAR 点用于初始化”误写成训练持续消费几何。正式链路必须同时具备真实 Face4 稀疏 range 损失、KNN-PCA 表面法线锚定和新生高斯表面准入。

### 9.2 问题现象

原 `face4_circle_person_v23g` 的 train/val 清单中 `with_depth_count=0`，全部 `depth_path=null`。Trainer 虽然实现了 Face4 depth crop 与 range loss，但真实雪堆训练输入没有给它任何 Face4 LiDAR 深度。另一方面，`tangent_proposal` 只接入旧 error-weighted MCMC；当前 snow 使用的 gradient split/clone/cull/reset 路径会按各向同性随机分裂新点，没有执行 LiDAR 父点门或切平面出生位置约束。

### 9.3 修改文件

- `cloudstudio_3dgs/data/face_lidar_geometry.py`
- `cloudstudio_3dgs/data/depth_cache.py`
- `cloudstudio_3dgs/geometry/lidar_projection.py`
- `cloudstudio_3dgs/training/face_dataset.py`
- `cloudstudio_3dgs/training/trainer.py`
- `cloudstudio_3dgs/training/tangent_proposal.py`
- `cloudstudio_3dgs/training/default_strategy_adapter.py`
- `cloudstudio_3dgs/training/mipmap_type2_contract.py`
- `cloudstudio_3dgs/training/mipmap_loss_schedule.py`
- `cloudstudio_3dgs/pipeline/mipmap_gate.py`
- `tools/extract_face_lidar_geometry_manifest.py`
- `tools/build_face4_lidar_geometry.py`
- `tools/audit_face4_lidar_geometry_pair.py`
- `tools/audit_lidar_first_face4_readiness.py`
- `tools/build_mipmap_type2_parameter_spec.py`
- `tools/build_mipmap_loss_schedule_audit.py`
- `tests/test_face_dataset.py`
- `tests/test_depth_cache.py`
- `tests/test_default_strategy_adapter.py`
- `tests/test_mipmap_gate.py`
- `tests/test_mipmap_loss_schedule.py`
- `tests/test_mipmap_type2_contract.py`

### 9.4 修改内容

1. 新增独立签名 `face4_sparse_lidar_geometry`，绑定原 Face4 RGB Manifest、新 AT 原鱼眼 LiDAR depth Manifest、split 和 Dataset SHA。新增深度不会重签 RGB/Mask 清单，因此已有 renderer mask、Tile crop 和相机内参绑定保持有效。
2. `FaceCacheDataset` 可以从独立几何根加载真实稀疏 range/confidence，与 RGB、mask、range-scale、sensor coordinates 使用同一 Tile crop，并继续执行 `cx-=x, cy-=y`。
3. Face4 训练启用正 `lidar_range_weight` 时，Trainer 强制要求签名 Face4 LiDAR geometry，禁止再次出现“配置写了深度损失但输入没有深度”的静默空跑。
4. 经典 split/clone 在候选阶段先执行 LiDAR planarity/support 硬门；未通过的 Gaussian 可以继续渲染，但不能作为几何增长父点。
5. 允许增长的父点通过带种子的局部切平面 proposal 生成 clone/split 子点；法向偏移按局部 spacing 限制，且可同时初始化表面方向与最短轴。父点门与出生门不一致时直接抛错，禁止静默产生离面子点。
6. 新 `lidar_first_face4` 参数合同将 DA2、mesh、BilateralGrid 和 SIFT 标成延后可选实验；当前 loss 为 RGB `0.6/0.4`、真实稀疏 LiDAR range `0.05`、LiDAR surface normal `0.01`、DA2/mesh 全部 `0`。
7. 原 v24a 使用 `LAS→鱼眼整数像素→Face4 整数像素`，会把鱼眼栅格量化误差带入 Face4。正式算法改为每张物理图使用已接受 AT 的 `c2w` 将完整 LAS 变换到相机坐标，再用各 Face 的 `R_face/K_face` 直接投影；只在目标 Face4 做一次像素量化和一次最近欧氏距离 Z-buffer，最后与已签名 Face4 圆形/FoV/人物组合 mask 相交。
8. 直接投影的审计缓存保留原 LAS `source_index` 和同一 Face 像素的 `support_count`；Trainer 缓存不重复保存训练代码不消费的逐像素索引，但在签名 Manifest 顶层绑定完整 LAS SHA256、点数、AT Dataset SHA、原 Face SHA 和原深度参数 SHA。两者的 `range_m` 均保持 float32；Trainer 缓存仅把 confidence 量化为 `uint8/255`，最大量化误差不超过 `1/255`。
9. train/val 必须分别生成到同一版本的独立根目录，所有非空记录只能使用 `depth/...` 相对路径，并同时记录 `shape`、`valid_pixels`、`valid_fraction` 和文件 SHA。任何旧 `faces/..._depth.npz` 路径、缺失文件、不同 source Face SHA 或不同存储 profile 都判定为过渡产物，禁止进入 Trainer。
10. 完整 LAS 直接投影固定单 worker，以免并发复制 703 万点相机坐标和投影临时数组导致 RAM 峰值终止。正式顺序为先生成 val 探针、验算覆盖率和单位磁盘占用，再按实测容量放大 train；输出目录没有最终签名 Manifest 时一律视为未完成缓存。

### 9.5 验证方式与结果

- CPU 定向回归：`123 passed`，另 `tests/test_mipmap_type2_contract.py` 为 `3 passed`。
- train Face4 LiDAR：`1224` Face，`1169` 个有真实命中，`55` 个无命中，总有效像素 `306,689,262`；Manifest SHA256 为 `cdb23f1e46cc604acf65ef3e5a53645d79f4d1b13fe2ee5070f4070995158f4f`。
- 原 v24a/val 虽声称 `144` Face、`126` 个有命中，但非空路径仍指向旧 `faces/..._depth.npz`，独立根目录缺少对应文件，且记录缺少 `shape/valid_fraction`；该 SHA 不再作为可训练证据。
- v24b/val 直接投影审计版：`144` Face，`126` 个有真实命中，`18` 个无命中，总有效像素 `53,822,705`；126 个文件 SHA 全部通过，原 LAS 索引范围为 `0–7,036,346`，支持数范围为 `1–367`，Manifest SHA256 为 `f5ce663599e3a0a61d2273dce493ebc715280706860d3313d8c556312f9789ef`。
- v24b/val 已由 `FaceCacheDataset` 在 DA2 未接入的条件下抽样加载首/中/末 Face，深度像素分别为 `453,551 / 312,199 / 113`，相机、Face、renderer mask 与几何签名一致。
- 磁盘预检：v24b 审计版 val 为 `0.599 GiB`；保持 float32 range、使用 `uint8/255` confidence 并省略 Trainer 未消费的逐像素 provenance 后，实测同批数据为 `0.280 GiB`，预计完整 train+val 为约 `2.66 GiB`。该预检避免在仅余约 `3.71 GiB` 时盲目写出预计约 `5.1 GiB` 的审计版训练缓存。
- 5 个 Tile 均实载一个签名深度样本并在 crop 后保留 `167,655–721,027` 个真实 LiDAR 像素；联合 readiness 审计 SHA256 为 `143276963cf9333aa290978c4df3e4fe3bc3b65d610dd011bd397b1d93c0256e`。
- 新参数规范 SHA256 为 `cb9f2cf3f3ca860e676d1a118dc3af628e0b244ca9dd68ea6358fe261d128521`；新 loss schedule 审计 SHA256 为 `d34d763e6d81656dff004ac012bdc851b332b47e68745d40fbef17c9dcf2a170`。

### 9.6 当前状态与下一步

状态为 `DIRECT_FACE4_REFERENCE_PARTIAL_PAUSED`，仍固定 `training_allowed=false`。旧 v24a train 与旧布局 val 不能混用；v24c 全场投影已按要求停止，当前只有 `552` 个部分 NPZ、约 `1.22 GiB`，没有最终 Manifest，因此只能作为参考实现样本，Trainer 必须拒绝。不得继续生成完整 v24c，也不得直接启动长训练。

### 9.7 全场投影暂停与 Tile 深度复用优化（2026-08-28）

问题现象：参考实现对 `306×4×7,036,347` 个点—Face 组合执行直接投影，候选约 `8,612,488,728`。其数学意义是验证 `LAS→Face4` 单次量化与 Z-buffer，但不适合作为每次训练的正式预处理。仅改成“每 Tile 全点×全部 Tile Face”仍有 `3,627,153,195` 个候选，只减少 `2.37×`。

否决的锚点替代路线：LiDAR 可见性观测表与 5 个 Tile core+halo/crop 相交后只有约 `1.47M` 条候选，规模很小；将锚点吸附回精确 Tile LAS 的 25 万点审计也达到每块 `94.704%–98.864%` 接受率，最近点距离 P95 为 `5.66–16.22 mm`。但是，逐点对全 LAS 参考深度的 10 万条审计中，`>10 cm` 遮挡错误为 `1.1748%`；改为使用全部可用候选并先做每视图 Z-buffer 后，`697,568` 条候选的错误仍为 `1.1933%`，没有达到 `<0.5%` 门槛。锚点距离、法向距离、像素偏移和源相机 range 差均不能稳定识别这些跨遮挡面异常。因此观测表只能负责 Tile/view/crop 选择，禁止作为像素深度真值；K=8 扩邻路线暂停，不能以增加同表面密度冒充可见性恢复。

正式优化流程从本节起固定为：

1. 完整 LAS 只在“原始鱼眼 + 新 AT 位姿”阶段执行一次最近 range Z-buffer，生成已签名、可复用的真实 LiDAR 深度权威缓存；后续 Tile 训练不得再次枚举完整 LAS×Face。
2. 已签名 Face4 LiDAR range 缓存继续承担像素级深度监督。每个 Tile 严格按自己的 Face4 crop 选择像素，再通过 `Face pixel + euclidean range + c2w` 反投影到世界坐标，只保留位于该 Tile `core+halo` 的点。
3. Tile 精确 LAS PLY 与 K7/K30 几何独立承担高斯初始化、尺度、朝向、LiDAR 父点门、切平面 proposal 和“新生高斯不离面”；不能把深度 sidecar 的稀疏像素误当完整表面初始化。
4. 每个 Tile 生成独立签名 `face4_sparse_lidar_geometry`，绑定原 Face4 SHA、原始 LiDAR depth SHA、上游 Face4 LiDAR geometry SHA、Tile inputs SHA、Tile ID 和三维边界。文件保留 float32 range，confidence 量化为 `uint8/255`，省略 Trainer 不消费的逐像素 provenance。
5. RGB view 不因 LiDAR 稀少而删除；无有效深度的 view 仅关闭 range loss，仍保留 RGB。DA2 和完整 mesh 保持关闭，不参与此基线。
6. 各 Tile 严格串行生成和训练；一个 Tile 结束后释放深度包、KD/surface field 和 CUDA 缓存。不得把 5 个 Tile 同时常驻内存或显存。
7. 内容审计必须逐文件复验 SHA、crop 边界和反投影三维边界，并通过真实 `FaceCacheDataset` crop 加载；短 GPU smoke 必须证明真实 range loss 有非零梯度，受约束 clone/split 全部经过表面 proposal。

实现与验证：新增 `cloudstudio_3dgs/data/tile_face_lidar_geometry.py`、`tools/build_tile_face4_lidar_geometry.py`、`tools/audit_tile_face4_lidar_geometry.py` 和定向单元测试。Tile_1 的 `374/374` 个 view 均有真实 LiDAR range；crop 内 `89,938,146` 个像素经三维边界过滤后保留 `44,487,571` 个（`49.4646%`），独立包约 `0.233 GiB`，Manifest SHA256 为 `b2f4b2fea5d52960ef817f309d0f7a4a63fab3c363a334685e2f2f547426020d`。全文件/全像素内容审计状态为 `CONSUMPTION_READY`，签名 `db2a7ee2d8fa735f10fde0278c278e8114b81e5f2a17b647cd06d6bea0de0e50`。

Tile_1 低显存 GPU 组件 smoke 已通过：真实 crop 中 `268,854` 个 LiDAR 像素产生非零梯度；5,000 点表面场触发 `4,817` 个受约束 clone，全部经过 proposal；峰值显存 `76 MiB`，签名 `e01bdac6ac720e0ac9e7cf81be392512b875b87bf297f592be9bf1232c9c0e61`。该结果只证明深度和贴面增长组件连通，不证明 rasterizer、step 500 生命周期或最终图像质量。

当前状态为 `TILE_1_RANGE_CONSUMPTION_READY`，仍固定 `training_allowed=false`。下一步应以同一签名算法串行生成 Tile_0/2/3/4，随后做短 Trainer raster smoke；只有 batch 中 `depth_mask` 非空、`lidar_range_loss` 非零、第一次 densification 实际触发 LiDAR 父点门与切平面 proposal、Gaussian 数量/显存/loss 均正常，才允许进入逐 Tile 长训练。

### 9.8 Tile_1 两步 Trainer raster smoke（2026-08-28）

问题现象：本机锁定 gsplat wheel 提供 `projection_ut_3dgs_fused` 和 `rasterize_to_pixels_from_world_3dgs`，但不提供经典 `projection_ewa_3dgs_fused_fwd` 与 `rasterize_to_pixels_3dgs_fwd`。Face4 pinhole 默认经典路径因此无法进入真实 raster；仅打开 UT 后仍会落回缺失的经典二维 raster。eval3d 能完成前向和反向，但它不提供 DefaultStrategy 所依赖的经典 `means2d.grad`，因此禁止把两步实现 smoke 误写成 densification 已验证。

修改文件：

- `cloudstudio_3dgs/training/backend.py`
- `cloudstudio_3dgs/training/trainer.py`
- `tests/test_training.py`
- `outputs/snow-20260224-full-20260825/snow_tile1_lidar_first_raster_smoke2_v24f.config.json`

修改内容：新增显式 `pinhole_with_ut` 兼容开关；仅在该开关开启时让 pinhole 使用 `UT + eval3d + RGB-Ed`。Backend 标记输出为 `euclidean_ray_range_m`，Trainer 不再重复乘 Face4 `depth_to_range_scale`。`implementation_smoke_only` 固定最多 2 步、禁止致密化和正式评估，并跳过依赖 `means2d.grad` 的策略梯度隔离及 post-step；默认 pinhole 正式训练路径不变。

验证方式：`tests.test_mcmc_runtime + tests.test_mipmap_gate` 共 `31 passed`；此前 Face4/门禁定向测试共 `34 passed`，eval3d range 语义回归 `1 passed`。最终按 B 方案锁定 `SH0` 的真实 Tile_1 smoke 使用 `971,903` 个 LiDAR 初始化高斯、签名 Face4 RGB/mask、Tile range sidecar 和 K7/K30 几何完成 2/2 CUDA steps，训练状态为 `IMPLEMENTATION_SMOKE_COMPLETE`，峰值显存 `957,911,552 bytes`，最终高斯数保持 `971,903`，LiDAR robust-log-Huber loss 为 `0.0003253764`。checkpoint SHA256 为 `e2b8ef731a20e59051f1e778bf199d24dba3641b0f8b3e685275d98052a8efe7`，Manifest 合同 SHA256 为 `6abe7e0590f4d132c5dc1ee3f492e9e6751d8944ebc5c55f65fe236e4c9b5d57`。

当前状态：真实 raster、LiDAR range loss、反向、优化器和 checkpoint 链路已经连通，但证据范围严格为 `IMPLEMENTATION_SMOKE_ONLY_NOT_TRAINING_READY`。两步使用随机视图，loss 从首步 `0.43987` 到末步 `0.55465`，不能用于判断收敛或画质。正式训练前仍必须单独解决 eval3d 下的增长统计来源，并在第一次真实 clone/split/cull/reset 边界验证 LiDAR 父点门、切平面出生、Gaussian 数量和显存；在此之前不得启动长训练。

### 9.9 固定拓扑优先的优化评估重排（2026-08-28）

问题现象：此前 Snow 配置仍以 `default_3dgs` 和推迟 refine window 间接表达“暂不增长”，无法从签名合同区分严格固定拓扑、只裁 opacity 和自适应增长；训练也没有明确记录 Phase A 几何冻结、Phase B 小 LR 几何优化、逐损失梯度冲突、参数实际更新和 point-to-plane drift。Tile 清单还把 `training_and_export_box` 同时用于训练和导出，存在 halo 直接拼接、重叠高斯重复交付的风险。仅报告保留像素的深度误差也可能通过丢弃难点像素得到虚假低尾差。

修改文件：

- `cloudstudio_3dgs/training/topology_policy.py`
- `cloudstudio_3dgs/training/optimization_audit.py`
- `cloudstudio_3dgs/training/tile_ownership.py`
- `cloudstudio_3dgs/evaluation/lidar_accuracy_coverage.py`
- `cloudstudio_3dgs/training/trainer.py`
- `tools/audit_tile_lidar_accuracy_coverage.py`
- `tools/build_tile_core_ownership_contract.py`
- `tools/build_fixed_topology_evaluation_plan.py`
- `tools/build_fixed_topology_evaluation_readiness.py`
- `configs/snow_tile1_fixed_topology_a0_smoke_v25a.json`
- `tests/test_topology_policy.py`
- `tests/test_optimization_audit.py`
- `tests/test_tile_ownership.py`
- `tests/test_lidar_accuracy_coverage.py`
- `tests/test_training.py`

修改内容：

1. Trainer 新增显式 `strict_fixed / opacity_prune_only / adaptive_growth` 拓扑合同。A0 每步强制 Gaussian 数量不变；A1 只允许在指定 completed-step 执行一次 opacity prune，任何出生、重定位或非授权数量变化立即失败；增长策略在 A0/A1 中不执行。
2. 固化 Phase A/B/C：A 只优化颜色、曝光和 opacity，means/scale/quat 的 LR 为 `0`，range/normal 继续计算但优化权重为 `0`；B 以配置比例开放几何 LR 并启用 robust range 与 surface normal；C 维持受约束几何优化并标记 gap diagnostic，不自动出生。
3. audit step 记录 means/scale/quat/opacity 的总梯度、实际更新量、RGB/range/normal 分项梯度范数与夹角、range 朝向性检查以及相对初始化表面的 point-to-plane P50/P95/P99/最大值。
4. Tile merge 固定为 `core = 唯一最终所有权，halo = 仅训练上下文`。共享边界采用最小 Tile ID 唯一归属；正式合并必须先逐 Tile core cut 再 concatenate，禁止直接拼接 halo。
5. Gate 0 报告同时保留 accuracy 与 coverage 分母，输出 per-view P5/P10/P50/P95/P99、低/中/高 confidence、深度不连续边缘、5/10 cm 尾差。range-only sidecar 无法可靠给出低入射角和语义墙角分层时明确写 `NOT_AVAILABLE`，不伪造通过。

验证方式与结果：

- 新增定向测试与既有门禁组合回归均通过；Python 语法、diff whitespace 和 UTF-8 检查通过。
- Tile_1 Gate 0：候选 `89,938,146`，core+halo 保留 `44,487,571`，coverage `49.4646%`；per-view coverage P5/P10/P50 为 `10.6945% / 17.0641% / 50.8994%`。低/中/高 confidence coverage 为 `62.9318% / 41.6849% / 34.7981%`；深度不连续边缘为 `33.7877%`，非边缘为 `52.1447%`。保留像素的 range P50/P95/P99/最大误差均为 `0`，`>5 cm` 和 `>10 cm` 均为 `0`。审计 SHA256 为 `0e7b7f5af1ade4444c154add9d41a0b6723e990f2860e88bacaeca8170af5ba4`。
- A0 SH0 方向 smoke：`971,903` 个 Gaussian 完成 Phase A 冻结和 Phase B `0.1×` 几何 LR 两步；Phase A means/scale/quat 更新严格为 `0`，Phase B point-to-plane P95 `1.741 μm`、最大 `1.858 μm`，数量保持不变。Phase B range baseline `0.0003260`，朝 LiDAR 拉回 1 cm 后 `0.0002365`，反向推远后 `0.0004514`，方向门通过。Manifest SHA256 为 `fe3bad68bd8a474e287aa285ae1073febe1ba338a81a9a5970dae6970de19eda`。
- 单视图梯度只作为调权起点：Phase B 配置加权 RGB/range 对 means 的 L2 分别为 `0.037338 / 0.00004347`，约差 `859×`；opacity 的 RGB-range cosine 为 `-0.6767`，scale 为 `-0.3966`，说明 `0.05/0.01` 尚不能冻结为产品权重，必须在 A0 多视图 audit 中看分布。
- full-resolution `factor=1`、SH0、strict-fixed 单步 smoke 通过，峰值显存 `1,137,484,800 bytes`，Manifest SHA256 为 `88fafa6023ce7a3687a5f3869e0c1bbd101d570500b527e41578c7201e3b9725`。
- 5 Tile core ownership 合同 SHA256 为 `895490a3ab5be96d0f317f852f2a2e05413c41838176a091502802c7054bf0ee`；固定拓扑评估计划 SHA256 为 `7d00e622bdf542f71d18bfd42eddb322d804bfb5860837a533b09ea25844309d`。

准备的两个清洁基线严格共用权威输入、374 个训练 Face、SH0、seed 42、factor 1、loss、Phase 和评价 cadence：A0 为完全固定拓扑；A1 仅在 Phase A 的 `1870` 步 warm-up 后执行一次 opacity `<0.01` prune。两者均为 `5V + 10V + 5V = 7480` 步，Phase 长度 `1870 / 3740 / 1870`。

当前状态为 `FIXED_TOPOLOGY_EVALUATION_PREPARED`，readiness SHA256 为 `c1815151d71b913cccd2d2394b6377b845399e1c3dd8d420ac8bbf91d1bea5d7`。本状态仍固定 `training_allowed=false`，需要明确晋级后才运行 7480-step A0/A1；adaptive growth 继续禁止。A0/A1 运行期间必须输出多视图梯度分布和 Phase C 结构缺口图，最终 5 Tile 合并前必须完成逐 Tile core cut 数量和唯一性审计。

### 9.10 固定拓扑评估晋级与实时监控（2026-08-28）

问题现象：用户明确要求继续 A0 后，Trainer 正确拒绝了 `UPSTREAM_DATA_READY`，因为旧的 `TRAINING_IMPLEMENTATION_READY` 合同强制要求经典 split/clone/cull，与本轮 `strict_fixed / opacity_prune_only` 实验相冲突。直接复用旧门会把新路线伪装成已启用增长，手工改 `training_allowed` 又会破坏签名和 fail-closed 语义。训练产物此前也只在 checkpoint 保存最后一个 batch 指标，无法提供多轮自动刷新曲线。

修改文件：

- `cloudstudio_3dgs/pipeline/mipmap_gate.py`
- `cloudstudio_3dgs/training/trainer.py`
- `tools/promote_fixed_topology_evaluation_gate.py`
- `tools/training_monitor_server.py`
- `tools/training_monitor.html`
- `tests/test_mipmap_gate.py`

修改内容：新增独立签名状态 `FIXED_TOPOLOGY_EVALUATION_READY`。Promotion 必须复验上游门、readiness、评估计划和 A0/A1 配置，绑定不依赖 gate 路径的 arm fingerprint，只允许 `strict_fixed` 与 `opacity_prune_only`，并显式保持 `adaptive_growth_allowed=false`。Trainer 按自己的 run、Tile、Phase、factor、SH、DA2、出生控制和拓扑配置重算 fingerprint；不匹配的配置无法借用该门。新增非权威 `monitor/progress.jsonl`，后续训练每 10 步记录 loss、RGB/SSIM、LiDAR range/normal、Gaussian 数量、阶段、显存与最新几何审计；监控服务只读 checkpoint/Manifest/JSONL，将多轮数据缓存并通过本机网页每 5 秒刷新，支持任意数值指标和多轮叠加。

验证方式：门禁、拓扑和 Trainer 回归共 `74 passed`，相关 Python 文件语法检查通过；晋级门 SHA256 为 `2779d90eeddb2f833ffbf5ca76e2164ae2cbfbd249126e9f05aa93dddb6e32c9`。A0 已使用该门进入真实全分辨率训练：Phase A step 1870 时 Gaussian 数量保持 `971,903`，means/scale/quat 更新与 point-to-plane drift 均为 `0`，LiDAR 方向性通过；Phase B 首个持久化检查点 step 2244 的 point-to-plane P95 为 `0.0000004109 m`，仍无拓扑变化。监控 API 已返回 A0/A1 两轮状态、GPU、指标和持久化曲线点；页面地址为 `http://127.0.0.1:8792/`。

当前状态：A0 正在运行，A1 未启动，自适应增长继续禁止。当前 A0 进程在本次遥测代码加载前启动，因此本轮曲线以 374 步 checkpoint 为粒度；A1 和后续轮次将使用每 10 步 JSONL。监控数据不是质量签名，最终判断仍以 checkpoint、run manifest、原鱼眼评估和 Phase C 缺口审计为准。

### 9.11 V26a LiDAR 约束经典生长边界与续跑（2026-08-28）

问题现象：A0 在 step 2618 的固定拓扑 PLY 中，雪面主体细致、加载快，但人物和移动物体干扰对应的离面大高斯在 SuperSplat 中非常明显。固定 97.19 万点无法通过拓扑生命周期删除这些错误结构，也不能只在真实覆盖缺口补点。第一版 V26a 使用 `pinhole_with_ut=true` 的 eval3d 兼容路径；该路径不提供 DefaultStrategy/AbsGS 所需的 `means2d.absgrad`，因此第 1 步按合同失败，未生成伪造的经典生长结果。切换真正的经典二维路径后，首次 step 500 生命周期已触发，但旧 MCMC 遥测用 `step > start` 判断事件，而竞品式精确生命周期用 `step >= start`，导致证据封装拒绝 step 500 的合法快照并停止。

修改文件：

- `cloudstudio_3dgs/pipeline/mipmap_gate.py`
- `cloudstudio_3dgs/training/backend.py`
- `cloudstudio_3dgs/training/checkpoint.py`
- `cloudstudio_3dgs/training/default_strategy_adapter.py`
- `cloudstudio_3dgs/training/runtime_evidence.py`
- `cloudstudio_3dgs/training/trainer.py`
- `tools/build_v26a_boundary_gate.py`
- `tools/promote_v26a_evaluation.py`
- `tools/run_with_prebuilt_gsplat.py`
- `tests/test_default_strategy_adapter.py`
- `tests/test_mipmap_gate.py`
- `tests/test_training.py`

修改内容：Face4 V26a 固定使用包含经典 EWA 投影和二维 raster 的 gsplat 3DGS 扩展，`pinhole_with_ut=false`，保留真实 `means2d.absgrad`；不允许以 eval3d 世界坐标梯度代理 AbsGS。新增 `ADAPTIVE_GROWTH_BOUNDARY_READY / ADAPTIVE_GROWTH_EVALUATION_READY` 两级签名门：先在 step 502 受控停点核验首次 clone/split/cull、LiDAR 出生守卫、scale、数量和显存，只有签名报告 PASS 才允许从原 checkpoint 连续恢复到 7480 步。跨门恢复只允许 `trainer_config_sha256` 从边界配置切换到评估配置；Dataset、Face4、LiDAR、坐标变换、初始化、尺度标定和 gsplat runtime 任一身份变化仍 fail-closed。step 2618 设置独立保留 checkpoint，导出后训练继续。

验证方式与边界结果：相关门禁、训练和遥测回归共 `127 passed`，Python 语法检查通过。真实全分辨率边界在 step 500 执行 clone `251,672`、split 父点 `6`（子点 `12`）、cull `238,447`，数量从 `971,903` 变为 `985,134`，净增 `13,231`。LiDAR 门检查 `257,723` 个候选，支持 `251,678` 个并拒绝 `6,045` 个；`251,684` 个新生点全部经过切平面 proposal，应用率 `100%`，回退父点比例 `0.6313%`。生命周期后 scale P95 为 `0.01520 m`、最大 `0.19761 m`；峰值训练显存 `2,054,993,408 bytes`，低于 7.5 GiB 门；MCMC 噪声步数为 `0`。边界报告 SHA256 为 `3deea1f1b3172a2814bcbec9307c87c520d3a5d5dc5503a97dfb022e2fc24332`，评估门 SHA256 为 `e68771d7c03ea4cc783ba736bd67d856cd36cb72b8af96d2c2f6d32e8b6a0712`。

当前状态：V26a 已从 step 502 原优化器、采样器和生命周期状态恢复并越过 step 2618，训练继续运行；A1 与纯 MCMC 均未启动。step 2618 独立 checkpoint 保留 `360,078` 个 Gaussian，完整 SH0 PLY 为 `24.5 MB`，SHA256 为 `6c9fc158f6261c992c27e032e20e4e1f4a7454dcd7afb9afcf871f9733fe0bed`。其最长轴 P50/P95/最大值为 `1.73/6.59/24.92 cm`，最短轴 P50/P95 为 `0.15/0.47 cm`；轴比 P50/P95 为 `11.91/37.46`，说明浮空大球预期会比 A0 少，但当前模型更稀疏且存在偏扁高斯，必须以同一 SuperSplat 视角实际比较，不能只凭 loss 或文件大小宣称更好。实时监控页面继续使用 `http://127.0.0.1:8792/`，已叠加 A0、V26a 边界和 V26a 续跑曲线。

### 9.12 五 Tile 全域合并、独立天空与原鱼眼外观精修（2026-08-28）

问题现象：全局 MCMC 重分配虽然改善部分低透明度点，但同一原鱼眼验证上比固定拓扑 A0 低 `0.71 dB`，不能替换用户已经确认细致的 LiDAR 表面。五 Tile checkpoint 直接串接还会保留 halo 双层；A0 合并后仍有 `12,002` 个主体 Gaussian 的最长轴超过 `8 cm`，其中 `23` 个超过 `20 cm`。原鱼眼验证缺少天空时白色背景又会显著拉低全帧指标。竞品 PLY 的刚性配准 ICP 中位残差为 `52.9 cm`，超过 `15 cm` 门，不能伪造逐帧竞品指标。

修改文件：

- `cloudstudio_3dgs/training/trainer.py`
- `tools/merge_v28_tile_checkpoints.py`
- `tools/compose_gaussian_checkpoint_layers.py`
- `tools/build_raw_fisheye_post_refine_gate.py`
- `tools/evaluate_checkpoint_validation.py`
- `tools/export_gaussian_ply.py`
- `tools/import_gaussian_ply.py`
- `tools/align_gaussian_ply.py`
- `tests/test_export_gaussian_ply.py`

修改内容：五 Tile 合并按唯一 core owner 丢弃 halo/场外重复点；合并 checkpoint 补齐可 warm-start 的 schema。新增签名的 raw-fisheye post-refine lineage：只允许已有 Face4/LiDAR 全流程产物 warm start、`strict_fixed`、means/scale/quat LR 全为 `0`，并允许重新初始化原鱼眼曝光辅助参数。独立照片天空以 100,000 个 SH0 Gaussian 烘焙，和主体分层组合。导出与验证支持只对前 N 个主体 Gaussian 应用尺度上限，不缩小天空壳。导入/配准工具会自动建立输出目录，避免不存在父目录时写盘失败。

验证方式与结果：2-step smoke 峰值显存 `2.89 GiB`，means/scale/quat 最大逐元素变化均为 `0`。250 步和 500 步原鱼眼精修持续提升；500 步 surface-only 为 PSNR `13.595 dB`、SSIM `0.5727`、LiDAR depth MAE `0.644 m`，相比未精修 A0 的 `13.324 / 0.5615 / 0.650 m` 全部改善。加入独立天空后为 `18.174 dB / 0.5539`；旧 v17d 为 `18.437 dB / 0.5704`，但其 depth MAE 为 `4.585 m`，因此 V30 保留了接近旧外观分数并显著改善真实几何。主体 `8 cm` 尺度 cap 前后图像指标不变。相关 Trainer、门禁、Surface、Tile 与导出回归分别通过 `76 passed` 和 `58 passed`，Python 语法及 UTF-8 乱码检查通过。

最终交付 PLY 为 `full_area_final_v30/exports/snow_full_area_v30_final_refine500_photo_sky_sh0_pruned005.ply`：`6,334,817` 个 SH0 Gaussian，`430.8 MB`，opacity `<0.005` 的死点已移除，SHA256 为 `eb455a21b86a0021b67b25642568a311cc4df80fcd5c30406cb8b72f96d412dc`。主体所有轴不超过 `8 cm`；全文件最大轴 `1.2851 m` 只来自独立 `210 m` 天空壳。1000 步候选在 G: 仅剩 `40 MB` 时因 checkpoint 写盘失败，已只删除其 `478,482,432` 字节失败临时文件并回退到已通过的 V30，未删除正式结果。

续训修正：原 1000 步候选会从 A0 重复执行已经完成的前 500 步，而且每 50 步原子写入约 1.18 GB checkpoint，在低磁盘空间下没有必要。`build_raw_fisheye_post_refine_gate.py` 新增显式 warm-start、继承曝光辅助参数、自定义 run ID 与 checkpoint 间隔参数；下一轮从 refine500 的 `latest.pt` 继承主体颜色、透明度和曝光，几何仍严格冻结，只再执行 500 步并把 checkpoint 间隔放宽到 250 步。验证要求包括配置/门禁签名通过、warm-start 文件存在、means/scale/quat LR 均为零、点数保持不变，以及中间/最终原鱼眼指标不低于 V30；未通过则不晋级。

续训结果：用户确认永久删除 23 个已核对的旧 smoke/V28/V29 checkpoint、evaluation 与 export 目录，实际删除 `32.85 GiB`，G 盘可用空间由 `0.29 GiB` 增至 `34.69 GiB`；V30、refine500、A0 合并 checkpoint、原始数据、源码和 JSON 证据均保留。第二段 500 步从 refine500 warm start 正常完成，`7,036,339` 个主体 Gaussian 全程保持不变，峰值显存 `3,117,250,048 bytes`。step 250 surface 为 `13.6977 dB / 0.57627 / 0.61672 m`，step 500 surface 为 `13.80315 dB / 0.57952 / 0.61391 m`。与同一 100k 照片天空组合后的 V31 整图均值达到 `18.77258 dB / 0.55984`，但用户视觉复核及我方原始鱼眼渲染均明确发现墙体透明、背景颜色穿透墙面以及远树缺失。复盘确认旧门禁仅使用整图平均 PSNR/SSIM，未检查前景 alpha、背景泄漏、墙体 ROI 和远树保留率；同时 V31 使用的是 SH0 照片球壳，而不是竞品式独立 SH1 天空。因此上述数值只作为失败实验记录，V31 交付晋级结论撤销。

当前状态：`V31_VISUAL_REVIEW_FAILED_RETRAIN_REQUIRED`。新路线回退到 A0 全域主体检查点：位置、尺度、旋转和 opacity 全部冻结，仅允许短程颜色/曝光适配；天空独立为 SH1，且天空掩膜必须排除远树与建筑。新门禁除 PSNR/SSIM 和 LiDAR range 外，必须同时检查墙体/地面前景 alpha、开关天空前后的背景泄漏、远树 ROI 保留率及最差视图，未通过前不得重新标记为交付。

### 9.13 竞品路线交叉核对与重训阻塞（2026-08-28）

问题现象：V26a 把“竞品经典生命周期阈值”和我方简化 loss/光度模型直接组合，step 2618 只剩 `360,078` 个高斯；V30/V31 又把 100k SH0 照片球壳合入 surface PLY。两者都混淆了竞品已证实路径、DLL 未启用能力和 CloudStudio 自有增强，导致同名参数不等于同样行为。

交叉核对：重新读取竞品实际任务、四个 level-0、surface/sky PLY，并与二进制静态审计逐项比对。确认 snow 使用四 Tile 串行、K7/K30 LiDAR 初始化、`20V` 无放回视图、gradient split/clone/cull/reset、MCMC 关闭、BilateralGrid 与条件 SIFT；surface 终态 DC-only，独立 sky 为 100k SH1。surface `gs.ply` SHA256 为 `C10026FEFF1DD645273E1620C7BA7E1A08C8727282734D847CC1CBA3D81DF8C4`，sky `sky.ply` 为 `C55F326E99AF6F58C3A8E00B63687CAFCEAFBFE20984C710B6EE05D8D2CDA3E5`。仍未知 label 33 类名、surface SH-rest 活跃方式、cap 倍率 C、每次 lifecycle 事件数和 `TrainBackground` 完整 loss。

修改内容：训练合同把 LiDAR 出生守卫和唯一 core ownership 明确标记为 CloudStudio 增强；sky 改为独立于 surface PLY 的产品，不再要求合并进主体流；表面 degree 1 的描述收紧为“参数结构已证实、实际 SH-rest 是否参与训练未知”。完整决策见 `results/diagnostics/snow-20260828-mipmap-route-crosscheck-and-retraining-decision.zh-CN.md`。

当前状态：`COMPETITOR_CROSSCHECK_COMPLETE_RETRAIN_BLOCKED`。新长训继续禁止。下一步先补真实 renderer semantic/dynamic mask 消费和 per-camera BilateralGrid 或等价受控光度模型，再生成单 Tile 502-step 签名边界；该边界必须同时通过数量、opacity、scale、LiDAR 距离、墙体 alpha、远树 ROI 和天空泄漏门，才允许继续下一完整 epoch。

### 9.14 V33a 竞品式生命周期与 LiDAR 出生守卫复核（2026-08-29）

问题现象：V33a 已使用竞品 snow 的主机制而非 MCMC，即 projected-gradient clone/split/cull/reset，并额外加入 LiDAR 父点支持与切平面出生守卫、逐相机 PPISP、SH0 surface、DA2/POS 关闭。step 502 首次生命周期健康，但继续执行同一组 cull/reset 阈值后点数快速崩塌，说明“机制方向正确”不等于“现有阈值已经与我方 loss、mask、可见性和 LiDAR 初始化完成定标”。

修改文件：

- `cloudstudio_3dgs/pipeline/mipmap_gate.py`
- `cloudstudio_3dgs/training/ppisp.py`
- `cloudstudio_3dgs/training/trainer.py`
- `tools/build_v26a_boundary_gate.py`
- `tools/promote_v26a_evaluation.py`
- `tests/test_mipmap_gate.py`
- `tests/test_ppisp.py`

修改内容：PPISP 增加确定性的 warmup + exponential decay 学习率并由 Trainer 每步消费；Trainer 记录 PPISP 实际学习率和 RGB/depth mask 有效像素比例。门禁增加 step `503–2618` 的受控 review 状态，并将首次边界最低保留率提高到 `90%`，同时要求父点和子点 LiDAR 支持均值不低于 `90%`。跨门 resume 仍绑定 Dataset、Face4、LiDAR、初始化和 runtime 身份。

验证方式与结果：定向回归 `101 passed`，Python 语法和 diff 检查通过。两步真实 raster smoke 完成，`971,903` 个高斯不变，RGB mask 有效比例为 `97.71% / 95.54%`，depth mask 实际非空，峰值显存约 `1.14 GiB`。真实 step 502 生命周期从 `971,903` 变为 `933,870`：clone `244,484`、split 父点 `5`、cull `282,522`；父/子 LiDAR 支持均值为 `95.726% / 95.280%`，点到 LiDAR 最近距离 P95 `4.837 mm`，最大轴 `0.19993 m`，峰值显存约 `2.47 GiB`。该边界报告为 PASS。

连续 review 到 step 1002 后只剩 `468,203` 个高斯，为初始点数的 `48.17%`。六次生命周期累计 clone `857,210`、split 子点净增 `484`、cull `1,361,394`。step 600 opacity reset 把上限压到 `0.2`，step 700 随后一次删除 `361,844`；其中低 opacity 候选 `359,585`，证明 collapse 主要来自 opacity cull/reset 相互作用，不是 LiDAR 出生守卫拒绝过强。step 502 原鱼眼指标为 PSNR `6.9918 dB`、SSIM `0.45841`、alpha `0.17971`，与 A0 step 2618 接近，仅证明首次边界未立即恶化，不代表最终晋级。

当前状态：`V33A_BOUNDARY_PASS_CONTINUED_CULL_FAILED`。MCMC 降级为未来固定预算重分配实验，不再作为当前 backbone；A0 只作覆盖基线；主线固定为 `LiDAR planar surfel + projected-gradient adaptive topology + LiDAR-safe birth + observation/coverage-aware cull`。下一步先增加按 opacity/world-scale/screen-radius 分项的 cull 遥测，再做 Tile_1 的受控短 A/B；在 step 1002 总保留率、单次净删除、前景 alpha、墙体/远树 ROI、LiDAR 距离和尺度全部通过前，禁止五 Tile 或全域长训。

### 9.15 V34a 观察量感知 Cull 修复（2026-08-29）

问题现象：V33a 的 projected-gradient 出生和 LiDAR 切平面守卫已经通过，但 opacity reset 后的下一次生命周期会直接按单帧累计状态删除所有低于阈值的点。step 700 一次删除 `361,844`，造成墙面、地面和远景覆盖崩塌。共享 MCMC 遥测还会把经典路线 refine 前的低 opacity 数量误记为 relocation，虽然嵌套 `classic_lifecycle` 正确，但监控汇总具有误导性。

修改文件：

- `cloudstudio_3dgs/training/default_strategy_adapter.py`
- `cloudstudio_3dgs/training/backend.py`
- `cloudstudio_3dgs/training/runtime_evidence.py`
- `cloudstudio_3dgs/training/trainer.py`
- `cloudstudio_3dgs/pipeline/mipmap_gate.py`
- `tools/build_v26a_boundary_gate.py`
- `tests/test_default_strategy_adapter.py`
- `tests/test_mcmc_runtime.py`
- `tests/test_mipmap_gate.py`

修改内容：新增签名的 `observation_aware` opacity cull policy。低 opacity 点必须累计至少 4 次有效观测、连续两个生命周期保持低值，并离上次 reset 超过 200 步；每次 opacity 删除最多为当前人口的 5%。world-scale 与 screen-radius 删除继续即时生效。策略状态中的观察数和连续低值是逐 Gaussian tensor，会随 gsplat duplicate/split/remove 同步变换并写入 checkpoint。生命周期证据拆分 opacity/world/screen 原因。经典路线的通用 telemetry 现固定 relocation 为 0，并以真实 clone/split/cull 重算 added/pruned；resume 会修复内存中的旧统计，不修改旧 checkpoint 文件。

验证方式与结果：定向生命周期、门禁、Trainer、PPISP 和 MCMC 回归共 `163 passed`，Python 语法与 diff 检查通过。真实 step 502 从 `971,903` 变为 `1,214,533`，clone `244,483`、split 父点 `5`、cull `1,858`；其中 opacity cull 为 0，全部删除来自 world/screen 异常。继续到 step 1002 后为 `1,573,987`，失败 V33a 同阶段仅 `468,203`。六轮累计生成 clone/split children `761,663`、cull `159,060`，另有 `519` 个 split parent 被替换，净增 `602,084`；没有单轮大规模 collapse。峰值显存低于 `2.7 GiB`。

几何与图像门：可见点到 LiDAR 距离 P95 `9.023 mm`、超过 `0.3 m` 为 0；最长轴 P95 `26.40 mm`，无 `>0.5 m` 巨型高斯；轴比 P50 `5.68`。原鱼眼 PSNR `6.9944 dB`、SSIM `0.45452`、alpha `0.18801`、LiDAR alpha `0.27570`。覆盖相对 V33a step 502 改善，但 SSIM 略降；且 `64.25%` 的高斯 opacity 低于 `0.1`，说明人口平衡仍未完成。

当前状态：`V34A_CULL_COLLAPSE_FIXED_QUALITY_GATE_PENDING`。完整 step 1002 PLY 已导出用于与 A0/V33a 同视角视觉比较；在墙体、远树、雪面和动态区域 ROI 通过前，不继续 2618 或五 Tile。下一轮只允许受控 population-equilibrium A/B，不得重新启用 MCMC，也不得把低 opacity 直接等价为几何无效。

### 9.16 V35 屏幕足迹条件 Split 门禁（2026-08-29）

问题现象：V34a 已阻止人口崩塌并保持 LiDAR 几何，但 step 1002 PLY 仍有明显模糊大高斯。可见高斯投影足迹 `>5 px` 的比例为 `39.4%`；最长轴 `>2 cm` 的可见点为 `79,483`，其中只有 `9,524` 个同时满足轴比 `<3`。这说明问题不只是“圆球过多”，还包括高轴比薄盘在当前视角占据过大屏幕足迹。全局按 2 cm Split 会破坏竞品依赖大薄盘覆盖平滑面的容量分配。

修改文件：

- `cloudstudio_3dgs/training/default_strategy_adapter.py`
- `cloudstudio_3dgs/training/trainer.py`
- `cloudstudio_3dgs/pipeline/mipmap_gate.py`
- `tools/build_v26a_boundary_gate.py`
- `tests/test_default_strategy_adapter.py`
- `tests/test_mipmap_gate.py`

修改内容：保留竞品确认的 `split_scale_m=0.2`，新增签名策略 `lidar_surface_screen_detail`。只有高 projected-gradient、LiDAR 支持、最长轴 `>0.02 m` 且过去一个 lifecycle 窗口真实最大 raster radius `>0.0035` 的父点才从 clone 改为 split。低 opacity 删除受 5% 上限约束时，`lowest_opacity_per_footprint` 使用 gsplat 实际累计 raster radius 排序；仅在缺失 radii 时回退到世界尺度。构建工具仍使用显式 `--detail-split-2cm` 开关，但门禁同时绑定物理尺度和屏幕阈值，不能退化成全局 2 cm Split。

验证方式：相关 lifecycle、门禁、MCMC 遥测与 PPISP 回归 `113 passed`；修改模块 Python 语法检查通过。新增单元测试证明只有同时超过物理与屏幕阈值的高梯度点 Split，屏幕足迹不足的大薄盘和物理尺寸较小的清晰点继续 clone。尚未启动 V35 训练，故当前没有画质 PASS 结论。

当前状态：`V35_SCREEN_DETAIL_ARM_IMPLEMENTED_NOT_RUN`。下一步只允许独立 Tile_1 step 1002 短臂；A0/V34a 复用既有产物。先比较固定 ROI 清晰度、alpha、SSIM、投影足迹分布、纹理—密度相关、LiDAR P95/P99、点数和显存，再决定是否继续 2618。opacity-mean `0.01`、无 LiDAR 视觉补洞、独立天空和五 Tile 长训均不得与本短臂同时开启。

### 9.17 V35–V38 症状补丁路线终止与 lifecycle 时序复盘（2026-08-29）

问题现象：V36b 已能让低纹理区高斯数量下降，但大量保留点 opacity 很低，真实 alpha 不足，墙面和地面出现背景穿透；V37a 只把高斯压得更薄，没有恢复表面不透明度。准备中的 V38a 使用 LiDAR alpha `0.95` 和 2 cm 体素累计 alpha `0.5` 试图补洞，但世界体素 opacity 乘积不包含相机视角、遮挡顺序、投影重叠和椭球方向，不能冒充真实渲染覆盖，也不是竞品生命周期。

修改文件：`cloudstudio_3dgs/training/default_strategy_adapter.py`、`cloudstudio_3dgs/training/lidar_normals.py`、`cloudstudio_3dgs/training/trainer.py`、`cloudstudio_3dgs/pipeline/mipmap_gate.py`、`tools/build_v26a_boundary_gate.py`、`tools/promote_v26a_evaluation.py` 及对应测试。

修改内容与复盘：相关 alpha、局部保护、比例压平和 screen-detail 能力均保持显式可关闭，不作为下一主臂默认值。源码确认当前训练循环是 `backward -> Adam -> topology`，而竞品恢复顺序为 `backward -> Split/Clone/Cull -> Adam`；当前还使用了 `revised_opacity=true`，与竞品 Split 子点重复父 opacity 的证据不符。PPISP 只能视为受控光度模型，不能标成 `[N_camera,12,8,16,16]` BilateralGrid 等价实现。

验证方式：alpha/局部保护/比例压平相关定向回归 `143 passed, 1 deselected`；排除项是当前经典 3DGS 动态库不支持的 3DGUT 尺度测试。V38a 在约 step 240 强制停止，未到首次 topology 边界，不产生晋级结论。

当前状态：`V38A_ABORTED_VENDOR_SEMANTICS_REWORK_REQUIRED`。下一步必须先增加签名的 pre-optimizer lifecycle，并证明 Split/Clone/Cull 后当前步梯度和 Adam state 按行正确继承；竞品语义臂关闭 LiDAR 硬出生守卫、局部保护、删除上限、detail split、薄盘和 alpha 自定义损失，使用普通梯度、厂商阈值、原始 opacity 复制和 immediate Cull。CPU 合成测试、2-step GPU smoke、step 502、reset 后短 review 依次通过前，禁止长训和全域扩展。

### 9.18 V40–V46 透明度生命周期定标与 602 步早停（2026-08-29）

问题现象：V40a step 502 PLY 的主体结构可用，但雪堆略透明且细节提升有限。继续使用每 300 步 opacity reset 和每 100 步 immediate Cull 会在 step 602 后快速降低覆盖；只降低 Cull 阈值或延长到 step 1002 仍会让 PSNR、alpha 和最差视图同时下降。

修改文件：`cloudstudio_3dgs/training/default_strategy_adapter.py`、`cloudstudio_3dgs/training/trainer.py`、`cloudstudio_3dgs/pipeline/mipmap_gate.py`、`tools/build_v26a_boundary_gate.py`、`tools/promote_v26a_evaluation.py`、`tools/evaluate_checkpoint_validation.py`、`tests/test_default_strategy_adapter.py`、`tests/test_mipmap_gate.py`。

修改内容：增加签名的 deferred reset、普通梯度 `7.5e-5`、uniform Cull `0.04`、当前可见高斯 opacity sparsity `0.001`、仅尺度异常 Cull，以及 screen-aware detail Split 对照配置。生命周期事件新增梯度分位数、梯度/opacity 阈值扫描和 Cull 原因遥测；验证器增加 metrics-only 模式，374 视图评估不再写出数 GiB 中间图。pre-optimizer Split 现消费配置中的 `revised_opacity`，旧厂商语义配置保持 `false`，只有显式 screen-detail 研究臂可设为 `true`。

真实结果：V42a step 600 的梯度 P50/P95/P99 为 `2.11e-5 / 7.71e-5 / 1.50e-4`，证明原 `1.5e-4` 只选择约最高 1% 父点。改为 `7.5e-5` 后，V45b step 600 选择 `39,257` 个父点；其中 clone `39,170`、split 父点 `87`，仅删除 world/screen 异常 `767` 个，最终为 `863,951` 个高斯。374 个 Tile_1 Face4 视图上，V45b step 602 达到 PSNR `10.45167 dB`、alpha 均值 `0.60189`、alpha P05 均值 `0.08731`、alpha 低于 `0.95` 的像素比例 `0.60039`，均优于此前 V42b step 602 的 `10.42081 / 0.59933 / 0.08675 / 0.61299`。

失败边界：V44a 把 opacity sparsity 从 `0.01` 降到 `0.001`，step 1002 仅恢复到 `9.54298 dB / alpha 0.58510`；V45a 完全关闭 opacity Cull 后 step 1002 也只有 `9.64908 / 0.59199`，证明 600 步后的退化不只是删点，而是当前 PPISP/opacity/拓扑联合目标继续训练会降低覆盖。V46a 将 `5,267` 个父点改为 screen-aware Split 后，PSNR/alpha 降到 `10.27435 / 0.58494`；V46b 启用 revised opacity 后进一步降到 `10.17067 / 0.57562`，因此 screen-detail 分支不晋级。

验证方式：定向生命周期与门禁回归 `71 passed`，修改模块 Python 语法检查通过。V45b PLY 包含 `863,951` 个 SH0 Gaussian、大小 `58.7 MB`；最长轴 P50/P95 为 `7.4/18.0 mm`，可见高斯中 `25.3%` 的屏幕宽度仍大于 5 px，说明当前候选主要修复覆盖和首轮容量分配，尚未达到竞品终态薄盘形态。

当前状态：`V45B_STEP602_CURRENT_TILE1_CHAMPION_LONG_RUN_BLOCKED`。当前最佳 PLY 为 `v45_geometry_cull_only/training_tile1_v45b_growth7p5e5_geometrycull_opacity1e3_review602/exports/snow_tile1_v45b_step602_sh0_full.ply`。不得从该节点直接长跑或扩展五 Tile；下一轮必须从 V45b 的固定候选出发，单独研究不降低真实 alpha 的形状优化或光度模型，不能同时重新启用 opacity Cull、screen-detail Split、MCMC、DA2 或天空。

### 9.19 V47–V48 覆盖地板与形状单变量探针（2026-08-29）

问题现象：V45b 的 SH0 结构和短程指标可用，但白色训练背景允许白雪通过降低 opacity 或缩小投影足迹来接近目标颜色，导致 SuperSplat 更换背景时出现透明雪堆。此前验证器又没有消费 Face4 LiDAR sidecar，`lidar_alpha_*` 指标为空，无法用真实 LiDAR 像素约束表面覆盖。V37 虽证明比例压平能改变高斯形状，却同时放开颜色、opacity、位置和生命周期，无法判断锐化是否会再次造成透明。

修改文件：

- `cloudstudio_3dgs/pipeline/mipmap_gate.py`
- `cloudstudio_3dgs/training/checkpoint.py`
- `cloudstudio_3dgs/training/ppisp.py`
- `cloudstudio_3dgs/training/trainer.py`
- `tools/build_v47_surface_alpha_probe.py`
- `tools/build_v48_surface_shape_probe.py`
- `tools/evaluate_checkpoint_validation.py`
- `tools/promote_v26a_evaluation.py`
- `tests/test_mipmap_gate.py`
- `tests/test_training.py`

修改内容：验证器现在从签名配置加载 Face4 LiDAR range sidecar，并支持确定性分层抽样和背景色挑战；日常短臂固定评估 48 个视图，8 个视图用于黑背景透明度挑战，只有晋级候选才运行全部 374 个 Tile_1 视图。V47 增加签名的 opacity-only 探针，冻结 means/scales/quats/SH0/PPISP 和 lifecycle，只用真实 LiDAR alpha 调整 opacity。V48 再从 V47e step 652 独立恢复，冻结点数、means、SH0、opacity 和 PPISP，仅允许 scales/quats 更新，同时保留 `LiDAR alpha=0.95` 的覆盖地板；V48b 使用 scales/quats LR `0.003/0.0005`、法线对齐与比例压平权重 `0.1/0.1`、目标短轴/切向几何均值比例 `0.15`，严格停止于 step 702。

验证方式与结果：定向 Trainer/门禁回归 `71 passed, 1 deselected`；排除项仍是当前环境未构建 3DGUT 的已知 raster smoke。V47e 在固定 48 视图上相对 V45b 将 PSNR `10.2554→10.3004 dB`、LiDAR alpha 均值 `0.9448→0.9537`，证明覆盖损失可以关闭白背景捷径，但它不优化细节。V48a 强度过低，中位轴比仅 `2.162→2.177`。V48b 保持 `863,951` 点，means、SH0、opacity 与 PPISP 最大差均为 0；中位最短轴 `3.568→2.963 mm`，轴比 `2.162→2.599`。同一 48 视图上 PSNR 为 `10.5218 dB`、alpha 均值 `0.63623`、alpha P05 `0.07659`、LiDAR alpha 均值/P05 为 `0.95679/0.84879`。8 视图黑背景挑战相对 V47e 的 PSNR `10.4260→10.5031 dB`、alpha `0.57452→0.58502`、LiDAR alpha `0.95542→0.95905`，没有出现靠白背景掩盖的覆盖退化。深度 MAE 从 V47e 的 `0.5519 m` 轻微变为 `0.5639 m`；由于 means 完全冻结，这反映 splat 足迹和可见性变化，不是中心漂移，仍需视觉 ROI 复核。

当前状态：`V48B_SHAPE_PROBE_NUMERIC_PASS_VISUAL_REVIEW_PENDING`。候选 PLY 为 `v48_surface_shape/training_tile1_v48b_scalequat_strong_ratio0p15_review702/exports/snow_tile1_v48b_step702_shapeonly_sh0_full.ply`，完整保留全部 `863,951` 个高斯。它只证明“覆盖受控条件下可以安全变薄并提高短程指标”，还没有达到竞品轴比约 12 的终态，也没有恢复纹理驱动的容量重分配。SuperSplat 墙面、雪堆和远景 ROI 通过前，不启动长训、拓扑增长或全 Tile。

### 9.20 V49–V50 五 Tile 覆盖修复、形状精修与全域原鱼眼协调（2026-08-29）

问题现象：V30 的全域外观较完整，但最终导出按 opacity `<0.005` 删除 `801,522` 个主体点，造成墙体、雪面和远景被独立天空或背景穿透。单 Tile V48b 已证明“先修覆盖、再改形状”不会移动中心，但尚未覆盖全场景；短阶段若继续随机有放回采样，又会反复看到少量照片而遗漏其他方向。

修改文件：

- `cloudstudio_3dgs/pipeline/mipmap_gate.py`
- `cloudstudio_3dgs/training/exposure.py`
- `cloudstudio_3dgs/training/trainer.py`
- `tools/build_v49_full_area_surface_quality_phase.py`
- `tools/build_v50_full_area_raw_fisheye_phase.py`
- `tools/evaluate_checkpoint_validation.py`
- `tests/test_mipmap_gate.py`
- `tests/test_training.py`

修改内容：五个既有 A0 Tile 均执行 50 步 opacity-only LiDAR alpha 覆盖修复，再执行 50 步 scales/quats-only 形状精修；means、SH0、点数和拓扑始终冻结。每一段配置同时绑定配置 SHA256 与 warm-start checkpoint SHA256，Trainer 启动时 fail-closed 验签。五 Tile 以唯一 core owner 合并，halo 重复点删除但不进行 opacity 裁剪。全域随后在原始鱼眼上执行 50 步短覆盖验证、250 步 Fisher–Yates 无放回 opacity-only 覆盖和 250 步 SH0 颜色/曝光协调；固定拓扑短臂允许无放回采样不足 20 epoch，自适应竞品臂仍强制 20 个完整 epoch。验证器继承 Trainer 的 3DGUT pinhole 路径，避免评估与训练 raster 模式不一致。

验证方式与结果：六个旧且已被替代、无 PLY/SOG 的训练目录经用户明确确认后永久删除，共释放约 `32.041 GiB`；G 盘可用空间一度由 `8.816 GiB` 增至 `41.828 GiB`。五 Tile shape checkpoint 分别保留 `1,895,788 / 971,903 / 1,477,056 / 1,850,945 / 1,076,290` 个高斯，峰值显存均低于 `2.28 GiB`。core-owner 合并从 `7,271,982` 个带 halo 点得到 `7,036,339` 个唯一主体点。V49 全域 36 张原鱼眼验证为 PSNR `13.7846 dB`、SSIM `0.56825`、LiDAR alpha `0.80846`、深度 MAE `0.60871 m`；V50b 覆盖后提升到 `14.0019 / 0.5755 / 0.849 / 0.59961 m`，最弱视图 LiDAR alpha 从约 `0.66` 提升到 `0.736`。V50c 颜色协调后为 `14.097 dB / 0.5799`，逐张量 SHA256 证明 means/scales/quats/opacities 与 V50b 完全一致，仅 SH0 改变。8 张黑背景挑战的 LiDAR alpha 从 V49 的 `0.88194` 提升到 V50c 的 `0.9069`，最弱视图从 `0.67095` 提升到约 `0.748`。

当前状态：`V50C_FULL_AREA_SURFACE_NUMERIC_PASS_VISUAL_REVIEW_PENDING`。无裁剪主体 PLY 为 `full_area_surface_quality_v50/exports/snow_full_area_v50c_surface_no_prune_sh0.ply`，完整包含 `7,036,339` 个 SH0 Gaussian，大小 `478.5 MB`，SHA256 为 `4bbf1ef211c68eb4dac4fc6f97f59a5ca9be44198fe9fef1fd3e43e36ff857f0`。可见高斯最长轴 P50/P95 为 `9.2/27.2 mm`，仅 `1.5%` 的可见点投影宽度超过 5 px。独立天空继续保持关闭：竞品天空确认使用 SH1，而仓库当前只有签名 SH1 初始化、没有独立天空训练器；旧 SH0 照片球壳已证实会透过前景串色，不得合入 V50c。下一步必须先通过 SuperSplat 墙体、雪堆、远树和 Tile 接缝视觉复核，再实现/验收真正独立的 SH1 天空优化与分层导出。

### 9.21 V63a 生产级跨视图 mesh 过滤与首次深度边界（2026-08-29）

问题现象：V61 全域仍有透明墙面、地面空洞和接缝；稀疏 LiDAR range 只覆盖少量像素，无法证明连续墙面和雪面的存在。已有 BPA mesh sidecar 又把所有插值像素当成同等可信，缺少逐像素跨视图一致性、遮挡与深度边界处理，因此不能直接进入 7480 步主训练。

修改文件：`cloudstudio_3dgs/data/mesh_geometry.py`、`cloudstudio_3dgs/geometry/mesh_cross_view_filter.py`、`tools/filter_mesh_geometry_cross_view.py`、`tools/build_v63_mesh_mainline.py`、`tools/evaluate_v63_boundary_gate.py`、`tests/test_mesh_geometry.py`、`tests/test_mesh_cross_view_filter.py`。

修改内容：对每个 Face4 选择最近两个不同物理相机且相同 face id 的邻居，执行逐像素三维重投影；深度边界与遮挡区域不作错误冲突票，容差为 `max(5 cm, 0.5% × 深度)`。真实 LiDAR 像素固定标为锚点；一致像素按支持率赋予置信度；无法观测或仅遮挡的像素保守保留并降权；只有至少两个可观测冲突且无一致支持的像素才剔除。sidecar 逐像素写入 `confidence` 与 `source_type`，Manifest 使用确定性 SHA256 签名。V63a 绑定过滤后 sidecar、关闭 DA2、保持严格固定拓扑，按竞品 `[5V,10V,5V]` 调度运行，并在零基 step 1871 首次启用 mesh depth 后，于 completed step 1872 受控停止。

验证方式与结果：374/374 个 sidecar 文件及 SHA256 通过，输入 `497,036,801` 个有效像素，输出 `497,034,304`，仅剔除 `2,497` 个重复跨视图冲突；其中 `42,740,977` 个 LiDAR 锚点、`272,917,438` 个跨视图支持像素、`181,375,889` 个不可观测保守保留像素，视图保留率 P05 为 `99.9971%`。定向单元测试 `9 passed`。V63a checkpoint 确认 `step=1872`、mesh depth 权重 `0.5`、首次 loss `0.00182`，高斯数恒定 `971,903`，峰值显存约 `2.52 GB`。相对同一运行 step 1870，固定 48 视图 PSNR `+0.00217 dB`、SSIM `-0.000007`、alpha P05 `+0.000141`、LiDAR alpha P05 `+0.000019`，深度 MAE 增加 `3.31 mm`（约 `0.89%`），说明首次接入没有立即破坏 RGB/alpha，但一帧 mesh-depth 更新不足以证明收益。

失败门禁：尺度审计发现最大轴超过 `0.2 m` 的高斯从 step 374 的 `51` 个持续增加到 step 1872 的 `196` 个，最大轴达到 `8.975 m`；其中 `52` 个超过 `0.5 m`，且中位 opacity 约 `0.983`。虽然可见高斯到 LiDAR 最近点 P95 仅 `2.12 mm`、超过 `0.3 m` 的可见漂浮点为 0，但这些高 opacity 巨型高斯会用模糊大球掩盖空洞，不能带入无 Cull 的固定拓扑长跑。

当前状态：`V63A_BOUNDARY_BLOCKED_BY_OVERSIZED_GAUSSIANS`。签名联合门禁为 `results/diagnostics/snow-tile1-v63a-boundary1872-joint-gate.json`，明确 `long_training_allowed=false`、`adaptive_growth_allowed=false`。不得继续 7480 步，也不得接入 Grow/Cull；下一步先用竞品 `0.2 m` 世界尺度约束的固定拓扑等价替代（硬夹或强尾部尺度屏障）做单变量短边界复验，同时确保 PSNR、alpha P05、LiDAR alpha 和深度不退化。

V63b 单变量复验：保持 V63a 的数据、损失、点数、学习率和 1872 步边界完全不变，只启用 `max_world_size_m=0.2` 硬保险丝；opacity、尺度软正则、各向异性正则和 lifecycle 仍关闭。结果最大轴严格降到 `0.2 m`，超过 `0.5 m` 的高斯从 `52` 个降到 0，深度 MAE 从 `0.3753 m` 改善到 `0.1686 m`；但同一 48 视图 PSNR 从 `19.1451` 暴跌到 `13.5354 dB`，alpha 均值/P05 从 `0.90575/0.54718` 降到 `0.75057/0.19411`，LiDAR alpha P05 从 `0.98040` 降到 `0.89789`。这证明 V63a 的高覆盖与 RGB 指标显著依赖少量巨型高斯掩盖固定 LiDAR 拓扑的真实覆盖缺口，简单限幅会重新暴露空洞。

修正后的结论：V63b 同样不得晋级；现有一对一 LiDAR 固定拓扑无法同时满足“小而薄”和连续 alpha。下一步不再微调 opacity 或继续放宽尺度，而是从已过滤 mesh 的高置信、非 LiDAR 锚点区域提取受控 surfel 初始化，只补真实连续表面的覆盖缺口；随后在 0.2 m 保险丝下重新执行短边界联合门禁。该初始化门禁通过前，7480 步训练、Grow/Cull、DA2 和全 Tile 扩展全部保持关闭。

### 9.22 V64 mesh completion 收益上限与 cap-aware 生命周期修正（2026-08-29）

问题现象：V63b 证明固定 LiDAR 拓扑在限制最大尺度后会暴露真实覆盖缺口。V64 先从生产级跨视图 mesh sidecar 中仅提取高置信、非 LiDAR 锚点、非深度边界且原模型 alpha 不足的位置，用局部 BPA 三角形的切平面 surfel 补洞；所有补点都必须位于真实 LiDAR 邻域，且不允许 range-only 插值冒充连续表面。外部竞品逆向材料同时恢复出一个此前缺失的生命周期分支：达到绝对 Gaussian cap、禁止继续 densify 时，Cull 不再沿用普通阈值，而是将 opacity 阈值乘 `0.25`，world/screen 尺度阈值各乘 `5`。

修改文件：`cloudstudio_3dgs/geometry/mesh_completion.py`、`cloudstudio_3dgs/training/surface_initialization.py`、`cloudstudio_3dgs/training/default_strategy_adapter.py`、`cloudstudio_3dgs/pipeline/mipmap_gate.py`、`tools/build_v64_mesh_completion_initialization.py`、`tools/build_v64_mesh_completion_fixed.py`、`tests/test_mesh_completion.py`、`tests/test_surface_initialization.py`、`tests/test_default_strategy_adapter.py`、`tests/test_mipmap_gate.py`。

修改内容：新增签名 `signed_precomputed_surfel` 初始化，逐点消费预计算的 xyz、三轴尺度和四元数，禁止 Trainer 重新估计几何。V64C 使用 `5,871` 个严格 completion surfel，总点数 `977,774`；V64D 放宽到仍需 alpha 与跨视图 mesh 支持、但允许单次支持的 source-type 3，使用 `9,634` 个 completion surfel，总点数 `981,537`。两者都继承 V63b 的学习率、mesh depth/normal 调度与 `0.2 m` 世界尺度保险丝，在 completed step 1872 受控停止。经典生命周期新增显式 `vendor_capacity_cull_profile=exact_relaxed_at_cap`：只有运行前人口已经达到签名绝对 cap 时才跳过出生并进入宽松 Cull；正常 densify 事件仍执行普通早/晚阈值。遥测记录实际 opacity/world/screen 阈值和是否处于 capacity-limited 模式。

验证方式与结果：V64D 全程保持 `981,537` 点，峰值显存约 `2.53 GB`，训练时 LiDAR point-to-plane 原始量约 `1.28e-5 m`。同一确定性 48 视图上，V63b / V64C / V64D 的 PSNR 分别为 `13.53543 / 13.57877 / 13.58002 dB`，SSIM 为 `0.592693 / 0.593522 / 0.593592`，alpha P05 为 `0.194113 / 0.211504 / 0.211408`，LiDAR alpha P05 为 `0.897893 / 0.902966 / 0.903244`，深度 MAE 为 `0.168606 / 0.168886 / 0.168666 m`。V64D 相比 V64C 仅增加约 `0.00125 dB`，说明继续加密固定 completion 已进入收益饱和；它可以补局部小洞，但不能单独形成竞品式纹理驱动容量分配。cap-aware Cull 的 adapter 与签名门禁定向回归为 `74 passed`。

竞品材料适用边界：材料中的 `22,452,075` 点属于 house0305，不是 snow；snow 四 Tile surface 终态仍为 `6,018,902` 点。材料反汇编得到的内部 `0.15` 也不能直接覆盖 snow 运行级恢复的 `0.2 m` split/world-cull 阈值，二者在 scene-scale/host 参数映射完成前只允许作为单变量 A/B。snow surface 继续使用 SH0，MCMC 与 redundancy cull 继续关闭；独立天空保留为后续 SH1 阶段。

当前状态：`V64_FIXED_COMPLETION_SATURATED_V65_CAP_AWARE_BOUNDARY_READY`。V64C/D 保留为固定拓扑安全基线，不继续 7480 步。下一轮 V65 只在 Tile_1 做竞品等价短边界：普通 projected gradient、pre-optimizer `Split -> Clone -> Cull`、绝对 cap、cap-aware 宽松 Cull、SH0、redundancy/MCMC 关闭；先在首次 lifecycle 和首次 cap-only lifecycle 核验点数、三类 Cull 数、真实 alpha、PSNR、深度、尺度和显存。两处边界都通过后，才允许继续一个完整 epoch；不得直接扩展全场景。

### 9.23 V65 容量滞回、尺度保险丝与早期 mesh-depth 对照（2026-08-30）

问题现象：V65a 按普通早期 Cull 在首次 step 500 事件中从 `981,537` 点直接删除 `324,404` 点，只剩 `661,444`，证明当前 LiDAR-first 初始化不能直接承受 `opacity<0.1`。V65b 以 `985,000` 诊断 cap 触发恢复出的宽松 Cull 后保留 `977,293` 点，但 relaxed world threshold 允许最大轴达到约 `0.519 m`；继续使用“仅精确触顶才宽松”的实现时，step 600 因人口略低于 cap 又回到普通 Cull，一次删除约 `340k`，形成容量振荡。即使点数得到保护，竞品每 300 步 opacity reset 仍会在当前训练环境把大量可用高 opacity 压到 `0.2`，造成真实 alpha 与 PSNR 明显下降。

修改文件：`cloudstudio_3dgs/training/default_strategy_adapter.py`、`cloudstudio_3dgs/pipeline/mipmap_gate.py`、`tools/build_v26a_boundary_gate.py`、`tools/promote_v26a_evaluation.py`、`tests/test_default_strategy_adapter.py`、`tests/test_mipmap_gate.py`。

修改内容：新增显式 CloudStudio 稳定化配置 `cloudstudio_relaxed_near_cap_0p99`，当生长后人口达到签名绝对 cap 的 `99%` 时继续使用宽松容量维护 Cull，避免在 cap 附近反复切换普通/宽松阈值。宽松 opacity Cull 与世界尺度保护拆开：opacity 使用 `0.25 × prune_opa_late`，但 Trainer 每步仍把最大轴限制在 `0.2 m`。promotion 工具能够审核并签名该容量事件，同时支持将 reset 延后到 step 3000，以及在 DA2 关闭时让已签名、跨视图过滤的 mesh depth 从 step 1 以权重 `0.5` 消费。所有能力均为显式单变量配置，不冒充竞品原样复刻。

验证方式与结果：定向门禁与生命周期回归 `75 passed`。V65d step 600 在 99% 滞回下由 `977,177` 点新增 `5,946`、删除 `5,628`，保持 `977,495`，不再崩塌；但 exact reset 使 PSNR `12.239→11.089 dB`、alpha `0.6889→0.6148`、LiDAR alpha P05 `0.9051→0.8069`，因此被门禁拒绝。V65e 延后 reset 后，step 602 达到 PSNR `13.079 dB`、SSIM `0.5857`、alpha `0.7217`、alpha P05 `0.1898`、LiDAR alpha P05 `0.909`。V65f 继续到 step 1002，六次事件始终维持约 `976k–978k` 点，PSNR `14.997 dB`、SSIM `0.6033`、alpha `0.7987`、alpha P05 `0.2706`，但 mesh depth 在竞品 `5V=1870` 边界前为 0，深度 MAE 升到 `0.3002 m`。

早期 mesh-depth 与细节 Split：V65g 从 step 1 开启跨视图 mesh depth 后，step 1002 PSNR/SSIM 为 `15.016/0.6036`，深度 MAE 改善到 `0.2669 m`，但六轮仍全部为 Clone。V65h 在同一配置上启用 `>2 cm` 且屏幕足迹 `>0.0035` 的 screen-detail Split；step 600–1000 累计 Split parent `10,853`、生成子点 `21,706`，最终 `975,931` 点，最大轴严格为 `0.2 m`。同一 48 视图达到 PSNR `15.015 dB`、SSIM `0.5976`、深度 MAE `0.1354 m`、alpha `0.756`、alpha P05 `0.239`、LiDAR alpha P05 `0.89`，相对 V64D 固定拓扑的 `13.580/0.5936/0.1687 m/0.756/0.211/0.903`，已在画质、低分位覆盖和深度上形成更好的综合平衡。

补充几何健康门：V65h 没有 `>0.5 m` 巨型高斯，最长轴 P50/P95 为 `8.00/19.23 mm`，但可见高斯中 `31.3%` 的屏幕宽度仍超过 5 px；同时出现 `1,525` 个 opacity `>0.1` 且距初始化表面超过 `0.3 m` 的点，其中 `56` 个超过 `1 m`。同一审计下 V64D 对应数量为 0，说明这些不是初始化远场，而是大父高斯 screen-detail Split 的随机三维偏移。V65i 将 LiDAR tangent proposal 应用于所有新生点后把 `>0.3 m` 降到 2，但 PSNR/alpha 降到约 `11.454 dB/0.68`；V65j 只保留位置投影仍为 `11.424 dB`，证明全量 guard 对 Clone 的重定位和候选拒绝会破坏覆盖，不能晋级。

当前状态：`V65H_STEP1002_VISUAL_CANDIDATE_SPLIT_OUTLIER_GATE_BLOCKED`。完整保留 `975,931` 个 SH0 Gaussian 的 PLY 为 `outputs/snow-20260224-full-20260825/v65h_nearcap_scaleguard_deferredreset_meshfromstart_screendetail_review1002/snow_tile1_v65h_step1002_sh0_full.ply`。在 SuperSplat 完成墙面、雪堆、地面、远树与空洞 ROI 复核前，不继续到 3000 reset 或 7480 步。下一代码边界必须把 LiDAR tangent 约束限定到随机偏移的 Split 子点，Clone 保持父点位置、尺度和旋转；通过画质、alpha 与 `>0.3 m` 新生点联合门后，才允许推进一个完整 Tile epoch。

### 9.24 V66 严格 mesh admission、直接 Gaussian normal 与 5V 门禁（2026-08-30）

问题现象：V63 跨视图过滤仍把 `source_type=4` 的“不可观测保守保留”像素送入 Trainer，且 mesh normal 监督来自渲染深度差分，不是 Gaussian 自身最短轴。对齐后的 DA2 Manifest 只保存每视图 RANSAC 参数，实际张量仍在原始 DA2 cache；若把 Manifest 目录误作数据根，会在首个有效 DA2 视图失败。旧 `0.2 m` 世界尺度保险丝还采用硬裁切，和竞品逐步 Shrink 语义不同。

修改文件：`cloudstudio_3dgs/data/mesh_geometry.py`、`cloudstudio_3dgs/training/backend.py`、`cloudstudio_3dgs/training/bilateral_grid.py`、`cloudstudio_3dgs/training/exposure.py`、`cloudstudio_3dgs/training/face_dataset.py`、`cloudstudio_3dgs/training/regularization.py`、`cloudstudio_3dgs/training/trainer.py`、`tools/build_lidar_mesh_candidate.py`、`tools/build_strict_mesh_admission.py`、`tools/build_v66_competitor_5v_gate.py`、`tools/smoke_direct_gaussian_normals.py` 及对应测试。

修改内容：独立签名 strict mesh cache 只允许 `source_type=2/3`，完全拒绝 `source_type=4`；逐视图 LiDAR overlap P95 超过 `0.10 m` 或缺失统计时整视图关闭 mesh。Trainer 读取 strict Manifest 后再次 fail-closed 检查 source type。Gaussian normal 改为当前最短尺度轴经四元数旋转后的直接可微 normal，前中期对齐 mesh normal，15V 后可用 `0.01` 与 rendered-depth normal 自洽。曝光模型增加两组物理相机共享的 RGB gain+bias，并串接按相机共享、恒等初始化的 `16×16×8`、12 通道 3D BilateralGrid。世界尺度超限点每步三轴共同乘 `0.8`，不再逐轴硬截到 `0.2 m`。surface 训练继续独立于 SH1 sky；生成白/黑背景两份签名 5V 配置，但不并发训练。

验证方式与结果：strict cache 共 374 个视图、约 `2.919 亿` 个允许像素；其中原生 LiDAR anchor `42,740,977`，跨视图一致支持 `272,917,438`，type 4 实际进入数为 0。逐视图门禁保留 268 个、关闭 106 个。10 cm 空间块隐藏 20% LiDAR 后重建 mesh，总体 point-to-mesh P50/P95 为 `1.52/4.84 cm`；雪、墙、地面、植被和深度边界 proxy 的 P95 分别为 `4.57/4.68/4.85/4.84/4.89 cm`。这些分类明确是 RGB/几何 proxy，不冒充人工语义真值；25 cm 隐藏块五类 P95 都约 12 cm，已作为失败边界保留。直接 normal CUDA smoke 的 quaternion/scale/最短轴 scale 梯度范数分别为 `0.3296/0.0397/0.0151`，证明梯度真实穿过 rasterizer。相关定向回归为 `90 passed`。

当前状态：`V66_BLACK_5V_RUNNING_WHITE_ARM_SIGNED_NOT_STARTED`。黑背景臂采用竞品 `[5V,10V,5V]` 时序，DA2 仅在 363/374 个 RANSAC 有效视图存在，mesh depth 在严格 `step>5V` 后开启，mesh normal 前中期为 `0.05`；固定拓扑运行只到首次跨过 5V 的 completed step 1872。当前不得启动白背景并行臂、自适应 Grow/Cull、全场景或独立天空长训；5V 后必须联合审查 alpha、最差视图、PSNR/SSIM、held-out LiDAR、墙厚、短轴/轴比、低 opacity、floaters 和 PLY 视觉。

5V 实测与阻断：黑背景臂已在 completed step 1872 停止，点数恒定 `971,903`，峰值训练显存约 `2.79 GiB`，最大轴 `<0.2 m`，可见高斯距 LiDAR 超过 `0.3 m` 的 floater 为 0。与 V64D 使用同一黑背景、同一 24 个确定性视图比较，V66 的 PSNR `10.801 vs 15.139 dB`、SSIM `0.1948 vs 0.5299`、alpha 均值 `0.3462 vs 0.7941`、alpha P05 `0.0084 vs 0.3175`、LiDAR alpha P05 `0.0822 vs 0.8893`、深度 MAE `0.1100 vs 0.1065 m`。V66 opacity `<0.1` 比例达到 `61.90%`，墙面有效厚度 P95 为 `13.40 cm`，短轴/轴比 P50 为 `3.78 mm / 2.03`。签名联合门禁状态为 `BLOCKED`，白背景臂、长训和自适应生长均禁止。

根因修正：检查 step 374 辅助参数发现 per-camera bias 已到约 `0.158`，BilateralGrid 相对恒等的最大偏移约 `0.159`。旧实现把 gain+bias 和 grid 仿射应用于背景合成后的 RGB，未把 bias 乘渲染 alpha；因此全透明像素也能被 photometric bias 直接涂色，形成“RGB 下降、opacity 消失”的确定性捷径。现已改为对 premultiplied foreground 应用线性项，所有 affine bias 必须乘 alpha，并原样恢复 `(1-alpha)×background`；透明黑/白背景不变量新增回归，定向测试更新为 `91 passed`。下一轮必须从 V64D 安全基线重新开始，先做 374 步 alpha-safe photometric 单变量臂；不得从失败的 V66 checkpoint 续训，也不得同时开启 Grow/Cull。

### 9.25 V67 alpha-safe 重启与 5V 联合门禁（2026-08-30）

问题现象：V66 的 gain+bias/BilateralGrid 能越过 alpha 直接涂背景，导致 opacity、PSNR 和结构同时坍塌；失败 checkpoint 不具备续训价值。V64D checkpoint 已含旧 exposure gain，而 BilateralGrid 的恒等初始化又不是全零，原 warm-start 辅助参数契约无法表达“丢弃已有 gain，并以零 bias、恒等 grid 重新开始”。

修改文件：`cloudstudio_3dgs/training/trainer.py`、`tools/build_v67_alpha_safe_photometric_probe.py`、`tools/evaluate_v66_5v_gate.py`、`tests/test_training.py`。修改内容：warm-start 明确允许签名为 fresh 的源辅助参数被忽略；非 fresh 参数仍必须完整匹配，未知参数仍 fail-closed；BilateralGrid fresh 参数允许有限的仿射恒等张量，其余 fresh 参数仍强制全零。V67 从 V64D step 1872 的 `981,537` 个安全高斯开始，光度辅助参数全部重新初始化；先运行 374-view 单变量探针，通过与 V64D 相同 24 视图门禁后，才签名并重新运行到首次跨过 5V 的 step 1872。DA2、Grow/Cull 和 sky 均关闭，mesh 只允许 source type 2/3。

验证方式与结果：warm-start 与 alpha-safe BilateralGrid 定向测试 `61 passed`。V67A step 374 相对 V64D 同视图白背景，PSNR 仅变化 `-0.022 dB`，alpha 基本不变，LiDAR alpha P05 提升约 `0.05`，深度 MAE 从 `10.65 cm` 降至 `9.03 cm`，证明透明捷径已被切断。V67B step 1872 白背景 PSNR/SSIM 为 `14.765/0.6224`，黑背景挑战为 `15.187/0.5337`，分别优于 V64D 的 `14.480/0.6167` 与 `15.139/0.5299`；深度 MAE 为 `9.15 cm`，alpha 均值/P05 为 `0.7988/0.3297`，LiDAR alpha P05 为 `0.9572`。无超过 `0.2 m` 高斯，无超过 `0.3 m` 可见 floater。

当前状态：`V67B_5V_ALPHA_DEPTH_PASS_SHAPE_BLOCKED`。墙面有效厚度 P95 为 `11.26 cm`，高于 10 cm 门槛；可见高斯短轴 P50 为 `3.05 mm`，轴比 P50 仅 `3.01`，仍未形成竞品式薄盘。opacity `<0.1` 比例为 `41.80%`，已通过当前 50% 门槛，因此下一步不再改 alpha，而应单变量标定 mesh-normal 对齐与 flatten/shape 学习率；在墙厚和轴比通过前，不继续 7480 步、不启用 Grow/Cull、不扩展全场景。

### 9.26 V68–V70 最短轴单向压平与覆盖安全回收（2026-08-30）

问题现象：V67B 的 `flatten_mode=absolute_m` 使用 20 mm 上限，而当前最短轴中位数仅约 3 mm，压平损失几乎恒为 0。V68 改用尺度无关的短轴/切向几何均值比例后，强臂虽把最短轴 P50 从 `3.05 mm` 降到约 `2.77 mm`、轴比升到 `3.49`，却同时把最长轴和屏幕足迹撑大；可见高斯宽度超过 5 px 的比例从约 `35.1%` 增到 `37.7%`，说明原比例损失可以通过放大两个切向轴作弊，会制造新的模糊大高斯。

修改文件：`cloudstudio_3dgs/training/lidar_normals.py`、`cloudstudio_3dgs/training/trainer.py`、`cloudstudio_3dgs/pipeline/mipmap_gate.py`、`tools/build_v69_shortest_only_shape_probe.py`、`tools/build_v70_shape_appearance_settle.py`、`tests/test_lidar_normals.py`、`tests/test_mipmap_gate.py`。修改内容：新增签名模式 `tangent_ratio_shortest_only`，仍以切向两轴几何均值对短轴做尺度归一化，但对分母停止梯度，保证优化只能缩短最短轴，不能通过扩张切向轴降低损失。门禁仅允许固定点数、冻结 means/opacity/SH0/曝光/BilateralGrid 的 50/150 步形状探针；新增梯度单测明确验证最短轴梯度非零、两个切向轴梯度严格为 0。随后 V70 从 V69B 独立恢复，冻结全部几何和光度辅助参数，仅用低学习率 SH0/opacity 加 `LiDAR alpha=0.95`、3 px 膨胀覆盖地板进行 100 步外观回收。

验证方式与结果：法线损失与签名门禁定向测试 `41 passed`。V69A 50 步保持 `981,537` 点且无任何 Grow/Cull，最短轴 P50 约 `2.8 mm`、轴比 P50 `3.3`，超过 5 px 的比例保持约 `35.1%`，没有重现 V68B 的切向膨胀。V69B 150 步把最短轴 P50 继续降到 `2.4 mm`、轴比升到 `3.8`，最长轴 P50 仍约 `9.1 mm`；无超过 0.2 m 高斯、无超过 0.3 m 可见 floater。相对 V67B 的同一 24 视图，V69B 白背景 PSNR/SSIM 从 `14.7645/0.62238` 提升到 `14.8083/0.62343`，黑背景 PSNR 从 `15.1872` 提升到 `15.2707`，深度 MAE 从 `9.154 cm` 降到 `8.924 cm`；alpha 均值轻微下降约 `0.0015`，因此没有直接继续压平。V70 100 步外观回收后白背景为 `14.8252/0.62364`，黑背景为 `15.279/0.5345`，深度 MAE `8.947 cm`，alpha 均值/P05 `0.79767/0.32751`，LiDAR alpha P05 `0.95700`，基本恢复 V67B 的覆盖水平并保留形状收益。

当前状态：`V70A_TILE1_NUMERIC_DIRECTIONAL_PASS_VISUAL_REVIEW_PENDING`。完整无 opacity 裁剪 PLY 为 `outputs/snow-20260224-full-20260825/v70_shape_appearance_settle/snow_tile1_v70a_step100_sh0_full.ply`，包含全部 `981,537` 个 SH0 Gaussian。V70A 数值上优于 V67B，且没有透明捷径、巨型高斯或几何漂移；但轴比仍远低于竞品约 12，墙面有效厚度 P95 的 RANSAC 指标仍约 11 cm，并主要受冻结中心相对混合平面的残差支配。必须先做 SuperSplat 墙面、雪堆和远景视觉复核；通过前不继续压平、不启动 7480 步、不启用 Grow/Cull，也不扩展全 Tile。

### 9.27 V71 五 Tile 全场安全精修、full-halo 合并与原鱼眼协调（2026-08-30）

问题现象：用户完成 V70A PLY 人工复核后确认其局部效果可接受，并授权稍作优化后扩展全场。旧 V49 五 Tile 基线均存在，但 Tile_2/3/4 的 opacity 中位数仅约 `0.045/0.058/0.022`；如果只照搬 V70 的 100 步低学习率外观回收，黑背景 12 视图 alpha 均值仍只有 `0.54/0.39/0.31`，会复现透明墙面、地面和接缝，不能直接合并为最终结果。

修改文件：`tools/build_v49_full_area_surface_quality_phase.py`、`tools/build_v50_full_area_raw_fisheye_phase.py`、`docs/MIPMAP_ALIGNED_FACE4_PIPELINE_SOP.zh-CN.md`。修改内容：Tile 阶段新增签名 `shortest_shape` 和 `appearance` 模式，支持 150/250/500 步边界、`tangent_ratio_shortest_only`、冻结 exposure bias/BilateralGrid、关闭 DA2/mesh/rendered-normal 干扰，并显式绑定 `retain_full_halo`。五块从各自 V49 coverage50 安全 checkpoint 串行执行 150 步单向压平、100 步低学习率 SH0/opacity 回收；黑背景门禁不通过后，再统一执行 250 步 opacity-only 覆盖修复。合并保留全部训练 halo，不作 core 裁切或 opacity 过滤。全场随后以原始鱼眼执行 250 步无放回 SH0/曝光协调、250 步黑背景 opacity-only 覆盖校准和最后 250 步白背景 SH0/曝光回收。全场合并 checkpoint 缺少 per-camera exposure 辅助参数时，颜色构建器现在明确将 `exposure_log_gains` 标为 fresh 初始化；若调用方显式要求继承，则仍执行严格匹配。

验证方式与结果：五块 shape150、appearance100、coverage250 全部串行完成，点数始终固定为 `1,895,788 / 971,903 / 1,477,056 / 1,850,945 / 1,076,290`，峰值显存均低于约 `2.24 GiB`。每块均无超过 `0.2 m` 巨型高斯、无距 LiDAR 超过 `0.3 m` 的可见 floater。coverage250 后五块黑背景 LiDAR alpha P05 分别为 `0.85/0.88/0.82/0.80/0.84`，均明显高于 appearance100 的 `0.79/0.78/0.46/0.51/0.45`。full-halo 合并保留 `7,271,982/7,271,982` 个高斯。原鱼眼全场第一次颜色协调将白背景 12 视图 PSNR 从 `8.124` 提升到 `14.401 dB`；黑背景覆盖校准再把 LiDAR alpha P05 从 `0.85` 提升到 `0.89`。最终颜色回收后，白背景 PSNR/SSIM 为 `14.530/0.5651`，黑背景为 `8.193/0.3235`，深度 MAE `0.615 m`；白/黑背景共享 alpha 均值约 `0.71`、LiDAR alpha 均值/P05 约 `0.97/0.89`。全场可见高斯最短轴/最长轴 P50 为 `3.5/8.9 mm`，轴比 P50 `2.6`，仅 `1.7%` 的可见高斯投影宽度超过 5 px。

当前状态：`V71_FULL_AREA_RETAIN_HALO_NUMERIC_PASS_VISUAL_REVIEW_REQUIRED`。最终 PLY 为 `outputs/snow-20260224-full-20260825/full_area_safe_refine_v71/final/snow_full_area_v71_final_retain_halo_sh0_full.ply`，大小 `494,495,193` bytes，SHA256 为 `94553c2228d5ceeb6dd3d0eea471a1c14605569c317bcc666c298b9d76f3588e`，完整保留全部 `7,271,982` 个 SH0 Gaussian。相对旧全场路线，V71 明确提高了弱 Tile 与 LiDAR 表面的覆盖，并通过 full-halo 避免硬裁接缝；但最终交付仍以 SuperSplat 的墙面、雪堆、地面、远树和 halo 重叠区人工复核为准。人工复核前不启用 Grow/Cull、DA2、SH1 sky 或 7480 步长训。

### 9.28 澳洲 B 最新分支交叉核对与 V72 起点（2026-08-30）

问题现象：V71 人工复核仍弱于竞品，主要表现为墙面/地面透明、远景缺失和高斯偏厚偏圆。远端 `origin/machine-b/uk-quality` 已从 `51049d7` 更新到 `db2409d`，比当前已提交主线 `77e73ff` 多 69 个提交。当前工作树包含 V63–V71 尚未统一提交的研发修改，因此只执行 `git fetch --prune origin` 更新远端引用，不在脏工作树上直接 merge/pull，也不覆盖现有现场。

B 机器新增证据修正了三项判断。第一，训练期大面积天空若只由白背景承担，会持续奖励 surface opacity 下降；开启生命周期后会把这种压力兑现为删穿。其无 reset 生长虽跑到约 `9.2M` 点、空洞率约 `3.8%`，但产生约 `2.5M` 漂浮；加入 tangent birth guard 后几乎不变，说明主要漂浮来自已有高斯位置漂移，而不是新生点。第二，竞品交付天空是独立 `100k SH0`，不是必须 SH1；照片烘焙穹顶应在训练期提供天空归属，并在导出时与 surface 分开。第三，V71 已实现的超限高斯三轴共同乘 `0.8` 与 B 最新 `ShrinkBigScaleGS` 等价；不再恢复逐轴硬 clamp。B 的 house0305 单 Tile/18.76M 初始化结论不能直接套到 snow，但“按常驻高斯而非视图像素估算显存”和“全分辨率优先”应进入下一轮规划。

修改文件：`cloudstudio_3dgs/training/lidar_normals.py`、`cloudstudio_3dgs/training/sky_layer.py`、`tools/augment_checkpoint_sky.py`、`tests/test_tangent_isotropy.py`、`tests/test_sky_layer.py`。新增 `weight_tangent_isotropy`，在同一 LiDAR 平面锚集合上惩罚两条切向轴比值偏离 1，使“短轴压平”得到薄圆盘而不是薄针；默认权重为 0，旧配置不变。天空 checkpoint 增加 `--prebaked` 路径，严格检查参数形状、有限值和 SHA256，并把现有照片烘焙穹顶原样附加到 surface checkpoint。

验证方式与结果：定向回归 `38 passed`。已将 `full_area_v28/sky/snow_photo_baked_sky_dome_e90.pt` 的 100,000 个照片烘焙 SH0 天空高斯无损附加到 V71 最终 7,271,982 个 surface 高斯，得到 `v72_bmachine_absorption/warm_start/snow_v71_plus_prebaked_sky_e90.pt`，总数 `7,371,982`，未执行 opacity 裁剪、训练或 Grow/Cull。

当前状态：`V72_PREBAKED_SKY_WARM_START_READY_FULLRES_PROBE_PENDING`。下一步不是直接长训，而是全分辨率短 A/B：保留 V71 作为 surface-only 对照，V72 使用训练期照片天空；先验证白/黑背景、surface-only alpha、天空泄漏和远景，再启用软 point-to-plane 位置锚与薄圆盘形态。只有已有高斯漂移受控、surface opacity 不再下降后，才允许测试 projected-gradient 生命周期。

### 9.29 V72 全分辨率天空所有权 A/B 与分层检查点守卫（2026-08-30）

问题现象：V72 派生暖启动虽记录 `sky_layer`，但 Trainer 保存下一代 checkpoint 时会丢失 surface/sky 行边界。后续导出、几何约束或继续训练将无法可靠区分前 7,271,982 个 surface Gaussian 与末尾 100,000 个 sky Gaussian。旧全场 raw-fisheye 配置还会在训练结束后渲染全部验证图，短 A/B 会把大量时间浪费在尚未晋级的候选上。

修改文件：`cloudstudio_3dgs/training/checkpoint.py`、`cloudstudio_3dgs/training/trainer.py`、`cloudstudio_3dgs/training/sky_layer.py`、`tools/build_v50_full_area_raw_fisheye_phase.py`、`tests/test_training.py`、`tests/test_sky_layer.py`。

修改内容：新增带 SHA256 的 `model_layers` 契约，约束行序必须为连续的 `surface_then_sky_contiguous_rows`，并兼容旧 `sky_layer`。Trainer warm-start 读取并向下一代 checkpoint 传播该契约；含分层模型强制 `strict_fixed`，禁止暖启动剪枝和全局尺度倍率，避免行边界失效。全场 raw-fisheye 构建器新增 `--factor {1,2,4}` 与显式签名的 `--defer-final-evaluation`：训练结束先保存模型，再以独立确定性抽样评估晋级，晋级后才做全量评估。

验证方式与结果：选定回归集 `97 passed`。factor=1、7,371,982 Gaussian、10-step photo-baked sky smoke 已完成训练步并原子写入 checkpoint；Trainer 峰值显存为 `6,731,443,200 bytes`。新 checkpoint 的 `model_layers_sha256=8fadcce389a459cbe883f05b3c0676c936d7658707ae201c0334186e27948576`，surface 为 `[0,7271982)`，sky 为 `[7271982,7371982)`，点数与 strict-fixed 拓扑保持不变。smoke 的全量最终评估在 checkpoint 落盘后人工中止；这不是训练失败，而是用于确认并消除低价值全量评估路径。

当前状态：`V72_FACTOR1_PREBAKED_SKY_374_RUNNING`。正式臂冻结 means/scales/quats/opacities，只训练 SH0 颜色与 per-camera exposure；完成后先做 24/36 帧白/黑背景及 surface-only alpha 联合门禁。只有优于 surface-only control 才运行对照臂或进入下一阶段，当前仍不启用 Grow/Cull/reset。

### 9.30 竞品 DLL 对抗性复核、生命周期容量语义修正与 V84 短 A/B（2026-08-31）

问题现象：V73–V83 已恢复普通投影位置梯度、无放回逐 epoch 视图采样、半径权重、Tile 外衰减和训练前生命周期，但仍混有若干 CloudStudio 自定义保护，不能仅凭“参数看起来接近”认定与竞品等价。尤其当前实现先完成 Grow，再按增长后的点数决定是否进入容量受限 Cull；这会在一次事件内提前切换 Cull 模式，与竞品行为不一致。

DLL 证据：对原始 `gaussian_splat.dll` 的机器码重新核验。`GaussianSplatModel::AfterTrain` 在 RVA `0xCE460–0xCE488` 先读取增长前点数并与容量上限比较，随后冻结本轮容量状态；投影梯度载体在 RVA `0xCD58E–0xCD5DC` 从模型偏移 `0x70` 取得保留的 rasterizer 投影张量，读取 `.grad()`、`.detach()` 后计算 L2 norm。构造函数 RVA `0xC8D1C–0xC8D41` 显示受保护前缀长度来自预载 `GaussianSplatData` 行数，而不是 LiDAR anchor 数量；导出接口又分别存在 `InitialParameters(PointCloud)`、`InitialParameters(GaussianSplatData)`、`RunScaffold` 与 `TrainBackground`。因此 Snow 的独立 surface 训练不得把 LiDAR 点误设为 protected prefix；只有预载 scaffold/background 与 surface 拼接在同一模型时才适用该保护。

修改文件：`cloudstudio_3dgs/training/default_strategy_adapter.py`、`tests/test_default_strategy_adapter.py`、`tools/build_v84_opacity_scope_revisit_ab.py`、`docs/MIPMAP_ALIGNED_FACE4_PIPELINE_SOP.zh-CN.md`。修改内容：容量模式改为在 `_grow_mipmap` 前计算并冻结，随后 Grow 与 Cull 共用该增长前状态；新增 `capacity_limited_pre_growth` 和 `capacity_trigger` 遥测；增加跨越容量阈值的回归测试。V84 从 V83A step 3299 的完全相同 checkpoint 构造 100 步、无拓扑事件的两臂签名 A/B：控制臂保持当前 `visible_current_view` opacity mean，竞品臂改为 surface 全体 opacity mean；随机种子、无放回视图序列、DA2/mesh、优化器和其余损失全部保持一致。

验证方式：生命周期与门禁定向回归 `83 passed`；扩大测试集除一个本机 gsplat 未以 `GSPLAT_BUILD_3DGUT=1` 构建而无法运行的 CUDA 渲染尺度测试外，其余 `142 passed`。该项是运行环境算子缺失，不是本次逻辑回归通过。V84 仅用于判定在当前 dense geometry 与无放回视图条件下，竞品式全体 opacity 正则是否仍会造成透明表面；两臂在完成确定性 24 视图白/黑背景、alpha、LiDAR alpha、深度和 opacity 分布联合审查前，不允许启动下一次 Grow/Cull 或全场长训。

当前状态：`V84_OPACITY_SCOPE_REVISIT_AB_READY`。已确认 V83 当前仍低于 `4,278,000` 容量上限，因此这次容量语义修复不会改变此前 V83A step 3299 的数值结果，只会修正未来首次跨过上限的生命周期事件。当前已证实方向是“竞品投影梯度生命周期 + dense geometry + 无放回视图”；仍待短 A/B 定标的是 opacity 正则作用域。观察保护 Cull、激进 screen-detail Split 和部分天空代理仍属于 CloudStudio 偏差，不能在本轮被表述为已复刻竞品。

### 9.31 Snow type-2 参数反证、V85 生命周期失败边界与 V86 定标方向（2026-08-31）

问题现象：另一主机把 DLL 构造函数默认的 `split=0.01 / grad=2e-4 / screen=0.1` 误写成 Snow High/type-2 运行值，并据此准备修改训练参数。对同一 DLL 的 preset 分支复核证明两组数可以同时存在：构造默认值随后被 type-2 覆写为 `split=0.2 m / grad=1.5e-4 / screen=0.15`。因此 Snow 不得按构造默认值改写任务参数。该主机随后已承认遗漏 preset 覆写。其 house 全密度/1.5 cm 抽稀尺度对比也不能直接套到本机 Snow Tile_0；本机签名初始化实际为 `2,851,911` 点，切向尺度 P50 `8.255 mm`、法向尺度 P50 `4.128 mm`，不存在其报告的 `19.3 mm` 初始化尺度。

修改文件：`cloudstudio_3dgs/training/default_strategy_adapter.py`、`cloudstudio_3dgs/training/trainer.py`、`cloudstudio_3dgs/pipeline/mipmap_gate.py`、`tests/test_default_strategy_adapter.py`、`tests/test_mipmap_gate.py`、`tools/build_v85_recovered_snow_lifecycle_probe.py`、`tools/build_v85b_recovered_lifecycle_settle.py`。修改内容：Snow type-2 保持 `1.5e-4 / 0.2 m / 0.15`；恢复 DLL 证据支持的 `reset=300`、禁用 split revised-opacity，并移除机器码中“计算但未被消费”的 growth opacity gate。容量状态固定在 Grow 前计算。V85 从相同初始化跑到首个 step-500 生命周期；V85b 精确恢复到 step 702，观察 step 600 reset 与 step 700 第二轮 Grow/Cull，不改变训练参数。

验证方式与结果：定向生命周期与门禁测试 `83 passed`，续跑门禁测试 `17 passed`。V85 step 500 从 `2,851,911` 点 Clone `65,384`、Split `0`、Cull `1,105,029`，剩余 `1,812,266`。V85b step 600 Clone `61,667`、Cull `144,094` 并 reset；step 700 Clone `23,412`、Cull `437,460`。step 702 最终仅 `1,315,791` 点，累计出生 `150,463`、删除 `1,686,583`，三轮 Split 均为 0。确定性 24 原鱼眼验证为 PSNR `5.481 dB`、SSIM `0.1120`、深度 MAE `1.046 m`；代表视图已经出现建筑、雪堆和边缘缺失。V85/V85b 因此不得晋级长训。

当前状态：`V85_IMMEDIATE_CULL_FAILED_V86_CALIBRATION_REQUIRED`。下一版不得继续 immediate opacity Cull，也不得用训练步数掩盖结构错误。高置信定标范围冻结为：普通 `1.5e-4` 投影梯度、`0.2 m` 竞品世界分界、`0.01 m + 0.0025` screen-detail Split、split children `/1.6` 且复制父 opacity；`observation_aware` Cull 至少 64 次观测、连续 2 次低 opacity、早晚阈值均 `0.05`；reset 延迟到 3000；容量上限 `4,278,000`；opacity mean 权重先降到 `0.0025` 且只作用当前可见 surface；mesh depth 从 step 1 为 `0.5`、mesh normal `0.05`、后期 rendered-normal `0.01`；黑背景 surface 与独立天空分离。`8 mm` 候选因缺少真实 A/B 证据且不在现有签名允许档内被门禁拒绝，V86 使用已在 V83 真实触发 Split 的 `10 mm` 激进档。step-502 边界先采用既有签名 `observation_cull_v2_conservative`（单轮上限 `2%`、reset grace 200），因为连续两轮门使首次事件不执行 opacity Cull；计划中的 `0.5%/300-step` 慢 Cull 必须在边界通过后作为独立签名续跑档实现和测试，禁止绕过门禁直接写入配置。该组是基于 V65h/V83/V85 失败边界形成的 CloudStudio 质量臂，不冒充竞品 bit-exact。只允许先跑 step 502/1002 联合门禁，通过 alpha、PSNR/SSIM、深度、Split 比例、Cull 净变化、尺度和 floater 后再冻结长跑参数。

### 9.32 V86/V86b 证据定标边界与正确 Tile 评估口径（2026-08-31）

问题现象：V85 证明直接恢复竞品 Snow type-2 的 immediate Cull/reset 会破坏本机 Snow，但此前原鱼眼 24 视图评估又把单个 Tile_0 模型用于全场验证相机，若干相机几乎看不到该 Tile，导致 alpha 和 PSNR 被错误拉低。另一个名为 `white` 的旧输出实际继承 checkpoint 黑背景，不能作为白背景挑战证据。

修改文件：`tools/build_v86_evidence_calibrated_boundary.py`、`tools/build_v86b_evidence_calibrated_review.py`、`docs/MIPMAP_ALIGNED_FACE4_PIPELINE_SOP.zh-CN.md`。修改内容：V86 从签名 dense Tile_0 初始化开始，采用普通半径加权投影梯度 `1.5e-4`、竞品 `0.2 m` 世界尺度分界、`10 mm + 0.0025` screen-detail Split、无 revised opacity、64 次观测/连续两轮/单轮至多 2% 的受限 Cull、reset 延迟到 3000。V86b 从 V86 step502 精确 checkpoint 续跑到 step1002，不改变任何训练、loss、梯度、Split、Cull、reset 或容量参数。评估改为绑定配置中 Tile_0 的 624 个 Face4 视图，并使用确定性分层固定 24 视图分别覆盖黑、白背景。

验证方式与结果：V86 step502 从 `2,851,911` 点开始，Clone `74,219`、Split parent `126,149`、Split children `252,298`、Cull `1,003`，得到 `3,051,276` 点；首次生命周期由错误的 Clone-only 转为 Split 主导。V86b step600/700/800/900/1000 后的点数依次为约 `3.130M / 3.173M / 3.212M / 3.254M / 3.296M`，显存峰值约 `4.76 GiB`，没有出现 V85 的百万级瞬时删除。相同 24 个 Tile Face4 视图中，step502→1002 的黑背景 PSNR `10.905→11.321 dB`、SSIM `0.4240→0.4391`；白背景 PSNR `10.072→11.045 dB`、SSIM `0.5411→0.5605`；整体 alpha `0.551→0.614`，LiDAR 像素 alpha `0.869→0.918`。但深度 MAE 从 `8.32 cm` 上升到 `10.04 cm`，所以仍不得直接晋级全程长跑。

当前状态：`V86B_STEP1002_VISUAL_COVERAGE_PASS_GEOMETRY_GUARD_REQUIRED`。step1002 完整 SH0 PLY 已导出 `3,296,227` 点。尺度审计为最长轴 P50 `9.23 mm`、最短轴 P50 `1.89 mm`、轴比 P50 `4.81`；相比竞品约 `0.7–0.8 mm` 短轴和约 `12` 的轴比仍偏厚、偏圆。下一轮应从同一 step1002 checkpoint 做短程受控几何锐化：保留当前 Split/Cull 和 photometric 参数，只提高 shortest-axis flatten/mesh-normal 约束并增加弱 point-to-plane 守卫；不得同时修改梯度阈值、背景、采样顺序或容量上限。验收必须同时要求 alpha 不回退、黑白 PSNR/SSIM不回退、深度 MAE不再恶化、短轴下降且实际 PLY 无新增雾团。

### 9.33 V87 最高纹理档 Split 瓶颈、8 mm 锐化门禁与全程晋级（2026-08-31）

问题现象：V86b PLY 已明显优于早期 Clone-only 路线，但砖石、墙板分界和雪边仍比竞品模糊。对相同 LiDAR 初始化和竞品只读 0.5 m 图像纹理 voxel 做回归后，V86b 的纹理—密度相关已达 `rho=0.4231`，证明 projected-gradient 主链不是失效；低纹理前两档净减少约 `6%/3%`，中间两档净增加约 `14%/18%`，但最高纹理五分位净增中位数回落为 `0%`。V86b 最长轴 P50 仅 `9.23 mm`，而 screen-detail Split 的物理门为 `>10 mm`，使最精细区域的小高斯只能 Clone、不能继续缩小。

修改文件：`cloudstudio_3dgs/training/default_strategy_adapter.py`、`cloudstudio_3dgs/training/trainer.py`、`cloudstudio_3dgs/pipeline/mipmap_gate.py`、`tests/test_default_strategy_adapter.py`、`tools/build_v87_ultrasharp_boundary.py`、`tools/build_v87b_ultrasharp_full.py`、`docs/MIPMAP_ALIGNED_FACE4_PIPELINE_SOP.zh-CN.md`。修改内容：新增显式签名 `lidar_surface_screen_detail_ultrasharp` 档，只将物理 Split 门从 `10 mm` 降到 `8 mm`，保持普通投影梯度 `1.5e-4` 和屏幕半径 `0.0025` 不变。短轴比例目标从 `0.10` 调为 `0.08`，flatten 权重从 `0.02` 调为 `0.05`，增加 `0.01` 的 2 cm Huber point-to-plane 软守卫；采样、RGB/SSIM、DA2/mesh、Cull、reset、背景和容量均不变。

验证方式与结果：新增 8–10 mm 高梯度高斯进入 Split 的单元测试，定向生命周期和门禁回归 `84 passed`。V87 从 step1002 续跑到 step1202，两轮后点数从 `3,296,227` 增至 `3,399,399`；最长轴 P50 `9.23→9.12 mm`，最短轴 P50 `1.89→1.50 mm`，轴比 P50 `4.81→6.07`，opacity `<0.1` 比例 `19.97%→19.19%`。固定 24 个 Tile Face4 视图中，黑背景 PSNR `11.321→11.403 dB`、SSIM `0.4391→0.4461`；白背景 PSNR `11.045→11.254 dB`、SSIM `0.5605→0.5652`；整体 alpha `0.614→0.622`，LiDAR alpha `0.918→0.929`。深度 MAE `10.04→10.68 cm`，增加约 `6.4 mm`，未出现离面失控。纹理—密度相关进一步升至 `rho=0.4627`，cross-view luma MAD 相关升至 `0.5042`。

当前状态：`V87_ULTRASHARP_BOUNDARY_PASS_FULL12480_RUNNING`。全程臂从签名 step1202 checkpoint 续跑至 step12480，不再更改参数；checkpoint 每 500 步滚动覆盖，只在最终步永久保留，避免磁盘再次被中间 checkpoint 填满。step1300 后点数约 `3.450M`、显存峰值约 `4.92 GiB`，仍低于 `4.278M` 容量和 7.5 GiB 显存门。后续必须重点检查 step3000 reset、首次到达容量上限后的净 Cull、最短轴是否继续逼近竞品以及 raw-fisheye/PLY 是否重新出现天空污染或雾状大高斯。

### 9.34 V87 全程尺度尾部回归与 V88 定向修复（2026-08-31）

问题现象：V87 Tile_0 完成 step12480 后，24 个固定 Face4 视图的白背景 PSNR/SSIM 从 step1202 的 `11.254/0.5652` 提升到 `14.448/0.6511`，alpha 从 `0.625` 提升到 `0.730`；但渲染深度 MAE 从 `10.68 cm` 回归到 `24.57 cm`。结构审计显示可见高斯最长轴 P95 已增至 `7.23 cm`，19.7% 的可见高斯投影宽度超过 5 px，轴比 P50/P95 为 `26.1/168.4`，并有 20.22% opacity 低于 0.005。说明中心点到 LiDAR 的约束仍健康，但无尺度尾部正则的长跑让少量超宽、极薄高斯以覆盖和 PSNR 换取深度偏移与雾状细节。

修改文件：`tools/build_v88_tail_repair.py`、`docs/MIPMAP_ALIGNED_FACE4_PIPELINE_SOP.zh-CN.md`。

修改内容：新增签名 V88 200 步稳定阶段，严格从 V87 step12480 checkpoint 恢复，冻结 means，且 topology 生命周期已在 step9360 后停止。只对超过 3 cm 的高斯执行竞品式逐步 `×0.8` 全轴收缩；对最大 5% 尺度尾部和轴比超过 64 的极端高斯施加软正则，同时以低学习率继续优化 scale/quaternion/opacity/SH0，并把 LiDAR alpha 权重从 0.02 提至 0.05，防止收缩尾部时重新打穿表面。

验证方式：构建器必须通过 `TrainerConfig`、签名 Gate、Python 编译和 UTF-8 乱码检查；真实 GPU 只跑到 step12680。晋级要求同时复核固定 24 视图黑/白背景 PSNR/SSIM/alpha/depth、尺度 P50/P95、>5 px 比例、opacity、floater 和实际 PLY。V88 未通过前不得复制到其余 Tile。

当前状态：`V87_FULL_COMPLETE_V88_SCALE_TAIL_REPAIR_PREPARING`。
### 9.35 V89：其余四分块的 Mesh 失败关闭与 LiDAR 可靠全量臂

Tile_1 的 0.5 m 空间块隐藏 10% 验证显示：整体点到重建表面 P95 为 0.221 m，雪、墙、地面、植被及深度边界代理类别 P95 均超过 0.10 m。该结果不满足生产 Mesh admission 门禁，因此不得把 Tile_0 的 Mesh/DA2 配置机械复制到其余块。

新增 `tools/build_v89_remaining_tiles.py`：沿用 V87 已验证的普通 projected-gradient、screen-aware ultrasharp Split、观察感知 Cull、LiDAR 切平面出生与 20V 生命周期，但对未通过块状隐藏门禁的 Tile fail-closed 关闭 Mesh depth、mesh normal 和 DA2，恢复权重 0.05 的真实稀疏 LiDAR range，并绑定各 Tile 的独立 LAS、K7/K30 几何、Face4 LiDAR sidecar、core box、视图数和 1.5 倍容量上限。此臂是可靠退化路径，不把不可信稠密几何用于长训练；每 5V 保存 checkpoint，最终仍需独立质量门禁后才能合并。

### 9.36 V88b 墙面空洞复核与可信 Mesh alpha 覆盖修复

问题现象：用户在 V88b Tile_0 PLY 的墙面近景确认存在黑缝、局部孔洞和糊斑。该视觉失败与数值审计一致：约 20% 高斯 opacity 低于 0.005，约 31% 位于 0.005–0.1；稀疏 LiDAR alpha 只约束离散射线，无法保证射线之间的连续墙面累计 alpha。V88b 因此撤销“最佳成品”表述，只保留为几何较稳的修复起点。

修改内容：Trainer 新增默认关闭的 `mesh_alpha_weight/mesh_alpha_target`。它只在已签名、已排除 source_type=4、且通过逐视图 P95 门禁的 Mesh 像素上，对真实渲染累计 alpha 的不足部分施加置信度加权平方损失；不把 RGB 全有效区或天空当作前景，不改变 Mesh depth/normal 的独立消费语义。下一步只允许冻结 means/scales/quats、关闭 lifecycle 和 opacity sparsity 的短程 opacity/SH0 修复，并以用户指出的墙面 ROI、黑白背景 alpha、深度和大高斯联合验收；不得用全局硬 alpha 或放大高斯掩洞。
