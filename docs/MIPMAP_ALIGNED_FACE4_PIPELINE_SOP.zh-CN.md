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
16. `tile_gaussian_training`：各 Tile 按 High/type-2 执行 `20×Tile视图数` 步、训练内部 SH degree 1/表面产品 DC-only、RGB `0.6 mean-L1 + 0.4(1-SSIM)`、DA2 `0.5`、LiDAR/mesh depth `0.5→0.25`、mesh normal `0.05`、后期自洽 normal `0.01` 和 opacity-mean `0.01`。snow 分支固定使用 gradient split/clone/cull/reset，禁止启用 MCMC relocation 或 redundancy cull 冒充竞品。
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
- 竞品 High/type-2 已直接确认训练参数 degree 1，而表面终态为 DC-only、天空为 degree 1；不能把“最终表面 PLY 无 SH rest”解释成训练全程只用了 SH0。
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
