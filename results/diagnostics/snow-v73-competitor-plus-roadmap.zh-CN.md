# Snow V73 竞品等价主线与超越路线图

日期：2026-08-30

状态：`TILE0_COMPETITOR_BOUNDARY502_RUNNING`

## 决策

下一次全量训练不再使用 V72 的五块固定拓扑，也不把 MCMC 作为主生命周期。主线为：

`四块 X/Y/Y → K7/K30 LiDAR surfel → Face4 → 光度解耦 → mesh/DA2 分阶段几何 → projected-gradient Split/Clone/Cull → 后段停止增长精修 → 独立 sky → retain-halo 合并与接缝验收`

CloudStudio 超越竞品的增强不在 step 0 全部开启。它们按门禁依次加入：screen/pixel-aware detail split、可信 mesh coverage birth、贡献度 prune。任一增强使 alpha、held-out LiDAR 或最差视图退化时自动回到竞品等价 checkpoint。

## 已量化差距

| 项目 | V72 | MipMap | V73目标 |
|---|---:|---:|---:|
| 空间 Tile | 5 | 4 | 4 |
| surface Gaussian | 7,271,982 | 6,018,902 | 由生命周期决定，不预设终值 |
| 最短轴 P50 | 3.193 mm | 0.767 mm | 先进入 0.7–1.5 mm 区间 |
| 轴比 P50 | 2.832 | 12.544 | 8–16，且不靠针状异常作弊 |
| opacity P05 | 0.00387 | 0.05468 | 不低于 0.05 且无穿孔 |
| 跨 Tile 0.1 mm 重合近邻 | 90.255% | 0.0082% | 显著低于 5% |
| gradient 高/低纹理密度比 | 未形成 | 约 3.74× | 先超过 2×，再逼近/超过竞品 |

V72 的 374 步 strict-fixed 只训练 SH0 和相机 gain，means/scales/quats/opacities 全冻结；它不可能形成纹理驱动容量重分配。继续增加这类步数没有意义。

## 论文证据如何使用

### 第一层：必须复刻的竞品闭环

- 普通 projected-XY gradient、opacity eligibility、Split/Clone/Cull、周期 opacity reset。
- backward 后、Adam 前执行生命周期。
- K7 邻距尺度与 K30 normal 初始化。
- 0.6 L1 + 0.4 DSSIM、每相机 gain+bias、BilateralGrid。
- mesh depth/normal 与成功 RANSAC 标定的 DA2，按 `[5V,10V,5V]` 时序启用。
- surface SH0 与 sky 分层。B 机最新交叉证据指向竞品交付的 100k sky 为 SH0；SH1 只保留为 CloudStudio 后续方向性 A/B，不作为竞品等价前提。

### 第二层：用于超过竞品、但必须后置的增强

- AbsGS 证明普通向量梯度会发生 gradient collision；只在普通梯度臂已健康后，才标定 AbsGrad 阈值，不能把 `1.5e-4` 原样套给 AbsGrad。
- Pixel-GS 证明可见视图简单平均会稀释大 footprint Gaussian 的增长信号。第一版利用现有 raster radius 做 pixel-aware 权重/分裂诊断，距离缩放抑制近相机 floaters。
- ResGS 用 residual split 保留 coarse coverage 并添加缩小残差 Gaussian，适合雪面/墙面“不能删穿但需要更细”的场景；它应替代 brutal split 的增强臂，而不是改变竞品等价臂。
- SAD-GS 比较 Gaussian 投影轴与局部结构频率，并要求多视图一致后做按轴细分。它适合作为后期高频 ROI 增强，不适合在本轮首次生命周期前全局启用。

## V73 分阶段路线

### Gate 0：输入与显存（已通过）

- 四块树：`X→Y/Y`。
- Tile 点数：2,851,911 / 1,477,056 / 1,850,945 / 1,076,290。
- Tile 视图：624 / 470 / 607 / 505。
- K7/K30 几何已生成。
- 最大 Tile_0 factor=1 一步 smoke 峰值 1,943,262,720 bytes。
- 首次 smoke 发现 Tile_0 的孤立 LiDAR 邻域令原始 K7 最大切向尺度达到 7.537 m，第一步触发 281 次 0.2 m world clamp。加入 `surface_initialization.maximum_scale_m=0.2` 后重新运行，初始/最终最大轴均为 0.200000003 m，world clamp 为 0，峰值显存不变。正式 V73 必须保留初始化上限，不能让巨型初始化存活到 step 500 再由 lifecycle 清理。

### Gate 1：单 Tile 竞品等价 step 502

选择覆盖墙、雪堆和细节边界的 Tile。只运行到首次真实 lifecycle 后：

1. `absgrad=false`，`grow_grad2d=1.5e-4`；
2. `opacity>0.15` 才出生；
3. `>0.2 m` Split，否则 Clone；
4. early/late opacity Cull 0.10/0.05，世界/屏幕异常 Cull；
5. reset 每 300 step，cap 0.2；
6. `backward → lifecycle → Adam`；
7. 不启用 LiDAR 硬出生守卫、局部 alpha 保护、各向异性惩罚、MCMC 或自定义 screen-detail 分支。

所有 epoch 相关步数必须由该 Tile 的实际视图数 `V` 动态生成，禁止继续复用旧 Tile_1 的硬编码常数：`max_steps=20V`、`prune_switch_step=10V`、`mcmc_refine_stop_iter=refine_scale2d_stop_iter=15V`。首次生命周期边界仍为 step 502，因为出生从 step 500 开始、每 100 step 执行一次；但最大容量改为由初始 Gaussian 数签名生成，而不是固定 220 万。当前 Tile_0 有 624 个视图、2,851,911 个初始 Gaussian，因此其总步数为 12,480、增长停止为 9,360、prune 切换为 6,240，cap 必须高于初始数量并通过 step-502 实测增长率再冻结。

晋级条件：增长候选局部化；高纹理五分位净增长高于低纹理五分位；无全场爆发；alpha/held-out LiDAR/最差视图不退化；无大球、floaters、OOM、NaN。

### Gate 2：单 Tile 5V

完整启用竞品光度与几何环境：gain+bias、BilateralGrid、成功 RANSAC 的 DA2、严格 source_type 2/3 mesh depth、Gaussian normal 对 mesh normal。检查：

- alpha P05 与黑/白背景挑战；
- PSNR/SSIM/LPIPS 和 edge local-SSIM；
- held-out LiDAR P50/P95、point-to-plane P95；
- shortest-axis、轴比、opacity 分布；
- `birth/split/clone/cull/count` 时序；
- detail/smooth density ratio `R_rho`。

### Gate 3：CloudStudio 增强臂

只从 Gate 2 最佳 checkpoint 分叉：

1. Pixel-aware detail：按 raster footprint 加权普通梯度，候选必须跨视图稳定；
2. residual split：保留 coarse parent coverage，新增缩小 residual child，并守恒初始 alpha；
3. coverage birth：仅在 `render alpha低 + source_type 2/3 mesh存在 + 跨视图一致` 的表面位置出生；
4. contribution prune：累计真实可见贡献与 residual reduction，在切平面 cell 保持 minimum occupancy，禁止删成洞。

四项不得同时首次启用。按 `pixel-aware → residual split → coverage birth → contribution prune` 顺序逐项验收。

### Gate 4：相邻双 Tile 接缝

训练一对相邻块并 retain halo。要求：

- 0.1 mm 几乎重合双层从 V72 的 90.255% 大幅下降；
- overlap/non-overlap alpha 一致；
- 真实渲染无亮度线、透明线和密度断层；
- 相邻块共享相机的 gain+bias/Bilateral 统计没有系统偏移。

竞品 Merge 本身是 append-only，不存在已证实的后融合。若完整等价训练后仍有接缝，再增加自研 half-open owner 或边界距离 blend，明确标为 CloudStudio 发布增强。

### Gate 5：四块全量

四块严格串行；每块按自身 V 运行 20V，前 5V 建立覆盖，中 10V 完整生命周期，后 5V 停止出生并精修。选择验证集最佳 checkpoint，不默认最后一步。surface 与 sky 分开导出，最后 retain halo 合并并执行统一质量审计。

## V73a step 502 实测与 V73b 校准门（2026-08-30）

V73a 在 500 step 预热后首次触发已恢复的 `Split → Clone → Cull → Adam` 生命周期：初始 `2,851,911`，Clone `57,787`、Split 父点 `0`、Cull `1,102,874`，终态 `1,806,824`。Cull 中 `1,102,236` 个候选来自 opacity `<0.10`，世界尺度异常为 `0`、屏幕异常为 `800`；低 opacity 候选的观测次数 P50/P95 为 `124/147`，因此不是“从未看见”导致，而是当前光度/透明度动力学在首次边界产生了过强退出。与此同时，初始化尺度上限 `0.2 m` 与仅 `>0.2 m` 的 vendor world split 组合，使 Split 通道在本轮为零。

V73b 保留竞品已确认的 `500 step` 预热和 `100 step` 周期，但明确作为 CloudStudio 校准增强臂：普通梯度阈值改为 `1.0e-4`，开启 `>2 cm` 且真实屏幕半径 `>0.0035` 的 detail Split，首轮 Cull 使用统一 `0.05`，并跑到 step 602 观察两次生命周期。该版本不得描述为逐 bit 竞品复刻；其门禁目标是增加真实细节出生，同时避免 V73a 首轮 36.6% 的净删点风险。

### 网页监控 PSNR 接入（2026-08-30）

问题现象：训练网页此前只有当前单视图总 loss，不能直接观察 RGB 重建质量，而且不同视图间的 loss 不具备严格可比性。

修改文件：`cloudstudio_3dgs/training/losses.py`、`cloudstudio_3dgs/training/trainer.py`、`tools/training_monitor.html`、`tests/test_training.py`。

修改内容：训练端按有效 RGB mask 计算真实 MSE PSNR，渲染结果先裁剪到 `[0,1]`，并将 `rgb_psnr_db` 写入持久化进度；网页卡片显示“当前视图 PSNR”，多轮曲线选择器可叠加该指标。该值仍是当前采样视图指标，不替代固定验证集均值。

验证方式：新增单元测试覆盖无效像素排除和预测值裁剪，并执行 Python 语法检查、目标测试与中文乱码扫描。

当前状态：代码已接入；修改前已启动的 V73b 进程不会热加载新指标，下一次续跑或新训练才会写出 PSNR。

### V73b step 602 复盘与 V73c 控制变量续跑（2026-08-30）

问题现象：V73b 在 step 602 的 PLY 与 24 视图验证仍显示明显低 alpha，黑背景 PSNR 10.568 dB、白背景 PSNR 9.242 dB，平均 alpha 0.481、LiDAR 像素 alpha 0.780。此时若立即增加 alpha loss 或修改 Cull，会把训练时序问题误判为参数问题。

直接原因：checkpoint telemetry 证明第二次生命周期发生在 step 600，并同时执行 `opacity_reset=true`；V73b 的受控停止点 602 仅比 reset 晚两个 completed step，正处于透明度恢复最低谷。两轮累计 Clone 229,205、Split parent 60,708、Split child 121,416、Cull 535,260；首轮净减少 205,593，第二轮仅净减少 39,754，说明 V73b 已显著缓解 V73a 的一次性大规模误删，但还没有观察 reset 后的恢复过程。

结构审计：总计 2,606,564 个 Gaussian，其中 opacity `<0.1` 为 34.40%；可见点最长轴 P50/P95 为 11.35/28.18 mm、短轴 P50 为 5.07 mm、轴比 P50 为 2.18。可见点到 LiDAR 最近邻 P95 为 1.74 mm，超过 30 cm 的 floaters 仅 228 个，没有大于 0.2 m 的 Gaussian。几何没有失控，但薄盘和纹理驱动重分配仍未成熟；纹理梯度与体素净变化仅 `ρ=0.109`，且样本仅 81 个，不能晋级全量。

修改文件：新增 `tools/build_v73c_post_reset_settle.py`，生成独立签名续跑配置和 gate。

修改内容：V73c 不改变任何 loss、学习率或生命周期阈值，从 V73b step 602 精确 resume；在 completed step 699 保存下一次生命周期前的 checkpoint，在 step 702 保存 step-700 生命周期后的 checkpoint。这样先比较 opacity 自然恢复，再单独量化第三次 Split/Clone/Cull 的增益和损伤。

验证方式：分别对 699/702 执行相同 24 视图黑白背景 PSNR/alpha、Gaussian health、尺度、纹理密度和 PLY 视觉检查，并核对网页记录的真实 masked RGB PSNR。

当前状态：V73c 配置生成器已实现，等待签名检查和 100-step 续跑。

## 全量自动停止条件

- Gaussian 数达到每 Tile 签名 cap；
- point-to-plane P95 超过 2 cm 或 held-out depth P95 超过 10 cm；
- 黑背景挑战出现新增透明洞；
- 大于 0.2 m Gaussian 非零且不能在一个维护周期内清除；
- 最差视图结构指标连续两个 epoch 下降；
- 显存超过 7.5 GiB；
- detail/smooth density ratio 下降且 smooth coverage 同时下降。

## V74a 观察窗口 Cull 修正（2026-08-30）

问题现象：V73c step 700 的普通投影梯度仍产生 39,699 个 Clone 和 47,460 个 Split 子点，但同一事件按 opacity `<0.05` 一次删除 477,672 个 Gaussian，净减少 414,243；纹理—密度相关性明显提高，但黑背景 PSNR、alpha P05 和 LiDAR 区域 alpha 同时下降。说明容量重分配方向正确，死亡控制器动作过快。

修改文件：新增 `tools/build_v74a_observation_cull_boundary.py`。

修改内容：V74a 从与 V73a 相同的确定性 Tile_0 输入重新开始，保持普通梯度 `1.5e-4`、生长 opacity `0.15`、world Split `0.2 m`、`revised_opacity=false` 和 vendor pre-optimizer 顺序不变。仅将 opacity Cull 改为观察窗口判定：至少累计 64 次有效观察、连续 2 个生命周期低于阈值、reset 后 200 step 宽限、单事件最多删除当前点数的 5%；达到签名绝对容量上限时才启用已恢复的宽松维护分支。V74a 不使用局部体素 alpha 代理、不使用 MCMC、不启用自定义 2 cm detail Split。

验证方式：配置与 gate 必须通过签名和 `TrainerConfig.validate()`；训练只授权到 step 602，分别审计 step 500/600 的候选、实际 Cull、净点数变化、PSNR、alpha、held-out LiDAR 和 PLY，不通过则不延长。

当前状态：实现完成，等待窄范围测试、配置生成和 Tile_0 step-602 GPU 边界验证。

V74a 实测：step 500 Clone 57,788、Split 0、仅因屏幕尺度删除 800；opacity 候选 1,102,275 个但因连续事件不足全部保护。step 600 Clone 39,173、Split 0，持续低 opacity 候选按 5% 上限删除 147,403，另删除 398 个屏幕异常点；终态 2,800,271，相对初始化仅净减 1.81%。固定 24 视图黑背景 PSNR 10.803 dB、alpha 0.508、LiDAR alpha 0.800、depth MAE 9.18 cm，均优于 V73b 同步数的 10.568/0.481/0.780/7.87 cm 中前三项，但深度略退化。可见点到 LiDAR P95 为 1.25 mm、无 30 cm floaters，证明安全 Cull 有效；另一方面 Split 仍为 0、可见短轴 P50 5.1 mm、轴比 P50 2.2，细节形态未解决。

后续单变量：新增 `tools/build_v74b_screen_detail_boundary.py`。V74b 与 V74a 使用相同输入、随机序列、loss、普通梯度阈值和 observation Cull，只启用 `lidar_surface_screen_detail`：world scale 超过 2 cm 且屏幕半径超过 0.0035 的高梯度 Gaussian 进入 Split，并使用签名的 revised opacity。仍只授权到 step 602；必须同时提升真实 Split 数、短轴/轴比和固定验证画质，且不得牺牲 alpha 或产生 floaters，才能晋级。

V74b 实测不晋级：两轮产生 26,306 个 Split parent，但固定24视图黑背景 PSNR 10.624 dB、alpha 0.484、LiDAR alpha 0.779，均低于 V74a；同时新增 232 个距离 LiDAR 超过30 cm的可见 Gaussian，中位短轴和轴比仍约5.1 mm/2.2。depth MAE 从9.18 cm改善到7.73 cm，但不足以抵消覆盖和视觉回退。detail Split 后续必须重新标定出生位置/尺度，不能直接进入长跑。

下一步新增 `tools/build_v74c_observation_cull_settle.py`：从已通过的 V74a step602 精确 resume，不修改任何参数；保存 step699 生命周期前 checkpoint，并在step702受控停止。用于判断安全 Cull 在reset恢复后能否达到或超过V73c的PSNR/alpha，同时避免step700再次大规模误删。

V74c 实测通过短边界：step700 仅 Clone 17,292，opacity Cull 因 reset 宽限为0，只删除230个屏幕异常点，终态2,817,333；没有重现V73c单轮净删414,243的问题。固定24视图黑背景由V74a step602的PSNR 10.803/alpha 0.508/LiDAR alpha 0.800提升到step702的11.548/0.621/0.888，白背景PSNR 10.754；并超过V73c step702的11.169/0.56/0.86。可见点到LiDAR P95为1.38 mm，超过30 cm的floaters为0。

剩余门禁未通过：step702 depth MAE为16.79 cm，高于V74a step602的9.18 cm；纹理—密度相关仅`ρ=0.147`，可见最短轴P50约5.1 mm、轴比P50约2.2，尚未形成竞品式薄盘和强纹理容量重分配。因此V74c是当前最佳安全短边界和后续基线，但不授权直接四块长训。下一步需重新标定detail Split的出生位置、尺度和opacity守恒，优先解决V74b出现的232个floaters与alpha回退，再做单Tile扩展。

## 当前已完成与阻塞

已完成：四块计划、四块 LiDAR PLY、四块 K7/K30 几何、四块稀疏 LiDAR sidecar、最大块全分辨率显存 smoke，以及 Tile_0 的生产级 mesh/DA2 输入和动态 step-502 门禁。

Tile_0 的 624 个 Face 已完成正确的 `raw mesh(source_type=1) → 跨视图分类(2/3/4) → 排除 type 4 → 每视图 P95≤10 cm` 顺序。跨视图层有 109,059,467 个原生 LiDAR anchor、324,397,484 个多视图支持像素和 643,786,409 个不可观测 type 4；最终严格 sidecar 启用 416 个视图、禁用 208 个视图，共保留 330,093,921 个 source 2/3 有效像素。DA2 仅有 388/624 个视图通过 mesh-native RANSAC 标定。旧的“直接对 source_type=1 做 2/3 admission”会生成全空监督，已被停止且未生成最终 Manifest。

V73 Tile_0 门禁使用 `V=624` 动态签名：总步数 12,480、增长停止 9,360、prune 切换 6,240、初始 2,851,911、临时边界 cap 4,278,000，只授权到 step 502。配置与训练门均已通过 `TrainerConfig.validate()`。普通 Python 入口首先暴露当前 `csrc.pyd` 缺少 EWA 算子；该失败发生在首帧前且没有 checkpoint。随后通过已验证含 `projection_ewa_3dgs_fused_fwd` 与 `rasterize_to_pixels_3dgs_fwd` 的 `csrc.3dgs-only.backup.pyd` 重启，现已进入真实 EWA 训练；step 60 时仍为 2,851,911 个 Gaussian，峰值显存 3,826,780,672 bytes，DA2 与 mesh-normal 在有效视图上均产生非零 loss。

阻塞：正在运行并等待验收 Tile_0 step 502；通过后再生成其余三块严格 mesh/DA2、单 Tile 5V、相邻双 Tile seam 自动门。完成这些之前不启动四块长训。

## 证据

- `results/diagnostics/snow-v72-vs-mipmap-seam-density-review.zh-CN.md`
- `results/diagnostics/snow-20260827-mipmap-gaussian-initialization-training-static-audit.zh-CN.md`
- `results/diagnostics/mipmap_image_voxel_regression_audit.summary.json`
- `outputs/snow-20260224-full-20260825/adaptive_tile_plan_lidar_visibility_4tile_v73/adaptive_tile_plan.json`
- `outputs/snow-20260224-full-20260825/tile_training_inputs_lidar_4tile_v73/tile_inputs_manifest.json`
- `outputs/snow-20260224-full-20260825/tile_initialization_geometry_k7_k30_4tile_v73/tile_geometry_manifest.json`
- `outputs/snow-20260224-full-20260825/v73_four_tile_competitor_equivalent/smoke_tile0_factor1_initcap/training/run_manifest.json`
- `outputs/snow-20260224-full-20260825/v73_four_tile_mesh/Tile_0/strict_source23_p95_010/mesh_geometry_manifest.json`
- `outputs/snow-20260224-full-20260825/v73_four_tile_mesh/Tile_0/da2_aligned`
- `outputs/snow-20260224-full-20260825/v73_four_tile_competitor_equivalent/tile0_boundary502/trainer.config.json`

外部一手资料：AbsGS arXiv 2404.10484；Pixel-GS ECCV 2024；ResGS ICCV 2025；SAD-GS SIGGRAPH 2026 项目论文。

## 2026-08-30 V75a：修复表面出生父子错配并恢复受约束细节 Split

问题现象：V74c 已恢复覆盖，但纹理—密度相关性约为 `0.147`，可见高斯最短轴 P50 约 `5.06 mm`、轴比 P50 约 `2.20`，细节仍不够锐利。V74b 开启 screen-detail Split 后出现 `232` 个距离 LiDAR 超过 `30 cm` 的可见漂浮点，因此不能直接晋级。

根因：竞品顺序为 `Split → Clone`。该顺序下新生张量尾部实际排列为“两个 Split 子点在前、Clone 副本在后”，但原实现给 LiDAR 切平面 proposal 的父点排列却是“Clone 父点在前、Split 父点在后”。启用表面出生守卫时，这会把新生高斯配到错误父点的局部表面。

修改文件：`cloudstudio_3dgs/training/default_strategy_adapter.py`、`cloudstudio_3dgs/training/trainer.py`、`cloudstudio_3dgs/pipeline/mipmap_gate.py`、`tests/test_default_strategy_adapter.py`、`tests/test_mipmap_gate.py`、`tools/build_v75a_surface_detail_boundary.py`。

修改内容：严格按执行顺序构造新生尾部的父点映射；新增混合 Clone/Split 回归测试；允许已签名的 `exact_relaxed_at_cap` 与 `observation_cull_v1` 组合启用 LiDAR tangent newborn guard。V75a 从 V74c step 702 恢复，只在 step 800 引入一次受约束细节事件：普通 projected-gradient 阈值仍为 `1.5e-4`，高梯度且世界尺度大于 `2 cm`、屏幕半径大于 `0.0035` 的高斯执行 Split；子点在 LiDAR 局部切平面内出生，最短轴初始化到表面法线方向，不支持的父点禁止出生。其余 loss、采样、学习率和 Cull 控制保持不变。

验证方式：`python -m pytest tests/test_default_strategy_adapter.py tests/test_mipmap_gate.py -q`，结果 `77 passed`；相关 Python 文件通过 `py_compile`；V75a 配置与训练门禁均完成签名并通过 `TrainerConfig.validate()` 与 `verify_gate()`。

当前状态：代码与短边界配置完成，下一步运行 V75a 到 step 802，联合检查 Split/Clone 数量、LiDAR 支持率、漂浮点、最短轴、纹理—密度相关性、alpha、PSNR 和 PLY 实际视觉；在此门禁通过前不启动四块长训。

## 2026-08-30 V76a：激进细节 Split 单事件标定

问题现象：V75b step 1002 的纹理—密度相关性提升到约 `0.212`，但固定 24 视图黑背景 PSNR 约 `11.245 dB`、alpha 约 `0.590`，可见高斯轴比 P50 仍只有约 `2.3`。训练仅覆盖约 `1002/624=1.61` 个视图 epoch，步数不足以作为终态；同时 step 900 的 reset 与 143,690 点 Cull 后画质回退，不能直接沿原参数盲目长训。

修改文件：`cloudstudio_3dgs/training/default_strategy_adapter.py`、`cloudstudio_3dgs/training/trainer.py`、`cloudstudio_3dgs/pipeline/mipmap_gate.py`、`tests/test_default_strategy_adapter.py`、`tools/build_v76a_aggressive_detail_boundary.py`。

修改内容：新增签名的 `lidar_surface_screen_detail_aggressive` 细节策略，将普通 projected-gradient 阈值降到 `1.0e-4`；高梯度且世界尺度超过 `1 cm`、屏幕半径超过 `0.0025` 的高斯进入 Split。新生点继续经过 LiDAR 表面支持门禁，但允许更大的切向移动；新生最短轴厚度系数降到 `0.25`，最小厚度 `0.5 mm`；增加只优化最短轴的弱压平项，不压缩两个切向轴。V76a 从 V75b step 1002 恢复，只授权一个 step-1100 生命周期事件并在 step 1102 停止。

验证方式：相关适配器和门禁测试结果 `78 passed`。step 1100 实际得到 52,308 个视觉候选、47,799 个 LiDAR 支持父点、4,788 个 Clone、43,011 个 Split parent、86,022 个 Split child；表面出生 90,810 个，仅删除 653 个屏幕异常点，终态 2,774,583。固定 24 视图黑/白背景 PSNR 为 `11.449/10.716 dB`，alpha `0.603`，LiDAR alpha `0.900`，depth MAE `12.89 cm`。纹理梯度密度相关性提升到 `ρ=0.2686`，亮度 MAD 相关性为 `ρ=0.2896`。可见短轴 P50 约 `4.7 mm`、轴比 P50 `2.6`；可见点到 LiDAR P95 `5.73 mm`，超过 30 cm 的可见点 44 个。

当前状态：V76a 证明“更低梯度阈值 + 更积极 Split + 表面安全出生”能显著增强纹理容量重分配，并小幅恢复 PSNR/alpha；但深度略退、短轴仍明显厚于竞品约 `0.7–0.8 mm`，且存在少量漂浮点，因此尚不授权四块长训。

## 2026-08-30 V76b：隔离 step-1200 opacity reset

问题现象：V75b 的 step-900 事件同时发生强 Cull 和 opacity reset，无法判断画质下降主要来自哪一项；若原样继续，step 1200 会再次把 opacity 统一压到不超过 `0.2`，可能破坏刚形成的细节覆盖。

修改文件：新增 `tools/build_v76b_aggressive_detail_noreset_boundary.py`。

修改内容：从 V76a step 1102 精确恢复，保留所有 loss、学习率、生长、Split、Cull 和 LiDAR 表面出生参数，仅将 reset profile 与周期从 `exact_every300/300` 改为已签名兼容的 `deferred_every3000_compatibility/3000`；只授权到 step 1202，以隔离观察 step-1200 的生长/Cull，而不叠加 opacity reset。

验证方式：生成配置必须通过 `TrainerConfig.validate()` 和 gate 签名验证；step 1202 后重复固定 24 视图黑白背景、alpha、LiDAR alpha、depth、尺度、纹理密度和漂浮点审计。

当前状态：配置和签名门禁已完成，正在执行 step-1202 受控边界；通过前不延长为多 epoch 长跑。

V76b 实测不晋级：step 1200 在关闭 opacity reset 后仍新增 100,620 个 Gaussian、Cull 142,766 个，终态 2,687,539。固定 24 视图黑/白 PSNR `11.265/10.409 dB`，alpha `0.5863`，LiDAR alpha `0.8912`，均低于 V76a step 1102；纹理相关升到 `ρ=0.2930`、短轴 P50 降到 `4.4 mm`、轴比升到 `2.8`。说明锐利化有效，但 5% observation Cull 明显快于有效容量出生。

V76c 新增签名的 `observation_cull_v2_conservative`，将单事件 Cull 上限降到 2%；执行层、训练配置层和 gate 层均已支持，回归测试为 `79 passed`。同一 step-1200 事件新增 100,618、Cull 57,859，终态 2,772,446，避免了点数塌缩；但 PSNR/alpha 与 V76b 几乎相同，证明被多保留的主要是低贡献点，step-1200 即时画质下降并非 Cull 数量单独造成。

V77a 在 V76c 基础上增加签名的弱 LiDAR alpha 约束 `weight=0.02,target=0.95`。黑背景 PSNR、alpha、LiDAR alpha 小幅改善为 `11.284 dB/0.5887/0.8941`，仍不足以恢复 V76a。V77b 将 opacity sparsity 的作用域从全部 Gaussian 改为 `visible_current_view`，当前帧仅约 3.66% Gaussian 接受该正则；结果为 `11.274 dB/0.5889/0.8942`，说明未观察点被全局正则压低并非唯一主因。

关键隔离结果：V77b 同次运行的 step 1199 位于生命周期事件前，固定 24 视图黑/白 PSNR 已达到 `11.804/11.107 dB`，alpha `0.6350`、LiDAR alpha `0.9085`；step 1200 执行约 44,591 个 Split 后，仅优化 2 步就评估，指标骤降。由此确认此前门禁把“新生子点尚未收敛”误判成持续退化。

V78a 从 V77b step 1202 精确恢复，不改任何 loss、学习率或生命周期参数，在下一次 step-1300 事件前停止于 step 1299，让新生点完成 97 步 settle。结果黑/白 PSNR `11.815/11.284 dB`，alpha `0.6428`、LiDAR alpha `0.9215`，短轴 P50 `4.2 mm`、轴比 P50 `2.9`，纹理相关 `ρ=0.2893`，成为当前最佳综合色彩/覆盖节点。代价是固定抽样 depth MAE 升到 `18.55 cm`，后续必须控制法向漂移并引入跨视图一致的无 LiDAR 补洞准入，不能全局取消几何守卫。

当前训练节奏修正为：`生命周期生长/Split → 约 100 step settle → 事件前保存与联合评估 → 再决定下一事件`。不再把事件后 2 步 checkpoint 当作最终画质节点，也不再以“距最近 LiDAR 超过 30 cm”单项否决；无 LiDAR 覆盖区将采用跨视图一致性、深度不冲突和持续残差联合准入。

V78b 从 V78a step 1299 继续执行 step-1300 生命周期，并在下一次 step-1400 事件前停止于 step 1399。事件实际 Clone 14,083、Split parent 38,059、Cull 56,935，opacity reset 关闭，终态 2,767,281；点数基本稳定，说明容量正在从低贡献区转移到高梯度细节区。完整 settle 后黑/白 PSNR 进一步提升到 `11.870/11.365 dB`，alpha `0.6453`、LiDAR alpha `0.9214`，纹理相关 `ρ=0.3063`；短轴 P50 降到 `4.0 mm`、轴比 P50 升到 `3.1`。depth MAE 为 `18.43 cm`，相对 V78a 没有继续恶化。V78b 成为新的最佳恢复点，证明训练步数不足且“事件后完整 settle”是必要条件。

## 2026-08-30 V79a/V80a：恢复竞品半径加权梯度主链

问题现象：V78b 已有 2,767,281 个 Gaussian，但砖石、屋檐和墙板等高频结构仍偏糊。旧控制器直接使用 gsplat 的二维梯度计数平均，无法与逆向恢复出的 MipMap 阈值同量纲比较。

修改文件：`cloudstudio_3dgs/training/default_strategy_adapter.py`、`tests/test_default_strategy_adapter.py`、`tools/build_v79a_mipmap_gradient_probe.py`、`tools/audit_mipmap_gradient_candidates.py`、`tools/visualize_mipmap_gradient_candidates.py`、`tools/build_v80a_mipmap_gradient_control.py`。

修改内容：新增可签名的 MipMap 等价梯度统计。每视图计算投影位置原始梯度 L2，乘 `0.5*max(1600,width,height)`，再以 raster radius 为权重累积；Tile core 外乘 `0.1`，最终以 `weighted_gradient_sum/radius_sum` 作为 100-step 窗口分数，同时保存最大归一化屏幕足迹。V79a 只旁路记录，不改变旧生长控制；V80a 才使用恢复分数和竞品 High 阈值 `1.5e-4` 控制一次 step-1500 生命周期，然后稳定到 step 1599。

验证方式：适配器测试 `64 passed`；新增单测证明 probe 的屏幕尺度、半径权重、Tile 衰减数值正确，并证明生产 profile 实际控制增长选择。V79a 图像空间投影显示新增候选明显集中在屋檐细构件、门窗轮廓、雪—地交界和碎雪表面，不是均匀撒点。粗 0.5 m voxel 候选相关审计不具判别力，不能替代图像空间证据。

当前状态：V80a step 1500 有 71,612 个梯度候选，其中 65,278 个父点通过表面支持；执行 Clone 22,495、Split parent 42,783、生成 108,061 个子点、Cull 57,827，终态 2,774,528。固定 24 视图黑/白 PSNR 为 `11.964/11.802 dB`，SSIM 为 `0.4815/0.5828`，alpha `0.6667`、LiDAR alpha `0.9356`，depth MAE `18.61 cm`。纹理梯度密度相关从 V78b 的 `ρ=0.3063` 提升到 `0.3503`，跨视图亮度 MAD 相关从 `0.3115` 提升到 `0.3798`。可见短轴 P50 为 `3.5 mm`、轴比 P50 `3.5`，仍有 `12.8%` 可见 Gaussian 宽于 5 px；因此方向通过，但砖石锐度仍需 PLY 视觉确认，尚不直接扩展四块长训。

V80b 从 V80a step 1599 原样继续一个生命周期和完整稳定窗口。step 1600 有 81,114 个恢复梯度候选、73,880 个表面支持父点；执行 Clone 25,610、Split parent 48,270、Cull 57,893，终态 2,790,515。固定 24 视图黑/白 PSNR 继续升至 `11.994/12.040 dB`，SSIM `0.4813/0.5849`，alpha `0.6737`、LiDAR alpha `0.9390`，depth MAE 改善至 `18.01 cm`。纹理梯度密度相关进一步升至 `ρ=0.3622`，亮度 MAD 相关 `ρ=0.3947`；可见短轴 P50 降至 `3.3 mm`、轴比 P50 升至 `3.8`。但宽于 5 px 的可见 Gaussian 从 12.8% 增至 13.6%，可见点到 LiDAR P95 从 8.58 mm 增至 9.25 mm，超过 30 cm 的可见点从 100 增至 120。V80b 综合指标继续通过，但收益开始递减，下一步应优先视觉检查砖石锐度和过宽高斯，而不是立刻无限重复周期。
