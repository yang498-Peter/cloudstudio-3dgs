# Snow V72 与 MipMap 接缝及高斯密度复盘

日期：2026-08-30

状态：`V72_NOT_PROMOTED_RETRAINING_BLOCKED_PENDING_EQUIVALENT_TILE_GATE`

## 问题现象

V72 完整 PLY 在 SuperSplat 的中心点显示中存在明显分区线和不自然密度变化；表面高斯仍接近 LiDAR 均匀采样，未形成竞品在纹理边缘密集、同色区域稀疏且连续的分布。

## 直接证据

### 1. V72 没有执行容量重分配

V72 正式臂只运行 374 步，配置为 `strict_fixed`。训练期间高斯数始终为 7,371,982；surface 为 7,271,982，sky 为 100,000。means、scales、quats、opacities 学习率均为 0，只训练 SH0 颜色和 per-image gain。竞品 lifecycle 从 step 500 开始、每 100 步执行一次，因此 V72 在时间和参数两方面都不可能产生 Clone/Split/Cull 或纹理驱动的位置重排。

V72 同时关闭 BilateralGrid、DA2、mesh depth 和 mesh normal；照片损失为 0.8 L1 + 0.2 SSIM。竞品 High/type-2 使用 0.6 L1 + 0.4 DSSIM、DA2/mesh depth/mesh normal、训练内 per-camera photometric nuisance model 和 BilateralGrid。

### 2. 我们的 halo 是几乎完全重合但参数冲突的双层

V71/V72 surface 按五个 Tile 的完整 halo 直接拼接，未删除任何重叠点。对 V72 相邻 Tile 的 2 cm 近邻做跨块审计：

| 指标 | V72 | MipMap snow |
|---|---:|---:|
| 跨块近邻样本数 | 250,966 | 146,055 |
| 最近距离 P50 | 0.027 mm | 4.968 mm |
| 最近距离 P95 | 0.215 mm | 15.902 mm |
| 距离不超过 0.1 mm | 90.255% | 0.0082% |
| 距离不超过 1 mm | 95.315% | 3.906% |

V72 各相邻 Tile 对中，近邻颜色绝对差 P50 为 0.028–0.036、P95 为 0.088–0.165；opacity 绝对差 P50 为 0.040–0.109、P95 为 0.723–0.912。也就是说，同一 LiDAR 表面位置保留了两套几乎重合、但颜色和透明度冲突的 Gaussian。视角变化或排序变化会改变两层的主导关系，形成可见接缝和不稳定外观。

MipMap 最终 PLY 同样保留四块 halo，并且 Merge 是 append-only；但它的跨块点已经经过每块独立的 xyz/scale/rotation/opacity 优化和生命周期重排，不是几乎逐点重合的 LiDAR 双层。不能把差异误归因于“竞品做了 merge 后融合”。

### 3. 我们比竞品多一块，而且弱 Tile 共享视图更少

CloudStudio 当前计划受 6.5 GiB 保守预算驱动，产生五块；MipMap snow 为 `X -> 左右分别 Y` 的四块。我们的五块有效 Face4 视图数为 476/374/470/607/505；竞品四块为 656/644/607/595。额外切出 Tile_1 后，相邻块共享照片监督减少，并新增一条 halo 边界。

### 4. 终态形状和 opacity 分布不等价

| 指标 | V71/V72 surface | MipMap surface |
|---|---:|---:|
| Gaussian 数 | 7,271,982 | 6,018,902 |
| 最长轴 P50 | 8.513 mm | 10.465 mm |
| 最短轴 P50 | 3.193 mm | 0.767 mm |
| 轴比 P50 | 2.832 | 12.544 |
| opacity P05 | 0.00387 | 0.05468 |
| opacity P50 | 0.13994 | 0.11485 |
| opacity 最小值 | 0.00121 | 0.05000 |

V72 surface 的几何和 opacity 继承 V71；V72 冻结了这些参数。我们的中心更多、更厚、更圆，并保留约 88 万个低 opacity/dead Gaussian；竞品使用少量更大但更薄的表面盘覆盖低纹理平面，并把最终 opacity 下限收敛到约 0.05。

### 5. 竞品纹理密度关系是结果级事实

竞品 4,235 个稳定 0.5 m voxel 中，`N_GS/N_LAS` 与图像梯度的 Spearman 相关为 0.364，Tile 内秩相关为 0.382；与纹理熵相关为 0.369/0.386。图像梯度从最低到最高五分位时，`N_GS/N_LAS` 中位数从 0.173 增至 0.646，约 3.74 倍。

这不是显式 Canny/纹理撒点，而是以下闭环的结果：普通 projected-XY gradient 决定高梯度点 Clone/Split；opacity mean 正则、周期 opacity cap 和 Cull 让低贡献点退出；mesh depth/normal 保持表面连续并允许剩余 Gaussian 变成薄盘；BilateralGrid 和相机颜色模型降低曝光/阴影造成的伪梯度。

## 为什么此前自适应实验没有复现竞品

此前不存在一条同时满足完整竞品条件的真实运行：

- V33 使用 AbsGrad、LiDAR 硬出生守卫、anisotropy penalty、PPISP，并且 lifecycle 顺序不是已恢复的 pre-optimizer vendor 顺序。
- V42–V48 虽修正为 pre-optimizer 顺序，但又关闭或弱化 means/opacity/Cull，修改梯度阈值，并继续使用 PPISP；没有同时启用竞品 DA2、mesh depth/normal 和 BilateralGrid。
- V71/V72 为修复透明问题退回 strict-fixed；它们只能改变形状、颜色或 opacity，不能改变中心密度。

因此不能用这些实验否定竞品经典 lifecycle；它们验证的是若干不完整或自定义组合。

## 下一步唯一允许的优化路线

### Gate A：输入与四块计划

1. 将当前五块恢复为与竞品同构的四块 `X/Y/Y`，优先合并当前 Tile_0 与 Tile_1；先做显存 smoke，不启动长训。
2. 每块使用带 0.2% 空间 halo 的 LAS、至少与竞品相当的共享 Face4 视图集合和 128 px 图像 halo。
3. 保留 append-only halo 作为竞品等价臂；同时新增只读 seam audit，不先做自定义融合。

### Gate B：单 Tile 竞品等价 502-step 边界

必须同时满足：

- Face4、SH0 surface、独立 SH1 sky，不把 sky 混入 surface lifecycle；
- `[5V,10V,5V]` 时序、Fisher-Yates 每 epoch 无放回；
- 0.6 L1 + 0.4 DSSIM；
- 通过门禁的 DA2 affine depth、mesh depth、mesh normal；
- per-camera `exp(a)*rgb+b` 与 BilateralGrid；
- `backward -> Split/Clone/Cull -> Adam step -> opacity cap`；
- `absgrad=false`、gradient threshold 1.5e-4、opacity eligibility 0.15；
- 关闭 LiDAR 硬出生守卫、局部 coverage Cull、自定义 alpha loss、anisotropy penalty 和 screen-detail 扩展；
- 使用竞品六组 Adam 学习率，让 means/scales/quats/opacities 真正可训练。

先停在 step 502，只检查第一次真实 lifecycle，不允许直接跑满。

### Gate C：密度和几何联合晋级

首次事件后必须同时检查：

- 高纹理五分位净增长高于低纹理五分位，相关方向为正；
- clone/split/cull 数及候选分布没有全场爆发；
- alpha、最差视图 PSNR/SSIM、held-out LiDAR 和 point-to-plane 不退化；
- opacity 低值尾部开始退出，但不出现墙面/雪面穿孔；
- shortest-axis、轴比向竞品薄盘分布移动；
- 无 NaN、OOM、大球和浮点。

只有 Gate C 通过，才继续到 5V 边界。只有相邻两块都通过，才做双 Tile seam 检查；四块全量训练排在最后。

### Gate D：双 Tile 接缝门禁

合并两个相邻块后必须比较：

- 跨块 0.1 mm 几乎重合近邻比例，相对 V72 的 90.255% 必须大幅下降；
- halo 颜色/opacity 差、真实渲染接缝和视角稳定性；
- 重叠区 alpha 与非重叠区一致；
- 中心点密度不存在硬分区线。

若完整竞品等价训练后仍有接缝，才允许增加 CloudStudio 自有发布增强：half-open owner、按边界距离 blend 或跨 Tile 颜色约束。该增强必须标记为自研，不得称为竞品已证实步骤。

## 当前决定

V72 只保留为“全分辨率照片天空与固定 surface 的视觉诊断”，不晋级、不续训。下一次计算应是四块输入/视图重建和单 Tile 502-step 竞品等价边界，不是继续调 V72 opacity、背景色或颜色步数。

## 证据路径

- `outputs/snow-20260224-full-20260825/v72_bmachine_absorption/training_factor1_prebaked_sky374/run_manifest.json`
- `outputs/snow-20260224-full-20260825/full_area_safe_refine_v71/merged_halo/merge_report.json`
- `results/diagnostics/snow-v71-vs-mipmap-final-ply-structure.json`
- `results/diagnostics/mipmap_image_voxel_regression_audit.summary.json`
- `results/diagnostics/snow-20260827-mipmap-gaussian-initialization-training-static-audit.zh-CN.md`
- `results/diagnostics/snow-20260827-mipmap-tile-view-selection-audit.json`
- `outputs/snow-20260224-full-20260825/adaptive_tile_plan_lidar_visibility_v23q/adaptive_tile_plan.json`
