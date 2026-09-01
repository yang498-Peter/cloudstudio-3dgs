# Snow Tile_0 V87→V90 失败复盘与专家审查材料

更新时间：2026-09-01（Asia/Singapore）

## 1. 结论

V90 没有修复 V88b 的墙面空洞，只得到非常小的数值改善，实际 PLY 仍在墙板、门、屋檐和窗边出现黑洞、涂抹与不连续覆盖。因此：

- V87、V88b、V90 均不得晋级全场交付；
- 当前问题不是 PLY 导出过滤，因为 V90 以 `min_opacity=0` 导出全部 4,191,083 个高斯后仍有空洞；
- 当前问题也不是单纯“训练步数不足”。V87 已训练到 12,480 步，但用大尺度、厚高斯和低透明度雾化换取了覆盖；V88b 对超过 0.05 m 的高斯逐步执行每步 `×0.8` 三轴等比收缩，最终把最大尺度从约 0.2 m 压到 0.05 m 后，几何/深度指标改善，却暴露出真实拓扑覆盖不足；
- V90 只在既有位置上训练 opacity/color，位置、尺度、旋转和拓扑全部冻结，无法在缺口处生成新的小高斯，因此只能把已有覆盖稍微变实，不能重建缺失结构。

最核心的失败链是：

```text
V87 在视觉残差驱动下增长到容量上限
  → 部分大/厚/低透明度高斯以模糊方式遮盖缺口
  → V88b 在停止生长后，累计触发 738,962 次超过 5 cm 的逐步 ×0.8 收缩事件
  → 原来的模糊遮盖被移除，但没有同步生成替代的小高斯
  → 墙、门、屋檐出现真实覆盖缺口
  → V90 仅抬升既有 opacity，不能弥补空间拓扑缺失
```

## 2. 数据与输入

| 项目 | 实际值 |
| --- | --- |
| 空间块 | MipMap Tile_0 |
| 初始 LiDAR Gaussian | 2,851,911 |
| V87/V88b/V90 终态 Gaussian | 4,191,083 |
| 训练图像 | 签名 Face4，全分辨率，Tile 选择后 624 个虚拟视图 |
| 采样 | `with_replacement`，每步一张完整 Face4 视图；12,480 步约等于 20 个 624-view epoch |
| 颜色模型 | SH0 |
| 背景 | 黑色 `[0,0,0]` |
| RGB loss | `0.6 L1 + 0.4 local Gaussian SSIM` |
| 相机 | 已接受独立 AT；两组共享 KB4 原始鱼眼内参；Face4 为派生虚拟相机 |
| 初始化 | 完整 LAS 的 Tile_0 子集 + K7 尺度 + K30 法线 |
| mask | 鱼眼有效圆、FoV、人物、Face4 renderer mask、surface/sky exclusion |
| mesh | Tile_0 `strict_source23_p95_010`，只允许原生 LiDAR anchor 与跨视图一致支持 |
| mono depth | DA2 对齐 sidecar，仅标定成功视图有效 |

关键签名输入：

- Face4：`outputs/snow-20260224-full-20260825/face4_circle_person_v23g/train/face_manifest.json`
- renderer mask：`outputs/snow-20260224-full-20260825/renderer_masks_v23i/train_renderer_mask_manifest.json`
- Tile LiDAR：`outputs/snow-20260224-full-20260825/tile_training_inputs_lidar_4tile_v73/Tile_0/initialization_full_lidar.ply`
- 初始化几何：`outputs/snow-20260224-full-20260825/tile_initialization_geometry_k7_k30_4tile_v73/Tile_0/initialization_geometry_k7_k30.npz`
- mesh：`outputs/snow-20260224-full-20260825/v73_four_tile_mesh/Tile_0/strict_source23_p95_010/mesh_geometry_manifest.json`

## 3. V87 完整长跑参数

权威完整配置：

`outputs/snow-20260224-full-20260825/v87_ultrasharp_detail/tile0_full12480/trainer.config.json`

| 参数组 | 值 |
| --- | --- |
| max steps | 12,480 |
| warm start | V87 boundary step 1,202 |
| learning rate | means `1.6e-5`; scales `5e-3`; quats `1e-3`; opacity `5e-2`; color `2.5e-3` |
| RGB | L1 `0.6`; SSIM `0.4`; `local_gaussian` |
| sparse LiDAR range | `0` |
| DA2 depth | `0.5` |
| mesh depth | `0.5` |
| mesh normal | `0.05` |
| rendered-depth/normal consistency | `0.01` |
| LiDAR alpha | `0.02`, target `0.95` |
| LiDAR normal | align `0.02`; flatten `0.05`; point-to-plane `0.01`; Huber delta `0.02 m` |
| opacity sparsity | `0.0025`, visible-current-view scope |
| scale/anisotropy penalty | `0`; max reference ratio `8`; max anisotropy `256` |
| world scale limit | `0.2 m` |
| topology | `adaptive_growth`, `default_3dgs` |
| gradient source | total loss |
| gradient profile | `mipmap_radius_weighted_v1` |
| growth threshold | `1.5e-4`, ordinary signed gradient，`absgrad=false` |
| lifecycle | start `500`; every `100`; stop `9,360` |
| capacity | 4,278,000 |
| split | world `0.2 m`; detail scale `0.008 m`; screen radius `0.0025` |
| prune | opacity `0.05/0.05`; scale `0.2 m`; screen `0.15` |
| reset | every `3,000`; opacity cap `0.2` |
| cull | observation-aware; min observations `64`; consecutive events `2`; max event fraction `2%` |
| birth guard | LiDAR tangent; planarity `0.6`; support `0.1`; tangent factor `3`; tangent sigma `0.65`; normal offset `0.05`; thickness factor `0.25`; min thickness `0.5 mm`; reject unsupported |
| exposure | per-camera gain LR `0.005`, bias LR `0.001`, gain/bias regularization `0.01`, max log gain `ln2`, max bias `0.25` |
| BilateralGrid | enabled; LR `0.002`; grid `16×16×8`; TV `5.0`; warmup `1/30`; final LR multiplier `0.01` |

重要差异：V87 配置中的 `competitor_loss_schedule_enabled=false`，DA2/mesh 权重不是已调查竞品的 `[5V,10V,5V]` 分段时序，而是长程固定开启。这是与竞品仍未等价的一处明确事实。

## 4. V88b 尾段修复参数

权威完整配置：

`outputs/snow-20260224-full-20260825/v88b_balanced_tail_repair/tile0_step12680/trainer.config.json`

| 参数组 | 值 |
| --- | --- |
| resume | V87 step 12,480 |
| 追加步数 | 200（到 12,680） |
| means LR | `0` |
| scales/quats/opacity/color LR | `0.001 / 0.0002 / 0.02 / 0.001` |
| DA2/mesh depth/mesh normal/self-normal | `0.5 / 0.5 / 0.05 / 0.01` |
| LiDAR alpha | `0.1`, target `0.95` |
| opacity sparsity | `0` |
| progressive world-size threshold | `0.05 m`；越界时每 optimizer step 三轴统一 `×0.8` |
| 生长 | 已超过 stop step 9,360，实际不再出生/分裂 |
| 点数 | 固定为 4,191,083 |
| world clamp events | 738,962 |

V88b 的设计目的是压制 V87 的大尺度雾化和几何漂移。它不是逐轴硬 clamp，而是对越界椭球每步三轴统一 `×0.8`；但它在“停止增长之后”才大规模缩小覆盖半径，导致原有大高斯覆盖被拿掉，却没有细粒度高斯补位。这是目前最强的直接失败解释。

## 5. V90 mesh-alpha 修复参数

权威完整配置：

`outputs/snow-20260224-full-20260825/v90_mesh_alpha_wall_repair/tile0_step12880/trainer.config.json`

| 参数组 | 值 |
| --- | --- |
| resume | V88b step 12,680 |
| 追加步数 | 200（到 12,880） |
| means/scales/quats LR | `0 / 0 / 0` |
| opacity/color LR | `0.02 / 0.0005` |
| topology | 不触发；点数保持 4,191,083 |
| RGB | `0.6 L1 + 0.4 local SSIM` |
| DA2/mesh depth/mesh normal/self-normal | 全部 `0` |
| LiDAR alpha | weight `0.1`; target `0.95`; dilation radius `3 px` |
| mesh alpha | weight `0.2`; target `0.95` |
| mesh alpha admission | `RGB valid ∩ mesh valid ∩ finite positive mesh range/confidence` |
| opacity sparsity | `0` |
| background | black |
| duration | 166.56 s |
| peak VRAM | 3.71 GB |

20 条训练监控记录中，mesh alpha 支持比例均值 `27.01%`，最小 `0`，最大 `96.35%`；4/20 个采样视图完全没有 mesh alpha 支持。该 loss 只能在签名 mesh 覆盖处增实，无法约束无 mesh 的门窗、屋檐、反光或缺失 LiDAR 区域。

## 6. 量化结果

固定、确定性分层抽样的同一组 24 个 Tile Face4 视图：

| 版本 | 背景 | PSNR dB | SSIM | alpha mean | alpha P05 | alpha<0.95 | LiDAR alpha mean/P05 | depth MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V87 | white | 14.4481 | 0.6511 | 0.7303 | 0.1644 | 33.39% | 0.9798 / 0.9084 | 24.57 cm |
| V87 | black | 12.7376 | 0.5312 | 同左 | 同左 | 同左 | 同左 | 同左 |
| V88b | white | 13.0352 | 0.6126 | 0.6435 | 0.1601 | 41.31% | 0.9594 / 0.7854 | 8.285 cm |
| V88b | black | 11.6280 | 0.4438 | 同左 | 同左 | 同左 | 同左 | 同左 |
| V90 | white | 13.0993 | 0.6169 | 0.6457 | 0.1603 | 40.79% | 0.9626 / 0.7999 | 8.287 cm |
| V90 | black | 11.6580 | 0.4488 | 同左 | 同左 | 同左 | 同左 | 同左 |

V90 相对 V88b：white PSNR 仅 `+0.064 dB`，SSIM `+0.0043`，alpha mean `+0.0022`。这远不足以修复肉眼可见的结构性空洞。

V90 Gaussian health：

- 最大轴 P50/P95/P99/max：`6.93 / 41.99 / 47.97 / 50.00 mm`；
- opacity P50/mean：`0.0948 / 0.3145`；
- opacity `<0.005`：852,041（20.33%）；
- opacity `0.005–0.1`：1,266,095（30.21%）；
- 真正 `opacity>0.1` 的可见高斯仅 2,072,947；
- 轴比 `>10`：4,038,860，`>30`：1,343,961，`>100`：183,987；
- 最短轴 `<0.1 mm`：146,501；
- 墙面有效厚度 P50/P95：`18.75 / 70.01 mm`；
- 可见高斯到 LiDAR 最近点 P95：`16.29 mm`；超过 30 cm 为 1,465 个。

这说明当前不是单一“高斯过圆”。相反，模型已存在大量薄盘/针状高斯，但约一半总人口仍处于低贡献状态；局部表面缺少正确位置、尺度和 opacity 组合。

## 7. 为什么先前自动检查错误放行

1. **PSNR/SSIM 被平均值掩盖。** 24-view mean 不能保证墙面局部最差 ROI 连续；少量好视图可以稀释门窗和屋檐的严重缺口。
2. **LiDAR alpha 只检查稀疏射线邻域。** V90 的 LiDAR alpha mean 已到 0.9626，但整面墙 alpha 仍只有 0.6457。稀疏 anchor 通过不等于像素面覆盖通过。
3. **深度 MAE 可因收缩大高斯而改善。** V87→V88b depth MAE 从 24.57 cm 降到 8.29 cm，但覆盖同时下降。深度更准和表面完整不是同一指标。
4. **没有局部 ROI 最差分位门禁。** 当前缺少墙、门、屋檐、窗边的固定 polygon/patch alpha P05、黑背景泄漏率和结构指标。
5. **没有保留关键中间 checkpoint。** V87 长跑只保留 boundary 1,202 与 final 12,480，生命周期 500–9,360 之间的崩塌时间点无法直接回滚确认，只能依赖 telemetry 或重跑短段。
6. **训练单步指标方差很大。** `with_replacement` 每步一张图是常见 SGD，但不能把任意最后一张图的 loss/PSNR 当作晋级依据；必须使用固定视图集合和固定 ROI。

## 8. 当前阻碍

### 已证实的工程阻碍

- 可信 mesh 覆盖不全，V90 监控中 20% 采样记录完全无 mesh 支持；不能把 mesh alpha 当作全场存在性真值。
- Tile_1/2/3 的 0.5 m block holdout mesh P95 约 22–24 cm，未达到 10 cm 门禁，不能复制 Tile_0 的 mesh 主线。
- 当前输出空间主要保留 final checkpoint，缺少生命周期中间态，无法低成本二分定位第一次产生局部孔洞的 step。
- 已经到达约 4.19M 容量平台；旧策略在容量附近进行大量“出生+删除”，但缺少面向真实渲染覆盖的局部竞争门禁。

### 尚未定论、需要专家意见的算法问题

1. V87 的大尺度覆盖中，哪些是合理低纹理薄盘，哪些是几何作弊雾？目前统一的 5 cm 收缩触发阈值仍然过于粗糙。
2. 对没有可信 mesh 的窗、反光、屋檐和远处结构，应使用多视图 photo-consistency 的软自由出生，还是保留 LiDAR 父点并放宽切平面移动？
3. 应在生长阶段直接加入真实渲染 alpha coverage 梯度，还是只将 coverage 作为 cull 的 veto？
4. 竞品 `[5V,10V,5V]` loss 时序与当前固定 DA2/mesh 权重的差距有多大；是否应先严格恢复时序再调 growth/cull？
5. 在容量上限附近，是否应该从“先 grow 再按 opacity cull”改成局部表面预算替换：细节 ROI split，低纹理 ROI 合并/退出，同时保证渲染 alpha 下限？

## 9. 建议的下一步短门禁

不从 V90 继续训练。建议从 V87 step 12,480 或更早的可用高覆盖节点做一个 200–400 步 V91：

1. 保留 V87 的现有覆盖，不再对所有超过 5 cm 的高斯无差别触发收缩；沿用逐事件 `×0.8` shrink，但仅处理真实投影足迹过大且几何残差不合格的高斯。
2. 重新打开 100-step 周期的 screen-aware detail Split，允许在墙板缝、石材、门边和屋檐生成 2–8 mm 小高斯。
3. means 使用低 LR，并在可信 mesh 区约束 point-to-plane；无 mesh 区允许沿视图一致残差小幅自由移动。
4. opacity cull 暂停或将 coverage-deficit pixel 的贡献高斯设为不可删；严禁再次只根据粒子 opacity 低就删除表面最后覆盖者。
5. mesh/LiDAR alpha 仅作为 coverage veto，不作为全局硬填充；添加固定墙/门/屋檐 ROI 的黑背景 alpha P05 门禁。
6. 每 100 步保留 checkpoint 与无过滤 PLY；只跑到首次生命周期边界和下一次 settle，先看 ROI，再决定长跑。

晋级必须同时满足：

- 用户标记的墙/门/屋檐 ROI 不再出现黑洞；
- black-background alpha 与最差视图不低于 V87，同时 depth MAE 明显优于 V87；
- 无 0.2 m 雾状大高斯重新出现；
- 细节边缘密度增加，平滑墙面仍连续而非均匀堆点；
- 固定 24-view PSNR/SSIM、局部 alpha、geometry、scale、opacity、floater 全部联合通过。

## 10. 证据文件

- V87 config：`outputs/snow-20260224-full-20260825/v87_ultrasharp_detail/tile0_full12480/trainer.config.json`
- V87 manifest：`outputs/snow-20260224-full-20260825/v87_ultrasharp_detail/tile0_full12480/training_ewa/run_manifest.json`
- V88b config：`outputs/snow-20260224-full-20260825/v88b_balanced_tail_repair/tile0_step12680/trainer.config.json`
- V88b manifest：`outputs/snow-20260224-full-20260825/v88b_balanced_tail_repair/tile0_step12680/training_ewa/run_manifest.json`
- V90 config：`outputs/snow-20260224-full-20260825/v90_mesh_alpha_wall_repair/tile0_step12880/trainer.config.json`
- V90 manifest：`outputs/snow-20260224-full-20260825/v90_mesh_alpha_wall_repair/tile0_step12880/training_ewa/run_manifest.json`
- V90 PLY：`outputs/snow-20260224-full-20260825/v90_mesh_alpha_wall_repair/tile0_step12880/snow_tile0_v90_step12880_sh0_full.ply`
- V90 health：`results/diagnostics/snow-tile0-v90-step12880-gaussian-health.json`
- V90 white/black metrics：`results/diagnostics/snow-tile0-v90-step12880-validation24-white/validation_summary.json` 与 `...validation24-black/validation_summary.json`

当前状态：**FAIL / 不晋级 / 不启动剩余 Tile 长跑。**

## 11. 复盘后新增短实验（V91/V91b）

### V91：边界条件失败

V91 原计划在 step 12,500 触发一次生长，但运行条件为
`step < refine_stop_iter`。配置误把 stop 写为 12,500，导致实际没有发生
任何新生长；它只做了 120 步 settle，点数保持 4,191,083。该版本不用于
判断算法方向。

### V91b：真实单次保覆盖生长

权威配置：
`outputs/snow-20260224-full-20260825/v91b_coverage_preserving_regrowth/tile0_boundary12600/trainer.config.json`

- source：V87 step 12,480；
- 生长窗口：start 12,480，stop 12,501，仅 step 12,500 可触发；
- 容量上限：4,600,000；
- growth：ordinary gradient `1.5e-4`，parent opacity `>=0.10`；
- detail split：8 mm + screen radius 0.0025；
- cull：opacity/scale/screen 阈值均禁用；
- means/scales/quats/opacity/color LR：`5e-6 / 0.002 / 0.0005 / 0.01 / 0.001`；
- 从 step 12,500 settle 到 12,600。

实际运行：

- 生长 90,650，删除 0；
- 终态 4,281,733；
- 峰值 VRAM 5.68 GiB；
- 训练时间 153.89 s。

固定 24-view white 指标：PSNR `14.688`，SSIM `0.6447`，alpha mean
约 `0.75`，depth MAE `30.40 cm`；black PSNR `12.969`，SSIM `0.5330`。

V91b 证明“保留 V87 覆盖 + 小规模受约束增长”能够提高 alpha 与黑/白背景
PSNR，但大尺度尾部仍存在：最大轴 P95/P99/max 为
`58.52 / 144.6 / 200.0 mm`，墙面有效厚度 P95 `83.36 mm`，depth MAE
反而高于 V87。即覆盖与几何仍处于跷跷板两端。

下一实验 V92 的唯一变量应为屏幕投影足迹收缩：只处理实际渲染半径超过约
5 px 的高斯，不再按世界尺度统一收缩。目标是在保留 V91b 新生 90,650 个
细节高斯和低纹理覆盖的同时，降低大尺度遮挡、墙厚与 depth MAE。

### V92：逐步 screen clip 失败

V92 从 V91b step 12,600 继续 20 步，固定 topology、means、scales 和
quats 的梯度学习，只开启已有 no-grad screen clip：

- max screen fraction `0.0035`（约 5 px）；
- 单次 hardness `1.10`；
- world threshold 仍为 `0.2 m`；
- opacity/color LR `0.005 / 0.0005`。

实际累计 screen clip events 为 `7,873,707`，即平均每步约 393,685 个
高斯被重复收缩。结果：

- depth MAE：V91b `30.40 cm` → V92 `18.10 cm`，有改善；
- white PSNR/SSIM：`12.7595 / 0.5412`，显著下降；
- alpha mean/P05：`0.6855 / 0.1310`；
- alpha `<0.95`：`48.38%`；
- LiDAR alpha mean/P05：`0.9453 / 0.7270`；
- black PSNR/SSIM：`12.072 / 0.4484`。

V92 再次以破坏覆盖换取深度改善，不晋级、不导出 PLY。根因不是阈值单独
偏小，而是该算子按“当前单视图、每个 optimizer step”重复处理同一高斯，
没有累计观测、冷却时间、每粒子总收缩预算，也没有 alpha/coverage veto。
下一版不能继续只放宽 `max_screen_fraction`，必须先改算子生命周期语义。

### 新增资源阻碍

G 盘当前只剩约 `2.14 GiB`。三个新增实验目录占用：V91 `1.64 GiB`、
V91b `1.95 GiB`、V92 `1.68 GiB`。V91 是未触发生长的边界失败，V92 是
已定性失败，二者的大 checkpoint 可以在用户明确授权后删除；保留配置、
gate、manifest 和指标即可复盘。在释放空间前，不应继续创建约 1.8 GiB/轮
的双份 latest/final checkpoint。
