# 14 — B 线：块外内容替身背景（stand-in backdrop）设计与准备（2026-09-12）

状态：**盘点、代码阅读、B1 设计、工具与测试、三份配置及其 CPU 校验已完成；未做任何 GPU 执行**（本机 CPU-only；未提交）。所有数字来自本目录 `14_standin_backdrop/*.json`（CPU 只读计算，未动任何 checkpoint / 缓存）；工具 `tools/audit_competitor_tile_coverage.py`、`tools/build_standin_backgrounds.py`；测试 `tests/test_competitor_tile_coverage.py`、`tests/test_standin_backgrounds.py`。

出发点（README 2026-09-11 17:15 行、K2/K3 行、门平面核查行、`11` §5–7）：切片训练合成 `final = render + (1 − α)·backdrop`，而 `tile_backgrounds_v9/Tile_*` 的 backdrop **只有天空穹顶**——没有树、没有邻切片、没有门洞后面的室内。于是切片盒内没有 LiDAR 几何的像素只能由切片自己的高斯去"画"：Tile_1 20k 有 34% 高斯离初始化云 > 0.2 m、35% 在训练框外（`10` §1），K2/K3 在天空里留下深色树替身团块，门前灰雾的真正内容在盒外 2.65 m 之后（`13`）。文献里每个分块系统都先给块一个块外内容的替身（VastGaussian / CityGaussian 的全场粗先验、H-3DGS 的邻块支架、BlockGaussian 的辅助高斯），冻结、不回传、合并时按所有权裁掉。B 线就是把这一项落到我们的 backdrop 机制上。

## 1. 盘点：磁盘上有什么、谁拥有什么

### 1.1 可作替身的全局 / 邻块模型

| 资产 | 内容 | 覆盖 | 可用性 |
|---|---|---|---|
| `delivery_g9/merged.pt`（1.78 GB，19.31M）+ `house0305_g9_merged.ply`（1.41 GB） | G9d 四切片 `core_owner_only` 合并 | 只保留各自 **core_box** 内的高斯（合并丢弃 5.26M） | 按构造不含任何切片 core 之外的内容；且是 R1d 之前一代 |
| `delivery_R1d/`：`merged.pt` **已在发布后删除**（delivery_report 记 2.46 GB、26.71M）；`exports/candidate_R1d/house0305_R1d_merged.ply`（1.88 GB，18.09M，opacity ≥ 0.05）+ `house0305_R1d_sky.ply` | R1d 合并导出 | 同上 core-only | 同上；PLY 可回读（`tools/import_gaussian_ply.py`），但没有 halo |
| 四个 R1d 切片 checkpoint（`checkpoints/latest.pt`，含优化器状态） | tile0 `tile0_R1_range0_20k` 3.50 GB / 11.46M；tile1 `tile1_R1d_20k` 1.02 GB / 3.34M；tile2 `tile2_R1d_20k` 2.28 GB / 7.47M；tile3 `tile3_R1d_cap13m_20k` 3.02 GB / 9.89M | 各自 training_and_export_box + 长出去的悬浮体；合并时各丢 1.24M / 1.23M / 1.50M / 1.47M（在 core 外） | **B1 替身 (a) 的来源**：未合并、带 halo、带各自"画树"的高斯 |
| `probes/sky_house0305.pt`（5.6 MB） | 照片烘焙天空穹顶：100k 个 SH0 高斯，半径 250 m，仰角 −15°..88°，222 帧采样，opacity logit 4（=0.982） | 只有远场天空色块，无树 | 现有 backdrop 的唯一内容；B1 继续用 |
| 切片前的全场训练 | **本数据谱系（`house0305_sop_v8/v9`）没有**：`house0305_sop/` 下 `stage_c_*`、`v13`–`v23` 等所有配置都带 `tile_inputs_manifest`（当年 tile_inputs 只有 Tile_0 一块也算切片）；`3dgs-runs/` 根目录 122 份 `probe_*_config.json` 是更早战役（`house0305_manifest_ba` / `face_cache_ba` / `init_18m` planar_surfel，如 `probe_e1` 16k），checkpoint 在 `probes/<arm>/`，**位姿谱系不同（BA 后的 manifest）**，作替身前得先做配准核查 | — | 不推荐；**没有现成的全场粗模型，B0 必须新训** |

### 1.2 竞品 vs 我方：盒子分区与 LiDAR 支持（`coverage_usa_training_box.json`、`coverage_r1d_training_box.json`）

竞品 `USAgs.ply`（22,452,075 高斯）按 `probes/usa_gs_alignment.json`（model→LAS，`means @ Rᵀ + t`，配准残差中位 3.5 cm）变到 s1_local；我方取 R1d 导出 PLY（18,088,789）。先按四个 `training_and_export_box` 分类：

| 模型 | 盒并集外 | Tile_0 | Tile_1 | Tile_2 | Tile_3 |
|---|---|---|---|---|---|
| 竞品（training box） | **0（0.00%）** | 8,557,589 | 2,030,161 | 4,592,071 | 7,272,254 |
| 竞品（core box） | 0 | 8,433,755 | 2,042,088 | 4,542,236 | 7,433,996 |
| R1d 导出 | 0 | 6,737,962 | 1,396,291 | 4,324,375 | 5,630,161 |

**盒并集外是空集**：tile plan 把四个盒子铺满了一个外扩 ~25 m 的 LAS 包围盒（x −81.3..92.1，y −75.9..80.3，z −8.1..33.9），竞品模型范围 x −57.8..56.8 / y −65.6..59.8 / z −5.4..30.9 整体落在里面。"谁都不拥有的内容"不是一块空间，得换口径量：**离 LiDAR 多远**（到 18,409,149 点的 `house0305_init_full/sparse_pc.ply` 的最近距离 > 0.5 m）：

| 模型 | LiDAR 无支持（> 0.5 m） | 占比 | Tile_0 | Tile_1 | Tile_2 | Tile_3 | z p50 / p95 (m) | LAS xy 范围之外 |
|---|---|---|---|---|---|---|---|---|
| 竞品 | **283,839** | 1.26% | 76,183 | **1,686** | 93,036 | 112,934 | 9.2 / 22.0 | 5,564 |
| R1d 导出 | **630,252** | 3.48% | 150,672 | **111,078** | 179,139 | 189,363 | 5.7 / 23.5 | 419 |

读法：
* 竞品的无支持高斯是**树冠**：z 中位 9.2 m，俯视图（`coverage_usa_lidar_support.png`，红=无支持）里是围着房子半径 20–55 m 的一圈树丛与远处的树排；几乎全在 Tile_0/2/3 的盒子里，Tile_1 盒（穿过房子的 y ∈ [−5.06, −1.52] 窄带）里只有 1,686 个。
* 我方无支持高斯是 2.2 倍，z 中位 5.7 m，俯视（`coverage_r1d_lidar_support.png`）是弥散的，**Tile_1 里 111,078 个（竞品的 66 倍）**——这些不是树，是悬浮体；与 `10` §1 的 34% / 框外 35% 一致。
* 所以 Tile_1 视角里的树、远景，其 3D 位置在**邻切片的盒子里**；邻切片的 raw checkpoint（合并前）里确实有它们（各自长出来的、模糊的树替身），这就是替身来源 (a) 的依据。竞品盒外仅 5,564 个（0.02%）在 LAS xy 范围之外，"远场超出扫描范围"的内容可以忽略。
* `coverage_usa_inside_outside.png` 是任务要求的"按盒内/盒外上色"图——全蓝（0 个盒外）——保留作证据。

### 1.3 视角裁剪覆盖（`crop_coverage.json`）

3,536 个 Face4 面里 30 个不属于任何切片；**只有 3.4% 的照片像素落在所有切片裁剪之外**（pitch_up 面 99.3% 被某个切片的裁剪框覆盖，pitch_down 90.6%）。也就是说树、天空这类像素**几乎总在某个切片的裁剪里被监督**——问题不是"没人看到"，而是：看到它的切片（例如 Tile_0 的 pitch_up 裁剪）用自己的高斯去画，位置可能落在 Tile_2 的盒里，合并时被 `core_owner_only` 丢掉；而 Tile_2 自己的裁剪围绕的是 Tile_2 的 LiDAR 回波，未必包含那棵树。这决定了 B0 全场粗先验的意义在**跨切片一致性**（同一棵树被 884 张照片一起约束，而不是每个切片各画一份过拟合的烟雾），不是覆盖。

### 1.4 R1d 邻块替身的实际规模（`standin_plan_R1d_neighbours.json`，plan-only 实跑，CPU 4 分钟）

`tools/build_standin_backgrounds.py --plan-only` 真实加载三个邻块 checkpoint、剔除 Tile_1 `training_and_export_box` 内的行、opacity ≥ 0.05、按各自中位曝光增益烘入颜色：

| 来源 | 输入 | Tile_1 盒内剔除 | opacity < 0.05 剔除 | 保留 | 烘入增益 |
|---|---|---|---|---|---|
| tile0_R1_range0_20k | 11,458,698 | 454,280 | 4,017,917 | 6,986,501 | 0.8818 |
| tile2_R1d_20k | 7,469,977 | 666,045 | 1,980,688 | 4,823,244 | 0.9154 |
| tile3_R1d_cap13m_20k | 9,889,343 | 138,949 | 2,985,998 | 6,764,396 | 0.9327 |
| **替身合计 + 穹顶** | | 1,259,274 | | **18,574,141 + 100,000 = 18,674,141** | 参数 1.60 GiB（float32） |

邻块伸进 Tile_1 盒里的 1.26M 行（halo + 悬浮体）被剔除，Tile_1 盒内仍只由 Tile_1 自己负责。

## 2. 背景库现状（代码事实）

* **烘焙**（`tools/build_view_backgrounds.py`）：`FaceCacheDataset.camera_sample(i)`（只取相机，不读图）→ `backend.render(dome, sample, with_range=False, background_rgb=(1,1,1))`（穹顶不透明 0.982，白底几乎不露）→ `clamp → uint8 → PIL 缩小 4×（BILINEAR）→ PNG`；manifest `views[sample_id] = {file, height, width}`（缩小后的尺寸）+ `{split, dome_source, dome_sha256, downsample, background_rgb}`，`write_view_background_manifest` 对整个 body 做 sha256 签名。**没有任何 mask**——渲染 mask（鱼眼圆、FoV、低权重、行人）只在损失侧作用，backdrop 是整面渲染。`view_backgrounds_v9`：3,536 面，2.7 分钟（46 ms/面，PNG 编码主导）。
* **切片裁剪**（`tools/build_tile_view_backgrounds.py`）：曾经的"压扁"故障——库在存图尺寸 ≠ 请求尺寸时用 `interpolate` **缩放**、从不裁剪，切片裁剪视角请求的是裁剪框尺寸，整面被压进框里（Tile_1 86% 视角受影响，误差 17–60/255）。修法是数据侧：把 4× 缩小的整面先 BILINEAR 放回面尺寸、再按 `tile_inputs` 的 `{x, y, width, height}` 切出，逐切片存成 `tile_backgrounds_v9/Tile_N`（Tile_1：1,829 面，其中 1,445 裁剪、384 整面；`cropped_view_count` 记在 manifest），让库只做它做得对的"原尺寸取用/放大"。
* **消费**（`ViewBackgroundLibrary`）：构造时校验 `manifest_sha256`（body 篡改即拒绝）、`views` 非空；`background_for(image_id, H, W)` 缺视角**直接抛错**（fail-closed，从不回退常量色）；解码后的 uint8 图**无上限缓存在内存**——Tile_1 1,829 个裁剪合计 4.84 G 像素 = **13.5 GiB**（现在的穹顶裁剪库就是这个体量，本机 31.6 GB 内存跑过 K2/R1d）；尺寸不符才 bilinear 缩放。
* **训练器**（`trainer._render_supervision_loss`）：每步 `background_for(sample.image_id, H=tensors["rgb"].shape[0], W)` → `backend.render(..., background_rgb=<HxWx3 张量>)`，光栅器内 `rgb + (1 − α)·background`（`backend.py` 541–549，深度不合成）→ 之后才乘逐帧曝光增益 `rgb_gain`、算 L1/SSIM/alpha/天空项。`validate()` 只检查 manifest 与 root 成对存在（`trainer.py` 836–850）；**训练契约 / identity 不签 backdrop manifest 的 sha**（`contract_dict` 无 background 项）——B1 与 R1d 的唯一差别正是这个文件，要在 identity 里补记 `background_manifest_sha256`（见 §6）。
* 天空监督（K 线）与 backdrop 的分工：K2 把天空 mask 像素的 α 压向 0 并免除光度项，剩下的像素仍由 backdrop 补。B2 = K2 + 替身时，天空 mask 之外（树枝、被标为"其他"的团块）的像素第一次有了替身可解释。

## 3. B1 设计：`backdrop(view) = render(dome + standin \ box(Tile_1))`

1. **替身层** = (a) 其余三块 R1d checkpoint（raw，合并前）+ (b) B0 全场粗先验；每层：按各自 `auxiliary_params.exposure_log_gains` 中位增益烘入颜色（与 `merge_v28_tile_checkpoints.py --harmonize-exposure` 同一 DC 公式 `sh0·g + (g − 1)·0.5/C0`，另把 shN 也乘 g，合并工具没乘）→ 剔除 **Tile_1 `training_and_export_box`**（可选外扩 `--exclude-margin-m`）内的行 → 剔除 sigmoid(opacity) < 0.05 的行（交付导出阈值）→ 可选 `--anchor-ply + --max-anchor-distance-m`（到 LiDAR 最近点 > d 的行剔除，cKDTree 有界查询）。
2. **合成**：穹顶（SH0，shN 零填到 3 带）与替身**拼成一个参数集，一次光栅化**（深度排序由光栅器负责），白色常量做最底层；不走"先渲穹顶再叠替身"的两次 alpha 合成，少一次 α 记账，且与训练时穹顶+高斯共同排序的语义一致。
3. **相机**：`FaceCacheDataset(..., tile_views=Tile_1.views).camera_sample(i)`——与训练逐位相同的裁剪主点与尺寸；**在裁剪分辨率直接渲染并存图**（`downsample` 缺省 1），manifest 里的 height/width 就是裁剪尺寸，库永远原样取用，压扁问题在构造上不可能出现。工具核对渲染尺寸 = 相机尺寸、sample id 无重复、渲染集合 = 切片视角集合，否则拒绝写 manifest。
4. **manifest**：schema 不变（`views` + 签名），元数据保留 `split / dome_source / dome_sha256 / background_rgb / downsample / tile_id`，新增 `source_tile_inputs_manifest_sha256`、`render_resolution: tile_crop` 与 `standin` 溯源块（各来源 path / sha256 / step / 输入数 / 三类剔除数 / 保留数 / 烘入增益；剔除盒与 margin；opacity 地板；anchor 规则；曝光帧；渲染耗时）。训练器一行不改即可消费；`ViewBackgroundLibrary` 的签名与 fail-closed 规则原样生效。
5. **曝光帧**：穹顶是照片帧；替身烘入各自增益后也在照片帧（`--target-gain 1.0`）。训练时合成结果再乘 Tile_1 的逐帧增益（R1d 中位 0.935）——backdrop 与穹顶一样承受约 7% 的系统性偏亮，与现状一致；若要放进 Tile_1 的"未增益帧"，`--target-gain 0.9347`。
6. **B0 全场粗先验的规格**（`house0305_global_coarse_B0_10k.json`，由 `14_standin_backdrop/make_b_line_configs.py` 从 `tile1_R1d_20k.json` 派生并 **`TrainerConfig.validate()` 通过**）：

   | 项 | R1d Tile_1 | B0 | 说明 |
   |---|---|---|---|
   | tile_inputs / mipmap_tile_id / geometry manifest | 有 | **去掉** | 无裁剪，`FaceCacheDataset` 迭代全部 **3,536** 面（884 × 4，空 mask 过滤 0） |
   | initialization_ply / geometry | Tile_1 全 LiDAR 3.42M + `mipmap_k7_k30` npz | `house0305_init_2m/sparse_pc.ply`（**1,863,918** 点，5.3 cm 体素，同一 LAS sha `8bed917b…`，s1_local）+ `lidar_init_geometry.npz`（k=16） | 现成，不用再建 |
   | surface_initialization.mode | mipmap_k7_k30 | **planar_surfel** | mipmap 模式硬绑切片输入 |
   | metric_scale_calibration.mode | precomputed | **knn**（k7 算术均值） | precomputed 只允许 mipmap 初始化 |
   | 视角/步数 | 1,829 × 20 = 36,580，停 20k | **3,536 × 20 = 70,720**，`controlled_stop_after_steps` **10,000** | epoch 置换采样硬性要求 max_steps = 20 × 视角数 |
   | cap_max | 15M | **3M** | 粗先验 |
   | 增长 | refine_stop 14k | **8k**（`refine_stop_iter` / `refine_scale2d_stop_iter` / `mcmc_refine_stop_iter`），`prune_switch_step` = 70,720 // 2（契约要求） | 最后 2k 步只做 opacity 收敛 |
   | backdrop | `tile_backgrounds_v9/Tile_1` | `view_backgrounds_v9`（整面穹顶，4× 缩小、库放大——整面视角下是正确的） | |
   | 其余（损失、alpha 地板 0.1/6 px、DA2 0.15、法向、sh1、lr、生命周期档位、gate `gates_v9/gate_17_training_da2.json`） | 同 | **同** | gate 绑定的是 dataset / split / face4 sha，与切片无关，校验通过 |

   **切片外路径是活的**：`validate()` 对 B0 PASS（gate、面缓存、渲染 mask、DA2、vis6 LiDAR 几何、人像 mask、planar_surfel 几何文件全部核过）；训练入口 `train()` 在 `tile_inputs_manifest is None` 时 `tile_views=None`、不建 ownership 盒，其余分支相同。未在 GPU 上跑过的部分：kNN 尺度标定在 1.86M 点上的耗时（估 1–2 分钟）、planar_surfel 初始化与 `LidarNormalAlignment` 用 k=16 npz 的口径（`probe_e1` 当年就这么跑的）、3,536 面 × 白/穹顶整面渲染的显存（Tile_0 11.4M 高斯峰值 11.2 GiB；B0 ≤ 3M，预计 < 6 GiB）。
7. **可选的更便宜路径**：先做 **B1a = R1d + 仅邻块替身**（不训 B0，替身 18.57M 现成），如果门前灰雾与天空树烟雾已经消失，B0 只是锦上添花；若邻块替身自己的悬浮体被烘进 backdrop 而 Tile_1 长出"纠正替身"的高斯，再上 B0 或 anchor 过滤。

## 4. GPU 步骤（按序；估时来自 `monitor/progress.jsonl` 实测：Tile_1 0.307 s/步、Tile_0 0.533 s/步、K2 115 min；穹顶烘焙 46 ms/面）

前置：`set PYTHONPATH=C:\Peter\cloudstudio-3dgs-work`，`env_machine_b.cmd`，与 R1d 相同的 gsplat 扩展（`00_runtime_identity.json`）。

| # | 步骤 | 命令 | 估时 | 产物 / 门禁 |
|---|---|---|---|---|
| 1 | **B0 全场粗先验训练** | `python tools\train_gsplat.py --config C:\Peter\3dgs-runs\house0305_sop\house0305_global_coarse_B0_10k.json` | **75–90 min**（10k 步 × 0.35–0.45 s/步：整面 4.24 MP 比 Tile_1 裁剪均值 2.65 MP 大 60%，但高斯 ≤ 3M；+ 5 min 初始化/标定） | `global_coarse_B0_10k/checkpoints/latest.pt`（step 10000，controlled_stop）；先看 `compare` 出图：树、远景是否成形，别只看 loss |
| 2 | **替身 backdrop 烟测** | `python tools\build_standin_backgrounds.py --config ...\tile1_R1d_20k.json --dome C:\Peter\3dgs-runs\probes\sky_house0305.pt --standin-checkpoint <tile0 latest.pt> --standin-checkpoint <tile2> --standin-checkpoint <tile3> --standin-checkpoint <B0 latest.pt> --output ...\tile_backgrounds_B1\Tile_1 --limit 24` | **5 min**（加载 4 个 checkpoint 3–5 min CPU + 24 面） | 24 张 PNG，**必须人工对照照片**：pitch_up `img_8a5782…`（屋檐+树枝+蓝天）、门洞 `img_9f8e6e67::yaw_plus_35`、`img_02167f…`；替身里的墙/树/门后室内应出现在正确位置。显存若溢出：`--min-opacity 0.1`（替身约减半）|
| 3 | **替身 backdrop 全量** | 同上去掉 `--limit` | **15–25 min**（1,829 面 × 0.3–0.5 s 渲染 ~21M 高斯 + PNG 线程编码；磁盘估 2–4 GB，比穹顶裁剪 618 MiB 大，内容高频） | `tile_backgrounds_B1/Tile_1/background_manifest.json`（签名；`standin` 块）；`ViewBackgroundLibrary` 加载一次确认签名与 1,829 视角 |
| 4 | **B1 = R1d + 替身** | `python tools\train_gsplat.py --config ...\tile1_B1_standin_20k.json` | **~105 min**（= R1d 102 min；backdrop 解码同体量） | `tile1_B1_standin_20k/checkpoints/latest.pt`；训练中每 5k 步看一次 `compare` 出图（教训：只盯代理指标看不见背景错配） |
| 5 | **B2 = K2 + 替身** | `python tools\train_gsplat.py --config ...\tile1_B1_standin_K2sky_20k.json` | **~115 min**（= K2） | 同上 |
| 6 | **每臂评估** | `build_three_way_compare --sample-ids roi_compare_ids.json` → `score_compare_roi --match-brightness`；`build_offtrajectory_compare` + `score_offtrajectory_strips`（18 帧）；`audit_surface_anchor.py --config <臂>`；morph；`freeze_run_identity` | **~10 min/臂** | `compare_roi_compare_ids.roi_bm.txt`、`offtraj.log`、`10_surface_anchor/<臂>_audit.json`、`identity/<臂>.json` |

合计约 **5.5–6.5 h GPU**（B0 → 替身 → B1 → B2 → 评估）。B1a（跳过 B0）约 3 h。

## 5. 判读协议（配对基线 `tile1_R1d_20k` 与 `tile1_R1d_rerun_20k`；K2 作第二基线）

复跑噪声带（README 08:11 行）：离轨 PSNR **±0.5 dB**、ROI ±7%、整板 ±4%；赢面门槛 ROI ≥ +15%、整板 ≥ +10%，PSNR 只作否决（−1 dB 以上算变差）。

1. **门叶 ROI**（49 视角、亮度归一 ours/ref，`score_compare_roi --match-brightness`）：R1d 0.315–0.366，K1 0.453，K2 0.465，K3 0.472。B1 的假设是门洞像素由替身（邻块的门后室内 / B0）解释、Tile_1 不再在门洞里捏灰雾，ROI 应 ≥ K2；若 B1 ≈ R1d 而 B2 ≈ K2，说明门洞的雾来自 alpha 地板（`13` 记 6 px 膨胀覆盖 99.9% 门洞像素）而非缺替身——那时下一刀是 O 线归属遮罩 + 替身。
2. **离面比例**（`audit_surface_anchor.py`：> 0.2 m 份额、远组 z p50、框外份额）：R1d 0.338 / 2.46 m / 0.353，K2 0.367。这是本机制的几何读数：树/檐外凸/门洞雾都不再需要 Tile_1 的高斯，> 0.2 m 份额应显著下降，**框外份额尤其应下降**（框外像素现在有替身）。分不清"悬浮体消失"和"真实内容消失"处按 `11` 的要求用出图判。
3. **天空 / 树视角出图**（`figures/`：pitch_up `img_8a5782`、`img_02167`、`img_a4bc51`，与 `tile1_R1d_vs_K2.jpg`、`tile1_K2_vs_K3.jpg` 同帧）：R1d 的棕色树状烟雾应被替身里的树取代（模糊但在正确位置）；K2 天空里的深色团块应消失（它们是 mask 外的树替身高斯，现在有替身了）；墙面帧 `img_a415` 不应变差。
4. **离轨 PSNR**（18 帧）：R1d 15.16 / 复跑 16.14，K2 15.62，K3 14.66（否决线）。替身对新视角的作用应为正（BlockGaussian 报 +0.1 dB 量级），但带宽 ±0.5 dB，只作否决项。
5. **整板 6 帧亮度归一锐度**：R1d 0.076–0.082，K2 0.107；B 线不直接作用于墙面，应不降。
6. **训练遥测**：`rgb_psnr` 应高于 R1d（backdrop 解释了更多像素）；终态高斯数应低于 K2 的 455 万（K2 +36% 是排除天空监督后容量流向别处）；`lidar_alpha_support_fraction` 不应变。合并侧：Tile_1 core 外的份额（R1d 36.9% 被合并丢弃）应下降。
7. **替身本身的审计**（一次性）：对 `standin` 溯源块里的保留行跑 `audit_surface_anchor`（> 0.5 m 份额 = 烘进 backdrop 的悬浮体量）；若 > 5%，做 `--max-anchor-distance-m 0.5` 的对照替身（代价：无回波的树冠会被一起剔掉，`11` §7.2 的四类内容）。

## 6. 风险

* **重复计数**：邻块 halo 与悬浮体伸进 Tile_1 盒内的 1.26M 行已剔除；反过来 Tile_1 自己长在盒外的高斯（R1d 有 35%）与替身画同一像素——训练里 backdrop 已能解释该像素时光度损失不再需要它们，opacity 会掉、cull 清掉，这是想要的方向；但**已存在的过拟合行不会立刻消失**，且 Tile_1 仍可能新长高斯去"纠正"替身的错误（替身糊、错位）。合并时 `core_owner_only` 不变，替身从不进合并，Tile_1 盒外行照旧丢弃。
* **替身模糊被烘成真值**：backdrop 冻结、无梯度，替身的模糊与自身悬浮体成了 Tile_1 那些像素的"照片"，Tile_1 无法改善它们；替身若错（曝光帧偏差 ~7%、配准、邻块自己的树烟雾），Tile_1 会在前面长不透明内容去补差。缓解：opacity 地板、anchor 过滤对照、B0 的跨视角一致性、`--limit` 烟测人工对照、pitch_up 出图判。
* **合并时的所有权**：B1 只改 Tile_1 的训练目标，交付仍是四切片 core 合并——树、远景仍不在交付里（竞品有 284k 树冠高斯，我们交付 0）。若要交付含树，需要定所有权规则：切片拥有各自 core；B0 只贡献"离 LiDAR > d 的行"或"没有任何切片高斯的体素"。这是后续决定，不在 B1 范围内。四块都改 B 线时替身来源是**上一代**（R1d）的其他三块——CityGaussian 的粗→细两阶段，不是循环依赖。
* **契约不签 backdrop**：训练契约/identity 不含 background manifest sha；B1 与 R1d 唯一差别在这个文件。跑前把 `background_manifest.json` 的 `manifest_sha256` 记进 `identity/tile1_B1_*.json`（`freeze_run_identity` 之外手工加一键），否则事后分不清用的哪一版替身。
* **内存 / 显存 / 磁盘**：backdrop 缓存 13.5 GiB（与现状同）；替身渲染 ~21M 高斯参数 1.6 GiB + 光栅缓冲，推理无优化器状态，16 GB 卡应可（tile3 训练崩溃是 12.5M 高斯 + Adam 14 GiB 的另一回事）；溢出时 `--min-opacity 0.1` 或 `--downsample 2`（缓存降到 3.4 GiB，库均匀放大 2×，无错配只有轻微模糊）。磁盘 2–4 GB。
* **B0 自身**：它是一个 3M 高斯、10k 步的粗模型，同样没有替身（只有穹顶）、同样会长悬浮体；它的价值是全场一致性与树的粗形状，不是锐度。B0 的悬浮体也会被烘进 backdrop——anchor 过滤对照就是为它准备的。
* **K2 组合**：B2 里天空 mask 像素 α→0 且免光度，backdrop 的替身在这些像素上只提供颜色，不冲突；但 mask 腐蚀/守卫带内的树枝像素现在有替身可解释，可能改变 K2 的"深色团块"成因判断——按 §5.3 出图。

## 7. 文件

* 工具：`tools/audit_competitor_tile_coverage.py`（任意 PLY × 签名 tile inputs：盒内/盒外、逐切片、opacity 子集、LiDAR 支持子集、俯视 PNG）；`tools/build_standin_backgrounds.py`（CPU：加载 / 曝光烘入 / 盒剔除 / opacity / anchor / 拼层 / plan；GPU 入口：裁剪相机渲染 + 签名 manifest + `standin` 块；`--plan-only`、`--limit`、`--verify-against`）。
* 测试（本机 `pytest tests/test_standin_backgrounds.py tests/test_competitor_tile_coverage.py`：13 通过；连同 `test_view_backgrounds`、`test_tile_ownership*`、`test_export_gaussian_ply`、`test_gaussian_health` 34 通过）：行选择三规则的归因计数与顺序、盒退化拒绝、anchor 缺点拒绝、shN 零填拼层、曝光公式与合并工具 DC 公式一致、`build_standin` 端到端溯源、假后端渲染写库→`ViewBackgroundLibrary` 原尺寸取用/签名/缺视角 fail-closed/篡改拒绝、尺寸不符与重复 id 拒绝；覆盖工具的分类/变换方向/有界查询/端到端（opacity、anchor、PNG）。
* 数据（本目录 `14_standin_backdrop/`）：`coverage_usa_training_box.json`、`coverage_usa_core_box.json`、`coverage_r1d_training_box.json`、`coverage_usa_inside_outside.png`、`coverage_usa_lidar_support.png`、`coverage_r1d_lidar_support.png`、`crop_coverage.json`（+ `crop_coverage.py`）、`standin_plan_R1d_neighbours.json`、`b_line_config_validation.json`（+ `make_b_line_configs.py`）。
* 配置（`C:\Peter\3dgs-runs\house0305_sop\`，均由 `make_b_line_configs.py` 生成、可重生成）：`house0305_global_coarse_B0_10k.json`（validate **PASS**）；`tile1_B1_standin_20k.json`、`tile1_B1_standin_K2sky_20k.json`（与基线只差 `background_image_manifest/root`、`run_id`、`output_dir`、`lineage`；替身库建成前 validate 按设计**只**在 `background image manifest is missing` 处失败，指回穹顶库后 PASS）。

## 8. 未验证门禁

GPU 上一步都没跑：B0 训练（kNN 标定耗时、planar_surfel + k16 法向口径、整面显存）、替身渲染（显存、耗时、PNG 体积）、B1/B2 训练与 §5 全部读数、`--limit` 烟测的人工对照。曝光帧偏差 ~7% 未做对照臂。`identity` 补记 backdrop sha 仍是手工步骤。README 索引行未加（本文件即记录；由执行 GPU 步骤的人在跑完后补行）。
