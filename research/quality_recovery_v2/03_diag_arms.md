# WP03 小实验臂：诊断视角集、短日程配置与 ROI 评分口径

日期 2026-09-11 · 分支 `cloudstudio-3dgs-work` · CPU-only（无训练、无 GPU、未提交）

目的：把任务书 §6 表中的 U0 / U1（3–5 视角）/ DIAG（20–60 视角）以及 §13 的 G0/G1 做成**普通训练臂**——同一训练器、同一 Tile 输入（初始化 PLY、几何 npz、背景库、Face4 缓存原样引用），只把 `views` 限制到看得见诊断区域的父图，日程缩到 H=3000 步。所有配置都通过 `TrainerConfig.from_dict(...).validate()`（含 PLY/npz 哈希与几何↔输入绑定检查），没有削弱任何检查。

## 1. 工具

| 文件 | 作用 |
|---|---|
| `tools/build_diagnostic_set.py` | 区域框 + Tile + 覆盖 CSV → 每父图严格有效可见性表（`per_image_coverage.csv`）→ 预设选图 → **派生**签名 Tile 输入/几何 manifest（引用原文件，不复制）+ `selection.json`（覆盖数字 + ROI 在每个 Tile 裁剪内的像素框） |
| `tools/make_diagnostic_arm_config.py` | 基础臂配置 + 预设目录 → `research_rescaled_horizon_v1` 合同下的 H=3000 臂配置（复用 `make_rescaled_schedule_config.rescale_to_horizon`），R1/G0/G1 变体，`--validate` 走训练器验证，`--eval` 写评估配置 |
| `tests/test_diagnostic_set.py` | 21 项：选图规则、派生 manifest 签名与绑定、ROI 框、H=3000 合同解析、G0/G1 字段、磁盘上已生成配置的合同回归 |
| `run_configs/house0305_tiles/diag_v2/` | 10 个臂配置 + 2 个评估配置 |
| `C:\Peter\3dgs-runs\house0305_sop\diag_v2\<region>\` | `per_image_coverage.csv`、`U0_1/ U1_5/ DIAG_40/`（各含 `tile_inputs_manifest.json`、`tile_geometry_manifest.json`、`selection.json`），合计 732 KB |

### 1.1 每父图"严格有效可见性"

沿用 `audit_observation_coverage.py` 的 strict 档（0.1 m + 3 % range 带内有 vis6 返回且无更近返回、face mask 有效），但**只在该父图属于本 Tile 的面裁剪里判定**（其它面不训练）。区域采样点直接取 `03_observation_coverage.csv` 里该区域的 2000 个 LiDAR 样本，随机取 400（seed 20260911）。每张父图记录：`support_fraction`（严格支持样本占比）、`range_min/median`、面像素足迹、原图 64 px 窗 Laplacian 方差中位/P90、亮度中位、按面拆分的支持数。677（Tile_1）/ 694（Tile_0）张父图各约 7 分钟。

### 1.2 选图规则（`select_views`，纯函数）

* 候选 = `support_fraction ≥ 0.5`；不够 `count` 时放宽到第 `count` 名的支持率，但**不低于 0.2 地板**（记录 `threshold_relaxed / min_support_fraction_applied`）。
* 排序：支持率按 0.1 分箱降序 → 距离升序 → 清晰度降序（一张 1.4 m 处看 1.6 m 宽门扇的图天然覆盖不了整框，0.49 与 0.52 的差别不该压过距离）。
* **U0**（1 张）：≤ 5 m 的近视角优先——先在近池内用 0.5 门槛，空则近池放宽到 0.2 地板，再空才退到远视角；池内取 LapVar 最高。
* **U1**（5 张）：左右物理相机轮流取（同一 rig 帧的左右两张都合法——它们是两台相机，正是"跨相机匹配"要的）。
* **DIAG**（40 张）：同样轮流取，相机可用时各半，余量按名次。

## 2. 选出的视角与覆盖数字

（完整清单与逐图数字见各 `selection.json`；`range` 为该图有效样本的中位距离，LapVar 为原图 64 px 窗 Laplacian 方差。）

### 室内门扇 `indoor_door_leaf_Tile_1`（表 677 张父图；support ≥ 0.5 仅 33 张：左 6 / 右 27；≥ 0.2 有 224 张）

| 预设 | 图数 / Tile 面视角 | 相机 | support min/med/max | range med (m) | LapVar min/med/max | luma med | 备注 |
|---|---|---|---|---|---|---|---|
| U0_1 | 1 / 4 | right | 0.38 | 4.12 | 1308 | 64 | `img_4a58d805e5b8e03110fdfac7`；近池在 0.5 门槛下为空，按规则放宽到地板（`near_pool_stage = near_and_support_floor`）；ROI 落在 2 个面裁剪 |
| U1_5 | 5 / 13 | 3 右 + 2 左 | 0.52 / 0.61 / 0.62 | 6.78–7.18 | 179 / 528 / 907 | 49 | `img_d691623e…`(R 6.78 m) `img_954f086e…`(L 7.17) `img_8a578215…`(R 6.97) `img_dad684e3…`(L 7.18) `img_9f8e6e67…`(R 7.06)；5 个不同 rig 帧 |
| DIAG_40 | 40 / 113 | 34 右 + 6 左 | 0.415 / 0.52 / 0.635 | 1.15–7.62（中位 7.12） | 179 / 689 / 1122 | 57 | 门槛放宽到 0.415（候选 87）；39 个 rig 帧；63 个面视角内有 ROI 框 |

读法：门扇的高支持视角几乎全在 ~7 m 的走廊/对面位置、亮度中位 37–80，只有两三张 1.2–1.5 m 的近图（support 0.49）；左相机看门扇的机会明显少于右相机（6 vs 27）。这和 03 号报告"≤ 5 m 有效视角室内只有室外 28 %"一致，也说明 U1 的"匹配良好视角"在室内只能是远距离组。

### 室外砾石 `outdoor_gravel_Tile_0`（表 694 张；support ≥ 0.5 有 200 张：左 124 / 右 76）

| 预设 | 图数 / Tile 面视角 | 相机 | support min/med/max | range med (m) | LapVar min/med/max | luma med | 备注 |
|---|---|---|---|---|---|---|---|
| U0_1 | 1 / 4 | left | 0.97 | 3.42 | 9076 | 146 | `img_39a2d47accaabf667edfaff8`（也是 03 号 CSV 里 643/2000 个样本的最佳清晰度图）；无放宽 |
| U1_5 | 5 / 20 | 3 左 + 2 右 | 1.00 | 2.51–3.28 | 5056 / 7705 / 8389 | 152 | `img_03f32cc8…`(L 2.51) `img_13691d73…`(R 3.18) `img_2c48bbf5…`(L 2.69) `img_d1b98043…`(R 3.28) `img_0bd60641…`(L 2.75) |
| DIAG_40 | 40 / 151 | 20 + 20 | 0.905 / 1.0 / 1.0 | 2.47–4.75（中位 3.85） | 2660 / 7081 / 8846 | 162 | 无放宽（候选 200）；40 个 rig 帧；63 个面视角内有 ROI 框 |

## 3. 派生 manifest 如何过训练器检查（未削弱任何检查）

训练器 `validate()` 要求：Tile 输入 manifest 自签名 + PLY 哈希一致；几何 manifest 自签名 + npz 哈希一致；**几何 manifest 的 `tile_inputs_manifest_sha256` 必须等于所用 Tile 输入 manifest 的签名**；门禁只绑定 `tile_plan_manifest_sha256`。做法：

* 派生 Tile 输入 manifest：只保留目标 Tile（`tile_count = 1`），`views` 过滤为选中父图的 Tile 面视角（顺序、裁剪 x/y/w/h 原样），`initialization.path/sha256`、boxes、`tile_plan_manifest_sha256` 原样，加 `diagnostic` 溯源块后用仓库同一 `canonical_json_bytes` 规则重新签名；`tile_inputs_root` 仍指向 `tile_inputs_v9`，所以 PLY 按原路径校验。
* 派生几何 manifest：只保留目标 Tile，`geometry.path` 改为相对派生目录的 `../../../tile_geometry_v9/Tile_1/initialization_geometry_k7_k30.npz`（sha256/bytes 原样），`tile_inputs_manifest_sha256` 改绑到派生输入 manifest，`sign_tile_geometry_manifest` 签名。
* 阴性对照（真实文件上跑过）：用原 v9 几何 manifest 配派生输入 → `Tile geometry manifest is bound to different Tile inputs`；改动派生 manifest 里一个裁剪 x → `Tile input manifest signature mismatch`；`controlled_stop_after_steps = 2000` → 合同 `controlled_stop_reaches_declared_final_lr` 拒绝。
* 陷阱：相对路径若要爬超过 3 级会先撞 Windows MAX_PATH（未解析路径 282 字符时 `is_file()` 为 False），工具此时改写绝对路径。

## 4. H=3000 的解析日程（`research_schedule_contract`，两区所有臂相同，仅 means LR 底数按 Tile 不同）

| 字段 | 值 | 占 H | 来源 |
|---|---|---|---|
| `max_steps` | 3000 | 1.0 | H；`controlled_stop_after_steps` 移除，跑满后写 run_manifest |
| `refine_start_iter` | 500 | 0.167 | 训练器 exact-lifecycle 规则，不变 |
| `refine_stop_iter` = `refine_scale2d_stop_iter` = `mcmc_refine_stop_iter` | 2100 | 0.70 | 基础臂实际执行比例 14000/20000，按 refine_every=100 取整；等于合同上限 0.7 |
| `refine_every` | 100 | — | 不变 |
| `reset_every` | 300 | 0.10 | vendor `exact_every300`，不变 |
| `prune_switch_step` | 1500 | 0.5 | H // 2；晚期阈值 0.05 首次可达（6 次事件，首个 1500） |
| grow/cull 事件 | 16 次：500…2000 | | 最后一次出生 2000，之后 1000 步无出生 |
| reset 事件 | 5 次：600、900、1200、1500、1800 | | 最后一次 reset 后 1200 步 |
| `sh_degree` / interval | 1 / 0 | | 全阶从第 0 步 |
| `learning_rates.means` | Tile_1 1.9773e-5（=0.0032×0.0061790）；Tile_0 2.4857e-5（=0.0032×0.0077678） | | 冻结为基础臂实际优化器底数（`surface_initialization_report.json: tangent_scale_median`），`means_step_fraction` 显式 null |
| 末步 means LR | 1.98e-7 = 声明终值 ×1.0015 | | 衰减真正到 ×0.01 |
| `checkpoint_every` | 3000 | | min(基础 5000, H) |

每图访问次数：U0 750 epoch、U1 室内 231 / 室外 150、DIAG 室内 26.5 / 室外 19.9（对比 v9 交付 20 epoch）。

G0/G1 与 R1 的差别只在 `default_strategy.lifecycle_execution_order: post_optimizer_gsplat`（两者）与 `densification_gradient_source: rgb_only`（仅 G1），其余日程字段逐字段相同（`diag.variant_fields` 记录）。

## 5. 验证结果

* 10 个臂配置全部 `TrainerConfig.from_dict(...).validate()` 通过（`.venv-train` python，CPU，含 PLY/npz 哈希）：`diag_{indoor_door_leaf_Tile_1,outdoor_gravel_Tile_0}_{1_R1,5_R1,40_R1,40_G0,40_G1}`。
* `resolved_schedule` 对每个臂 `mismatches = []`；`schedule_contract_fields.resolved` 与解析值逐项相等（训练前会再次核对）。
* 单元测试：`tests/test_diagnostic_set.py` 21 项通过；连同 `test_schedule_contract`、`test_audit_observation_coverage` 共 57 项通过。

## 6. 评估配置与 ROI → 对比条映射

`run_configs/house0305_tiles/diag_v2/<region>_eval.json` = 该区域 DIAG_40 R1 臂的输入/视角原样（`tile_inputs_manifest` 指向 DIAG_40 派生 manifest，`mipmap_tile_id` 不变），`run_id = diag_<region>_eval`，去掉 `diag` 块，加 `diag_eval` 块。现有评估器（`build_three_way_compare.py`、`build_offtrajectory_compare.py`）都从配置的 `tile_inputs_manifest + mipmap_tile_id` 取视角，所以用这个配置评任何一个臂的 checkpoint 就是"在区域自己的视角上评"：

```
python tools/build_three_way_compare.py --config run_configs/house0305_tiles/diag_v2/<region>_eval.json \
    --checkpoint <arm>/checkpoints/latest.pt --output <arm>/compare --frames <view_count> [--reference-ply … --reference-alignment …]
python tools/build_offtrajectory_compare.py run_configs/house0305_tiles/diag_v2/<region>_eval.json <arm>/checkpoints/latest.pt <arm>/offtraj <frames>
```

映射口径（`diag_eval.strip_layout` 与 `roi_in_crops`）：

* 三路对比条 `compare_<k>_<sample_id[:18]>.png` = [photo | ours | reference]，每格就是该 `sample_id` 的 **Tile 裁剪原尺寸**（无缩放），格间 8 px 深灰；第 i 格起点 `x = i × (crop.width + 8)`。`--frames` 是按步长抽样，给 ≥ view_count 才逐视角出图。
* `selection.json / diag_eval.roi_in_crops[sample_id].roi = {x0, y0, x1, y1}` 是区域采样点投影到该面（`FaceSpec.directions_to_pixels`，像素中心 +0.5，索引 `rint(c − 0.5)`）后减去裁剪 `x/y` 的 2–98 % 分位框，即直接是每格内的数组索引；ROI 评分 = 三格各取 `[x0 + i(w+8), x1 + i(w+8)) × [y0, y1)` 后只在框内算（逐通道仿射匹配后的 PSNR、ours/photo 的 Laplacian 方差比等，沿用 `score_compare_strips.py` 的口径）。`roi = null` 的面视角不看该区域（室内 113 个面视角中 63 个有框、室外 151 中 63 个）。
* 离轨对比 `offtraj_*.png` = [ours | reference]，相机已移位，ROI 框不适用，整格评分。
* 注意：U0/U1 臂在 DIAG_40 视角上评时，未训练过的视角是"新视角分数"，DIAG 臂则是训练视角分数，两者分开报。

## 7. 未能满足 / 残留

* **相机平衡**：室内 DIAG_40 是 34 右 + 6 左——左相机 support ≥ 0.415 的父图只有 6 张，不是选图缺陷而是采集几何；室内 U0 也无法同时满足"≤ 5 m 且 support ≥ 0.5"，按规则放宽到 0.38。
* **区域外高斯的命运**：初始化仍是整块 Tile 的 PLY（Tile_1 342 万点），只有诊断视角给梯度；不可见高斯经 reset 后不再恢复会被 cull 掉——这是区域诊断的预期副作用，不是交付；`diag.notes` 已写明。若要"局部容量充足"以外还保留其它区域，需要空间裁剪初始化（未做）。
* **背景库 / DA2 / vis6 缓存**：按 sample_id 查表，子集不需要重建；未在 GPU 上跑通任何一个臂（CPU-only），训练侧行为未验证。
* `03_roi_provisional.json` 仍是临时 ROI；`01_roi_registry.json` 落地后应重跑 `build_diagnostic_set.py`（每区约 7 分钟）并重新生成配置。
* 未提交任何文件；`research/quality_recovery_v2/README.zh-CN.md` 有其它任务的未提交改动，本任务未触碰。

## 8. F3：数据侧去掉 pitch_up 面（DIAG_40_F3，2026-09-11，CPU-only、未训练、未提交）

**动机**：DIAG-40 室内 R1 的树色/天空色悬浮烟雾集中在 `pitch_up_56` 面（README "悬浮体成因线索" / "DIAG-40 室内 F2"）；F2 把树/天空像素从光度监督里遮掉后悬浮体原样存在。F3 是同一假设的数据侧版本：**40 张父图不变，直接去掉它们的全部 `pitch_up_56` 面视角**，其余面、配方、日程合同、cap 规则、LiDAR 缓存与 R1_c134 完全相同。若悬浮体随天空视角一起消失，即是"悬浮体来自被 pitch_up 视角放到错误深度/无人管的高斯"的数据侧证据。

**工具改动**（`tools/build_diagnostic_set.py`、`tools/make_diagnostic_arm_config.py`，`tests/test_diagnostic_set.py` +7 项）：

* `--exclude-faces FACE[,FACE]`：在签名前从派生 `views` 里删掉这些 face_id；缺省不给时输出**字节相同**（排除信息只在启用时才进入签名的 `diagnostic` 块）；任一父图会剩零个面则拒绝（错误信息列出图 id）。`selection.json` 记 `excluded_faces` 与 `face_exclusion{view_count_before/after, views_removed, face_counts_before/after, parent_images_kept}`。
* `--preset-name DIAG_40_F3`：输出目录名，不碰 `DIAG_40`。
* `--reuse-selection <DIAG_40/selection.json>`：父图 id 与顺序、覆盖行、policy 原样复用（校验 region / count / Tile 输入 sha 一致），不重算 7 分钟的逐图表；`selection.json` 记 `reused_selection{path, sha256, parent_image_ids_identical}`。
* `make_diagnostic_arm_config.py`：变体 `F3`（配方 = R1，差别全在 preset），`--label-suffix c134`（run_id / `diag.variant` = `F3_c134`），`--cap-init-multiplier 1.34`（`cap_max = floor(1.34 × Tile 初始化点数)`，点数取派生 manifest 的签名值并与 PLY 头 `element vertex` 交叉核对；记 `diag.cap_rule` / `diag.cap_policy`），`--note`；`diag.data_variant` 带出 preset 的排除/复用记录。

**视角数（父图 40 / 40 不变）**：

| 区域 | R1_c134 面视角 | F3 面视角 | 去掉 | 剩余面构成 | 含 ROI 框的面视角 |
|---|---|---|---|---|---|
| 室内 `indoor_door_leaf_Tile_1` | 113 | **73** | 40 × pitch_up | yaw−35 34 / yaw+35 32 / pitch_down 7 | 63 → 50 |
| 室外 `outdoor_gravel_Tile_0` | 151 | **115** | 36 × pitch_up | yaw−35 40 / yaw+35 39 / pitch_down 36 | 63 → 63 |

每图访问次数：室内 41.1 epoch（R1 26.5）、室外 26.1（19.9）——H=3000 不变，视角少了 epoch 自然变多，判读时要记住这一点。

**核对**（脚本逐项比对，全部为真）：父图 id 与顺序、`selected` 行、region 框、inputs 记录与 DIAG_40 相同；F3 的 `views` = R1 的 `views` 去掉 pitch_up 后原顺序、裁剪 x/y/w/h 逐字段相同；保留视角的 ROI 框与 DIAG_40 逐项相同；派生输入 manifest 按 `tile_inputs_v9` 根校验 PLY 哈希通过，几何 manifest 绑定到 F3 输入 sha 且 npz 哈希通过；`DIAG_40/` 三个文件 mtime 未变、sha 仍与 `R1_c134` 配置里记录的绑定一致。两份配置 `TrainerConfig.from_dict(...).validate()` 通过（`.venv-train`，CPU，含 PLY/npz 哈希）；与 `R1_c134` 逐字段比对，差异仅：`run_id`、`output_dir`、`tile_inputs_manifest`、`initialization_geometry_manifest`、`schedule_contract_fields.arm/moved_from_base`、`diag` 块（variant、manifest sha、view_count/sample_ids、cap_rule、data_variant、note）；`schedule_contract_fields.resolved`、`cap_max`（室内 4579208 / 室外 9440001）、`face_lidar_geometry_manifest`（vis6）相同。

**ROI compare 视角**：`DIAG_40/roi_compare_ids.json` 室外 45 个全在 F3 里；**室内 49 个里有 5 个是 pitch_up 面、不在 F3 数据集里**：`img_552ccaef…::pitch_up_56`、`img_f1799d46…::pitch_up_56`、`img_d0d851b2…::pitch_up_56`、`img_26e10d65…::pitch_up_56`、`img_c1d98062…::pitch_up_56`。对 F3 而言它们是**新视角**，评分时要么剔除、要么单列（`roi_compare_ids_u1.json` 的 5 个 yaw 面两区都完整）。

**文件**：

| 文件 | sha256 |
|---|---|
| `diag_v2/indoor_door_leaf_Tile_1/DIAG_40_F3/tile_inputs_manifest.json` | `059b3210c151e4b35b214d7bfcda13f9d94b27fe3c3fc85a79b24029f298e995` |
| `diag_v2/indoor_door_leaf_Tile_1/DIAG_40_F3/tile_geometry_manifest.json` | `75fc600cced956a2c1946cf3bbd6dda2679375490d2762e7d7886653947b49da` |
| `diag_v2/outdoor_gravel_Tile_0/DIAG_40_F3/tile_inputs_manifest.json` | `806c687c21716f6effc7526ee8c010bb3887cb335cb233a487c8e5f2b1b4a276` |
| `diag_v2/outdoor_gravel_Tile_0/DIAG_40_F3/tile_geometry_manifest.json` | `2839f286f76f44de5a6949f875cbef5a7978597efec4402954b220244de80c5b` |
| `diag_indoor_door_leaf_Tile_1_40_F3_c134.json`（runs 根目录；副本在 `run_configs/house0305_tiles/diag_v2/`） | `d6ac4c2ffd7b617ee2936783f201c5395b4ea38087eca8d92d43182d78ef55dc` |
| `diag_outdoor_gravel_Tile_0_40_F3_c134.json`（同上） | `02ba3faf4e4059199dd283ccf25545ce3a9438a91a4c823945a8c6208797c6ef` |

`tests/test_diagnostic_set.py`：改动前 21 通过 / 9 失败，改动后 28 通过 / 同样 9 失败——失败全是 `GeneratedConfigsOnDiskTests` 对既有 `*_ci.json` / `*_X1_c134.json` 的 `run_id`（带 `-` 而非 `_`）≠ 文件名 stem 的子测试，本任务未动它们；两份 F3 配置的 stem / run_id / output_dir 叶名一致，通过该回归。

**残留**：未在 GPU 上跑；室内 F3 的 ROI compare 只剩 44 个视角（5 个 pitch_up 为新视角）；`pitch_down_56` 面保留（室内仅 7 个）。
