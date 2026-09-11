# WP05 §8.3 后续 — vis6 缓存修复：`build_face4_lidar_geometry.py` warp 路径真正施加隐藏点剔除并重建 `face4_lidar_train_vis6f`

日期 2026-09-11 · 分支 `cloudstudio-3dgs-work`（研究分支，未提交） · CPU-only（`.venv-train` python，`CUDA_VISIBLE_DEVICES=""`，未训练、未占用 GPU）

> 前置：`05_visibility_alpha_audit.md` §3.4 判定 `face4_lidar_train_vis6` 名不副实——warp 分支只把 `visibility_cell_px = 6` 写进 manifest，从未调用 `visible_point_mask`。本文记录 builder 的修复、回归测试、自检工具、重建后的缓存/切片几何、以及在修复缓存上可跑的 D0/D1 臂。

## 0. 结论先行

1. **缺陷位置（修复前 `tools/build_face4_lidar_geometry.py`，HEAD `853f489`）**：warp 分支 251–267 行 `warp_sparse_depth_to_face(...)` 之后只做 `face_valid &= supervision_mask` 与 `keep &= finite & 0 < conf ≤ 1`，没有任何距离比较；331–335 行按 `projection_config.visibility_cell_px > 0` 给 `projection` 字段加 `_visibility_filtered` 后缀；`projection_config.to_dict()` 原样写出 cell/tolerance/margin。仓库里 `visible_point_mask` 只有两个调用点：`lidar_projection.py:216`（`project_camera_points_to_face`，即 `--point-cloud` 直投路径）与 `:353`（`project_lidar_depth`，鱼眼缓存构建）——warp 路径两者都不经过。所以 v9 缓存的 9.59 亿"有效像素"里含穿墙/穿玻璃/门扇多副本回波，manifest 却宣称已过滤。
2. **修复**：warp 分支在 supervision-mask 相交之前，对整张 warp 栅格的 z-buffer 胜者调用 `visible_point_mask`（新函数 `face_visibility_keep_mask`，139 行；调用 335–337 行）。栅格是最近距离 z-buffer，每个 6 px cell 上"胜者的最小距离 = 全部投影点的最小距离"，所以对胜者过滤与直投路径"先过滤后 z-buffer"保留的点集**逐点相同**（差别只在直投路径的 `support_count` 计数）。直投路径本来就在 `project_camera_points_to_face` 内过滤，现在通过新增的 `stats` 出参（`lidar_projection.py:223-224`）回报候选/保留计数。
3. **manifest 现在记录请求与实际施加**：新增顶层 `visibility_filter = {requested{cell,tolerance,margin} | null, applied, applied_in, stats_basis, candidates, kept, removed, removed_fraction}`，每条 record 多一个 `visibility{candidates, kept, removed}`（仅过滤开启时）。旧缓存没有这个块（`null`），这就是区分"只记参数"与"真正过滤"的判据。
4. **重建结果 `face4_lidar_train_vis6f`**：3536 面 / 有效像素 **550,863,673**（旧 959,390,055，保留 57.4 %）；栅格层面 981,976,228 个回波中剔除 418,048,295（**42.6 %**）；4.4 GB（旧 7.6 GB）；单进程 **39 min 14 s**（旧缓存按文件时间戳约 10 min，过滤的 `np.minimum.at` 是新增成本）。manifest sha `5e855da4…`（旧 `98305c46…`）。
5. **自检 `--verify-visibility`**：同一 5+5 个 DIAG_40 面视角，旧缓存 3×3 窗宽松规则违反率 室内均值 8.6 %（最大 24.8 %）/ 室外 9.7 %（最大 32.0 %），重跑过滤会再丢 31 % / 28 %；修复缓存两项在全部 10 面上**恰为 0.000**（cell 过滤对 18 px 邻域最小值判定，界住任意 3×3 窗；再施加一次是幂等）。
6. **可用臂**：D0f/D1f 配置已落盘并通过 `TrainerConfig.from_dict(...).validate()`，与 c134 基线的 D0/D1 构成 2×2（§5）。切片 Tile_0/Tile_1 的按切片 LiDAR 几何（步骤 L）也从修复缓存重建（§4），但 v9 训练配置（含诊断臂）直接绑定全量缓存，切片几何目前只被审计工具消费。

## 1. 缺陷与修复（代码）

| 文件 | 修复前 | 修复后 |
|---|---|---|
| `tools/build_face4_lidar_geometry.py` warp 分支 | 251–267：warp → `&= supervision_mask` → 有限/置信度过滤 → 写 npz | 335–337：warp → **`face_visibility_keep_mask`**（z-buffer 胜者上的 `visible_point_mask`，配置 cell/tolerance/margin）→ `face_valid = visible & supervision_mask` → 同前 |
| 同上 direct 分支 | `project_camera_points_to_face` 内部已过滤，无计数 | 310：传 `stats=` 拿到 `visibility_candidates/kept` |
| 同上 manifest | 331–335 仅按参数加 `_visibility_filtered` 后缀 | 444：`visibility_filter` 块；382：record 级 `visibility` |
| 同上 CLI | 无自检 | 644：`--verify-visibility --output <cache> [--sample-id … | --sample-ids-file selection.json | --sample-count N --seed S] [--verify-report out.json]`；构建参数改为按需必填 |
| `cloudstudio_3dgs/geometry/lidar_projection.py` | — | `project_camera_points_to_face(..., stats=None)`，223–224 行填计数；行为不变 |

过滤放在 supervision-mask 相交**之前**：mask 外的回波仍参与"最近表面"判定，与直投路径（mask 最后施加）一致；审计 §3.4 的探针是在已 mask 的缓存像素上做的，二者的差别只在 mask 边缘。

### 1.1 回归测试 `tests/test_build_face4_lidar_geometry.py`（7 项，全部通过；`tests.test_lidar_projection`、`tests.test_tile_face_lidar_geometry` 共 16 项通过）

* 纯函数：48×48 栅格，5 m 墙每 3 px 一个回波 + 8 m"墙后"回波每 6 px 一个（各自独占像素，z-buffer 不会去掉）。cell 6：墙后全部剔除、墙全部保留、`candidates = kept + removed`；cell 0：全保留。
* 直投 stats：既有 `test_hidden_point_filter…` 场景（4 个 3 m 前景 + 1 个 7 m 漏点）在 `project_camera_points_to_face(stats=)` 下报 `{candidates 5, kept 4}`。
* 端到端 warp：合成鱼眼相机（64×64，KB4）+ 一张 front 面（fx 40）+ 签名的 dataset / Face4 / 鱼眼深度 manifest；同一输入 `--visibility-cell-px 6` 与 0 各建一次。cell 0 缓存保留 8 m 回波、`visibility_filter.applied = false`、record 无 `visibility`；cell 6 缓存 8 m 回波为 0、5 m 墙像素数与未过滤缓存相等、`applied = true`、`applied_in = warped_face_raster_zbuffer_winners_before_supervision_mask`、`removed > 0`、磁盘 manifest 与返回值一致。
* 自检：对 cell 0 缓存按 cell 6 规则验证，违反率与重跑剔除率 > 0，`manifest_visibility_filter.applied = false`；对 cell 6 缓存两项 = 0，`applied = true`；manifest 无过滤参数又不给 `--visibility-cell-px` 时拒绝。

## 2. 重建（步骤 F）

命令（与 v9 vis6 同一条路径：无 `--point-cloud`，warp v8 鱼眼稀疏深度缓存；单进程）：

```
python tools/build_face4_lidar_geometry.py
  --face-manifest  C:\Peter\3dgs-datasets\house0305_sop_v9\face4_train\face_manifest.json
  --face-root      C:\Peter\3dgs-datasets\house0305_sop_v9\face4_train
  --dataset-manifest C:\Peter\3dgs-datasets\house0305_sop_v8\dataset_manifest.json
  --depth-manifest C:\Peter\3dgs-datasets\house0305_sop_v8\depth\depth_manifest.json
  --depth-root     C:\Peter\3dgs-datasets\house0305_sop_v8\depth
  --output         C:\Peter\3dgs-datasets\house0305_sop_v9\face4_lidar_train_vis6f
  --storage-profile audit --workers 1 --visibility-cell-px 6 --visibility-tolerance 0.2 --visibility-margin-m 0.1
```

为什么不用 `--point-cloud colorized.las`：旧缓存 `intermediate_fisheye_raster = true`（warp 路径），修复缓存走同一路径才能让 2×2 对照只差"过滤有没有做"；直投路径会同时改变量化次数（1 vs 2）与置信度语义，是另一臂。colorized.las 的 sha `8bed917b…` 通过 v8 depth manifest 作为身份锚进入 manifest（`source_point_cloud_sha256`），与旧缓存相同。

| 项 | 旧 `face4_lidar_train_vis6` | 新 `face4_lidar_train_vis6f` |
|---|---|---|
| manifest sha | `98305c46e0a73e3d606ea985469d3467982fc0590073fa8f14ba995ce8917bc7` | `5e855da4878eb76aea4843aa910f297ec622368cfe329eda23ee75f253efdf6d` |
| `projection` | `kb4_forward_splat_nearest_range_zbuffer_visibility_filtered`（名不副实） | 同名，且 `visibility_filter.applied = true` |
| `visibility_filter` | 无 | requested {6, 0.2, 0.1}；applied_in `warped_face_raster_zbuffer_winners_before_supervision_mask`；candidates 981,976,228 / kept 563,927,933 / removed 418,048,295（0.4257） |
| 面数 / 有 depth 面数 | 3536 / 3536 | 3536 / 3536 |
| 有效像素 | 959,390,055 | **550,863,673**（0.574） |
| 大小（audit profile，float32 置信 + provenance 列） | 7.6 GB | 4.4 GB |
| 运行时间 | ≈ 10 min（文件时间戳 04:13–04:23） | 39 min 14 s（13:52:34–14:31:48） |
| 绑定 | source_face `449e7f9b…`，source_depth `af8fb540…`，dataset `33f5f814…` | 相同（训练器 gate 绑定 `lidar_depth_manifest_sha256` 不变） |

按面（栅格剔除率 = removed / candidates，mask 前；保留率 = 新/旧有效像素）：

| 面 | 旧有效像素 | 新有效像素 | 保留 | 栅格剔除 |
|---|---|---|---|---|
| pitch_down_56（地面） | 16,828,960 | 13,414,037 | 0.797 | 0.206 |
| pitch_up_56（天花/树冠） | 145,282,149 | 78,361,215 | 0.539 | 0.460 |
| yaw_minus_35 | 390,265,532 | 224,113,225 | 0.574 | 0.426 |
| yaw_plus_35 | 407,013,414 | 234,975,196 | 0.577 | 0.423 |

逐面分布：栅格剔除率均值 0.311、p10 0.000、p50 0.264、p90 0.680、最大 0.950；保留率 p10 0.319 / p50 0.738 / p90 1.000。DIAG_40 视角集合：室内 113 面视角 43.18 M → 27.08 M（保留 0.627，视角中位 0.667，p10 0.436），室外 151 面视角 45.14 M → 31.86 M（0.706，中位 0.817，p10 0.491）。与审计 §3 的判读一致：地面几乎不动，pitch_up 与两个 yaw 面（门窗、树冠、屋檐、家具边界）丢得最多。

## 3. 自检数字（`--verify-visibility`，每区 5 面，`random.Random(7)` 从 DIAG_40 `view_sample_ids` 抽样；报告在 scratch，未入库）

指标定义：`windows_loose_violation_frac` = 有回波的 3×3 像素窗中 `farthest > 1.2·nearest + 0.1 m` 的比例（审计 §3.4 的口径）；`rerun_removed_fraction` = 对缓存自身像素再跑一次 `visible_point_mask(cell 6)` 会丢的回波比例。

| 面视角 | 区 | 旧：3×3 违反率 | 旧：重跑剔除 | 新：违反率 | 新：重跑剔除 | 新 manifest `visibility` (cand/kept/removed) |
|---|---|---|---|---|---|---|
| `img_72d329aa…::yaw_plus_35` | 室内 | 0.1027 | 0.2384 | 0.000 | 0.000 | 413,253 / 315,569 / 97,684 |
| `img_26e10d65…::yaw_plus_35` | 室内 | 0.2481 | 0.3625 | 0.000 | 0.000 | 641,817 / 407,476 / 234,341 |
| `img_88226fd0…::pitch_up_56` | 室内 | 0.0177 | 0.2264 | 0.000 | 0.000 | 182,326 / 140,881 / 41,445 |
| `img_d0d851b2…::pitch_down_56` | 室内 | 0.0013 | 0.0578 | 0.000 | 0.000 | 25,118 / 23,694 / 1,424 |
| `img_0cfe3b00…::pitch_up_56` | 室内 | 0.0619 | 0.6730 | 0.000 | 0.000 | 249,246 / 82,116 / 167,130 |
| `img_70756332…::yaw_plus_35` | 室外 | 0.0822 | 0.1538 | 0.000 | 0.000 | 550,102 / 465,659 / 84,443 |
| `img_38fa922a…::yaw_plus_35` | 室外 | 0.3200 | 0.4221 | 0.000 | 0.000 | 654,497 / 378,090 / 276,407 |
| `img_ba1f9bf2…::pitch_down_56` | 室外 | 0.0010 | 0.0898 | 0.000 | 0.000 | 27,124 / 24,433 / 2,691 |
| `img_074273e7…::pitch_up_56` | 室外 | 0.0727 | 0.7263 | 0.000 | 0.000 | 229,458 / 63,883 / 165,575 |
| `img_0bd60641…::yaw_plus_35` | 室外 | 0.0095 | 0.0226 | 0.000 | 0.000 | 234,986 / 229,625 / 5,361 |
| **均值 / 最大** | 室内 | 0.086 / 0.248 | 0.312 / 0.673 | 0 / 0 | 0 / 0 | |
| | 室外 | 0.097 / 0.320 | 0.283 / 0.726 | 0 / 0 | 0 / 0 | |

审计 §3.4 在 v8 鱼眼缓存上量到 12.7–53.5 %（并非同一批面、且是鱼眼栅格而非 Face4 栅格），这里在 Face4 栅格上的旧值 0.1–32 % 与之同量级；修复后为 0 是构造保证而不是抽样运气（§0.5）。

## 4. 按切片 LiDAR 几何（步骤 L，仅 Tile_0 / Tile_1）

`tools/build_tile_face4_lidar_geometry.py --tile-inputs …\tile_inputs_v9\tile_inputs_manifest.json --tile-inputs-root …\tile_inputs_v9 --tile-id N --face-manifest …\face4_train\face_manifest.json --dataset-manifest …\v8\dataset_manifest.json --source-geometry-manifest …\face4_lidar_train_vis6f\face_lidar_geometry_manifest.json --source-geometry-root …\face4_lidar_train_vis6f --output-root C:\Peter\3dgs-runs\house0305_sop\tile_face_lidar_v9f\Tile_N`

| 切片 | 视角 | 非空视角 | 源缓存与视角重叠像素 | crop 内像素（三维框过滤前） | 切片像素（crop ∩ training_and_export_box） | 大小（uint8 置信，无 provenance） | 运行时间 | manifest sha |
|---|---|---|---|---|---|---|---|---|
| Tile_0 | 2132 | 2121 | 404,741,693 | 330,929,829 | **223,199,229** | 1.3 GB | 3 min 47 s | `e0ef9c082573520e510a0c85fd5b63405e1a6910e8eeb0bd4ccb96a1becb7e8c` |
| Tile_1 | 1829 | 1796 | 356,677,634 | 244,739,987 | **82,895,738** | 468 MB | 2 min 12 s | `422889dc0a32876b4c035e69e733a4549dee1033fa4aafbaf496a2d05f37c9ba` |

两个 manifest 的 `source_face_lidar_geometry_manifest_sha256 = 5e855da4…`（vis6f）、`source_depth_manifest_sha256 = af8fb540…`、`tile_inputs_manifest_sha256 = 7af17f43…`（tile_inputs_v9）。v9 之前没有落盘的按切片几何可比（`tile_face_lidar_v9` 不存在），所以这里没有旧/新对照；同一工具在 v8 Tile_1 上的记录是 374 视角 / 44.5 M 像素（`MIPMAP_ALIGNED_FACE4_PIPELINE_SOP.zh-CN.md`）。两个切片并行构建（CPU），日志在各自 `Tile_N.build.log`。

注意：v9 训练配置（`run_configs/house0305_tiles/v9/tile{0,1}_*.json`）和全部诊断臂的 `face_lidar_geometry_manifest/root` 直接指向**全量** train 缓存，训练器在 `FaceCacheDataset` 里按 tile view 的 crop 切片；没有单独的"按切片几何"配置键，`tile_inputs` manifest 也不绑定 LiDAR 几何 sha。`tile_face_lidar_v9f` 因此目前只服务 `tools/audit_tile_lidar_accuracy_coverage.py` 之类的审计，以及需要更小侧车的场景（其 manifest 也是合法的 `face4_sparse_lidar_geometry`，`source_depth_manifest_sha256` 沿用，可直接换进配置）。Tile_2/3 未建。

## 5. 2×2 臂矩阵（DIAG_40，c134 基线：`cap_max = 1.34 × 切片初始化点数`）

配置文件（同时落在 `C:\Peter\3dgs-runs\house0305_sop\` 与 `run_configs/house0305_tiles/diag_v2/`）：

| 臂 | 缓存 | `lidar_alpha_support_mode` | 文件 / run_id（= 文件 stem = output_dir 名） |
|---|---|---|---|
| D0 | 旧 vis6 | dilated（缺省） | `diag_<region>_40_R1_c134.json`（既有；run_id `…_40_R1-c134`） |
| D1 | 旧 vis6 | strict_visibility | 未落盘；= D0 加一行（sha 见下） |
| **D0f** | vis6f | dilated | `diag_<region>_40_R1_c134_v6f.json` |
| **D1f** | vis6f | strict_visibility | `diag_<region>_40_D1_c134_v6f.json` |

`trainer_config_sha256`（`sha256(canonical_json(TrainerConfig.contract_dict()))`，全部四臂 `from_dict(...).validate()` 通过，含 manifest/PLY/npz 哈希与几何↔输入绑定）：

| 区 | D0 | D1 | D0f | D1f |
|---|---|---|---|---|
| indoor_door_leaf_Tile_1 | `894ce916fef80a4857f02592ebff4eac354f8c213c3a8bab3c0ed9d45424ce83` | `65b9663c830da35baaa7c171e425117009dfe1ef3c7570a810b49a94e049cf44` | `b911fa1c16115e374046ffc82defb423a3114742c487ac9b9f1a6d2d9f915d8d` | `8496154765f6721a71f6291ed743433f53720325a419b35aa076284bdaaed9a6` |
| outdoor_gravel_Tile_0 | `7d14bee41c543ff72cd062f3d25eed63318051d2242b2ea72a145cafe2c56ef9` | `5596faafb92d341e6d983c921d9626ec955008a94a4f78bf210707d280ddb930` | `6261af01e18a0d6bcf9fc33715650e8d1b83ab785b65b3791476178b3b366899` | `54f9c5d43bfe3b675351f685d083359419e4ff136876d09c1f8f6f1c01f4b0e1` |

合同扁平化差分（两区相同）：D0→D0f **只差 1 键** `/face4_lidar_geometry/manifest_sha256`（`98305c46…` → `5e855da4…`）；D0f→D1f 与 D0→D1 都只差 7 键，全在 `/loss_contract/lidar_alpha_coverage/`（`source`、`support_mode`、`strict_visibility/{search_radius_px 6, tolerance 0.03, margin_m 0.1, edge_erosion_px 3, rule}`）。日程、学习率、生命周期、cap、alpha 权重/目标/半径、视角集合、切片输入/几何全部相同。

新配置相对 c134 基线的改动只有：`run_id`/`output_dir`（`…/diag_v2/<region>/runs/<stem>`）、`face_lidar_geometry_manifest/root`、（D1f）`lidar_alpha_support_mode`，以及 `diag.variant`（`R1_c134_v6f` / `D1_c134_v6f`，让 `tests/test_diagnostic_set.py::GeneratedConfigsOnDiskTests` 的 stem = run_id = `diag_run_id(...)` 规则对这四个文件成立；该测试对既有 `_ci` / `X1_c134` 文件的 9 个失败是它们自己的 stem/run_id 不一致，与本任务无关）和 `diag.cache_variant`（记录 base/diag 的几何路径与 sha、支持模式、来源说明）。

**诊断 tile inputs manifest 不需要重导**：`diag_v2/<region>/DIAG_40/tile_inputs_manifest.json` 的 `diagnostic` 块只记 `derived_from_tile_inputs_manifest_sha256`、parent images、view 列表、区域框；不含 LiDAR 几何 sha，训练器也没有 tile_inputs ↔ face_lidar_geometry 的绑定检查（只检查 geometry ↔ face cache、geometry ↔ gate 的 depth manifest）。旧几何 sha 只出现在 `selection.json.inputs.lidar_geometry_manifest_sha256`——它记录的是**选视角时**用来算支持分数的缓存。若按修复缓存重跑 `tools/build_diagnostic_set.py --lidar-geometry-manifest …vis6f…`，支持分数会变、选出的 40 张父图可能不同，2×2 就不再共享视角集合；本轮刻意保持视角集合不变。

## 6. 判读建议与残留

* 先跑 D0f（只换缓存）对 D0：回答"穿透回波当支持"本身伤不伤室内；再跑 D1f 对 D0f：在干净栅格上"过度外扩"还剩多少害处。审计预期：discontinuity 类被拒大幅下降、interior_band / erosion 类基本不变（§3.4 表中过滤后 r=6 严格带违反率仍 17–32 %）。
* 修复缓存的 `lidar_range_weight` 交付臂（0.05）和 03 号报告的 loose/strict 有效视角计数都应在 vis6f 上重算，本轮未做。
* 训练器/`FaceCacheDataset` 读取 manifest 时不看 `visibility_filter`；它只是证据字段。若要让训练器拒绝"只记参数未过滤"的旧缓存，需要在 `trainer.py` 的几何绑定检查里加 `applied` 断言——会让所有旧配置失效，属于另一决定。
* 直投路径（`--point-cloud`）本轮只加了计数，未重建；它的 `applied_in = project_camera_points_to_face_before_face_zbuffer`。
* 磁盘：vis6f 4.4 GB + 切片几何（§4）；机器剩余约 34 GB（重建期间其他任务也在写盘）。
* 工作树同时有别的任务在改 `trainer.py` / `exposure.py`（曝光曲线），本轮的 `validate()` 是在那份工作树 trainer 上通过的；本任务未触碰这两个文件。
