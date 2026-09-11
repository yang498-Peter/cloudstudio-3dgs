# WP05 §8.3 — 可见性与 alpha 支持审计：当前 radius-6 max-pool 构造 vs 严格可见性支持

日期 2026-09-11 · 分支 `cloudstudio-3dgs-work`（研究分支，数据只读，未提交） · CPU-only（torch 仅在 CPU 上用于交叉核对）

> 数字来自 `05_visibility_alpha_overlays/support_fractions.csv`（264 行 = 室内 DIAG_40 的 113 个面视角 + 室外 DIAG_40 的 151 个面视角）与 `summary.json`，工具 `tools/build_alpha_support_overlays.py`。叠加图 48 张（每区 12 个代表视角 × 整面 1/4 缩放 + ROI 1:1）在同目录。
> 训练器旋钮 `lidar_alpha_support_mode` 与共享模块 `cloudstudio_3dgs/training/alpha_support.py` 已入工作树；测试 `tests/test_alpha_support.py` 9 项。

## 0. 结论先行

1. **当前构造的支持区不是"LiDAR 表面"，而是"任何一个返回 6 px 内的一切"。** 室内 DIAG_40 面视角上当前支持覆盖 RGB 有效像素的 85.1 %（像素加权；视角中位 89.3 %），门扇 ROI 内 99.6 %；室外 79.0 %，砾石 ROI 91.2 %。
2. **按严格规则（窗内所有返回落在最近返回的 0.1 m + 3 % 带内，否则视为深度断边并外扩 3 px 剔除）重算，室内当前支持像素的 73.9 % 被拒（视角中位 82.0 %），门扇 ROI 内 95.5 %；室外 46.4 %（中位 51.6 %），砾石 ROI 内只有 5.1 %。** 被拒像素中 75 %（室内）/ 72 %（室外）连 vis6 自身的宽松规则（20 % + 0.1 m）都不满足，即窗内混着相差 20 % 以上距离的返回；只违反 3 % 严格带的"斜面/小台阶"占被拒的 15 % / 16 %；纯粹因边缘外扩而丢的占 10 % / 12 %。
3. **根因不只在膨胀——输入栅格本身未做隐藏点过滤。** 名为 `face4_lidar_train_vis6` 的 Face4 LiDAR 几何是从 v8 鱼眼稀疏深度缓存（`kb4_ray_zbuffer_v1`，`projection` 里没有 visibility 字段）warp 得到的；`tools/build_face4_lidar_geometry.py` 的 warp 分支只把 `visibility_cell_px = 6` **写进 manifest**、命名为 `_visibility_filtered`，从未调用 `visible_point_mask`。直接在 v8 鱼眼缓存上验证：室内 `img_02167f65` 3×3 窗有 53.5 % 违反宽松规则，若真正施加 cell-6 过滤会丢掉该图 63.7 % 的返回并把违反率压到 0.000；另 4 张（室内 2、室外 2）同样 12.7–26.2 % → 0.000、丢 20–38 %。这意味着 alpha 支持、range 监督和 03 号报告的"loose/strict 有效视角"计数都在消费一张含穿墙/穿玻璃/门扇多副本返回的栅格。已作为独立任务标出（修 builder + 回归测试，不在本任务范围内重建缓存）。
4. **D1 臂（仅换支持模式）在现有栅格上等价于"室内几乎关掉 alpha 地板、室外保留地面但剔掉树冠与屋檐"。** 室内严格支持只剩 RGB 像素的 22 %（门扇 ROI 4.5 %），室外 42 %（砾石 ROI 86.5 %）。它能回答"过度外扩的支持是否伤害室内"，但与"栅格本身脏"混杂；干净的判读需要先重建过滤后的栅格再跑 D0/D1。

## 1. 当前构造在哪里、做什么（行号为本工作树当前状态）

| 环节 | 位置 | 内容 |
|---|---|---|
| 稀疏 LiDAR 载入 | `cloudstudio_3dgs/training/face_dataset.py:820-821` | `depth_range, depth_confidence, depth_valid = load_sparse_depth(npz).to_dense()`；`depth_mask = rgb_mask & depth_valid`（rgb_mask = renderer mask PNG > 0） |
| 切片裁剪 | `face_dataset.py:924-925` | 三张栅格按 tile view 的 `x,y,w,h` 切片，之后所有池化都在裁剪后的张量上做（裁剪边界当"无返回"填充） |
| 张量化 | `trainer.py::_tensor_sample`（2713-2725 行） | `range_m` float32、`confidence` float32、`depth_mask` bool |
| 支持区与权重 | `trainer.py:2996-3006`（原为内联代码，现调用 `alpha_support.lidar_alpha_support`） | `valid = depth_mask & isfinite(conf) & conf > 0` → `conf0 = where(valid, conf, 0)` → 半径 r>0 时 `max_pool2d(conf0, kernel 2r+1, stride 1, pad r)` → `support = pooled > 0`；`mask = rgb_mask & support` |
| 损失 | `trainer.py:3007-3019` | `support_fraction = mask.mean()`（全裁剪像素）；`deficit = relu(target − alpha)`；`loss = Σ pooled_conf · deficit² / Σ pooled_conf`，再乘 `lidar_alpha_weight` |
| 配置 | `trainer.py:275-282` | DIAG_40 R1_ci：`lidar_alpha_weight 0.1`、`target 0.95`、`dilation_radius_px 6`、`surface_alpha_floor_profile true`、`lidar_range_weight 0.0` |
| 合同 | `trainer.py:2110-2135` | `loss_contract.lidar_alpha_coverage.source = signed_lidar_depth_mask_confidence_max_dilated`，`dilation_radius_px 6` |

即：13×13 窗内只要有一个已签名返回，像素就要求 alpha ≥ 0.95，权重是窗内最大置信度；返回的**距离从不参与**支持判定。膨胀不看邻域返回是否同一表面，所以近物体轮廓向远表面外扩 6 px、远表面向近物体外扩 6 px，天空缝隙（树枝之间、屋檐之下）只要 6 px 内有枝条返回也被要求不透明（室外 `pitch_up_56` 面：当前支持 79 %，严格 19 %，被拒 76.5 %，见 `outdoor_gravel_Tile_0__07__…yaw_minus_35.png` 第 2/4 面板中树冠与天空之间的蓝/红）。

### 离线复现精度

`tools/build_alpha_support_overlays.py` 用 `alpha_support.lidar_alpha_support_numpy`（numpy 双胞胎）逐视角重算；`tests/test_alpha_support.py::test_dilated_mode_is_bit_identical_to_legacy_inline_construction` 把训练器现在调用的 torch 版钉在旧内联代码上（mask 与权重 `torch.equal`），`test_numpy_and_torch_twins_agree` 把 numpy 版钉在 torch 版上；工具本身又对每区前 5 个真实视角在 CPU torch 上重跑训练器函数并比对 `rgb_mask & support`，两区 10/10 完全一致（`summary.json.regions.*.torch_cross_check`）。**没有不能离线复现的部分**：tile ownership masking 在诊断臂中未启用（工具遇到启用会拒绝），person mask 已烘进 renderer mask，渲染 alpha 不属于支持区构造。

## 2. 严格规则与被拒像素的分类

* `strict_visibility`（`alpha_support.py:46-50, 94-143`）：同一 (2r+1)² 窗（r = `lidar_alpha_dilation_radius_px` = 6，与当前口径同尺度）取最近返回 d_min 与最远返回 d_max；`agrees = d_max ≤ (1 + 0.03)·d_min + 0.1 m`（与 `tools/audit_observation_coverage.py` 的 `STRICT_TOLERANCE / STRICT_MARGIN_M` 相同）；`discontinuity = 有返回 & ¬agrees`；再把 discontinuity 外扩 3 px（`STRICT_VISIBILITY_EDGE_EROSION_PX`）从支持中剔除。权重仍是当前的池化置信度，**只有 mask 变**。r = 0 时严格模式与当前模式相同（无窗即无分歧）。
* 被拒像素（当前支持 ∧ ¬严格支持）按窗内返回分歧程度拆三类：
  * `discontinuity`：`d_max > (1 + 0.2)·d_min + 0.1`，连 vis6 宽松规则都不满足（真断边或穿透返回）；
  * `interior_band`：只违反 3 % 严格带（斜面、曲面、≤ 20 % 的小台阶）；
  * `edge_erosion`：窗内一致，但落在 discontinuity 的 3 px 外扩内。
* 单元夹具（`tests/test_alpha_support.py`）：稀疏栅格墙（每 3 px 一个 5 m 返回）+ 30 px 宽前景杆（2 m）。当前模式全图支持；严格模式墙的内部（含无返回的格点间隙）与杆的内部保持支持，杆轮廓两侧各 18 px 带（含 3 px 外扩）被拒；随机稀疏图与逐像素循环参考实现逐位一致（r = 0/2/4）。

## 3. 测得的过度外扩（DIAG_40 面视角）

### 3.1 按区域（像素加权 / 视角中位 [p10, p90]）

| 指标 | 室内门扇 Tile_1（113 面视角） | 室外砾石 Tile_0（151 面视角） |
|---|---|---|
| 当前支持 / RGB 有效像素 | **0.851** / 0.893 [0.736, 0.998] | **0.790** / 0.855 [0.567, 0.928] |
| 严格支持 / RGB 有效像素 | **0.222** / 0.165 [0.073, 0.362] | **0.423** / 0.429 [0.171, 0.620] |
| 被拒 / 当前支持 | **0.739** / 0.820 [0.597, 0.927] | **0.464** / 0.516 [0.000, 0.791] |
| 其中 discontinuity（违反 20 % 规则） | 0.555 / 0.595 | 0.334 / 0.323 |
| 其中 interior_band（只违反 3 % 带） | 0.109 / 0.129 | 0.074 / 0.074 |
| 其中 edge_erosion | 0.076 / 0.063 | 0.056 / 0.051 |
| 返回落在本窗最近返回严格带内的比例（栅格纯度） | 0.51 [0.31, 0.64] | 0.70 [0.45, 1.00] |
| 3×3 窗违反宽松规则的比例 | 0.148 [0.018, 0.378] | 0.047 [0.000, 0.290] |
| 返回距离中位 m | 6.9 [3.7, 20.5] | 5.6 [1.8, 15.8] |

### 3.2 门扇 / 砾石 ROI（`selection.json.roi_in_crops`，每区 63 个面视角）vs 整面

| ROI 内 | 室内门扇 | 室外砾石 |
|---|---|---|
| 当前支持 / RGB | 0.996（中位 1.000） | 0.912（中位 1.000） |
| 严格支持 / RGB | **0.045**（中位 0.021 [0.001, 0.099]） | **0.865**（中位 0.929 [0.722, 0.988]） |
| 被拒 / 当前 | **0.955**（中位 0.979） | **0.051**（中位 0.017） |
| discontinuity / interior_band / erosion | 0.827 / 0.086 / 0.042 | 0.0005 / 0.035 / 0.015 |

读法：砾石是平坦的"内部"表面，严格规则几乎不动它（被拒 5 %，且全是斜面带与边缘外扩，没有一个真断边）；门扇是"门窗/薄结构边"的极端：ROI 内当前 100 % 支持，严格只剩 2–5 %，因为窗内返回同时含门扇、门框、玻璃后的走廊/室外（`indoor_…__02__img_88226fd0…__roi.png` 第 1 面板：门扇范围内的返回密度与墙面一样高，第 4 面板整块红 + 门框条纹橙/紫）。03 号报告已指出门窗格中心的 LiDAR 中位距离比门扇远 2 m，这里给出它在 alpha 支持上的后果：**训练器在门扇的每个像素上都要求 alpha ≥ 0.95，包括其实透过玻璃看到远处的像素，以及多副本门扇造成的相互矛盾的深度**。

### 3.3 按面与相机（像素加权，被拒 / 当前）

| 面 | 室内 当前→严格 | 室内被拒 | 室外 当前→严格 | 室外被拒 |
|---|---|---|---|---|
| `pitch_down_56`（地面） | 0.520→0.462 | **0.111** | 0.595→0.589 | **0.010** |
| `pitch_up_56`（天花/树冠/屋檐） | 0.853→0.163 | 0.809 | 0.792→0.186 | 0.765 |
| `yaw_minus_35` | 0.930→0.237 | 0.746 | 0.889→0.436 | 0.509 |
| `yaw_plus_35` | 0.910→0.199 | 0.782 | 0.865→0.467 | 0.460 |

相机：室内左 0.991→0.065（被拒 0.934）、右 0.844→0.230（0.728）；室外左 0.800→0.416、右 0.778→0.431。室内左相机的 6 个 DIAG 父图全是远距离（~7 m）看门的视角，窗口在 7 m 处横跨 9 cm，混入更多层。

朝下的地面面几乎不受影响（室内 11 %、室外 1 %），说明严格规则本身不"误杀"连续表面；被拒集中在 pitch_up 与两个 yaw 面，即天花板/树冠/屋檐、门窗、家具边界——正是任务书 §8.3 点名的"门窗、细枝、遮挡边界"。

### 3.4 栅格本身：vis6 名不副实

| 图 | 区 | v8 鱼眼缓存 3×3 窗违反宽松规则 | 施加 cell-6 过滤后 | 过滤会丢的返回 |
|---|---|---|---|---|
| `img_02167f65` | 室内 | 0.535 | 0.000 | 63.7 % |
| `img_d691623e` | 室内（DIAG U1 视角） | 0.215 | 0.000 | 29.6 % |
| `img_05cc6ae9` | 室内 | 0.225 | 0.000 | 32.8 % |
| `img_03f32cc8` | 室外（U1 视角） | 0.127 | 0.000 | 20.3 % |
| `img_39a2d47a` | 室外（U0 视角） | 0.262 | 0.000 | 38.2 % |

证据链：`face4_lidar_train_vis6/face_lidar_geometry_manifest.json` 写着 `projection = kb4_forward_splat_nearest_range_zbuffer_visibility_filtered`、`intermediate_fisheye_raster = true`、`source_depth_manifest_sha256 = af8fb540…`；后者是 `house0305_sop_v8/depth/depth_manifest.json`（`algorithm_version kb4_ray_zbuffer_v1`，`projection` 只有 min/max range、theta、confidence 五个键）；`tools/build_face4_lidar_geometry.py` 的非 direct 分支（`warp_sparse_depth_to_face` 之后）只做 `keep &= finite & 0 < conf ≤ 1`，`visible_point_mask` 只在 `project_camera_points_to_face`（direct 分支）与 `project_lidar_depth`（鱼眼缓存构建）里被调用。上表"施加过滤后"是用 `DepthProjectionConfig(visibility_cell_px=6)` 对缓存自身像素调用 `visible_point_mask` 得到的（scratch 探针，未写入任何缓存）。

后果：(a) 03 号报告 §2 的 loose/strict 有效视角计数是在未过滤栅格上算的，"strict 遮挡比 0.77"里一部分是穿透返回造成的假遮挡；(b) 当前 alpha 地板在室内把穿透返回也当支持；(c) range 监督（交付臂 `lidar_range_weight 0.05`，诊断臂为 0）同样在消费它。修 builder 并重建栅格会改变 `face_lidar_geometry_manifest_sha256` → 训练器身份，属于新臂。

## 4. 旋钮与 D0/D1 臂

### 4.1 `lidar_alpha_support_mode: "dilated" | "strict_visibility"`

* 字段 `trainer.py:282`（默认 `"dilated"`），`from_dict` 键 `:548`，校验 `:907-918`（未知值拒绝；非默认模式要求 `lidar_alpha_weight > 0`），vendor pre-optimizer 生命周期门 `:1394-1402`（非默认模式只允许在 `surface_alpha_floor_profile` 下，与 dilation 的放行口径一致）。
* 合同 `:2110-2135`：默认模式下 `loss_contract.lidar_alpha_coverage` **键集合不变**（`enabled/source/target/dilation_radius_px/loss`），所以所有既有配置的 `trainer_config_sha256` 逐字节不变；`strict_visibility` 时 `source = signed_lidar_depth_mask_strict_visibility`，并加 `support_mode` 与 `strict_visibility = {search_radius_px, tolerance 0.03, margin_m 0.1, edge_erosion_px 3, rule}`。`run_manifest.json.trainer_contract` 与 `trainer_config_sha256` 随之变化。
* 损失 `:2996-3006` 调用 `alpha_support.lidar_alpha_support(...)`，从已加载的 `range_m / confidence / depth_mask` 张量现算，**无新缓存**；`range_m` 张量只要 `depth_range_m` 存在就一直在（`_tensor_sample`）。

### 4.2 D0 / D1 精确差分（DIAG_40）

D0 = 仓库现有 `run_configs/house0305_tiles/diag_v2/diag_<region>_40_R1_ci.json`（未改动），D1 = 同一 JSON 只加一行 `"lidar_alpha_support_mode": "strict_visibility"`（run_id / output_dir 相应改名，例如 `…_40_D1-ci`）。两者都通过 `TrainerConfig.from_dict(...).validate()`（含 manifest / PLY / npz 哈希与几何↔输入绑定）。

| 区 | D0 `trainer_config_sha256` | D1 `trainer_config_sha256` |
|---|---|---|
| indoor_door_leaf_Tile_1 | `e916cd4f6b2be361c574ec09a016adc072255de88f9f47b9aeb7caad99840ab2` | `4cab9b9c5ef62c1e3582c5c22b46a4b76466669fdb71f61e1ca11e7730f7f855` |
| outdoor_gravel_Tile_0 | `86d21881a957f80324c4aa8aae9fe7212738198bca82209fe7d5a10ed91fa441` | `2ac3fede93519c101bee7ea0aecbd35813fb3a08f16f605b17347041bcea657c` |

合同扁平化后两区各只有 7 个键不同，全部在 `/loss_contract/lidar_alpha_coverage/` 下：`source`（`…confidence_max_dilated` → `…strict_visibility`）、新增 `support_mode`、`strict_visibility/{search_radius_px 6, tolerance 0.03, margin_m 0.1, edge_erosion_px 3, rule}`。日程、学习率、生命周期、cap、alpha 权重/目标/半径全部相同。

D1 在现有栅格上的实际含义（§3）：室内 alpha 地板覆盖从 RGB 像素的 85 % 降到 22 %（门扇 4.5 %），室外从 79 % 降到 42 %（砾石 86.5 %，树冠/屋檐几乎归零）。判据建议：若 D1 室内锐度/ROI 指标相对 D0 改善而室外砾石不变，说明过度外扩的支持在室内确实有害；若室内无差别，则 alpha 支持不是室内糊的主因，注意力应回到栅格纯度与 03 号报告的其它假设。两种结果都不能替代过滤后栅格上的重跑。

`run_configs/` 不在本任务的写范围内，D1 配置未落盘；上面差分由 scratch 脚本在内存中生成并校验。

## 5. 测试与验证

* 新增 `tests/test_alpha_support.py`：9 项通过（`.venv-train` python，`CUDA_VISIBLE_DEVICES=""`，CPU）。系统 python 无 torch 时 4 项 torch 用例跳过、5 项 numpy 用例仍跑。
* 触及的既有套件（同一环境）：`tests/test_training.py`、`test_vendor_parity_lifecycle.py`、`test_surface_only_route.py`、`test_diagnostic_set.py`、`test_schedule_contract.py`、`test_shape_knob_plumbing.py`、`test_training_presets.py` 合计 **134 通过 / 11 失败**：
  * `test_training_presets.py` 3 项——本机已知失败（任务说明），未处理；
  * `test_training.py::RenderScaleContractTests::test_rendered_footprint_matches_linear_metric_scale` 1 项——需要 GPU 渲染，在 `CUDA_VISIBLE_DEVICES=""` 下 `Invalid device id`；与本改动无关，**GPU 门禁未验证**；
  * `test_diagnostic_set.py::GeneratedConfigsOnDiskTests::test_repo_diag_configs_satisfy_contract` 7 个子用例——`*_ci.json` 文件名 stem 用下划线而 `run_id` 用连字符（`…R1_ci` vs `…R1-ci`），是另一任务生成 `_ci` 配置时留下的既有不一致，与本改动无关（断言只比较 stem 与 run_id）。
  * `test_locked_patch_runtime_is_verified_before_importing_runtime` 被 deselect（需要干净的 gsplat 检出验证，本机训练环境有本地补丁）。
* 工具运行：264 视角 539 s，CPU；两区各 5 视角 torch 交叉核对全一致。
* 未做：任何训练、任何缓存重建、任何 GPU 使用。

## 6. 残留与注意

* 严格规则用同一 13×13 窗，仍是启发式；在 7 m 处窗宽 9 cm，斜面/曲面（`interior_band`）在室内占被拒的 15 %。若要减少误杀，应改为局部平面拟合或按距离缩放窗口，不在本轮范围。
* 严格模式**不区分"近表面"与"远表面"**，分歧窗整体剔除；杆状细结构窄于窗宽时在严格模式下得不到任何 alpha 支持（由光度项接管）。
* `edge_erosion_px = 3` 是常量而非旋钮，写进合同；改动它就是另一臂。
* 栅格未过滤是更上游的缺陷；本文所有"当前 vs 严格"数字都是在这张栅格上测得，过滤后重测预期：discontinuity 类被拒大幅下降、interior_band 与 erosion 类基本不变（§3.4 表中过滤后 r=6 的严格带违反率仍有 17–32 %）。
* 叠加图目录 48 MB、未提交；CSV 以 `sample_id` 为键可与 `03_observation_coverage.csv`、`selection.json` 关联。
