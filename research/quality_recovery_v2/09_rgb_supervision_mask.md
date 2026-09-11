# 09 — F-line: `rgb_supervision_mask` 光度归属遮罩（2026-09-11）

状态：旋钮、契约、门禁拒签、测试、两区 F1 配置与 CPU 监督像素占比估计已完成；**F1 尚未训练**。本文记录机制、契约键、占比表与判读口径。所有数字来自 `09_supervision_fraction/*.json`（本目录，`tools/estimate_rgb_supervision_fraction.py` 生成），CPU 只读计算，未动缓存。

## 1. 动机（README 17:15 行）

DIAG-40（H=3000）室内 R1/D1f 的失败形态是墙面与门叶前的**半透明树色/天空色悬浮团**，不是模糊。切片训练的逐视角背景库（`tile_backgrounds_v9/Tile_*`）只从天空穹顶渲染（`dome_source: probes/sky_house0305.pt`），损失是 `final = render + (1 − alpha) · backdrop`。于是裁剪内凡是照片有内容、切片没有几何的像素（pitch_up 面里的树、屋檐外远景、相邻切片的东西）只能由切片自己的高斯去"画"——它们落在任意深度并遮挡其他视角。G/X/D 三线中性与此一致：它们都没碰这个监督。

F-line 的最小验证：把光度监督限制在切片"能合法拥有"的像素上，其余像素零梯度、交给背景。

## 2. 机制：旋钮做了什么

`TrainerConfig`（`cloudstudio_3dgs/training/trainer.py`）新增两个字段：

| 字段 | 取值 | 缺省 |
|---|---|---|
| `rgb_supervision_mask` | `all` \| `lidar_support` | `all`（字节不变） |
| `rgb_supervision_dilation_radius_px` | int，[0, 64]；模式非 `all` 时必须 > 0 | 24 |

构造（`cloudstudio_3dgs/training/rgb_supervision.py`，torch 与 numpy 双胞胎，测试互钉）：

```
valid      = depth_mask & isfinite(confidence) & (confidence > 0)      # 签名的面 LiDAR 回波
support    = max_pool2d(valid, kernel 2r+1, stride 1, pad r) > 0       # 与 dilated alpha 支持同一算子
supervised = rgb_mask & support                                        # rgb_mask = 渲染器 mask ∧ 行人 ∧（若开）切片归属
```

`all` 模式直接把 `tensors["rgb_mask"]` 本身（同一对象）交给下游，损失路径与旧代码逐算子相同。`lidar_support` 模式下：

| 损失项 | 处理 | 归一化 |
|---|---|---|
| RGB L1 | 只在 `supervised` 上算 | 监督像素数（`masked_rgb_l1` 的 mean 本来就是按 mask 像素数） |
| SSIM（`local_gaussian`） | 传入 `supervised` 作 mask：mask 外像素先清零再卷积，只有中心有效且覆盖 ≥ `ssim_min_valid_fraction` 的窗贡献；**mask 外像素梯度恒为 0** | 有效窗数；稀疏 mask 可能一个窗都不够覆盖 → 新增 `allow_empty=True` 返回接图的 0 而非抛错 |
| SSIM（`global_moments`） | 传入 `supervised` | mask 像素数 |
| RGB 梯度 L1（权重 0 时不算） | 传入 `supervised` | 有效邻对数 |
| LiDAR 支持 RGB L1（`lidar_rgb_l1_weight`，本配置 0） | 原本 `rgb_mask & dilate(depth_mask)`，现在再 ∧ `supervised` | 不变 |
| **DA2 深度项** | `da2_valid = da2_mask & finite & >0` 再 ∧ `supervised` | 不变（confidence 全 1 的加权 L1） |
| 训练视角 PSNR（遥测） | 同 L1 的像素 | — |
| LiDAR range / alpha 地板 / mesh 项 | **不动**（本来就只在 LiDAR 支持上） | — |
| 曝光 gain | **不动**：`rendered = rendered * gain` 仍在遮罩之前、按视角整体应用；gain 的梯度只来自监督像素 | — |

关于 DA2 的决定：DA2 项不是光度项（比较的是渲染射线距离），但它的目标是**从照片预测**的单目深度、按该视角 LiDAR 回波做 scale/shift 对齐——在 LiDAR 支持之外它是对切片没有几何的内容的外推。若只遮 RGB 不遮 DA2，深度项会继续要求"这里该有一个面"而颜色又无约束，等于半截措施；所以 DA2 与光度项一起遮。契约里的 `masked_terms` 明确列出。若 F1 有效而需要拆因，再加独立开关（F2）。

空支持视角：`lidar_support` 下若某视角一个监督像素都没有（无回波的面，或裁剪内无回波），L1/SSIM/梯度项返回 `rendered.sum() * 0`（接在图上、梯度为零），`rgb_psnr = None`，`rgb_supervised_fraction = 0`。`all` 模式保留损失函数原有的空 mask fail-closed 报错。

遥测：每步 progress 记录多两个键 `rgb_supervised_fraction`（= 监督像素 / rgb_mask 像素，`all` 下恒 1.0）与 `rgb_supervised_pixels`。

## 3. 契约与门禁

`contract_dict()["loss_contract"]` **只在模式 ≠ `all` 时**多一个键（默认配置的契约与 `trainer_config_sha256` 逐字节不变，测试钉住；`all` 下改半径也不进契约）：

```json
"rgb_supervision_mask": {
  "mode": "lidar_support",
  "source": "signed_lidar_depth_mask_confidence_max_dilated",
  "dilation_radius_px": 24,
  "combined_with": "rgb_mask & dilated_support",
  "masked_terms": ["rgb_l1", "rgb_ssim", "rgb_gradient_l1", "lidar_rgb_l1", "da2_depth"],
  "normalisation": "supervised_pixels",
  "empty_support_policy": "zero_photometric_loss_for_view",
  "exposure_gain": "applied_unchanged_before_masking"
}
```

校验：模式非 `all` 时必须有 LiDAR 输入（`depth_manifest/depth_root` 或 `face_lidar_geometry_manifest/root`），半径必须 > 0。

门禁：`pipeline/mipmap_gate.py::advance_adaptive_growth_gate` 与拒签 `schedule_contract` 同一位置、同一措辞拒签 `rgb_supervision_mask != "all"`（"research departure and is recorded by the trainer, not this gate"）；签名仍先校验。实现/表面冻结门禁只绑数据身份，不受影响。

## 4. 测试

`tests/test_rgb_supervision_mask.py`（8 项，CPU 合成样本）：默认路径与内联 oracle 逐位相等、支持覆盖全 rgb_mask 的 `lidar_support` 与默认逐位相等；遮罩外像素梯度精确为 0、L1 按监督像素归一；DA2 项被同一遮罩截断；空支持视角零损失可反传；numpy/torch 双胞胎与逐像素 oracle 一致；契约只在非缺省多键；校验拒绝无 LiDAR 输入 / 半径 0 / 未知模式；自适应增长门禁拒签（显式 `all` 仍可签，篡改仍先报签名错）。

回归（本机 CPU，`.venv-train`）：`tests/test_schedule_contract.py` + `test_alpha_support.py` + `test_exposure_curve.py` 改前 49 通过、改后 49 通过；`test_training.py`、`test_training_presets.py`、`test_densification_gradient_source.py`、`test_mipmap_loss_schedule.py`、`test_diagnostic_set.py` 与新模块合跑 149 通过 / 13 失败，13 个失败与改动无关且在干净 HEAD 导出树上同样失败：1 个需要 gsplat CUDA 扩展（本机无 CUDA）、3 个 `test_training_presets`（预设 `geometry_regularization` 不匹配，以及 `test_legacy_preset_executes_global_masked_moment_loss` 的假 Backend 本来就没有 `.torch`）、9 个 `test_diagnostic_set` 在盘配置子测试（既有配置的 `run_id` 用 `-ci`/`-c134` 连字符而非文件名）。两份 F1 配置的在盘子测试通过（16 → 18 subtests passed）。

## 5. 两区 F1 配置

由 `diag_<region>_40_R1_c134.json` 派生（`C:\Peter\3dgs-runs\house0305_sop\diag_<region>_40_F1_c134.json`，副本在 `run_configs/house0305_tiles/diag_v2/`），与 R1 的差异只有：`run_id`（= 文件名 stem）、`output_dir`（`diag_v2/<region>/runs/<stem>`）、`rgb_supervision_mask: lidar_support`、`rgb_supervision_dilation_radius_px: 24`、`diag.variant: F1_c134`、`diag.supervision_variant`（记 base 臂 sha、base run_id、两个字段的 base/diag 值与说明）。缓存仍是 R1 的 `face4_lidar_train_vis6`（未过滤），对照 R1 时只差契约里 `loss_contract.rgb_supervision_mask` 一个键（脚本断言）。两份都过 `TrainerConfig.from_dict(...).validate()`。

## 6. 监督像素占比估计（CPU，真实缓存）

`tools/estimate_rgb_supervision_fraction.py`：对 DIAG_40 的每个视角（tile_inputs 裁剪）读渲染器 mask PNG 与面 LiDAR npz，按训练器同一顺序（先裁剪再膨胀）算 `supervised = rgb_mask & max_pool(valid, r)`，报告 supervised / rgb_mask。ROI 用 `selection.json` 的 `roi_in_crops` 框。缓存 = R1/F1 配置的 vis6。

### 6.1 半径 24（F1 配置）

| 区 | 范围 | n | 中位 | 最小 | Q1–Q3 |
|---|---|---|---|---|---|
| 室内门叶 Tile_1 | 全部视角 | 113 | 0.993 | 0.891 | 0.979–1.000 |
| | pitch_up | 40 | 0.963 | 0.891 | 0.943–1.000 |
| | yaw | 66 | 0.998 | 0.967 | 0.987–1.000 |
| | pitch_down | 7 | 0.994 | 0.987 | 0.989–0.999 |
| | ROI（全部） | 63 | 1.000 | 0.997 | 1.000–1.000 |
| | ROI pitch_up / yaw | 13 / 50 | 1.000 / 1.000 | 1.000 / 0.997 | — |
| 室外砾石 Tile_0 | 全部视角 | 151 | 0.994 | 0.537 | 0.944–1.000 |
| | pitch_up | 36 | 0.941 | 0.537 | 0.891–1.000 |
| | yaw | 79 | 0.988 | 0.646 | 0.944–0.996 |
| | pitch_down | 36 | 1.000 | 0.973 | 0.999–1.000 |
| | ROI（全部） | 63 | 1.000 | 1.000 | 1.000–1.000 |
| | ROI pitch_down / yaw | 18 / 45 | 1.000 / 1.000 | 1.000 / 1.000 | — |

像素加权保留：室内 **0.980**、室外 **0.960**；没有任何回波的视角 0 个。

**判读**：半径 24 下 F1 几乎什么都没遮。原因是两条代码事实：(1) 面 LiDAR 缓存是**全场景**投影（`source_point_cloud_points` 625 万，不按切片框裁），树、屋檐、邻切片的墙都有回波——"LiDAR 支持"≠"切片拥有"；(2) 回波密度中位 14%（每 7 个像素 1 个回波），49×49 的窗把一切回波 24 px 内的像素都盖住。真正被遮的只有**天空与无回波区**：室外天空帧 `img_2c48::pitch_up` 0.537、`img_03f3::pitch_up` 0.607（README 14:11 行点名的天空烟雾帧），室内三个悬浮体视角 `img_02167` 三面 1.000、`img_8a5782::pitch_up` 0.943、`img_a4bc51::pitch_up` 0.921——**室内烟雾所在的树/屋檐像素基本仍在监督内**。门叶 / 砾石 ROI 两区都 ≥ 0.997，即 F1 不会伤 ROI。

### 6.2 半径敏感性（同缓存）

| 区 | r | 全部中位 | pitch_up 中位 | pitch_down 中位 | yaw 中位 | ROI 中位（最小） | 像素加权 |
|---|---|---|---|---|---|---|---|
| 室内 | 6 | 0.893 | 0.843 | 0.497 | 0.958 | 1.000 (0.985) | 0.851 |
| 室内 | 12 | 0.960 | 0.916 | 0.877 | 0.985 | 1.000 (0.989) | 0.939 |
| 室内 | 24 | 0.993 | 0.963 | 0.994 | 0.998 | 1.000 (0.997) | 0.980 |
| 室外 | 6 | 0.854 | 0.827 | 0.600 | 0.897 | 1.000 (0.615) | 0.790 |
| 室外 | 12 | 0.956 | 0.894 | 0.957 | 0.962 | 1.000 (0.979) | 0.926 |
| 室外 | 24 | 0.994 | 0.941 | 1.000 | 0.988 | 1.000 (1.000) | 0.960 |

缩小半径先砍掉的是 pitch_down（近地面回波稀疏的裁剪）和室外砾石 ROI（r=6 时 ROI 最小 0.615），而不是树/天空——半径不是把"LiDAR 支持"变成"切片拥有"的旋钮。

### 6.3 切片归属拆分（`--tile-ownership`，r=24，vis6）

用训练器 `tile_ownership_masking` 同一函数（`face_dataset.tile_ownership_masks`，训练框 + 0.5 m 边距，foreign 膨胀 15 px；K 按裁剪平移，`c2w = c2w_base @ [R_face]`）把每个视角的回波分成切片内 / 切片外：

| 区 | 量 | 全部中位（最小） | pitch_up 中位 | yaw 中位 | pitch_down 中位 | ROI 中位（最小） |
|---|---|---|---|---|---|---|
| 室内 Tile_1 | 回波在切片框内的份额 | 0.449 (0.128) | 0.362 | 0.508 | 1.000 | — |
| | foreign 区（膨胀 15 px）占 rgb_mask | 0.359 | — | — | — | — |
| | 监督：`lidar_support` ∧ ¬foreign（F1 + 现成 `tile_ownership_masking`） | 0.605 (0.243) | 0.489 | 0.640 | 0.994 | 1.000 (0.908) |
| | 监督：只膨胀框内回波（假想 F2 构造） | 0.639 (0.224) | 0.523 | 0.689 | 0.994 | 1.000 (0.913) |
| 室外 Tile_0 | 回波在切片框内的份额 | 0.827 (0.155) | 0.485 | 0.681 | 1.000 | — |
| | foreign 区占 rgb_mask | 0.085 | — | — | — | — |
| | 监督：`lidar_support` ∧ ¬foreign | 0.861 (0.313) | 0.721 | 0.814 | 1.000 | 1.000 (0.910) |
| | 监督：只膨胀框内回波 | 0.860 (0.309) | 0.736 | 0.819 | 1.000 | 1.000 (0.928) |

逐视角（室内三个悬浮体视角，`sup` = F1 r=24 保留，`¬foreign` = F1 ∧ 切片归属，`owned` = 只膨胀框内回波）：`img_02167::pitch_up` 1.000 / 0.698 / 0.721（框内回波 43%）、`img_8a5782::pitch_up` 0.943 / **0.323** / 0.329（框内回波仅 22%——就是 `figures/backdrop_vs_photo_pitchup.jpg` 那张树/屋檐面）、`img_a4bc51::pitch_up` 0.921 / 0.598 / 0.616；`img_26e10`（墙板比参考还锐的视角）四面 ¬foreign 0.63–0.998。室外天空帧 `img_2c48::pitch_up` 0.537 / 0.350 / 0.314，`img_03f3::pitch_up` 0.607 / 0.429 / 0.406，近砾石 `img_9294::pitch_down` 1.000 / 1.000 / 1.000。

**判读**：室内 DIAG-40 视角里中位只有 45% 的回波属于 Tile_1 自己（pitch_up 36%）；把切片外回波的 15 px 邻域也遮掉后，监督面积从 0.98 落到 **0.61**（pitch_up 0.49），而门叶 ROI 仍 ≥ 0.91（中位 1.000）。室外切片框大得多（y 跨 71 m），回波 83% 在框内，监督只落到 0.86，砾石 ROI ≥ 0.91。也就是说，"切片能合法拥有"的像素集合是由**切片归属**而不是 LiDAR 支持决定的：F1（只用 LiDAR 支持）遮的是天空，F1 ∧ `tile_ownership_masking: true` 才会遮到室内烟雾所在的树/屋檐像素。两个旋钮都已存在、可直接组合（`tile_ownership_masking` 在数据集侧改 rgb_mask，本估计用同一函数与训练器缺省 0.5 m / 15 px 复现）。

### 6.4 修复缓存 vis6f（隐藏点剔除后，r=24）

vis6f（`face4_lidar_train_vis6f`，剔除 42.6% 穿透回波）在 r=24 下与 vis6 **逐项相同**：室内全部中位 0.992（vis6 0.993）、pitch_up 0.963、像素加权 **0.980**；室外全部 0.994、pitch_up 0.941、像素加权 **0.960**，ROI 两区仍 ≥ 0.997。隐藏点剔除砍的是回波，不是回波所在的区域——24 px 的膨胀把剩下的回波又铺满了同一片面积。换缓存不改变 F1 的遮罩面积（`indoor_vis6f_r24.md`、`outdoor_vis6f_r24.md`）。

## 7. F1 怎么判

F1 对 R1_c134 的配对读数，口径固定为（README 16:13 行起）：

1. **亮度归一 ROI-all ours/ref**：`tools/build_three_way_compare.py --sample-ids` 渲染 `roi_compare_ids.json`（室内 49 / 室外 45 视角），`tools/score_compare_roi.py --match-brightness --selection DIAG_40/selection.json`；对照表 `07_diag40_roi_scores.md` 第二表（R1 室内 0.881 / 室外 0.595，噪声带 ±5%）。ROI 两区在 F1 下保留 ≥ 0.997 的监督，ROI 读数**不应**因遮罩下降；若下降即遮罩伤了正常监督（看 SSIM 窗覆盖）。
2. **离轨 PSNR**（18 帧，R1 室内 14.79 / 室外 15.16）：这是对悬浮体最敏感的数字——挡在墙前的团在新视角上错位。判赢的门槛 ≥ +0.3 dB（X1 的 −0.4 dB 曾被判为真变差）。
3. **出图**：`figures/DIAG40_indoor_R1_views.jpg` 同六视角（`img_02167` / `img_8a5782` / `img_a4bc51` 三视角为主）的 R1 vs F1 并排，看烟雾是否消失、墙面/门叶是否露出背景色块（遮罩外像素由背景库补，背景库本身是糊穹顶，pitch_up 面的树区若变成一片蓝棕色块是**预期**而非缺陷）；室外看 `img_2c48` / `img_03f3` pitch_up 的天空烟雾。放大条（zoom sheet）沿用 `build_three_way_compare` 的裁剪。
4. 训练遥测：`rgb_supervised_fraction` 逐步均值应与 §6.1 的像素加权值一致（室内 ≈ 0.98、室外 ≈ 0.96）；若明显偏低说明数据集侧 rgb_mask 与本估计不同（例如行人 mask），先对账再判。
5. 形态分位（short p50、opacity p50、frac<0.1）与终态高斯数与 R1 对齐，排除 cap/生命周期混入。

**预期与后手**：按 §6.1，r=24 的 F1 只对天空/无回波像素零梯度，对室内烟雾所在像素几乎不动，所以更可能的结果是"室外天空烟雾减轻、室内悬浮体原样"。若如此，F1 不是反证假设，而是反证"LiDAR 支持"这个近似——下一步是 F2：`lidar_support` ∧ 切片归属（现成旋钮 `tile_ownership_masking: true`，§6.3 给出它会砍掉多少），或在 rgb_supervision 里直接用"切片框内回波"的膨胀作支持（§6.3 的"只膨胀框内回波"一行即其占比，尚未实现为训练器旋钮；两者在 DIAG-40 上占比几乎相同，先用现成旋钮）。建议 GPU 排程上 F1 与 F1+归属（`F2 = F1 + tile_ownership_masking: true`，配置未生成）背靠背，各 ≤ 30 min。

## 8. 残留风险

- 估计工具不复现行人 mask 与 `tile_ownership_masking` 对 rgb_mask 的修改（R1 两者均未开/无行人）；vis6 缓存含穿透回波，其"支持"比真实可见表面宽（05_vis6_cache_fix）。
- `lidar_support` 下评估路径（`_compare_pose_candidate`、验证损失）同样遮罩；离轨 PSNR 由评估器独立算，不受影响。
- 稀疏 mask 可能让 SSIM 没有一个覆盖 ≥ 0.8 的窗（`allow_empty` 返回 0）：F1 两区 ROI 保留 ≥ 0.997，风险只在 pitch_up 天空边缘。
- 未训练；本文所有关于 F1 效果的说法都是预期。
