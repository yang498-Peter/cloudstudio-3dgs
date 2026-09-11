# 10 — S-line: `surface_anchor_prune` 表面锚定剪枝（2026-09-11）

状态：测量、旋钮、契约、门禁拒签、测试、审计工具、三份 S1 配置已完成；**S1 尚未训练**（本机 CPU-only，未提交）。所有数字来自 `10_surface_anchor/*.json`（本目录，`tools/audit_surface_anchor.py` 生成），CPU 只读计算，未动任何 checkpoint / 缓存。

## 1. 测量：高斯离 LiDAR 表面有多远

`tools/audit_surface_anchor.py --config <臂配置>`：读 `checkpoints/latest.pt`（CPU）、配置里的 `initialization_ply`（切片输入签名的全 LiDAR 初始化云）与 `tile_inputs_manifest` 里该 Tile 的 `training_and_export_box`，对每个高斯算到初始化云最近点的**精确欧氏距离**（scipy `cKDTree`，k=1），报告超过 0.05 / 0.1 / 0.2 / 0.5 / 1.0 m 的份额（全部 / opacity ≥ 0.05）、距离分位、按 0.2 m 切开的 z 分位、框外份额。

| checkpoint | 步 | 高斯数 | 锚点数 | >0.05 | >0.1 | **>0.2** | >0.5 | >1.0 | 距离 p50/p90/p99 (m) | z p50 近/远 (m) | 框外 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `tile1_R1d_20k` (Tile_1 全视角) | 20000 | 3,336,404 | 3,417,320 | 0.475 | 0.397 | **0.338** (可见 0.340) | 0.268 | 0.200 | 0.041 / 2.29 / 6.29 | 0.37 / **2.46** | **0.353**（框外且远 0.267） |
| `diag_indoor_door_leaf_Tile_1_40_R1_c134` (H=3000) | 3000 | 4,421,950 | 3,417,320 | 0.142 | 0.106 | 0.070 (可见 0.066) | 0.027 | 0.007 | 0.005 / 0.11 / 0.87 | 0.96 / 14.70 | 0.061 |
| `diag_outdoor_gravel_Tile_0_40_R1_c134` (H=3000) | 3000 | 9,292,725 | 7,044,777 | 0.035 | 0.021 | 0.011 | 0.002 | 0.000 | 0.004 / 0.013 / 0.21 | 0.40 / 5.09 | 0.009 |

读法：

* tile1 全视角 20k 模型确认了 README 晚间行的数字：**34% 的高斯离最近 LiDAR 点 > 0.2 m，27% > 0.5 m，20% > 1 m**，且这部分坐得高（z p50 2.46 m，近表面组 0.37 m），35% 在切片训练框之外——就是 viewer 里看到的悬浮体与檐口/天空外凸。opacity ≥ 0.05 的子集份额几乎一样（0.340），说明它们不是死质量，是在渲染的。
* DIAG-40 短日程（H=3000）两区远得多的份额低一个量级（室内 7.0%、室外 1.1% > 0.2 m），但室内远组的 z p50 已到 14.7 m（框顶 33.9 m，天空方向）——远高斯在 3000 步已经出现在同一位置，只是数量还没长起来；tile1 的 34% 是 20k 步 135 次增殖事件累积的结果。这决定了 S1 在 DIAG-40 上的读数会比全视角臂温和（§6）。
* 距离是到**初始化云**的距离，不是到"真实表面"：LiDAR 没回波的地方（玻璃、细杆、扫描盲区、扫描时在动的物体）本来就没有锚点，§7 记风险。

代价（本机 14 线程 CPU，`cKDTree` 3.42M 点建树 1.2 s）：`query(k=1, distance_upper_bound=0.3)` 3.34M 点 **1.22 s**（0.37 µs/点），平铺到 16.7M 点 **6.4 s**；无上界的完整距离（审计用）2.75 µs/点、3.34M 点 9.2 s。训练器里 `LidarNormalAlignment` 已经在每个 refine 事件后对全人口做一次同样的 CPU `cKDTree` k=1 查询（`LidarNormalAnchors.refresh`，`means.detach().cpu()`——**它不是 GPU 查询**，任务书的假设有误），所以 S1 每个 cull 事件多付的是同一量级的一次有上界查询（cap 15M 时 ≈ 6 s，100 步一次），加上出生守卫对候选父本（几千到几十万行）的一次小查询。GPU 体素哈希版本评估后暂不做：不带 padding 需要变长 gather kernel；带 padding 时 0.3 m 格子在 1 cm 间距的 LiDAR 面上要装 ~900 点，200k 格子就是 GB 级。

## 2. 机制：旋钮做了什么

新模块 `cloudstudio_3dgs/training/surface_anchor.py`（numpy 核心 + torch 桥；`nearest_surface_distance` 就是实现本身，审计工具与训练器调同一函数），`TrainerConfig` 新增字段 `surface_anchor_prune`：

| 字段 | 取值 | 缺省 | 含义 |
|---|---|---|---|
| `enabled` | bool | `false`（字节不变） | 总开关 |
| `max_distance_m` | >0 | 0.3 | 到初始化云最近点的距离超过它 = 无支持 |
| `start_step` | ≥0 | 0 | 首个允许剪枝的步；S1 设在首次 opacity reset 之后的第一个 cull 事件 |
| `every` | int \| null | null | 额外节奏：为 null 只在 cull 事件剪；设了则整除该值的步也剪（用于 `refine_stop_iter` 之后） |
| `min_age_steps` | ≥0 | 0 | 出生不足这么多步的行豁免（clone 落在父本上、split 偏移是父本尺度的一部分，出生当下的距离就是父本的距离） |
| `outside_box` | `keep` \| `prune` | `keep` | `prune`：切片 `training_and_export_box` 各面外扩 `max_distance_m` 之外的行也剪 |
| `reject_unsupported_parents` | bool | `false` | 增长候选父本若 > `max_distance_m` 则不得 clone/split |

作用点都在既有的经典生命周期里（`DefaultStrategyAdapter`，仅 `exact_mipmap_lifecycle` 路径；`validate()` 要求 `densification_strategy = default_3dgs` + `exact_mipmap_lifecycle`，否则拒绝而不是静默无效）：

1. **cull 事件**（refine 步的 `_prune_mipmap`、`post_refine_cull`）：`step ≥ start_step` 时对全人口做一次有上界查询，`far = 距离 > max_distance_m`，`outside = 框外（若 prune）`，`candidates = far | outside`，去掉 `age < min_age_steps` 的行，再去掉本事件 opacity/尺寸 cull 已经选中的行（归因唯一），**并进同一个 `remove` 调用**——Adam 动量、`grad2d/count/radii`、`_cloudstudio_birth_step/kind` 血统列与既有 cull 走完全相同的重索引路径（测试钉住 `exp_avg` 与血统对齐）。
2. **额外节奏**（`every`）：非 cull 步上单独 `remove`，事件 `kind = surface_anchor_prune`；`pre_optimizer_vendor` 顺序下该步的梯度按既有 provenance 机制重映射（测试钉住），与 `post_refine_cull` 同步时不重复剪。
3. **出生**：`_grow_mipmap` 里在梯度/opacity 门之后、cap topk 之前，只查询候选父本行，远父本从 `eligible` 里去掉。这是**纯父本遮罩**，不搬动新生儿，所以没有 `tangent_proposal.reject_unsupported_births` 那种对新生儿尾部布局的依赖——现有守卫在 `pre_optimizer_vendor` 下抛错是因为它要按 post-optimizer 布局改写新生儿位置；S1 **不需要**换执行顺序，配置保持 R1d 的 `pre_optimizer_vendor`（单一改动臂）。任务书写"像现有守卫一样禁止 vendor 顺序、S1 走 post_optimizer_gsplat"，这里有意偏离并记录：多改一个顺序会把 G0 那条"顺序中性"的结论也押进 S1 的判读里。

不做的事：不动 opacity、不搬高斯、不新增高斯；近表面但透明/过大的行仍交给既有 cull。`lifecycle_dry_run` 下不剪、不计数。

遥测：每个事件 `last_lifecycle_event["surface_anchor_prune"]` = `{population, candidates, far_count, far_fraction, outside_count, protected_young, already_removed_overlap, pruned_far, pruned_outside, pruned, remaining, pruned_total}`；`cull_reasons.surface_anchor_count`、`growth_diagnostics.surface_anchor_rejected_count`。progress 记录新增 `surface_anchor_pruned_total`（累计，存于 strategy state 的普通 int，随 checkpoint 续训）、`surface_anchor_far_fraction`（剪之前 > `max_distance_m` 的份额）、`surface_anchor_pruned_event`、`surface_anchor_rejected_parents`。run manifest `densification.resolved.surface_anchor_prune`（无此机制时为 null，与 `surface_birth_guard` 同款）。

## 3. 契约与门禁

`contract_dict()["strategy"]` **只在 `enabled` 时**多一个键（默认配置与显式 `enabled: false` 的契约逐字节相同、`trainer_config_sha256` 不变，测试钉住）：

```json
"surface_anchor_prune": {
  "enabled": true, "max_distance_m": 0.3, "start_step": 700, "every": null,
  "min_age_steps": 100, "outside_box": "prune", "reject_unsupported_parents": true,
  "distance": "exact_nearest_initialization_point_euclidean_m", "box_margin_m": 0.3
}
```

校验：`max_distance_m > 0`、`start_step ≥ 0` 且 `< max_steps`、`every` 为 null 或正整数、`min_age_steps ≥ 0`、`outside_box ∈ {keep, prune}`；`enabled` 要求 `default_3dgs` + `exact_mipmap_lifecycle` + `adaptive_growth`；`outside_box: prune` 要求 `tile_inputs_manifest` + `mipmap_tile_id`。

门禁：`pipeline/mipmap_gate.py::advance_adaptive_growth_gate` 与拒签 `schedule_contract` / `rgb_supervision_mask` 同一位置、同一措辞拒签 `surface_anchor_prune.enabled`（"research departure and is recorded by the trainer, not this gate"）；显式 `enabled: false` 仍可签，篡改仍先报签名错。

`schedule_audit.lifecycle_events` 未改：S1 三份配置 `every = null`，事件表与 R1 相同；若将来设 `every`，审计事件表不会列出额外的锚定剪枝步（已知缺口）。

## 4. 测试

`tests/test_surface_anchor_prune.py`（14 项，CPU）：距离双胞胎 vs 暴力 O(N·M) 逐点相等、有上界查询恰在界外读 inf；框判定含外扩；配置校验拒绝非法值；审计表在合成 checkpoint 上按阈值/可见/框计数；cull 事件剪远留近且 `exp_avg` 与血统列按存活行重索引、遥测归因（`pruned_far`/`pruned_outside`/`remaining`/`pruned_total`）；`start_step` 之前与无该机制逐位相同、后 refine 非节奏步无事件、`state_dict` 为 null；`min_age_steps` 豁免当事件新生儿、下一事件再剪；`outside_box: keep/prune` 对"有锚点但在外扩框外"的行分别留/剪；`every` 在 refine stop 后单独剪且 vendor 顺序下当步梯度重映射、与 `post_refine_cull` 同步不重复；`reject_unsupported_parents` 只让近父本出生；`TrainerConfig` 契约只在 enabled 多键、校验拒绝 MCMC/无 tile 输入/`start_step ≥ max_steps`/非对象/非法 `outside_box`；自适应增长门禁拒签；`tools/audit_surface_anchor.py` 子进程在合成 PLY + checkpoint 上复现表格。

回归（本机 CPU，`.venv-train`）：新模块 + `test_schedule_audit`、`test_schedule_contract`、`test_alpha_support`、`test_rgb_supervision_mask`、`test_post_refine_cull`、`test_default_strategy_adapter`、`test_mipmap_gate`、`test_gaussian_lifecycle`、`test_densification_gradient_source`、`test_diagnostic_set` 合跑 224 通过 + 32 子测试通过、9 个 `test_diagnostic_set` 在盘配置子测试失败为既有问题（09 §4 记录的 `-ci`/`-c134` 连字符 run_id，干净 HEAD 同样失败，与本改动无关；两份 S1 diag 配置的在盘子测试通过）。`test_schedule_audit` 的源码奇偶测试钉住 `_step_post_backward_mipmap` 的 refine-stop 早返回文本，额外节奏因此挂在 `_post_refine_cull` 的两个"不 cull"分支上而不是改早返回。

## 5. 三份 S1 配置

均由 `C:\Peter\3dgs-runs\house0305_sop\` 下的基线派生、通过 `TrainerConfig.from_dict(...).validate()`（含 PLY/npz 哈希与 manifest 绑定检查），副本在 `run_configs/house0305_tiles/`。派生脚本断言：顶层只改 `run_id` / `output_dir` / `surface_anchor_prune` / 溯源块，契约去掉 `strategy.surface_anchor_prune` 后与基线逐字节相同。

| 配置 | 基线 | 契约 sha（基线 → S1） | 溯源 |
|---|---|---|---|
| `tile1_S1_anchor_20k.json`（`run_id house0305-t1-S1anchor`，`v9/`） | `tile1_R1d_20k.json` | `eca85467…` → `9719f342…` | `lineage`（base + 单一改动描述，同 ladder 台账写法） |
| `diag_indoor_door_leaf_Tile_1_40_S1_c134.json`（`diag_v2/`） | `diag_indoor_door_leaf_Tile_1_40_R1_c134.json` | `894ce916…` → `765b8ed5…` | `diag.variant = S1_c134`、`diag.anchor_variant` |
| `diag_outdoor_gravel_Tile_0_40_S1_c134.json`（`diag_v2/`） | `diag_outdoor_gravel_Tile_0_40_R1_c134.json` | `7d14bee4…` → `94c19a0f…` | 同上 |

旋钮取值（三份相同；两套日程的解析事件表在 500–1100 步完全一致：refine 500/600/700…，`reset_every 300` → 首次 reset 在 600）：`max_distance_m 0.3`、`start_step 700`（首次 reset 之后的第一个 cull 事件；600 那次 cull 虽然也在 reset 之后——`reset_before_cull: true`——但那是 reset 事件本身，按任务书取其后一个）、`every null`、`min_age_steps 100`（一个 refine 间隔）、`outside_box prune`、`reject_unsupported_parents true`。tile1 20k 日程下这意味着从 700 到 13900 的 133 个 refine 事件都剪，`refine_stop 14000` 后不再剪（`post_refine_cull_every` 未设、`every` 为 null）；若 20k 终态仍有远高斯，后手是 `every: 100` 让 14000–20000 段继续剪。

## 6. S1 怎么判

配对基线：tile1 用 `tile1_R1d_20k`，DIAG-40 用 `diag_<region>_40_R1_c134`。口径沿用 09 §7，外加本机制专属的第 1 条：

1. **PLY / checkpoint 几何审计**（本机制的直接读数）：训练完对 `latest.pt` 跑 `tools/audit_surface_anchor.py --config <S1 配置>`，与 §1 表配对。判"机制起效"的门槛：tile1 的 >0.2 m 份额从 0.338 落到 ≤ 0.05（0.3 m 阈值 + 100 步豁免下，终态残余应只剩最后一两个 refine 间隔的新生儿），框外份额从 0.353 落到 ≈ 0（外扩 0.3 m 之外为 0，0–0.3 m 带内允许残余），远组 z p50 不再高于近组。同时看遥测 `surface_anchor_far_fraction` 逐事件曲线：健康形态是首次剪枝后骤降、之后每事件 < 5%（新生儿被 100 步豁免后下一事件补剪）；若每事件持续 ≥ 20% 说明增殖在反复往远处生（出生守卫没拦住 split 子代的位移），要看 `surface_anchor_rejected_parents` 是否为零。viewer 里看檐口/天空外凸与墙前悬浮团是否消失，以及**墙面/门叶是否出现洞**（下面第 4 条）。
2. **亮度归一 ROI-all ours/ref**（`build_three_way_compare --sample-ids roi_compare_ids.json` + `score_compare_roi --match-brightness`）：门叶 / 砾石 ROI 都在 LiDAR 面上、离锚点 < 0.3 m，剪枝**不应**碰它们；ROI 读数若下降 ≥ 噪声带（±5%）即剪枝误伤了近表面几何（看 `pruned_far` 里 opacity 高的行的位置分布）。
3. **离轨 PSNR**（18 帧）：悬浮体最敏感的数字，判赢门槛 ≥ +0.3 dB；预期 S1 在这里最先出现差异。
4. **出图**：09 §7 同六视角 R1 vs S1 并排；被剪掉的天空/树高斯所在像素改由背景库补，pitch_up 面的树区变成穹顶色块是**预期**；要盯的是墙面/门叶/地面是否露出背景色（= 近表面行被误剪，或 `min_age` 太短让还没落到面上的 split 子代被剪）。
5. **形态分位**（short p50、opacity p50、frac<0.1）与终态高斯数：S1 终态数应明显低于 R1d（tile1 20k 的 34% 远高斯直接没了），对齐 cap/生命周期后再比锐度。
6. 训练遥测对账：`surface_anchor_pruned_total` 终值 ≈ Σ 事件 `pruned`；首个事件（700）的 `far_fraction` 应与 R1d 同步 checkpoint 的审计值同量级（tile1 无 700 步 checkpoint，可用 DIAG-40 室内 3000 步的 0.070 作量级参考）。

**预期与后手**：全视角 tile1 上远高斯是 34%，机制直接命中 README 里 viewer 可见的问题，S1 最可能的结果是"外凸与悬浮消失、离轨 PSNR 涨、ROI 持平"；风险是剪掉的高斯本来在给树/天空像素供色，背景库穹顶补不出树的结构，训练视角 PSNR 可能降——这是接受的取舍（那些像素不是切片该拥有的，09 §6.3）。DIAG-40 上远高斯只有 7% / 1%，S1 读数可能落在噪声带内；若 DIAG-40 中性而 tile1 明显改善，结论是"机制对长日程累积的远高斯有效、对短日程无害"，不按 DIAG-40 判负。后手：`every: 100` 延伸到 refine stop 之后；`max_distance_m` 0.2 / 0.5 敏感性；`reject_unsupported_parents` 单独关掉的拆因臂。

## 7. 风险

* **没有 LiDAR 回波的真实结构**：玻璃、细杆/栏杆/电线、扫描盲区（视线被挡）、远处建筑。这些地方初始化云没有点，照片有内容，S1 会在每个 cull 事件把那里长出的高斯剪掉，画面退回背景库。0.3 m 的阈值容忍了 LiDAR 噪声与配准偏移（03 号报告门锁边缘 ≈ 10 mm），但容忍不了整块缺失的面。审计工具的 `far_fraction` 分不清"悬浮体"和"无回波的真实面"，只能靠出图判。
* **扫描时的动态物体**：人、开关中的门（`scan-time-moving-objects` 记录里的门后树）。LiDAR 记下的是扫描时刻的位置，照片里的位置可能不同；两者都在 0.3 m 内时无事，门叶开合角大时门叶高斯会被剪。
* **切片边界**：`outside_box: prune` 用外扩 0.3 m 的训练框；halo 区里的邻切片几何本来就由合并时的所有权决定，剪掉不影响交付，但训练视角里跨框的像素会失去前景、改由背景补——与 `tile_ownership_masking` 的动机一致，副作用形态相同（09 §6.3）。
* **`min_age_steps` 与 split 子代**：split 子代按父本尺度偏移，父本本身 > 0.2 m（`split_scale_m`）时子代可落到锚点 0.3 m 外；100 步豁免期后若还没被拉回就会被剪。这是设计意图（远父本的子代不该活），但若观察到 `pruned_far` 里新生儿占比很高而人口停滞，说明增殖预算在被剪枝消耗，应先开 `reject_unsupported_parents`（S1 已开）再看。
* **续训**：`pruned_total` 存在 strategy state（普通 int），`remove/duplicate/split` 不动它；warm-start 到新配置时按新 state 从零计。
* **代价上限**：cap 15M 时每个 cull 事件 ≈ 6 s CPU（有上界查询）+ 一次 GPU→CPU 拷贝（15M×3×8 B = 360 MB），100 步一次；与 `LidarNormalAnchors.refresh` 同量级，未在 GPU 机上实测，真机首次运行看 progress 里事件间隔是否拉长。
