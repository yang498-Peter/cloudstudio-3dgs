# CloudStudio 3DGS SDK 总览（`cloudstudio3dgs_sdk`）

面向对象：没有参与 house0305 研究过程、需要在**新数据集**上跑出同一套交付质量的工程/交付团队。

这个包把 house0305 的交付配方冻结成可复用的配方对象（默认 **b5sky**：不带填充层、身体 + 冻结天空层
成对交付、样本预取开；**b5fill2** 保留给复现 2026-09-14 那一批交付件），并在现有
`tools/pipeline.py`（job-state 机、`gpu.lock` 租约、`queue` / `deliver` 子命令）之上提供四个阶段的
驱动。它**不重写** `tools/pipeline.py`；数据接入由 `cloudstudio3dgs_sdk.ingest` 负责，
`bundle.load_dataset_bundle()` 把接入层的缓存图跑完 CPU 那一半并投影成训练器的路径契约。

```
python -m cloudstudio3dgs_sdk run --dataset <路径> --work <路径> [--profile b5sky] \
    [--dry-run] [--stages prepare,train,deliver,report] [--vram-gib 15.9] [--force] \
    [--adapter s1_fisheye] [--run-dir <processed>] [--pipeline-gate <gate.json>]
```

两条进入路径，第 9 节各有一段实操：

* **已经准备好的场景**（house0305，或上一轮 SDK 跑过的数据集）：`adopt` 从 as-run 配置生成
  `prepare_manifest.json`，`run` 直接采纳并验签，不重建任何缓存。
* **全新数据集**：`run` 的 prepare 阶段走接入层。CPU 缓存自动建；第一个 GPU 缓存把准确命令打印出来
  停下；就绪门禁（gate chain）必须用门禁工具链单独产出后经 `--pipeline-gate` 传入——SDK 不伪造门禁。

---

## 1. SDK 做什么

| 模块 | 职责 |
| --- | --- |
| `profile.py` | 冻结、带版本、带 `profile_sha256` 的配方对象 `PROFILE_B5FILL2`。**每个旋钮附实测出处**。 |
| `plan.py` | profile + 已发现的数据集 → 具体步骤清单（切片数、cap、每臂名字与配置、替身背景、合并/填充/导出/电池），**不执行任何东西**。 |
| `requirements.py` | 预检：显卡/显存、磁盘余量 vs 计划估算、gsplat 扩展与锁、python/torch 版本、外部权重是否在本机。结构化 PASS/FAIL，FAIL 即拒绝开工。 |
| `project.py` | `Project(dataset_root, work_root, profile)` 与四个阶段方法 + `run_all()`。幂等、带 state sidecar、可续跑、输入 sha 不符即拒绝。 |
| `bundle.py` | `prepare()` 调用的接口定义：`PreparedScene`（路径契约）+ `DerivedCaches`。接入实现是 `cloudstudio3dgs_sdk.ingest` 的事。 |
| `__main__.py` | CLI：`run` / `preflight` / `profile`。 |

**配方是数据，不是代码分支。** 引擎读 profile，永不按 profile 名字分支；新增一个配方 = 在
`PROFILES` 里多一个对象，不改 `plan.py` / `project.py`。

---

## 2. 四个阶段

| 阶段 | 设备 | 内容 | 产物 |
| --- | --- | --- | --- |
| `prepare` | **纯 CPU** | 数据接入（委托 `ingest`）→ 写全部 arm 配置与 `pipeline.json` → 天空 mask（SegFormer）→ 天空穹顶 → 每切片归属缓存 | `prepare/prepare_manifest.json`、`caches/`、`runs/*.json` |
| `train` | GPU | 全场背景库 → 粗先验 B0（10k）→ 每切片替身背景 → 每切片交付臂（20k，`controlled_stop`） | `runs/<arm>/checkpoints/latest.pt` |
| `deliver` | CPU + GPU | 四切片合并（含填充层）→ 导出 PLY → 阈值对照 → 回读 → 电池 + 形态（+ 有参考件时的同口径对比）→ 冻结身份 | `runs/delivery_<tag>/`、`report/delivery_<tag>_identity.json` |
| `report` | CPU | profile 门槛 vs 实测读数，附全部未实测旋钮与遗留问题 | `report/<tag>_report.{json,md}` |

阶段之间的实际训练/合并仍然走 `tools/pipeline.py queue` 与仓库里既有的 `tools/*.py`；SDK 只负责
**计划、预检、编排、记账**。

### 工作目录布局

```
<work>/
  prepare/prepare_manifest.json      prepare 的契约产物，下游只读这个
  caches/                            sky_masks/ sky_dome.pt view_backgrounds/ ownership/ backdrops/
  runs/                              tools/pipeline.py 的 run_root：<arm>.json、<arm>/、delivery_<tag>/
  pipeline.json                      给 tools/pipeline.py 的本机配置
  sdk_state/stage_<阶段>.json        阶段状态 sidecar
  logs/                              每一步的 stdout/stderr
  report/                            交付报告与身份
```

---

## 3. Profile 契约

一个 `Profile` 由若干**只读数据段**组成（`runtime`、`dataset_contract`、`tiling`、`trainer_base`、
`tile_rules`、`coarse_prior`、`backdrop`、`merge`、`export`、`battery`、`acceptance`、`cost_model`），
外加 `external_assets`、`provenance`、`open_questions`。三条硬规则：

1. **profile 里没有路径。** 路径属于数据集，由 `plan.py` 从 `prepare()` 记录的内容解析。
   （`tests/test_sdk_profile.py::test_profile_holds_no_absolute_paths` 钉住这一点。）
2. **每个值都带 `Provenance`**，写明是哪一次实测定下来的，以及这份实测值多少：
   `measured` → `extrapolated` → `inherited` → `inferred` → `unmeasured`。
   没人测过的旋钮**允许存在**，但必须自报，报告会把它们逐条列出来。
3. **`profile_sha256` 覆盖全对象**（含 provenance）。运行时记录它；续跑时不一致即拒绝。
   改一个旋钮 = 一个新配方，而不是悄悄改掉旧配方。

深冻结是真的：`profile.trainer_base["cap_max"] = 1` 抛 `TypeError`，嵌套块同样，列表是 `tuple`。

### B5fill2 的关键旋钮与出处（摘）

| 旋钮 | 值 | 出处 |
| --- | --- | --- |
| `tile_rules.cap_ratio_of_initialisation` | 1.756 | Tile_1 的 6.0M = 1.756 × 3,417,320 初始化点；容量-质量曲线 4.5M→门叶 ROI 0.405、6M→0.501、7.75M→0.528 |
| `runtime.max_gaussians_per_gib_vram` | 781,250 | tile3_R1d 在 12.48M 高斯 / 14.0 GiB 上 CUDA 崩溃；Tile_0 的 cap 因此从 12.4M 手工降到 11.0M |
| `tiling.epochs_for_max_steps` | 20 | 四个 as-run 配置的 `max_steps` 恰好 = 20 × 各自视角数；粗先验 3536 面 → 70720 |
| `trainer_base.controlled_stop_after_steps` | 20000 | 越训越糊：死质量单调升、PSNR 5–10k 达峰；20k 是停点不是预算 |
| `trainer_base.default_strategy.reset_every` | 300 | reset 周期是 cull 崩塌的根因；vendor-exact 档配合 `reset_optimizer_state=keep` |
| `trainer_base.sky_supervision` | erosion 4 px / 无 LiDAR 守卫 6 px / alpha 0.5 | 交付臂实跑值（设计文档提的是 8 / 24，交付用的是 4 / 6） |
| `trainer_base.tile_ownership_*` | 开 / margin 0.5 m / 膨胀 15 px | 四切片空中高斯下降 53–73%；B6 把膨胀放到 40 px 没换来任何东西 |
| `merge.fill` | 体素 0.2 m / clearance 0 / min_opacity 0 | 240 万粗先验行保留 52.9 万；battery alpha p05 0.189 → 0.733、PSNR 18.05 → 19.11，只多 1.4% 高斯 |
| `export.min_opacity` | 0.05（另出 0 / 0.01 / 0.05 对照） | 交付阈值；对照把"被阈值删掉多少"写进记录 |
| `battery.views` | 48，**必须同时报 alpha 覆盖率** | 否则覆盖缺口会被读成"糊" |

`python -m cloudstudio3dgs_sdk profile b5fill2` 打印完整的 provenance 与遗留问题。

### 派生规则复现 as-run 配置

`tests/test_sdk_plan.py::test_house0305_derivations_reproduce_the_as_run_configs` 用 house0305 的真实
切片几何跑一遍派生规则，四个切片的 `cap_max` / `max_steps` / `prune_switch_step` **全部逐值等于**
`3dgs-runs/house0305_sop/tile{0,1,2,3}_B5_cap6_20k.json` 里实跑的值：

| 切片 | 视角 | 初始化点 | cap（规则） | cap（as-run） | max_steps | prune_switch |
| --- | --- | --- | --- | --- | --- | --- |
| Tile_0 | 2132 | 7,044,777 | 11.00M（1.756× = 12.4M，被 16 GiB 上限钳住） | 11,000,000 | 42640 | 21320 |
| Tile_1 | 1829 | 3,417,320 | 6.00M | 6,000,000 | 36580 | 18290 |
| Tile_2 | 1684 | 3,309,574 | 8.00M（1.756× = 5.81M，低于自身上代 7.47M，被下限规则抬起） | 8,000,000 | 33680 | 16840 |
| Tile_3 | 2317 | 5,651,827 | 9.90M | 9,900,000 | 46340 | 23170 |

这是"配方可以表达成数据"这句话唯一有意义的证据。

---

## 4. 新数据集需要提供什么

`prepare()` 的输出契约是 `bundle.PreparedScene`，字段按来源分组：

* **影像 / 位姿 / 内参**：`dataset_manifest`、`recording_root`、`face_cache_manifest` + `face_cache_root`
  （训练真正采样的 Face4 展开）、`renderer_mask_manifest`。
* **划分**：`split_manifest`（rig-frame 划分，train / val / golden）。
* **监督遮罩**：`mask_*`、`person_mask_*`（人从光度目标里去掉）、`mono_depth_*`（DA2 相对深度）。
* **LiDAR**：`lidar_cloud`（客户原始点云）、`depth_*`（逐面投影回波）、`face_lidar_geometry_*`。
* **切片**：`tile_inputs_manifest` / `tile_inputs_root`（逐切片：core box、training_and_export_box、
  视角列表、LiDAR 初始化 PLY 与点数）、`tile_geometry_manifest`；`global_init_ply` /
  `global_init_geometry`（粗先验用的 2 m 抽稀全场初始化）。
* **门禁**：`pipeline_gate`（训练器拒绝在缺它时启动）。
* **派生缓存**：`DerivedCaches`（天空 mask、天空穹顶、全场背景库、逐切片归属缓存）。

**全新数据集由 `bundle.load_dataset_bundle()` 准备**：调用 `cloudstudio3dgs_sdk.ingest.load_dataset`
（适配器 S1 鱼眼 rig / COLMAP / 纯针孔目录，缺省自动探测，`--adapter` 指定）与 `plan_caches` 得到签名
缓存图，按依赖顺序建 CPU 缓存；遇到第一个输入就绪的 GPU 缓存抛 `GpuStepRequired`（带准确命令），
不在 prepare 阶段占 CUDA；粗先验用的全场初始化在这里用 `tools/build_lidar_init.py` 按配方抽稀建出。
它**不做**两件事，缺了就按名字拒绝：

* 没有 LiDAR 点云的采集件——配方从 LiDAR 初始化每个切片，跑不了。
* 就绪门禁（`pipeline_gate`）。鱼眼数据的训练器拒绝在缺签名门禁时启动，而门禁链（七个工具：
  `build_mipmap_frontend_gate` → `advance_mipmap_renderer_mask_gate` → `advance_mipmap_lidar_depth_gate`
  → `advance_mipmap_da2_gate` → `advance_mipmap_tile_gate` → `promote_surface_frozen_training_gate`
  → `bind_monocular_depth_gate`）不属于接入层。缓存建好后用它们对着 `<work>/caches` 产出门禁，再
  `--pipeline-gate PATH` 传入；不传就拒绝并把工具链原样打印出来。

**分体采集件**（S1Mapper 的 `..._Raw_Data`（相机 + info）与 `..._Processed_by_S1Mapper`（`ImgPose.txt` +
上色 LAS）两个目录）：`--dataset` 指向 Raw_Data，`--run-dir` 指向 Processed 目录，`--adapter s1_fisheye`。
`run_dir` 只在给出时才传给适配器，没有这个关键字的适配器不受影响。

接入侧的适配器、缓存图与自动切片规则见 `docs/sdk_ingestion.zh-CN.md`。

对**已经准备好的场景**（house0305，或上一轮 SDK 跑过的数据集）不走这条路：`adopt` 子命令从各切片
的 `config_as_run.json` 与粗先验配置生成 `<work>/prepare/prepare_manifest.json`（校验其中每条路径、记录
摘要），`Project.prepare()` 直接采纳并验签，被采纳的缓存**永不在原地重建**。

两个 `DatasetBundle` 不要混淆：

* `cloudstudio3dgs_sdk.ingest.bundle.DatasetBundle` 是**采集件**（图像表、内参、位姿、rig、时间戳、
  划分、点云引用、能力集）——并行任务拥有。
* `cloudstudio3dgs_sdk.bundle.PreparedScene` 是**已准备场景的路径契约**——只有路径与计数，是交付引擎
  消费的那一层投影。

---

## 5. Fail-closed 规则

house法则："宁可拒绝，不要跑出一个没人能重建的交付件。"

1. **阶段链上的摘要比对。** 每个阶段把自己每个产物的摘要写进 sidecar；下一个阶段开工前重算上游
   产物的摘要，只要有一个对不上就拒绝，并说明是哪个文件、怎么不一样。
   摘要策略与 `tools/pipeline.py` 一致：≤ 256 MiB 的文件算真 sha256，更大的（checkpoint、合并 PLY）
   记 size + mtime，并且记录里写明用的是哪一种，**不会有人把 stamp 当成 hash 读**。
2. **profile 变了就拒绝续跑。** 上游阶段记录的 `profile_sha256` 与当前进程不符 → 拒绝，要求新开工作根。
   同样地，`_stage_is_current()` 不会把"另一个 profile 下完成的阶段"当成已完成而跳过。
3. **GPU 阶段先过预检。** `train` / `deliver` 开工前跑 `requirements.preflight()`，任何 required FAIL
   直接 `PreflightFailed`，什么都不启动。
4. **本检出跑不出这个配方就拒绝。** 计划里的步骤如果需要本检出没有的工具能力，该步骤带 `blocking`，
   预检的 `checkout_supports_profile` 变 FAIL（见第 7 节）。
5. **阶段幂等。** 状态为 `COMPLETE` 且记录的产物摘要仍然吻合 → 跳过；删掉任一产物即重新打开该阶段。
   `--force` 才会重跑。
6. **外部权重不随交付发布。** profile 里任何 `ships_in_delivery: true` 的外部资产直接判 FAIL。

### 外部资产与许可

天空 mask 用 `nvidia/segformer-b4-finetuned-ade-512-512`（revision `2641fd1e`），许可是
**NVIDIA Source Code License-NC（非商业）**。权重**只用于在训练期间派生监督遮罩**；权重本身、以及
任何由它派生的模型文件，**都不进交付件**，交付的 PLY 里不含 SegFormer 的任何输出。预检会把这段话
原样打印在检查结果里，不管权重在不在本机。

---

## 6. house0305 干跑实录

数据集摘要取自真实的 `tile_inputs_v9/tile_inputs_manifest.json`（884 张训练图 × 4 面 = 3536 个训练视角，
全场初始化 1,863,918 点），并用 `--prior-checkpoint` 提供上一代 R1d checkpoint（有它们就不需要 seed 代）：

```
python -m cloudstudio3dgs_sdk run \
    --dataset D:/scenes/house0305 --work D:/work/house0305 --profile b5fill2 --dry-run \
    --summary house0305_summary.json --vram-gib 16 \
    --prior-checkpoint 0=.../tile0_R1_range0_20k/checkpoints/latest.pt \
    --prior-checkpoint 1=.../tile1_R1d_20k/checkpoints/latest.pt \
    --prior-checkpoint 2=.../tile2_R1d_20k/checkpoints/latest.pt \
    --prior-checkpoint 3=.../tile3_R1d_cap13m_20k/checkpoints/latest.pt
```

```
plan house0305 profile=b5fill2@2026.09.14
  profile_sha256 a8e2d7129f40cfedb83d24a7ee7c5d1033731413c260901bcf1dfa1e736254e6
  plan_sha256    a927b4f115a9ceedc6ff92bfb7c58b2f2c6f363d119b75f2ba5b6d8244abf285
  dataset        D:\scenes\house0305
  work           D:\work\house0305
  tiles          4 (Tile_0 2132v init 7.04M cap 11.00M, Tile_1 1829v init 3.42M cap 6.00M,
                    Tile_2 1684v init 3.31M cap 8.00M, Tile_3 2317v init 5.65M cap 9.90M)
  generations    delivery

[prepare] 8 steps, 65m, 0.8GB disk
   ingest_dataset                     external        0s         -  [unmeasured]
      note: adapters, the signed cache graph and the automatic tiling rule live in
            cloudstudio3dgs_sdk.ingest; an existing verified prepare_manifest.json is adopted instead
   write_arm_configs                  cpu             1s         -  [measured]
   sky_masks                          cpu            35m     0.1GB  [unmeasured]
      note: SegFormer b4 ADE20k, NVIDIA non-commercial licence: supervision masks only,
            nothing derived from it ships
   sky_dome                           cpu             5m     0.0GB  [unmeasured]
   ownership_Tile_0                   cpu             7m     0.2GB  [measured]
   ownership_Tile_1                   cpu             6m     0.2GB  [measured]
   ownership_Tile_2                   cpu             5m     0.2GB  [measured]
   ownership_Tile_3                   cpu             7m     0.2GB  [measured]

[train] 10 steps, 11.4h, 23.5GB disk
   global_view_backgrounds            gpu             4m     0.3GB  [extrapolated]
   train_global_coarse_b5fill2        gpu            42m     0.8GB  [extrapolated]
      note: tile-free whole-scene prior; feeds both the backdrop and the merge fill layer
   backdrop_Tile_0                    gpu             2m     2.8GB  [measured]
      note: sky dome + the other tiles' checkpoints + the coarse prior, own box excluded
   train_tile0_b5fill2_delivery       gpu           3.1h     3.0GB  [extrapolated]
   backdrop_Tile_1                    gpu             2m     2.4GB  [measured]
   train_tile1_b5fill2_delivery       gpu           2.0h     3.0GB  [extrapolated]
   backdrop_Tile_2                    gpu             2m     2.2GB  [measured]
   train_tile2_b5fill2_delivery       gpu           2.5h     3.0GB  [extrapolated]
   backdrop_Tile_3                    gpu             2m     3.0GB  [measured]
   train_tile3_b5fill2_delivery       gpu           2.9h     3.0GB  [extrapolated]

[deliver] 9 steps, 12m, 13.0GB disk
 ! merge_tiles                        cpu             4m     2.7GB  [extrapolated]
      note: fill layer: coarse-prior rows kept only where no merged tile gaussian occupies
            their 0.2 m voxel
      BLOCKING: the profile asks for a fill layer this checkout cannot produce:
            merge_v28_tile_checkpoints.py has no --fill-checkpoint; tools/pipeline.py deliver
            does not forward the fill arguments (the flag exists on the research branch only)
   export_ply                         cpu            30s     2.1GB  [extrapolated]
   threshold_control                  cpu             2m     6.3GB  [extrapolated]
   reimport_ply                       cpu            60s     1.9GB  [extrapolated]
   battery                            gpu            30s         -  [measured]
      note: alpha coverage is reported beside PSNR; a coverage gap reads as blur otherwise
   morphology                         gpu            30s         -  [extrapolated]
   compare_matched                    gpu             2m         -  [unmeasured]
   offtrajectory                      gpu             2m         -  [unmeasured]
   freeze_identity                    cpu            30s         -  [measured]

[report] 1 steps, 10s, 0.0GB disk
   acceptance_report                  cpu            10s     0.0GB  [measured]

total 12.7h wall, 37.2GB peak disk [unmeasured]
WARNING Tile_0: cap clamped 12.40M -> 11.00M by the 16 GiB VRAM ceiling
WARNING Tile_2: the 1.756x rule gives 5.81M, below its previous final population 7.47M;
        the floor rule raised the cap to 8.00M
BLOCKED merge_tiles: ... (见上)
```

（原始输出每一步还会打印完整 argv，这里为了可读性省掉了 `$ ...` 行；`--plan-json` 可以把同一份
计划写成 JSON。）

对照真实战役：四个切片实跑 121 / 152 / 168 / 187 分钟 = 10.5 小时，加粗先验与替身背景，与计划里
的 11.4 小时一致。两条 WARNING 恰好就是当时手工做的两处 cap 修正。

没有上一代 checkpoint 时（真正的新数据集），计划会多出一个 **seed 代**：训练时间从 12.7 h 变成
23.1 h，磁盘从 37.2 GB 变成 49.2 GB，并给出对应警告。

---

## 7. 配方里还不能表达成数据的地方

`profile.open_questions` 把这些写进了 profile 本体，`report` 阶段会原样输出。

| id | 问题 | 现状 |
| --- | --- | --- |
| `pipeline-fill-passthrough` | **阻塞项。** `eng` 检出的 `tools/merge_v28_tile_checkpoints.py` 没有 `--fill-checkpoint`，`tools/pipeline.py deliver` 也不转发填充参数；这些只存在于研究分支 `fix/loss-parity-and-split`（`cloudstudio-3dgs-work`）。 | 预检 `checkout_supports_profile` 直接 FAIL，正式跑之前必须先把这段合过来 |
| `backdrop-bootstrap` | 切片 N 的替身背景要渲染**其它切片的 checkpoint**。house0305 用的是上一代 R1d/R1；全新数据集没有上一代，只能先跑一遍 seed 代，GPU 时间翻倍。 | 已表达为 `tile_rules.seed_generation_overrides` + 计划里的 `seed` 代，但这组 override 本身是 **INFERRED，从未 A/B 过** |
| `cap-floor` | cap 下限规则需要"该切片自己上一代的最终人口"，新数据集拿不到，规则不会触发。 | 规则是数据（`cap_floor_policy` + `cap_floor_headroom` 1.07），输入只有复跑时才有 |
| `fill-vs-sharpness` | 填充层买到覆盖（battery alpha 0.712 → 0.947、PSNR 18.05 → 19.11），卖掉离轨锐度（0.444 → 0.369）。两项差异都远超复跑噪声带，是**真实取舍不是最优点**。 | `merge.fill.enabled` 与体素参数是单调旋钮；B5fill2 选的是覆盖 |
| `no-reference-model` | `tools/pipeline.py` 的配置 schema **强制**要求 `reference_ply` / `reference_alignment` / `delivery_baselines`，其 deliver 会跑同口径三方对比与离轨对比。新场景的第一次交付根本没有竞品件。 | 计划在 `has_reference_model=false` 时省掉这两步，但 pipeline 配置 schema 仍然要求这些路径——应当改成可选 |
| `tile-plan` | 把任意场景切成几块、怎么切，是接入侧的规则。 | 委托给 `prepare()`；`cloudstudio3dgs_sdk.ingest.tiling` 已有 slab 切分实现 |
| `prepare-step-source` | `plan.py` 把 prepare 的缓存步骤写死了；`cloudstudio3dgs_sdk.ingest.plan_caches` 已经能从采集件推导同一张图，而且带依赖绑定与 staleness 规则。 | 集成点：等 ingest 的 API 稳定后，prepare 的步骤应改为由 `plan_caches` 提供 |

另外三个 `cost_model` 条目是 **UNMEASURED**（`sky_mask_seconds_per_face`、`sky_dome_seconds`、
`compare_seconds`）：从来没人单独计时，只是为了能给出总时长而估的。预检的 `profile_confidence`
会 WARN 列出它们，`report` 也会。

---

## 8. 测试

全部 CPU、无 GPU、无网络：

| 文件 | 数量 | 覆盖 |
| --- | --- | --- |
| `tests/test_sdk_profile.py` | 24 | 深度不可变、sha 稳定性与敏感性、provenance 覆盖与置信度、profile 里没有绝对路径、外部资产不随交付发布 |
| `tests/test_sdk_plan.py` | 30 | 派生规则（cap 比例 / 下限 / VRAM 上限、步数、prune 切换）、**house0305 四切片复现 as-run 配置**、两切片合成场景的步骤清单与顺序、seed 代、估算与置信度、计划 sha、blocking 检测 |
| `tests/test_sdk_project.py` | 29 | 摘要（sha / size+mtime 两条路径）、state sidecar 形状与历史、prepare 采纳与委托、幂等跳过与 `--force`、上游产物变更/消失/profile 变更的拒绝、阻塞步骤、全流程 `run_all`、阈值对照记录 |
| `tests/test_sdk_requirements.py` | 22 | python / torch / gsplat 锁与扩展 / 无卡 / 显存不足 / cap 超卡容量 / 磁盘不足（含安全系数边界）/ 权重缺失与许可声明 / 检出不支持配方 / 工具缺失 / `raise_for_status` |
| `tests/test_sdk_cli.py` | 31 | `--stages` 解析（排序、去重、未知项）、`--prior-checkpoint` 解析、参数必填与退出码、干跑输出内容（阶段、总计、cap、置信度、argv、警告、BLOCKED）、`--plan-json`、`preflight` 与 `profile` 子命令 |
| **合计** | **136** | |

上表是 2026-09-14 的快照；之后又加了 `test_sdk_bundle_build.py`（全新数据集构建路径）、
`test_sdk_adopt.py`（采纳 as-run 配置）、`test_sdk_discover.py` 与接入层套件。**条数以 runner 为准**：

```
python -m pytest -q tests/test_sdk_*.py tests/test_ingest_*.py
```

---

## 9. 常见操作

```bash
# 只看这台机器能不能跑（不跑任何东西）
python -m cloudstudio3dgs_sdk preflight --dataset <路径> --work <路径> --summary <摘要.json>

# 只准备（纯 CPU 的机器也可以）
python -m cloudstudio3dgs_sdk run --dataset <路径> --work <路径> --stages prepare

# 中断后续跑：已完成且产物摘要吻合的阶段会被跳过
python -m cloudstudio3dgs_sdk run --dataset <路径> --work <路径>

# 打印配方与全部出处
python -m cloudstudio3dgs_sdk profile b5sky
python -m cloudstudio3dgs_sdk profile b5sky --json
```

### 9.1 已准备好的场景：adopt，然后 run

house0305 就是这样跑的（`day_plan26`，2026-09-18）。四份切片 `config_as_run.json` + 粗先验配置
生成清单；天空穹顶、天空层 PLY、竞品参考模型都是可选，给了就校验并记进清单：

```bash
python -m cloudstudio3dgs_sdk adopt --work <work> --profile b5sky --scene-tag house0305 \
    --tile-config <runs>/tile0_.../config_as_run.json  (每个切片一份) \
    --coarse-config <runs>/house0305_global_coarse_B0_10k.json \
    --sky-dome <probes>/sky_house0305.pt --sky-ply <exports>/house0305_f5_sky_20260903.ply \
    --reference-ply <竞品.ply> --reference-alignment <刚体对齐.json> \
    --eval-config <runs>/house0305_sop/delivery_eval.json

# 先干跑：输出里不能出现 "<prepare:" 占位符，否则清单不完整
python -m cloudstudio3dgs_sdk run --dataset <数据集> --work <work> --profile b5sky --vram-gib 15.9 \
    --dry-run --prior-checkpoint 0=<上一代 tile0 latest.pt> ... (每个切片一份)

python -m cloudstudio3dgs_sdk run --dataset <数据集> --work <work> --profile b5sky --vram-gib 15.9 \
    --prior-checkpoint 0=... --prior-checkpoint 1=... --prior-checkpoint 2=... --prior-checkpoint 3=...
```

注意点：

* `adopt` 从各切片 `monitor/progress.jsonl` 的最后一条读取上一代最终种群，切片 cap 按配方规则**重新推导**，
  不照抄手工配置里的数字（house0305 Tile_2 手工 8.0M，按规则 6.8M；Tile_0 12.4M 按 15.9 GiB 的卡钳到 10.9M）。
* `--vram-gib` 要填这张卡**实际可用**的数（RTX 5070 Ti 是 15.9，不是 16）；b5sky 的 `min_vram_gib` 是 15.5。
* `--reference-ply` / `--reference-alignment` 必须成对。不给时，逐臂的 `offtraj` / `compare` 与交付的
  `*_matched` 四步以及它们的评分器按名字报 `skip`，形态、battery、身份冻结、成对 alpha 门禁照常跑——
  一个新场景的首次交付没有竞品可比，不能让训练四十分钟之后死在比对步骤上。
* **不要**把整个 `run` 再包一层 `tools/gpu_lease.py`：流水线自己对 `<work>/runs/gpu.lock` 逐训练步取租约，
  外层同文件的租约会让第一个训练步拒绝自己的祖先进程（2026-09-18 实测）。
* 配方 sha 变了（改任何旋钮）就换一个新的 `--work`，旧工作根不会被续跑。
* **`--eval-config` 对 house0305 是必须的。** battery 的验证缓存不是配置里的路径，是评估器按名字从训练缓存
  派生的（`face4`→`face4_val`、`_train`→`_val`，规则在 `cloudstudio_3dgs/training/validation_paths.py`）。
  house0305 用 v9 缓存训练，但 v9 没有 `face4_lidar_val_vis6`；历次交付都是拿 v8 的 `delivery_eval.json`
  评的。不传时 prepare 会按派生规则预查验证缓存，缺了就在训练前拒绝，而不是训练 7 小时后死在 battery。
* 续跑时预检只按**还没跑的步骤**要磁盘（`Plan.pending()`），报告里同时给整计划数字；已存在的产物不再计费。
* `write_arm_configs` / `delivery_eval_config` / `acceptance_report` 是 `refresh` 步骤：每次 prepare/report 都重写
  （内容不变则字节相同），不会因为"输出已存在"而留下过期的评估配置或报告。
* 采纳了参考模型后，交付阶段会多出 `compare_matched` / `offtrajectory` / `offtrajectory_score` 三步（argv 与
  `tools/pipeline.py deliver` 一致），报告的离轨锐度门禁读 `offtrajectory_scores.json` 的中位 `sharp_ratio`。
  交付完成后才补上参考模型也行：计划多出的步骤会让 deliver 只跑这三步，然后重开 report。

house0305 首跑结果（2026-09-19，eng `749b34a`）：五项门禁全部 PASS——成对 alpha p05 0.907 / PSNR p10 15.08 /
离轨锐度 0.468 / 16.08M 高斯 / 短轴 p50 0.456 mm；与手工 B5 交付（0.898 / 15.05 / 0.454 / 16.72M）同量级。
GPU 总耗时约 6.5 h（粗先验 31 min，四切片 77–110 min，交付评分 < 3 min）。

### 9.2 换一套新数据

```bash
# 1. 是什么数据？（分体采集件：--dataset 指 Raw_Data，--run-dir 指 Processed_by_S1Mapper）
python -m cloudstudio3dgs_sdk.ingest.cli detect <dataset>

# 2. 只准备：CPU 缓存自动建，遇到第一个 GPU 缓存停下并打印命令
python -m cloudstudio3dgs_sdk run --dataset <raw> --run-dir <processed> --adapter s1_fisheye \
    --work <work> --stages prepare
# 3. 在有 CUDA 的机器上按打印的命令建 GPU 缓存，重复第 2 步直到 prepare 只剩门禁一项

# 4. 门禁工具链对着 <work>/caches 产出签名门禁（第 4 节列的七个工具，顺序固定）
# 5. 带门禁完整跑
python -m cloudstudio3dgs_sdk run --dataset <raw> --run-dir <processed> --adapter s1_fisheye \
    --work <work> --pipeline-gate <work>/gate/pipeline_gate.json --vram-gib 15.9
```

没有上一代检查点时不给 `--prior-checkpoint`，计划自动多出一代种子训练（替身背景要有东西可渲染）。
磁盘按 `--dry-run` 打印的 `peak disk` 预留；house0614 的 900 张子集估算约 57 GiB 缓存。
