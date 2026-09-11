# 12 — K-line: `sky_supervision` 天空监督（2026-09-11）

状态：训练器旋钮、数据集绑定、损失项、增殖阻断、契约、门禁拒签、测试、K1 配置已完成；**K1 尚未训练**（本机 CPU-only，未提交）。天空 mask 由另一子任务生成（`tools/build_sky_masks.py`、`cloudstudio_3dgs/data/sky_masks.py`，提交 `9f23dfa`）：写本文时 `sky_mask_val` 已建、`sky_mask_train` 仍在 CPU 上跑（`faces/` 305 张），`sky_mask_train.json` 尚不存在，所以 K1 的 `validate()` 目前只在"manifest 缺失"这一步失败（§7）。

## 1. 问题与方向

README 晚间行的事实：切片训练的逐视角背景库只有天空穹顶（`tile_backgrounds_v9/Tile_*`，`dome_source = probes/sky_house0305.pt`），损失是 `final = render + (1 − α)·backdrop`。照片里是屋檐 + 树枝 + 蓝天，穹顶是一片糊的蓝/棕色块，于是切片自己长出高斯去画天空和树，这些高斯悬在墙前、檐上。定量：`tile1_R1d_20k` 全模型 334 万高斯 **34% 离最近 LiDAR 点 > 0.2 m**（27% > 0.5 m、20% > 1 m），远组 z p50 2.46 m（近组 0.37 m），35% 在切片训练框外；竞品对齐后同一盒内 208 万高斯只有 1.8% > 0.2 m。

用户方向：**无 LiDAR 处允许自由生长，天空是特例**——天空像素上切片不得作画，这些像素归穹顶。S1（`surface_anchor_prune`，10 号报告）的硬距离规则只作上界探针；K1 直接给出"天空像素上 α → 0"的损失，这正是文献综述（11 号）里 `render + (1−α)·backdrop` 系统（Street Gaussians / OmniRe / Splatfacto-W / Urban RF）都带、我们缺的那一项。

## 2. 机制：旋钮做了什么

新模块 `cloudstudio_3dgs/training/sky_supervision.py`，`TrainerConfig` 新增字段 `sky_supervision`：

| 字段 | 取值 | 缺省 | 含义 |
|---|---|---|---|
| `enabled` | bool | `false`（字节不变） | 总开关；关时数据集不读 mask、损失不看、契约无键 |
| `mask_manifest` | 路径 | null | 签名的 `face4_sky_mask_cache` manifest（`sky_mask_train.json`） |
| `mask_root` | 路径 | null | manifest 里 `mask_path` 的根目录 |
| `alpha_weight` | ≥0 | 0.5 | 天空 α 项权重 |
| `alpha_target` | [0,1] | 0.0 | 天空像素被拉向的 α |
| `exclude_photometric` | bool | `true` | 光度项不看天空像素 |
| `exclude_mono_depth` | bool | `true` | DA2 项不看天空像素 |
| `growth_block` | bool | `true` | 主要在天空里被看到的高斯不得增殖 |
| `mask_erosion_px` | [0,256] | 8 | 原始天空标签先腐蚀这么多像素（防线 1） |
| `require_no_lidar_within_px` | [0,256] | 24 | 任一 LiDAR 回波周围这么大的方窗内不算天空（防线 2）；0 关闭 |

**有效天空 mask**（三处消费的都是它，不是原始标签）：

```
effective_sky = erode(sky_label, mask_erosion_px) & ~dilate(depth_mask, require_no_lidar_within_px) & rgb_mask
```

腐蚀 = 补集的膨胀再取补（零填充，图像边界外不算非天空，所以裁剪面顶边的天空不会被腐蚀掉）；`depth_mask` 是数据集已经带在样本上的**签名 LiDAR 回波**（tile 裁剪与 ownership 遮蔽之后的那份），扫描仪打到的面不管标签怎么说都不是天空；最后与渲染器 mask 相交。torch 版在设备上用 `max_pool2d`（ownership 遮蔽的教训：全分辨率膨胀每步在 CPU 上做慢 3.2×），numpy 版给审计用，测试钉住两者与逐像素 oracle 相同。

数据集（`face_dataset.py`）：`FaceCacheDataset(sky_mask_manifest_path, sky_mask_root)` 与渲染器 mask 同款绑定——验签、`source_face_manifest_sha256` 必须等于 Face4 cache 的 `face_manifest_sha256`、split 一致、无重复；**被选中的面没有记录 → 构造时 `ValueError`**（fail-closed，与背景库一致）；PNG 首次访问验 SHA、形状按面核对、`sky_pixels` 与记录核对；tile 裁剪与 `rgb_mask` 同一矩形。样本新增字段 `TrainingSample.sky_mask`（原始标签，裁剪后），`identity` 只在配置了时多 `sky_mask_manifest_sha256`（identity 会展开进 checkpoint identity，无条件加键会改每次运行的形状）。验签走 `cloudstudio_3dgs/data/sky_masks.py` 的 `verify_sky_mask_manifest`（可导入时为准；本模块保留一份同签名规则的最小回退，不创建那个文件）。

### 2.1 哪些项被遮蔽（精确清单）

复用 `rgb_supervision.py` 的 mask 管道：`rgb_supervision_mask(..., exclude=effective_sky)` 在 `all` / `lidar_support` 两种模式下都把天空像素从监督集里去掉（`all` 且无 `exclude` 时仍返回 `rgb_mask` 同一对象，缺省逐字节不变）。`exclude_photometric: true` 时被遮蔽的是：

| 项 | 遮蔽后 |
|---|---|
| RGB L1（`masked_rgb_l1`） | 只在非天空监督像素上取均值 |
| SSIM（local_gaussian / global_moments） | 同上（local 版 `allow_empty`） |
| RGB 梯度 L1 | 同上 |
| LiDAR 支持 RGB L1 | `rgb_mask & lidar_rgb_mask`，rgb_mask 已去天空 |
| 训练视角 PSNR（遥测） | 同一像素集 |

`exclude_mono_depth: true` 时 **DA2 深度项** `da2_valid &= ~effective_sky`（与光度遮蔽独立生效；对齐后的单目深度没有天空概念，天空像素不得索要任何距离的面）。

**不遮蔽的**：LiDAR range 项（本来就只在回波上）、LiDAR α 地板（`lidar_alpha_*`，在膨胀 6 px 的回波支持上）、mesh 项、几何正则、曝光 gain（照旧在遮蔽前施加）。α 地板与天空 α 项在像素集上**不相交**：防线 2 的 24 px 窗 ≥ 地板的 6 px 膨胀，任一回波周围既是地板支持又不可能是有效天空。整个监督集被天空吃空的视角（全天空裁剪）走图连通的零，不触发 loss 函数的空 mask 报错（与 `lidar_support` 同一路径）。

### 2.2 天空 α 项定义

`backend.render` 返回 `alpha[0, ..., 0]`，即光栅化器**合成背景之前**的累计 α——合成用的就是它：`rgb = rgb + (1 − alpha)·backdrop`。天空项：

```
sky_alpha_loss = mean_{p ∈ effective_sky} | alpha(p) − alpha_target |
loss += alpha_weight · sky_alpha_loss
```

按有效天空像素数归一；没有有效天空像素的视角给图连通的零。梯度方向测试钉住：对每个有效天空像素 ∂loss/∂α = alpha_weight / N > 0（下降步降低 α），非天空像素为 0；两高斯玩具模型里覆盖天空像素的高斯 opacity logit 梯度 > 0（被压低），覆盖墙面像素的为 0。

### 2.3 增殖阻断（`growth_block`）——状态：**已实现，用文档里的廉价代理**

逐高斯"增长梯度主要来自天空像素"的精确归因不可行：`means2d.grad` 每高斯每视角只有一个二维向量，事后无法按像素拆分。实现的是任务书列出的代理：`SkyGrowthBlock`（`sky_supervision.py`）挂在 `DefaultStrategyAdapter` 上，每个 refine-stop 之前的步与 `_update_state` 同处累加——可见（`(radii > 0).all(-1)`）高斯的投影中心 `floor(means2d)` 落在本步有效天空 mask 内记一次命中，可见即记一次观测（`state["_cloudstudio_sky_seen" / "_sky_hits"]`，人口长度张量，随 gsplat 的 duplicate/split/remove 重索引，与 `grad2d/count` 同一时刻清零）；`_grow_mipmap` 在梯度门、锚定父本门之后、cap topk 之前，把 `hits/seen ≥ 0.5` 的候选父本从 `eligible` 去掉（纯父本遮罩，不搬新生儿，与执行顺序无关；K1 保持 R1d 的 `pre_optimizer_vendor`）。有效天空 mask 由损失通过 `info["cloudstudio_sky_mask"]` 交给策略（同一个 `info` dict 到达 `strategy_post_step`）；开了阻断而 info 里没有 mask 时 `RuntimeError`，不静默跳过。`validate()` 要求 `growth_block` 只在 `default_3dgs` + `exact_mipmap_lifecycle` + 活跃拓扑策略下开（其它策略从不读它，开了会是静默无效）；`growth_block: false` 可单独跑 α 项 + 遮蔽。

遥测：`growth_diagnostics.sky_growth_blocked_count / sky_growth_blocked_total`（总数存 strategy state 普通 int，随 checkpoint 续训）；progress 记录 `sky_alpha_loss`、`sky_pixel_fraction`（有效天空占 rgb_mask 的份额）、`sky_alpha_mean`（天空像素上的渲染 α 均值，K1 的直接读数）、`sky_growth_blocked_parents`、`sky_growth_blocked_total`；`info` 另带 `cloudstudio_sky_pixels / cloudstudio_sky_raw_pixels`（腐蚀 + LiDAR 防线去掉了多少）。梯度审计在 enabled 时多一项 `sky_alpha`。离轨 / compare 评估器**未改**：评估渲染照旧合成同一背景。

## 3. 契约、校验、门禁

`contract_dict()["loss_contract"]` **只在 `enabled` 时**多 `sky_supervision` 键（缺省与显式 `enabled: false` 的契约逐字节相同，测试钉住）；`growth_block` 时 `strategy` 多 `sky_growth_block` 键：

```json
"sky_supervision": {
  "enabled": true, "alpha_weight": 0.5, "alpha_target": 0.0,
  "exclude_photometric": true, "exclude_mono_depth": true, "growth_block": true,
  "mask_erosion_px": 8, "require_no_lidar_within_px": 24,
  "alpha_source": "accumulated_alpha_before_backdrop_composite",
  "alpha_loss": "mean_abs_alpha_minus_target_over_effective_sky_pixels",
  "effective_sky": "erode(sky_mask, mask_erosion_px) & ~dilate(depth_mask, require_no_lidar_within_px) & rgb_mask",
  "excluded_photometric_terms": ["rgb_l1", "rgb_ssim", "rgb_gradient_l1", "lidar_rgb_l1", "train_view_psnr"],
  "excluded_depth_terms": ["da2_depth"],
  "exclusion_normalisation": "supervised_pixels",
  "growth_block_rule": {"min_sky_fraction": 0.5, "observation": "projected_centre_inside_effective_sky_mask", "window": "since_last_refine_event"},
  "manifest_sha256": "<sky_mask_train.json 的 sky_mask_manifest_sha256>",
  "manifest_consumed_by_dataset": true, "source_face_manifest_bound": true
}
"strategy.sky_growth_block": {"min_sky_fraction": 0.5, "observation": "...", "window": "since_last_refine_event"}
```

路径不进契约，manifest 按签名绑定（同渲染器 mask）。校验：`enabled` 要求 `mask_manifest + mask_root`、面缓存训练；manifest 必须存在（**错误信息带 manifest 路径**）、根目录存在、验签、`source_face_manifest_sha256` 等于面缓存 sha、split 一致；`alpha_weight ≥ 0`、`alpha_target ∈ [0,1]`、两个像素旋钮 ∈ [0,256]；权重 0 且三个开关全关时拒绝（什么都不做）；关着却给了路径也拒绝。门禁 `advance_adaptive_growth_gate` 与拒签 `rgb_supervision_mask` / `surface_anchor_prune` 同一位置同一措辞拒签 `sky_supervision.enabled`。

## 4. 测试

`tests/test_sky_supervision.py`（**11 项**，CPU，合成数据）：缺省路径逐字节不变（无天空张量时 loss 等于 oracle、有张量也被忽略、`all` 仍返回 rgb_mask 同一对象、契约无键且等于显式 disabled）；有效 mask torch/numpy 双胞胎 = 逐像素 oracle，含 LiDAR 防线清掉窗内天空、两防线关闭、无回波跳过防线 2、形状不符报错；α 项数值 = 天空像素上 mean|α − target|、总 loss 组成、梯度只落在天空像素且方向压低 α、两高斯玩具模型只压天空高斯的 opacity、`alpha_target` 生效；光度遮蔽（天空像素零梯度、按剩余像素归一、`supervised_pixels/fraction`、关掉后恢复整幅、与 `lidar_support` 复合、全天空视角图连通零、enabled 无张量报错）；DA2 遮蔽（两开关四种组合）；数据集（绑定 sha、identity 只在配置时多键、tile 裁剪同矩形、缺记录 / 缺文件 / 篡改 PNG / 错绑 / 篡改签名 / 只给一半路径六种拒绝）；契约只在 enabled 多键且去键后等于基线、`growth_block: false` 只少 strategy 键；校验 12 种拒绝含 **缺失 manifest 报错带路径**、错绑、split 不一致、篡改签名、`growth_block` 非经典生命周期，以及正确绑定时通过；门禁拒签；增殖阻断（观测/命中计数、恰好一半阻断、少于一半放行、克隆后计数器长度跟随人口、`reset` 清零、无阻断时无键、缺 info mask 报错）；阻断在 `_step_post_backward_mipmap` 里的接线（非 refine 步累加、refine 步消费并清零）。

回归（本机 CPU，`.venv-train`，pytest）：任务书指定集 `test_sky_supervision + test_rgb_supervision_mask + test_schedule_contract + test_alpha_support + test_surface_anchor_prune + test_training` **125 通过、1 失败**——`test_training.py::RenderScaleContractTests::test_rendered_footprint_matches_linear_metric_scale`，"gsplat CUDA extension is not available"，即既有的 gsplat 懒加载 CUDA 失败。相邻套件 `test_face_dataset + test_default_strategy_adapter + test_mipmap_gate + test_post_refine_cull + test_gaussian_lifecycle + test_densification_gradient_source + test_schedule_audit + test_view_backgrounds + test_renderer_masks + test_tile_ownership + test_training_presets + test_sky_masks` **173 通过 + 6 子测试通过、3 失败**——三项都是 `test_training_presets.py` 的预设夹具（`'Backend' object has no attribute 'torch'`，失败点是改动前就有的 `rgb_supervision_mask(backend.torch, …)` 调用），09 号报告 §4 已记录为 HEAD 同样失败。

## 5. K1 配置

`C:\Peter\3dgs-runs\house0305_sop\tile1_K1_sky_20k.json`（副本 `run_configs/house0305_tiles/v9/`）由 `tile1_R1d_20k.json` 派生，派生脚本断言顶层只改 `run_id`（`house0305-t1-K1sky`）/ `output_dir` / `sky_supervision` / `lineage`（S1 台账写法：base + `base_config_sha256`（R1d 文件字节 sha）+ 单一改动描述），关掉 `sky_supervision` 后契约与 R1d 逐字节相同。旋钮取值即上表缺省：`alpha_weight 0.5`、`alpha_target 0`、三个开关全开、腐蚀 8 px、LiDAR 窗 24 px；`mask_manifest = …\house0305_sop_v9\sky_mask_train.json`、`mask_root = …\sky_mask_train`。生命周期执行顺序、LiDAR α 地板（0.1 / 0.95 / 6 px）、DA2、曝光、cap 15M 全部沿用 R1d。

`TrainerConfig.from_dict(K1).validate()` 结果：**FAIL，且只败在 manifest 缺失**——`FileNotFoundError: sky mask manifest is missing: C:\Peter\3dgs-datasets\house0305_sop_v9\sky_mask_train.json`；同一 dict 把 `sky_supervision.enabled` 置 false 后 `validate()` **PASS**（含 PLY/npz 哈希与 manifest 绑定检查），说明其余全部通过。manifest 落地后需重跑一次 `validate()`（多做验签、Face4 绑定、split 三项检查）并记下契约 sha（`contract_dict()` 读 manifest 绑定 `manifest_sha256`，现在算不出）。基线契约 sha 复算：`tile1_R1d_20k` `eca85467…`、`tile1_S1_anchor_20k` `9719f342…`，与 10 §5 一致；K1 的待 manifest。

## 6. K1 怎么判

配对基线 `tile1_R1d_20k`；S1 是上界探针，三者同口径：

1. **远离表面比例**（本机制的几何读数）：训练完 `tools/audit_surface_anchor.py --config tile1_K1_sky_20k.json`，与 10 §1 表配对。K1 不剪任何东西，只让天空像素不再要求高斯：预期 > 0.2 m 份额从 0.338 明显下降但不会像 S1 那样到 ≤ 0.05（树枝、邻切片内容仍会长）；重点看远组 z p50（2.46 m，檐/天空带）是否回落、框外份额是否下降。分不清"悬浮体消失"和"真实内容消失"的地方按 11 号报告的要求用出图判。
2. **训练遥测**（第一手证据）：`sky_alpha_mean` 应从早期（R1d 等价值，天空像素上高斯把 α 推向 1）单调降到 ~0；`sky_alpha_loss` 同步；`sky_pixel_fraction` 是数据事实（pitch_up 面高、pitch_down 面 0）；`sky_growth_blocked_parents` 每次 refine 的阻断数应先高后低（没有天空高斯就没有可阻断的）。若 `sky_alpha_mean` 卡在 0.3 以上说明 0.5 的权重压不过光度项/α 地板——先看 `cloudstudio_sky_raw_pixels − sky_pixels`（防线吃掉了多少），再调权重。
3. **亮度归一 ROI-all ours/ref**（`build_three_way_compare --sample-ids roi_compare_ids.json` + `score_compare_roi --match-brightness`）：门叶 / 砾石 ROI 都在 LiDAR 面上、离任何天空像素 > 24 px，天空项**不应**碰它们；下降 ≥ 噪声带（±5%）即防线漏了（过曝墙被标天空又没被 LiDAR 窗救回），看 `figures` 里 mask 叠图。
4. **离轨 PSNR**（18 帧，`build_offtrajectory_compare` + `score_offtrajectory_strips`）：悬浮体最敏感的数字，判赢门槛 ≥ +0.3 dB。
5. **viewer 裁剪 PLY**：与 README 42 行同一门叶区裁剪（opacity ≥ 0.05），看檐口上的成片凸起与墙前悬浮团是否消失、树枝区是否退回穹顶色块（**预期**，那些像素不是切片该拥有的）、墙面 / 门叶 / 地面是否露出背景色（= 防线漏标，误伤）。
6. 与 S1 对照：K1 保留"无 LiDAR 处自由生长"，S1 不保留；若 K1 的远离比例落到 S1 的量级而 ROI / 离轨不差于 S1，说明 34% 里的大头就是天空拟合，S1 的硬规则可退役；若 K1 只降一半，剩下的是树 / 邻切片内容，走 B 线（全场粗先验做背景替身）。

## 7. 风险与缓解

* **分割错误**：玻璃、过曝的白墙、亮金属屋顶被 SegFormer 标成天空，α 项会把那里强制透明、光度项不再纠正——这是本机制唯一能"删掉真实内容"的路径。两道防线都做成了旋钮：`mask_erosion_px 8` 去掉标签边界的半像素级错分；`require_no_lidar_within_px 24` 让扫描仪打到的面周围一律不算天空——玻璃 / 过曝墙通常有回波（玻璃框、墙面），纯天空没有。代价是靠近屋檐边缘 24 px 内的真天空也不再受 α 项约束（那一带正是檐口凸起长出来的地方，可能压不干净）；K1 若檐口凸起仍在，先试 `require_no_lidar_within_px 12`。审计：`sky_raw_pixels − sky_pixels` 逐视角看防线吃掉的份额；mask 叠图看被救回的是墙还是天。
* **无回波的真实结构落在天空里**：细树枝、电线、远处建筑轮廓与天空混标（分割给的是像素级标签，树枝间隙是天空）。K1 在树枝像素上仍允许生长（标签不是天空），只在标为天空的间隙压 α；结果是树会"稀疏化"而不是消失——出图时看 pitch_up 面。
* **代理的漏与误**：投影中心在天空 = 阻断；一个脚跨在檐口边的大高斯中心在墙上不会被阻断（漏），一个中心刚过檐口线的墙面高斯会被阻断一次 refine（误，下一窗口重算）。`min_sky_fraction 0.5` 与 `since_last_refine_event` 窗口是常量，未做旋钮；若阻断数长期为 0 或恒等于候选数，先怀疑 `info["radii"]` 的可见性约定（非 packed 路径要求两个分量都 > 0）。
* **续训 / 热启动**：计数器随 topology ops 重索引，人口长度不符时重建为零（不报错）；`sky_growth_blocked_total` 存 strategy state 普通 int。
* **代价**：每步两次 `max_pool2d`（8 / 24 px 窗）+ 一次索引，全在 GPU；未在 GPU 机上实测。
* **未验证门禁**：真机训练、`sky_mask_train.json` 落地后的 `validate()`、GPU 上的 `_step_post_backward_mipmap` 全链（CPU 测试只覆盖到接线）、评估器口径未变但未重跑。
