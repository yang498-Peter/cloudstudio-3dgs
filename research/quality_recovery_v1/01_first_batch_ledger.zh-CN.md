# 质量恢复 v1 — 第一批证据台账（WP00 / WP01 / WP02，2026-09-05）

机器可读版本：`evidence_registry.json`（由 `tools/build_evidence_registry.py` 从磁盘工件重建）。本批**没有启动任何训练**。

## 1. PASS / FAIL / NOT_RUN

| WP | 项目 | 结论 | 证据 |
| --- | --- | --- | --- |
| WP00 | 身份冻结：G9 候选、G9d 四切片、`delivery_g9/merged.pt` + 两份交付 PLY | PASS | `identity/*.json`（config_as_run、15 项输入、checkpoint、编译扩展 `60e7a458…`、gsplat `f2d14131` + 锁定补丁反向应用干净） |
| WP00 | 缓存依赖 DAG | PASS | `cache_dependency_dag.json`：44 节点 / 87 边 / 23 个受保护根目录 |
| WP00 | CI 分诊 | PASS | `ci_failure_triage.md`：37 errors = 26 torch + 9 pytest + 1 gsplat + 1 pycolmap；56 个函数式用例从未被 `unittest discover` 收集 |
| WP00 | cpu 通道（本地 torch-free venv 复现） | PASS | `ci/collection-cpu-local.json`：collected 535 / passed 302 / NOT_RUN torch 218、pycolmap 11、gsplat 4 / 违规 0；10 个"应 skip 却 fail"的用例单列 |
| WP00 | torch-cpu 通道 | NOT_RUN | 本机无 CPU torch，只能在 Actions 验证；`torch==2.11.0` CPU wheel 可用性待 CI |
| WP00 | cuda 通道（B 机） | FAIL（已知） | `ci/collection-cuda.json`：collected 778 / passed 774 / skipped 0 / failed 4 = gate1 他人 WIP 的 preset `geometry_regularization` 问题，非本批范围 |
| WP00 | 审计脚本阴性对照 | PASS | cpu 通道 junit 用 torch-cpu 策略判 → 219 条违规（红） |
| WP02 | `_opacity_summary` RNG 隔离 | PASS | `tests/test_lifecycle_rng_isolation.py` 4 用例：修复前 3 失败（CPU 流前进、CUDA 流前进、两次读数不一致），修复后全过 |
| WP02 | cap 预算在实跑分支上被消费 | PASS | `CapacityBudgetOnSelectedPathTests` 2 用例：`pre_optimizer_vendor` + cap=5、3 候选 → `capacity_rejected_count=1`；无 cap → None/0。修复前 KeyError |
| WP02 | 预算遥测进 progress.jsonl | PASS（代码） | `lifecycle_capacity_cap / lifecycle_capacity_rejected` 两个键；只有下一次训练才会有读数 |
| WP02 | checkpoint 携带 RNG 流 | PASS（既有） | 所有训练 checkpoint 含 `torch_rng_state / cuda_rng_state / sampler_state`（identity JSON），`load_checkpoint` 恢复；合并产物不含（合理） |
| WP01 | checkpoint→PLY→checkpoint 张量往返 | PASS | means/scales/opacities/sh0/shN max_abs = 0；quats 2.4e-7（归一化 ulp） |
| WP01 | 同 backend 渲染往返 | PASS | 4 帧 max_abs 0.0012–0.0032（≤0.8/255），PSNR 115–120 dB；重复渲染差 0（前向光栅化确定性） |
| WP01 | 阴性对照 | PASS（修复后） | 相机平移 3.5 cm → PSNR 20.8；换背景 → 26.8；SH 降到 DC → 33.5（修复前 = 120.08，与往返差分逐像素相同，即对照无效） |
| 回归 | adapter / vendor parity / mipmap gate 套件 | PASS | 83 用例 OK；训练 venv + env 全量 pytest 774/778，失败 4 项同上 |

## 2. 本批发现（按对既有结论的杀伤力排序）

### 2.1 评估器把 SH1 模型钳成 DC-only（WP01 阴性对照抓到）
`GsplatBackend.render` 用 `min(self.sh_degree, active_sh_degree)`，而 `tools/sharpness_metrics._load_backend` 把 `self.sh_degree` 设成评估配置的 `sh_degree`。`delivery_eval.json` 与 `delivery_eval_v9.json` 都写着 `sh_degree: 0`（DC 时代复制的）。结果：**2026-09-05 之前对 G9/G9d 的每一个数字——battery、三方对比条、离轨迹条、锐度比值——都是 DC-only 渲染。** `merged.pt` 的 shN 不是零（|coef| 均值 0.054，87% > 0.01），所以不是"SH1 没学到"，是评估器瞎了。交付 PLY 在外部 viewer 里以 SH1 显示，和我们打分的图不是同一张。

修复：`_load_backend(config, sh_degree=None)`，并在 `evaluate_probe_views`、`build_three_way_compare`、`build_offtrajectory_compare`、`sharpness_metrics` 主流程、`roundtrip_checkpoint_ply` 中以模型阶数 `sqrt(K0+KN)-1` 覆盖配置。

用修复后的评估器重评**同一个** `delivery_g9/merged.pt`（无训练）：

| 指标 | F6（DC 模型） | G9 旧读数（DC-only 渲染） | G9 正确读数（SH1） |
| --- | --- | --- | --- |
| battery PSNR / p10（48 视角，v8 eval） | 19.33 / 16.70 | 18.74 / 16.44 | **19.49 / 16.83** |
| 训练视角锐度 ours/photo（同 6 帧，参考 0.574） | 0.156 | 0.175 | **0.211** |
| 训练视角锐度（v9 8 帧，参考 0.813） | — | 0.132 | **0.155** |
| 离轨迹（同 6 帧 ×3 位移） | 见 §2.1a | 见 §2.1a | 见 §2.1a |

之前写下的"G9 对 F6 仅 +12%，battery 低于 F6"是评估器缺陷；正确读数是锐度 +35%、battery 略高于 F6（差值在复跑噪声内）。**离参考仍差 2.7×。**

图：`wp01_roundtrip/zoom_dc_vs_sh1_vs_ref.png`（photo | ours DC | ours SH1 | reference，同帧同裁剪）。

### 2.1a 离轨迹（修复后）

| 集合 | 臂 | 锐度 ours/ref（中位） | PSNR vs 参考 1/4 res（中位） |
| --- | --- | --- | --- |
| 同 6 帧 ×3 位移（18 条） | F6 | 0.425 | 17.07 |
| | G9 DC-only 渲染 | 0.408 | 17.45 |
| | **G9 SH1** | **0.469** | 16.84 |
| v9 8 帧（F6 18 / G9 24 条） | F6 | 0.451 | 17.80 |
| | G9 DC-only 渲染 | 0.349 | 17.62 |
| | **G9 SH1** | **0.418** | 17.10 |

解读：SH1 打开后离轨迹锐度从"低于 F6"翻成高于 F6（+10% / −7%，两组方向不一致，均在噪声内）。PSNR vs 参考反而下降 0.3–0.6 dB——参考 PLY 是 DC-only（K=1），在位移视角上 SH1 的视角相关颜色必然偏离一个常色参考，加上 3.5 cm 对齐残差把该指标压在 ~17–20 dB 的地板上，所以这个 PSNR 不能用来判 SH1 好坏。离轨迹的可用读数只有锐度比值，且 G9 仍只有参考的 0.42–0.47。

### 2.2 遥测 randperm 真的改写了训练 RNG（WP02）
阈值 100 万元素；G9 候选 tile0 clone 父集峰值 1,847,926；G9d tile0 42 次 refine 事件越线、tile3 28 次、tile1 0 次。调用点在 `_grow_mipmap` 相对 280 行，`split_selected` 在 307 行，gsplat split 用无 generator 的 `torch.randn`。即遥测读数改变了 split 的偏移采样。同配方复跑 −16% 人口的分叉中，这是一个已证实的确定性来源；GPU atomicAdd 非确定性只是剩余部分。

### 2.3 cap_max 在 G9 实跑分支上是活预算（WP02，纠正旧结论）
`backend.py:204` → `settings["capacity_cap"]` → `_grow_mipmap` topk 预算与 `relaxed_cull_at_capacity`。"cap 完全失效"只对 `exact_mipmap_lifecycle=False` 的 inner 路径成立。G9/G9d 两次峰值都低于 cap，所以拒绝数应为 0，但历史遥测没有该键，无法反证；新键 `lifecycle_capacity_rejected` 从下次训练起记录。

### 2.4 gsplat 检出是脏的，但等于锁定 commit + 锁定补丁
`git apply --check -R` 反向应用干净；补丁 sha 与锁一致。编译扩展只在 `env_machine_b.cmd` 设的 `TORCH_EXTENSIONS_DIR=.torch-ext` 下可加载；不经该脚本启动的 venv 会去 Claude LocalCache 找扩展并报"CUDA extension not available"——这是环境问题，不是代码缺陷。

## 3. 作废 / 降级的历史结论
- 作废：所有含 G9/G9d SH1 模型的指标对比（G9 vs F6 +12%、battery 18.74/16.44、离轨迹 −4%、"SH1 是否值得"的判断）。
- 降级：同配方复跑 16% 差异 = "GPU 非确定性"→ 现为"遥测 RNG 污染 + GPU 非确定性"，比例未知，需要修复后复跑一次才能量化（属于后续 WP，本批不跑）。
- 保留：密度旋钮（grow_grad2d 5e-5）是唯一移动锐度的旋钮（G3/G9 对 G1/G8 均为 DC 对 DC 或经同一评估器，差异 +45% 远超噪声）。

## 4. 下一步最小任务（供负责人决定）
按方案决策树，测量可信度已恢复到"评估器带阴性对照 + 身份冻结"，可以进入 WP03/WP04 的最小版本：
1. **WP04 最小任务（推荐先做，无训练，~1 h）**：用修复后的评估器和冻结身份，把 G9 候选 tile0 与 G9d tile0（同配方复跑）重评一次，得到修复后的"评估噪声底"；再用修复了 RNG 的代码复跑一次 tile0 G9d（20k，约 2.5 h，**需负责人授权**）量化 randperm 污染占复跑分叉的比例。
2. **WP03 最小基线**：只在 Tile_0 上用最小配方（不带 alpha/normal/DA2 辅助项）跑一次，建立"最少假设"的锐度/battery 基线，与 G9 同帧同评估器比较。需授权。
3. 用户关心的"切片不自适应"属于方案 WP08/09 架构决策，本批不动。

## 5. 未验证门禁
- torch-cpu CI 通道、Actions 上的两个新 job（需 push 才能跑）。
- `lifecycle_capacity_rejected` 的真实读数（需一次训练）。
- 修复后复跑分叉比例（需一次训练）。

## 6. 研究循环结果（Tile_0，20k，seed 42，修复后评估器；规则见 `02_research_loop_plan.zh-CN.md`）

| 臂 | 变量 | 人口 | 训练视角锐度 ours/photo（6 帧中位） | 离轨迹锐度 ours/ref（18 条中位） | 短轴 p50 / 轴比 p50 | 判定 |
| --- | --- | --- | --- | --- | --- | --- |
| G9 候选 | 基线（cap 12M） | 9.26M | 0.281（参考 0.396） | 0.509 | 0.437 mm / 11.8 | 基线 |
| G9d t0 | 基线（cap 10M） | 7.75M | 0.259 | 0.482 | 0.449 mm / 11.8 | 基线（同配方复跑，−8% / −5%） |
| R0 | RNG 修复复跑（= G9d t0 配方，cap 10M） | 7.74M | 0.252 | 0.477 | 0.450 mm / 11.7 | 复现 G9d：人口 −0.15%、锐度 −2.7%、离轨迹 −1% |

**R0 结论（改写噪声认知）**：RNG 修复后同配方复跑与 G9d t0 几乎重合，所以本轮噪声带取 **锐度 ±3%、离轨迹 ±2%、人口 ±1%**。此前"同配方复跑差 16%"的候选 vs G9d 差异**不是噪声，是 cap**：R0 的新遥测 `lifecycle_capacity_rejected` 在 91 次 refine 事件上非零，单次最多拒绝 159 万候选（人口峰值 8.97M 未到 cap，但候选数 > cap 余量即被 topk 截断）。候选 cap 12M 峰值 10.75M / 终值 9.26M，G9d cap 10M 峰值 8.97M / 终值 7.75M——密度差 16% 全由 cap 解释。交付四切片的 cap 为 min(10M, 2×点数)，tile1/2 只有 6.8M/6.6M，等于全部切片都被 cap 掐着增殖。峰值显存约 0.9 GiB / 百万高斯（候选 10.75M → 9.42 GiB），16.3 GiB 卡可承受 cap 15M。

由此调整队列：D1（3.5e-5，cap 12M）在启动 5 分钟后终止（cap 会混淆阈值效应），改为 **C1 = 候选配方 + cap 15M**（单变量），随后 D1 改在 cap 15M 上跑。
| C1 | cap 12M → 15M（候选配方） | 11.50M（峰 13.27M，VRAM 11.29 GiB） | **0.298**（+6%） | **0.563**（+11%） | 0.424 mm / 11.8；long p95 34 mm（候选 38） | **赢，成为新基线**；cap 仍在 23 次事件截断（最多 125 万）；frac<0.1 升到 0.482 |
| D1 | C1 + `grow_grad2d` 5e-5 → 3.5e-5 | 11.53M（cap 绑定，36+ 次截断） | 0.287（−3.7%） | 0.514（−9%） | 0.438 mm / 11.3 | **拒**：cap 绑定时降阈值只是让 topk 选到更差的候选；首次运行在 step 8800 随宿主进程退出而中断，重跑于 19:22–22:32 |
| C2 | C1 + cap 15M → 17M | 12.07M（峰 13.9M，VRAM 11.8 GiB，**0 次截断**） | 0.302（+1.3%） | 0.545（−3%） | 0.422 mm / 11.8 | **中性**：cap 不再绑定后自然人口约 12M、死质量 0.487；阈值 5e-5 下密度已饱和，再抬 cap 无收益 |
| K1 | C1 + `post_refine_cull_every` 100（14k→20k 全程） | 3.77M（14k 时 12.76M，14.1k 9.79M，15k 6.4M，之后每 500 步再掉 2–5%，到 20k 仍在降） | 0.274（−8%） | 0.519（−8%） | 0.522 mm / 13.1；long p95 66 mm；opacity p50 0.73、frac>0.9 0.31 | **拒（按此配置）**：全程持续 cull 过度侵蚀，幸存者变大变实来补洞。但 15k 快照死质量已从 48% 降到 2.4% 且形态仍同 C1 → 机制有效、窗口太长 |
| K2 | C1 + cull 每 100 步、仅限 14000–15000 窗口（新旋钮 `post_refine_cull_until`） | 6.43M（窗口内 11.5M → 6.43M，之后冻结） | 0.267（−10%） | 0.495（−12%） | 0.471 mm / 12.4；long p95 47.5 mm；frac<0.1 回升到 0.269 | **拒，并推翻"死质量"假设**：清掉 opacity<0.1 的高斯后锐度反而明显下降，而且 5k 步后又有 27% 掉回 0.1 以下。这些低 opacity 高斯不是废物，是产生锐度的半透明叠层（竞品 opacity 中位数也只有 0.20）。K 线关闭；"透明/悬浮影子"要按位置（离面漂浮）定位，不能按 opacity 一刀切 |
| B1 | C1 + AbsGS 式增殖：`absgrad` True、`grow_grad2d` 8e-4、`lifecycle_execution_order` post_optimizer_gsplat | 排队（P2 之后） | | | | 文献中绝对梯度增殖专门缓解高频区糊化；vendor 顺序要求平梯度，故需一起换顺序（两变量，作为方向试探） |
| A1 | C1 + `da2_depth_weight` 0.15 → 0 | 11.50M（峰 13.27M，22 次截断） | 0.294（−1.3%） | 0.540（−4%） | 0.428 mm / 11.7 | **中性**：DA2 不是糊化来源，保留 |
| E1 | C1 + 日程 20k → 30k（refine stop 14k → 21k） | 排队（R1 之后） | | | | 竞品训练 49,560 步，是我们 2.5×；日程长度是唯一没对齐过的"竞品变量" |
| P2 | C1 + rig 位姿光度精调 + 去掉位姿绑定的 LiDAR 距离/alpha 项 | 排队（E1 之后） | | | | 根因排序①配准：G7 失败是因为位姿动了而 LiDAR 目标没动，这次一起放开 |
| L1 | C1 + `lidar_alpha_weight` 0.1 → 0.05 | 排队（A1 之后） | | | | 监督冲突：alpha 地板把枝间/天空推成饱和 |
| R1 | C1 + `lidar_range_weight` 0.05 → 0 | 排队（L1 之后） | | | | 监督冲突：LiDAR 距离项可能压平细纹理 |

**死质量血统审计（`tools/audit_dead_mass.py`，`dead_mass/*.json`，2026-09-06）**：C1 / 候选 / R0 三个 checkpoint 结构完全一致——**95% 的终态高斯出生在 10k–14k 这最后一个增殖窗口**，7.04M 个 LiDAR 初始化高斯只剩约 3 千个（0.03%）；死质量（opacity<0.1）46–48%，其中 clone 出生的死亡率 0.50–0.52、split 0.41–0.44；死高斯不是没被看到（观测计数 5–99 的桶死亡率一样是 0.48），不是地下（<0.5%），也不是大团（死亡率随长轴增大反而下降：1–5 mm 桶 0.56，>50 mm 桶 0.17）——它们是 14k 附近最后一次 reset 后没能恢复的冗余小高斯。机制：`_step_post_backward_mipmap` 在 `step >= refine_stop_iter` 直接返回，14k 之后既不 reset 也不 cull，这些高斯就被冻结进交付（导出只靠 `--min-opacity 0.05` 过滤，0.05–0.1 的半透明层就是用户看到的"透明/悬浮影子"），同时在 10k–14k 期间占着 cap 余量。下一臂 K1：增长停止后继续按周期只做 cull（新旋钮，默认关闭 = 厂商 parity）。

**针状/条状取向审计（`tools/audit_needle_orientation.py`，`needles/tile0_C1_vs_reference.json`）**：定义"针"= 长轴 ≥10 mm、轴比 ≥4、贴在 LiDAR 平面（planarity>0.5、距离<15 cm）且长轴与法向夹角 ≤30°；"条"= 同条件但夹角 ≥60°。C1 每千个贴面高斯有 1.73 根针、102 条条纹；**竞品是 2.60 根针、149 条条纹**——竞品的细长高斯比我们还多。用户看到的"针状凸起"不是我们细长高斯更多，而是我们的少而不透明（Tile_0 活跃仅 5.96M，竞品全场景活跃 17.6M、opacity 0.26 半透明叠加），单根条纹裸露；竞品把同样的条纹埋在密集半透明叠层里成为纹理。结论：不上形状惩罚（G1/G2 已证中性，且会掐掉竞品也在用的形态），继续攻活跃密度与死质量。Laplacian 锐度对针不敏感（针反而抬高它），出图必须看。

C1 解读：密度确实是杠杆——人口 +24% 换来训练视角锐度 +6%、离轨迹 +11%，针状长尾还缩短了。但 48% 的高斯 opacity < 0.1（参考 18%），也就是 15M 预算里约 5.5M 是死质量在占余量；下一步除了阈值/cap，还要考虑把死质量清出预算。显存按 0.85 GiB/百万高斯峰值算，cap 17M（峰约 15M → ~12.8 GiB）仍可行，列为 C2 候选。

注：tile 配置本身 `sh_degree: 1`，所以这两条基线的条带本来就是 SH1 渲染，不受 §2.1 缺陷影响，可直接与后续臂比较。参考形态：短轴 0.43 mm、轴比 10.2、opacity p50 0.197、frac<0.1 0.18；我们 frac<0.1 为 0.44–0.46（死质量仍是参考的 2.5×）。
