# WP06 X1 — 每相机时间曲线曝光（frozen camera curve）：定义、全场拟合数字、X0/X1 判读

日期 2026-09-11 · 分支 `cloudstudio-3dgs-work`（研究分支，未提交） · CPU-only，无训练、无渲染

> 本文是 `06_exposure_findings.md` §8 X1 臂的实现说明与拟合读数。曲线由
> `tools/fit_exposure_curve.py` 从四个 R1 切片 checkpoint 已学到的每图 gain
> （`06_photometric_consistency.csv`，2793 行）拟合，**没有训练任何模型**；X0/X1 的
> 训练视角对照留待 GPU 队列。

## 0. 落地文件

| 文件 | 内容 |
|---|---|
| `cloudstudio_3dgs/training/exposure.py` | `ExposureCompensationConfig` 新增 `mode / knot_seconds / prior_weight / frozen_curve`（缺省 `per_image`，契约字典只在 `mode != per_image` 时多出键）；`ExposureCurve` 模型；曲线文件读写与纯 python 求值助手 |
| `cloudstudio_3dgs/training/trainer.py` | `camera_curve` 分支：从 dataset manifest 取 `timestamp_ns`，构造 `ExposureCurve`，注册 `auxiliary_params["exposure_curve_knots"]`；frozen 时**不建优化器**；`warm_start_fresh_auxiliary` 支持 `exposure_curve_knots`；frozen 曲线计入 `frozen_photometric_nuisance` |
| `tools/fit_exposure_curve.py` | 全场曲线拟合（最小二乘 + 平滑 + 软锚 + Huber IRLS + 切片偏置），输出曲线 JSON、逐行 CSV、edf、λ 扫描、留一切片验证 |
| `tools/make_exposure_curve_arm_config.py` | 由 X0（`*_40_R1_c134`）派生 X1 配置，`--validate` 走 `TrainerConfig.from_dict(...).validate()` |
| `research/quality_recovery_v2/06_exposure_curve_scene.json` | **冻结的全场曲线**（sha256 `75a1e5d4…bdb2f5`）；同一字节复制到 `C:\Peter\3dgs-runs\house0305_sop\exposure_curve_scene_R1.json`，X1 配置引用后者 |
| `research/quality_recovery_v2/06_exposure_curve_fit.csv` | 2793 行：每图×切片的学到 gain、曲线值、切片偏置、拟合值、残差、Huber 权重 |
| `run_configs/house0305_tiles/diag_v2/diag_{indoor_door_leaf_Tile_1,outdoor_gravel_Tile_0}_40_X1_c134.json` | X1 臂配置（与 `C:\Peter\3dgs-runs\house0305_sop\` 中同名文件逐字节相同） |
| `tests/test_exposure_curve.py`（13 项）、`tests/test_fit_exposure_curve.py`（5 项） | 见 §5 |

## 1. 曲线定义（代码事实）

* **参数化**：每台物理相机 `c` 一条分段线性 log-gain 曲线 `f_c(t)`，节点在
  `t_k = origin + k·K`，`K = knot_seconds`（缺省 10 s）。图像 `i` 的 gain 为
  `g_i = exp(clamp(f_{cam(i)}(t_i), ±ln 2))`，`t_i` 是 dataset manifest 的 `timestamp_ns`；
  clamp 与 per_image 相同（`max_abs_log_gain`）。节点值是唯一参数；两台相机的节点拼成一个
  一维张量 `exposure_curve_knots`（checkpoint 里的 auxiliary 参数名），与 per_image 的
  `exposure_log_gains` 互斥。
* **节点数**：`floor(span/K) + 2`，最后一个节点 ≥ 该相机最后一帧；网格之外常数外推。
  house0305 跨 221.0 s → 每相机 24 个节点（0–230 s），共 48 个参数（对比每图 884 个）。
* **应用位置不变**：仍只在 `_render_supervision_loss` 里乘到渲染上（`rgb_gain`），
  验证/评估/交付渲染 gain=1.0；`decoupled_ssim` 语义不变（gain 仍是标量）。
* **先验（可学习模式）**：`prior_weight · mean_j (k_{j+1} − k_j)²`（每相机相邻节点 L2 平滑）
  + `mean_anchor_weight · SmoothL1(mean_i∈c f_c(t_i), 0; β)`——锚在**该相机训练图像上的曲线均值**
  （是节点的固定线性泛函，构造时预计算），不是节点均值；不施加 per_image 的零拉 L2；
  硬 zero-mean 投影在此模式下被 `validate()` 拒绝（P8 probe 结论）。
* **frozen_curve**：给出路径时从文件读节点、`requires_grad=False`、`make_optimizer()` 返回
  `None`、`prior_loss()` 恒 0；配置的 `knot_seconds` 必须等于文件的，图像所属相机必须在文件中。
  契约字典多出 `frozen_curve_sha256`，所以 `trainer_config_sha256` 钉住曲线内容。
* **per_image 缺省字节不变**：`ExposureCompensationConfig().to_dict()` 仍是原 7 个键；
  R1 臂的 `{"enabled": true, "learning_rate": 0.005}` 契约与旧版逐键相等（测试钉住）。

## 2. 全场拟合（`fit_exposure_curve.py`，无训练）

模型：`log g[i, T] = f_{cam(i)}(t_i) + b_T`。`b_T` 是每切片一个 nuisance 偏置——每块模型帧
自己的亮度（合并烘焙掉的中位 gain 0.88–0.93），不是曝光，不得进入共享曲线。
观测取**应用过的**（clamp 后）log gain；Huber IRLS（δ = 0.15，10 轮）压制 41 行 clamp 饱和值
与每块残差离群；`λ_s = 3`（平滑）、`λ_a = 1e4`（二次软锚，锚在各相机唯一图像上的曲线均值）。

### 2.1 主要数字

| 项 | 值 |
|---|---|
| 观测 / 唯一图 / 时长 | 2793 行 / 884 图 / 221.0 s |
| 节点 | 每相机 24（0–230 s），曲线参数 48；含 4 个切片偏置共 52 |
| **有效自由度**（hat 矩阵迹） | 曲线 **36.6**（left 18.1 / right 18.5）；含切片偏置 40.6 |
| 曲线范围（log） | left [−0.223, +0.250]，right [−0.303, +0.254]；全场 886 图求值 \|log g\| P50 0.073 / P95 0.236 / max 0.303，无饱和 |
| **每相机均值** | left +0.00099 / right −0.00099（软锚残留；两相机的 2 % 常数差被平分给模型） |
| 切片偏置 `b_T` | T0 −0.114 / T1 −0.116 / T2 −0.148 / T3 −0.066（gain 0.892 / 0.890 / 0.863 / 0.936；与合并中位 0.882 / 0.935 / 0.915 / 0.933 同量级，是每块 canonical 帧比照片亮 7–14 % 的那项） |
| 残差 RMS（全部 / 去饱和行） | **0.1507 / 0.1378**；MAD 0.076；P95 \|r\| 0.319 |
| 基线 RMS：只有切片偏置 / 偏置 + 相机常数 | 0.1945 / 0.1942（相机常数解释不了任何东西，与 06 §5 R² < 0.02 一致） |
| **R²（对只有切片偏置）** | 0.400（去饱和行 0.428）；left 0.412、right 0.392 |
| 按切片 RMS | T0 0.142 / T1 0.151 / T2 0.163 / T3 0.148 |
| 按环境 RMS | indoor **0.205**（n 902）/ covered 0.157（343）/ outdoor **0.108**（1411）/ stationary 0.062（137） |
| 饱和行（41） | 残差 RMS 0.52、均值 −0.45、平均 Huber 权重 0.31；21.8 % 的行权重 < 1 |

读法：曲线吃掉每图 gain 方差的 40 %（去饱和 43 %），与 06 §5 "camera×10 s 块 adj R² 0.43–0.53"
一致——那是每块各自拟合块均值的上限，这里是**一条跨四块共享**的曲线。剩下的 60 % 是逐帧、
且 06 §2 已证约一半跨切片不可复现的模型残差；X1 按设计不承载它。室内残差 0.205 ≈ 室外的 1.9×，
与 06 §1 的离散比例一致。

### 2.2 平滑权重扫描（GCV 与 edf）

| λ_s | RMS | edf（曲线） | GCV |
|---|---|---|---|
| 0.1 | 0.1489 | 44.3 | 0.01441 |
| 1 | 0.1493 | 41.0 | 0.01439 |
| **3** | **0.1507** | **36.6** | 0.01448 |
| 10 | 0.1561 | 28.5 | 0.01504 |
| 30 | 0.1654 | 20.1 | 0.01626 |
| 100 | 0.1760 | 12.4 | 0.01804 |
| 1000 | 0.1887 | 3.9 | 0.02072 |

GCV 偏好 λ ≤ 1（几乎不平滑），但残差 lag-1 自相关 0.8–0.9（06 §5）使 GCV 系统性欠平滑；
λ 由 §2.3 的跨切片迁移决定。

### 2.3 留一切片验证（"frozen for all tiles" 的实测依据）

在三块上拟合，预测第四块的 gain（该块自己的偏置取中位数，自由）：

| 留出 | 预测残差 RMS | 只有偏置 | R² | left / right |
|---|---|---|---|---|
| Tile_0 | 0.1555 | 0.1840 | 0.286 | 0.145 / 0.163 |
| Tile_1（室内门叶所在） | 0.1684 | 0.2208 | **0.418** | 0.163 / 0.172 |
| Tile_2 | 0.1849 | 0.1948 | 0.099 | 0.162 / 0.206 |
| Tile_3 | 0.1625 | 0.1879 | 0.252 | 0.149 / 0.174 |

* 三块学到的曲线能解释第四块 10–42 % 的方差，Tile_1（室内主战场）最好、Tile_2 最差
  （right 相机 0.206：Tile_2 的 right gain 有本块特有的成分）。λ_s ∈ {0.3, 1, 3, 10, 30} 的留一
  平均 RMS 分别 0.1691 / 0.1684 / **0.1678** / 0.1701 / 0.1757，取 λ_s = 3。
* 同一扫描下 `K = 5 s` 的留一平均 RMS 0.160（λ 0.3–1，edf ≈ 80），比 10 s 好 5 %；
  `K = 20 s` 0.178。本轮按 06 §8 的 10 s 定义交付，5 s 是 X1 若不劣于 X0 之后的下一档。

## 3. "冻结给所有切片"保证了什么、不保证什么

保证（构造上成立，`tests/test_exposure_curve.py::test_frozen_curve_reproduces_given_gains` 钉住）：

1. 同一张图在任何切片、任何步数的 gain 恒等（06 §2 的跨切片差中位 0.18、室内 0.29 → **0**），
   合并后接缝两侧不再各自"拟合"过不同的每图残差；
2. 曝光自由度从每块 638–784 降为全场 48，且训练中不再更新——gain 不能再吸收模型残差；
3. 每相机曲线均值 ≈ 0（|mean| ≤ 1e-3），全场亮度自由度留在模型里；四块的"canonical 比照片亮
   7–14 %"不再通过每块不同的 gain 中位数表达，而由每块模型各自吸收 `b_T`。

不保证 / 需要注意：

* **子集臂的均值不为零**：DIAG-40 只训练 40 张父图，冻结曲线在这 40 张上的均值
  室内 left −0.172（6 张）/ right +0.025（34 张），室外 left −0.102 / right −0.016。
  该均值会由 DIAG 模型的亮度吸收，所以 X1 与 X0 的 canonical 帧亮度**不同**（X0 的 canonical
  比照片亮约 10 %，X1 的约等于照片扣除上述均值）。判读见 §4。
* 曲线只是每图 gain 的可迁移部分（R² 0.1–0.42），不是 06 §4 oracle 的多视图一致性下限；
  室内剩余的逐帧亮度矛盾（06 §4：learned 只做了 1/3 的工作）X1 不会减少，反而会把 learned
  在室内吃掉的那 16 % CV 还回一部分（06 §4：10 s 块变体 −10 %，learned −16 %）。
* `merge_v28_tile_checkpoints.py --harmonize-exposure` 读 `exposure_log_gains`，X1 checkpoint
  没有该键会报错——X1 切片合并时**不应**再做每块中位烘焙（均值已锚定），这是设计意图，
  但合并工具尚未为 `exposure_curve_knots` 增加分支（未做）。
* `render_spec.py::_resolve_exposure` 不区分 mode；X1 的 `training` 子记录仍报告
  `mean_anchor_weight` 等旧字段，`mode`/`frozen_curve_sha256` 只在 `trainer_config_sha256` 里（未做）。

## 4. X0 / X1 如何判

* **X0** = `diag_<region>_40_R1_c134`（已跑：`diag_v2/<region>/runs/…_40_R1_c134`），per_image，
  L2 1e-2，无锚，每块独立学习。不新建文件。
* **X1** = 同一配置 + `exposure_compensation.mode = camera_curve, knot_seconds 10, frozen_curve =
  exposure_curve_scene_R1.json`（`prior_weight 0.01`、`mean_anchor_weight 1.0` 在 frozen 下无效，
  记录给将来的可学习臂）。`run_id` `diag_<region>_40_X1-c134`，`output_dir` 叶子 `…_40_X1_c134`。
  两份配置与 X0 只在 `run_id / output_dir / exposure_compensation / diag` 四个键上不同
  （`diag.variant = X1`，`diag.x1` 记录基配置与曲线 sha256）；均已通过
  `TrainerConfig.from_dict(...).validate()`。
* 判据（06 §8，室内、室外分别报，不取平均）：
  1. **配对 CV**：`audit_exposure_gains.py` §4 的同一表面跨视图亮度 CV，同一 LiDAR 采样点、同一
     有效视角集合下配对比较 X0 `learned` 与 X1（gain = 冻结曲线求值，可在训练前离线算——
     本轮未算，需要照片投影），报告逐点比值中位与改善点比例；预期室内 X1 略劣于 X0
     （−10 % vs −16 %）、室外 X1 优于 X0（−11 % vs −1 %）。
  2. **canonical 渲染锐度**：训练视角与离轨迹（`build_three_way_compare` / `build_offtrajectory_compare`，
     `01_eval_protocol.json` 的 RenderSpec 指纹逐字段一致）。因 §3 的亮度差，锐度比值必须在
     **亮度归一后**比较（Laplacian 方差随亮度平方缩放）：或对 X0/X1 渲染先匹配平均亮度到参考，
     或同时报告 train-compensated 帧（X0 乘每图 gain、X1 乘曲线 gain）。只报 canonical 一帧的锐度
     不能判臂。
  3. **保留集 P10**（`02` 口径）与同帧 PSNR，同样两帧各报一次。
  4. **跨切片一致性**：X1 构造为 0，无需测；接缝两侧 DC 亮度差要等四块 X1 全量跑完后在合并件上量。
* 进入下一级（每相机 RGB 3 维、PPISP）的条件不变：X1 不劣于 X0，且室内 raw CV 仍显著高于
  X1 后残差。

## 5. 验证

* 新测试 18 项通过：`tests/test_exposure_curve.py` 13（节点处与节点间求值、网格外常数外推、
  梯度只到两个 tap、clamp；软锚把每相机均值压到 |mean| < 1e-3 而保留曲线形状、左右对冲仍被罚；
  平滑项对常数曲线为 0；frozen 曲线逐图复现文件 gain、无优化器、两份不同子集同图同 gain、
  `knot_seconds` 不符 / 相机缺失 / 时间戳缺失被拒；per_image 缺省契约逐键相等且
  `ExposureCompensator` 索引/分组/前向/先验/报告与显式 `mode="per_image"` 完全相同；
  `TrainerConfig` 契约键只在 camera_curve 下增长、warm-start fresh 名按 mode 校验；
  trainer 源码钉住 camera_curve 分支为 per_image 的显式替代）；`tests/test_fit_exposure_curve.py` 5
  （合成已知曲线 + 4 切片偏置 + 2 % 饱和离群：节点误差 < 0.03、偏置误差 < 0.01、离群 Huber 权重 < 0.5、
  去饱和 R² > 0.9、留一切片 R² > 0.9、edf 单调；CLI 往返进 `ExposureCurve` 逐图相等）。
* 既有：`tests/test_schedule_contract.py` 27 通过（`python -m unittest tests.test_schedule_contract` OK）、
  `tests/test_alpha_support.py` 9 通过、`tests/test_audit_exposure_gains.py` 17 通过、
  `tests/test_training.py` + `test_render_spec.py` + `test_ppisp.py` 111 通过、1 失败
  （`test_rendered_footprint_matches_linear_metric_scale`：gsplat lazy CUDA 对象，本机无 CUDA 约束下
  的既有失败，与本次无关）。
* 干跑（CPU）：用真实 dataset manifest + DIAG-40 tile_inputs 构造 frozen `ExposureCurve`：
  室内 40 图（left 6 / right 34）gain 0.842–1.132，室外 40 图（20 / 20）0.838–1.081；全场 886 图
  （含 2 张 val）均在曲线覆盖范围内（最后一帧 221.0 s < 最后节点 230 s）。
* 未做：任何训练与渲染；X1 变体的 §4 配对 CV；合并工具与 render_spec 对 camera_curve 的分支。
