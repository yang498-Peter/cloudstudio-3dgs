# CloudStudio 专业缺陷扫描（snow 表面初始化第 1 轮）

- 扫描协议：`v2`
- 扫描深度：`module-deep`
- 扫描基线：`926ce5ab8438198f2b0e21257f3469a135fa6d80`
- 提交窗口：`edf52a7d278786d0b2235ac5b396a029a5533905..926ce5ab8438198f2b0e21257f3469a135fa6d80`
- 扫描时间：`2026-08-26`
- 包含范围：`LiDAR 初始化 -> PCA 局部几何 -> Trainer 参数 -> gsplat 运行时 -> checkpoint/锐度审计`
- 排除范围：`训练中出生高斯、Browser/WebView2、安装包、外部 PLY Viewer、发布 CI`
- 结论：**CONDITIONAL GO**
- 判定适用范围：`表面初始化源码、真实 snow CUDA 长训、完整评估和 PLY 结构`
- 修改边界：`verify-and-fix`

## 结论摘要

发现并修复 1 个既有 P1 产品缺陷：PCA 法向已经由初始化工具产出，但 Trainer 丢弃该文件，
导致全部高斯按等轴球和单位旋转开始训练。修复后，414 万 snow 高斯的尺度分布与同源 SH2
竞品接近；按排除天空、只统计 LiDAR 几何支持像素的修正口径，2k 固定步数锐度能量从
旧 30k 的 `0.707` 提升到 `0.777`。Pass B 又发现并修复
1 个修复中新引入的 P1 门禁缺陷：补丁 checkout 的哈希验证最初未覆盖未跟踪文件。47 个
测试及登记补丁上的真实 CUDA 尺度测试通过；warm 2k + 6k 正式训练、36 帧 LPIPS 报告和
三份 PLY 已完成。最终 6k 主模型的几何区域锐度能量提高约 12.3%，但 PSNR/深度/空洞
回归，综合仍为 MIXED；外部
PLY Viewer 尚未执行，故不能给成熟竞品画质 GO。

## 覆盖模型与责任矩阵

### 生产者—消费者图

```text
colorized.las -> build_lidar_init(PCA) -> sparse_pc.ply + lidar_init_geometry.npz
              -> Trainer scale/quaternion initialization -> gsplat CUDA
              -> signed checkpoint -> sharpness/scale audit -> PLY export
```

| 边 | 首次成功 | 重跑失败/迟到 | A→B/owner | 数值/格式边界 | 取消/恢复 | 状态 |
|---|---|---|---|---|---|---|
| LAS -> PCA 初始化 | 112 万、414 万真实 snow 均完成 | 确定性 seed 42 | 同一 CLI | 点数/哈希/行数 | 未测取消 | covered |
| PCA -> Trainer | 414 万 SH2 CUDA 烟测和 2k 完成 | 缺文件/行数错/NaN 失败关闭 | 几何 SHA 绑定身份 | mm-cm 尺度 | resume 未测 | covered |
| Trainer -> gsplat | 锁定补丁实跑 | patch/diff/untracked 负控 | 运行时证据进 manifest | 8 GB 显存 | 进程取消未测 | covered |
| checkpoint -> 评估 | 1.5k/3k/4.5k/6k 永久保留 | latest 与 best 分开判读 | run/split 身份独立 | PSNR/SSIM/depth/LPIPS/sharpness | 完成 | covered |

### 未执行矩阵单元

- `PLY -> 外部 Viewer 固定视角`：未执行；这是客户可视质量 GO 的阻塞项。
- `训练中 densification`：最新固定步数证据已判为净负，本轮有意排除，不属于遗漏。
- `Browser/WebView2/安装包/CI`：本轮算法模块不触及这些消费者，保持 NOT_RUN。

## 测试预言机审计

| 检查 | 结果 | 证据 |
|---|---|---|
| 源/中间/结果/重导入身份可区分 | yes | LAS、PLY、geometry、config、checkpoint 哈希分别绑定 |
| 协议与真实 start/poll/result/cancel 一致 | partial | start/训练/结果真实执行；cancel 未执行 |
| 未主动执行用户 workaround | yes | 没有用旧 best checkpoint 或后处理锐化替代训练 |
| 至少一条真实算法/运行时边界 | yes | 414 万 SH2，8 GB GPU 完成 warm 2k + 6k 正式训练 |
| 乱序、失败和取消被确定性制造 | partial | 坏几何/篡改运行时已制造；取消未执行 |

## 第一轮结果与独立复审

### Pass A：症状/差异扫描

- 从“旧模型 64.8% 可见高斯宽于 5 px”反向追到初始化链：PCA 文件存在但 Trainer 无消费者。
- 红测先因 `surface_initialization` 模块不存在失败；修复后法向对齐、非平面回退和坏数据
  三组用例通过。

### Pass B：正交攻击

- 独立审查：`同一 agent 的正交 Pass B；当前协作策略不允许自行创建子代理`
- 未复用的维度：`运行时供应链篡改`、`数值非有限/行数边界`、`8 GB CUDA 容量和竞品结构尺度`
- 新发现：`1，R1-P1-RUNTIME-UNTRACKED-BLIND-SPOT`
- 重复/排除：`densification 退化为最新提交已知结论，不重复计数`

## 新发现

### R1-P1-PCA-SURFACE-DISCARDED：Trainer 丢弃已计算的 LiDAR 表面几何

- 发现类别：`product`
- 来源分类：`pre-existing`
- 问题现象：30k 模型仍像大块半透明球，雪块、沥青和建筑细节发糊。
- 最小场景：非 z 轴平面法向输入时，旧 Backend 仍生成单位四元数和三轴相同尺度。
- 根因链：`--with-pca` 产出法向/特征值 -> Trainer 只读 PLY XYZ/RGB -> Backend 固定单位旋转 ->
  8.64 cm 球状高斯覆盖多个像素 -> 空间高频无法表达。
- 客户影响：PLY 可导出但不可作为成熟产品画质交付。
- 代码位置：`cloudstudio_3dgs/training/trainer.py`、`backend.py`、`surface_initialization.py`。
- 独立复现：`tests/test_surface_initialization.py`。
- 正向控制：414 万真实 snow 2k 几何区域锐度能量 `0.777`，最长轴 p50 `1.31 cm`。
- 与历史问题的区别：BA 已通过且无法修复该模糊；这是表示尺度/方向断链，不是 POS 刚体误差。
- 修复边界或建议：只消费与初始化 PLY 行数严格一致、哈希绑定的 PCA 文件。
- 当前状态：`fixed；正式锐度改善通过，综合质量 MIXED`

### R1-P1-RUNTIME-UNTRACKED-BLIND-SPOT：补丁运行时身份初版未覆盖未跟踪文件

- 发现类别：`gate-infrastructure`
- 来源分类：`repair-introduced`
- 问题现象：初版 locked-patch 验证只哈希 `git diff HEAD`，未跟踪 Python/CUDA 文件不在 diff 中。
- 最小场景：模拟 checkout diff 为空但 `git ls-files --others` 返回 `rogue.py`。
- 根因链：放宽 clean checkout -> 只验证 tracked diff -> untracked 文件可改变 import/build -> 证据不完整。
- 客户影响：可能把未锁定运行时误报为可复现模型。
- 代码位置：`cloudstudio_3dgs/training/backend.py`。
- 独立复现：`test_runtime_verification_rejects_untracked_checkout_files`。
- 正向控制：当前登记补丁返回 `source_kind=locked_patch`，补丁和 checkout diff SHA 均匹配。
- 与历史问题的区别：这是本轮兼容登记补丁时引入的门禁风险，不是产品渲染缺陷。
- 修复边界或建议：登记补丁继续允许；任何未跟踪运行时文件一律失败关闭。
- 当前状态：`fixed`

## 修复中新风险

- `R1-P1-RUNTIME-UNTRACKED-BLIND-SPOT` 已在 Pass B 中修复并增加耐久负控。

## 门禁接线与强制执行

| 测试 | 聚焦 | static/numeric | 默认套件 | Browser/WebView2 | 发布预检 | CI |
|---|---|---|---|---|---|---|
| `tests/test_surface_initialization.py` | PASS | PASS | 可发现 | N/A | NOT_RUN | absent |
| runtime lock 负控 | PASS | PASS | `tests.test_training` 可发现 | N/A | NOT_RUN | absent |
| 414 万真实 CUDA | PASS(long run) | PASS | 非默认 | N/A | PLY 结构 PASS | absent |

- 发现方式：`unittest discovery + 显式真实 GPU 命令`
- 失败短路检查：`本轮不含 browser/release lane，未伪报执行`
- 未接线风险：`真实长训仍是人工/实验门禁，不在默认单元测试中`

## 验证证据

```text
python -m unittest tests.test_training.TrainingContractTests tests.test_reproducible_baseline tests.test_surface_initialization -v
Ran 23 tests ... OK

python tools/train_gsplat.py --config ...snow_dense4144k_planar_sh2_smoke10.config.json
10 steps complete; peak torch VRAM 3,828,968,960 bytes

python tools/train_gsplat.py --config ...snow_dense4144k_planar_sh2_probe2k.config.json
2,000 steps complete; peak torch VRAM 3,977,996,288 bytes

python tools/train_gsplat.py --config ...snow_dense4144k_near_fixed_warm2k_sh2_final6k_v4.config.json
6,000-step phase complete; peak torch VRAM 3,683,377,152 bytes

python tools/sharpness_metrics.py ... --views 0 10 20 30 --exclude-sky --crop-metrics
step6000 geometry energy 0.794; agreement 0.453; holes 0.7%; crop energy 0.819

python tools/evaluate_run.py ... --lpips --require-complete
status COMPLETE; 36 frames; PSNR 18.67; SSIM 0.577; LPIPS 0.463; depth MAE 4.74 m

python tools/export_gaussian_ply.py ...
two full SH2 PLYs with 4,143,881 Gaussians and one optional opacity-pruned PLY exported
```

## 已检查但未发现新缺陷

- 四元数 wxyz 约定与 gsplat Backend 参数顺序一致，三组轴向法向数值测试通过。
- 414 万点 PCA 文件与 PLY 行数、哈希一致；约 94.6% 点按表面 surfel 初始化。
- 固定拓扑容量 4,143,881 未发生出生或删除，未复现历史 runaway densification。

## 边界与未验证项

- 36 帧完整 LPIPS 和 PLY 导出已完成；外部 PLY Viewer 固定视角仍 NOT_RUN。
- 两个竞品 PLY 坐标帧不同，本轮只比较文件内结构参数，不比较跨帧 agreement。
- Browser/WebView2、安装包、发布 CI 均 NOT_RUN。

## 并行修改说明

- 冻结后新增修改：工作区原有 `trainer.py` warm-start/checkpoint retention 等并行改动继续存在；
  本轮未回退、未整文件格式化。
- 受影响复现是否重跑：`yes，47 个测试、真实 CUDA 尺度测试和正式 GPU run 已执行`

## 建议修复顺序

1. 在外部 Viewer 以固定视角检查主/备 PLY 的雪面、地面和建筑边缘。
2. 下一轮针对 45.9% 低不透明度薄雾和 0.7% 几何区域空洞做单变量优化，不能用后处理伪装。

## 发布判定

- 源码与自动化：`GO（47 个测试和真实 CUDA 尺度测试通过）`
- Browser/WebView2：`NOT_RUN / 不适用本算法扫描`
- 真实数据/外部软件：`真实 snow 长训、完整报告和 PLY 结构 GO；外部 Viewer NOT_RUN`
- 安装包/发布环境：`NOT_RUN`
- 最终适用结论：`CONDITIONAL GO；结构锐度改善通过，但综合质量 MIXED，外部 Viewer 未验收`

## 2026-08-26 第二轮追加审计

- Stage 2 / 2 cm POS 先验：`PASS`。306 张图重投影 p50 从 `1.657 px` 改善到
  `1.092 px`，固定基线和尺度门通过；求解器 `NO_CONVERGENCE` 作为剩余风险保留。
- 在线姿态：强/弱先验候选均由训练器自动 `REJECTED` 并回退原姿态，未改善固定验证集。
- upstream DefaultStrategy：raw-fisheye 3DGUT 缺少所需屏幕空间梯度，两次探针均在
  checkpoint 前失败关闭；改用显式暖启动 opacity 行裁剪，不伪报默认策略可用。
- 透明度裁剪：0.05 删除 39.37%，保留 2,512,501 高斯且综合质量近似；0.10 继续损失
  PSNR/SSIM；5 cm 尺度硬截断导致 PSNR `12.95 dB`，判为负结果。
- LiDAR RGB 支持区和 RGB 梯度损失：均有签名契约、公式测试和真实 CUDA A/B；最多只
  带来微小 PSNR/SSIM 收益，锐度下降，未晋级。
- v11a 中心微调：36 帧 `COMPLETE`，PSNR `18.7327 dB`、SSIM `0.57842`、LPIPS
  `0.45882`、深度 `4.7346 m`、覆盖率 `99.697%`；LiDAR NN p95 `4.40 mm`，无 >30 cm
  漂浮点。锐度 `0.768/0.455`，仍低于 v4/6000 的 `0.794/0.453`。
- 新 PLY：`snow-20260224-dense4144k-sh2-v11a-balanced.ply`，4,143,881 SH2 高斯，
  679,597,491 bytes，SHA256
  `fe38b7abf81d98066873a60112f7f20464b335bcd92edf9512eb2e9cf1ce5893`。
- 自动测试：`Ran 54 tests ... OK`，包含真实 CUDA 尺度测试。
- 追加判定：`CONDITIONAL GO` 不变。v11a 是综合平衡候选，v4/6000 是高锐度候选；
  成熟竞品画质和外部 PLY Viewer 固定视角仍 `NOT_RUN/未通过`。
