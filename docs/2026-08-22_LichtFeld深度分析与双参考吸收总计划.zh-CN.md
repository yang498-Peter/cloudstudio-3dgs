# LichtFeld Studio 深度分析 + Spirula 双参考吸收总计划（2026-08-22）

## 1. 仓库确认

- **URL**: https://github.com/MrNeRF/LichtFeld-Studio（作者 MrNeRF/janusch，README 确认 MCMC/bilateral grid/3DGUT/headless/Python 插件/MCP）
- **本地克隆**: `C:\Peter\lichtfeld-studio`（--depth 1）
- **Commit**: `268d5fd4d2f8579f803a424a2dd793127a429778`（2026-08-22 00:21 +0200，与克隆同日，极活跃）
- **License**: GPLv3（根 LICENSE，每文件 SPDX `GPL-3.0-or-later`）。**任何代码禁止复制进我们 Apache-2.0 仓库；只允许思路/公式/超参重实现。** 例外通道：LichtFeld 的 PPISP 设计与 Spirula 相同，上游是 NVIDIA `nv-tlabs/ppisp`（Apache-2.0），从上游 port 不受 GPL 约束。
- 技术栈：C++23 + CUDA 12.8+，自研 tensor 库（无 LibTorch），主光栅化路径 FastGS（pinhole），3DGUT 路径为 gsplat 移植 + 3DGRUT 相机代码（`src/training/rasterization/gsplat/Cameras.cuh` 头注明 "referenced from 3DGRUT codebase"）。
- 默认超参真相源：`src/core/include/core/parameters.hpp`（`OptimizationParameters` 结构体内联默认值）+ `src/core/parameters.cpp`（各策略 preset 覆盖）。

## 2. 逐项机制分析（含文件路径与公式）

### 2a. MCMC 工程化（`src/training/strategies/mcmc.cpp` + `src/training/kernels/mcmc_kernels.cu`）

**MCMC 默认超参**（`parameters.hpp`，mcmc_defaults 即结构体默认）：
- iterations 30k；refine_every=100，start_refine=500，**stop_refine=25000**（最后 5000 步纯收敛，无 relocate/add，但噪声继续、此时已衰减到 ~2% 初值）
- max_cap=1M；min_opacity=0.005；opacity_reg=0.01，scale_reg=0.01；sh_degree_interval=1000
- means_lr 1.6e-5 → 1.6e-7（×scene_scale，见 2c）；scaling_lr 5e-3 恒定；opacity_lr 0.025；shs_lr 2.5e-3

**Error-score weighted relocation/add 抽样**（对照我们 gsplat.MCMCStrategy 按 opacity 抽样的最大差距）：
1. 每步 loss 时从 SSIM workspace 取出 per-pixel `(1-SSIM)` error map（`trainer.cpp:6932` 起，`launch_ssim_to_error_map`）。
2. raster backward 里按 splat 累计两个量（`fastgs/rasterization/include/kernels_backward.cuh:867`）：`weight += α·T`（blending_weight）、`error += α·T · pixel_error` → 写入 `densification_info` 两行。
3. `MCMC::post_backward`（mcmc.cpp:723）：每步把 info 行 1（E_k^π，即该视角下的误差质量）与历史做 **max-over-views** 进 `_error_score_max`；每 2 个 refine 窗口（=200 步）清零重新累计（滑动窗口）。
4. `get_sampling_weights()`（mcmc.cpp:189）：relocate 的 src 抽样和 add_new 的抽样权重 = `error_score_max.clamp_min(1e-12)`（冻结 splat 置零）——**新增/搬迁 splat 直接落到"最近视角里画错最多"的地方**。
5. relocation 公式 = MCMC 论文 Eq.9 原版（`mcmc_kernels.cu:56`，binomial 系数表进 `__constant__`，n_max=51，ratio=同一 src 被抽中次数 +1 后 clamp）。

**噪声注入**（`mcmc_kernels.cu:161-248`）：
- `noise = covar(R·S²·Rᵀ) · randn · noise_factor`，`noise_factor = current_lr · sigmoid(-100·(opacity-0.005))`（即 `1/(1+exp(100·op-0.5))`，高 opacity splat 几乎不动）。
- **`current_lr = optimizer.get_lr() × 5e5`（mcmc.hpp: NOISE_LR=5e5）——噪声严格跟随 means LR 调度衰减 100×**。与我们已实现的"噪声随 means LR 衰减 100×"完全同构，互相印证。
- 每步注入、无独立噪声 buffer（fused curand Philox kernel）。

**Resume 可复现**（mcmc.cpp:22 `deterministic_mcmc_seed`）：每个随机操作的种子 = splitmix64 风格 hash(绝对 iteration ⊕ 操作常量流 ID)。relocate/add/noise 各用不同 stream 常量。**断点续训产生与不间断训练完全相同的 MCMC 突变序列，无需序列化 CUDA RNG 状态。**

**Optimizer moment 纪律**：
- relocate 后对 **src 行和 dst 行都** reset Adam m/v（mcmc.cpp:155 `update_optimizer_for_relocate`→`relocate_params_at_indices_gpu`，理由注释：src 的 opacity/scale 变了、dst 拿到全新参数）。
- 软删除（remove_gaussians，mcmc.cpp:819）：deleted mask 置位 + rotation 清零（作为死亡标记，dead 检测 = `opacity<=min_opacity OR |rot|≈0`）+ 六个参数组的 m/v/grad 全部清零 + error score 清零。
- capacity：init 时按 max_cap 直接 cudaMalloc 预留全部参数+optimizer 状态（growth_factor 1.5 允许超出）；add_new 前 `preflight_grow_capacity` 预检，失败则整体中止**不产生半更新模型**（mcmc.cpp:532）；add 后 `trim_memory_pool()` 防池膨胀。

**对照我们**：gsplat.MCMCStrategy 的 add 抽样与误差无关（按 opacity 多项式抽样）、resume 随机性依赖全局 RNG 状态、无 error-map 通道。error-weighted 抽样是最大可吸收差距；moment reset gsplat 已做（binoms 路径内部处理），但"src 行也 reset"这一细节值得核对 gsplat 1.5.3 行为。

### 2b. 曝光/外观校正（头号 bug 相关，价值最高）

**三层体系：bilateral grid → PPISP → PPISP controller**，顺序为 render → bilagrid → PPISP → loss（trainer.cpp:6353-6370）。

**PPISP**（`src/training/components/ppisp.cpp`、`src/training/kernels/ppisp_math.cuh`；上游 Apache-2.0 nv-tlabs/ppisp）：
- 每帧 1 个曝光参数（EV，`exp2f(clamp(raw, -16, +16))`，ppisp_math.cuh:179、279）+ 每帧 8 个颜色 homography latent；每物理相机 vignetting（3 通道 × [cx,cy,α0,α1,α2]）+ CRF（3 通道 × [toe,shoulder,gamma,center]）。
- **防"增益吃掉模型亮度"的锚定机制（我们要找的东西，明确存在）**，`ppisp.hpp:57`：

  ```
  float exposure_mean = 1.0f; // Encourage exposure mean ~ 0 (resolve SH ambiguity)
  ```

  实现（ppisp.cpp:640-648, 784-791）：

  ```
  L_anchor = w_exposure_mean · SmoothL1( mean_i(EV_i), β=0.1 )
  ∂L/∂EV_i = w · smooth_l1'(mean) / N_frames        （每帧均分同一梯度）
  ```

  只约束**数据集级均值为 0**，单帧完全自由。注释直接点名这是为了解决 SH（模型亮度）与曝光的模糊性——正是我们 30k 退化的头号根因。颜色 latent 同样有 mean 锚（β=0.005，先过 ZCA pinv 白化再取均值）。其余正则：vig 光心 0.02、vig 非正性 0.01、vig/CRF 通道方差 0.1。
- 优化器：自研 Adam（β 0.9/0.999，eps 1e-15），lr 2e-3，**warmup 500 步（从 0.01× 线性升）+ 指数衰减到 0.01×**（PPISPConfig）。
- CRF 已 clamp 输出，PPISP 开启时跳过外部 [0,1] clamp（trainer.cpp:6380）。

**Decoupled D-SSIM（LichtFeld 独有，第二道防线）**（trainer.cpp:6371-6377、`kernels/ssim.cuh:176-230`、ssim.cu:1182）：
- 外观模型开启时，photometric loss 用**解耦公式**：L1 + SSIM 的亮度(μ)项作用在 **corrected image**（过了外观校正），SSIM 的对比度/结构(σ)项作用在 **raw render**（不过外观校正），backward 返回两路梯度（`grad_corrected` 走外观链、`grad_raw` 直达 raster）。
- 效果：外观参数**只能通过亮度匹配获利，无法帮模型隐藏结构误差**；同时结构梯度不被增益缩放污染。这从损失函数层面切断了"增益↔内容"简并的另一半通道。

**PPISP controller**（`ppisp_controller.hpp` + trainer.cpp:5754、6217-6346）：
- 小 CNN（3 层固定随机 conv 特征 + 4 层可训 FC，1601→128→128→128→9），输入**渲染图**，输出 9 个 ISP 参数（1 曝光 + 8 颜色），每物理相机一个 controller。
- 默认在**最后 5000 步激活**（`resolved_ppisp_controller_activation_step`），期间默认冻结高斯（`ppisp_freeze_gaussians_on_distill=true`），把 per-frame 查表蒸馏成"看图预测"——**novel view / 导出渲染时也能给出合理曝光**，回答了"训练帧有 gain、验证/漫游帧没有"的账目问题（不过 in-loop eval 本身不加 ISP，见 2f）。

**Bilateral grid**（`components/bilateral_grid.cpp`）：16×16×8（x,y,亮度导引），lr 2e-3，warmup 1000 步 + 衰减到 0.01×，TV 权重 10.0（trainer.cpp:7216 加进 loss），自研 fused Adam。igs+ 预设 TV=5.0。

### 2c. LR / 优化器调度（`strategies/strategy_utils.cpp:98-137`、`optimizer/scheduler.cpp`）

- Adam：β1 0.9、β2 0.999、**eps 1e-15**（比 torch 默认 1e-8 小 7 个量级，小梯度参数步长更真实）。
- 参数组 LR：means = `means_lr × scene_scale`（场景尺度归一）；sh0 = shs_lr；**shN = shs_lr / 20**（高阶 SH 恒定低速，这是它们除 degree 解锁外的第二道 SH 管理）；scaling/rotation/opacity 恒定。
- 调度：**只有 means 衰减**，`gamma = 0.01^(1/iterations)` 每步连乘 → 全程恰好 100×（scheduler.cpp:145）。存在 WarmupExponentialLR（线性 warmup + 指数衰减）供 PPISP/bilagrid 用，主参数不 warmup。
- SH 渐进：每 sh_degree_interval=1000 步 `increment_sh_degree()`（3000 步到 deg3），与我们相同。

### 2d. 正则化全家桶（`losses/regularization.cpp` + `kernels/regularization.cu`）

- **scale reg**：`0.01 · mean(exp(scaling_raw))`——MCMC 论文原版 **post-exp 域**（regularization.cu:36）。梯度 `w/N · exp(x)`，对大 splat 指数放大。（Spirula 明确批评此形式 not scale invariant 并改用 log 域——**这一项 Spirula 的方案更好**。）
- **opacity reg**：`0.01 · mean(sigmoid(opacity_raw))`，梯度 `w/N · σ(1-σ)`。持续把低效 splat 压向回收阈值，与 MCMC relocation 配合。
- SH 无显式正则（靠 shN lr/20）。mask 相关另有 alpha 惩罚（mask_opacity_penalty_weight=1.0，power=2）。
- normal 监督可选：normal_loss_weight 0.05、normal_consistency（depth→normal 自洽）0.05、normal_flatten（压最小轴）1.0。
- 另有 ADMM 稀疏化剪枝（`components/sparsity_optimizer.cpp`，sparsify_steps 15000、prune_ratio 0.6）——压缩用，非质量项。

### 2e. Depth Anchor（`src/training/depth_anchor/depth_anchor_cache.{hpp,cpp}`）

- 用途：单目深度先验 → 每相机 robust affine 对齐（把 SfM 稀疏点投影进相机、采样先验，**同时拟合 disparity 和 depth 两个 affine 候选**，一次 trimmed refit，返回 scale/shift/floor/corr/samples；数据集级再决定用哪个模型）。GPU 投影单线程串行、CPU RANSAC 进工作池与下一相机的深度解码重叠。
- **Cache 生命周期（我们该学的部分）**：
  - sidecar 文件放在深度先验目录旁（`depthAnchorSidecarPath`）。
  - **Fingerprint = FNV-1a hash over：schema 版本 + 每相机(名字, fx, fy, cx, cy, W, H, pose 参数) + 深度文件(size + mtime)**（depth_anchor_cache.cpp:280-321）。任何一项变化（重跑 SfM、重生成深度、改 resize）→ hash 不匹配 → 强制重算。
  - **刻意排除**点云子采样数和先验分辨率——注释说明 robust fit 对它们不变，且两个生产者（trainer / preprocess 工具）下它们不同；排除后两边可共享同一 cache。
- 对照我们：metric LiDAR ray-range 不需要 affine 对齐，但我们的 ray cache 应采用同样的 fingerprint 纪律（相机内参+位姿+源文件 size/mtime+schema 版本；刻意排除与结果无关的项），失配即作废，杜绝陈旧 cache 静默污染训练。

### 2f. 评估 / checkpoint 生命周期

- **Eval**（`metrics/metrics.cpp`）：仅在 `enable_eval` 且 iteration ∈ eval_steps（默认 {7000, 30000}）跑；PSNR/SSIM，**对 raw render 计算，不过 PPISP/bilagrid**（metrics.cpp 无任何 ppisp/bilateral 引用）——即它们的账目口径与我们一样是 gain=1 结账，靠 exposure-mean 锚保证这个口径公平。报告文件里记录 "Best PSNR (at iteration)" 但**不保留 best 权重**——只在固定 save_steps（默认 {7000,30000}）存盘。
- **Checkpoint**（`checkpoint.hpp/cpp`）：LFKP 流 = iteration + 策略（含 optimizer 全部 m/v + scheduler 状态）+ 全参数 + bilateral grid + PPISP + controller + 稀疏化状态，可嵌入 .licht 工程；MCMC 自己带 magic/version 段（mcmc.cpp:965）。配合 2a 的绝对-iteration 种子 → **resume 逐位可复现**。
- **对照我们**：我们的 golden 视图训练中选优（每 1000 步存 best_golden.pt）**领先** LichtFeld（它没有 best 保留）；但我们应核对 best_golden.pt 是否含足够状态用于"从最优点续训"（LichtFeld checkpoint 含 optimizer/scheduler 全状态）。

### 2g. 3DGUT 相对 gsplat 上游

- `src/training/rasterization/gsplat/`：gsplat CUDA 算子的 C++ 移植（Projection/UT/Rasterize/SH/Relocation），相机代码引自 3DGRUT（Cameras.cuh 头注释）。相机模型枚举：PINHOLE/ORTHO/FISHEYE/EQUIRECTANGULAR/THIN_PRISM_FISHEYE（`core/camera_types.h`），含 rolling shutter 类型。
- **限制**：`--gut` 路径与其 joint-codec 量化优化器不兼容，**sh_degree > 0 直接拒绝**（trainer.cpp:2733："GUT/gsplat training with sh_degree > 0 is unsupported"）。主战场是 FastGS（pinhole，用 `--undistort` 展开畸变）。
- **结论：3DGUT 上 LichtFeld 无可吸收的演化**——我们（gsplat 1.5.3，SH3 + 190° raw fisheye 直渲）在该轴上反而领先它。它的 fisheye 大 FOV 答案实质是"undistort 或不做"，Spirula 的面分裂才是正统参考。

## 3. 与 Spirula 对应项对照（谁更适合我们）

| 主题 | LichtFeld | Spirula | 更适合我们的 |
|---|---|---|---|
| 曝光锚定 | **exposure-mean SmoothL1 锚（w=1.0, β=0.1）+ 注释点名解决 SH 模糊性** | PPISP 同款 mean 正则 + color_shift_reg（EMA sign 漂移惩罚） | **LichtFeld 为主**（机制最小、直接可移植到我们的标量 gain）；Spirula color_shift_reg 作第二道防线 |
| 外观↔结构解耦 | **Decoupled D-SSIM（σ 项走 raw render）**，含 masked 变体 | 无对应（overexposure_reg 较弱） | LichtFeld 独有，强烈建议吸收 |
| MCMC 抽样升级 | error map=(1−SSIM)，α·T 聚合，max-over-views，2 窗口滑动，直接换 relocation/add 权重 | loss-map 更讲究（Tukey+Canny/ssim_cs、幂变换、无放回抽样）+ 长轴分裂 | **LichtFeld 为主教科书**（改动面小、与 gsplat MCMC 一一对应）；Spirula 的 robust 条件化与长轴分裂作二期增强 |
| 巨型高斯钳制 | 无等价物（靠 opacity/scale reg + 抽样回收） | **每步视角尺度钳 + max_world_size 硬 clamp** | Spirula（LichtFeld 在此为空白） |
| scale/opacity reg 形式 | post-exp 原版 0.01/0.01 | **log 域尺度不变版** + erank | Spirula 形式，LichtFeld 默认值佐证 0.01 量级 |
| 噪声衰减 | noise = means_lr × 5e5（随调度衰减 100×） | 独立调度 80→0.8 | 两者一致；**我们已实现**，两个参考互相验收 |
| refine 收尾期 | stop_refine=25k/30k（最后 5000 步） | max(25000, T−5000) | 完全一致；双参考背书 |
| SH 管理 | 1000 步/级解锁 + **shN lr = sh0/20** | in-the-wild 预设干脆不解锁 | 我们诊断 deg1 验证最优 → 先按 Spirula 关闭/推迟解锁，长期加 LichtFeld 的 shN 低速比 |
| Resume 可复现 | **种子=hash(绝对 iteration)** | 未特别处理 | LichtFeld |
| 相机级 ISP（vignetting/CRF） | PPISP（GPL 实现，Apache 上游） | PPISP（部分文件直接 Apache 头） | 都指向 **nv-tlabs/ppisp 上游直接 port（零 GPL 风险）** |
| 空间变化校正 | bilagrid 16×16×8, TV 10, warmup 1000 | bilagrid 同形状 + ppisp 网格化变体, Adagrad | 用 **gsplat 自带 fused_bilagrid（Apache）**，超参照抄两家共识 |
| fisheye 大 FOV | 无（gut 路径 SH0 限制） | **面分裂 warp_to_pinhole 全套** | Spirula 独占 |
| 显存/量化 | joint Adam codec、SH u16、VRAM ledger（深度绑定自研 tensor 库） | 块量化 + fused bwd+optim | 都难直接进 PyTorch 栈；**缓** |
| checkpoint 生命周期 | 全状态可嵌入 + 版本化段 + capacity preflight | （未深挖） | LichtFeld；但 best-checkpoint 保留是**我们领先** |
| depth cache | **fingerprint(内参+位姿+文件 size/mtime)+sidecar+刻意排除无关项** | （未深挖） | LichtFeld |

## 4. 双参考吸收总清单（按对我们当前真实问题的价值排序）

GPL 纪律：除标注"Apache 上游"外，一律**公式/思路重实现**，禁止复制两家（均 GPLv3）任何代码。

| # | 吸收项 | 主参考 | 解决什么 | GPL 风险 | 工作量 | 验收指标 |
|---|---|---|---|---|---|---|
| 1 | **曝光均值锚定**：把我们 per-frame L2 prior 换成/叠加数据集级（或每物理相机级）均值锚 `w·SmoothL1(mean_i g_i, β)`（公式见 §5） | LichtFeld（ppisp.cpp:640） | **头号 bug**：增益↔模型亮度联合简并漂移；左相机尾部塌 1dB | 安全（一行数学） | 0.5 天 | 30k 时 mean(gain) ≈ 0（<0.02 EV）；val（gain=1 口径）尾部不再下降 |
| 2 | **Decoupled D-SSIM**：L1+亮度项吃 corrected，SSIM σ 项吃 raw render，两路梯度 | LichtFeld（ssim.cuh:176, trainer.cpp:6371） | 增益无法遮蔽结构误差；结构梯度不被增益缩放 | 安全（公式重实现；PyTorch 下可用两次 SSIM 分量组合或自定义 autograd） | 1–2 天 | 开关 A/B：train/val gap 收窄；曝光参数漂移幅度下降 |
| 3 | **每步视角尺度钳 + max_world_size 硬 clamp** | Spirula（DensifyScoring.cu） | 巨型高斯/长训结构退化 | 安全 | 0.5–1 天 | max scale < 场景直径 5% |
| 4 | **refine 收尾期**：relocation/add 在 `min(25k, T−5k)` 停 | 两者（完全一致） | 30k 尾部扰动；8k 探针 vs 30k 终点不可比 | 安全 | 0.5 天 | 最后 5k 步 val 曲线单调不降 |
| 5 | **error-weighted MCMC 抽样**：per-pixel (1−SSIM) → 投影/α·T 聚到 splat → max-over-views（窗口 2×refine_every）→ 换 relocation+add 抽样权重 | **LichtFeld**（机制最小闭环）；Spirula robust 条件化二期 | 新增 splat 落点与"画错的地方"无关（val −0.47dB 的过拟合/欠拟合错配） | 安全（gsplat 下可先用投影中心采样 error map 近似 α·T 聚合） | 3–5 天 | 同预算 val PSNR ↑；floater 数↓ |
| 6 | **log 域 scale reg + opacity reg**（形式 Spirula，权重 0.01 两家共识） | Spirula | 大 splat 恒定回拉、低效 splat 送回收 | 安全 | 1 天 | scale P99 收窄，PSNR 不降 |
| 7 | **SH 策略修正**：先把 sh_degree_interval 关闭或推后（我们实测 deg1 验证最优）；若保留 SH3，shN 参数组 lr = sh0/20 | Spirula（in-the-wild）+ LichtFeld（/20） | SH3 只赚 0.2dB 还助长视角过拟合 | 安全 | 0.5 天 | val deg-sweep 最优点=训练配置 |
| 8 | **PPISP 逐相机 ISP**（vignetting+颜色+可选 CRF；曝光锚即 #1 的超集） | 上游 nv-tlabs/ppisp（Apache-2.0，可直接 port） | 宽 FOV vignetting、白平衡漂移，标量 gain 罩不住 | **零风险（Apache 上游）** | 2–3 天 | 逐帧 gain 直方图收窄；PSNR ↑ |
| 9 | **fisheye 面分裂训练** | Spirula 独占 | 2912² batch-state int32 上限；190° 单相机极限 | 安全（几何数学重实现） | 3–5 天 | 全分辨率可开 packed；无 seam |
| 10 | **resume 确定性种子**：所有随机操作种子 = hash(绝对 step ⊕ 操作 ID) | LichtFeld（mcmc.cpp:22） | 断点续训可复现、消融可信 | 安全 | 0.5 天 | resume 与直跑逐位一致（短程冒烟） |
| 11 | **LiDAR ray cache fingerprint**：schema 版本+内参+位姿+源文件 size/mtime，失配强制重算；刻意排除与拟合无关项 | LichtFeld（depth_anchor_cache.cpp:280） | 陈旧 cache 静默污染 | 安全 | 1 天 | 改任一输入后 cache 自动作废 |
| 12 | bilateral grid 16×16×8 + TV 10 + warmup（用 gsplat 自带 fused_bilagrid） | 两者共识超参；gsplat Apache 实现 | 空间变化曝光 | 零风险 | 1–2 天 | 关闭时 PSNR 差值可量化 |
| 13 | relocation 时 src+dst 双侧 moment reset 核对；capacity preflight 原子性 | LichtFeld | 搬迁 splat 带陈旧动量乱飞；OOM 半更新 | 安全 | 0.5 天（主要是核对 gsplat 行为） | 新增 splat 500 步内 opacity 达标 |
| 14 | PPISP controller 蒸馏（末 5000 步冻高斯，CNN 看渲染图出 ISP 参数） | LichtFeld | novel view/导出时的曝光账目 | 安全（结构重实现） | >1 周，**缓**（先靠 #1 把口径校平） | 漫游渲染无亮度跳变 |
| 15 | loss-map robust 条件化（Tukey+Canny、幂 4.0、无放回抽样）+ 长轴分裂 | Spirula（引公开论文） | #5 的二期增强 | 安全（按论文） | 5–8 天 | 边缘锐度/floater 进一步改善 |
| 16 | 多尺度 loss 金字塔 | Spirula | 高分辨率低频残差 | 安全 | 1–2 天 | 下采样 PSNR 改善 |

不吸收：两家的自研 tensor/量化优化器栈（与 PyTorch 不合）；LichtFeld 的 FastGS/undistort 路径（我们 3DGUT 领先）；Vulkan/GUI/MCP 层。

**分工总结**（按任务书原则验证成立）：LichtFeld = Trainer core 教科书（曝光锚定、decoupled SSIM、error-weighted MCMC、resume 种子、checkpoint 原子性、cache fingerprint、shN/20）；Spirula = 尺度钳制、log 域正则、fisheye 面分裂、loss-map 精加工、显存专项。两家在噪声衰减 100×、refine 收尾 5000 步、opacity/scale reg 0.01、bilagrid 16×16×8+TV10 上**独立收敛到同一答案**——这些可视为业界共识直接定版。

## 5. 对曝光简并 bug 的直接建议（含 LichtFeld 锚定公式）

我们的诊断：per-image 增益吸收全局亮度 → 模型整体变暗 → 验证按 gain=1.0 结账吃亏，左相机尾部塌 1dB。我们现有 L2 prior（每帧 g→0）的问题：它同时惩罚**真实的**帧间曝光差和**病态的**集体漂移，权重调大伤前者、调小拦不住后者。

LichtFeld 的答案是把这两件事分开——**只锚均值，放开个体**：

```
设每帧对数增益 g_i（我们的标量曝光，ln 域；LichtFeld 为 EV=log2 域）
L_anchor = w · SmoothL1( (1/N)·Σ_i g_i , β )
梯度：∂L/∂g_i = w · smooth_l1'(mean) / N       （每帧同一方向、均分）
LichtFeld 默认：w = 1.0，β = 0.1 EV（≈0.069 ln）；EV 硬 clamp ±16（我们已有 ±ln2 更紧，保留）
```

落地为四步组合（按性价比排序）：

1. **均值锚替换/叠加 L2 prior**：保留很弱的 per-frame L2（防个别帧跑飞），主力换成上式。**建议按物理相机分组各自锚**（`mean(g_left)→0`、`mean(g_right)→0`）——我们左相机尾部塌 1dB 正是"左相机组集体漂移"的形态，全局单锚允许左右互相抵消后仍各自漂。分组锚同时使"验证 gain=1"对两相机都公平。
2. **decoupled D-SSIM**（清单 #2）：SSIM σ 项吃 raw render，从梯度通道上让增益只能做亮度搬运工。
3. **诊断量常态化**：每次 eval 记录 `mean(g)` 按相机分组曲线 + 渲染图全局亮度均值曲线。锚定生效的直接证据 = 两条曲线在 30k 全程平直；回归判据 = |mean(g)| < 0.02。
4. 若之后引入 PPISP（清单 #8），其 exposure_mean/color_mean 正则天生就是本机制的推广形态，w=1.0/β=0.1 直接沿用；再叠 Spirula 的 color_shift_reg（EMA sign 漂移惩罚）防"校正方向性变暗"。

训练视角过拟合（train +0.36 / val −0.47）的对应组合：#5 error-weighted 抽样（把容量花在真误差处）+ #7 SH 冻结 + #6 正则 + #4 收尾期；曝光锚本身也会回收一部分 val 损失（当前 val 在 gain=1 下低估了模型真实质量）。

---
### 附：LichtFeld 来源文件速查
- MCMC 策略/种子/moment 纪律：`src/training/strategies/mcmc.cpp`
- MCMC 核（Eq.9、噪声、抽样）：`src/training/kernels/mcmc_kernels.cu`（NOISE_LR 在 `strategies/mcmc.hpp:94`）
- error score 聚合：`src/training/rasterization/fastgs/rasterization/include/kernels_backward.cuh:867,963`
- 默认超参：`src/core/include/core/parameters.hpp:124-264`、preset 在 `src/core/parameters.cpp:507-560`
- 曝光锚定/PPISP 正则：`src/training/components/ppisp.cpp:625-763`、config 默认 `components/ppisp.hpp:47-62`
- 曝光公式/EV clamp：`src/training/kernels/ppisp_math.cuh:159,179,278`
- Decoupled D-SSIM：`src/training/include/lfs/kernels/ssim.cuh:176-230`、`kernels/ssim.cu:1182`、接线 `trainer.cpp:6371`
- LR/优化器：`src/training/strategies/strategy_utils.cpp:98-150`、`optimizer/scheduler.cpp`
- opacity/scale reg 核：`src/training/kernels/regularization.cu`
- depth anchor cache：`src/training/depth_anchor/depth_anchor_cache.{hpp,cpp}`
- checkpoint：`src/training/checkpoint.{hpp,cpp}`；eval：`src/training/metrics/metrics.cpp`
- 3DGUT 移植：`src/training/rasterization/gsplat/`（SH>0 限制在 `trainer.cpp:2733`）
- PPISP controller：`src/training/components/ppisp_controller.hpp`、蒸馏接线 `trainer.cpp:5754,6217`
