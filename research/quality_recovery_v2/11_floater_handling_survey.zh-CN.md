# 11 — 文献综述：3DGS 如何处理悬浮体、越界增长、LiDAR 锚定与天空/远景

日期 2026-09-11 · 只读调研（WebSearch / WebFetch 读原始论文 HTML、官方代码与文档），无代码改动、无训练、无 git 操作 · 引用一律给 URL

> 阅读边界：所有机制描述来自论文正文或官方代码；两篇 ScienceDirect 论文（UAV LiDAR "depth-guided prune"、"Direct LiDAR-supervised surface-aligned 3DGS"）只读到摘要，正文被 403 挡住，文中标注"仅摘要"。ARSGaussian 的 PDF 本机无法解析文本，只用了 arXiv 摘要与检索摘录。"证据"一栏只写论文自己报告的数字，不做推断放大。
> 本文对应本目录 `10_surface_anchor_prune.md`（S1：到初始化云 > 0.3 m 硬剪）、`09_rgb_supervision_mask.md`（F1/F2 监督遮罩、切片归属）与 `08` 中期报告里的失败形态：墙前半透明树色/天空色团、屋檐/天空方向外凸、tile1 20k 有 34% 高斯离最近 LiDAR 点 > 0.2 m、35% 在切片框外。

## 0. 结论先行

1. **文献里没有一篇用"到 LiDAR 点的欧氏距离 > 阈值就剪"作为主机制。** 距离到表面被用作**软损失**（LI-GS、SuGaR、Structured-Li-GS、2DGS/GOF/RaDe-GS 的深度畸变）、**增长引导**（ARSGaussian、Pixel-GS 的近相机梯度缩放）或**距离加权的剪枝阈值**（LI-GS）。出现过的硬空间规则只有三类：物体包围盒（LiDAR-RT）、切片/块的所有权盒（VastGaussian、CityGaussian、H-3DGS、BlockGaussian，都在**合并时**剪、训练时不剪）、以及"离别的块更近就删"（H-3DGS）。
2. **所有分块方法都先回答"这个块用什么去画不属于它的像素"**，再谈剪：全局粗先验（CityGaussian）、邻块脚手架 + 天空盒（H-3DGS）、块外稀疏点长成的辅助高斯（BlockGaussian）、空域可见性选相机 + 覆盖选点（VastGaussian）。我们的切片只有天空穹顶背景库，树/屋檐外远景/邻切片墙没有任何"替身"，这正是 BlockGaussian 点名的 supervision mismatch——文献给的答案是替身或遮罩，不是剪。
3. **`render + (1 − alpha) · backdrop` 这种合成在文献里从不单独出现**：Street Gaussians、OmniRe、Splatfacto-W、Urban Radiance Fields 都配一条 **alpha/opacity 损失**把天空像素的累计 alpha 压向 0，否则高斯与背景层之间没有梯度上的偏好，高斯照样去画天空。
4. **LiDAR 射线的视线（free-space）损失**是驾驶域处理"表面前方悬浮体"的标准手段（Urban RF → NeuRAD → SplatAD），它按**射线**而不是按**点距离**惩罚，天然放过无回波（玻璃、细枝、扫描盲区）的方向；Urban RF 的消融同时提醒：单独去掉空域项指标反而略升，近表面项才是关键，空域项不是免费的。
5. 对我们：优先级应是 **越界像素归属/替身 → 天空 alpha 损失 → 只限增长（不剪） → 射线视线约束 → 距离加权软衰减**；S1 那样的硬 0.3 m 剪枝放到最后，且要加可见性门。

## 1. 通用生命周期层面：opacity reset、剪枝、增殖判据

| 方法 | 机制 | 硬/软 | 报告的副作用 / 证据 |
|---|---|---|---|
| 原始 3DGS（[Kerbl et al. 2023](https://ar5iv.labs.arxiv.org/html/2308.04079)，[官方代码](https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/scene/gaussian_model.py)） | 每 100 步增殖（τ_pos = 0.0002，split 缩放 1.6）；剪 opacity < ε_α（代码 0.005）；每 3000 步把 opacity 压到 min(α, 0.01)；周期性剪世界空间过大（scale > 0.1 × 场景范围）与屏幕足迹过大（代码 max_radii2D > 20 px，reset 之后才启用）的高斯 | 全部硬规则 | 论文原话：优化"会卡在靠近输入相机的悬浮体上……造成高斯密度不合理增长"，reset 就是为此。reset 是全局同一冲击，没有按位置区分 |
| Revising Densification（[Rota Bulò et al. ECCV 2024](https://arxiv.org/html/2404.06109)） | 用逐像素误差（SSIM）按贡献分摊给高斯做增殖判据；clone 时把 opacity 改成 α̂ = 1 − √(1 − α) 消除"clone 后两层叠加变更不透明"的偏差；总预算上限；**用每次增殖后 opacity 减 0.001 的渐进衰减替代硬 reset** | 软（预算为硬） | 作者称硬 reset 会破坏误差统计、造成训练不稳；gsplat 的 `revised_opacity` 即此（[gsplat strategy 文档](https://docs.gsplat.studio/main/apis/strategy.html)） |
| 3DGS-MCMC（[Kheirkhah et al. NeurIPS 2024](https://arxiv.org/html/2404.09591)） | opacity 与 scale 各加 L1 正则（λ_o = λ_Σ = 0.01）；opacity < 0.005 视为"死"，每 100 步按活体 opacity 多项式采样搬迁；位置噪声按 σ(−100(α − 0.005)) 门控，只扰动低 opacity 者；总数 cap | 软正则 + 硬 cap | 论文未讨论薄结构/透明物损失；噪声理论上可能把窄高斯推出支撑区，靠 opacity 门控缓解。gsplat `MCMCStrategy` 缺省 cap 1M、noise_lr 5e5 |
| Mip-Splatting（[Yu et al. CVPR 2024](https://arxiv.org/html/2311.16493)） | 按训练相机 max(f/d) 算每个高斯的最大采样频率，用 3D 平滑滤波给尺寸加**下界**；2D Mip 滤波替代 dilation | 硬下界 | 解决缩放走样，**不处理悬浮体，也没有尺寸上界**——挡不住向天空长大的高斯；每 100 步重算采样率有开销 |
| AbsGS（[Ye et al. 2024](https://arxiv.org/html/2404.10484)） | 视空间梯度按像素取绝对值再累加，避免大高斯的子梯度相消；阈值提到 0.0004/0.0008 | 软判据 | 报告"过重建"区域被正确 split，内存约减半；未报告悬浮体副作用 |
| Pixel-GS（[Zhang et al. 2024](https://arxiv.org/html/2403.15530)） | 增殖梯度按覆盖像素数加权；**近相机梯度缩放** f = clip((z / (0.37 · 场景半径))², 0, 1) 抑制近相机悬浮体 | 软 | 消融最有说服力：只加像素加权时 Tanks&Temples LPIPS 从 0.194 恶化到 0.239（悬浮体增多），加上距离缩放回到 0.178。说明**增殖越激进越需要增长侧的空间门** |
| FSGS（[Zhu et al. 2023](https://arxiv.org/html/2312.00451)） | 2k/5k/7k 步把 opacity reset 到 0.05 清悬浮体；Pearson 单目深度正则 | 硬 | 少视角设定；不能泛化到训练未观测的遮挡区 |

## 2. 贡献/可见性/孤立度剪枝

| 方法 | 机制 | 硬/软 | 副作用 / 证据 |
|---|---|---|---|
| RadSplat（[Niemeyer et al. 2024](https://arxiv.org/html/2403.13806)） | 重要性 = 所有训练射线上 max(α·T)；< 0.01（默认）或 0.25（轻量）剪，训练中做两次；渲染时按相机簇做可见性过滤 | 硬阈值 | 轻量版高斯数 ≈ 1/10 而质量持平；只与训练视角贡献有关，**对在训练视角里"有用"的悬浮体无效** |
| Mini-Splatting（[Fang & Wang 2024](https://arxiv.org/html/2403.14166)） | 按累计混合权重做**随机采样**而非确定性剪；用渲染深度重新初始化 | 软（采样） | 作者称采样比硬剪更保几何；**天空区域深度不可靠**、保留过少时远景失真 |
| TrimGS（[Fan et al. 2024](https://arxiv.org/html/2406.07499)） | 贡献 = 归一化的 α^γ·T^(1−γ) 像素和，取 top-5 视角均值；每 1000 步剪最低 10%；大尺度强制 split；深度法线 L1 | 硬（比例） | 明确报告**户外 PSNR 略降**；剪枝必须配几何正则才稳 |
| TIDI-GS（[2026](https://arxiv.org/html/2601.09291)） | 候选 = 低可见计数 ∧ 低 opacity ∧ 低学习重要性 ∧ 低梯度 EMA；**细节守卫**豁免高频 SH、高局部色方差、线状各向异性；再按 kNN 孤立度排序删，带自适应上限；每 400 步 | 硬但多门 | 只针对室内有限深度；作者说无界室外深度先验不可靠。守卫设计本身说明单一信号会误杀线状结构 |
| SparseGS（[Xiong et al. 2023](https://arxiv.org/html/2312.00206v2)） | 逐像素比较 alpha 混合深度与"众数深度"（最大权重高斯的深度）的相对差；直方图双峰 + dip test 自适应阈值；把众数高斯之前（含）的全部高斯删掉；20k 步做一次 | 硬（自适应阈值） | 消融 +0.25 dB（16.93 → 17.18）；依赖深度已收敛；少视角设定 |
| UAV LiDAR 几何感知 3DGS（[Int. J. Appl. Earth Obs. 2025，仅摘要](https://www.sciencedirect.com/science/article/pii/S1569843225002377)） | "depth-guided prune"：比较高斯自身深度与渲染深度剪悬浮高斯；LiGAGC 损失（深度/法线/曲率/局部一致） | 硬 | 阈值与副作用正文未读到 |

## 3. 面对齐几何正则（软）

- **2DGS**（[Huang et al. SIGGRAPH 2024](https://arxiv.org/html/2403.17888v3)）：深度畸变 L_d = Σ ω_i ω_j |z_i − z_j|（有界场景权 1000）把同一射线上的 splat 挤到一起，法线一致性权 0.05。作者明列局限：**假设完全不透明，玻璃/复杂透射失败**；增殖偏向纹理丰富区，几何细结构可能欠表达；正则过强会过平滑。
- **GOF**（[Yu et al. 2024](https://arxiv.org/html/2404.10772)）：opacity 场取所有训练视角的最小值；深度畸变项对部分梯度 detach，作者说明是为避免它自己制造悬浮体。半透明物体未处理。
- **RaDe-GS**（[Zhang et al. 2024](https://arxiv.org/html/2406.01467)）：L = L_color + 100·L_distortion + 5·L_normal；反光面困难。
- **SuGaR**（[Guédon & Lepetit 2023](https://arxiv.org/html/2311.12775)）：用"理想 SDF"（最近高斯主导、扁平、α = 1）与实际密度的 L1 差把高斯拉到面上，早期加 opacity 熵项逼二值化，之后**删 opacity < 0.5**；作者承认对齐阶段渲染质量略降。
- 这一族对我们的意义是：它们把"离面"变成损失而不是删除，但代价是把半透明层压扁——与本仓记忆"低 opacity 叠层不是死质量、是锐度来源"直接冲突，需要按 ROI 验证。

## 4. LiDAR / 深度锚定：初始化、深度损失、射线视线、距离约束

### 4.1 只用 LiDAR 做初始化 + 深度损失（最常见，没有剪枝）

| 方法 | LiDAR 用法 | 剪枝 | 证据 / 局限 |
|---|---|---|---|
| Street Gaussians（[Yan et al. ECCV 2024](https://arxiv.org/html/2401.01339)） | 聚合 LiDAR 0.15 m 体素下采样、**剔除训练相机不可见的点**、远处补 SfM 点；深度 L1 只取最好的 95% 像素（λ = 0.01） | 无 LiDAR 剪枝 | 刚体动态、依赖跟踪 |
| DrivingGaussian（[Zhou et al. CVPR 2024](https://arxiv.org/html/2312.07920)） | LiDAR 作初始化，先切掉动态前景；按 LiDAR 深度分 N 个 bin **渐进加入远景**，避免过早引入远处造成尺度混乱 | 无 | 极小物体、全反射材质失败 |
| LiHi-GS（[2024](https://arxiv.org/html/2412.15447v1)） | 距离图 L1 + "LiDAR 可见性 α = 1"损失（覆盖相机看不到的 360° 区） | 无；明确**没有 free-space 项** | 深度误差略高于 NeuRAD |
| LetsGo（[Cui et al. 2024](https://arxiv.org/html/2404.09748)，手持 LiDAR 车库——与我们最像） | 网格重采样初始化；深度 L1 λ = 0.8，深度取射线与高斯交点而非中心；LOD 多分辨率 | 无 | 报告深度正则"有效缓解地面悬浮体"；LOD 切换伪影 |
| GTLR-GS（[2026](https://arxiv.org/html/2603.23192)） | 置信度加权（图像 Laplacian）的度量深度 λ = 1、法线对齐 | 无 | 未分析薄结构/玻璃 |
| TCLC-GS（[Zhao et al. 2024](https://arxiv.org/html/2404.02410v1)） | LiDAR 建八叉树 SDF → 网格 → 渲染**稠密**深度监督，比稀疏点更抗过拟合 | 无 | 32 线 LiDAR 上退化 |
| OmniRe（[Chen et al. 2024](https://arxiv.org/html/2408.16760)） | LiDAR 稀疏深度损失 + 天空 opacity 损失 | 无 | 不建光照模型 |

### 4.2 射线视线（free-space / line-of-sight）监督——"表面前方不该有东西"的标准写法

- **Urban Radiance Fields**（[Rematas et al. CVPR 2022](https://ar5iv.labs.arxiv.org/html/2111.14643)）：把每条 LiDAR 射线拆成空域项（击中点之前 ε 之外权重平方积分为 0）、近表面项（权重分布匹配以击中深度为中心的高斯核）、之后项（可丢）；**ε 随训练指数退火**，作者说小 ε 早期会伤性能；天空像素由分割掩膜强制零密度、由球面 MLP 出色。消融：单独去掉空域项指标略升，三项合用最好——**近表面项是关键，空域项主要压悬浮体但不是免费的**。
- **NeuRAD**（[Tonderski et al. CVPR 2024](https://arxiv.org/html/2311.15260)）：对 τ > ε（ε ≈ 0.1 m）之外的样本做权重衰减；**ray drop**（无回波射线）单独建模——无回波的射线只衰减到传感器量程，不给深度监督。去掉 ray drop 掉约 1.9 dB，作者点名它能避免**透明面和远处无回波区**的假悬浮体。
- **SplatAD**（[Hess et al. 2024](https://arxiv.org/html/2411.16816v3)）：3DGS 版，line-of-sight 损失惩罚 **LiDAR 回波之前的 opacity**，作者说明"把 LiDAR 点投到图像做深度监督"会因相机-LiDAR 位姿差造成错误体雕刻——**射线要在 LiDAR 帧里算**；MCMC 增殖 cap 5M；远景用视差均匀采样到 10 km 的随机点。
- **LiDAR-RT**（[Zhou et al. 2024](https://arxiv.org/html/2412.15199)）：距离/强度 L1 + ray-drop BCE + Chamfer；世界空间梯度增殖；**采样落在物体包围盒外的高斯直接剪**（硬空间规则，仅对已知包围盒的物体）；长序列高斯数暴涨。

### 4.3 距离到表面的约束：软损失、增长引导、加权阈值

- **LI-GS**（[Jiang et al. 2024](https://arxiv.org/html/2409.12899v1)）：LiDAR 转平面约束 GMM，surfel 位置/形状控制点/法线三路对 GMM 面的加权距离损失；**剪枝阈值按到 GMM 面的距离加权——离面越远越容易被剪**，不是硬距离切；组件减少约 45%；作者指出纯光度损失会"放错位置"，GMM 补的是相机主轴方向的约束。
- **ARSGaussian**（[Yao et al. 2024，摘要](https://arxiv.org/abs/2412.18380)）：LiDAR 点"自适应引导高斯沿几何基准生长与分裂，解决过度生长和悬浮体"，检索摘录称降低距离阈值会让全部高斯收敛到 LiDAR 框架内、总数随之下降——这是**增长侧**的距离门，正文阈值与消融本机未能读取。
- **Structured-Li-GS**（[2026](https://arxiv.org/html/2606.27509)）：高斯挂在 LiDAR 体素锚点上，偏移损失法向权 5、切向权 1，扁平损失；**不增殖**，高斯数约为 Scaffold-GS 的 1/3。
- **Direct LiDAR-supervised surface-aligned 3DGS**（[2026，仅摘要](https://www.sciencedirect.com/science/article/abs/pii/S0141938226000120)）：LiDAR/SfM 位置与法线作直接约束，可微面对齐损失同时管位置与形状。
- 共同点：**距离只进入损失或阈值权重**，没有一篇在训练中按固定米数删；LiDAR-RT 的包围盒剪只对物体。

## 5. 分块训练：一个块拥有什么，块外像素怎么监督

| 方法 | 训练时块外内容 | 相机分配 | 合并时所有权 | 关于空域悬浮体的说法 |
|---|---|---|---|---|
| VastGaussian（[Lin et al. CVPR 2024](https://arxiv.org/html/2402.17427)） | 边界外扩 20% 选数据；**空域感知可见性**：把 cell 的 AABB 投到图像，占比 ≥ 25% 的相机入选；再把这些相机看得到的所有点都加进来（coverage-based） | 按 cell 体积投影可见性，不按相机位置 | **删掉原始区域（外扩前）之外的高斯** | 明确：只用表面点算可见性会漏掉空域监督，空中长悬浮体；块外物体入镜却没有点会造成深度歧义 |
| CityGaussian（[Liu et al. ECCV 2024](https://arxiv.org/html/2404.01133)） | 先用全部观测训 30k 步全局粗先验，再在收缩空间分块；粗先验"防止悬浮体过拟合块外区域" | 视角按"去掉块 j 后 SSIM 变化 ≥ ε"分给块，加相机在块内者 | **按空间盒过滤**后直接拼接 | 粗先验显著减少块间干扰；航拍+街景混训反而退化 |
| Hierarchical 3DGS（[Kerbl, Meuleman et al. SIGGRAPH 2024](https://arxiv.org/html/2406.12080)） | 块内 SfM 点 + **邻块脚手架高斯**；块外的粗环境与天空盒只临时优化 opacity/SH；**天空盒 = 场景直径 10 倍的球面上 10 万个高斯** | 块内相机，或在 2 倍块范围内且块内 ≥ 50 个 SfM 点的相机 | **块外的基元若离别的块更近就删** | 增殖判据改用屏幕梯度**最大值**而非均值；单目深度按 SfM 定标做正则；逐图曝光优化"去掉为解释亮度差而生的假高斯" |
| BlockGaussian（[Wu et al. 2025](https://arxiv.org/html/2504.09048v2)） | 用监督视角看得到、但在块外的稀疏点初始化**辅助高斯**，与块内一起优化（mini-batch 稳定），**合并时裁掉** | 可见性感知 | 裁掉辅助高斯后直接合并 | 伪视角几何约束（扰动位姿 → 用渲染深度 warp 回参考图做 L1）针对**空域悬浮体**，rubble 场景 +0.10 dB（26.23 → 26.33） |

模式：(a) 每个块都有块外内容的替身；(b) 块外只冻结或只轻优化；(c) 相机按"看得到这个块的体积"分配；(d) 所有权在合并时按盒或按最近块裁决。**没有一篇在块训练过程中按距离删块外高斯**，也没有一篇让块用自己的高斯去画没有替身的块外像素。

## 6. 天空 / 远景 / 逐视角背景

| 方法 | 表示 | 合成与损失 | 报告的局限 |
|---|---|---|---|
| Street Gaussians（[链接](https://arxiv.org/html/2401.01339)） | 1024² 立方体贴图 | C = C_G + (1 − O_G)·C_sky；**渲染 opacity 对 Grounded-SAM 天空掩膜做 BCE（λ = 0.05）** | — |
| OmniRe（[链接](https://arxiv.org/html/2408.16760)） | 可优化环境纹理 | 同上合成；**opacity 损失把高斯 opacity 对齐到非天空掩膜** | 不建光照 |
| Splatfacto-W（[Xu et al. 2024](https://arxiv.org/html/2407.12306)） | 3 阶 SH 天空，由逐图外观嵌入经 MLP 出系数 | 同上合成；**alpha 损失只作用于"背景模型已解释得好"（残差低）的像素**，不需要语义掩膜 | 只能表示低频，云层差 |
| Urban RF（[链接](https://ar5iv.labs.arxiv.org/html/2111.14643)） | 球面 MLP | 天空射线上所有采样密度 → 0 | 依赖分割 |
| H-3DGS（[链接](https://arxiv.org/html/2406.12080)） | 场景直径 10 倍球面上 10 万高斯 | 与场景一起渲染，块训练时只动 opacity/SH | 目的是让各块的天空一致 |
| EVolSplat（[Miao et al. 2025](https://arxiv.org/html/2503.20168)） | 固定半径 100 m、随相机平移的半球高斯，opacity = 1、几何固定 | 前后景分开渲染再合成 | 作者承认远景几何只是近似，背景有伪影 |
| 两阶段户外 GS（[2025](https://arxiv.org/html/2510.09489)） | 深度阈值分出背景，测地球壳 [R_i, R_o] 上初始化 | **壳损失**（不许跑出球壳）+ **平面性损失**（不许向场景中心长径向尖刺） | 报告天空/极远物体无伪影，但只是定性 |
| 夜间驾驶重建（[2026](https://arxiv.org/html/2602.13549)） | MLP 天空 vs 立方体贴图 | — | MLP 低频偏置，恢复不了云和太阳 |

要点：天空层的**表示**（贴图/SH/球面高斯）各异，但**都有一条把天空像素 alpha 压向 0 的损失**；Splatfacto-W 的残差门控版本是唯一不需要语义掩膜的写法。我们的背景库合成式与它们相同，缺的是这条损失（`09` §1 记录当前损失只有合成项）。另外文献里的背景层是**全局共享**的（同一贴图/球面被所有视角约束），而我们是逐视角渲染的穹顶图，这意味着背景本身不受多视角一致性约束，高斯与背景之间的分工只能靠 alpha 损失和掩膜定义。

## 7. 对我们的启示

设定回顾：LiDAR 稠密房屋扫描；四个切片各自训练，损失 `render + (1 − alpha) · backdrop`，backdrop 只来自天空穹顶；失败形态是墙前半透明团、屋檐/天空方向外凸；tile1 20k 有 34% 高斯离初始化云 > 0.2 m、35% 在切片框外、远组 z 中位 2.46 m；室内 DIAG-40 视角里中位只有 45% 的回波属于本切片（`09` §6.3）。S1 已实现"> 0.3 m 硬剪 + 远父本禁生 + 框外剪"。

### 7.1 候选干预（按 文献证据 × 契合度 排序）

| # | 干预 | 文献证据 | 契合度与理由 | 主要风险 |
|---|---|---|---|---|
| 1 | **越界像素归属 + 远景替身**：训练损失只监督"本切片拥有"的像素（F2 = LiDAR 支持 ∧ 切片归属，`tile_ownership_masking`），或给切片一个冻结的替身层（邻切片粗高斯 / 全场粗先验 / 天空盒）去解释树、屋檐外远景、邻切片墙；合并时按最近切片裁决所有权 | 强且一致：VastGaussian、CityGaussian、H-3DGS、BlockGaussian 四篇都这样做，且 BlockGaussian 直接把"监督视角里有块外内容"命名为悬浮体成因 | 最高。`09` 已量化室内悬浮体视角里 55% 回波不属本切片；这是唯一直接对应"切片高斯被迫画树"的干预。替身层比遮罩更保训练视角 PSNR，但需要邻切片粗模型的一次预训练（CityGaussian 的 30k 粗先验思路） | 遮罩会让被遮像素改由背景补、训练视角 PSNR 下降（可接受）；替身层若也参与优化会重演 BlockGaussian 的"辅助高斯监督不足"问题，应只优化 opacity/SH 或完全冻结 |
| 2 | **天空 / 背景 alpha 损失**：对背景库能解释得好的像素（残差门控，Splatfacto-W）或天空掩膜像素（Street Gaussians/OmniRe）把累计 alpha 压向 0 | 强：四篇都配此项，无一例外 | 高。我们已有合成式与逐视角背景，只缺这条损失；残差门控版不需要语义分割，天空掩膜版可由"无回波 ∧ 视线朝上"近似 | 逐视角穹顶本身是糊的，残差门可能只在纯天空像素打开；树冠边缘会被判为"背景解释得好"而失去高斯——需要与 #1 的归属遮罩配合 |
| 3 | **只限增长、不剪**：把 S1 的 `reject_unsupported_parents`（远父本不得 clone/split）和框外父本禁生单独成臂；可再加 Pixel-GS 式近相机梯度缩放 | 中强：Pixel-GS 消融（无增长门悬浮体增多）、H-3DGS 改 max 梯度、ARSGaussian 增长沿 LiDAR、LiDAR-RT 采样出盒即剪、Revising Densification 预算 | 高。`10` §1 已证明远高斯是 133 次增殖累积出来的（3000 步时 7% → 20k 时 34%），源头在增长侧；增长门不会在墙面/门叶上开洞，是最便宜、最不伤 ROI 的第一刀 | 已存在的远高斯不会消失，需要配 opacity reset/衰减把它们自然淘汰；对短日程臂读数会很温和 |
| 4 | **LiDAR 射线视线约束**（替代点距离）：在面 LiDAR 缓存（vis6 修复版）上，对有回波的像素惩罚 LiDAR 深度 − ε 之前的累计 alpha，ε 从大到小退火；无回波像素不给任何惩罚 | 中强：Urban RF、NeuRAD、SplatAD 三代沿用；SplatAD 明确指出要在 LiDAR 帧里算射线以免体雕刻错误 | 高。它精确对应"墙前悬浮体"：只有真正遮住 LiDAR 表面的高斯才被惩罚，玻璃/细枝/扫描盲区方向没有回波就不受影响；比 S1 的欧氏距离更贴近可见性语义 | Urban RF 消融显示空域项不是免费的；我们的 LiDAR 与鱼眼相机位姿不同（`03` 记录门边 ≈ 10 mm 偏移），ε 至少要盖住配准误差；只在"严格可见性"支持像素上做（`05` 审计的 strict 规则），否则穿墙回波会把墙后高斯当悬浮体 |
| 5 | **距离加权的软衰减**替代硬阈值：opacity L1 权重或剪枝 opacity 阈值随到 LiDAR 距离单调上升（如 λ(d) = λ₀·clip((d − 0.3)/0.7, 0, 1)），配出生年龄豁免 | 中：LI-GS 的距离加权剪枝阈值、MCMC opacity L1、Revising Densification 的渐进衰减、SuGaR 的 SDF 拉回 | 中。让光度上必要的远高斯（玻璃后的树、LiDAR 漏掉的面）通过"挣到 opacity"存活，而只靠单视角过拟合的悬浮体在衰减下先死 | 与本仓"低 opacity 叠层是锐度来源"的读数相冲，权重需按 ROI 锐度验证；实现要改损失而非生命周期 |
| 6 | **贡献/孤立度剪枝**（RadSplat max 贡献、TIDI-GS kNN 孤立 + 线状守卫） | 中：数字扎实但都是通用压缩/清理 | 低-中。对训练视角里"有用"的悬浮体无效（它们正是靠过拟合训练视角活着的），只作为收尾清理 | TIDI-GS 自己就要加线状守卫，说明孤立度会误杀栏杆/枝条 |

### 7.2 为什么"> 0.3 m 一律剪"太绝对

S1 的 0.3 m 是对**初始化云**的欧氏距离，而初始化云 ≠ 真实表面，它会删掉四类照片里有、LiDAR 里没有的内容：

1. **玻璃与透射**：LiDAR 在透明面上无回波或回波在后方（`grep` 到的 LiDAR 玻璃文献与 2DGS 的"假设完全不透明"局限一致）；门叶上 6 格玻璃的反射层高斯会被当悬浮体。
2. **细结构**：枝条、栏杆、电线在 LiDAR 里只有零星回波，TIDI-GS 为此专门加线状守卫；ARSGaussian 之类航拍论文也报告细杆只有两三个点。
3. **扫描时的动态物体与开合的门**：LiDAR 记录扫描时刻位置，照片是另一个时刻（本仓 `scan-time-moving-objects` 记录）；两者相差 > 0.3 m 就整体删除。
4. **LiDAR 盲区与远处建筑**：视线被挡的面、超出量程的远景。合并后这些像素只能由糊穹顶补，训练视角 PSNR 会下降——`10` §6 已把这当作接受的取舍，但它同时意味着 S1 的读数会把"悬浮体消失"和"真实内容消失"混在一起，审计工具的 `far_fraction` 分不开两者（`10` §7 自己承认）。

文献侧的旁证：没有一篇把固定米数的点距离当训练期剪枝主机制；最接近的 LI-GS 用的是距离**加权**阈值，ARSGaussian 用在**增长侧**；Urban RF 的消融说明就连按射线的空域惩罚都可能略伤指标。硬空间规则出现的地方都是**所有权裁决**（合并时按盒/最近块）或**有明确包围盒的物体**，而不是"离面就删"。

### 7.3 更软的替代与建议的臂顺序

- **只限增长**（#3）：S1 关掉 `cull` 侧、只留 `reject_unsupported_parents` + 框外父本禁生，先看远高斯份额是否停止累积（`10` §1 的 3000 步 → 20k 步曲线是现成读数）。
- **归属遮罩 / 替身**（#1）：F2 已是现成旋钮；替身层可先用邻切片 R1 checkpoint 的 opacity ≥ 0.05 子集冻结加载，只做合成不回传梯度，验证"有替身后切片是否还长树"。
- **天空 alpha 损失**（#2）：先做 Splatfacto-W 式残差门控（不依赖分割），再评估是否需要显式天空掩膜。
- **射线视线约束**（#4）替代点距离：在 strict 可见性像素上、ε 退火（起点 ≥ 0.3 m，终点 ≈ 配准误差的 3–5 倍）；无回波像素零惩罚，天然放过玻璃与细枝。
- **距离加权衰减**（#5）：若 #1–#4 后仍有远高斯残留，再把 S1 的硬剪改成随距离升高的 opacity 阈值 + 年龄豁免，而不是一刀切。
- **硬剪只作最后一道、且加可见性门**：仅当高斯同时满足"离初始化云 > d"且"至少 k 条 strict 可见 LiDAR 射线穿过它到达更远的回波"才剪——这把 S1 变成 4.2 节的射线语义，玻璃/盲区方向因没有穿透射线而免疫。d 的敏感性（0.2/0.5）与 `10` §6 后手一致。

判读口径沿用 `09` §7 / `10` §6：离轨 PSNR（≥ +0.3 dB）、亮度归一 ROI-all、六视角出图看墙面/门叶是否露背景色、`far_fraction` 曲线；任何臂都要同时报"删掉的高斯在训练视角里的真实贡献"（RadSplat 的 max α·T 可直接复用为审计量），把"悬浮体消失"和"真实内容消失"分开计数。

## 8. 引用索引

3DGS 原文 https://ar5iv.labs.arxiv.org/html/2308.04079 · 官方代码 https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/scene/gaussian_model.py · Revising Densification https://arxiv.org/html/2404.06109 · 3DGS-MCMC https://arxiv.org/html/2404.09591 · gsplat strategy https://docs.gsplat.studio/main/apis/strategy.html · Mip-Splatting https://arxiv.org/html/2311.16493 · AbsGS https://arxiv.org/html/2404.10484 · Pixel-GS https://arxiv.org/html/2403.15530 · FSGS https://arxiv.org/html/2312.00451 · SparseGS https://arxiv.org/html/2312.00206v2 · RadSplat https://arxiv.org/html/2403.13806 · Mini-Splatting https://arxiv.org/html/2403.14166 · TrimGS https://arxiv.org/html/2406.07499 · TIDI-GS https://arxiv.org/html/2601.09291 · UAV LiDAR 3DGS（摘要）https://www.sciencedirect.com/science/article/pii/S1569843225002377 · 2DGS https://arxiv.org/html/2403.17888v3 · GOF https://arxiv.org/html/2404.10772 · RaDe-GS https://arxiv.org/html/2406.01467 · SuGaR https://arxiv.org/html/2311.12775 · Street Gaussians https://arxiv.org/html/2401.01339 · DrivingGaussian https://arxiv.org/html/2312.07920 · LiHi-GS https://arxiv.org/html/2412.15447v1 · LetsGo https://arxiv.org/html/2404.09748 · GTLR-GS https://arxiv.org/html/2603.23192 · TCLC-GS https://arxiv.org/html/2404.02410v1 · OmniRe https://arxiv.org/html/2408.16760 · Urban Radiance Fields https://ar5iv.labs.arxiv.org/html/2111.14643 · NeuRAD https://arxiv.org/html/2311.15260 · SplatAD https://arxiv.org/html/2411.16816v3 · LiDAR-RT https://arxiv.org/html/2412.15199 · LI-GS https://arxiv.org/html/2409.12899v1 · ARSGaussian（摘要）https://arxiv.org/abs/2412.18380 · Structured-Li-GS https://arxiv.org/html/2606.27509 · Direct LiDAR-supervised surface-aligned 3DGS（摘要）https://www.sciencedirect.com/science/article/abs/pii/S0141938226000120 · VastGaussian https://arxiv.org/html/2402.17427 · CityGaussian https://arxiv.org/html/2404.01133 · Hierarchical 3DGS https://arxiv.org/html/2406.12080 · BlockGaussian https://arxiv.org/html/2504.09048v2 · Splatfacto-W https://arxiv.org/html/2407.12306 · EVolSplat https://arxiv.org/html/2503.20168 · 两阶段户外 GS https://arxiv.org/html/2510.09489 · 夜间驾驶重建 https://arxiv.org/html/2602.13549
