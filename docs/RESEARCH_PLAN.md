# 自研 3DGS 处理管线 — 调研报告与开发计划
# Proprietary 3DGS Pipeline for SLAM Scanner Data — Research Plan & Handoff Document

> 交付对象:Claude Code(后续研究与开发协作)
> 日期:2026-07-02
> 状态:调研完成,准备进入 Phase 1 测试

---

## 1. 项目背景与上下文 (Context)

### 1.1 我是谁 / 项目定位
- LiDAR/SLAM 扫描仪产品经理,负责硬件产品 + 自研软件 CloudStudio(Web 端点云平台,Node.js + Python 后端,Potree 渲染,已开源 BSD-2-Clause)。
- 目标:**替代当前使用的第三方收费 3DGS 软件(mipmap)**,做出自己的 3DGS 处理软件,且效果要优于/追平收费方案。
- 长期上,3DGS 管线可能集成进 CloudStudio 产品体系(遵循 CloudStudio 现有架构约定:新路由 append 到 server.js、i18n ASCII-safe、不直接改 viewer.html、输出目录按 `projects/<projectName>/<module>/<runId>/` 规范)。

### 1.2 输入数据(硬件端已解决的部分)
| 输入 | 说明 | 对 3DGS 的意义 |
|---|---|---|
| 两个鱼眼相机照片 | 已知镜头参数(内参/畸变模型待确认具体形式:等距/KB/OpenCV fisheye) | 免去自标定;需鱼眼投影支持 |
| 照片 POS | 每张照片的位姿(SLAM 轨迹插值/优化后) | **免去 COLMAP/SfM**,绕开最脆弱环节 |
| 高精度 SLAM 点云 | 已处理完成的高精度稠密点云 | 最优初始化 + 深度监督来源 + 几何约束 |
| 相机镜头参数 | 标定内参 | 直接可用 |

**核心判断:这套输入组合把标准 3DGS 最难的两个环节(SfM 位姿、稀疏初始化)直接绕过,项目性质是"中等难度工程整合",不是科研攻关。**

### 1.3 竞品/对标
- mipmap(当前收费第三方):质量基准线,目标是追平或超越。
- 行业竞品扫描仪厂商(GreenValley、NavVis、FARO 等)多数也在做 3DGS 输出,自研管线是产品差异化点。

---

## 2. 需求目标 (Requirements)

### 2.1 功能需求
1. **输入**:双鱼眼图像序列 + POS(位姿)+ 相机内参 + 高精度点云(LAS/LAZ/PLY,坐标系与 POS 一致)。
2. **输出**:标准 3DGS PLY / SPZ / SOG 格式,可在 Web 查看器(未来接 CloudStudio)播放。
3. **核心优化功能**(超越收费软件的卖点):
   - 去除行人/人体
   - 去除移动物体(车辆、开门等)
   - 去除漂浮物(floaters)/ 雾状伪影
   - 利用高精度点云做几何约束,几何精度优于纯照片 3DGS
4. **商用许可安全**:全链路组件必须商用友好(Apache 2.0 / MIT / BSD)。**禁止使用原版 INRIA 3DGS 代码(非商用许可)。**

### 2.2 质量目标(验收基准)
- 留出测试视角:室内静态场景 PSNR ≥ 28–30,LPIPS 尽量低(与 mipmap 同数据对比)。
- 文字/标牌在 1:1 缩放下可读(标定质量的试金石)。
- 平面(墙面/地面)渲染深度与 LiDAR 点云偏差 < 2–3 cm(室内)。
- 离开采集轨迹 1–2 m 的新视角无明显尖刺/雾状伪影。
- 处理时间:单场景(数百张图)在单张消费级 GPU(如 RTX 4090)上 ≤ 1–2 小时(离线可接受)。

---

## 3. 深度调研结果 (Research Findings, 截至 2026-07)

### 3.1 技术底座:gsplat(定论)
**选 gsplat(nerfstudio-project/gsplat),Apache 2.0,商用友好。** 理由:
- 2024 年 v1.4 合入 Fisheye-GS,鱼眼数据集完整支持。
- 2025-04 集成 NVIDIA 3DGUT(CVPR 2025 Oral):用 Unscented Transform 替代 EWA splatting,sigma 点可在**任意非线性投影**下精确投影,原生支持鱼眼/畸变/卷帘快门,保持光栅化速度。gsplat 中通过 `--with_ut --with_eval3d` 启用,注意 **3DGUT 只支持 MCMC 致密化策略**。
- 2026-01 集成 PPISP(bilateral grid 的替代)做训练视角曝光/外观补偿。
- 2026-03 新增:LiDAR 光栅化(spinning-lidar 相机模型、depth/hit-distance 渲染模式,`pip install "gsplat[lidar]"`)、TorchScript 部署导出、3DGUT 外部畸变扩展。
- gsplat 与 Nerfstudio 均为 Apache 2.0。

**备选参考**:nv-tlabs/3dgrut(3DGUT/3DGRT 官方仓库,2025-07 起支持多传感器 COLMAP 数据集;注意核查其 License 是否为 Apache——gsplat 内的 3DGUT 集成已确认可商用路径)。

### 3.2 鱼眼支持
- 两条路线:(A) 去畸变成针孔 → 损失边缘视场,浪费鱼眼优势;(B) 原生鱼眼投影(3DGUT 或 Fisheye-GS)→ 推荐。
- 3DGUT 论文实测:直接在原始畸变鱼眼图上训练(全像素利用)比先去畸变再训练质量更好。
- **关键坑**:2025-08 研究(arXiv 2508.06968)对 200°/160°/120° FoV 实测,Fisheye-GS 和 3DGUT 都在 **160° 达到最佳**,200° 边缘畸变拖垮质量。→ 若我们镜头 ≥180°,应重投影/mask 到有效 ~160°。
- 若 SfM 失败场景需要补位姿(理论上我们不需要),UniK3D 单目深度可做初始化备胎。

### 3.3 LiDAR 点云的三重利用(核心竞争力)
1. **稠密初始化**:点云直接转 Gaussian 初值。注意需降采样(参考 GauU-Scene:原始点云太密,需 subsample;经验量级 50 万–200 万点起步)。可用点云投影到图像取色作为初始颜色(Gaussian-LIC 做法)。
2. **深度监督**:点云投影到每张鱼眼生成稀疏/半稠密深度图,监督渲染深度(LiGSM、UAV-LiDAR 3DGS 等均验证有效)。DN-Splatter(WACV 2025,基于 gsplat 实现,代码开源 maturk/dn-splatter)提供成熟的深度损失设计:**gradient-aware logarithmic depth loss + TV 正则**,并有法向监督(从深度或 Omnidata 单目法向)。实测深度监督同时提升深度指标和 RGB 指标,显著减少 floaters。
3. **几何引导致密化**:ARSGaussian 等工作用 LiDAR 作为几何基准约束 Gaussian 生长/分裂方向(沿点云局部切平面),抑制过度生长与漂浮物。

### 3.4 动态物体/瞬态剔除(三层方案)
| 层 | 方法 | 适用 | 代码 |
|---|---|---|---|
| L1 显式语义 mask | Grounded-SAM / SAM2 / YOLO-seg 分割"人/车"等已知类,loss 屏蔽 | 已知类别,最可控 | SAM2 (Apache 2.0) |
| L2 自动瞬态剔除 | SpotLessSplats:预训练特征(Stable Diffusion 特征)聚类 + 鲁棒优化,轻量 MLP 与 3DGS 同步训练识别瞬态像素;或 T-3DGS(无监督分类网络 + mask 传播,对视频序列边界更稳) | 兜住 L1 漏掉的不可预测物体 | SLS 官方未完整开源训练代码,需自实现或找社区复现;T-3DGS 有代码 |
| L3 几何漂浮物 | 深度监督 + 几何致密化 + scale 正则 + opacity 剪枝 | 空中乱长的 Gaussian | 内置于训练管线 |
| 加分项 | 点云-照片交叉验证:照片中出现但点云中不存在的物体 → 动态物强信号 | 我们独有的数据优势 | 自研规则 |

### 3.5 质量问题的算法根因(已在前期讨论中确认)
- 总根因:原版 3DGS 只有光度监督,欠约束。
- 糊/文字不清 → 位姿/标定亚像素误差("共识模糊")+ 致密化不足 + 曝光不一致。
- 混色 → SH 过拟合(浮空高斯靠视角相关颜色作弊)+ 深度错误的 alpha 混合串色。
- 尖刺 → 针状各向异性高斯过拟合训练视角 + 放大走样(Mip-Splatting 解决走样部分)。
- 雾状凸出 → 深度歧义 + 椭球不擅长表达薄平面(2DGS 扁盘基元是对症药)。
- **训练步数不是这些问题的旋钮**;结构性问题需正则/先验/输入质量解决,步数过多反而过拟合。

### 3.6 需要的正则化工具箱
- Scale 正则(惩罚极端长宽比)→ 治尖刺。
- MCMC 策略(3DGUT 强制)自带 opacity/scale 正则项。
- Mip-Splatting 式 3D 频带限制 → 治缩放走样(gsplat 有 antialiased 模式)。
- 曝光补偿:PPISP 或 bilateral grid,**必开**(SLAM 扫描仪自动曝光,双鱼眼间 + 轨迹间亮度跳变)。
- 2DGS / 法向监督:若平面质量仍不足的备选升级。

---

## 4. 技术选型汇总 (Decisions)

| 环节 | 选型 | 备注 |
|---|---|---|
| 训练/渲染核心 | gsplat (Apache 2.0) | 禁用 INRIA 原版 |
| 鱼眼投影 | 3DGUT (`--with_ut --with_eval3d`) | MCMC 策略;FoV 裁到 ≤160° |
| 致密化 | MCMC | 3DGUT 要求 |
| 初始化 | LiDAR 点云降采样 + 投影取色 | 50w–200w 点起步,做消融 |
| 深度监督 | DN-Splatter 式 log-depth loss + TV | 点云投影深度图,权重先强后弱 |
| 曝光补偿 | PPISP / bilateral grid | 必开 |
| 语义剔除 | SAM2/Grounded-SAM 生成 mask → loss 屏蔽 | 人/车优先 |
| 瞬态剔除 | SpotLessSplats 思路自实现(或 T-3DGS) | Phase 3 |
| 数据格式 | 输入 COLMAP-style 目录(自写转换器:POS+内参 → cameras/images.txt;点云 → points3D) | 兼容 gsplat simple_trainer |
| 评估 | PSNR/SSIM/LPIPS on held-out views + 渲染深度 vs 点云 | 见 §6 |

---

## 5. 开发路线 (Roadmap)

### Phase 0:数据与环境准备(第 1 周)
- [ ] 确认鱼眼畸变模型(等距/Kannala-Brandt/OpenCV fisheye?)与 3DGUT 支持的参数格式映射。
- [ ] 确认 POS 格式、坐标系(点云坐标系 vs 相机坐标系 vs 世界系)、时间戳对齐方式。
- [ ] 确认相机-LiDAR 外参标定值及其精度评估方式(重投影误差)。
- [ ] 搭环境:CUDA + PyTorch + gsplat(源码装),跑通官方 MipNeRF360 benchmark 确认环境正常。
- [ ] 选 1–2 个测试场景:一个干净静态室内场景(验证上限),一个有行人干扰的场景(验证剔除)。
- [ ] 同一数据在 mipmap 上跑一遍,留存输出作为对照基准。

### Phase 1:最小闭环(第 2–3 周)— 验证可行性
- [ ] 写数据转换器:POS + 内参 + 鱼眼图 → gsplat 可读格式(COLMAP-style);点云降采样 → points3D 初始化。
- [ ] **验证重投影正确性(最关键一步)**:把点云投影到若干张鱼眼图上叠加显示,目视检查边缘对齐(误差应 <2px)。这一步不过关,后面全白费。
- [ ] gsplat + 3DGUT + MCMC,静态场景裸跑(无深度监督、无剔除),开 PPISP 曝光补偿。
- [ ] 输出 PLY,held-out 视角评估,与 mipmap 对比。
- **Gate:清晰度接近 mipmap、无系统性糊(若糊 → 回查标定/位姿,不要先调训练参数)。**

### Phase 2:几何增强(第 4–5 周)
- [ ] 点云投影深度图生成管线(处理遮挡:z-buffer / 点云可见性剔除)。
- [ ] 接入深度监督 loss(参考 dn-splatter 实现,log-depth + TV),权重调度:早期强、后期衰减。
- [ ] Scale 正则 + opacity 剪枝调参。
- [ ] 消融实验:初始化点数(50w/100w/200w)× 深度监督权重,记录指标矩阵。
- **Gate:漂浮物/雾状伪影显著减少;渲染深度 vs 点云误差达标;离轨迹视角无明显崩坏。**

### Phase 3:动态剔除(第 6–8 周)
- [ ] L1:SAM2/Grounded-SAM 批量生成人/车 mask,loss 屏蔽;评估行人场景。
- [ ] 点云-照片交叉验证规则:渲染深度与点云深度冲突区域 → 疑似瞬态 mask(自研,差异化点)。
- [ ] L2:实现 SpotLessSplats 式鲁棒 loss(residual 聚类 + MLP 分类),或先试 T-3DGS 开源代码评估效果。
- [ ] 有人场景 vs mipmap 对比(mipmap 若无剔除功能,这就是核心卖点)。
- **Gate:行人干扰场景中人体残影基本消除,不误删静态结构。**

### Phase 4:产品化(第 9–12 周)
- [ ] 封装 CLI / Python API:`process(images, pos, intrinsics, pointcloud) → gs.ply`,参数预设(fast/quality)。
- [ ] Web 查看器集成方案调研(CloudStudio + Potree 旁挂 3DGS 渲染:候选 gsplat.js / Spark / PlayCanvas SuperSplat 内核,注意各自 license)。
- [ ] 输出压缩格式(SPZ/SOG)评估,文件大小 vs 质量。
- [ ] 多场景回归测试集 + 自动化指标报告。
- [ ] 文档 + 内部演示。

---

## 6. 质量评估体系 (Evaluation Protocol)
1. **数据划分**:每 8 张训练留 1 张测试(COLMAP 惯例),测试视角绝不参与训练。
2. **图像指标**:PSNR / SSIM / LPIPS(测试视角)。同时记录训练视角指标,**train-test gap 作为过拟合诊断**。
3. **几何指标**:渲染深度图 vs LiDAR 点云投影深度:AbsRel、RMSE、<2cm/<5cm 比例。
4. **压力测试**:相机移出采集轨迹 1–2 m + 2×/4× 放大截图,人工检查尖刺/雾/混色。固定几个标准机位做版本间对比。
5. **文字可读性**:场景中标牌/文字特写作为定性金标准(直接反映标定质量)。
6. **横向对比**:每个 Phase 末与 mipmap 同数据对比(盲测打分 + 指标)。

---

## 7. 风险与可能的问题 (Risks & Open Issues)

### 高风险
1. **相机-LiDAR 外参 / 时间同步精度不足** → 系统性糊,任何算法救不了。缓解:Phase 1 重投影检查作为硬门槛;必要时在训练中开位姿微调(gsplat 支持 pose refinement;参考 3R-GS/约束优化类工作处理粗位姿)。
2. **卷帘快门**:若鱼眼是 rolling shutter 且运动中采集,投影模型需含时间项。3DGUT 原生支持 rolling shutter——确认我们相机的快门类型,若是卷帘,把行曝光时间参数喂给 3DGUT。
3. **双鱼眼间的色彩/曝光差异**:PPISP 能补,但若差异极端需预先做双机白平衡对齐。

### 中风险
4. **场景规模**:扫描仪单趟可能覆盖数千平米、数千张图,单卡显存撑不住 → 分块(chunk)训练 + 合并方案(参考 hierarchical 3DGS / VastGaussian 思路),Phase 4 再解决,Phase 1–3 用中小场景。
5. **点云与照片的遮挡不一致**:点云投影深度图必须做可见性剔除,否则"穿墙深度"会污染监督。
6. **SpotLessSplats 无完整官方开源**:自实现有工作量;先用 SAM2 mask + 交叉验证兜底,SLS 作为增强。
7. **玻璃/镜面/无纹理白墙**:LiDAR 与照片都各有盲区,反光面深度监督要设置置信度降权。

### 低风险 / 待确认
8. 鱼眼实际 FoV(≥180°?)→ 决定是否裁剪到 160°。
9. 室内为主还是室内外混合?室外天空区域点云无覆盖,深度监督需 mask 掉天空(语义分割天空类)。
10. 目标 GPU 环境(客户侧本地?云端?)→ 影响 Phase 4 打包方式。
11. mipmap 输出的具体格式与质量指标,用于精确对标。

---

## 8. 关键参考资源 (References)

**代码库(优先级排序)**
- gsplat: github.com/nerfstudio-project/gsplat (Apache 2.0) — 核心底座;README + docs/3dgut.md 必读
- dn-splatter: github.com/maturk/dn-splatter — 深度/法向监督实现参考(基于 gsplat)
- 3dgrut: github.com/nv-tlabs/3dgrut — 3DGUT 官方,多传感器数据集支持
- SAM2 (Meta, Apache 2.0) — 语义 mask
- T-3DGS — 瞬态剔除备选实现

**论文**
- 3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting (CVPR 2025 Oral, arXiv 2412.12507)
- Fisheye-GS (ECCV 2024 WS, arXiv 2409.04751)
- 3DGS with Fisheye Images: FoV Analysis (arXiv 2508.06968) — 160° 最优结论
- DN-Splatter (WACV 2025, arXiv 2403.17822) — 深度/法向监督设计
- SpotLessSplats (arXiv 2406.20055, ACM TOG) — 瞬态剔除
- T-3DGS (arXiv 2412.00155) — 瞬态剔除(视频序列)
- Gaussian-LIC (arXiv 2404.06926) / LiGSM (arXiv 2503.05425) — LiDAR 初始化+深度监督
- ARSGaussian — LiDAR 引导致密化
- Constrained Optimization for 3DGS from Coarse Poses & Noisy LiDAR (arXiv 2504.09129) — 粗位姿容错
- Mip-Splatting — 缩放走样
- 2DGS — 平面质量备选升级

**许可证红线**
- ✅ gsplat / Nerfstudio / SAM2:Apache 2.0
- ❌ INRIA 原版 3DGS(graphdeco-inria/gaussian-splatting):非商用,禁止引入任何代码
- ⚠️ 每引入一个新仓库先查 LICENSE,记录在项目 NOTICE 文件

---

## 9. 给 Claude Code 的工作指引 (Instructions for Claude Code)

1. **从 Phase 0 开始**,逐项打勾推进;每个 Phase 的 Gate 不通过不进入下一阶段。
2. **Phase 1 的重投影验证是全项目最高优先级检查点**,提供可视化脚本(点云投影叠加鱼眼图)。
3. 所有实验记录指标到统一 CSV/表格(场景、配置、PSNR/SSIM/LPIPS、深度误差、训练时长、高斯数量),方便消融对比。
4. 数据转换器、训练配置、评估脚本分模块组织,风格与 CloudStudio 现有 Python 工具链一致(Open3D/PyTorch 栈)。
5. 遇到 gsplat API 变动以仓库最新 main 分支文档为准(3DGUT/LiDAR 功能更新很快)。
6. 涉及许可证的任何代码引入,先停下确认。
7. 开发机为 Mac(主力)+ 需要 CUDA GPU 训练环境——训练部分需在 Linux/Windows CUDA 机器上跑,Mac 上只做数据处理与脚本开发,注意环境分离。

---

## 附:待用户确认的问题清单
- [ ] 鱼眼 FoV 具体度数?畸变模型(KB4 / 等距 / OpenCV fisheye)?
- [ ] 快门类型:全局 or 卷帘?
- [ ] POS 文件格式样例(字段、坐标系、时间戳)?
- [ ] 相机-LiDAR 外参标定的重投影误差数据?
- [ ] 主要应用场景:室内 / 室外 / 混合?典型场景面积与照片数量级?
- [ ] 训练 GPU 资源(型号、显存)?部署目标(客户本地 / 云端)?
- [ ] mipmap 输出样例文件(用于对标)?
