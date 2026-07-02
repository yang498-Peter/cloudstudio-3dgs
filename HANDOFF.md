# HANDOFF — 新机器接力指南(2026-07-02 打包)

> 给接手的 Claude Code(或人):从这份文件读起。项目在旧机器(RTX 5070 Laptop 8GB)
> 上完成了全部数据侧工作,卡在 CUDA 编译太慢;你在新机器上从"编译 gsplat"继续。

## 0. 一句话状态

MVP S1 扫描仪 → 3DGS 的数据链路**全部打通并验证**(位姿约定、鱼眼标定、点云初始化、
圆形 mask、COLMAP 数据集),训练环境代码就绪但 gsplat **尚未在任何机器上编译成功过**
(旧机器编译到 >60 分钟被主动中止,原因和加速方法都已查明,见 §2)。

## 1. 这个压缩包里有什么

```
cloudstudio-3dgs\
├── HANDOFF.md            ← 本文件
├── README.md             ← 项目总览 + 阶段状态
├── docs\
│   ├── RESEARCH_PLAN.md  ← 调研报告 + 分阶段路线(plan of record,先通读)
│   ├── S1_DATA_FORMAT.md ← S1 数据格式规格(全部经真实数据核验,位姿约定已钉死)
│   ├── DATASETS.md       ← 旧机器上的数据清单(新机器没有这些原始数据!)
│   └── references\       ← mipmap 竞品 SDK 数据契约逆向文档
├── tools\                ← inspect_recording / reproject_check / make_fisheye_masks
├── converter\            ← s1_to_colmap(已实测) / s1_to_nerfstudio / s1_common
├── train\
│   ├── ENV.md            ← 环境搭建全记录(踩坑清单)
│   ├── setup_new_machine.cmd  ← ★ 一键编译 gsplat + 装依赖
│   ├── run_smoke_gs2.cmd      ← ★ 一键冒烟训练
│   └── patches\          ← 对 gsplat 的 3 处修改(已应用在 external\gsplat,补丁文件备份)
├── external\gsplat\      ← gsplat 源码(commit f2d1413,补丁已应用,glm submodule 已带)
├── data\gs2_keyframes\   ← ★ 训练就绪数据集(174 张鱼眼关键帧 + masks + images_4
│                            + 101 万 LiDAR 点 points3D.bin),开箱即训
└── experiments\runs.csv  ← 实验指标表(空模板)
```

git 仓库完整(7 个 commit),`external\`/`data\` 不入库但随包携带。

## 2. 新机器上的操作步骤(按顺序)

1. **前置安装**(手动):NVIDIA 驱动、CUDA Toolkit 12.8+、VS2022 BuildTools(C++ x64)、
   Python 3.12、`pip install torch --index-url https://download.pytorch.org/whl/cu128`。
   解压到 **纯 ASCII 路径**(mipmap 和我们自己都踩过中文路径的坑)。
2. **`train\setup_new_machine.cmd`** — 编译 gsplat + 装筛选过的依赖。脚本内置了旧机器
   查明的全部坑位修复:
   - `VSLANG=1033`(中文 MSVC 横幅让 torch 编译探测崩溃)
   - `DISTUTILS_USE_SDK=1`(torch 强制要求)
   - `TORCH_CUDA_ARCH_LIST=12.0` + `MAX_JOBS`(**只编 sm_120,这是旧机器编译
     >60 分钟的主因;设了之后应该 ~10 分钟**)
   - MSVC 不认 `__builtin_clzll` 的源码修复(已打在源码里,patches\ 有备份)
   - **绝不要 `pip install -r examples\requirements.txt`**——它钉死 torch==2.9.1,
     会把 cu128 的 torch 降级,Blackwell 直接废
3. **`train\run_smoke_gs2.cmd`** — 冒烟训练(3DGUT+MCMC,10k 步,728px,cap 100 万高斯)。
   预期:eval 在 5k/10k 步输出 PSNR/SSIM/LPIPS(`results\gs2_smoke\stats\`),
   10k 步导出 PLY(`results\gs2_smoke\`)。首次 eval 会联网下载 LPIPS 权重。
   如果新卡显存 >8GB,可改 `--data_factor 2` + `--strategy.cap-max 2000000` 提质量。

## 3. 冒烟成功的判定(研究计划 Phase 1 Gate 的前哨)

- 训练全程不 NaN、显存不爆;
- 渲染 eval 图(`results\gs2_smoke\renders\`)里建筑结构清晰、无整屏糊;
- **重点看清晰度而不是 PSNR 绝对值**(黑边参与 metric 会虚高,见 §4 已知问题 1);
- 若整体糊 → 按研究计划 §5 Phase 1 的原则:先查数据(位姿/标定),不要先调训练参数。
  但位姿链路已在旧机器重投影验证过(点云贴合 <2px 目视),糊的嫌疑应先落在
  训练配置/降采样上。

## 4. 已知问题与设计决策(新 agent 必读)

1. **mask 只挡 loss,不挡 metric**:圆外黑边像素不进 L1/SSIM 训练损失(patch 已实现),
   但 eval 的 PSNR/SSIM/LPIPS 仍算全图 → 指标会虚高。正式对比 mipmap 前要改成
   只在 mask 内算指标(改 simple_trainer eval 段,或后处理 stats)。
2. **数据集只有 174 个关键帧**(SDK 按 2m/15° 阈值筛的),对 3DGS 偏稀。加密方案:
   converter 改用 `ImgPose.txt`(1238 张全量位姿)+ `info/calibration.json` 组帧
   ——格式都已在 docs/S1_DATA_FORMAT.md §4 钉死,s1_to_colmap.py 加个 `--use-imgpose`
   即可(注意 ImgPose 是 c2w/OpenCV 轴,用 s1_common.GL_TO_CV 换算,别搞混)。
   但新机器没有原始数据,需要把 `G:\S1\2026-06-17_12-40-48gs2` 一并拷过去。
3. **FoV 消融未做**:mask 现在封在 190°(镜头实际成像圈更大)。研究计划 §3.2 说
   160° 最优 → 用 `tools\make_fisheye_masks.py --theta-max-deg 80` 重生成 mask
   跑对照(数据集里 masks\ 可直接覆盖,不用重导数据)。
4. **位姿约定(已锁死,勿重推导)**:solver `transforms.json` = c2w/OpenGL(nerfstudio
   原生);`ImgPose.txt` = c2w/OpenCV。验证过程见 docs/S1_DATA_FORMAT.md §4/§5。
5. **曝光补偿(PPISP)未开**:依赖没装(git+CUDA 编译)。S1 双鱼眼自动曝光,亮度跳变
   是已知风险 → 冒烟如果出现明显明暗斑块,优先装 ppisp 或开 bilateral grid
   (simple_trainer `--post_processing` 相关,PPISP 要求 MCMC,我们正好是)。
6. **ECEF 红线**:任何时候只用局部坐标系产物训练,`geo\ecef_*` 一律不碰。

## 5. 冒烟之后的路线(研究计划 Phase 1 收尾 → Phase 2)

1. 全分辨率/半分辨率正式跑(新卡显存决定),`--max_steps 30000` 默认值。
2. 与 mipmap 对标:旧机器 `G:\Tersus3DGSResults\20260609-234131-s1-3dgs\3D\model-gs-ply\gs.ply`
   (4855 万高斯/2.5GB)。同数据对比需要把 mipmap 对应场景确认清楚(见 DATASETS.md TODO)。
3. Phase 2 =深度监督:LiDAR 点云投影深度图 + DN-Splatter 式 log-depth loss。
   simple_trainer 已有 `--depth_loss` 雏形(colmap parser 的 load_depths 用 points3D
   投影稀疏深度),可以先开这个开关白嫖一版,再上稠密深度图管线。
4. 所有实验记入 experiments\runs.csv(列已定好)。

## 6. 旧机器上没带走的东西(需要时再传)

- 原始扫描记录(75GB,G:\S1 等,见 docs/DATASETS.md)——加密帧/换场景时需要
- mipmap 成品软件和它的输出(对标用)
- CloudStudio 主仓库(只读参考,本工程不依赖它)

## 7. 给新 agent 的开场提示词(复制可用)

> 读 HANDOFF.md 和 docs/RESEARCH_PLAN.md。当前任务:在这台新机器上执行
> train\setup_new_machine.cmd 编译 gsplat(预计 ~10 分钟,失败就按 train\ENV.md
> 的坑位清单排查),然后跑 train\run_smoke_gs2.cmd 冒烟训练,把 results\gs2_smoke
> 的 eval 渲染图和 stats 指标整理给我看,按 HANDOFF.md §3 判定冒烟是否通过。
> 通过后按 §5 路线推进;改 gsplat 源码时同步更新 train\patches\。
