# CloudStudio 3DGS — MVP S1 扫描数据的 3D Gaussian Splatting 处理管线

> **换机接力先读 [HANDOFF.md](HANDOFF.md)**。该文档是 2026-07-02 的历史现场记录；
> 当前实施顺序和验收边界以 [持续优化实施计划](docs/IMPLEMENTATION_PLAN.zh-CN.md) 为准。

独立于 CloudStudio 本体的 3DGS 处理软件工程。目标:用开源可商用组件(gsplat/Apache-2.0 路线)
处理 MVP S1 扫描仪输出(双鱼眼图像 + 逐帧位姿 + 高精度 SLAM 点云),替代第三方收费 3DGS 软件(mipmap),
并在动态剔除/几何精度上超越它。

> 本工程与 CloudStudio 仓库(`G:\cloudstudio-windows-XL-0424`)完全分离,只读取扫描仪数据,
> 不 import 其代码、不共享依赖。未来产品化后再考虑以独立服务形式接入 CloudStudio。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) | 调研报告与分阶段开发计划(plan of record) |
| [docs/S1_DATA_FORMAT.md](docs/S1_DATA_FORMAT.md) | **MVP S1 真实数据格式规格**(Phase 0 交付物,基于真实数据核验) |
| [docs/DATASETS.md](docs/DATASETS.md) | 本机可用的真实数据清单(原始记录 / 解算成品 / mipmap 对标输出) |
| [docs/IMPLEMENTATION_PLAN.zh-CN.md](docs/IMPLEMENTATION_PLAN.zh-CN.md) | 深度审查后的逐阶段实施、验证、提交和真实数据验收计划 |
| [NOTICE.md](NOTICE.md) | 第三方组件 license 台账(红线:禁止 INRIA 原版 3DGS 代码) |

## 目录结构

```
cloudstudio-3dgs/
├── docs/          # 计划、数据格式规格、数据清单
├── tools/         # 独立工具脚本
│   ├── inspect_recording.py   # 体检一条 S1 记录(标定/图像/位姿/点云是否齐全)
│   ├── reproject_check.py     # Phase 1 硬门槛:点云投影到鱼眼图叠加显示,验证标定+位姿+坐标约定
│   ├── build_lidar_init.py    # 确定性 voxel 初始化、预算和覆盖率报告、可选 PCA
│   ├── build_per_image_masks.py # 逐图 valid/static/depth-valid mask Manifest
│   ├── build_depth_cache.py   # KB4 ray-range、z-buffer、confidence 稀疏缓存
│   ├── build_split_manifest.py # Rig Frame 级 temporal/spatial/manual 正式切分
│   ├── evaluate_run.py        # masked 图像/深度指标与 HTML 质量报告
│   ├── build_corrected_pose_set.py # 关键帧 SE(3) 修正的 Rig 时间传播
│   ├── build_ba_match_graph.py # 仅训练集的双目/时序/空间回环匹配图
│   ├── run_hloc_aliked_lightglue.py # 锁定运行时的 ALIKED + LightGlue
│   ├── run_hloc_triangulation.py # 已知 POS 位姿几何验证与三角化
│   └── run_rig_ba.py          # 固定双相机 Rig、POS 先验和分阶段 BA
├── converter/     # S1 数据 → gsplat/nerfstudio 可读格式的转换器
│   ├── s1_common.py           # 共享:确定性 voxel 初始化、位姿换算(c2w_gl→w2c_cv)、PLY/四元数
│   ├── s1_to_colmap.py        # → COLMAP 格式(gsplat simple_trainer 输入,已实测)
│   └── s1_to_nerfstudio.py    # → nerfstudio transforms.json 格式
├── cloudstudio_3dgs/           # 产品侧正式 Python 模块
│   └── data/                    # 确定性数据 Manifest 与 S1 输入读取
├── experiments/   # 实验记录(runs.csv 指标矩阵)
└── NOTICE.md
```

## 快速开始(数据侧,无需 GPU)

```powershell
# 体检一条记录
python tools/inspect_recording.py G:\S1\2026-06-17_12-40-48gs2

# 建立正式数据清单；默认计算图片和点云内容哈希，不静默跳过缺失图片
python -m cloudstudio_3dgs.data.manifest `
  --recording G:\S1\2026-06-17_12-40-48gs2 `
  --run G:\S1\2026-06-17_12-40-48gs2\process\2026-06-17_12-40-48gs2_3 `
  --output G:\3dgs-datasets\gs2_manifest

# 定量数据 QA；未通过时返回非零退出码并写出 JSON/HTML/叠加图
python -m cloudstudio_3dgs.evaluation.data_qa `
  --recording G:\S1\2026-06-17_12-40-48gs2 `
  --run G:\S1\2026-06-17_12-40-48gs2\process\2026-06-17_12-40-48gs2_3 `
  --output G:\3dgs-datasets\gs2_qa

# 确定性 LiDAR 初始化；默认 40 万目标，严格小于 100 万 Gaussian 上限
python tools/build_lidar_init.py `
  --run G:\S1\2026-06-17_12-40-48gs2\process\2026-06-17_12-40-48gs2_3 `
  --output G:\3dgs-datasets\gs2_lidar_init `
  --config configs/lidar_init_8gb.json

# 为每张有位姿图片建立独立 mask 路径；PR-06/PR-07 再分别填入 static/depth-valid
python tools/build_per_image_masks.py `
  --manifest G:\3dgs-datasets\gs2_manifest\dataset_manifest.json `
  --output G:\3dgs-datasets\gs2_masks `
  --theta-max-deg 95

# 从 PR-04 voxel PLY 建立逐图稀疏 LiDAR ray-range 缓存；去掉 --max-images 才是全量
python tools/build_depth_cache.py `
  --manifest G:\3dgs-datasets\gs2_manifest\dataset_manifest.json `
  --mask-manifest G:\3dgs-datasets\gs2_masks\mask_manifest.json `
  --point-cloud G:\3dgs-datasets\gs2_lidar_init\sparse_pc.ply `
  --output G:\3dgs-datasets\gs2_depth `
  --workers 4 `
  --max-images 12

# 按完整 Rig Frame 建立正式切分；左右相机永远进入同一 split
python tools/build_split_manifest.py `
  --manifest G:\3dgs-datasets\gs2_manifest\dataset_manifest.json `
  --output G:\3dgs-datasets\gs2_evaluation\split_manifest.json `
  --mode temporal_block `
  --validation-fraction 0.1 `
  --nearest-train-warning-m 0.25

# 对签名 run manifest 中的完整验证集生成 JSON + HTML；缺 LPIPS/深度/资源项时明确 PARTIAL
# 已在兼容训练环境安装可选 LPIPS 时再追加 --lpips，避免改动锁定的 CUDA PyTorch
python tools/evaluate_run.py `
  --run-manifest G:\3dgs-runs\run_manifest.json `
  --split-manifest G:\3dgs-datasets\gs2_evaluation\split_manifest.json `
  --output G:\3dgs-runs\quality

# 将 transforms 关键帧相对 ImgPose 的修正传播到完整 Rig 时间轴；原始位姿永不覆盖
python tools/build_corrected_pose_set.py `
  --manifest G:\3dgs-datasets\gs2_manifest\dataset_manifest.json `
  --transforms G:\S1\2026-06-17_12-40-48gs2\process\2026-06-17_12-40-48gs2_3\transforms.json `
  --output G:\3dgs-datasets\gs2_poses

# PR-10：只从 train Rig 建立双目、同侧时序和空间回环图；validation 不得参与
python tools/build_ba_match_graph.py `
  --manifest G:\3dgs-datasets\gs2_manifest\dataset_manifest.json `
  --split-manifest G:\3dgs-datasets\gs2_evaluation\split_manifest.json `
  --output G:\3dgs-datasets\gs2_ba\match_graph.json `
  --hloc-pairs G:\3dgs-datasets\gs2_ba\pairs.txt

# 在独立可选环境中按 upstream/rig_ba.lock.json 安装精确版本后提取与匹配
python tools/run_hloc_aliked_lightglue.py `
  --image-dir G:\3dgs-datasets\gs2_colmap\images `
  --pairs G:\3dgs-datasets\gs2_ba\pairs.txt `
  --output G:\3dgs-datasets\gs2_ba\features `
  --require-cuda

# 从已有 POS/COLMAP 模型自动裁出 train-only 已知位姿模型并三角化
python tools/run_hloc_triangulation.py `
  --image-dir G:\3dgs-datasets\gs2_colmap\images `
  --reference-model G:\3dgs-datasets\gs2_colmap\sparse\0 `
  --pairs G:\3dgs-datasets\gs2_ba\pairs.txt `
  --features G:\3dgs-datasets\gs2_ba\features\features-aliked-n16.h5 `
  --matches G:\3dgs-datasets\gs2_ba\features\matches-aliked-lightglue.h5 `
  --feature-runtime-manifest G:\3dgs-datasets\gs2_ba\features\feature_runtime_manifest.json `
  --output G:\3dgs-datasets\gs2_ba\triangulation

# Stage 1 只优化 Rig 位姿；Stage 2 可优化 fx/fy；Stage 3 才可选 k1/k2
python tools/run_rig_ba.py `
  --model G:\3dgs-datasets\gs2_ba\triangulation\sfm `
  --manifest G:\3dgs-datasets\gs2_manifest\dataset_manifest.json `
  --match-graph G:\3dgs-datasets\gs2_ba\match_graph.json `
  --output G:\3dgs-datasets\gs2_ba\ba_stage1 `
  --through-stage stage_1 `
  --position-prior-stddev-m 0.05

# 重投影验证(全项目最高优先级检查点):
# 把解算点云投影回原始鱼眼图,输出多种坐标约定的叠加图供目视比对
python tools/reproject_check.py `
  --run-dir  G:\S1\2026-06-17_12-40-48gs2\process\2026-06-17_12-40-48gs2_3 `
  --raw-dir  G:\S1\2026-06-17_12-40-48gs2 `
  --out-dir  experiments\reproject_gs2

# 导出 gsplat 训练数据集(COLMAP 格式;首个数据集已出在 G:\3dgs-datasets\gs2_keyframes)
python converter/s1_to_colmap.py `
  --run-dir  G:\S1\2026-06-17_12-40-48gs2\process\2026-06-17_12-40-48gs2_3 `
  --raw-dir  G:\S1\2026-06-17_12-40-48gs2 `
  --out-dir  G:\3dgs-datasets\gs2_keyframes `
  --init-points 400000
```

## 可复现环境基线

CPU 数据工具依赖由 `pyproject.toml` 和 `uv.lock` 锁定：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\doctor.ps1
```

CUDA 训练环境另需 NVIDIA 驱动、CUDA Toolkit 12.8、VS2022 BuildTools、
Python 3.12 和 PyTorch 2.11.0+cu128。执行 `scripts\bootstrap.ps1 -Training` 时，
脚本会按 `upstream/gsplat.lock.json` 检出精确上游提交、校验补丁 SHA256、检查补丁可应用性，
再调用 Windows 构建脚本。不要直接安装 gsplat 的 `examples\requirements.txt`，它可能替换已选定的 CUDA PyTorch。

当前历史训练数据集仍包含约 1,019,218 个初始化点，超过旧 smoke 的 1,000,000 个 Gaussian 上限，
所以 `baselines/gs2_smoke.baseline.json` 继续标记为阻塞。PR-04 已从同一真实 LAS 独立生成
376,906 点的新初始化 PLY，结果锁定在 `baselines/gs2_lidar_init.baseline.json`；只有把新 PLY 接入训练数据集并完成 GPU smoke 后，
才能解除训练基线的阻塞状态。

## 关键事实(已用真实数据核验,详见 S1_DATA_FORMAT.md)

- 鱼眼模型:**OPENCV_FISHEYE (Kannala-Brandt k1–k4)**,2912×2912,f≈788px → 估算 FoV ≥190°
  (研究计划 §3.2:训练时需评估裁剪/mask 到 ~160°)。
- 解算输出自带 **`transforms.json`(NeRF 风格逐关键帧位姿 + 逐帧内参)** 和
  **`ImgPose.txt`(全部图像位姿)** —— 免 COLMAP/SfM 的前提成立。
- **位姿约定已实测钉死**:`transform_matrix` = c2w/OpenGL(= nerfstudio 原生,直接喂);
  `ImgPose.txt` 四元数 = c2w/OpenCV。竞品 mipmap 的 SDK 数据契约整理见
  [docs/references/](docs/references/)(同输入、935 对均匀采样、带位姿微调先验)。
- 点云:`colorized.las / uncolorized.las`,与位姿同一局部坐标系(首帧位姿在原点附近)。
- **只用局部坐标系产物**做 3DGS(`ImgPose.txt` + `colorized.las`);
  绝不用 `geo/ecef_*`(ECEF 百万级坐标,float32 训练直接崩,CloudStudio 侧已有同款教训)。
- 训练底座:gsplat + 3DGUT(`--with_ut --with_eval3d`,MCMC 致密化)。本机 GPU 为
  RTX 5070 Laptop 8GB —— 够跑小场景冒烟,正式训练需更大显存的机器。

## 阶段状态

- [x] Phase 0(部分):数据格式核验 → `docs/S1_DATA_FORMAT.md`;mipmap 数据契约 → `docs/references/`
- [x] Phase 0(基线):CPU 依赖锁、gsplat 上游/补丁锁、bootstrap/doctor、CPU CI
- [ ] Phase 0(外部验收):空白新机器安装 + gsplat CUDA 编译 + GPU 冒烟
- [x] 路线 PR-01:确定性数据 Manifest、内容哈希、原子写出、缺失输入硬失败
- [x] 路线 PR-02:记录级标定、619 对真实左右 Rig、固定外参与量化诊断
- [x] 路线 PR-03:定量数据 QA、JSON/HTML/叠加图、可配置 fail-closed 门槛
- [x] 路线 PR-04:确定性 voxel LiDAR 初始化、RGB 位深识别、点数预算和 stride 覆盖率对照
- [x] 路线 PR-05:逐图片 valid/static/depth-valid mask 契约、统一 crop/factor 和 masked PSNR
- [x] 路线 PR-07:KB4 LiDAR ray-range、前表面 z-buffer、confidence、mask 和确定性稀疏缓存
- [x] 路线 PR-08:Rig Frame 切分、泄漏告警、masked PSNR/SSIM/LPIPS、深度指标和 HTML 报告
- [x] 路线 PR-09:关键帧 SE(3) 修正、鲁棒过滤、Rig 时间插值、基线保持和默认位姿回退门
- [x] 路线 PR-10:训练集匹配图、锁定 ALIKED/LightGlue/HLoc、固定 Rig + POS 先验分阶段 BA 与回退报告；真实特征/BA 验收仍为 `NOT_RUN`
- [x] Phase 1(前置):重投影验证初步通过(gs2 场景目视贴合,约定=c2w_gl),
      正式 Gate 需再覆盖 2–3 场景 + 逐点误差统计
- [x] Phase 1(前置):COLMAP 数据集导出实测通过(gs2_keyframes:174 图/2 相机/101 万点,
      w2c 位姿往返校验与基线一致)
- [ ] Phase 1(剩余):gsplat 安装(待批准外部仓库)→ 裸跑 3DGUT+MCMC → 对比 mipmap
- [ ] Phase 2:LiDAR 深度监督 + 正则
- [ ] Phase 3:动态物剔除(SAM2 mask + 点云-照片交叉验证)
- [ ] Phase 4:产品化(CLI/API、Web 查看器、SPZ/SOG)
