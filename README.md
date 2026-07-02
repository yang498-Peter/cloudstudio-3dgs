# CloudStudio 3DGS — MVP S1 扫描数据的 3D Gaussian Splatting 处理管线

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
| [NOTICE.md](NOTICE.md) | 第三方组件 license 台账(红线:禁止 INRIA 原版 3DGS 代码) |

## 目录结构

```
cloudstudio-3dgs/
├── docs/          # 计划、数据格式规格、数据清单
├── tools/         # 独立工具脚本
│   ├── inspect_recording.py   # 体检一条 S1 记录(标定/图像/位姿/点云是否齐全)
│   └── reproject_check.py     # Phase 1 硬门槛:点云投影到鱼眼图叠加显示,验证标定+位姿+坐标约定
├── converter/     # S1 数据 → gsplat/nerfstudio 可读格式的转换器
│   └── s1_to_nerfstudio.py
├── experiments/   # 实验记录(runs.csv 指标矩阵)
└── NOTICE.md
```

## 快速开始(数据侧,无需 GPU)

```powershell
# 体检一条记录
python tools/inspect_recording.py G:\S1\2026-06-17_12-40-48gs2

# 重投影验证(全项目最高优先级检查点):
# 把解算点云投影回原始鱼眼图,输出多种坐标约定的叠加图供目视比对
python tools/reproject_check.py `
  --run-dir  G:\S1\2026-06-17_12-40-48gs2\process\2026-06-17_12-40-48gs2_3 `
  --raw-dir  G:\S1\2026-06-17_12-40-48gs2 `
  --out-dir  experiments\reproject_gs2
```

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
- [ ] Phase 0(剩余):gsplat CUDA 环境搭建 + MipNeRF360 冒烟
- [x] Phase 1(前置):重投影验证初步通过(gs2 场景目视贴合,约定=c2w_gl),
      正式 Gate 需再覆盖 2–3 场景 + 逐点误差统计
- [ ] Phase 1(剩余):s1_to_nerfstudio 转换器实测 → 裸跑 3DGUT+MCMC → 对比 mipmap
- [ ] Phase 2:LiDAR 深度监督 + 正则
- [ ] Phase 3:动态物剔除(SAM2 mask + 点云-照片交叉验证)
- [ ] Phase 4:产品化(CLI/API、Web 查看器、SPZ/SOG)
