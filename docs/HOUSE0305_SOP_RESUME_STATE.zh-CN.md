# DA2 解除阻塞后的续跑清单

## 需要批准的下载（仅此一项）

| 项 | 来源 | 大小 |
|---|---|---|
| `depth_anything_v2` Python 包 | github.com/DepthAnything/Depth-Anything-V2 | 约 2 MB |
| `depth_anything_v2_vits.pth` | huggingface.co/depth-anything/Depth-Anything-V2-Small | 约 99 MB |

编码器由 `tools/build_da2_face_cache.py::_load_model` 写死为
`DepthAnythingV2(encoder="vits", features=64, out_channels=[48,96,192,384])`，
即 **Small**，不是 Base/Large。

## 批准后按顺序执行

设 `SRC` = 源码包解压根目录（内含 `depth_anything_v2/` 子包），
`CKPT` = `depth_anything_v2_vits.pth` 路径，
`OUT` = `C:\Peter\3dgs-datasets\house0305_sop_v8`。

### 第 13 步 DA2（train + val）

```
python tools/build_da2_face_cache.py --face-manifest %OUT%\face4\face_manifest.json ^
  --face-root %OUT%\face4 --dataset-manifest %OUT%\dataset_manifest.json ^
  --depth-manifest %OUT%\depth\depth_manifest.json --depth-root %OUT%\depth ^
  --model-source %SRC% --checkpoint %CKPT% --output %OUT%\da2_train --device cuda
```
val 同理，改 `face4_val` / `da2_val`。

### 第 14 步 独立天空

竞品交付件实测参数（本次审计所得，直接对齐）：
100,000 点、SH0、包围盒 403.5×404.0×283.6 m（半径约 202 m）、
最长轴 P50 2.056 m、轴比 P50 1.521（近各向同性）、opacity P50 0.100。

```
python tools/build_independent_sky_background.py --face-manifest %OUT%\face4\face_manifest.json ^
  --face-root %OUT%\face4 --mono-manifest %OUT%\da2_train\<manifest> --mono-root %OUT%\da2_train ^
  --output %OUT%\sky_train --sky-count 100000 --sky-radius-m 202 --sky-scale-m 2.056 --sky-opacity 0.100
```
先看工具默认值再决定是否覆盖；`--sky-count 100000` 与竞品完全一致，可直接用。

### 门禁推进链（严格前缀，不可跳）

```
tools/advance_mipmap_renderer_mask_gate.py   -> RENDERER_MASK_READY
tools/advance_mipmap_lidar_depth_gate.py     -> LIDAR_DEPTH_READY
tools/advance_mipmap_da2_gate.py             -> DA2_DEPTH_READY
tools/advance_mipmap_sky_gate.py             -> SKY_BACKGROUND_READY
tools/build_lidar_face4_adaptive_tile_plan.py（需 --sky-gate）
tools/advance_mipmap_tile_gate.py            -> UPSTREAM_DATA_READY
```

### 第 15 步之后

```
tools/materialize_lidar_tile_inputs.py
tools/build_mipmap_tile_geometry.py          （K=7 邻距尺度 / K=30 PCA 法向 / [d,d,0.5d]）
tools/build_tile_face4_lidar_geometry.py     （逐 Tile 侧车）
```

### 第 16 步训练配置（DLL 审计得出的修正）

免代码的配置项：

- `factor = 1`（**底线，不得退回 2**；我方 41 次历史训练全是半分辨率）
- `split_scale_m = 0.15`（DLL 硬编码值，我方原为 0.2）
- `capacity_cap` 用**绝对整数**（DLL 证实竞品是绝对上限，不存在倍率 C）
- `grow_grad2d` 必须先扫本场景梯度分位再定（竞品该值是宿主传入、不在二进制里；雪堆定标为 7.5e-5）
- RGB `0.6·L1 + 0.4·(1−SSIM)`；LiDAR range 0.05；LiDAR 法向 0.01；DA2/mesh 权重 0

需改代码（**尚未实施**）：

- 裁剪阈值缩放与前半段加倍。已用 capstone 独立复核，并更正了首轮报告的三处错误（加倍的是
  `0x128` 不是"阈值 A"；系数对应为 `0x128→×0.25`、`0x124→×5.0`、`0x130→×5.0`；两种调整
  **互斥不叠加**）。精确指令序列见 `COMPETITOR_BINARY_AND_ARTIFACT_AUDIT.zh-CN.md` §5.1。
  **仍不可实施**：三个偏移各自对应哪个物理量（透明度／屏幕半径／世界尺度）尚未确定，
  乘数已知而被乘数未知。
- 生命周期顺序 Split 先于 Clone（我方 `duplicate()` 在 501 行、`split()` 在 649 行）。两个掩膜
  互斥（`small` vs `~small`），实际只影响索引布局，优先级低。

形态目标（竞品实测）：高斯数 22,452,075、最短轴 P50 0.428 mm、轴比 P50 10.235。
我方 D1 对应为 7,453,193 / 5.006 mm / 2.297。

## 已就绪的产物（全部签名验证通过）

| 内容 | 路径 | 签名 |
|---|---|---|
| 训练数据集清单 | `%OUT%\dataset_manifest.json` | `33f5f814` |
| 圆掩膜 | `%OUT%\masks\mask_manifest.json` | `b96cffe2` |
| 人物掩膜 | `%OUT%\person_mask_manifest.json` | `52a27011` |
| split（796+90） | `%OUT%\split_manifest.json` | `a792f3c8` |
| Face4 train（3184 面） | `%OUT%\face4\face_manifest.json` | `19552ab4` |
| Face4 val（360 面） | `%OUT%\face4_val\face_manifest.json` | `040b0b06` |
| renderer 掩膜 train | `%OUT%\renderer_mask_train.json` | `20e632a7` |
| renderer 掩膜 val | `%OUT%\renderer_mask_val.json` | `bdbb3d6a` |
| LiDAR 深度（886 图） | `%OUT%\depth\depth_manifest.json` | `af8fb540` |
| Face4 LiDAR 侧车 train | `%OUT%\face4_lidar_train` | `bacd67bf` |
| Face4 LiDAR 侧车 val | `%OUT%\face4_lidar_val` | `e5ae6532` |

Face4 LiDAR 侧车已完成：train `3184/3184` 面全部有真实命中、有效像素 `1,263,932,760`；
val `360/360` 面全部有命中、`114,333,560` 像素。两个划分都是 **100% 覆盖**（雪堆同阶段为
95.5% / 87.5%）。共 7.4 GB，按 §9.4 的正式算法用完整 LAS 经 `c2w` + `R_face/K_face`
直接投影生成，未走鱼眼整数像素中转。每个逐 Tile 侧车都要绑定它作为上游。

可复用、无需重建：`house0305_mesh_full\tile_lidar_surface_bpa_full.ply`（位姿无关）。
已作废：`house0305_face4_geom_ba`（584 图 × 11 面的旧方案，且绑定旧 AT）。
