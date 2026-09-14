# SDK 数据接入层（cloudstudio3dgs_sdk/ingest）

本文件描述 SDK 的接入层：如何把 house0305 之外的数据集变成一个 `DatasetBundle`，
如何由 bundle 推导出训练配方需要的全部缓存，以及在没有人工切块的场景上如何自动
得到 tile 边界。

代码位置：

| 文件 | 职责 |
| --- | --- |
| `cloudstudio3dgs_sdk/ingest/bundle.py` | `DatasetBundle` 数据类 + 签名的 `bundle_manifest.json` |
| `cloudstudio3dgs_sdk/ingest/adapters/s1_fisheye.py` | MVP-S1 鱼眼双目 + 着色 LAS（house0305 的形状） |
| `cloudstudio3dgs_sdk/ingest/adapters/colmap.py` | COLMAP 二进制稀疏重建 + 可选 LiDAR 点云 |
| `cloudstudio3dgs_sdk/ingest/adapters/pinhole_folder.py` | 图片目录 + poses json（原生方言与 nerfstudio 方言） |
| `cloudstudio3dgs_sdk/ingest/caches.py` | `CachePlan`：缓存清单、依赖顺序、sha 绑定、跳过与 GPU 拒绝 |
| `cloudstudio3dgs_sdk/ingest/tiling.py` | 等点数板条切分规则，输出训练器可校验的 tile plan |
| `cloudstudio3dgs_sdk/ingest/cli.py` | `detect` / `bundle` / `tile` / `plan` 四个子命令 |

对应测试：`tests/test_ingest_bundle.py`、`tests/test_ingest_adapters.py`、
`tests/test_ingest_caches.py`、`tests/test_ingest_tiling.py`（共 101 个用例，全部
CPU、不 import torch）。

---

## 1. 适配器契约

每个适配器是一个模块，暴露四个名字：

```python
NAME: str          # 唯一标识，例如 "colmap"
VERSION: str
REQUIRES: str      # 一句话说明它要什么，出现在探测失败的报错里
def detect(path: Path) -> bool: ...
def load(path: Path, **kwargs) -> DatasetBundle: ...
```

约束：

1. **`detect` 只看结构，不看内容。** 判断依据是"哪些文件存在"，不读点云、不解码图片。
   这样一个结构正确但内容损坏的数据集会被 *识别* 然后 *带着具体原因被拒绝*，而不是
   悄悄掉到下一个适配器上。
2. **探测必须互斥。** `detect_adapter()` 要求恰好一个适配器认领路径；零个或多个都抛
   `DatasetDetectionError`，并把每个适配器的 `REQUIRES` 打印出来。
   已处理的一处歧义：S1 的 run 目录里也有 `transforms.json`，`pinhole_folder.detect`
   显式排除带 `info/calibration.json` + `camera/` 的目录。
3. **`load` 失败关闭（fail closed）。** 任何缺失都抛 `DatasetIncompleteError`，消息里
   必须含具体路径或键名。禁止让裸 `KeyError` / `FileNotFoundError` 逃出去——调用方是
   流水线 runner，它没法从 `KeyError: 'fl_x'` 推断是哪个文件的哪一行。
4. **`load` 只读。** 不写文件、不做格式转换。COLMAP 的稀疏点转 PLY 是显式的单独调用
   `colmap.export_sparse_point_cloud()`，不是 `load` 的副作用。

### 1.1 s1_fisheye：与现有输入逐字节一致

S1 适配器 **不重写** S1 读取逻辑。它调用
`cloudstudio_3dgs.data.manifest.build_manifest()`——也就是当初生成
`house0305_sop_v8/dataset_manifest.json` 的同一段代码——再把结果映射成 bundle，
并把原始 payload 保存在 `bundle.native_dataset_manifest`。

如果数据集旁边已经存在签名过的 `dataset_manifest.json`（或通过
`dataset_manifest=` 传入），适配器 **校验后原样复用**，不重建。因此 house0305 走 SDK
接入时 `manifest_sha256` 保持不变，下游所有缓存的绑定关系不会被无谓地打断。

### 1.2 bundle_manifest.json

`write_bundle_manifest()` 写出的清单用的是仓库统一的规范化方式
（`cloudstudio_3dgs.data.manifest.canonical_json_bytes`：`sort_keys=True`、
`separators=(",", ":")`、无缩进），签名键是 `bundle_manifest_sha256`，签名时把该键本身
排除。因此 bundle 的 sha 可以和训练器其它清单的 sha 直接比较、互相绑定。

`verify_bundle_manifest(payload, verify_artifacts=True)` 额外逐文件核对图片与点云的
sha256。

---

## 2. 能力集（capabilities）：不假设，只声明

通用数据集可能没有 LiDAR、没有时间戳、没有刚性双目。bundle 不去填默认值，而是导出
一个 `capabilities` 集合，`CachePlan` 按 token 拒绝：

| token | 含义 | 缺失时被挡住的缓存 |
| --- | --- | --- |
| `lidar_point_cloud` | 有场景点云 | `depth_cache`、`face_lidar_geometry`、`tile_plan`、`tile_inputs`、`tile_geometry`、`tile_ownership_*` |
| `capture_timestamps` | 每张图有纳秒时间戳 | `split_manifest`（时间分块与 rig 配对失效） |
| `camera_rig` | 多相机且都有 `transform_from_lidar` | rig-frame 级别的划分退化为逐图划分 |
| `declared_split` | 数据集自带 train/val | 无（改为由 `split_manifest` 生成） |
| `fisheye_source_images` | 源图是鱼眼 | `face_cache`（Face4 展开只对鱼眼成立） |
| `image_content_hashes` | 每张图有 sha256 | 缓存绑定无法核验 |

被挡住的缓存在计划里状态是 `BLOCKED`；`build(dry_run=False)` 碰到它会抛
`DatasetIncompleteError`，消息里点名缺的是哪个 token。

---

## 3. 缓存依赖图与代价

参考配方：`C:\Peter\3dgs-runs\house0305_sop\tile1_B5_cap6_20k.json`。
下面的依赖边就是配方里那些 `*_manifest` 字段之间的 sha 绑定关系。

```
dataset_manifest
├── mask_manifest
│   ├── person_mask_manifest      [GPU]
│   └── depth_cache               (需要 LiDAR)
├── split_manifest                (需要时间戳)
└── (以上四者共同喂给)
        face_cache                (需要鱼眼)
        ├── renderer_mask
        ├── face_lidar_geometry   (需要 LiDAR；含隐藏点剔除)
        ├── mono_depth            [GPU]  DA2 + LiDAR 仿射对齐
        └── sky_masks             (SegFormer，默认 CPU)
                └── tile_plan     (板条规则；需要 LiDAR)
                    └── tile_inputs
                        ├── tile_geometry          (K=7 间距 / K=30 PCA)
                        ├── tile_ownership_<id>    (每 tile 一份)
                        └── view_backgrounds_<id>  [GPU] 每 tile 一份
```

### 3.1 代价表（house0305：884 张训练图 → 3536 个 face）

| 缓存 | 设备 | 构建器 | 耗时 | 依据 | 产物体积 |
| --- | --- | --- | --- | --- | --- |
| `dataset_manifest` | CPU | `python -m cloudstudio_3dgs.data.manifest` | ~2 min | 估算（本机 sha256 约 1 GB/s，3.4 GB 图 + 0.7 GB LAS） | 1 MB |
| `mask_manifest` | CPU | `tools/build_per_image_masks.py` | ~1 min | 实测（v8 产物时间跨度 18:18 → 18:19） | 19 MB |
| `person_mask_manifest` | **GPU** | `tools/build_person_masks.py` | ~25 min | 估算（886 张过 maskrcnn_resnet50_fpn_v2 @800px） | 0.9 GB |
| `depth_cache` | CPU | `tools/build_depth_cache.py` | ~95 min | 估算（v8 跨度 18:19 → 20:16 为上界） | 9.8 GB |
| `split_manifest` | CPU | `tools/build_split_manifest.py` | < 1 min | 实测（v9 跨度） | 0.2 MB |
| `face_cache` | CPU | `tools/build_face_cache.py` | **31 min** | 实测（v9：02:14 → 02:45） | 25 GB |
| `renderer_mask` | CPU | `tools/build_renderer_mask_manifest.py` | **6 min** | 实测（v9：02:46 → 02:52，train+val，主要花在 sha 核验） | 1.3 MB |
| `face_lidar_geometry` | CPU | `tools/build_face4_lidar_geometry.py` | **72 min** | 实测（v9：03:11 → 04:23，`--visibility-cell-px 6`；不做隐藏点剔除的同一构建器是 19 min） | 7.6 GB |
| `mono_depth` (DA2) | **GPU** | `tools/build_da2_face_cache.py` | **53 min** 串行 / **11 min** 五分片 | 实测（v9 分片日志：0.9 s/face） | 1.6 GB |
| `sky_masks` | CPU | `tools/build_sky_masks.py` | **168 min** | 实测（v9 日志：21:46:06 → 00:34:40，6 线程 CPU） | 61 MB |
| `tile_plan` | CPU | `cloudstudio3dgs_sdk.ingest.cli tile` | ~1 min | 实测（单遍流式 LAS：18.76 M 点 2.7 s） | 10 MB |
| `tile_inputs` | CPU | `tools/materialize_lidar_tile_inputs.py` | ~3 min | 实测（单遍 LAS 6.2 s + 280 MB PLY 写入与 sha） | 280 MB |
| `tile_geometry` | CPU | `tools/build_mipmap_tile_geometry.py` | ~40 min | 估算（19.4 M 含 halo 点的 K=30 PCA，964 MB npz） | 964 MB |
| `tile_ownership_<id>` | CPU | `tools/build_tile_ownership_masks.py` | **4–8 min / tile** | 实测（v9，10 workers） | 190 MB / tile |
| `view_backgrounds_<id>` | **GPU** | `tools/build_tile_view_backgrounds.py` | ~12 min / tile | 实测（v9 日志：1829 个视角渲染 1.5 min，其余是 stand-in 组装） | 2.5 GB / tile |

冷启动一个 4-tile 场景的总量级：CPU 约 7 小时（`sky_masks` 与
`face_lidar_geometry` 两项就占了四小时），GPU 约 1.5 小时，磁盘约 50 GB。

`CacheSpec.cost_basis` 字段在代码里逐条标注了 `measured` 还是 `estimated`；测试
`test_every_cache_states_a_cost_and_its_basis` 保证这一列不会退化成空话。

### 3.2 跳过规则

`CachePlan.status_of()` 对每个缓存给出四种状态之一：

* `PRESENT` — 签名清单存在，且它记录的每一条上游 sha 都等于上游清单当前的 sha。跳过。
* `STALE` — 清单存在但某条绑定对不上（或者清单没签名）。**重建**，绝不静默复用。
  状态里的 `reason` 会打印是哪个键、记录值与实际值的前 12 位。
* `MISSING` — 没有清单。
* `BLOCKED` — bundle 缺能力（见第 2 节）。

绑定用点分路径表达，因为 Face4 的绑定藏在嵌套结构里，例如
`source_identity.dataset_manifest_sha256`。`person_mask_manifest_sha256` 标记为
`optional=True`：不做人像遮罩的数据集，face cache 里就没有这个键，这不算 stale。

### 3.3 GPU 拒绝

`build(dry_run=True)` 只打印计划；`build(dry_run=False)` 只跑 CPU 构建器，遇到 GPU
缓存立即抛：

```
GpuStepRequired: mono_depth is a GPU step, run through the SDK runner
                 (builder: tools/build_da2_face_cache.py)
```

理由：设备放置、排队和显存预算属于 SDK runner，接入层不该在这个进程里占一个 CUDA
context。另外，如果 profile 没提供某个必需路径（DA2 权重、trainer config 等），命令里
会留下 `<da2_checkpoint>` 这样的占位符，执行时抛 `CachePlanError` 并点名占位符，而不是
把字面量 `<da2_checkpoint>` 传给子进程。

---

## 4. 自动切块规则

house0305 的 4 个 tile 来自投影像素 kd 规划器
（`cloudstudio_3dgs.pipeline.adaptive_tiling`），它需要完整的 Face4 观测表和实测显存
预算才能切。缓存建好之后那是对的规划器；对接入层是错的——**切块必须在 per-tile 缓存
建之前就定下来**。

因此 `tiling.py` 实现一条独立的、确定性的规则：

1. 取 LiDAR 包围盒，按 `scene_padding_fraction`（默认 0.2，与生产规划器的 root box
   一致）向外扩，作为 root box。
2. 选切分轴：**未扩边**点包围盒里 X/Y 中较长的那一条；`axis` 参数可显式指定。
   永不切 Z——沿 Z 切会把地板和天花板分开。
3. 把该轴切成 `tile_count` 个 **等点数**（不是等长度）板条，切点取点数分位数。
   分位数由固定宽度直方图（`histogram_bins`，默认 4096）加桶内线性插值得到，再按
   `cut_decimals`（默认 6）取整。因此同一份点云永远给出同一组切点，与分块顺序、
   读取顺序、平台都无关。
4. `core_box` → `training_and_export_box`：给了 `overlap_margin_m` 就按绝对米数外扩，
   否则按 `halo_fraction_per_side`（默认 0.002，与
   `AdaptiveTilingConfig.spatial_halo_fraction_per_side` 一致）按盒子尺寸比例外扩。

### 4.1 参数表

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `tile_count` | 4 | 板条数 |
| `axis` | `"auto"` | `auto` / `x` / `y`；`z` 直接报错 |
| `scene_padding_fraction` | 0.2 | root box 每侧外扩比例 |
| `halo_fraction_per_side` | 0.002 | 导出 halo，按盒子尺寸比例 |
| `overlap_margin_m` | `None` | 给定则改用绝对米数 halo，覆盖上一项 |
| `histogram_bins` | 4096 | 分位数分辨率（house0305 上约 3 cm） |
| `cut_decimals` | 6 | 切点取整位数，保证确定性 |
| `minimum_slab_extent_m` | 1.0 | 板条薄于此值直接报错，提示减少 tile 数 |
| `resolution_level` | 1 | 只用于填 plan 里的 `bytes_per_pixel` |

产出的 plan 用的就是训练器已经在校验的 schema
（`adaptive_projected_pixel_kd_xy_v1`），所以
`cloudstudio_3dgs.training.tile_inputs.materialize_lidar_tile_inputs` 可以原样消费，
`verify_adaptive_tile_plan` / `verify_tile_inputs_manifest` 可以原样校验。
core box 保持对 root box 的**无缝无重叠划分**，这是
`cloudstudio_3dgs.training.tile_ownership` 分配 core 归属的硬性要求；只有 export box
才重叠。

`views` 字段需要投影观测表。没有观测表时 plan 里 `views_source: "deferred"`、
`views: []`，`CachePlan` 会在这个状态下拒绝物化 tile inputs——否则会写出一份没有任何
视角能训练的清单。

### 4.2 与 house0305 手工/自适应切块的对比

LAS：18,757,869 点，包围盒 X 跨度 123.612 m、Y 111.108 m、Z 29.883 m，所以规则选 X 轴。
切点：`x = 3.993404 / 7.131795 / 10.101953`。两种切法覆盖的 root box 体积完全相同
（1,139,765.23 m³），所以下面是同口径比较。

| | core box 实测点数 | max/min | training_and_export_box 实测点数 | 合计（halo 重复） |
| --- | --- | --- | --- | --- |
| SDK 板条（4 块，halo 0.002） | 4 691 018 / 4 685 082 / 4 691 844 / 4 689 925 | **1.001** | 5 133 594 / 4 721 828 / 4 715 589 / 4 900 456 | 19 471 467（+3.80%） |
| 现有 v9 tile（自适应 kd） | 6 799 907 / 3 334 980 / 3 186 167 / 5 436 815 | **2.134** | 7 044 777 / 3 417 320 / 3 309 574 / 5 651 827 | 19 423 498（+3.55%） |

（现有那一列的 t&e 点数与 `tile_inputs_v9` 清单里记录的
`initialization.point_count` 完全相同，说明这个对比是精确的。）

**两组边界不一样，这是预期的，不是缺陷。** 自适应规划器沿 X 切一刀后又在左半边沿 Y
切了两刀，目标是把 *投影像素负载* 和预测高斯驻留量压到显存预算以下；板条规则只看点
数，目标是每块初始化规模相当。结论：

* 点数均衡度显著更好（1.001 vs 2.134），halo 重复率相当（3.80% vs 3.55%）。
* 板条宽度极不均匀（85.1 / 3.1 / 3.0 / 81.8 m），因为 house0305 一半的点集中在 6 m 宽
  的 X 带里，而 20% 的外扩全部落进首尾两块。生产规划器切出的 Tile_1 也只有 3.5 m
  厚，这一点上两者行为一致。
* 板条薄的时候按比例的 halo 会很小（house0305 上 Tile_1 约 7 mm，生产的 Tile_1 同样
  是 7 mm）。改用 `overlap_margin_m=0.5` 在这个场景会让 halo 重复率从 3.80% 涨到
  **23.51%**，所以默认保留比例式 halo；薄板条场景要用绝对 margin 时需要知道这个代价。
* 板条规则 **不看显存**。它保证的是初始化规模均衡，不保证每块都能塞进某张卡；显存这
  一层仍然由 `cap_max` 和 runner 负责。

`tests/test_ingest_tiling.py::House0305ComparisonTest` 在本机有真实 LAS 时会跑这组
断言，没有就 skip。

---

## 5. 一个新数据集最少要提供什么

**必须**：

1. 图片文件本身。
2. 每张图的位姿（4×4 camera-to-world；COLMAP 的 world-to-camera 由适配器转换，
   nerfstudio 的 OpenGL 轴向由适配器翻转并在 `warnings` 里写明）。
3. 每台相机的内参：`width`、`height`、`fx`、`fy`、`cx`、`cy`、畸变模型与参数。

**强烈建议**：

4. 场景 LiDAR 点云（LAS/LAZ/PLY），与位姿同一坐标系、同一尺度。
5. 每张图的纳秒时间戳。
6. 多相机时每台相机相对 LiDAR 的外参（`transform_from_lidar`）。

### 5.1 缺项时的降级

| 缺什么 | 后果 |
| --- | --- |
| **LiDAR 点云** | `depth_cache` / `face_lidar_geometry` / `tile_plan` / `tile_inputs` / `tile_geometry` / `tile_ownership_*` 全部 `BLOCKED`。配方里的 `lidar_range_weight`、`lidar_alpha_weight`、`lidar_normal_alignment`、`surface_initialization`、`metric_scale_calibration` 全部失去输入；DA2 也没有可对齐的度量深度，只剩相对深度。COLMAP 数据集可以用 `colmap.export_sparse_point_cloud()` 把三角化点提升成点云，但那是显式调用：尺度、密度、噪声水平和实测 LiDAR 不是一回事，切块与表面初始化的结论不能直接搬。 |
| **时间戳** | `split_manifest` 的时间分块和 rig 配对失效，必须改喂 `--manual` 划分文件。`bundle.capabilities` 里没有 `capture_timestamps` 时 `split_manifest` 直接 `BLOCKED`。 |
| **rig 外参** | rig-frame 级别的 train/val 划分退化为逐图划分，同一时刻的左右图可能分到不同集合，保留集会偏乐观。 |
| **鱼眼源图** | `face_cache` `BLOCKED`。整条下游链路（renderer mask、face lidar geometry、DA2、sky mask、tile views、ownership）都以 Face4 的 `image_id::face_id` 为样本键，针孔数据集要接进来需要一条"直通 face cache"——目前 **没有实现**，见下一节。 |
| **图片 sha** | 缓存绑定无法核验，`bundle_manifest` 里写 `not_computed` 并加一条 warning。 |

### 5.2 接入层给不了、配方却需要的东西

以下是通用数据集 *结构上* 提供不了的，必须由 profile / runner / 人工补：

1. **模型权重**：Mask R-CNN（人像）、Depth Anything V2、SegFormer ADE20K。三个都不随
   仓库分发（SegFormer 是 NC 许可），必须由 profile 给路径。
2. **`mipmap_pipeline_gate`**：配方里那份签名的就绪门（`gates_v9/gate_17_training_da2.json`）
   由 gate 工具链产出，不属于接入层。
3. **`view_backgrounds` 的自举**：per-view 背景需要 *相邻 tile 已经训练好的 checkpoint*
   做 stand-in，还需要一份天空球探针。第一轮没有 checkpoint，只能先用纯 sky dome 跑，
   然后重建。这是一条真实的循环依赖，计划里用 `note` 标了出来。
4. **人像遮罩的人工复核**：`tools/finalize_person_mask_review.py` 的人工确认环节不在
   计划里。
5. **针孔数据集的 face cache 直通**：见 5.1 最后一行。
6. **`gsplat_lock`**：配方绑定的上游 gsplat 提交由环境决定，不是数据属性。
7. **坐标系语义**：bundle 只记录 `coordinate_frame` 字符串。位姿和点云是否真的同系
   同尺度，接入层无法证明，只能失败在下游。

---

## 6. 常用命令

```bash
# 这个目录是什么数据集？
python -m cloudstudio3dgs_sdk.ingest.cli detect <dataset>

# 载入并看能力集，顺便写出 bundle_manifest.json
python -m cloudstudio3dgs_sdk.ingest.cli bundle <dataset> --output <work>/bundle

# 只算 tile 边界（不需要 Face4 缓存）
python -m cloudstudio3dgs_sdk.ingest.cli tile \
    --point-cloud <scene.las> --output <work>/tile_plan/adaptive_tile_plan.json \
    --tile-count 4

# 打印缓存计划（不动任何文件）
python -m cloudstudio3dgs_sdk.ingest.cli plan <dataset> \
    --dataset-root <datasets>/<name> --run-root <runs>/<name> --report plan.json

# 只跑 CPU 那一半，遇到 GPU 缓存停下
python -m cloudstudio3dgs_sdk.ingest.cli plan <dataset> ... --execute
```

Python API：

```python
from cloudstudio3dgs_sdk.ingest import load_dataset, plan_caches

bundle = load_dataset(r"C:\Peter\testdata\S1\house0305")
plan = plan_caches(bundle, profile)          # profile 对象或 dict 都可以
print("\n".join(plan.build(dry_run=True)))
```
