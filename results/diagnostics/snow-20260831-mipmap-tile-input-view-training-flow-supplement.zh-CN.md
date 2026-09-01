# MipMap 逐 Tile 图片、视角与 3DGS 训练数据流补充

日期：2026-08-31
样本任务：`D:\mipmap-lite\1fad9647-f717-4cd4-9391-11f219d7e5d1\snow\snow-20260827`
性质：对既有静态审计、分块回放和 Snow 真实产物的补充说明；不运行厂商程序，不改动原任务数据。

## 1. 先给结论

MipMap 的“切片”和“逐切片训练”不是同一个文件完成的：

```text
原始鱼眼照片 + 相机标定/位姿 + LAS
              │
              ▼
AT/MVS：原始 342 张照片 -> 1368 个去畸变虚拟视图
              │
              ▼
divide_engine.exe
  自适应空间分块 + Tile ROI/halo
  根据点-照片 observation 图选择每块视图
  为每个视图生成投影矩阵和 image_rect 裁剪窗
              │
              ├─ block_reconstruction_Tile_i.json
              ├─ block_mvs/Tile_i.pb.bin
              └─ point_cloud/Tile_i_point_cloud.pb.bin
              │
              ▼
reconstruct_full_engine.exe -> mipmap_engine.dll
              │
              ▼
gaussian_splat.dll
  逐 Tile 建模 -> 逐视图随机置换训练 -> densify/cull
  -> Cut -> level-0/LOD -> 释放该 Tile -> 下一 Tile
              │
              ▼
MergeGSData + 背景/天空 -> PLY / SOG
```

所以：

- `divide_engine.exe` 里有切块、视图关联、裁剪和内存预算逻辑；
- `block_reconstruction_Tile_i.json` 只是块级任务配置，不包含实际训练视图清单；
- `Tile_i.pb.bin` 才是每块训练所需的相机、图片路径、姿态、裁剪窗、SfM 点和观测关系；
- `Tile_i_point_cloud.pb.bin` 是每块进入 Gaussian 初始化的 LiDAR 派生点云；
- 真正的 3DGS 初始化、渲染、loss、Adam、split/clone/cull 和输出逻辑在 `gaussian_splat.dll`；
- 四块是同一进程内严格串行训练，不是四块同时送入 GPU。

## 2. 逻辑具体在哪里

当前安装包没有厂商源码，以下“位置”是磁盘二进制、导出函数、反汇编 RVA 和实际任务产物的位置。

| 位置 | 已确认职责 | 证据级别 |
|---|---|---|
| `D:\Program Files\MipMap\MipMapLite\resources\resources\catch3d\divide_engine.exe` | 自适应分块、Tile ROI/halo、照片负载、视图选择、重投影与 crop、分块内存估值 | 静态调用链 + Snow 数值回放 |
| `...\reconstruct_full_engine.exe` | 读取块任务 JSON，调度 `ReconstructBlockPartion/ReconstructBlockTask/PostProcess3D` | 导入表直接证据 |
| `...\mipmap_engine.dll` | engine 编排、MVS/点云/GS 桥接 | 导出/导入和调用链证据 |
| `...\gaussian_splat.dll` | `GSTraining::Run/BatchTraning`、`InitialParameters`、renderer、loss、生命周期、Cut/Merge/LOD | 导出符号 + 反汇编直接证据 |
| `result\task\block_reconstruction_Tile_i.json` | 原始 LAS、两台鱼眼相机标定、局部坐标 offset、该 Tile ROI、质量和输出选项 | 实际任务文件 |
| `result\milestones\block_mvs\Tile_i.pb.bin` | 该 Tile 的实际训练 view、虚拟相机、3×4 W2C、crop、SfM 点/observation、bbox | protobuf 实际解码 |
| `result\milestones\point_cloud\Tile_i_point_cloud.pb.bin` | 该 Tile 的 LiDAR 派生初始化点云 | 实际产物 + 点数审计 |
| `result\milestones\classify\*.tif` | 语义/动态物体 renderer mask | 实际产物 + GS 读取调用点 |
| `.tools\decode_mipmap_mvs.py` | 只读解码 MVSBlock protobuf | 本项目分析工具，不是厂商逻辑 |
| `.tools\audit_mipmap_tile_view_selection.py` | 用 ROI 点的 observation 闭包复核每块 view 集 | 本项目验证工具，不是厂商逻辑 |
| `.tools\audit_mipmap_undistort_parent_views.py` | 复核 1368 虚拟 view 与 342 父照片的 4:1 映射 | 本项目验证工具，不是厂商逻辑 |

当前磁盘文件名是 `gaussian_splat.dll`，SHA-256 为 `A910B39DBAD956DC35E9E436ACFD0FB8D92364E03BB44B0401CBEF6BCB8D492E`。既有报告中的 `mipmap_gaussian_splat.dll` 是该模块的内部/历史命名；哈希相同，但不应继续当作当前安装目录的真实文件名。

## 3. 每个 Tile 的输入图片从哪里拿

### 3.1 原图先展开为虚拟视图

Snow 有 342 张父照片、两台 2912×2912 鱼眼相机。AT 阶段把每张父照片展开为 4 个方向不同、相机中心相同的去畸变虚拟 view：

```text
342 parent photos × 4 = 1368 virtual undistorted views
```

虚拟 view 的训练图片路径写在 `MVSImage.path`，形如：

```text
<task>\result\.temp\undistort\1.jpg
```

本任务 `keep_undistort_images=false`，任务结束后这些临时 JPG 已被清理；但是其路径、相机和裁剪信息仍保留在 `Tile_i.pb.bin`。这意味着现在可以证明“训练器被告知从哪里读”，但不能再对已删除 JPG 做逐字节复验。

### 3.2 不是把全部 1368 个 view 复制给每个 Tile

分块器从 Tile 的 core/halo 范围取得空间点，再沿 `point -> observation -> image` 图闭包收集能看到这些点的 view。相机中心不需要落在该 Tile 内；相邻 Tile 可以共享同一张照片或同一个虚拟 view。

Snow 的实际块 MVS：

| Tile | 虚拟相机 | 训练 view | 唯一图片路径 | SfM 点记录 | observation | metadata 总表 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 656 | 656 | 17,417 | 127,779 | 1,368 |
| 1 | 8 | 644 | 644 | 11,905 | 91,601 | 1,368 |
| 2 | 8 | 607 | 607 | 12,390 | 80,143 | 1,368 |
| 3 | 8 | 595 | 595 | 10,041 | 67,689 | 1,368 |

用导出 ROI 内点的 observation 图闭包预测 view 集，与真实 `Tile_i.pb.bin` 的 Jaccard 分别为：

```text
Tile_0 0.9924
Tile_1 0.9787
Tile_2 0.9918
Tile_3 0.9691
```

而且真实 Tile view 全部被预测集合覆盖。因此当前最可靠的 view 选择模型是“空间点可见性/观测图闭包”，不是“相机中心在哪块”或“每块复制全部照片”。

## 4. 每张图片的位置、视角和裁剪信息怎么表示

`Tile_i.pb.bin` 的关键 protobuf 关系是：

```text
MVSBlock
  camera[]            虚拟 pinhole/fisheye 相机内参
  image[]             该 Tile 实际训练 view
  point[]             SfM 空间点
  observation[]       图像像素 <-> 空间点关联
  image_meta_data[]   父图来源、原相机、时间戳、原始 C2W/位置
  bounding_box        带 halo 的块范围
  tight_bounding_box  实际选中点的紧包围盒
```

每个 `MVSImage` 至少提供：

| 字段 | 训练含义 |
|---|---|
| `img_id` | view ID，也是缓存、mask/depth 关联键 |
| `path` | 去畸变虚拟 JPG 路径 |
| `camera_id` | 关联虚拟相机内参 |
| `projection_matrix[12]` | 3×4 world-to-camera 变换；相机中心可由 `C=-R^{-1}t` 得到 |
| `image_rect` | 在虚拟图片上实际保留/训练的 `x,y,width,height` crop |
| `color_params` | view 级颜色参数初值/记录 |

`image_meta_data` 另外保存父照片的原始 position、`rotation_c2w`、原相机 ID、原 2912×2912 尺寸、鱼眼标定参数和 timestamp。审计证明 1368 个虚拟 view 全部可准确映射回 342 个父照片，映射后的相机中心最大误差仅 `7.26e-15 m`。虚拟 view 与父图相机中心相同，但朝向和投影面不同。

Snow 四块实际保存的 crop 范围并不都大于 256：最小宽度为 66–70 px，最小高度为 80–101 px，最大到 2912 px。这不与分块内存规划中的“256×256 最小计费矩形 + 128 px 上下文”矛盾：

- `656/644/607/595` 是保留给训练的上下文 view；
- 真正满足最小计费矩形、进入 `max_memory` 像素预算的只有 `541/545/476/484` 个 view；
- 训练 view 数不能直接替代分块器的内存计费 view 数。

训练阶段通过 `ImageCache` 按 view 路径加载/缓存 RGB，并按 `image_rect` 和该 view 的相机投影构造训练 Camera；`DepthCache`、mesh rasterizer 和 `map<view_id,MonoDepthInfo>` 为同一 view 提供几何监督。

## 5. 每个 Tile 的点云怎么导入训练

`block_reconstruction_Tile_i.json` 中每块都保留同一个原始 LAS 路径，同时带有独立 `roi_for_3d` 和 `task_index`。engine 根据 Tile ROI/halo 生成独立的：

```text
result\milestones\point_cloud\Tile_i_point_cloud.pb.bin
```

Snow 的块级 LiDAR 派生输入点数：

| Tile | GS 初始化点云 | 训练后 level-0 GS |
|---|---:|---:|
| 0 | 2,700,801 | 1,773,436 |
| 1 | 1,520,716 | 1,988,450 |
| 2 | 1,802,271 | 1,506,518 |
| 3 | 1,221,675 | 750,498 |

这排除了“每个 LAS 点固定一对一变成一个 Gaussian”以及“固定比例抽稀”的解释。`GaussianSplatModel::InitialParameters(PointCloud...)` 对该 Tile 点云执行的高置信初始化是：

```text
xyz = (point.xyz - offset) * scale
K=7 邻域，去掉 self 后取 6 个邻点平均距离 d
scale = log([d, d, 0.5d])
K=30 局部协方差/法向估计
rotation = 从 +Z 到 local normal 的最短弧四元数
RGB/255 -> SH0
opacity = logit(0.1)
```

随后为 xyz、scale、rotation、SH DC、SH rest、opacity 建立六组独立 Adam 参数组。

## 6. 每个 Tile 如何逐视图训练

Snow 为 `resolution_level=1`，对应 High/type 2。训练器持有该 Tile 的 `MVSBlock`，从其中的 view/image vector 取得 `V`：

```text
High schedule = [5, 10, 5] 个完整 view epoch
total_steps = (5 + 10 + 5) × V = 20V
```

若 Snow 的 PB view 数一一对应运行时 vector.size，则候选总步数为：

| Tile | V | High 候选步数 |
|---|---:|---:|
| 0 | 656 | 13,120 |
| 1 | 644 | 12,880 |
| 2 | 607 | 12,140 |
| 3 | 595 | 11,900 |

`BatchTraning` 每个 epoch 创建 `0..V-1`，用 `RandomDevice -> MT19937 -> Fisher-Yates` 做一次随机置换，然后每个 view 恰好训练一次。它不是每一步有放回随机抽图。

单步主链可概括为：

```text
view = next(permuted_tile_views)
rgb = ImageCache.load_and_crop(view.path, view.image_rect)
camera = build(view.camera_id, view.projection_matrix, view.image_rect)

rendered = GaussianSplatModel.forward(G, camera, semantic_render_mask)

loss = 0.6 * mean_L1(rendered.rgb, rgb)
     + 0.4 * (1 - SSIM(rendered.rgb, rgb))
     + valid DA2 mono-depth loss
     + scheduled LiDAR/mesh depth loss
     + mesh normal / late self-consistency normal loss
     + opacity / sky regularization

backward
split / clone / cull / opacity reset（满足阶段与阈值时）
Adam step
```

补充输入：

- `classify\*.tif` 通过 `Catalog::GetUndistortSegMapPath(image_id)` 读入 renderer mask；Snow 的组合是 `(seg != 255)`，并因 `remove_moving_object=true` 再与 `(seg != 33)` 相交；
- DA2 depth 不以独立图片持久化，而是以 `map<view_id,MonoDepthInfo>` 驻留内存；它先用当前 view 的 LiDAR/mesh raster depth 做 affine RANSAC 标定，再作为权重 0.5 的深度监督；
- 训练中还存在每相机 `exp(a) * rgb + b` 和 BilateralGrid 颜色校正，帮助共享照片覆盖的相邻 Tile 收敛到更一致的视觉结果。

## 7. 分块是如何“导入、训练、换下一块”的

Snow 的时间戳和显存曲线都支持严格串行：

```text
for Tile_0..Tile_3:
    load Tile_i MVS + Tile_i point cloud
    create Gaussian model + optimizer
    run 20V view steps
    Cut + save level-0 + create LOD
    destroy tile model/optimizer/transient tensors
    return VRAM to about 0.9 GiB baseline
```

实测 CUDA 峰值为 4.53/4.56/4.41/3.44 GiB；前三块 LOD 后均回到约 0.9 GiB。相机模型、AI engine 和部分全局缓存可在同一进程复用，但每块的点云、Gaussian 参数、Adam state 和渲染临时张量会重新创建。

训练结束后，Snow 的 `Cut` 仍按带 0.2% halo 的 block AABB 筛选，`MergeGSData` 只是依次 append 六组 Gaussian 属性，并不做空间去重、颜色重估或接缝融合。最终 PLY 是 Tile_0→1→2→3 的 level-0 数据串接后再追加背景/天空；相邻块一致性主要依赖共享 view、训练内颜色校正和 renderer 行为。

## 8. 已补齐与仍未闭合的逻辑

### 已补齐

1. 逐 Tile 的真实训练 view 不在任务 JSON，而在 `block_mvs\Tile_i.pb.bin`。
2. 图片磁盘路径、虚拟相机内参、3×4 W2C、camera center、crop、父图来源都已能从 PB 解析。
3. view 选择由 Tile 点的 observation 图闭包解释，Snow 四块 Jaccard 已达到 0.969–0.992。
4. 每块独立 LiDAR 派生点云进入 `InitialParameters`，不是直接把整份 LAS 同时交给四个训练器。
5. High 为 `20V`，每个 epoch 对全部 Tile view 做无放回随机置换。
6. RGB、语义 mask、DA2、LiDAR/mesh depth/normal、颜色校正、生命周期和块后处理的数据流已串起来。
7. 当前真实训练 DLL 文件名已更正为 `gaussian_splat.dll`。

### 仍不能宣称完全恢复

1. `divide_engine.exe` 受保护；1368 个虚拟 view 在切线负载阶段如何内部聚合成 342 个父照片组，结构证据很强，但精确分支仍未完全闭合。
2. 子节点候选 envelope 的 `observation support >= 2` 已被 Snow 数值精确验证，但尚未绑定到明确条件跳转。
3. `V=PB image_count` 对 Snow 是高置信映射；`20V` 是直接确定，但还缺运行时内存 trace 来逐字节确认该 vector.size。
4. 临时 undistort JPG 已因配置清理，不能对训练时实际读到的 RGB 内容和 cache eviction 次序做事后逐帧复验。
5. SegFormer 的 label 33 真实类别、阈值和加密模型精确网络仍未知。
6. 每次 split/clone/cull 的逐事件计数没有 checkpoint，无法从最终 PLY 反推出每粒 Gaussian 的完整身份轨迹。

## 9. 复核入口

只读复核四块 MVS 摘要：

```powershell
python .tools\decode_mipmap_mvs.py `
  'D:\Program Files\MipMap\MipMapLite\resources\resources\catch3d\divide_engine.exe' `
  '<task>\result\milestones\block_mvs\Tile_0.pb.bin' `
  '<task>\result\milestones\block_mvs\Tile_1.pb.bin' `
  '<task>\result\milestones\block_mvs\Tile_2.pb.bin' `
  '<task>\result\milestones\block_mvs\Tile_3.pb.bin'
```

进一步算法细节见：

- `snow-20260827-mipmap-gaussian-initialization-training-static-audit.zh-CN.md`
- `snow-20260827-mipmap-tiling-vram-reproduction-spec.zh-CN.md`
- `snow2-20260827-mipmap-live-pipeline-research.zh-CN.md`
