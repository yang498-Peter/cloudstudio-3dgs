# snow2-20260827 MipMap 实时处理流程与产物研究记录

## 1. 记录范围

- 任务目录：`D:\mipmap-lite\1fad9647-f717-4cd4-9391-11f219d7e5d1\snow2\snow2-20260827`
- 任务名称：`snow2-20260827`
- 任务 ID：`d2c51abf-6d67-4263-ab4a-fcbf8bd35744`
- 观察时间：2026-08-27 04:45:37 至 04:51:44（UTC+08:00）
- 检查方式：只读检查任务 JSON、结果目录、日志、进程和 GPU 状态；未停止任务，未修改 MipMap 任务目录中的任何文件。
- 截图证据：用户提供的界面截图显示任务最初处于 `Performing AT`，总计 2 个相机、342 张照片、342 张已定位照片，界面进度约 0.98%。截图仅作为状态证据，不作为操作指令。
- 当前结论边界：截至观察截止时间，AT 已成功完成，3D 重建阶段仍在运行；尚不能判定最终 Gaussian Splatting 产品成功生成。

## 2. 结论摘要

1. 修正路径后的 MPL 已被软件实际识别。本次任务成功读取 342 张照片和 1 个 LAS 输入，没有复现旧任务的 `Photo reading error`。
2. 软件首先运行独立的 AT（Aerial Triangulation，空中三角测量/影像联合定向）阶段，然后另起一次 `reconstruct_full_engine.exe --reconstruct_type 2` 进入 3D 重建阶段。
3. AT 已于 04:49:26 完成：342/342 张图像注册成功，最终重投影误差为 1.252814 px；POS RMS 为 X=0.018541 m、Y=0.015305 m、Z=0.006749 m。
4. AT 前处理确实执行了人物/移动目标分类：342 张输入图对应生成 342 个 `milestones\classify\*.tif` 掩膜文件，这与任务参数 `remove_moving_object=true` 一致。
5. 截至 04:51:44，任务状态为 `processing_3d`，3D 阶段已创建影像去畸变子任务并开始加载大量内存，但还没有落盘最终 `.ply` 或 SOG tiles。
6. 当前没有直接错误证据。3D 阶段是否最终成功、模型质量如何、点云和照片是否产生局部重影，仍需等待最终状态与产品文件验证。

## 3. 输入参数与运行环境

### 3.1 软件与硬件

| 项目 | 观察值 | 证据 |
| --- | --- | --- |
| MipMap Lite | 1.0.0 | `info.json` |
| SDK | 5.1.0.8 | `info.json`、`result\logs\log.txt` |
| CPU | Intel Core Ultra 9 275HX | SDK 日志 |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU | SDK 日志、`nvidia-smi` |
| GPU 显存 | 8150.5625 MB（日志值） | SDK 日志 |
| NVIDIA 驱动 | 610.62 | SDK 日志 |
| CUDA Runtime | 12.8 | SDK 日志中的 `12080` |
| 计算模式 | Standalone | `info.json` |

### 3.2 输入与输出选择

| 项目 | 观察值 |
| --- | --- |
| 任务类型 | LiDAR |
| 相机数 | 2 |
| 照片数 | 342 |
| 照片定位数 | 342 |
| LAS 输入 | `G:/S1/USA/2026-02-24_16-21-11snow/process/2026-02-24_16-21-11snow_2/2026-02-24_16-21-11snow_colorized.las` |
| 坐标语义 | `MVP S1 Local` |
| `resolution_level` | 1 |
| 人物/移动目标移除 | 开启，`remove_moving_object=true` |
| 3D 输出 | `gs_ply`、`gs_sog_tiles` |
| 普通 LAS/点云输出 | 未启用 |
| 纹理网格/OBJ/OSGB/GLB | 未启用 |

说明：界面将质量显示为 `Ultra High`，SDK 根日志将 `resolution_level=1` 描述为 `high (1)`。这是界面与 SDK 的命名差异，当前没有证据说明它导致了处理错误。

## 4. 实际时间线

| 时间 | 直接观察到的事件 | 关键文件/状态 |
| --- | --- | --- |
| 04:45:37 | 创建任务 | `info.json`、`photos.json`、`task.json` |
| 04:45:38 | 启动 AT 引擎 | 第一阶段进程参数为 `--reconstruct_type 1` |
| 04:45:48 | SDK 记录 `Start AT...` | `result\log.txt`、`result\logs\log.txt` |
| 04:46:00 | 完成 SfM 分块任务生成 | `.temp\sfm_block_partion.done`、`task\block_sfm_task_0.json` |
| 04:46:03～04:46:27 | 逐图生成分类掩膜 | 342 个 `milestones\classify\*.tif` |
| 04:46:41 | 生成图像匹配候选/结果列表 | `.temp\match\match_list_0.pb.bin` |
| 04:47:50 | 生成带相机与照片姿态的块交换 XML | `.temp\pre.xml` |
| 04:47:53～04:48:51 | 持续写入三角化统计 | `report\report.json` 由 236 B 增长至 61,305 B，随后继续增长 |
| 04:49:21 | SfM 块完成 | `.temp\sfm_block_0.done`、`report\sfm_block_0_report.json` |
| 04:49:25 | 生成正式 AT/MVS XML 与质量缩略图 | `AT\mvs.xml`、`thumbnail\*.png` |
| 04:49:26 | AT 完成 | `.temp\at.done`，日志进度 100% |
| 04:49:27 | 任务切换为 `processing_3d` | `info.json` 的 `currentRunType=2` |
| 04:49:27 | 启动第二阶段引擎 | 新进程参数为 `--reconstruct_type 2` |
| 04:49:45 | SDK 记录 `Start 3D Reconstruction...` | `result\log.txt`、SDK 日志 |
| 04:49:51～04:49:52 | 建立 3D 输出目录、ROI 与去畸变任务 | `3D\model-gs-ply\metadata.xml`、`milestones\mvs_roi.pb.bin`、`task\image_undistortion_task_0.json` |
| 04:51:44 | 最后一次快照仍为 3D 处理中 | `status=processing_3d`；最终模型文件尚未出现 |

## 5. 已生成文件及其作用

以下“作用”分为直接证据和合理推断。文件是否存在、大小和时间为直接证据；私有 `.pb.bin` 的内部字段没有解码，其用途主要依据目录名、任务类型和生成顺序推断。

### 5.1 任务定义与状态文件

| 文件 | 大小 | 直接证据/用途 |
| --- | ---: | --- |
| `info.json` | 2,642 B（初始观察值） | 任务状态、输入参数、软件/SDK 版本、开始时间和阶段状态 |
| `task.json` | 359,423 B | 传给引擎的完整任务；含 342 张图、2 个相机、LAS、输出开关 |
| `last_task.json` | 359,423 B | 当前任务参数的保存副本 |
| `photos.json` | 299,947 B | 照片清单与照片元数据 |
| `layers.json` | 417 B | 项目图层定义 |
| `result\at_task.json` | 359,459 B | AT 阶段使用/更新后的任务定义 |
| `result\r3d_task.json` | 359,459 B | 3D 阶段使用/更新后的任务定义 |

### 5.2 AT/SfM 中间文件

| 文件或目录 | 观察值 | 判断 |
| --- | ---: | --- |
| `result\milestones\classify\*.tif` | 342 个，共 4,645,728 B；单个约 13.6 KB | 与 342 张输入图一一对应的分类/人物移除掩膜 |
| `result\task\block_sfm_task_0.json` | 219,814 B | 单个 SfM 分块任务；`task_type=block_sfm` |
| `result\.temp\match\match_list_0.pb.bin` | 56,869 B | 图像对匹配列表或匹配中间结果 |
| `result\.temp\pre.xml` | 321,959 B | `BlocksExchange` XML；包含 FishEye 相机、342 张照片路径、组件和姿态 |
| `result\report\report.json` | 观察中从 236 B 增长到 92,637 B | 硬件信息及逐图三角化统计；运行时会被持续更新 |
| `result\milestones\mvs_block_0.pb.bin` | 5,131,252 B | SfM 块结果，后续 MVS/3D 输入 |
| `result\AT\mvs.xml` | 16,676,134 B | AT 完成后正式输出的相机/照片/稀疏重建交换文件 |
| `result\milestones\mvs_raw.pb.bin` | 4,426,813 B | AT 原始 MVS 里程碑数据 |
| `result\milestones\mvs.pb.bin` | 4,426,813 B | AT 整理后的 MVS 里程碑数据 |
| `result\milestones\cs.json` | 77 B | 坐标系统里程碑信息 |
| `result\.temp\at.done` | 0 B | AT 完成标记 |

### 5.3 质量预览文件

| 文件 | 大小 | 作用 |
| --- | ---: | --- |
| `result\thumbnail\overlap_map.png` | 31,614 B | 影像重叠关系预览 |
| `result\thumbnail\camera_1_residual.png` | 136,922 B | 相机 1 残差可视化 |
| `result\thumbnail\camera_2_residual.png` | 127,569 B | 相机 2 残差可视化 |
| `result\thumbnail\rgb_thumbnail.png` | 90,608 B | RGB 总览缩略图 |

### 5.4 3D 阶段已创建但尚未形成最终模型的文件

| 文件或目录 | 观察值 | 判断 |
| --- | ---: | --- |
| `result\milestones\roi.json` | 1,194 B | 3D 重建使用的范围定义 |
| `result\milestones\mvs_roi.pb.bin` | 4,426,813 B | 裁到 ROI 后的 MVS/相机输入 |
| `result\task\image_undistortion_task_0.json` | 219,868 B | 去畸变子任务；`task_type=image_undistortion`，`head_task=true`，支持恢复 |
| `result\.temp\undistortion_block_partion.done` | 0 B | 去畸变任务分块完成标记，不等于全部图像已经去畸变完成 |
| `result\3D\model-gs-ply\metadata.xml` | 153 B | Gaussian PLY 输出目录的元数据占位/初始化文件 |
| `result\3D\model-gs-sog-tile\` | 空目录 | 计划生成 SOG tiles，尚未有 tile 文件 |
| `result\milestones\splats\` | 空目录 | Gaussian splat 里程碑尚未落盘 |
| `result\.temp\undistort\` | 观察时为空 | 去畸变图可能尚在内存准备、尚未落盘，或稍后写入/临时生成 |
| `result\.temp\depth\` | 观察时为空 | 深度相关中间文件尚未生成 |

截至 04:51:41，`result` 目录共有 368 个文件，总计 41,791,077 B。这里的多数文件是 AT 中间产物和 342 个掩膜，不是最终 3D 产品。

## 6. AT 质量与成功判据

SDK 在 04:49:26 明确写出：

- 总照片：342
- 已注册照片：342
- 注册率：100%
- 最终重投影误差 `pixel_RMSE`：1.252814 px
- `POS_RMS_X`：0.018541 m
- `POS_RMS_Y`：0.015305 m
- `POS_RMS_Z`：0.006749 m
- AT 耗时：0.055833 h，约 201 秒
- 引擎进度：100%
- 明确状态：`AT Finished`

这些事实说明：

1. 修正后的照片路径可以被 SDK 读取。
2. 相机内参、外参和照片位姿格式至少满足 SDK 的 AT 输入要求。
3. 342 张照片都进入同一次成功的联合定向，没有在输入或注册阶段被丢弃。
4. AT 成功不能单独证明最终 Gaussian 模型质量，但已经排除了此前的照片路径读取故障。

## 7. 进程与资源行为

### 7.1 AT 阶段

- 引擎：`reconstruct_full_engine.exe`
- 第一阶段 PID：8360（观察时值，进程结束后不可复用为稳定标识）
- 参数：`--task_json <task.json> --reconstruct_type 1`；命令行中的桌面鉴权值未写入本文。
- 04:47:25 的瞬时 GPU 采样：GPU 利用率约 61%，显存占用约 288 MB。
- 04:48 左右的束平差/优化段主要消耗 CPU，GPU 瞬时利用率降至 0%；引擎 CPU 时间持续快速增加。
- 内部 AT 进度先快速增长到约 49%，随后较慢进行全局优化，最后跳至 100%。这与“先逐图特征/匹配，再做全局三角化和束平差”的行为一致。

### 7.2 3D 阶段

- 第二阶段 PID：52040（观察时值）
- 启动时间：04:49:27
- 参数：`--task_json <task.json> --reconstruct_type 2`
- 04:49:51 日志进度：0.5%
- 04:50:19～04:50:35：进程短暂处于低常驻内存/低活动状态，目录没有新增最终产品。
- 04:50:40 以后：工作集由约 645 MB 增长到约 1.3 GB；随后继续增长。
- 04:51:41：工作集 3,643,420,672 B（约 3.39 GiB），私有内存 5,884,289,024 B（约 5.48 GiB），说明正在加载/准备较大的 3D 数据结构。
- 04:51:44 的瞬时 GPU 采样：显存占用约 1,236 MB，GPU 利用率瞬时为 0%。单次 0% 采样不等于任务停滞；此时进程内存仍明显增长，且阶段刚完成数据准备。

## 8. 推测的完整处理流程

以下流程中，1～7 已有文件或日志直接支持；8～10 是依据任务输出开关和目标目录作出的后续流程推断，尚待任务实际产物确认。

1. **读取项目描述**：解析 `task.json` 中的 2 个鱼眼相机、342 张照片、照片位姿和 1 个 LAS。
2. **人物/移动目标分类**：为 342 张图生成 342 个分类掩膜，供后续匹配或训练时屏蔽动态区域。
3. **影像分块与候选配对**：生成 `block_sfm_task_0.json` 和 `match_list_0.pb.bin`。
4. **特征提取与图像匹配**：建立跨照片特征轨迹；运行时报告中出现逐图 `feature_count`、`track_count` 和重投影残差。
5. **三角化/束平差（AT）**：联合优化相机姿态与稀疏结构，并利用输入 POS 作为先验或约束。
6. **AT 质量评估与封装**：生成 `mvs.xml`、MVS 二进制里程碑、残差图、重叠图和 `at.done`。
7. **切换到 3D 重建**：重启同一引擎为 `reconstruct_type 2`，建立 ROI 和 `image_undistortion` 子任务。
8. **影像去畸变与训练数据准备（推断）**：将鱼眼相机图像变换为 Gaussian/MVS 阶段可使用的图像，并套用人物掩膜。
9. **LiDAR/相机数据联合初始化与 Gaussian 优化（推断）**：使用 LAS 作为几何/尺度/坐标基础，结合已优化相机与图像生成并优化 Gaussian splats。具体损失函数、迭代次数和 LiDAR 约束方式无法从当前明文日志确定。
10. **产品导出（推断）**：写出 `result\3D\model-gs-ply` 下的 Gaussian PLY，并在 `model-gs-sog-tile` 下生成 SOG 瓦片；随后更新任务为完成状态并生成最终报告/缩略图。

## 9. 当前尚未完成的验证

- `info.json` 尚未进入 completed/success 类终态。
- 尚未看到最终 Gaussian `.ply` 文件。
- 尚未看到 SOG tile 数据文件。
- 尚未验证最终模型能否在 MipMap 产品查看器中正常打开。
- 尚未验证雪地弱纹理、反光、动态人物移除和闭环区域是否有重影、孔洞或漂浮 Gaussian。
- 尚未核对最终模型坐标、尺度与输入 LAS 的几何一致性。
- 尚未统计最终 splat 数、PLY 大小、tile 数量、总耗时和峰值显存。

因此当前状态应表述为：**AT 已通过，3D 重建运行中；最终产品待验证**。

## 10. 关键证据路径

- 任务状态：`D:\mipmap-lite\1fad9647-f717-4cd4-9391-11f219d7e5d1\snow2\snow2-20260827\info.json`
- 引擎任务：`D:\mipmap-lite\1fad9647-f717-4cd4-9391-11f219d7e5d1\snow2\snow2-20260827\task.json`
- 明文摘要日志：`D:\mipmap-lite\1fad9647-f717-4cd4-9391-11f219d7e5d1\snow2\snow2-20260827\result\log.txt`
- SDK 详细日志：`D:\mipmap-lite\1fad9647-f717-4cd4-9391-11f219d7e5d1\snow2\snow2-20260827\result\logs\log.txt`
- AT 正式成果：`D:\mipmap-lite\1fad9647-f717-4cd4-9391-11f219d7e5d1\snow2\snow2-20260827\result\AT\mvs.xml`
- AT 统计：`D:\mipmap-lite\1fad9647-f717-4cd4-9391-11f219d7e5d1\snow2\snow2-20260827\result\report\report.json`
- 影像去畸变任务：`D:\mipmap-lite\1fad9647-f717-4cd4-9391-11f219d7e5d1\snow2\snow2-20260827\result\task\image_undistortion_task_0.json`
- 3D 输出根：`D:\mipmap-lite\1fad9647-f717-4cd4-9391-11f219d7e5d1\snow2\snow2-20260827\result\3D`
- 本次使用的修正 MPL：`G:\S1\USA\2026-02-24_16-21-11snow\process\2026-02-24_16-21-11snow_2\2026-02-24_16-21-11snow_fixed.mpl`

## 11. 数据保护说明

- 本次只读检查没有修改客户照片、LAS、MPL、任务 JSON、日志或任何运行中产物。
- 没有停止或重启 MipMap、没有改变任务参数、没有清理临时目录。
- 研究记录保存在工作区诊断目录，与客户数据和 MipMap 运行目录隔离。

## 12. 持续监控快照

### 2026-08-27 12:10:57（UTC+08:00）

直接证据：

- 系统重启后一度不可见的 `G:` 已重新挂载。
- 研究文件、任务所引用的照片目录和 `2026-02-24_16-21-11snow_colorized.las` 均重新可读。
- `info.json` 仍是旧运行的 `status=error`，更新时间仍为 04:53:17，错误码仍为 `1073807364`。
- 没有发现 12:05 后启动的 `reconstruct_full_engine.exe`。
- `result` 中没有 12:05 后新增或修改的文件；最新文件仍是 04:49:52 的旧运行产物。
- GPU 瞬时利用率为 0%，显存占用约 154 MB，没有训练负载。

当前判断：输入盘阻塞已经解除，但用户所说的“重新开始”尚未落实为新的计算进程。持续监控继续等待新的引擎实例，不能把旧运行的错误终态当成新运行结果。

### 2026-08-27 12:11:21～12:12:31（UTC+08:00）：新任务重新开始

用户补充背景：系统重启后曾忘记插入 G 盘，点击旧任务“继续”失败。该信息属于用户提供的操作背景；磁盘与任务状态证据显示，当时 G 盘确实不可读，旧任务没有恢复计算。

新一轮采用新的任务目录，而不是续写旧 `snow2` 目录：

- 新任务：`D:\mipmap-lite\1fad9647-f717-4cd4-9391-11f219d7e5d1\snow\snow-20260827`
- 新任务名称：`snow-20260827`
- 新项目 ID：`6e70fa07-a897-40da-9e26-d94a614ff809`
- 新任务 ID：`6032cb3c-d0cd-470c-a057-8ace47fe4e10`
- 创建/启动时间：2026-08-27 12:11:21
- 状态：`processing_at`
- `currentRunType=1`
- 新引擎 PID：46684
- 引擎参数：`--reconstruct_type 1`

输入可读性复核：

- `G:` 已挂载。
- 左右相机照片目录均可读。
- `2026-02-24_16-21-11snow_colorized.las` 可读。
- 新任务仍引用修正后的 `G:/S1/USA/...` 路径。

截至 12:12:31 的直接运行证据：

- SDK 已记录 `Start AT...`，输入 342 张图、2 个相机。
- AT 内部进度于 12:12:28 到达 23.054001%。
- 新 `result` 已有 350 个文件，共 5,646,764 B。
- 342 个 `milestones\classify\*.tif` 已全部重新生成。
- `.temp\match\match_list_0.pb.bin` 已于 12:12:22 生成，大小 56,869 B。
- 引擎 CPU 时间约 349.28 秒，工作集约 561 MB，私有内存约 948 MB。
- 12:12:31 的瞬时 GPU 利用率为 51%，显存占用约 397 MB，功耗约 39.38 W。

当前判断：新任务已经从头执行 AT，运行活动正常，尚未进入 3D/Gaussian 阶段。后续监控以 `snow\snow-20260827` 为唯一当前任务路径，旧 `snow2` 仅作为中断运行的历史对照。

### 2026-08-27 12:14:43（UTC+08:00）：质量档位核对与 AT 完成

用户说明本轮希望切换到 `High`，用于观察质量档位的影响。磁盘参数和安装包代码核对得到以下直接证据：

- 旧任务 `snow2-20260827` 的 `info.json` 和 `task.json` 均为 `resolution_level=1`。
- 新任务 `snow-20260827` 的 `info.json` 和 `task.json` 也均为 `resolution_level=1`。
- 安装版 `app.asar` 的 `getQualityAliasByLevel` 映射为：值 1 显示 `Ultra High`，值 3 显示 `Medium`，其余默认值 2 显示 `High`。
- MipMap SDK 根日志把值 1 输出成 `resolution level: high (1)`，但界面映射明确把值 1 显示为 `Ultra High`。该日志文字不能用来证明选择了 High。

结论：用户希望切换到 High，但当前任务实际持久化的仍是 `Ultra High / resolution_level=1`，与旧任务没有形成档位差异。当前运行结果不能用于比较 High 相对 Ultra High 的速度、显存或最终质量影响。具体是选择操作发生在任务创建之后、界面没有保存，还是 LiDAR 参数联动覆盖了选项，现有证据尚不能唯一确定。

新任务 AT 于 12:14:26 完成：

| 指标 | 旧任务（同为 level 1） | 新任务（同为 level 1） | 解释 |
| --- | ---: | ---: | --- |
| 注册图像 | 342/342 | 342/342 | 相同 |
| 重投影 RMSE | 1.252814 px | 1.251524 px | 差异约 0.00129 px，基本相同 |
| POS RMS X | 0.018541 m | 0.018780 m | 基本相同 |
| POS RMS Y | 0.015305 m | 0.015438 m | 基本相同 |
| POS RMS Z | 0.006749 m | 0.006826 m | 基本相同 |
| AT 时间 | 0.055833 h（约 201 s） | 0.044167 h（约 159 s） | 新任务约快 21%，但参数相同，可能受缓存和系统负载影响，不能归因于质量档位 |

12:14:27 后状态已切换为 `processing_3d`，新 3D 引擎 PID 为 7204，参数为 `--reconstruct_type 2`。12:14:43 日志记录 3D 阶段进度 0.5%。持续监控继续跟踪实际 PLY、SOG tiles、耗时和资源峰值。

### 2026-08-27 12:18:19（UTC+08:00）：当前 3D 子阶段分析

AT 已正式结束，直接证据包括：

- 12:14:26 日志明确记录 `Aerial Triangulation Finished`、进度 100% 和 `AT Finished`。
- 342/342 张图像注册成功，重投影 RMSE 为 1.251524 px。
- `.temp\at.done` 已生成。
- `AT\mvs.xml` 已生成，大小 16,589,796 B。
- `mvs_raw.pb.bin`、`mvs.pb.bin`、残差图、重叠图和 RGB 缩略图均已生成。

当前已进入 3D 重建：

- `info.json` 为 `status=processing_3d`、`currentRunType=2`，无错误字段。
- 第二阶段 `reconstruct_full_engine.exe` PID 7204 于 12:14:27 启动，参数为 `--reconstruct_type 2`。
- 12:14:35 日志明确记录 `Start 3D Reconstruction...`。
- 已生成 `milestones\mvs_roi.pb.bin`、`task\image_undistortion_task_0.json`、`.temp\undistortion_block_partion.done` 和 `3D\model-gs-ply\metadata.xml`。

12:18:19 的资源状态：

- 进程累计 CPU 时间约 451.5 秒。
- 工作集 4,740,980,736 B（约 4.42 GiB）。
- 私有内存 5,791,338,496 B（约 5.39 GiB）。
- GPU 瞬时利用率 35%，显存占用约 936 MB，功耗约 38.84 W。

尚未观察到：

- `.temp\undistort` 中的落盘图像；
- `.temp\depth` 深度文件；
- `milestones\point_cloud` 点云里程碑；
- `milestones\splats` Gaussian 里程碑；
- 最终 Gaussian PLY；
- SOG tile 文件。

阶段判断：当前不是 AT，也还不能证明已经进入 Gaussian 参数的正式迭代优化。最符合现有证据的解释是，3D 引擎正在读取 342 张鱼眼图像、人物掩膜、优化相机和 LAS，进行 ROI/MVS 整理、影像去畸变及 Gaussian/MVS 初始化。进程资源持续活动，当前没有停滞或报错证据。下一关键节点应是去畸变/深度/初始点云或 splat 里程碑开始落盘。

### 2026-08-27 12:23:24（UTC+08:00）：确认当前正在执行去畸变与语义分割

此前的“3D/Gaussian 输入准备”现在可以进一步细化。当前直接执行的子任务是：将 342 张 2912×2912、`projection_model=1` 的源相机图像转换为去畸变图，并为每张派生图生成低分辨率语义分类图。

直接证据：

- `result\task\image_undistortion_task_0.json` 明确写有 `task_type=image_undistortion`、`image_meta_data` 342 项、`img_id` 342 项、`keep_undistort_images=false`、`remove_moving_object=true`、`resolution_level=1`。
- 12:21:34 起，`.temp\undistort` 开始连续生成编号 JPG；同一时刻 `.temp\undistort_classify` 生成同编号 TIF。
- 12:23:24 时，两目录各有 882 个文件；最新配对为 `882.jpg` 与 `882.tif`，仍在持续增长。去畸变 JPG 合计约 771,027,142 B，分类 TIF 合计约 8,305,766 B。
- 抽查 `1.jpg`：尺寸为 1456×2912，画面是由原始鱼眼/广角相机数据展开后的直线投影图像。
- 抽查 `1.tif`：尺寸为 364×728，像素格式 Gray8；它是 `1.jpg` 长宽各 1/4 的离散标签图。全图只有 8 个像素值：0、1、2、3、4、6、7、34，证明它不是普通灰度缩略图或单一黑白掩膜，而是多类别语义分割标签图。
- 活跃进程加载了 `mipmap_classify.dll`、TensorRT `nvinfer_10.dll` 和 `nvonnxparser_10.dll`。
- `mipmap_classify.dll` 的可读符号包括 `SegFormerSeg`、`TensorRTSeg`、`TensorRTMonoDepth`、`Infer`、`InferColor`、`InferRaw`、`SetConfidenceThreshold`，以及“build engine from onnx”等字符串；因此分类 DLL 明确实现了 ONNX/TensorRT 推理、SegFormer 语义分割和单目深度接口。
- 12:22:01 的 3 秒采样中，进程新增 CPU 时间 9.688 秒，GPU 利用率 32%，显存占用约 1,312 MB，说明该批处理仍在实际计算。

算法判断：

- **已确认调用的算法类别**：相机模型驱动的影像去畸变/重投影，以及基于 TensorRT 的 SegFormer 类语义分割。JPG/TIF 的一一配对和离散标签直方图是实际执行证据，不只是安装文件存在。
- **高置信推断**：由于任务打开 `remove_moving_object=true`，语义标签图会在后续重建中用于屏蔽人、车辆等被配置为动态的类别；软件生成的是多类标签，之后再选择需排除类别，而不是直接只输出人物二值蒙版。
- **鱼眼展开方式的推断**：342 张源图已产生超过 342 张去畸变图，说明每张源图可能被切分/重投影为多个视向。若固定为 3 个视向，目标数可能是 1,026，但当前任务文件没有明文给出每张源图的派生视图数，因此该数量关系暂不作为已确认事实。
- **尚不能确认**：虽然分类 DLL 也暴露 `TensorRTMonoDepth`，但 `.temp\depth` 当前为 0 文件，不能据此声称单目深度网络已经运行。
- **明确尚未开始**：进程尚未加载 `mipmap_gaussian_splat.dll`，也未加载扩展目录中的 `torch*.dll`；`milestones\splats` 和 `milestones\point_cloud` 仍为空。因此 12:23:24 时尚未进入 Gaussian Splat 参数优化/训练。

阶段链可概括为：AT 优化相机参数 → 计算重建 ROI/分块 → 当前的鱼眼展开与语义分割 → 后续深度/点云或 LiDAR 初始化 → Gaussian Splat 优化 → 导出 GS PLY 与 SOG Tiles。后半段顺序仍需用后续实际模块加载、日志和产物时间戳验证。

### 2026-08-27 12:26:51（UTC+08:00）：去畸变完成，Gaussian 模块已加载

直接证据：

- G:、左右照片目录和任务引用 LAS 均可读。
- 去畸变与分类输出最终各为 1,368 个文件，恰好是 342 张源图的 4 倍，确认每张源照片被展开为 4 个派生视向；JPG 合计 1,183,587,451 B，语义分类 TIF 合计 12,920,672 B。
- 最后一对 `1368.jpg` / `1368.tif` 于 12:24:25 写入。
- 随后生成 `milestones\mvs_undistort.pb.bin`（5,103,548 B）、`AT\mvs_undistort.xml`（19,328,926 B）、`.temp\undistort_block_0.done` 和 `.temp\undistort.done`；完成标记时间为 12:24:26。
- 同一 PID 7204 仍在运行；12:26:51 累计 CPU 时间约 1,313.2 秒，工作集约 2.68 GiB，私有内存约 3.82 GiB。
- 进程现已加载 `mipmap_gaussian_splat.dll`、`torch_cpu.dll`、`torch_cuda.dll`、`c10_cuda.dll`、CUDA Runtime、cuBLAS 和 cuDNN；这些模块在 12:23:24 的上一快照中尚未加载。
- 12:26:35 GPU 瞬时利用率 25%，显存占用约 674 MB；引擎自身日志在 12:26:32 报告阶段进度 15.20%、G-RAM 使用约 0.90 GB。
- `.temp\depth`、`milestones\point_cloud`、`milestones\splats` 当前仍为 0 文件，最终 GS PLY 仅有早期 153 B 的 `metadata.xml`，SOG tile 尚未生成。

阶段判断：去畸变和语义分割已经正式完成。Gaussian/PyTorch/CUDA 模块的新增加载证明引擎已进入 Gaussian Splat 执行链，当前最可能处于数据装载、LiDAR/相机初始化或训练初始化/早期迭代。由于尚无 splat 里程碑落盘，暂不把“已经完成首轮可恢复 Gaussian checkpoint”视为已确认事实。日志进度从 12:24:50 的 15.50% 回到 12:25:18 的 15.05%，结合模块切换更符合子阶段独立进度重新计数，而非任务倒退。

### 2026-08-27 12:29:00（UTC+08:00）：背景 Gaussian 已生成，场景切为 4 个重建块

直接证据：

- G:、左右照片目录和 LAS 仍可读；任务仍为 `processing_3d`，无错误。
- 12:28:52 生成 `milestones\splats\gaussian_splat_background.pb.bin`，大小 5,600,004 B。这是首个 splat 里程碑。
- 同时生成 `3D\model-gs-ply\sky.ply`（5,600,362 B）和 `3D\model-gs-ply\ue\sky_full.ply`（24,801,532 B）。
- 两个 PLY 均为有效的 `binary_little_endian 1.0` PLY，各声明 100,000 个 vertex。`sky.ply` 包含位置、DC 颜色、透明度、尺度和旋转；`sky_full.ply` 还包含 45 个高阶球谐 `f_rest_*` 属性，符合 Gaussian Splat 属性结构。
- 12:28:53 生成 `task\tiles.json`，明确把场景划分为 4 个非空块 `Tile_0` 至 `Tile_3`；各块估算最大内存约 4.26–6.59 GB。
- 对应生成 4 个 `milestones\block_mvs\Tile_*.pb.bin` 和 4 个 `block_reconstruction_Tile_*.json`，随后写入 `.temp\block_cut.done`。
- 日志在切块后记录 `Images Count: 164`，表明引擎开始装载某个块所覆盖的派生视图；具体是哪个 Tile，明文日志尚未标注。
- 12:29:00 PID 7204 仍在运行，累计 CPU 时间约 1,480.8 秒；当前 GPU 瞬时利用率为 0%、显存约 582 MB，符合刚完成背景模型与切块、正在阶段切换/CPU 装载的瞬时状态。
- `.temp\depth` 和 `milestones\point_cloud` 仍为空，SOG tile 尚未生成。

阶段判断：Gaussian 执行已不再只是“模块已加载”。背景/天空 Gaussian 已实际生成并通过 PLY 头验证，场景主体随后被划为 4 个独立重建块。下一阶段应按 Tile 逐块执行主体 Gaussian 优化，之后再合并/导出最终 PLY 和 SOG Tiles。背景 PLY 是有效中间产物，但不是最终完整场景模型。

### 2026-08-27 12:33:30（UTC+08:00）：分块、块训练与合并机制专项分析

#### 1. 空间如何分块

`tiles.json` 的直接证据表明，本次不是三维八叉树分块，而是在共享完整 Z 范围的前提下对局部 XY 平面做 4 个矩形分区：

- 所有 Tile 的 Z 范围完全相同：`[-14.620715, 22.145558]`。
- `Tile_0`：左下，X `[-54.388954, 0.251152]`，Y `[-53.264386, 3.356187]`，估算最大内存 6.591 GB。
- `Tile_1`：左上，X `[-54.388954, 0.251152]`，Y `[3.152480, 48.792447]`，估算最大内存 6.042 GB。
- `Tile_2`：右下，X `[0.018278, 62.280985]`，Y `[-53.256914, -0.387518]`，估算最大内存 5.082 GB。
- `Tile_3`：右上，X `[0.018278, 62.280985]`，Y `[-0.591225, 48.799919]`，估算最大内存 4.264 GB。
- 四块使用同一个局部坐标偏移 `[2.767021, -4.222885, 1.415041]`，因此块输出无需重新配准即可回到同一坐标框架。
- 左右列 X 方向重叠约 0.232874 m；每列上下块 Y 方向重叠约 0.203706 m。软件故意保留窄空间重叠带，不是无缝硬切。

几何上表现为先沿 X 分成左右两列，再在两列内部用不同的 Y 位置分别切分。左右列的 Y 分割线分别约为 3.25 m 和 -0.49 m，并非一条贯穿全场的统一水平线。结合每块不同的 `max_memory` 估值，当时只能高置信推断这是按点云/照片密度和显存预算进行的递归二维负载均衡。2026-08-28 的后续静态反汇编已把该不确定性关闭：`divide_mode=2` 确实执行按照片投影像素负载平衡的递归平面 KD 切分；详见文末专项复刻结论。

#### 2. 照片如何分配到块

四个块 JSON 只保存 ROI、公共相机、LAS 和输出参数；真正的照片成员关系封装在 `milestones\block_mvs\Tile_*.pb.bin` 中。直接提取其中唯一的去畸变 JPG 编号得到：

- `Tile_0`：656 个派生视图；
- `Tile_1`：644 个派生视图；
- `Tile_2`：607 个派生视图；
- `Tile_3`：595 个派生视图；
- 四块合计覆盖 1,254 个不同派生视图，少于全部 1,368 个派生视图，说明部分视图可能只服务背景、没有主体块有效覆盖，或被可见性/质量条件排除。

块间共享视图数量与比例：

- 左下 `Tile_0` / 左上 `Tile_1`：共享 312，分别占两块 47.56% / 48.45%；
- 左下 `Tile_0` / 右下 `Tile_2`：共享 344，占 52.44% / 56.67%；
- 左上 `Tile_1` / 右上 `Tile_3`：共享 417，占 64.75% / 70.08%；
- 右下 `Tile_2` / 右上 `Tile_3`：共享 282，占 46.46% / 47.39%；
- 对角块也共享约 196–217 个视图，源于鱼眼四向展开、相机轨迹和宽视场覆盖。

这证明照片不是按“相机中心落入哪个 ROI”独占分配，而是允许一个视图进入多个块。结合 Gaussian DLL 的 `checkFrustum` CUDA 符号，高置信推断分配依据包括相机视锥与块 ROI 的可见/相交关系。大比例共享照片使相邻块边界从相同图像观测中学习，是降低独立训练接缝的重要条件。

#### 3. 每块如何训练

运行时证据显示 Standalone 模式采用单进程顺序训练，而非 4 块并行：

- 只有一个 PID 7204；4 个块任务的 `task_index` 为 0–3。
- 目前只生成 `Tile_0_point_cloud.pb.bin`（74,553,610 B）和 `3D\point-pnts\Tile_0` 分层点云；其余 Tile 尚无同类产物。
- 日志在进入首块后记录 `Images Count: 164`。Tile_0 的块 MVS 中包含 656 个派生视图，恰为 164×4，与每张源照片展开 4 个视向一致。
- 12:33:15 总进度 27.770054%，引擎自报 GPU 显存约 2.23 GB；12:33:30 实测 GPU 利用率 67%、显存约 2,024 MB，说明 Tile_0 已进入实际 GPU 优化。

块训练参数保持公共坐标系、相机内参、同一 LAS 和同一语义剔除设置，仅 ROI 与 `task_index` 不同。当前最符合证据的块训练链是：

1. 从全局 LAS/MVS 中按块 ROI 取得块点云及可见照片；
2. 用块点云初始化 Gaussian 的位置/颜色/尺度等参数；
3. 从该块共享照片中采样视图，执行可微 Gaussian 渲染；
4. 用图像重建误差反向传播，并通过 Adam 优化；
5. 依据梯度克隆或分裂 Gaussian，依据透明度和冗余度裁剪无效 Gaussian；
6. 保存每块可恢复的 splat/PLY 里程碑，再转到下一 `task_index`。

DLL 可读符号为这条链提供了算法级证据：

- 初始化：`InitialParameters(PointCloud...)`、`GetSubsampleKDTree`；
- 优化器：`InitialOptimizer`、`OptimizersStep`、`UpdateLearningRate`，并明确使用 PyTorch `Adam`；
- 渲染与损失：`forward`、`GetLoss`、`GetLossWithGradientWeight`、SSIM 日志；
- 几何约束接口：`GetDepthRegularizerLoss`、`GetNormalLoss`、`GetNormalGradientLoss`、`GetScaleLoss`、`GetOpacityLoss`、`GetSingleViewLoss`；哪些权重在本任务实际非零仍需配置/运行日志证明；
- 自适应增密：`CloneGS`、`SplitGS`，以及明文 `GS training clone point by grad`、`GS training split point by grad`；
- 清理与稳定：`CullGS`、`CullGSRedundancy`、`ShrinkBigScaleGS`、`Reset Opacity step`；
- MCMC 路径也存在 `AfterTrainMCMC`、`relocation_kernel` 和 `compute_dead_mask_kernel`，但当前尚无明文证据证明本次任务选择了 MCMC 模式；
- 色彩一致性接口：`ApplyColorHarmonization`，是否在块间实际调用需继续观察。

#### 4. 分块关系和最终合并

已确认的设计基础：

- 所有块共享同一个坐标偏移，因此不存在独立块间的二次位姿求解。
- 相邻块拥有窄空间重叠带，同时共享大量训练照片，使边界处受到共同图像观测约束。
- 天空/背景单独通过 `TrainBackground` 生成一次全局 `gaussian_splat_background.pb.bin` 和 `sky.ply`，不会要求每个主体块重复学习一套无限远背景。
- Gaussian DLL 提供 `GaussianSplatData::Cut(AlignedBox, SceneROI, bool)`、`SaveROI`、`MergeGSData`、`SavePly`、`SaveSplatPB`、`CreateLoD`、`CreateSogLOD`、`CutSogLoD` 和 `GenerateOrdering` 等接口。

据此得到的高置信合并模型是：每块可在带缓冲/重叠观测的范围内训练，但导出时用核心 ROI 对 Gaussian 集合做空间裁剪，移除重叠带中的重复 Gaussian；然后 `MergeGSData` 在公共坐标系中串接/合并各块主体数据，并只加入一次全局背景，最后生成完整 PLY 和空间分层的 SOG LOD。这个策略比直接把四块未经裁剪地叠加更能避免边界双层表面和 Gaussian 数量翻倍。

当前仍未直接确认的是：重叠带是否存在额外的权重羽化、颜色融合或基于质量的二选一。DLL 中存在 `ApplyColorHarmonization`、`MergeGSData` 和 Cut 接口，但没有足够运行时证据证明具体调用顺序。后续监控重点将记录：

- 每块 splat/PLY 的原始数量和 ROI；
- 块完成顺序及是否先裁剪再保存；
- 合并前后 Gaussian 数量是否等于各核心块之和加背景；
- 是否出现 color harmonization、merge、cut/ROI 明文日志或中间文件；
- 最终 PLY 坐标边界、重复点/双层面、接缝附近密度和 SOG LOD 层级。

### 2026-08-27 12:37:36（UTC+08:00）：Tile_0 持续优化

任务仍为 `processing_3d`，G:、左右照片和 LAS 可读。PID 7204 未变化，累计 CPU 时间约 2,800.1 秒；工作集约 4.65 GiB，私有内存约 8.00 GiB。GPU 瞬时利用率 75%，显存约 2,720 MB。日志总进度从 12:33:15 的 27.770054% 上升到 12:36:47 的 30.352173%，引擎自报 G-RAM 从约 2.23 GB 增至 2.71 GB。尚无新的 Tile splat/PLY；仍只有 Tile_0 块点云和全局背景 splat，因此判断 Tile_0 正在持续增密/优化，未发生块切换或错误。

### 2026-08-27 12:42（UTC+08:00）：Tile_0 输入点云与 LAS ROI 精确对照

为判断 Tile 输入点云是否由视觉/MVS/depth 扩充，直接解析 `3D\point-pnts\Tile_0\tileset.json` 引用的全部 PNTS 标准头，并对原始 LAS 做只读分块计数。

直接结果：

- 原始 `2026-02-24_16-21-11snow_colorized.las`：7,036,347 点，LAS 1.4 / PointFormat 7，尺度 0.0001 m；边界 X `[-39.7278, 47.8004]`、Y `[-46.5360, 33.2265]`、Z `[-2.1286, 14.9113]`。
- Tile_0 的 `tileset.json` 引用 441 个唯一 `.pnts`，所有文件魔数均为 `pnts`；逐文件读取 Feature Table 的 `POINTS_LENGTH` 后合计 **2,700,801 点**，合计文件大小 40,561,128 B。
- Tile_0 任务 ROI 为 X `[-54.388954, 0.251152]`、Y `[-53.264386, 3.356187]`、Z `[-14.620715, 22.145558]`，公共 offset 为 `[2.767021, -4.222885, 1.415041]`。
- 使用 ROI 直接过滤原始 LAS 得到 2,185,337 点；使用 `ROI - offset` 得到 929,697 点；使用 **`ROI + offset`** 得到 **2,700,799 点**。
- `PNTS / LAS(ROI+offset) = 2,700,801 / 2,700,799 ≈ 1.00000074`，绝对差仅 2 点。

结论：Tile ROI 坐标需要加公共 offset 后才能与原始 LAS 坐标对应；这个解释同时被点数近乎完全一致所验证。Tile_0 的层级 PNTS 点云与原始 LAS ROI 裁剪结果在点数上等价，差 2 点很可能来自边界舍入、量化或层级生成的极少量处理。当前证据强烈排除“先由 MVS/depth 生成数百万额外点，再作为 Tile Gaussian 初始化”的假设。MipMap 本次 LiDAR 模式的主体 Gaussian 主初始化点源，基本可以确认是原始 LAS 的 ROI 子集。

`milestones\point_cloud\Tile_0_point_cloud.pb.bin` 的 74,553,610 B 不是可直接用文件大小推算的裸点数组；PNTS 标准头给出的 2,700,801 才是当前可验证的 Tile 点数基线。后续待 Tile_0 最终 splat/PLY 生成后，将计算 `N_GS-final / 2,700,801`，量化 clone/split/cull 的净增密程度。

### Gaussian Lifecycle Audit：后续专项指标与当前可观测性

监控重点从普通总进度切换为 Gaussian 生命周期。Tile_0 当前已知基线为：LAS ROI 2,700,799 点，层级 PNTS 2,700,801 点。

当前初始化 checkpoint 可观测性：

- `milestones\splats\Tile_0` 目录于 12:29:36 创建，但截至 12:48:28 仍为空。
- 任务目录没有 Tile_0 中间 `.ply`、`.splat`、`.spz`、`.pt`、`.pth` 或 `.ckpt`；只有全局背景 splat/sky PLY、块 MVS 和 Tile_0 点云。
- 因而当前无法直接读取 `InitialParameters(PointCloud)` 刚完成时的 Gaussian 数、scale、opacity 或 orientation。为保持客户任务只读且不干扰训练，不对活跃进程做内存注入、调试附加或转储。
- 若首个 Tile 中间 checkpoint 后续出现，将立即解析；如果软件只在 Tile 完成时保存，则 `N_init` 只能通过最终点数、最近邻/父簇结构、DLL接口和下一次受控复现实验间接推断，不能伪装成已测事实。

生命周期审计表将按关键事件填写，而不是机械记录每一分钟：

| 时间/总进度 | Tile | LAS ROI点数 | Tile点云数 | 当前/最终GS数 | GS/点云比 | scale统计 | opacity统计 | dead% | GPU显存 | 事件 |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|
| init基线 | 0 | 2,700,799 | 2,700,801 | 待测 | 待测 | 待测 | 待测 | 待测 | 待关联 | `InitialParameters` 后暂无落盘 |
| 关键checkpoint | 0 | 同上 | 同上 | 待测 | 待测 | p10/p50/p90/p95 | p10/p50/p90 | `<0.005`等阈值 | 待测 | clone/split/reset/cull候选事件 |
| final/core | 0 | 同上 | 同上 | 待测 | 待测 | short/mid/long与aspect | sigmoid后分布（先验证编码） | 待测 | 待测 | Cut前后分别记录 |

出现可解析 Gaussian PLY/splat 后的固定分析顺序：

1. 读取 vertex/Gaussian 数和属性表，识别 SH 阶数；不能仅凭产品描述假设 SH0。
2. 验证 `scale_*` 是线性尺度还是 log-scale、`opacity` 是概率还是 logit，再计算真实尺度与透明度，避免错误解码。
3. 统计短/中/长轴 p10/p50/p90/p95、长短轴比、四元数范数和异常值。
4. 用原始 LAS 建最近邻索引，统计 Gaussian mean 到 LAS 的距离分桶：`<1 mm`、`1–5 mm`、`5–20 mm`、`>20 mm`。
5. 以 LAS 点为 parent，统计 5 mm / 10 mm 半径内 Gaussian 子点数 `0/1/2/3/4+`，识别克制局部 split 还是自由视觉增长。
6. 在 LAS 邻域计算 PCA normal，与 Gaussian 最短轴比较夹角，检验 LiDAR normal/PCA orientation 初始化假设。
7. 若有多个 checkpoint，形成 `进度—GS数—opacity—显存` 曲线，识别单调增密、锯齿式 `densify→reset/cull`、后期停止增密等 schedule。
8. 用相机模型计算可见 Gaussian 的 screen-space major-axis footprint，统计 p50/p90 及 `>3 px`、`>5 px`、`>8 px` 比例。
9. 将近LAS继承型与明显偏移/新增型 Gaussian 投影到照片，关联图像梯度、局部熵和边缘强度，判断新增点是否集中于碎石、屋檐、树枝等高频结构。
10. Tile_0/Tile_1 均完成后，专项比较 overlap 区位置、颜色、scale、opacity；再结合 Cut前后数量判断无缝效果来自共享照片一致性、核心ROI裁剪、颜色统一还是其他质量选择。

最优先交付的三个数字保持为：`Tile_0 N_init`（若有可观测checkpoint）、`Tile_0 N_final`、`Tile_0最终Gaussian→LAS最近邻距离分布`。

### 2026-08-27 12:53:18（UTC+08:00）：Tile_0 生命周期监控快照

直接证据：任务仍为 `processing_3d`；PID 7204 仍存活且响应正常，启动时间为 12:14:27，累计 CPU 约 4,095.81 秒、工作集约 4,278.8 MiB、私有内存约 9,860.1 MiB、104 个线程。`nvidia-smi` 瞬时记录 GPU 利用率 79%，显存 4,121/8,151 MiB。明文日志总进度从 12:46:35 的 32.417866% 上升到 12:51:41 的 33.967136%；引擎自报 G-RAM 在 3.82–4.25 GB 区间波动，说明训练仍持续消耗 GPU，但仅凭显存波动不能判定一次 clone/split/cull 事件。

文件证据：`milestones\splats` 仍只有 12:28:52 生成的全局背景 `gaussian_splat_background.pb.bin`；没有 Tile_0 checkpoint、PLY、splat 或块完成文件，日志也没有 Cut/Merge/块切换/终态明文。因此本次仅记录连续训练快照，不报告生命周期里程碑。

### 2026-08-27 12:55:13–12:59:38（UTC+08:00）：Tile_0 最终 Gaussian、六级 LOD 与 SOG 块完成

这是首次出现可解析的 Tile Gaussian，不是初始化 checkpoint。12:55:11 总进度到达 34.999985%，随后于 12:55:13 写出 `milestones\splats\Tile_0\gaussian_splat_level_0.pb.bin`；12:59:38 出现 `.temp\.tiles\Tile_0\Tile_0.done` 和块报告，确认 Tile_0 已完成。块报告写入 `reconstruction_time: 30.750000`；结合 12:29 左右进入 Tile_0，可推断该值单位很可能是分钟，但文件没有显式单位。

#### 二进制结构与 Gaussian 数量

- level 0 文件大小 99,312,420 B；首 4 字节小端无符号整数为 **1,773,436**，且 `4 + 1,773,436 × 56` 与文件大小严格相等，因此每个 Gaussian 为 14 个 float32、56 B。
- 使用同软件生成的背景 PB 与 `sky.ply` 对照字段重排，可确定 PB 记录顺序为：`x,y,z, scale_0..2, f_dc_0..2, opacity, rot_0..3`。导出的 PLY 则按常见 PLY 顺序把 `f_dc/opacity` 排在 `scale/rot` 前。
- Tile_0 最终 level 0 为 **1,773,436 GS**；相对 Tile 输入 PNTS 2,700,801 点，`N_GS-final / N_tile-point-cloud = 0.656633`。净结果比输入点少 927,365 个（34.34%）。这证明最终产物不是简单的一点一 Gaussian 原样保留，但单凭净数仍不能拆分训练中 clone/split、cull 与 Cut/ROI 各自贡献。
- 记录只有 DC 颜色 `f_dc_0..2`，没有 `f_rest_*`；后续 SOG 文件也只出现 `sh0.webp`。当前 Tile_0 输出证据支持 **SH0/DC-only**，不是更高阶 SH。

#### level 0 参数分布

PB 中原始 `scale_*` 大量为负，原始 `opacity` 也主要为负；按 3DGS 常见参数化，以 `exp(scale)` 和 `sigmoid(opacity)` 解码后得到合理的米制尺寸与 0–1 透明度。因此“log-scale/logit”是由字段范围和解码合理性共同支持的强推断；软件没有在明文元数据中直接声明变换函数。

- 三轴先按实际尺度排序后，短轴 p10/p50/p90/p95：**0.188/0.715/3.931/7.855 mm**。
- 中轴 p10/p50/p90/p95：**0.629/1.996/12.164/26.494 mm**。
- 长轴 p10/p50/p90/p95：**2.240/8.547/55.452/86.489 mm**。
- 长短轴比 p10/p50/p90/p95：**3.56/11.90/45.26/67.29**，说明 Gaussian 普遍高度各向异性，更接近贴合表面的薄椭球，而不是球形点精灵。
- opacity 概率 p10/p50/p90/p95：**0.0599/0.1148/0.3341/0.4776**；`opacity < 0.005/0.01/0.05` 均为 0。最终文件中没有按这些阈值定义的 dead Gaussian，说明低透明度元素已被过滤或训练结果本身受下限约束。
- 四元数原始范数 p10/p50/p90/p95：0.958/1.103/1.358/1.447；95.46% 与单位范数偏差超过 1%。因此文件保存的是未归一化旋转参数，渲染/协方差计算阶段很可能再归一化，不能把原始四元数直接当单位旋转。

#### 最终 Gaussian 到原始 LAS 的最近邻

使用已经验证的坐标关系 `Gaussian/local = LAS - coordinate offset`，对 2,700,799 个 LAS ROI 点建立只读最近邻索引，并查询全部 1,773,436 个 level 0 Gaussian mean：

- 最近邻距离 p10/p50/p90/p95/p99：**4.315 / 14.353 / 37.232 / 56.003 / 472.289 mm**。
- `<1 mm`：3,412（0.192%）。
- `1–5 mm`：231,161（13.035%）。
- `5–20 mm`：964,169（54.367%）。
- `≥20 mm`：574,694（32.406%）。

这组结果直接否定“最终 Gaussian mean 仍基本等于原 LAS 坐标”的简单模型：只有 13.23% 落在 LAS 的 5 mm 内，约三分之一已离最近 LAS 20 mm 以上。它更符合“LAS 提供主初始化/几何锚点，随后受照片损失驱动发生位置优化，同时经历裁剪和剔除”的机制。需要注意，当前没有初始化 checkpoint，因此还不能把位移精确分解为训练移动、split 子点偏移和 ROI Cut。

最近 LAS parent 聚类结果：5 mm 内共有 234,573 GS；按最近 LAS parent 计，0/1/2/3/4+ 子点的 LAS 数为 2,567,526 / 90,627 / 22,002 / 9,025 / 11,619，最大 37。10 mm 内共有 613,762 GS；对应 0/1/2/3/4+ 为 2,413,774 / 171,678 / 51,925 / 24,518 / 38,904，最大 78。局部确实存在一 LAS 对多 GS 的 split/聚集现象，但它不是最终全体 GS 的唯一结构。

#### LOD 与 SOG 分块的直接时序

Tile_0 在训练完成后连续写出六级 Gaussian LOD：

| LOD | Gaussian 数 | 相对上一级 | 写完时间 |
|---:|---:|---:|---|
| 0 | 1,773,436 | — | 12:55:13.046 |
| 1 | 886,490 | 0.499871 | 12:56:14.338 |
| 2 | 443,122 | 0.499861 | 12:57:01.199 |
| 3 | 221,460 | 0.499772 | 12:57:39.746 |
| 4 | 110,719 | 0.499950 | 12:58:16.674 |
| 5 | 55,314 | 0.499589 | 12:58:32.889 |

每一级都非常稳定地减半，这不是训练 checkpoint 曲线，而是训练后建立的离散 LOD 金字塔。12:58:36 开始写 `3D\model-gs-sog-tile\Tile_0`；`lod-meta.json` 明文声明 `lodLevels: 6`，空间树有 8 个叶节点，并把不同叶节点/LOD 用 `file + offset + count` 映射到 9 个容器目录。每个容器落盘为 `means_l.webp`、`means_u.webp`、`quats.webp`、`scales.webp`、`sh0.webp` 和 `meta.json`。这表明它先做空间树分叶，再对每个叶节点构造六级约 1/2 抽样 LOD，最后把量化后的 means/rotation/scale/SH0 分通道编码成 WebP；不是把整块只写成一个不可分割的 splat 文件。

### 2026-08-27 13:00:01–13:02:53（UTC+08:00）：切换到 Tile_1，并确认相邻块输入重叠

Tile_0 的 `.done` 于 12:59:38 写完后，13:00:01 立即生成 `Tile_1_point_cloud.pb.bin`，13:00:03 写出 Tile_1 层级 PNTS；13:02:49 总进度已到 47.677174%。PID 7204 未变，说明这是同一引擎实例内的串行块切换，而不是重启新进程。13:02:53 的 GPU 瞬时利用率 86%、显存 2,877/8,151 MiB，Tile_1 已进入 GPU 工作阶段。

直接解析 Tile_1 的 234 个 PNTS 文件：`POINTS_LENGTH` 合计 **1,520,716**。使用 Tile_1 ROI 加公共 offset 对原始 LAS 做只读计数，同样得到 **1,520,716**，差值 0、比值 1.0。这再次确认每个 Tile 的训练点云由原始 LAS ROI 直接裁切，不存在视觉/MVS/depth 点数扩充。

Tile_0 与 Tile_1 的训练 ROI 在本地 Y 方向明确重叠：Tile_0 上界 3.356186750，Tile_1 下界 3.152480487，重叠宽度 **0.203706263 m**，中线 Y=3.254333619。严格同时应用两块共享 X/Z ROI 后，该窄带内有 **33,172** 个原始 LAS 点。此前记录的 64,907 只按 Y 窄带过滤、漏加 X/Z 条件，现明确更正，不能继续作为 overlap LAS 基数。Tile_0 level-0 最终文件仍保留重叠带内 **47,660 GS**，其中中线下方 24,812、中线上方 22,848；其 Y 最大值也到达 Tile_0 ROI 上界。因此 Tile_0 的块里程碑是包含 halo/overlap 的训练结果，至少在保存 `milestones\splats\Tile_0` 时尚未按相邻块中线裁掉上半重叠区。后续若最终合并无重复层，Cut/SaveROI 更可能发生在全块完成后的 Merge 阶段，而不是每块训练结束前。待 Tile_1 完成后，将在同一 20.37 cm 重叠带逐点比较两块位置、颜色、scale 与 opacity，并检查最终合并实际保留哪一侧。

### 2026-08-27 13:07–13:12（UTC+08:00）：Tile_0 point-to-plane、法向对齐与局部密度审计

根据新的研究优先级，先对已完成的 Tile_0 做只读几何审计。为避免正在训练 Tile_1 时执行数百万次局部特征分解，最近点距离和 voxel 计数仍使用全量数据；PCA normal/point-to-plane 使用 level-0 中均匀等间隔抽取的 **150,000 GS**。每个样本先找最近 LAS anchor，再用 anchor 周围 24 个 LAS 邻点做 PCA。把 `curvature = λ0/(λ0+λ1+λ2) < 0.02` 且 `λ1/λ2 > 0.1` 定义为可靠表面邻域，可靠样本占 80.485%。这是确定性抽样统计，不冒充 1,773,436 个 GS 的全量 PCA。

#### point-to-plane 结果：强“纯切向重采样”假设目前不成立

150,000 样本的最近点距离 p50/p90/p95 与此前全量结果一致，为 14.368/37.067/56.019 mm，证明抽样没有明显偏离总体。

- 以最近 LAS anchor 为平面基点，位移的法向绝对分量 p10/p50/p90/p95/p99：**1.961/11.851/33.755/50.405/195.860 mm**；仅 24.153% 小于 5 mm，25.125% 大于等于 20 mm。
- 只看可靠表面邻域，法向分量 p10/p50/p90/p95/p99：**2.120/12.101/31.963/47.824/148.392 mm**；22.688% 小于 5 mm，24.165% 大于等于 20 mm。
- 使用 24 邻点质心定义局部 PCA 平面，可靠样本的点到平面距离 p50/p90/p95 为 **12.348/32.963/49.298 mm**，与 anchor 平面结论一致。
- 最近点位移的切向分量 p10/p50/p90/p95/p99：**1.802/5.330/15.571/21.809/270.528 mm**。中位数上法向位移反而约为切向位移的 2.2 倍。
- 在最近点距离已经 `≥20 mm` 的样本中，仅 **1.187%** 同时满足 anchor 法向分量 `<5 mm`；用邻域质心平面也仅 1.122%。

因此，用户提出的判别假设“32% 最近点超过 20 mm，但 95% 距 LiDAR surface 只有几毫米”在 Tile_0 上被当前数据否定。MipMap 的最终 GS 确实不是简单粘在 LAS 点上做纯切向重采样；相当一部分 Gaussian mean 相对局部 LAS 表面存在厘米级法向偏移。可能来源包括视觉损失把均值推向照片表面、LiDAR/影像表面偏差、局部动态/遮挡、split 后优化或后续质量选择；当前静态产物尚不能把这些原因分开。

#### Gaussian 最短轴与 LAS normal 高度一致

PB 四元数先归一化，并分别测试 `rot_0=w` 与 `rot_3=w` 两种解释。`rot_0=w` 时，可靠表面样本中 Gaussian 最短轴与 PCA normal 的夹角 p10/p50/p90/p95 为 **2.846°/9.198°/30.589°/54.694°**；71.855% 在 15° 内，89.748% 在 30° 内。若按 `rot_3=w` 解释，全样本中位角为 36.669°，明显更差。

这给出两项强证据：第一，该格式的四元数顺序很可能是 **wxyz**；第二，虽然 Gaussian mean 会离开 LAS 表面，但其薄轴方向大多仍由局部表面法向控制。更贴近当前数据的模型是“LiDAR 提供位置和 orientation 的强几何先验，视觉优化可在三维中移动 mean 并重新分配密度”，而不是“只允许沿切平面移动”。

#### 0.5 m / 1 m voxel 的局部 `N_GS/N_LAS`

这里使用 Tile_0 全量 2,700,799 LAS 和 1,773,436 GS，采用固定本地坐标网格：

- 0.5 m：LAS 占据 2,429 个 voxel，GS 占据 6,389 个，交集 2,299 个。仅 5.352% 的 LAS voxel 没有 GS；1.905% 的 GS 位于无 LAS 的 voxel。共享 voxel 的 `N_GS/N_LAS` p10/p50/p90/p95/p99 为 **0.075/0.386/3.000/8.100/38.050**；57.63% 小于 0.5，13.61% 大于等于 2。
- 1 m：LAS 占据 661 个 voxel，GS 占据 1,944 个，交集 631 个。4.539% 的 LAS voxel 没有 GS；0.971% 的 GS 位于无 LAS 的 voxel。共享 voxel 比率 p10/p50/p90/p95/p99 为 **0.115/0.502/5.810/17.750/77.500**；49.76% 小于 0.5，17.75% 大于等于 2。
- 按共享 voxel 内点数加权，GS/LAS 分别为 0.645（0.5 m）和 0.651（1 m），与全块净比例 0.6566 接近；但局部比率跨越两个以上数量级，表明训练/裁剪不是空间均匀下采样。

高分位极大值可能部分来自 LAS 分母很小的边缘 voxel，后续关联照片 gradient/visibility 时必须同时记录 `N_LAS` 并设置最小支持点数，不能只看裸比值。下一步把每个 voxel 的 `N_GS/N_LAS` 与可见相机数、投影有效像素数、梯度/局部熵关联；若高纹理且高可见区域系统性拥有更高比率，才能把“视觉信息驱动 Gaussian 密度重分配”从合理解释提升为实证。

监控优先级据此调整为：1）GS→LAS point-to-plane 法向/切向位移；2）0.5/1 m voxel 的局部 `N_GS/N_LAS`；3）local ratio 与图像 gradient/visibility；4）最终 scale/最短轴与 LAS normal；5）任何中间 checkpoint 的 GS 数量以还原 grow→cull 曲线。

### 2026-08-27 13:18:46–13:20:20（UTC+08:00）：Tile_1 level-0 与跨块 overlap 对照

13:18:46 写出 Tile_1 `gaussian_splat_level_0.pb.bin`，大小 111,353,204 B；按已验证的 4 B 计数头和 56 B/GS 记录解析为 **1,988,450 GS**。Tile_1 输入 PNTS 为 1,520,716 点，因此 `N_GS-final/N_point-cloud = 1.307575`，净增 467,734 GS（+30.76%）。这与 Tile_0 的 0.656633 形成鲜明对比：同一训练配置并非对每块执行固定比例抽样，而是按局部内容产生完全不同的净 grow/cull 结果。

Tile_1 level-0 参数：短轴 p10/p50/p90/p95 为 **0.102/0.403/2.395/4.568 mm**；中轴 **0.328/1.189/7.010/15.049 mm**；长轴 **1.415/7.247/46.916/70.701 mm**；aspect p10/p50/p90/p95 为 **4.38/16.27/68.86/104.93**。相较 Tile_0，Tile_1 的短轴更薄、各向异性更强。opacity 概率 p10/p50/p90/p95 为 0.0597/0.1177/0.4022/0.5931，`<0.05` 仍为 0，继续支持最终低 opacity 元素已被过滤/约束。

#### Tile_1 point-to-plane（150,000 个确定性等间隔样本）

方法保持一致：最近 LAS anchor，24 邻点 PCA，可靠规则 `curvature<0.02 && λ1/λ2>0.1`。可靠样本占 92.919%。

- 可靠邻域法向位移 p10/p50/p90/p95/p99：**2.414/10.930/36.185/54.196/147.028 mm**；21.676% `<5 mm`，23.037% `≥20 mm`。
- 可靠 PCA 平面距离 p50/p90/p95：11.059/37.128/55.612 mm。
- 全样本切向位移 p50/p90/p95：6.018/18.314/23.704 mm。
- 最近点距离 `≥20 mm` 的样本中，仅 2.638% 同时法向 `<5 mm`。
- Gaussian 最短轴与可靠 LAS normal 的夹角 p10/p50/p90/p95：**2.764°/8.875°/33.642°/61.085°**；71.993% 在 15° 内、88.441% 在 30° 内。

Tile_1 重复了 Tile_0 的核心模式：orientation 与 LAS normal 高度相关，但 mean 并非只沿切面移动，厘米级法向位移普遍存在。这使该结论从单块现象提升为两个相邻块的一致行为。

#### Tile_1 voxel 密度重分配

- 0.5 m 共享 voxel 的 `N_GS/N_LAS` p10/p50/p90/p95/p99：**0.139/0.941/7.307/18.200/76.758**，28.79% `≥2`；1.382% GS 落在无 LAS 的 voxel。
- 1 m 共享 voxel 的比率：**0.213/1.116/11.905/33.250/115.283**，31.25% `≥2`；0.695% GS 落在无 LAS 的 voxel。

Tile_1 的局部密度比整体显著高于 Tile_0，且净 GS 数超过 LAS 输入。这个跨块差异是“按内容重分配 Gaussian 密度”的直接证据，但尚未证明驱动变量一定是图像 gradient；仍需把 voxel ratio 与可见相机数、投影梯度和 LAS 支持点数联合建模。

#### 20.37 cm overlap：位置接近，但不是共享同一组 Gaussian

同一训练重叠带中，Tile_0 有 47,660 GS，Tile_1 有 31,881 GS，Tile_1/Tile_0 密度比 0.6689。以 Tile_1 GS 查询最近 Tile_0 GS：距离 p10/p50/p90/p95/p99 为 **0.957/2.592/8.497/13.766/69.763 mm**；77.70% 在 5 mm 内，96.81% 在 20 mm 内，但 1 μm 内精确重复为 0。

这说明两个块在共享 LAS/照片约束下恢复出了高度接近的表面位置，却各自保存独立 Gaussian，不是直接复用同一参数集合。对 `<5 mm` 的 24,772 对近邻：归一化 RGB L2 差 p10/p50/p90/p95 为 0.022/0.094/0.320/0.476；opacity 概率绝对差为 0.009/0.063/0.287/0.420；三轴排序后 log-scale 平均绝对差为 0.258/0.614/1.223/1.460（中位差对应约 1.85 倍尺度因子）。因此即使位置接近，颜色、透明度和尺度也没有被块间强制统一。

该结果进一步支持 `halo training → 保存带重叠的块结果 → 后续 Cut/SaveROI 按空间择一 → Merge`，而不是在 overlap 中对两块 Gaussian 做逐参数平均融合。如果最终直接叠加两组结果，会形成明显的双层密度和参数冲突；必须继续观察最终 Merge 前后的数量与边界裁剪。

13:19:39 和 13:20:20 又分别写出 level 1=994,436、level 2=497,160，继续保持约 1/2 LOD 递减；Tile_1 的后续 LOD/SOG 和 `.done` 尚在生成，本节不提前宣称块已完全结束。

### 2026-08-27 13:21–13:25（UTC+08:00）：Overlap 法向位移一致性天然实验

目标是区分“两个块从同一 LAS 表面确定性地向同一视觉表面修正”与“独立优化/MCMC 随机漂移”。分析使用严格共享 ROI 内的 33,172 个 LAS 点、Tile_0 的 47,660 GS 和 Tile_1 的 31,881 GS。所有对照都让两块使用同一个 LAS anchor、同一个 24 邻点 PCA 平面和同一条确定性定向的 normal。

第一套口径不按两块互相最近配对，而是按**共同最近 LAS parent**聚合，可避免“挑选空间上本来就很近的 pair”导致循环论证。限制 GS 距 anchor `<20 mm` 且 PCA 邻域可靠后，共有 2,227 个 LAS parent 同时被两块占用：

- Tile_0/Tile_1 的 parent 中位 signed normal displacement 同侧比例 **79.79%**；把 Tile_1 parent 顺序确定性错位作为对照时仅 50.25%。
- signed displacement Spearman `ρ=0.723`，绝对位移 Spearman `ρ=0.522`。
- 两块法向位移绝对差 p10/p50/p90/p95/p99：**0.331/1.966/7.360/10.367/15.586 mm**；错位对照的中位差为 5.898 mm。
- 两块位移都超过 5 mm 且方向相同的 parent 占 23.98%。

收紧到 `<5 mm` anchor 后仍有 1,269 个共同 parent：同侧 70.13%（错位 48.54%），signed Spearman `ρ=0.490`，法向差中位数 1.250 mm（错位 2.127 mm）。

第二套 mutual-nearest 一对一口径仅作佐证，因为其筛选本身偏向一致结果：5 mm 内有 6,760 对可靠 pair，同侧 94.79%、signed Spearman 0.982、法向差中位数 0.664 mm；20 mm 内 7,136 对，同侧 94.70%、Spearman 0.977、法向差中位数 0.703 mm。

共同 parent 结果已经足够说明：两个独立块相对同一 LAS 局部平面的法向修正具有明显同向和幅值相关性，远高于错位对照。这是目前最强的证据之一，表明 overlap 中相当一部分 mean 位移由共同的确定性约束驱动，更像相同照片监督/几何目标导致的 gradient-based surface correction，而不是独立随机 relocation。它不能排除 MCMC relocation 对局部 density 的作用；由于没有训练中 checkpoint，仍无法直接观察“低 opacity A 点瞬移到 B 点”的时间事件。

### 2026-08-27 13:25–13:27（UTC+08:00）：密度与 visibility/geometry 的初步 Spearman

发现 `AT\mvs_undistort.xml` 明文包含 1,368 个去畸变视图的完整内外参、Near/Median/FarDepth 与图像路径。用 500 个 tie-point measurement 验证投影公式 `q=R(X-C)`，重投影误差中位数约 `4.6e-7 px`、p90 约 `7.7e-7 px`，因此相机矩阵解释可靠。

本轮先计算不读取 1.102 GB 图像的 **frustum visibility proxy**：0.5 m LAS voxel 质心投影在图像边界内且深度为正即计可见；另记录 Near/FarDepth 门控版本。它没有做遮挡测试，不能称为真实 visibility。为抑制 `N_GS/N_LAS` 的小分母伪高值，只对 `N_LAS≥100` 的共享 voxel 做 Spearman。

Tile_0（1,505 个稳定 voxel）：

- density ratio vs frustum visibility `ρ=0.068`，vs depth-gated visibility `ρ=0.123`：很弱。
- vs LAS density `ρ=-0.108`，vs surface curvature `ρ=0.177`，vs camera distance `ρ=-0.208`。
- 75,000 GS 确定性抽样形成的 voxel 中位 normal displacement，在 1,352 个稳定 voxel 上：vs visibility `ρ=-0.354`，vs depth-gated visibility `ρ=-0.465`，vs curvature `ρ=0.224`，vs camera distance `ρ=0.551`，vs density ratio `ρ=-0.139`。

Tile_1（704 个稳定 voxel）：

- density ratio vs visibility `ρ=0.050`，vs depth-gated visibility `ρ=0.038`：同样接近零。
- vs LAS density `ρ=-0.195`，vs curvature `ρ=0.309`，vs camera distance `ρ=-0.046`。
- 75,000 GS 抽样在 657 个稳定 voxel 上：normal displacement vs visibility `ρ=-0.457`，vs depth-gated visibility `ρ=-0.473`，vs curvature `ρ=0.397`，vs camera distance `ρ=0.497`，vs density ratio `ρ=0.115`。

当前解释：仅用几何视锥代理时，“多视角区域获得更多 GS”没有得到支持；两个块的密度比更稳定地随 surface curvature 增加。mean 法向位移则在两个块中都表现为**越远、视角覆盖越少、曲率越高，位移越大**。这更像远距/低可见/复杂几何区域的不确定性和表面修正，而不是“高 visibility 自动增加密度”。但 visibility 代理没有遮挡，且还没有图像 gradient、entropy 或 photometric residual，结论只能作为初步相关性，不能当因果解释。

下一步读取图像时将直接使用已验证投影，把每个稳定 voxel 的多视角 gradient/entropy 做 median、p90 和有效观测数；考虑到去畸变 JPEG 共 1.102 GB，活跃训练期间不做全量并发解码，避免与 MipMap 争用磁盘/CPU。任务完成或进入低负载阶段后再执行受控扫描。局部 photometric residual 尚无软件落盘的 per-pixel residual，将优先寻找可验证的渲染残差或采用跨视图颜色离散度作为明确标注的代理，绝不把代理冒充原始 photo loss。

### 2026-08-27 13:21:43–13:32:11（UTC+08:00）：Tile_1 完成并切换到 Tile_2

Tile_1 六级 LOD 已完整写出：level 0–5 分别为 **1,988,450 / 994,436 / 497,160 / 248,479 / 124,179 / 62,005**，各级继续稳定约减半。13:21:43 写完 level 5 和 `levels_info.json`，随后生成 Tile_1 SOG；13:22:46 写入 `Tile_1.done` 和块报告。报告值 `reconstruction_time=23.116667`，结合块实际时间仍支持单位为分钟的推断。

13:23:41 开始写 `Tile_2_point_cloud.pb.bin`，13:23:45 完成 Tile_2 层级 PNTS，证明引擎在同一 PID 7204 内按 Tile_0→Tile_1→Tile_2 串行处理。直接解析 Tile_2 的 306 个 PNTS，`POINTS_LENGTH` 合计 **1,802,271**；原始 LAS 在 Tile_2 ROI+offset 内也为 **1,802,271**，差值 0。第三个块再次严格确认输入点云就是 LAS ROI 子集。

13:31:14 总进度 66.551437%，GPU 瞬时利用率 92%、显存 3,894/8,151 MiB；Tile_2 已进入训练，尚未生成自身 level-0。Tile_1 的完整生命周期/overlap 审计已完成，但仍没有任何训练中 checkpoint，因此 move/split/cull/relocate 的时间序列判别门仍未打开。

### 2026-08-27 13:39:09–13:40:21（UTC+08:00）：Tile_2 level-0 与初步几何审计

13:39:09 写出 Tile_2 level-0，大小 84,365,012 B，解析为 **1,506,518 GS**。相对输入 1,802,271 点，`N_GS-final/N_input = 0.835900`，净减少 16.41%；它位于 Tile_0 的 65.66% 与 Tile_1 的 130.76% 之间，再次证明块间密度分配高度非均匀。

Tile_2 短轴 p10/p50/p90/p95 为 0.382/1.266/6.433/11.350 mm，中轴 0.957/3.040/20.359/34.789 mm，长轴 3.643/14.479/67.550/102.078 mm，aspect 为 3.13/10.08/40.57/61.02。opacity p10/p50/p90/p95 为 0.0597/0.1155/0.3334/0.4722，`<0.05` 仍为 0。

对 100,000 个确定性等间隔 GS 做相同的 LAS 24 邻点 PCA，可靠邻域占 77.164%。最近 LAS 距离 p10/p50/p90/p95/p99 为 **5.761/17.335/104.862/712.891/3,205.373 mm**，43.70% `≥20 mm`；可靠邻域法向位移 p10/p50/p90/p95/p99 为 **2.789/14.170/49.654/96.460/659.787 mm**，18.14% `<5 mm`、35.77% `≥20 mm`。最近点 `≥20 mm` 的样本中，仅 1.943% 法向 `<5 mm`。

Tile_2 的高分位位移显著大于前两块，说明该块存在更多远离 LAS 的 Gaussian 或局部稀疏/遮挡区域；必须在后续 voxel、visibility、gradient 与最终空间边界中定位这些长尾，不能把 p95/p99 简化成全块统一偏移。13:39:48 和 13:40:20 已继续生成 level 1/2，任务正建立 LOD，Tile_2 尚未 `.done`。

### 2026-08-27 13:39:09–13:46:40（UTC+08:00）：Tile_2 完成并切换到 Tile_3

Tile_2 六级 Gaussian LOD 已完整写出。直接按 `4 B count header + 56 B/GS` 解析，level 0–5 点数依次为 **1,506,518 / 753,226 / 376,612 / 188,294 / 94,176 / 47,105**，每级继续稳定约减半。13:41:30 写完 level 5 与 `levels_info.json`，随后生成该块的 SOG/层级文件；13:42:18 写入 `block_reconstruction_Tile_2_report.json` 和 `Tile_2.done`。块报告给出 `reconstruction_time=19.516667`，结合文件时序继续支持其单位为分钟的推断。

13:42:30 写出 `Tile_3_point_cloud.pb.bin`（57,381,448 B），13:42:32 完成 Tile_3 层级 PNTS。直接解析 208 个 PNTS 的 `POINTS_LENGTH`，合计 **1,221,675 点**；按 Tile_3 ROI 与共同 offset `[2.7670208738,-4.2228845455,1.4150408912]` 对原始 LAS 做闭区间计数得到 **1,221,671 点**，仅差 4 点（0.000327%）。这与 Tile_0 的 2 点边界级差异、Tile_1/2 的完全相等一致，继续确认 Tile 输入点云是原始 LAS 的 ROI 子集，而不是 MVS/depth 新生成的稠密点云；微小差异最可能来自量化或边界包含规则，不能解释为视觉增密。

13:46:23 总进度 **83.360023%**，PID 7204 仍存活；13:46:40 进程累计 CPU 10,218.89 s，工作集 2.325 GiB、私有内存 7.896 GiB，GPU 利用率 75%、显存 2,696/8,151 MiB。直接证据表明 Tile_3 已进入活跃训练，尚未生成 level-0，也没有训练中 checkpoint。下一关键证据是 Tile_3 level-0 的净 GS/LAS 比及其几何分布，之后才进入四块 Cut/SaveROI、Merge、sky append 和最终 PLY/SOG 阶段。

### 2026-08-27 13:52:12–13:53:00（UTC+08:00）：Tile_3 level-0

13:52:12 写出 Tile_3 `gaussian_splat_level_0.pb.bin`，大小 42,027,892 B，解析为 **750,498 GS**。相对 PNTS 输入 1,221,675 点，`N_GS-final/N_input = 0.614319`，净减少 38.57%；这是四块中目前最低的净保留比例（Tile_0 65.66%、Tile_1 130.76%、Tile_2 83.59%），进一步排除固定比例下采样机制。

Tile_3 最终参数直接统计：短轴 p10/p50/p90/p95 为 **0.347/1.304/8.580/14.672 mm**，中轴 **0.890/3.485/27.111/44.559 mm**，长轴 **4.490/17.719/80.034/124.200 mm**，aspect 为 **3.242/11.571/51.473/82.584**；opacity 概率 p10/p50/p90/p95 为 **0.058/0.107/0.309/0.443**，`<0.05` 仍为 0。13:52:38 和 13:53:00 已写出 level 1=375,220、level 2=187,619，稳定约减半。Tile_3 正在生成剩余 LOD，尚未写 `.done`；因此本节只确认 level-0 净结果，不提前宣称块完成，也不从最终净数反推不可见的 grow/cull 时间曲线。

### 2026-08-27 13:53–13:58（UTC+08:00）：Tile_2 大法向位移的相邻块独立复现

本轮专门回答：Tile_2 的厘米级 mean 位移是否在相邻 Tile 的 overlap 中由另一次独立训练复现。统一方法为：严格共享 XYZ ROI、同一原始 LAS anchor、24 邻点 PCA 平面、同一个确定性定向 normal；每块先按最近 LAS parent 聚合 signed normal displacement，再对共同 parent 比较。mutual-nearest 没有用于主结论。确定性错位 Tile_2 parent 顺序作为“没有空间对应关系”的对照。近表面口径要求两块 GS 最近 LAS 距离 `<20 mm` 且 PCA 邻域可靠；大位移口径不设 20 mm 最近点门，因此必须同时报告最近 anchor 距离并降低极端长尾置信度。

#### Tile_0 ↔ Tile_2：23.287 cm 左右 overlap

严格 overlap 内原始 LAS 为 70,320 点；Tile_0/Tile_2 分别有 42,418/64,165 GS。近表面可靠共同 parent 共 4,740 个：

- signed 位移同侧率 **83.48%**，错位对照 50.53%；signed Spearman `ρ=0.751`，绝对位移 `ρ=0.491`。
- 两块法向位移绝对差 p10/p50/p90/p95 为 **0.505/3.173/11.087/14.790 mm**。
- Tile_0/Tile_2 绝对位移中位数为 8.431/7.273 mm，说明近表面区不只是方向一致，幅值也相当接近。

以 Tile_2 的大位移分段：

- **20–50 mm**：1,316 个共同 parent；同侧率 **96.43%**（错位 54.64%），signed `ρ=0.731`，绝对幅值 `ρ=0.269`；Tile_0/Tile_2 绝对位移中位数 23.937/27.132 mm，差值中位数 6.569 mm；66.34% 在两块中都 `>20 mm` 且同向。
- **50–100 mm**：99 个 parent；同侧率 **95.96%**（错位 53.54%），signed `ρ=0.736`，但绝对幅值 `ρ=0.048`；差值中位数 27.444 mm。方向高度复现，精确幅值已不稳定。
- **≥100 mm**：仅 15 个 parent；Tile_2 最近 LAS 距离中位数已达 264.37 mm，支持数和 anchor 可信度均不足，不作稳定复现结论。

#### Tile_3 ↔ Tile_2：20.371 cm 上下 overlap

严格 overlap 内原始 LAS 为 24,497 点；Tile_3/Tile_2 分别有 9,624/12,755 GS。近表面可靠共同 parent 952 个：同侧率 **72.58%**（错位 52.21%），signed `ρ=0.548`、绝对位移 `ρ=0.343`，法向差中位数 4.568 mm。该边界的一致性弱于 Tile_0↔Tile_2，但仍显著高于错位关系。

- Tile_2 **20–50 mm**：358 个 parent；同侧率 **76.82%**（错位 50.00%），signed `ρ=0.483`，绝对幅值 `ρ=0.051`；Tile_3/Tile_2 绝对位移中位数 16.581/26.262 mm，差值中位数 16.339 mm；32.12% 两块都 `>20 mm` 且同向。
- **50–100 mm**：仅 32 个 parent；同侧率 68.75%（错位 37.50%），signed `ρ=0.187`，证据较弱。
- **≥100 mm**：29 个 parent，Tile_2 绝对位移中位数 210.97 mm、Tile_3 仅 41.60 mm，绝对幅值相关为负且 anchor 距离很大，不能视为同一视觉表面的精确复现。

#### 当前判定

两条相邻边界都独立复现了 Tile_2 **中等厘米级（尤其 20–50 mm）位移的方向**；Tile_0↔Tile_2 还复现了相当接近的幅值。这是比单个 Tile point-to-plane 分布更强的直接证据：MipMap 的主要厘米级 geometry movement 并非随机漂移，存在由共同照片/几何目标驱动的确定性表面优化。因而当前主模型应是 **LiDAR-seeded、photo-guided Gaussian geometry**，而不是只优化固定 LiDAR surfel 的颜色/尺度。

但结果也明确划出边界：位移越极端，绝对幅值复现越差；Tile_2↔Tile_3 尤其明显。≥100 mm 长尾还伴随远 LAS anchor 和小样本，可能混合视角覆盖差异、边界/halo、错误 anchor、MCMC relocation 或独立块优化不稳定。没有训练中 checkpoint 时，不能从该终态对照单独证明 MCMC。最终 Cut/SaveROI 与 Merge 的审计应重点检查这些不一致区域是否被 core ownership 丢弃，从而判断 tile ownership 是否同时承担隐藏极端局部不稳定性的作用。

### 2026-08-27 13:53:53–13:54:36（UTC+08:00）：终态、最终 PLY/SOG 与 Cut/Merge 结论修正

Tile_3 六级 LOD 于 13:53:53 写完，level 0–5 为 **750,498 / 375,220 / 187,619 / 93,745 / 46,807 / 23,382**；块报告 `reconstruction_time=12.000000`。13:54:19 写入 Tile_3 块报告和 `.done`，13:54:23 写出紧凑 PLY，13:54:29 写出 UE full PLY、`tiles.json` 与 `rec.done`，13:54:31 明文日志报告 `Progress=100`、`3D Reconstruction Finished`，总重建时间 **1.665 h**。13:54:36 `info.json.status=complete`，PID 7204 随后退出，属于正常成功终态。

#### 最终 PLY 格式与数量

- `model-gs-ply/gs.ply`：337,058,875 B，binary little-endian，14 个 float32/vertex，header 363 B，`element vertex 6,018,902`；文件大小精确满足 `363 + 6,018,902 × 56`。
- `model-gs-ply/ue/gs_full.ply`：1,492,689,229 B，62 个 float32/vertex，header 1,533 B，同样为 **6,018,902 vertex**，大小精确满足 `1,533 + 6,018,902 × 248`。10,000 个确定性等间隔样本中，position、f_dc、opacity、scale、rotation 与紧凑 PLY 精确相等，45 个 `f_rest_*` 全为 0；它主要是 UE/通用 3DGS 属性布局扩展，不是新增 Gaussian。
- `sky.ply` 独立存在，100,000 vertex、5,600,362 B；`sky_full.ply` 也独立存在。最终两个主 GS PLY 的 vertex 数均不包含 sky，说明本任务没有把 sky append 到同一个 vertex stream。

四块 level-0 数量之和恰好为：

`1,773,436 + 1,988,450 + 1,506,518 + 750,498 = 6,018,902`

这与最终 PLY vertex 数**精确相等**。进一步按上述累计边界把最终 `gs.ply` 分成四段，每块确定性抽 1,000 条记录，与对应 `gaussian_splat_level_0.pb.bin` 比较：位置完全相等；按 PLY 字段顺序重排后，f_dc、opacity、scale、rotation 也全部精确相等。最终 PLY 的顺序就是 **Tile_0 → Tile_1 → Tile_2 → Tile_3**。

#### SOG tiles 也保留各块完整数量

`model-gs-sog-tile/tiles.json` 明文引用四个独立 `Tile_*/lod-meta.json`。递归汇总每个 LOD tree 的 count，分别与四块 PB 的 level 0–5 **逐级精确相等**；Tile_0/1/2 各 9 个 SOG 数据分片目录，Tile_3 为 7 个。四块 SOG 文件总大小约 124.60 MB，均包含 `means_l/means_u/quats/scales/sh0.webp + meta.json`，没有观察到跨块参数平均或合并后的第五个全局 block。

#### 对 Cut/SaveROI → Merge 推断的直接修正

终态证据否定了“本任务先把 overlap Cut 成不重叠 core，再合并到最终 PLY”的先前高概率推断：

1. 四块 level-0 自身包含 overlap 中的两套独立 GS；
2. 最终 PLY 数量严格等于四块完整数量之和；
3. 最终 PLY 的四段参数与各块 PB 抽样逐字段精确相等；
4. SOG 也按四个独立块保存完整 LOD 数量。

因此这次 MipMap Lite 1.0、高质量、`gs_ply + gs_sog_tiles` 路径的实际“Merge”是**字段重排 + 顺序串接/索引四个 block**，没有证据表明执行了 Gaussian 平均、去重、颜色协调或 core ownership 裁剪。DLL 中的 `GaussianSplatData::Cut`、`SaveROI` 字符串可能属于其他输出模式、条件分支或未在此任务调用；不能再用字符串存在推断本次运行已执行 Cut。

这也改变了对 overlap 的产品解释：Tile_0/1 与 Tile_0/2、Tile_2/3 的独立结果会同时进入最终 PLY；SOG 则由上层 block tree 决定加载，但数据本身仍完整保留。约 20 cm 的 halo 很窄，加上中等位移高度一致，可能使双层影响不明显；然而极端位移幅值不一致并没有被离线 Cut 隐藏。后续若要判断实际渲染是否避免双层，必须检查 MipMap Viewer/SOG runtime 是否在 block 边界做选择或裁剪，不能从产物生成管线假定已有 ownership。

#### 最终研究结论

本任务最稳健的模型是：**原始 LAS ROI 逐块精确提供初始化点；每块在照片监督与局部显存预算下独立重分配 Gaussian 数量并优化 mean/scale/orientation/opacity；主要 20–50 mm 表面修正在相邻块中可确定性复现，而极端长尾幅值不稳定；各块随后分别建立约二分 LOD，最终 PLY 直接串接完整块，SOG 保持四块独立索引。**

### 2026-08-28：分块算法和显存管理终态反汇编/回放补记

#### 直接静态证据

`divide_engine.exe`（SHA-256 `9DBBCB7059363B8460D643D73553432C035D5A0073962A17988CB53A3D0A747D`）RVA `0x3C690` 是 `divide_mode=2` 的递归节点处理函数。它以每块照片投影裁剪像素数而非 LAS 点数作为主要负载：节点少于 100 个锚点且小于 100,000 px 时停止；否则在估算负载超过预算且深度小于 10 时递归切分。

切分器 RVA `0x3D830/0x3E7C0/0x3E1F0` 直接证明：本任务 `pipeline_mode=1` 只评估 X/Y；每轴 64 个候选；每个锚点按关联 view 从相机位姿重新投影、round 并更新 per-image 前缀/后缀矩形；矩形至少 256×256，四周各扩 128 px；切线取左右像素负载穿零处。轴代价差小于 10% 时优先较长空间轴，否则优先较低最大子负载轴。因而此前“不宜命名 KD-tree”的边界已由新证据关闭：它确实是递归平面 KD，只是 cost 不是普通点数，而是 view crop pixel workload。

节点内存原始估值为 level-0/1 的 `pixel_load*5.5 B`；RVA `0x3C876` 再乘 `0.8` 后才与预算比较。叶 `tiles.json.max_memory` 保存未乘 0.8 的原始 GiB。启动参数 RVA `0x4A7F1..0x4A8C6` 把 `min_avali_memory_size` 设为 `min(GPU0 free GiB, 12, system available GiB)`。

`mipmap_gaussian_splat.dll` RVA `0xA6B13..0xA6B2B` 在每个偶数 global step 访问 `c10::cuda::CUDACachingAllocator::allocator` 并调用虚函数 `+0x60`；同一调用还位于对象清理路径 `0x59862..0x59870`。结合 PyTorch CUDA allocator ABI，高置信对应 `emptyCache()`。这解释了训练期显存波动和块结束后回落，但不表示每两步把活跃模型张量卸载到 CPU。

#### 运行回放测量

反解 0.2% 单侧 halo 后，根 core 为 X `[-54.280109411,62.156955725]`、Y `[-53.151596075,48.701530461]`，实际递归切线为根 X `0.142307`、左 Y `3.243397`、右 Y `-0.492836`。因此训练 overlap 是每个子块按自身 extent 各扩 0.2%，不是固定 20 cm。

在真实 core ROI 上用 `mvs_undistort.pb.bin` 重放裁剪面积并乘 5.5 B/px：Tile_0–3 得到 6.6978/6.0477/5.0894/4.2666 GiB；厂商记录为 6.5911/6.0416/5.0818/4.2639 GiB。Tile_1–3 相对误差均小于 0.2%，Tile_0 为 1.619%。因此 5.5 B/px 已被运行级校准；此前若把 `0.8*5.5=4.4` 当作导出的叶估值，需要以本节更正。

220 条 MemoryProfile 与产物时间戳对齐表明四块无并行训练。Tile_0–3 的实测训练 VRAM 峰值为 4.53/4.56/4.41/3.44 GiB；前三块 LOD 后均回到约 0.9 GiB。计划 `max_memory` 是图像工作量代理，不是 CUDA hard cap。

#### 兼容复刻产物和边界

已生成：

- `.tools/replay_mipmap_adaptive_tiling.py` 与 `snow-20260827-mipmap-adaptive-tiling-replay.json`；
- `.tools/audit_mipmap_tile_memory.py` 与 `snow-20260827-mipmap-tile-memory-audit.json`；
- 独立复刻规范 `snow-20260827-mipmap-tiling-vram-reproduction-spec.zh-CN.md`。

现有回放可得到正确的 X→Y/Y 拓扑、4 块、0.2% halo 和叶内存量级。三条切线相对厂商值仍差约 0.69–1.81 m，约一个 64-bin 间隔；主要未同构部分是全相机模型重新投影/round、root pre-clip 和 poor-split retry 的端点细节。因此状态是“可用兼容复刻”，不是“任意场景浮点级相同”。

未解决项仍是训练内部的 grow/cull/relocate 时间序列：整个运行未落盘任何中间 checkpoint，因此不能仅凭四块最终净比例判定 clone、split、cull 与 MCMC relocation 的具体 schedule。要把这一层从推断变为直接证据，需要软件提供迭代 checkpoint、调试日志或同场景受控重复实验；本轮只读终态数据不足以恢复不可见事件。

### 2026-08-28：分块数值闭环补正（取代上节“差一个 bin”结论）

进一步区分节点内的连续 ownership box 与离散 candidate-point envelope，并将 raw/undistort 两阶段数据角色分开后，snow 已不再存在 0.69–1.81 m 的切线误差。以启动时 `7.013 GiB` 预算运行回放，不强制深度、不强制切轴，自动得到 4 个叶块：

```text
root  X =  0.14230728149414062
left  Y =  3.243396759033203
right Y = -0.49283599853515625
```

四个预测叶块与厂商 Tile_0–3 的中心误差仅 `3.47e-9–3.97e-9 m`。因此上节“切线仍差约一个 64-bin 间隔”的描述已经过时，应以本节和独立复刻规范为准。

当前精确兼容模型为：raw 342-view observation graph 决定左右照片负载和切轴；undistort 点包围盒决定 64 个 float32 候选坐标；undistort fresh projection/crop 决定叶块 `max_memory`。根候选 X 范围 `[-53.4560470581,48.8680877686]` 的 candidate[33] 精确产生根切线；左子 candidate[39] 精确产生左 Y；右子在候选 envelope 使用 support≥2 点后精确产生右 Y。后两阶段的数值和运行文件时序完全闭合，但 raw/undistort 是否由同一厂商调用直接双路传入，以及 support≥2 的静态条件来源，仍需继续向上追调用链确认。

四个叶块的 undistort pixel load 更新为 `1,306,820,686 / 1,179,953,745 / 992,946,852 / 832,279,267`；对应 5.5 B/px 重放相对厂商 `max_memory` 的误差为 `+1.560% / +0.041% / +0.086% / -0.017%`。Tile_1–3 已低于 0.1%，Tile_0 仍有 1.56% 异常需要单独解释。

Tile view 选择也已审计：实际块 MVS 为 656/644/607/595 views；由导出 ROI 内点的 observation 图闭包预测，与实际集合 Jaccard 为 `0.9924/0.9787/0.9918/0.9691`，且实际 view 全部被预测覆盖。内存规划只计入通过 256 px crop 门限的 541/545/476/484 views，所以训练上下文 view 数不能直接替代 `max_memory` 的有效 image 数。

更新产物：

- `.tools/replay_mipmap_adaptive_tiling.py`；
- `results/diagnostics/snow-20260827-mipmap-adaptive-tiling-exact-budget-replay.json`；
- `.tools/audit_mipmap_tile_view_selection.py`；
- `results/diagnostics/snow-20260827-mipmap-tile-view-selection-audit.json`；
- `results/diagnostics/snow-20260827-mipmap-tiling-vram-reproduction-spec.zh-CN.md`。

证据边界仍需保留：raw 畸变类型 2 的专有投影尚未独立实现，当前回放显式使用保存的 observation xy 作 surrogate；snow 的数值闭环不能替代第二数据集泛化验证。

### 2026-08-28：MVS 输入调用链与 4:1 父 view 映射补正

随后静态追踪关闭了一个重要歧义。`divide_engine.exe` RVA `0x2FFF7..0x30079` 明确构造 `mvs_undistort.pb.bin` 路径、检查存在性；若存在就加载该文件，否则才回退 `mvs.pb.bin`。因此本次 division 递归实际读取 undistort MVS，不能再把精确兼容回放中的 raw/undistort 分工描述成“厂商同时将两个 PB 传入切分器”。上层参数构造 RVA `0x49DE0` 另行加载 `mvs.pb.bin`，主要服务缺省 ROI/场景参数初始化，与 division 入口选中的 MVSBlock 用途不同。

新增父 view 审计解释了为何 342-view 原图回放仍能精确模拟切轴：

- 1368 个 undistort image 用其 metadata 中的原相机 ID + timestamp 可全部精确映射到 342 张父照片，每组 4 个；1368/1368 与相机中心映射一致，到父照片中心的最大距离仅 `7.26e-15 m`；虚拟 `image_rect` 为 1456×2912，但 1368 条 metadata 均保留 2912×2912 原图尺寸；
- 18,847 个 undistort point 全部能按 float32 xyz 精确映射到 18,931 个原 MVS 点；
- 将 undistort image 折叠到父照片后，得到 137,468 个唯一 `(point,parent-image)` 对；它们 100% 存在于原 MVS 的 153,316 对中，无额外错误边；覆盖原图 observation graph 的 89.663%；
- 150,029 条 undistort observation 折叠后会合并为 137,468 个唯一父图对，说明部分同一点会进入同一父照片的多个虚拟 view。

因此更准确的当前模型是：厂商加载单个 undistort block，切分负载阶段存在某种 4:1 父照片聚合/筛选语义；候选点框与叶块内存仍来自 undistort 几何和虚拟 view crop。兼容工具用原图 observation xy 模拟该父照片负载，所以 snow 数值精确，但内部聚合公式仍未静态闭合。

新增只读产物：`.tools/audit_mipmap_undistort_parent_views.py` 与 `results/diagnostics/snow-20260827-mipmap-undistort-parent-view-audit.json`。
