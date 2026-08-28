# snow2-20260828 AT—LiDAR 受控实验：原始基线

更新时间：2026-08-28（UTC+08:00）
任务目录：`D:\mipmap-lite\1fad9647-f717-4cd4-9391-11f219d7e5d1\snow2\snow2-20260828`
证据镜像：`G:\cloudstudio-3dgs\results\diagnostics\snow2-20260828-live-evidence`

## 1. 目的与证据边界

本轮是 P1-1 的原始输入重复基线，用于先测量 MipMap AT 自身的运行间随机波动，再与后续“LAS 整体平移”和“相机时间戳偏移”任务比较。

单次原始运行不能单独证明 AT 是否消费 LiDAR。本轮可以直接回答：

1. 当前 AT 输入、输出、运行时文件和清理前临时文件是什么；
2. 相同输入重复 AT 的自然波动量级；
3. 后续受控实验需要超过什么噪声基线，才可以归因于 LAS 或时间戳改动。

## 2. 精确时间线

| 时间 | 直接证据 |
|---|---|
| 11:49:38 | MipMap 创建并启动任务 |
| 11:49:48 | `reconstruct_full_engine.exe` 开始 AT；342 图、2 相机、`reconstruct type=1` |
| 11:51:44 | 日志写出 `AT Finished`、进度 100%，并生成 `.temp\at.done` |
| 11:51:56 | 自动进入 `Start 3D Reconstruction` |
| 11:54:19 | 3D 进度达到 5.0% |
| 11:54:23 | 用户按要求在 UI 点击 Stop；`info.json.status=stop` |
| 11:54:56 | `reconstruct_full_engine.exe` 进程数已为 0 |

AT 报告耗时为 `1.733333 min`。Stop 由用户在 MipMap UI 中执行，监控没有修改、停止或注入 MipMap 进程。

## 3. 证据保全结果

终止后源目录和镜像目录逐项统计均为：

- 文件数：`3113`
- 总字节数：`1,263,955,837`
- 超过 512 MiB 而未复制的文件：`0`

镜像哈希清单：`snow2-20260828-live-evidence\evidence-manifest.sha256.csv`

- 清单行数：`3113`
- 清单 SHA-256：`2B3600B77692E076D5339F54B8C0E38491E539662B99F65BD6FF85787836684A`

已经保全的关键文件包括：

- `result\AT\mvs.xml`
- `result\AT\mvs_undistort.xml`
- `result\milestones\mvs.pb.bin`
- `result\milestones\mvs_raw.pb.bin`
- `result\milestones\mvs_roi.pb.bin`
- `result\milestones\mvs_undistort.pb.bin`
- `result\.temp\pre.xml`
- `result\.temp\match\match_list_0.pb.bin`
- `result\report\report.json`
- `result\report\sfm_block_0_report.json`
- `result\thumbnail\camera_1_residual.png`
- `result\thumbnail\camera_2_residual.png`

本轮还保全了通常会在任务完成后清理的完整派生视图：

| 临时目录 | 文件数 | 大小 | 说明 |
|---|---:|---:|---|
| `.temp\undistort` | 1368 JPG | 1,183,589,056 B | 342 张鱼眼各 4 个派生视图 |
| `.temp\undistort_classify` | 1368 TIF | 12,919,758 B | 派生视图分类结果 |
| `milestones\classify` | 342 TIF | 约 4.65 MB | AT 前原始鱼眼分类/动态掩膜 |

## 4. 当前 AT 直接结果

- 注册图像：`342/342`
- 移除图像：`0`
- tie point：`19,029`
- 图像重投影 RMSE：`1.250889 px`
- POS 修正范数 P50/P90/P95/max：`20.969/37.847/45.341/53.652 mm`

优化后共享相机参数：

| Camera | focal(px) | cx | cy | k1 | k2 | k3 | k4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 786.605522 | 1451.619510 | 1454.705887 | 0.0777858563 | -0.0078366315 | -0.0049876137 | 0.0002799772 |
| 2 | 784.597008 | 1458.945609 | 1453.393572 | 0.0809606594 | -0.0135317213 | -0.0011936431 | -0.0005275200 |

## 5. 与 2026-08-27 相同输入运行的重复性比较

对照任务：`D:\mipmap-lite\1fad9647-f717-4cd4-9391-11f219d7e5d1\snow\snow-20260827`。

经 JSON 结构化比较，两次运行以下输入完全一致：

- 342 个照片路径及顺序；
- 342 个照片时间戳；
- 初始 POS、初始 orientation、`pos_sigma`；
- 每图 `pre_calib_param`；
- 两组相机 metadata；
- LAS metadata 与 LAS 路径；
- `resolution_level=1`；
- `remove_moving_object=true`。

只有工作目录和任务身份不同。尽管输入一致，AT 输出并非逐字节确定：

| 指标 | 2026-08-28 | 2026-08-27 | 差异 |
|---|---:|---:|---:|
| residual RMSE | 1.250889 px | 1.251524 px | -0.000635 px |
| tie point | 19,029 | 18,931 | +98 |
| scene area | 634.915955 | 627.369202 | +7.546753 |

342 张相机的优化结果运行间差异：

| 指标 | P50 | P90 | P95 | max |
|---|---:|---:|---:|---:|
| 相机中心差 | 0.572 mm | 0.996 mm | 1.146 mm | 2.060 mm |
| 旋转角差 | 0.00897° | 0.01439° | 0.01667° | 0.04183° |

内参也存在小幅随机差异：

- Camera 1：focal `-0.00048 px`，主点差 `(-0.0577,+0.1334) px`；
- Camera 2：focal `+0.02386 px`，主点差 `(+0.0352,-0.0277) px`；
- 两次 `mvs.xml`、`mvs.pb.bin`、report 的 SHA-256 均不同。

结论：相同输入的 MipMap AT 具有毫米级位置、约百分之一度旋转和少量 tie-point 的自然随机波动。后续不能用“文件不相同”或亚毫米差异宣称 LAS/时间戳影响了 AT。

## 6. 对 P1-1 实验设计的修正

后续每种条件至少需要重复两次；最好再补一个原始基线，使 baseline 有 3 次运行。建议把“AT 对改动敏感”的判据设置为同时满足：

1. 相机中心差相对 baseline 分布显著放大，优先观察 P95 `>3.5 mm`；
2. 旋转差 P95 `>0.05°`；
3. 改动方向与相机变化具有系统相关性，而不是随机正负散布；
4. 图像 RMSE、tie point、内参变化和独立 LiDAR reprojection 指标共同支持；
5. 多次重复的组间差异大于组内差异。

阈值是基于当前两次重复的保守初始门槛，不是最终统计置信区间。

## 7. AT 是否消费 LiDAR：当前证据

### 直接证据

- 根 `task.json`、`at_task.json` 和 `block_sfm_task_0.json` 都携带 LAS 路径；
- AT 阶段输出只呈现照片、相机、特征、匹配、三角化和 POS/内参优化结果；
- `mvs.xml` 与 AT 报告没有 LiDAR residual、LiDAR 命中数或 LiDAR loss 字段；
- LiDAR 路径被传入 AT block task，不等于求解器实际读取或使用点坐标。

### 当前判定

**尚不能单凭原始基线确认 AT 是否消费 LiDAR。** 当前运行证据更像“视觉＋POS AT，LAS metadata 随完整任务下传”，但必须通过 LAS 平移 A/B 和 AT 期间的文件访问跟踪确认，不能把缺少报告字段直接当作未使用证明。

## 8. 其他重要观察

1. UI 截图显示 `Ultra High`，但 `info.json.resolution_level=1`，引擎明文日志写的是 `resolution level: high (1)`。本轮再次没有形成真正的 Ultra High 参数运行。
2. AT 前已经生成 342 张分类 TIF，支持动态掩膜在 AT/匹配前生效。
3. 1368 个去畸变 JPG 和 1368 个分类 TIF 已完整保存，为后续研究 Face4 投影、分类语义、Tile crop、图像梯度和 visibility 提供了不再依赖任务临时目录的原始证据。
4. 3D 只运行到 5%，没有进入 Tile Gaussian 长训练；该终态不能用于比较最终模型质量。

## 9. 下一步

按价值排序：

1. 在 AT 期间使用只读文件访问跟踪，记录引擎是否打开并读取 LAS，以及发生在预检还是求解阶段；
2. 使用隔离副本把 LAS 仅沿 X 整体平移 `+0.25 m`，照片、POS、时间戳和相机参数保持不变，仅运行到 AT 完成；
3. 对平移 LAS 条件至少重复两次，并与 baseline 组内噪声比较；
4. 再分别构造双相机时间戳 `+20/-20/+50/-50 ms` 条件；时间偏移必须保持照片文件和 POS 空间值不变；
5. 每轮 AT 完成立即镜像同一组文件并停止，不进入完整 3D 长训练。
