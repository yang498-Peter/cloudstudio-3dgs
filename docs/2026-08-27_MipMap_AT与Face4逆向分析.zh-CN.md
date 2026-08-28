# MipMap AT 与 Face4 逆向分析

日期：2026-08-27
数据：`snow-20260827`，342 张 2912×2912 双鱼眼影像
证据：`task.json`、`report/report.json`、`AT/mvs.xml`、`AT/mvs_undistort.xml` 与 1368 张实际派生 JPEG

## 1. 结论

MipMap 的空三和分面是两个先后阶段，不应混为一体：

1. 它先在 342 张原始完整鱼眼影像上做特征匹配、三角化和 AT/BA；
2. AT 同时优化每张影像的独立位姿，以及左右相机各自共享的鱼眼内参；
3. AT 完成后，才从每张优化后的实体鱼眼相机确定性派生 4 个零畸变视图；
4. 4-face 视图进入后续 MVS、深度和 Splat 流程，本身不是新的自由位姿变量。

因此 CloudStudio 不应先把 Face11/Face4 喂给 AT。正确对位路线是：

`raw fisheye AT -> corrected physical cameras -> Face4 undistortion -> MVS/training`

## 2. AT 前后位姿

### 2.1 坐标和符号已闭合

`task.json` 是 AT 前输入，`mvs.xml` 是 AT 后结果。XML 的旋转矩阵是 world-to-camera。

对每张影像：

```text
corrected_position = raw_position - report.pos_diff
mvs_center = corrected_position + gauge_translation
gauge_translation = (-2.767020873826, 4.222884545541, -1.415040891251) m
```

342 帧的固定 gauge translation 最大离差为 `7.55e-10 m`，所以 `pos_diff` 的方向和 XML 坐标关系均已确定，不是估计。

### 2.2 位置改正量

| 范围 | 均值 | P50 | P90 | P95 | 最大 |
|---|---:|---:|---:|---:|---:|
| 全部 342 帧 | 2.25 cm | 2.10 cm | 3.78 cm | 4.55 cm | 5.44 cm |
| 左相机 171 帧 | 2.31 cm | 2.12 cm | 3.75 cm | 4.17 cm | 5.03 cm |
| 右相机 171 帧 | 2.18 cm | 2.09 cm | 3.97 cm | 4.65 cm | 5.44 cm |

这个量级与 CloudStudio 先前 snow BA 的 P50 约 2.37 cm、P95 约 4.38 cm 接近，说明 2 cm 级 POS 先验方向正确。

### 2.3 姿态改正量

姿态改正按 `R_AT * R_raw^T` 的旋转角统计：

| 范围 | 均值 | P50 | P90 | P95 | 最大 |
|---|---:|---:|---:|---:|---:|
| 全部 342 帧 | 0.480° | 0.429° | 0.823° | 1.010° | 1.915° |
| 左相机 | 0.543° | 0.506° | 0.843° | 1.041° | 1.872° |
| 右相机 | 0.416° | 0.354° | 0.680° | 0.890° | 1.915° |

因此产品做的不是纯 XYZ 平移修正，而是完整逐影像 SE(3) 优化。

### 2.4 它没有使用硬固定双目 Rig

171 对左右影像的时间差很小，P50 约 `0.004 ms`、最大 `0.069 ms`。原始左右相机中心距离稳定在约 `76.80 mm`，标准差只有约 `0.005 mm`。

AT 后左右中心距离却变为：

- P50：92.08 mm；
- P95：113.44 mm；
- 范围：42.47–143.95 mm；
- 相对旋转相对输入变化 P50 0.373°，最大 0.985°。

这直接证明其优化变量是独立影像位姿，至少没有硬性固定左右相机外参。对 CloudStudio 的启示不是立刻放弃 Rig，而是必须增加三组对照：硬 Rig、软 Rig 先验、完全独立位姿。此前 Stage 2 不收敛，很可能与硬约束过强有关。

## 3. 鱼眼内参变化

`constant_parameters=[]`，两台相机的共享内参都允许优化。

| 相机 | 参数 | AT 前 | AT 后 | 变化 |
|---|---|---:|---:|---:|
| 左 | f(px) | 786.6352 | 786.6060 | -0.0292 |
| 左 | cx | 1454.3578 | 1451.6772 | -2.6806 px |
| 左 | cy | 1452.2178 | 1454.5725 | +2.3547 px |
| 右 | f(px) | 786.4943 | 784.5731 | -1.9211 |
| 右 | cx | 1454.6488 | 1458.9104 | +4.2617 px |
| 右 | cy | 1449.1511 | 1453.4213 | +4.2702 px |

KB4 的 k1–k4 也都发生小幅变化。右相机的焦距和主点调整明显大于左相机，说明只优化 pose、不优化共享鱼眼内参，无法完全对位竞品流程。

## 4. 四个虚拟视图的精确几何

`mvs_undistort.xml` 有 8 个 Photogroup、1368 张图：两台实体相机各 4 组，每组 171 张。每个派生视图与父相机中心完全相同，中心误差为 0；相对旋转在 171 帧上的最大离散仅约 `2.7e-6°`。

相机坐标采用 OpenCV 约定：+x 向右、+y 向下、+z 向前。

| Face | 光轴（父相机坐标） | 尺寸 | f(px) | pinhole FOV |
|---|---|---:|---:|---:|
| yaw -35° | `(-sin35°, 0, cos35°)` | 1456×2912 | 1039.691749 | 70.000°×108.941° |
| yaw +35° | `(+sin35°, 0, cos35°)` | 1456×2912 | 1039.691749 | 70.000°×108.941° |
| pitch up 56° | `(0, -sin56°, cos56°)` | 2912×1456 | 2308.921016 | 64.471°×35.000° |
| pitch down 56° | `(0, +sin56°, cos56°)` | 2912×1456 | 2308.921016 | 64.471°×35.000° |

四组主点都严格位于影像中心，k1–k4 全为 0。虽然 XML 标签仍写 `FishEye`，实际 JPEG 视觉和 `f = size / (2*tan(FOV/2))` 都证明它是零畸变透视子视图。

这是十字形覆盖，不是四个等尺寸 90° 方形 cubemap：

- 两个竖向大面覆盖中央左右区域；
- 两个横向窄面补充上、下区域；
- 对所有方位角完整覆盖到父光轴约 70°；
- 70°以外覆盖迅速下降，说明它主动丢弃鱼眼最外圈，而不是追求完整 190° 覆盖。

以源影像 1 为例，四个派生 ID 是 12、20、25、37。XML 没有显式 parent ID，但可通过“同中心 + 固定相对旋转”无歧义恢复全部 1368 项映射。

## 5. Face4 与 CloudStudio Face11 的差异

使用同一套 snow 左相机参数：

| 项目 | CloudStudio 当前 Face11 | MipMap Face4 |
|---|---:|---:|
| 每张鱼眼派生图数 | 11 | 4 |
| 每张鱼眼总派生像素 | 18,802,332 | 16,959,488 |
| 相机样本数下降 | 基准 | -63.6% |
| 像素量下降 | 基准 | -9.8% |
| 覆盖策略 | 完整约 190°，含极端边缘 | 完整到约 140°直径，主动舍弃外圈 |
| 形状 | 11 个方形面 | 2 竖 + 2 横矩形面 |

MipMap 的主要收益不只是像素减少，而是虚拟相机数从 11 降至 4、去掉高畸变外圈，并减少 visibility、采样和 epoch 语义膨胀。

当前 342 张原始 JPEG 共约 0.79 GiB；1368 张 Face4 JPEG 共约 1.10 GiB，只增加约 39%。CloudStudio 目前缓存 PNG、mask 和可选 depth，真实磁盘差距预计比像素比更大，应单独实测。

## 6. 建议的 AT 优化路线

### P0：先对齐竞品的 AT 变量和顺序

1. 继续直接在完整 KB4 鱼眼图上做特征匹配和 BA；
2. 342 张图全部使用独立 SE(3) 变量；
3. 每台实体相机共享一套 `f/cx/cy/k1..k4`，允许联合优化；
4. 位置使用 `sigma=(0.03, 0.03, 0.06)m` 的各向异性先验；
5. 左右 Rig 从硬约束改为可配置：`hard / soft / none`；
6. BA 收敛后才生成 Face4，四个 face 不再进入 BA 自由变量集合。

姿态先验在输入中存在初始矩阵，但没有显式 sigma；是否被加入目标函数无法从文件直接证明，实施时应做强度扫描而不是假设竞品配置。

### P1：实现精确 MipMap-style Face4

增加一个独立 face plan，固定使用本报告中的四个旋转、尺寸比例和 FOV。父相机使用 AT 后 pose 和 AT 后 KB4 内参完成重投影。

Face4 sample 必须保留 `parent_image_id`。训练 epoch、验证划分和采样权重按父影像计数，不能把 4 个 face 当作 4 次独立采集。

2026-08-27 已完成第一阶段实现：`plan_mipmap_face4()` 固化上述四组矩形虚拟相机几何，`build_face_cache.py --face-plan mipmap_face4` 可显式选择该计划，默认自适应完整 FoV 计划保持不变。几何、FOV、分辨率缩放与缓存兼容路径共通过 49 项相关测试（另 1 项外部真实清单测试按环境条件跳过）。真实雪堆全量缓存、训练和原始鱼眼回评仍是独立验收门槛，不能由单元测试替代。

### P2：三组受控实验

1. `raw-fisheye + 新 AT`：隔离 BA 改进；
2. `Face11 + 新 AT`：保留现有完整覆盖；
3. `MipMap Face4 + 新 AT`：验证外圈裁剪、样本数和 MVS/GS 可见性收益。

三组必须共用同一 corrected pose set、同一初始化点云、相同训练步数和相同原始鱼眼验证视角。

### P3：晋级门槛

- 注册率：342/342；
- 整体重投影 RMSE：目标不高于 1.5 px；
- POS correction：P95 不高于约 5 cm；
- 旋转 correction：P95 不高于约 1.1°；
- Solver 必须明确收敛，不能继续接受 `NO_CONVERGENCE`；
- 独立位姿模式必须同时报告双目 baseline 漂移，不能只看重投影误差；
- 最终使用原始鱼眼视角评估，不能只用派生 face 自证质量。

## 7. 证据边界

可以直接证明：逐影像 SE(3) 改正、共享鱼眼内参优化、Face4 精确几何、Face4 在 AT 后生成、非硬固定 Rig。

尚不能仅由 XML 证明：具体特征算法、BA robust loss、姿态先验权重、LiDAR 是否进入 AT 目标函数、后续 MVS 算法以及高斯优化损失。这些内容不能写成已确认事实。

## 8. 342 张 snow 对位 AT 实测

### 8.1 实现与输入

本轮没有使用派生 Face4 做 AT，完全在 342 张原始 `2912×2912` 鱼眼影像上运行：

- 两台物理相机各共享一套鱼眼内参；
- 每张影像保留独立 SE(3) 位姿，不建立硬双目 Rig；
- POS 先验使用 `sigma=(0.03, 0.03, 0.06)m`；
- 匹配图包含 1733 对：171 对同步双目、1348 对同侧时序、214 对同侧回环；
- 特征与匹配使用 ALIKED N16 + LightGlue，342 张覆盖完整；
- 全量三角化注册 342/342，得到 103269 个点、374704 次观测，初始平均重投影误差 1.60786 px。

对应实现文件：

- `cloudstudio_3dgs/ba/pycolmap_adapter.py`：独立位姿、共享内参、各向异性位置先验 BA；
- `tools/build_independent_at_pairs.py`：全量对位匹配图；
- `tools/run_independent_at.py`：两阶段 AT 和候选模型/报告输出；
- `tools/compare_independent_at_to_mipmap.py`：逐影像最终 pose 对位；
- `tests/test_pycolmap_ba.py`：独立位姿 BA 与非法 sigma 回归测试。

`python -m unittest tests.test_pycolmap_ba` 实测 9/9 通过。

### 8.2 竞品 tie-point 结构修正

直接按 `mvs.xml` 节点计数，最终 XML 中有 15725 个 `TiePoint`、111218 个 `Measurement`，平均每个 tie point 约 7.07 次观测。此前报告中的 18931 是竞品报告阶段统计，不能等同于最终 `mvs.xml` 的 tie-point 节点数。

我们的图更密：103269 个三维点、374704 次观测，但平均 track length 只有 3.63。这说明两者不仅求解器不同，对应点图的稀疏度和长轨迹结构也明显不同。

### 8.3 完整变量版 v22a/v22b

输出：`outputs/snow-20260224-full-20260825/ba/independent_at_mipmap_v22a/at`。

- pose + points 阶段：98 次迭代，`CONVERGENCE`；
- pose + points + 共享内参阶段：250 次达到上限，`NO_CONVERGENCE`；
- 将上限提高到 1000 后仍为 `NO_CONVERGENCE`，且结果几乎不再变化；
- 最终重投影 P50 0.91885 px，RMSE 1.70929 px；
- POS 修正均值 1.097 cm、P50 1.007 cm、P95 2.224 cm；
- 旋转修正均值 0.256°、P50 0.230°、P95 0.465°；
- 与竞品最终相机中心差 P50 1.169 cm、P95 2.716 cm；
- 与竞品最终旋转差 P50 0.261°、P95 0.553°。

完整变量版在最终 pose 上与竞品处于同一量级，但不能晋级，因为内参阶段没有明确收敛。继续增加迭代次数无效。

### 8.4 明确收敛的 pose-only v22c

输出：`outputs/snow-20260224-full-20260825/ba/independent_at_mipmap_v22a/at_pose_only_v22c`。

- 99 次迭代，`CONVERGENCE`；
- 重投影 P50 0.92165 px、RMSE 1.42378 px；
- 与竞品最终相机中心差 P50 1.191 cm、P95 2.746 cm；
- 与竞品最终旋转差 P50 0.348°、P95 0.621°。

该版本满足 342/342 注册、RMSE 不高于 1.5 px、POS/旋转修正范围和 Solver 明确收敛四项门槛，可作为当前可复现的 pose-only AT 候选。它没有复刻竞品的共享内参联合优化，因此不能宣称完整方法等价。

### 8.5 未完全重合的根因

竞品的物理鱼眼模型每台相机只有一个共享焦距 `f`，参数为 `f/cx/cy/k1..k4`。当前 PyCOLMAP 的 `OPENCV_FISHEYE` 使用 `fx/fy/cx/cy/k1..k4`，没有“单焦距 + KB4”内置模型。开放两个焦距后，焦距、主点和四阶畸变出现明显耦合，产生非正定步长并导致 1000 次仍不收敛。

因此本轮结论是：

- 独立 pose、共享物理相机内参、3/3/6 cm POS 先验、raw-fisheye 先 AT 后 Face4 的主路线已被复刻；
- 最终 pose 已达到厘米级、亚度级对位，但不是逐参数完全相同的解；
- pose-only 版本通过当前 AT 可用门槛；
- 完整竞品等价版仍需要实现受约束的单焦距 KB4 参数化，并重新验证显式收敛，不能用 `NO_CONVERGENCE` 候选进入签名训练 Manifest。

## 9. pose-only AT 的 raw-fisheye SH0 基线 v22e

2026-08-27 使用 v22c 候选生成了独立签名的 dataset、mask、person-mask、depth 与 evaluation manifest，并通过真实 Trainer 连接测试。正式基线保持 6,158,096 个 LiDAR 初始化高斯、SH0、2912×2912 原始鱼眼、无致密化，训练 2000 步：

- 输出：`training_independent_at_sh0_fullres_formal2000_v22e`；
- 训练耗时 4583.72 秒，峰值 VRAM 7,050,916,864 bytes；
- 最佳且交付 checkpoint 为 step 2000；
- 签名 run manifest SHA256：`a70cb8908645dc5f5da10ff9f2ee53d2947f74d196d48c09c29e7600bb20a586`；
- 36 视角原始鱼眼评估：PSNR 17.33573 dB、P10 15.18928 dB、SSIM 0.527811、深度 MAE 4.72780 m；
- 相对旧 B 路线同 step 2000：PSNR -0.24282 dB、P10 +0.06200 dB、SSIM +0.00956、深度 MAE +0.19796 m；
- 导出 PLY 保留全部 6,158,096 个高斯，418,750,945 bytes，SHA256 `8ddbdd238429efdcda9b5a34bc047d8346ef0b2cb2295ce22b41247a75b7a8ad`。

健康审计没有发现可见高斯中心距离 LiDAR 超过 30 cm，说明新 AT 没有引入明显 floater；但透明度 0.005–0.1 的雾状高斯占 27.44%，另有 470 个最长轴超过 0.5 m 的高不透明异常大高斯。可见高斯最长轴 P50 约 8.8 mm、P95 约 29.3 mm，明显窄于竞品 P95 约 89.6 mm。肉眼结果对应表现为近场雪堆和墙板结构可读，但墙面仍有半透明涂抹、屋檐和远景边缘发虚。

因此 v22e 证明 pose-only AT 能提高结构指标和弱视角下限，却不能单独解决透明拖影。下一受控变量固定为 MipMap-style Face4；训练继续使用 SH0 与同一 LiDAR 拓扑，原始鱼眼全分辨率验证不变。

## 10. 精确 Face4 训练验证 v22g/v22h

精确 Face4 缓存共包含 306 个父影像、1224 个派生样本，每个父影像固定四面且没有缺失；1177 个派生样本带 LiDAR depth。签名 Face Manifest SHA256 为
`92e65235e592fd2bbc331a6f818fd4e4e0bba0723d37ca132d4bd4967b3e0927`。几何严格采用竞品 XML 恢复值：左右面 `1456×2912`、焦距 `1039.691749 px`、yaw `±35°`；上下面 `2912×1456`、焦距 `2308.921016 px`、pitch `±56°`。

冷启动 `training_independent_at_mipmap_face4_sh0_formal4000_v22g` 跑满 4000 步、保持 6,158,096 个 SH0 高斯。最佳 Golden 在 step 3000：PSNR `14.6043 dB`、SSIM `0.50249`、深度 MAE `4.40854 m`；step 4000 的 36 帧原始鱼眼回评为 PSNR `14.57308 dB`、SSIM `0.51146`、深度 MAE `4.45824 m`。相对等像素预算 raw-fisheye 基线低约 `2.3 dB`，并出现蓝黄接缝、外圈覆盖不足，因此精确 Face4 冷启动不晋级。

为排除“Face4 只适合作为抛光阶段”，新增签名采样身份重绑定：它验证源/目标 Manifest、坐标系、初始化、运行时和所有参数形状，只复制模型与辅助参数，不恢复优化器、采样器或 RNG。`v22e@2000` 重绑定到 Face4 后运行 500 步低学习率抛光，`v22h` 选择 step 250：36 帧原始鱼眼 PSNR `17.35640 dB`、P10 `15.16723 dB`、SSIM `0.527621`、深度 MAE `4.70603 m`。相对 v22e 仅有 PSNR `+0.02067 dB`、深度改善 `2.18 cm`，但 P10 `-0.02206 dB`、SSIM `-0.000190`；薄雾高斯只减少 9825 个（约 `0.16` 个百分点）。该阶段安全但收益过小，不能作为最终质量突破。

## 11. 竞品尺度独立 sky 层 v22i

从未经 Face4 改写的 `v22e@2000` 增加 100,000 个确定性远场高斯：半径 `136.5 m`、初始尺度 `1.1 m`、初始透明度 `0.05`、SH0 RGB 约 `(0.68, 0.75, 0.93)`，下半球最低方向 z 为 `-0.4`。主层和 sky 使用签名边界，可分别导出；总数 6,258,096。

`training_v22e_sh0_competitor_sky100k_fullres_polish250_v22i` 在 `2912×2912` 原始鱼眼上优化 250 步，最佳 Golden 为 step 125。签名 run Manifest SHA256 为
`1e6309e9daaf8c01edd5a65e62e926784f3b933b43c025604e1a8e34c036a4b7`，峰值 VRAM `7,242,338,816 bytes`。最佳模型的 36 帧报告为：

- PSNR `17.38387 dB`、P10 `15.38460 dB`、SSIM `0.528287`；
- 深度 MAE `5.09255 m`、覆盖率均值 `99.782%`；
- 相对 v22e：PSNR `+0.04814 dB`、P10 `+0.19532 dB`、SSIM `+0.000476`，但深度 MAE 恶化 `0.36475 m`；
- 报告状态为 `PARTIAL`，唯一未运行项为 LPIPS，不能写成全指标完成。

主表面仍为 6,158,096 个高斯：最长轴 P50 `8.3 mm`、P95 `26.9 mm`，可见高斯最长轴 P50 `8.8 mm`、P95 `29.3 mm`；可见点距 LiDAR 超过 30 cm 的数量为 `0`。sky 层 100,000 个高斯的最长轴 P50 `1.101 m`、透明度 P50 `0.0498`。由于 sky 只带来小幅弱视角收益、肉眼主体几乎不变且深度明显退化，v22i 保留为分层导出/远景验证候选，不作为最终表面质量胜出。

导出文件：

- 完整：`snow_v22i_best125_combined.ply`，SHA256 `37d096437cabc327b7ef12039229efd5a0b59817cead506aea640f2bce2ad385`；
- 表面：`snow_v22i_best125_surface.ply`，SHA256 `f95905049b0e760030716745ab923df82235e128cddfa7266f75acb08d4e5e6d`；
- sky：`snow_v22i_best125_sky.ply`，SHA256 `af06af066bd65cadb4294cf32c571f0cafed015837bb777467b5e56eb64002cb`。

## 12. 全分辨率冻结几何收口 v22j

v22i 同时优化了几何，虽学习率很低，仍使深度指标退化。`v22j` 从同一未训练 sky 暖启动重新开始，继续使用全分辨率原始鱼眼，但冻结 means/scales/quats，只优化 SH0 颜色、不透明度与曝光 500 步。这一设计复用已验证的 B-H3 sky 收口语义，同时严格隔离“填远景/透明洞”与“移动 LiDAR 表面”两个机制。

运行 `training_v22e_sh0_competitor_sky100k_fullres_frozen_polish500_v22j` 完整结束并选择 step 500：

- 训练耗时 `1667.48 s`，峰值 VRAM `7,238,044,160 bytes`；
- 签名 run Manifest SHA256 `7ca813bd3cc57c6892693676ae2d64d26c72e2c56a23a9a0d0df74ab05f5e973`；
- 模型 SHA256 `273920a9094708b566fed39470819dde69fbce64e8e2e61ed01c565be69d98a3`；
- 36 帧正式质量报告状态 `COMPLETE`，SHA256 `84b2b82162747d573ac6171bef9b171c2df1e14baa72411a8bd6939fab906005`；
- PSNR `17.41438 dB`、P10 `15.56306 dB`、SSIM `0.528075`、LPIPS AlexNet `0.570268`；
- 深度 MAE `4.92163 m`，预测覆盖率均值 `99.775%`。

相对无 sky 的 v22e，v22j 的 PSNR 约 `+0.07865 dB`、P10 `+0.37378 dB`、SSIM `+0.000265`，但深度 MAE仍恶化约 `0.19383 m`。纯表面健康审计显示，雾状高斯从 1,689,718 个降到 1,421,530 个，比例从 `27.44%` 降到 `23.08%`；可见高斯增加到 4,700,777 个，距 LiDAR 超过 30 cm 的可见漂浮点仍为 `0`。同四视角、排除天空的锐度测试中，空洞从 `0.6%` 降到 `0.5%`，一致性从 `0.303` 升到 `0.311`，能量从 `0.389` 略降到 `0.382`。因此它确实改善透明洞和弱视角，不是靠移动表面或制造孔洞边缘，但提升仍有限，不能宣称全面超过成熟竞品。

最终 PLY：

- 完整产品候选：`snow_v22j_best500_combined.ply`，6,258,096 高斯，425,550,945 bytes，SHA256 `ac29c062bef0778bca1723b5913712c81ca10ad83065cfa28a597c59884e929a`；
- 纯表面：`snow_v22j_best500_surface.ply`，6,158,096 高斯，418,750,945 bytes，SHA256 `73320a8775965773efa0da60f213c4d27c082d7e8356d7ef120aff360e69b4f2`；
- 独立 sky：`snow_v22j_best500_sky.ply`，100,000 高斯，6,800,416 bytes，SHA256 `462b68aad048a474c1b07d2e1f711666fbbf80e58a8e66cbb314cd366092d1ae`。

## 13. 路线结论

本轮对竞品方向的结论不是“Face4 越多越好”，而是：raw-fisheye 独立 pose AT 建立可收敛基线，Face4 只适合极轻量抛光，独立低透明度 sky 用于远景，最终冻结 LiDAR 几何做外观收口。精确 Face4 冷启动已经实测失败；既有 `v21c` 增长实验虽新增 307,904 个高斯，500 步完整 PSNR 只有 `13.4136 dB`，因此在没有新的出生位置机制证据前不重复盲目生长。

当前新 AT 路线以 v22j combined PLY 为完整候选，以 surface PLY 为几何回退。它已完成训练、签名 Manifest、36 帧全分辨率评估、LPIPS、健康检查、纯表面空洞/锐度检查、分层 PLY 导出、PLY 复解析和哈希；外部独立 PLY Viewer 固定视角验收仍需单独执行，不能由训练渲染替代。

## 14. 竞品可执行调用链与算法模块核对

### 14.1 已确认的外部调用入口

竞品桌面端最终调用：

`D:\Program Files\MipMap\MipMapLite\resources\resources\catch3d\reconstruct_full_engine.exe`

该程序的帮助信息公开了 `-task_json`、`-reconstruct_type`、`-desktop_magic` 和 `-license_key` 参数。现有任务的运行记录表明：

- AT 阶段使用同一可执行文件并设置 `reconstruct_type=1`；
- 三维重建阶段设置 `reconstruct_type=2`；
- AT 配置为 `at_task.json`，三维配置为 `r3d_task.json`；
- 去畸变子任务固化在 `task/image_undistortion_task_0.json`，其中 `head_task=true`、`remove_moving_object=true`、`keep_undistort_images=false`、`resolution_level=1`；
- 插件路径由任务 JSON 指向用户目录中的 `extentions/gs_dlls` 和 `extentions/ml_dlls`。

`keep_undistort_images=false` 解释了为什么任务完成后只剩 `AT/mvs_undistort.xml`，而运行中生成的 `.temp/undistort/*.jpg` 和 `.temp/undistort_classify/*.tif` 被删除。

### 14.2 DLL 导入导出提供的直接证据

PE 导入表显示 `reconstruct_full_engine.exe` 直接依赖 `mipmap_engine.dll`、`mipmap_log.dll`、`mipmap_hardware.dll` 和 `mipmap_engine_util.dll`。关键导出如下：

- `mipmap_engine.dll`：`ReconstructAT`、`OptimizeAT`、`UndistortionTask`、`UndistortionPartionTask`、`ReconstructBlockTask`、`Reconstruct3D`；
- `mipmap_classify.dll`：`SegFormerSeg`、`TensorRTSeg`、`TensorRTMonoDepth`、`Infer`、`InferRaw`、`InferColor`、`SetConfidenceThreshold`、`MakeEngine`、`WarmUp`；
- `mipmap_gaussian_splat.dll`：`GSTraining::BatchTraning`、`InitialParameters(PointCloud)`、`GetMonoDepth`、`GetDepthRegularizerLoss`、`GetNormalLoss`、`GetNormalGradientLoss`、`GetScaleLoss`、`GetOpacityLoss`、`GetSingleViewLoss`、`AddNewGS`、`CloneGS`、`SplitGS`、`CullGS`、`CullGSRedundancy`、`RelocateGS`、`AfterTrainMCMC` 和 `TrainBackground`。

这些名称直接证明竞品二进制包含对应能力，但仅凭导出名称不能证明每项能力在 snow 任务中都被启用。能够由运行产物继续确认的是：

- `mipmap_classify.dll` 加载 TensorRT 10；
- `seg_v1.onx` 对应的 GPU TensorRT engine 在 Face4/去畸变阶段生成，证明该阶段实际运行了分割模型；
- `da2_v1.onx` 对应的 GPU TensorRT engine 在 Tile/高斯阶段开始时生成，证明该阶段实际运行了单目深度模型；
- 最终四个 Tile 的高斯数相对各自 LAS 初始化数既有减少也有增加，证明 snow 任务不是固定拓扑的全局 615.8 万点优化，而是发生了内容自适应的净增长/裁剪。仅凭最终点数还不能还原每次 `Add/Clone/Split/Cull/Relocate` 的具体调度。

模型容器位于 `C:\ProgramData\MipMap\MipMapLite\gdal_data`：`seg_v1.onx`、`seg_model_v1.onx`、`da2_v1.onx` 和 `voc.bin`。这些 `.onx` 不是普通 ONNX；ONNX Runtime 解析为 `InvalidProtobuf`，而 DLL 字符串包含 encrypted onnx 构建失败信息。因此当前只能通过竞品自身运行时做黑盒输入输出捕获，不应把它们当普通 ONNX 直接加载。

### 14.3 人物掩膜纠正

`milestones/classify/*.tif` 共 342 张、每张 `728×728`、只有 0/255 两个值。对 `1.tif` 与原始鱼眼图叠加后可见，它主要描述鱼眼有效圆和边界，并没有遮住画面左侧人物。因此不能再把这 342 张文件写成竞品人物/移动物体语义掩膜。

竞品真正的语义输出更可能是 Face4 后的 `.temp/undistort_classify/*.tif`：运行时数量为 1368，和 342×4 完全一致，但因 `keep_undistort_images=false` 已在成功收尾时删除。要确认其类别编码、人物覆盖率和形态学后处理，必须进行一次受控竞品重跑并在去畸变完成后保留该目录。

我们自己的 v22f/v22g 并非没有人物识别：

- `independent_at_training_person_masks_v22c/person_mask_manifest.json` 记录 342 张图、271 张有人、480 个实例；
- v22g 配置包含 `require_person_masks=true`；
- Dataset 以 `rgb_mask = valid_mask & static_mask` 排除动态区域；
- `tools/build_face_cache.py` 将这个组合掩膜与 RGB 一起投影到四个虚拟相机面；
- Face4 可视化复核确认人物轮廓已进入派生面无效区。

当前缺口不是“完全没做人影识别”，而是我们使用的是自己的检测器，且人工复核状态仍为 `PENDING`；尚未获得竞品 Face4 后分割模型的真实输出用于逐像素对位。

### 14.4 对当前失败原因的修正

v22g 只复刻了 Face4 几何、人物掩膜投影和 SH0 训练输入，没有复刻竞品完整的后半段。现有证据表明至少还缺：

1. 竞品共享单焦距 KB4 内参联合优化；当前 v22c 是 pose-only AT；
2. Face4 后的竞品语义分割输出与其阈值/类别合并规则；
3. `da2_v1` 单目深度及深度、法线、尺度、不透明度正则；
4. 按空间 Tile 选择相机和 LAS 子集，而不是一次全局训练；
5. 实际启用的高斯增长、克隆、分裂、裁剪、冗余裁剪和重定位调度；
6. 独立背景/天空训练与最后的 Tile/LOD 合并。

所以“Face4 冷启动低 2.3 dB”只能否定我们当时那套固定 LiDAR 拓扑的全局训练配置，不能据此否定竞品 Face4 路线本身。

### 14.5 下一次黑盒对位测试

下一次竞品测试应只改变中间产物保留项：从桌面端正常启动已授权任务，将 `keep_undistort_images` 设为 `true`，在 `undistort.done` 后复制 1368 张 Face4 JPG、1368 张 classify TIF、`mvs_undistort.xml` 和任务日志，然后停止后续 Tile 训练。验收项为：

- 派生图尺寸、焦距、yaw/pitch 与 XML 逐张一致；
- 语义 TIF 的尺寸、像素值集合、人物召回、非人物误杀和是否包含鱼眼有效区；
- 文件时间线与 `seg_v1` TensorRT engine 调用一致；
- 同一原图的四面 RGB 和掩膜可以由我们的 Face4 几何重投影到像素级对比。

该测试不需要解密模型，也不应绕过授权信息；它只捕获竞品正常执行时产生的输入输出。完成这一步后，才能决定是复刻其语义后处理，还是保留我们的检测器并补齐漏检区域。

## 15. 圆形有效区、人物掩膜前置与共享单焦距 KB4 AT 实测

### 15.1 问题现象与流程纠正

此前 HLoc 的 ALIKED 特征提取与 LightGlue 匹配没有加载掩膜，人物和鱼眼低质量外圈仍可能进入 tie point 图。仅在 3DGS 训练阶段排除人物不能修复已经被动态目标污染的 AT 位姿。因此本轮把掩膜前置到特征匹配之前，统一采用：

`AT 有效特征 = 半径 1200 px 的主点圆形有效区 & ~人物动态区`

1200 px 是 2912×2912 原图尺度；缩到 728×728 后与竞品 `milestones/classify/*.tif` 的圆形区域 IoU 分别为左 `0.9771`、右 `0.9789`。该掩膜只删除低质量外圈，不尝试拟合复杂边界。人物掩膜使用我们已有的 342 张逐图检测结果，并在图像 ID、源路径、源图 SHA256 和相机 ID 全部一致时才允许重绑定到 AT 后的新数据签名。

### 15.2 修改文件

- `cloudstudio_3dgs/data/mask_manifest.py`：增加按主点生成固定像素半径圆形有效区；
- `cloudstudio_3dgs/ba/hloc_mask_filter.py`、`tools/run_hloc_aliked_lightglue.py`：在 ALIKED 后、LightGlue 前过滤圆外和人物区域特征，并签名记录过滤身份；
- `cloudstudio_3dgs/ba/single_focal_kb4.py`、`tools/run_independent_at.py`：交替执行独立相机位姿/点 BA 与两组共享单焦距 KB4 内参优化；
- `tools/rebind_person_mask_base.py`：增加跨位姿数据清单的不可变图像身份验证后重绑定；
- `tests/test_image_sample.py`、`tests/test_pycolmap_ba.py`：覆盖圆形掩膜和单焦距 KB4 约束。

### 15.3 AT 结果

新的遮罩特征图从 918,786 个 ALIKED 特征中保留 713,607 个，删除 205,179 个（`22.3315%`）；LightGlue 重新匹配 1,733 对图像。三角化仍注册 342/342 张图，得到 75,112 个点和 285,555 个观测，平均轨迹长度 `3.8017`，说明过滤没有破坏全量覆盖。

`independent_at_circle_person_single_focal_kb4_v23e` 明确收敛：重投影误差 P50 从 `1.46036 px` 降到 `0.86844 px`，P95 为 `2.52094 px`；位置修正 P50 `1.008 cm`、P95 `2.316 cm`、最大 `3.240 cm`；旋转修正 P50 `0.2765°`、P95 `0.5214°`。最终左右相机分别为：

- left：`f=787.0446, cx=1454.3521, cy=1452.2099`；
- right：`f=786.5898, cx=1454.6595, cy=1449.1718`。

这与竞品相同的是“342 张原始鱼眼、逐图独立 pose、两组共享单焦距 KB4”的变量结构；数值没有伪装成完全一致。竞品的左右焦距为 `786.6060/784.5731`，其位姿修正 P50 为 `2.099 cm`，而本轮为 `1.008 cm`。差异仍可能来自特征/匹配器、圆形掩膜边界和竞品内部稳健核或先验权重，需用保留下来的竞品 tie point/Face4 中间产物继续对位。

### 15.4 1368 张 Face4 固化结果

通过验收的 AT candidate 已发布成签名训练数据集，SHA256 为 `735beec49ae0a48b08f0b70d5ae05645123e8e5fdfc8f8ce56236a28e3ef3e99`。重新绑定后的 train/val 图像 ID 与原始划分完全一致。按 `mipmap_face4` 固定几何生成：

- train：306×4=`1224` 张，0 跳过，Face Manifest SHA256 `307caccb62ec06da8041f2833f928654865b628bde2e51dd22f5d0cd70105b91`；
- val：36×4=`144` 张，0 跳过，Face Manifest SHA256 `5579bae10c9b2a3b80820221f0458d3a394fcf8b812a866bd030dc2efe4cd793`；
- 合计：342×4=`1368` 张，四个方向各 342 张。

人物最多的抽查帧 `img_51cfc781b27f6ab6d6f8b7dd` 在派生面 mask 中已形成对应黑色无监督区，证明人物掩膜不是只写清单，而是实际进入 Face4 训练输入。新 AT 对应的完整 LAS 深度现已覆盖 342/342 原图并独立验签；Face4 DA2 阶段流式把该深度重投影用于标定，不把旧深度清单混入本轮结果。

## 16. 2026-08-28 对比结论与流程门禁

完整对比后不能宣称当前 CloudStudio 雪堆结果整体优于竞品。v23e 的 AT、遮罩前置和签名链路已经达到同量级且更可审计，但竞品在 Face4 多类别语义、DA2、四 Tile 自适应高斯、独立天空及 SOG/LOD 交付上仍领先。双方 AT 的重投影 RMSE 分别为我方 `1.315863 px`、竞品 `1.251524 px`；我方与竞品最终相机中心差异 P50/P95 为 `1.340/3.038 cm`，旋转差异 P50/P95 为 `0.362/0.678°`。这些结果支持“路线接近”，不支持“结果相同”或“我方更好”。

已新增 `MIPMAP_ALIGNED_FACE4_PIPELINE_SOP.zh-CN.md`，把从输入、时间同步、原图遮罩、AT、Face4、renderer mask、深度、天空、Tile 到训练/评估/导出的 18 个阶段固定为不可跳过的连续序列。`cloudstudio_3dgs/pipeline/mipmap_gate.py` 和 Trainer 门禁会对 shared-single-focal KB4 数据 fail-closed：缺少签名门禁、缺少 Face4、阶段未完成、顺序改变或输入 SHA256 不匹配时均拒绝训练。

静态审计进一步确认 snow High/type-2 使用 `0.6 mean-L1 + 0.4 DSSIM`、`20×Tile视图数` 步和 gradient split/clone/cull/reset，本任务 MCMC 与 redundancy cull 均关闭。保留的 342 张 classify TIF 只有左右相机各一份固定二值图，不能冒充逐图多类别语义；训练 renderer 的可确认规则为 `(seg!=255)&(seg!=33)`，label 33 的类别名仍未知。

雪堆前半段的 DA2 结果保持不变：完整新 AT 深度覆盖 342/342 原图、绑定 `7,036,347` 点 LAS，342 个 NPZ 哈希全部通过。DA2 使用官方 Apache-2.0 Small 模型，按竞品 reciprocal 输出和 `a×mono+b` 的 `2000` 次 RANSAC/`1%`/`>5%`/OLS 规则逐 Face4 标定；train `1123/1224`、val `80/144` 通过，1368 个缓存 SHA 全通过且无非有限值。后续独立天空和 LiDAR Tile 计划已经在第 17 节补齐，当前门状态以该节为准。

## 17. 独立天空与 LiDAR 可见性分块落地（2026-08-28）

### 17.1 天空证据边界

旧 v22j 的 100k 固定球壳只是把确定颜色的远场高斯追加到表面检查点，不等于竞品 `TrainBackground` 的独立训练。本轮改为独立数据和模型：对每个 Face4，只接受 `Face4有效掩膜 & DA2尺度对齐有效 & 对齐深度>=30 m & 世界方向Z>=0` 的像素；单视图候选不足 1% 时整视图禁用，无效 DA2 对齐也直接禁用。该策略与竞品 `mono_depth<=0` 代理不完全相同，因为官方 DA2 在天空仍输出正值；差异已写入签名 Manifest，未伪装成完全复刻。

实测 train `438/1224`、val `17/144` 个视图提供天空/远景监督，总候选比例分别为 `1.4943%`、`0.3278%`。训练证据生成独立 `100,000` 个高斯的 SH degree 1 初始化；没有读取、追加或改写任何表面检查点。天空证据门签名为 `59a76d850d7eda323a15ced38149f7a243c43c247356c1e083c36f7b9a103803`。

### 17.2 为什么最终不用照片 AT 稀疏点切块

照片 AT 的 `75,112` 个稀疏点适合纯视觉流程，也是 MipMap 兼容无 LiDAR 输入的合理最低公分母；但雪堆已有完整 LiDAR。第一次用 AT 稀疏点自动取根包围盒时，离群点把范围扩展到 X `[-131.14,166.26]`、Y `[-127.35,182.79]` m，并得到 `Y→Y→X`，该结果被保留为诊断但不晋级。

正确实现是混合路线：

1. 完整 703 万点 LAS 头确定场景范围，20% padding 后为 X `[-57.233,65.306]`、Y `[-62.489,49.179]`、Z `[-5.537,18.319]` m；
2. 从每张原鱼眼的新 AT LiDAR z-buffer 均匀抽取 5000 个有效深度像素；
3. 用共享 KB4 内参把像素反投影成相机射线，用量测 range 恢复世界点；
4. 再用固定 Face4 虚拟相机投影，并与 Face4 人物/有效区 mask 相交；
5. 由这些真实 LiDAR 可见点同时决定空间负载、每块相机集合和每张图的 256 px 最小裁剪/128 px halo。

该过程产生 `1,710,000` 个 LiDAR 规划锚点、all/train `1,607,923/1,435,928` 个 Face4 观测。按实测可用显存约 `6.85 GiB` 并施加 `6.5 GiB` 保守上限，得到 5 个串行 Tile；主拓扑从 X 开始并在左右继续按 Y 分裂。5 Tile 不等于复刻失败：竞品 4 Tile 来自其 MVS 观测、ROI 和运行时预算；LiDAR 可见性更密，不能为了凑成 4 块丢弃负载证据。

### 17.3 修改与验证

新增正式模块 `adaptive_tiling.py`、`lidar_face4_observations.py`、`tile_scheduler.py` 和 `sky_background.py`，并新增天空/Tile 门禁推进工具。Tile 必须严格串行；训练配置可将 `cuda_empty_cache_interval_steps=2`，每块结束还必须 synchronize、清缓存并释放 Python 引用。`tiles.json.max_memory` 继续表示照片工作集估值，不冒充 CUDA 显存硬上限。

最终上游门文件为 `mipmap_aligned_training_ready_lidar_tiles_gate_v23q.json`，状态 `TRAINING_READY`、`training_allowed=true`，签名 `4cf63fa6e28c93e56c2aa74d744013cbfdf16c5af0a3375c3d4048a7041ce6ac`。这只允许进入 `tile_gaussian_training`；尚未代表训练质量、原鱼眼评估或最终 PLY/SOG/LOD 通过。
