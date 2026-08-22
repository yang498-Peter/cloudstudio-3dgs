# cloudstudio-3dgs 持续优化实施计划

更新时间：2026-08-21

## 1. 目标与执行规则

目标是把当前真实数据闭环原型逐步升级为可复现、可诊断、坐标正确、支持固定双鱼眼 Rig、可利用 LiDAR 深度监督并可产品化的 3DGS 管线。

每一阶段都遵循同一完成定义：

1. 只实现当前阶段的最小闭环，不提前混入后续能力。
2. 几何与投影改动先加入合成数值测试，再实现并跑绿。
3. 缺图、缺位姿、缺标定、缺 mask、缺深度或补丁不适用时必须失败，不允许静默跳过。
4. 新增第三方代码或模型前先核查代码和权重许可证，并更新 `NOTICE.md`。
5. 每阶段记录修改、验证结果和未完成的真实数据/GPU 门槛。
6. 精确暂存当前阶段文件，提交后推送；最终以本地和 `origin/master` 的 SHA 相等、分叉计数为 `0/0` 为同步依据。

## 2. 完整阶段路线

| 阶段 | 主要交付 | 自动化验收 | 仍需外部证据 |
|---|---|---|---|
| PR-00 可复现基线 | `pyproject.toml`、`uv.lock`、gsplat 锁、bootstrap/doctor、CPU CI、基线清单 | CPU 测试、Python 编译、补丁哈希与适用性 | 新机器从空环境安装、CUDA 编译、GPU smoke |
| PR-01 数据 Manifest | 正式 schema、S1 reader、输入哈希、原子写出 | 重复输入哈希一致；缺文件失败；中文/空格路径 | 真实记录可迁移检查 |
| PR-02 标定与 Rig | 读取 `calibration.json`、左右配对、固定外参、Rig 诊断 | 合成 Rig 平移/旋转误差小于 `1e-8` | 实际未配对帧和内参差异报告 |
| PR-03 定量 QA | 投影、边缘、时间、轨迹、RGB、范围和可见点统计 | 合成投影误差小于 `0.05 px`；阈值可配置 | 真实场景 overlay 与阈值标定 |
| PR-04 点云初始化 | 确定性 voxel、RGB 位深识别、预算守卫 | 重复输出一致；`init_points < cap_max` | 真实 LAS 覆盖率优于 stride |
| PR-05 逐图 mask | 每图 valid/static/depth mask 与统一 crop | factor 1/2/4、crop 像素一致；masked 指标测试 | 真实动态区域回放 |
| PR-07 LiDAR depth | KB4 range 投影、z-buffer、置信度和缓存哈希 | 合成平面/球面误差小于 `1 mm`；遮挡只留前表面 | 真实 LiDAR/图像边缘核验 |
| PR-08 正式评估 | Rig-aware split、masked PSNR/SSIM/LPIPS、质量报告 | 左右同 split；黑边不计入指标 | 固定 golden views 和 GPU 基准 |
| PR-09 位姿修正传播 | 关键帧 SE(3) 修正、鲁棒过滤、Rig 时间插值 | 合成轨迹恢复与无突跳测试 | LiDAR-edge、LPIPS、双边对比 |
| PR-10 Rig BA | 许可友好特征、时序/回环图、固定 Rig 分阶段 BA | Rig 基线固定、尺度不漂、参数边界失败 | 真实重投影 p50 改善至少 30% |
| PR-11 自有 Trainer | 直接调用 gsplat API，移除示例代码长期补丁依赖 | 合成鱼眼收敛、resume、mask/depth/split/坐标测试 | 与现有 smoke 无显著退化 |
| PR-12 Rig pose refinement | 每个 Rig frame 一个 SE(3) 增量与先验 | 左右基线不变、修正统计、无改善自动回退 | 真实训练消融 |
| PR-13 粗到细原图训练 | A/B/C 阶段、valid-aware crop、OOM 降级 | mask/depth crop 一致、checkpoint 跨分辨率 | 8GB 不 OOM、原图 LPIPS/清晰度改善 |
| PR-06 高级动态剔除 | 人/车/天空逐图 mask、鱼眼透视面映射、时序一致性 | 人工标注指标脚本、静态像素保护门 | 至少 50 张人工标注，static recall 不低于 95% |
| PR-14 曝光与颜色 | gain/bias、物理相机颜色矩阵，条件满足后评估 PPISP | 几何指标不退化、推理颜色策略明确 | masked CC-PSNR/LPIPS 和接缝评审 |
| PR-15 大场景瓦片 | core/halo、局部原点、任务队列、PLY/SPZ、场景清单 | 合成块拼接、单块加载/卸载 | 大场景无断层、浏览器内存释放 |
| 最终产品验收 | `inspect → prepare → pose → mask → depth → train → evaluate → export` | CLI 集成测试、运行 manifest 完整 | 真实 S1 数据、目标 GPU、Viewer 和坐标往返验收 |

## 3. 当前阶段记录：PR-00

### 问题现象

训练依赖上游工作树和手工步骤，缺少机器可读的上游提交/补丁锁、CPU 依赖锁、自动环境诊断和 CI。历史 handoff 记录了训练输入规模，但仓库内没有本次训练结果，不能把历史数字当成当前复验结果。

### 修改文件

- `pyproject.toml`、`uv.lock`
- `upstream/gsplat.lock.json`
- `scripts/bootstrap.ps1`、`scripts/doctor.ps1`
- `.github/workflows/unit-tests.yml`
- `configs/smoke_8gb.toml`
- `baselines/gs2_smoke.baseline.json`

### 修改内容

- 锁定 Python 3.12 CPU 工具依赖。
- 锁定 gsplat 提交、补丁路径与 SHA256、Torch/CUDA/编译架构契约。
- bootstrap 在任何构建前校验补丁哈希、上游提交和工作树，遇到不相关改动立即拒绝覆盖。
- doctor 输出 Python、Git、GPU、CUDA、Torch、gsplat、提交和补丁状态。
- GitHub Actions 运行 CPU 测试、源码编译、补丁哈希和针对锁定提交的适用性检查。
- smoke 配置明确禁用 world-space 归一化；基线结果明确标记为 GPU 未运行。
- 将历史 UTF-16 LE 补丁规范化为 UTF-8；旧编码会使 `git apply` 无法识别补丁。
- 现有 1,019,218 个初始化点超过 1,000,000 上限，基线被标记为阻塞，等待 PR-04 重新生成点云后再验收。

### 验证方式与当前状态

本阶段提交前必须运行：

```powershell
python -m unittest discover -s tests -v
python -m compileall -q converter tools tests
powershell -ExecutionPolicy Bypass -File scripts\doctor.ps1
git -C external\gsplat apply --reverse --check train\patches\gsplat-s1-fisheye-keep-distortion.patch
```

CPU 与静态门槛可在当前机器闭环；新建空环境 bootstrap、完整 CUDA 编译和 GPU 训练 smoke 必须分别记录真实结果。在这些外部门槛运行前，PR-00 只能声明“源码基线已锁定”，不能声明训练环境或画质验收通过。

远端首次 CPU CI 暴露了 Windows checkout 将补丁转为 CRLF、导致 SHA256 与本地 LF 文件不同的问题。
修复方式是在 `.gitattributes` 中强制 `*.patch` 使用 LF，并在单元测试中同时守卫 UTF-8、无 BOM、LF 和 SHA256；补丁适用性作业本身已通过。

## 4. 当前阶段记录：PR-01

### 问题现象

旧转换器直接从源目录生成 COLMAP 文件，没有一个稳定的数据身份层；图片身份依赖扁平化文件名，缺图时会打印并继续，输入内容、坐标语义、位姿来源和未完成的 Rig 配对也没有机器可读记录。

### 修改文件

- `cloudstudio_3dgs/data/schema.py`
- `cloudstudio_3dgs/data/s1_reader.py`
- `cloudstudio_3dgs/data/manifest.py`
- `tests/test_dataset_manifest.py`
- `README.md`

### 修改内容

- 从每条记录自己的 `info/calibration.json` 和 `ImgPose.txt` 建立确定性 Manifest。
- 每张有位姿图片保存稳定 ID、原始相对路径、相机侧、纳秒时间戳、内容哈希、位姿来源、c2w 和后续 split/mask/depth 槽位；原始目录中没有位姿的图片单独列入 `unposed_images`。
- 路径使用逻辑根和 POSIX 相对路径，不写入机器绝对路径；缺图、重复图、危险路径、零四元数和缺点云均立即失败。
- 默认计算图片、标定、位姿和点云 SHA256；显式跳过大文件哈希时，必须把未计算状态写入 Manifest warnings。
- 输出目录非空时默认拒绝；`--force` 仅原子替换 Manifest，不删除目录内其他文件。
- PR-01 不虚构左右 Rig 配对，`rig_frame_id` 暂为空并写入 `rig_pairing_pending_pr02`；真实配对与外参统计由 PR-02 完成。

### 验证方式与当前状态

合成测试覆盖重复构建哈希一致、中文和空格路径、相对路径、缺图硬失败、未入位姿集图片显式报告、跳过哈希的持久警告、非空目录拒绝、原子替换且保留其他文件。

真实 gs2 只读检查确认原始目录共有 1,246 张图（左右各 623），`ImgPose.txt` 只有 1,238 条有效位姿（左右各 619），两台物理相机均被识别，另 8 张原图已列入 `unposed_images`。这与旧文档“1,246 张原图”并不矛盾，但说明原图数不能直接当成可训练位姿数。

默认完整内容哈希已在真实数据上执行：首次读取约 4.4GB 图片和 1,023,660,519 字节点云耗时 16.819 秒；同输入第二次耗时 4.937 秒，两个 Manifest SHA256 均为 `0e4d65b37fb682df97fb2957d4c1ac08b3e8bcaf3c680f68a74989967f56b822`，没有 `not_computed` 哈希警告。真实输出写在被 Git 忽略的 `outputs/pr01-real-manifest/`，未修改原始记录。

## 5. 当前阶段记录：PR-02

### 问题现象

全帧转换器此前从 `transforms.json` 每侧第一张关键帧复制内参，左右图片没有稳定的 `rig_frame_id`，也没有证明逐帧左右相对位姿与记录级标定外参一致。普通逐图 pose optimization 因而可能破坏物理基线。

### 修改文件

- `cloudstudio_3dgs/geometry/rig.py`
- `cloudstudio_3dgs/data/schema.py`
- `cloudstudio_3dgs/data/manifest.py`
- `cloudstudio_3dgs/data/s1_reader.py`
- `converter/s1_to_colmap.py`
- `tests/test_rig.py`
- `tests/test_dataset_manifest.py`
- `tests/test_s1_to_colmap.py`

### 修改内容

- 从每条记录自己的 `info/calibration.json` 建立 `camera_from_lidar` 左右固定外参，计算 `expected_right_to_left`。
- 以 50 ms 容差进行确定性一对一时间戳配对；同一对左右图片共享稳定 `rig_frame_id`，未配对项完整列出。
- 诊断输出配对数、时间差 p50/p95/max、相对平移/旋转误差、外参散布和左右内参差异。
- `transforms.json` 只用于内参差异检查和候选位姿证据，不再作为全帧物理相机内参来源。
- COLMAP 全帧出口改为记录级标定；真实 gs2 的 1,238 帧只产生两组物理相机内参。
- 旋转微小角误差采用稳定的 `atan2(sin, cos)` 计算，避免 `acos(trace)` 把微弧度误差量化成零。

### 验证方式与当前状态

非单位旋转和非零平移的合成 Rig 测试恢复误差均小于 `1e-12`，超出容差的图片保持显式未配对。真实 gs2 得到 619 对、左右未配对均为 0；时间差 p50/p95/max 为 4,096 / 16,128 / 121,088 ns；相对平移误差 p50/p95/max 为 2.903 µm / 12.467 µm / 66.183 µm；相对旋转误差 p50/p95/max 为 1.531 / 7.026 / 17.305 µrad。记录级标定与 `transforms.json` 内参最大绝对差为 `1.735e-18`，全帧转换器确认 1,238 帧、两组唯一内参。

这些结果证明本次真实数据内部的固定 Rig 契约一致，但尚不等同于外部靶场标定精度认证；训练中 Rig-aware pose refinement 仍属于 PR-12。

## 6. 当前阶段记录：PR-03

### 问题现象

旧 `reproject_check.py` 只能生成四种坐标约定的目视叠加图和图内点比例，不能量化 KB4 投影、时间同步、轨迹异常、Rig 误差、RGB 范围、点云范围、逐帧可见点数或 LiDAR 深度边缘到图像边缘距离，也没有 fail-closed 训练前门槛。

### 修改文件

- `cloudstudio_3dgs/geometry/kb4.py`
- `cloudstudio_3dgs/evaluation/data_qa.py`
- `configs/qa_default.json`
- `tests/test_kb4.py`
- `tests/test_data_qa.py`
- `pyproject.toml`、`uv.lock`
- `.github/workflows/unit-tests.yml`

### 修改内容

- 建立可测试的 KB4 正投影/反投影，支持超过 180° 镜头并显式限制量程和最大入射角。
- 全量流式统计 LAS 点数、XYZ 范围、RGB min/median/p99/max 和纯黑比例；确定性抽取 QA 投影点。
- 为所有有位姿图片输出可见 LiDAR 点数和比例；按左右相机均匀选择帧，使用图像梯度距离变换与 LiDAR 邻域 range 跳变计算边缘距离并生成叠加图。
- 输出左右图像间隔、配对时间差、Rig 轨迹速度/角速度、固定外参误差和标定/`transforms` 差异。
- 所有阈值来自 JSON 配置。默认失败返回退出码 2；只有 `--allow-qa-warning` 才返回 0，且 `status=WARNING_OVERRIDDEN`、`override_used=true` 持久写入报告。
- 输出 `qa/report.json`、`qa/report.html` 和 `qa/overlays/*.jpg`；远端 CI 编译范围扩大到正式 Python 包。

### 验证方式与当前状态

合成 KB4 1,000 点像素往返最大误差低于 `0.05 px`，合成 LAS 和十一个 fail-closed gate 有单元测试。真实 gs2 全量统计 28,435,004 个点，范围为 `[-39.1872,-70.2405,-5.9661]` 至 `[103.0488,78.4596,15.3523]`，RGB 纯黑比例为 0.004107；50,000 点用于 1,238 帧投影，6 张边缘叠加图均成功生成，默认十一个 gate 全部通过，最终复验耗时 10.573 秒。

真实关键指标：可见点比例 p50=0.6594，边缘距离 p50=4 px，配对最大差 0.121088 ms，帧间隔最大 500.134 ms，轨迹最大速度 1.028 m/s、最大角速度 67.769°/s。故意把最低可见比例设为 0.99 时，默认退出码为 2、报告为 `FAIL`；显式 override 后退出码为 0，报告为 `WARNING_OVERRIDDEN` 且保留失败 gate。叠加图已人工抽查，绿色图像边缘与红色 LiDAR range 边缘整体贴合；这仍不是外部靶标的绝对像素精度认证。

## 7. 当前阶段记录：PR-04

### 问题现象

旧 `subsample_las` 按 LAS 写入顺序固定步长抽样，空间覆盖依赖文件排序；同时无条件执行 `rgb16 >> 8`，会把以 0–255 写入 16 位字段的有效颜色压成黑色。历史训练数据集还包含 1,019,218 个初始化点，超过 1,000,000 的 Gaussian 上限。工具首次直接以 `python tools/build_lidar_init.py` 启动时还暴露了仓库包不在脚本搜索路径的问题，已在工具和共享转换入口中显式加入仓库根目录。

### 修改文件

- `cloudstudio_3dgs/data/point_cloud.py`
- `tools/build_lidar_init.py`
- `converter/s1_common.py`
- `converter/s1_to_nerfstudio.py`
- `configs/lidar_init_8gb.json`、`configs/smoke_8gb.toml`
- `baselines/gs2_lidar_init.baseline.json`
- `tests/test_point_cloud.py`
- `README.md`、`NOTICE.md`

### 修改内容

- 以局部坐标 LAS 的确定性 voxel-grid 代表点替换 stride；同一体素默认选择最接近体素中心的点，按 `seed` 稳定选出的 20% 体素改选更靠体素边界的点，作为不引入外部模型的几何边界保留策略。
- 自动体素尺寸以表面占用模型起步，通过全量扫描收敛到目标点数的 90%–100%；任何输出都必须同时满足 `point_count <= target_points` 和 `point_count < cap_max`，否则失败。
- 全量直方图判断 LAS RGB 是 8 位值写入 16 位字段，还是实际 16 位值；前者直接保留，后者以 `/257` 四舍五入映射到 8 位。
- 报告包含输入 SHA256、坐标语义、范围、RGB 分布、体素调参轮次、占用密度、黑色比例、输出数组 SHA256，以及与相同规模 stride 的占用体素覆盖率对照。
- 可选使用 SciPy `cKDTree` 计算局部 PCA 法向、特征值和协方差；法向符号按最大绝对分量固定，保证可复现。
- CLI 以临时文件加原子替换写出 `sparse_pc.ply`、`lidar_init_report.json` 和可选 `lidar_init_geometry.npz`，已有输出默认拒绝覆盖。

### 验证方式与当前状态

八项合成/配置测试覆盖逐数组确定性、自动预算、8/16 位 RGB、密度有序点云覆盖率、点数上限、已签入基线预算和 PCA 平面法向。真实 gs2 的 28,435,004 点、1,023,660,519 字节 `colorized.las` 在三轮自动调参后使用 0.119599914 m 体素生成 376,906 点；点数低于 400,000 目标并严格低于 1,000,000 上限。输入 RGB 判定为真实 16 位，输出纯黑比例为 0.008047。

在最终体素尺度下，voxel 输出覆盖全部 376,906 个源占用体素；相同规模的 stride 样本只覆盖 140,656 个，覆盖率为 37.3186%，voxel 提升 62.6814 个百分点。两个独立全量运行的 PLY SHA256 均为 `66fbe6205f5fb8ebde104ef13a3735a9a0d091607ffa9a67936c7586eafc42df`，报告也逐字节一致。真实输出位于 Git 忽略的 `outputs/pr04-real-lidar-init*/`，没有写入原始记录。

PR-04 只完成初始化数据质量闭环；新 PLY 尚未接入正式训练数据集，GPU 训练、画质和显存门槛仍为 `NOT_RUN`，因此旧 `gs2_smoke` 基线继续保持阻塞，不能升级为训练 GO。

## 8. 当前阶段记录：PR-05

### 问题现象

旧 `make_fisheye_masks.py` 按物理相机生成一张圆形 mask，再复制给该相机所有图片；它无法表示同一相机在不同时刻出现不同人物、车辆或深度有效区。仓库也没有一个数据加载入口能保证 image、valid/static mask、depth、confidence 共享同一裁剪窗口，factor 下的尺寸和 masked 指标同样没有测试。

### 修改文件

- `cloudstudio_3dgs/data/image_sample.py`
- `cloudstudio_3dgs/data/mask_manifest.py`
- `cloudstudio_3dgs/evaluation/image_metrics.py`
- `tools/build_per_image_masks.py`
- `tools/make_fisheye_masks.py`
- `tests/test_image_sample.py`
- `baselines/gs2_masks.baseline.json`
- `README.md`

### 修改内容

- 正式训练 mask 定义为 `fisheye_valid & static_mask & optional_depth_valid`。每张图片以 `image_id` 指向自己的 mask 路径，不允许相机级隐式回退；static 尚未生成时显式使用 identity policy，depth-valid 尚未生成时显式标记为 PR-07 待办。
- 从记录级 OPENCV_FISHEYE 标定和 95° 半视场角生成几何 valid mask，输出逐图 `mask_manifest.json`；Manifest 绑定 PR-01 数据集 SHA256，自身也带可校验 SHA256，重复 image ID、共享 combined-mask 路径、不安全路径和篡改均失败。
- `prepare_image_sample` 先以一个全分辨率 `CropWindow` 同步裁剪 image、valid/static/depth-valid、depth、confidence，再以 factor 1/2/4 同步缩放；mask 使用 nearest，RGB 使用 Lanczos，depth/confidence 使用 nearest。
- depth 存在时，非有限、非正 range、非有限/非正 confidence 与显式 depth-valid 会共同收紧最终 mask；缺文件、尺寸不一致、空指标区域和不整除 factor 的 crop 均失败。
- 新增 masked MSE/PSNR；黑边或动态区只要 mask 为 false 就不计入指标，零有效像素不得返回伪造分数。
- 旧 `make_fisheye_masks.py` 保留给历史 COLMAP 文本模型实验，但启动时明确警告其相机级复制结果不能作为逐图 mask 已接入的证据。

### 验证方式与当前状态

合成测试证明同一相机的两张图片可加载不同 static mask；factor 1/2/4 下 image、mask、depth、confidence 尺寸一致；factor 1 的 crop 对四类数据逐像素等于原数组切片；depth/confidence 失效像素按公式剔除；masked PSNR 忽略 mask 外误差且空 mask 失败；两次 mask Manifest 与 PNG 输出确定性一致，篡改 Manifest 会失败。

真实 gs2 的 PR-01 Manifest 共 1,238 张有位姿图片。本阶段生成 1,238 个唯一逐图路径和 1,238 个 PNG，总计 22,305,046 字节；左/右几何 valid 比例分别为 0.701814 和 0.720464。两次独立运行的内部 Manifest SHA256 都是 `ad6a814005cb4bd457563fc6093d08281707fee0ea88f1ab594cc71f64eb633b`，JSON 文件 SHA256 都是 `eb4fa2c9691634d30402a694964fb5c0f205388106ff4c795d503af5636ed2c4`。真实首图加载得到 factor 1/2/4 的 2912/1456/728 方形 image 与 mask，尺寸全部一致。

PR-05 只建立逐图数据结构和几何 valid 基线；真实人物/车辆/天空 static mask 属于 PR-06，逐图 LiDAR depth-valid 属于 PR-07，自有 Trainer 消费该契约属于 PR-11。真实动态区域回放和训练画质仍为 `NOT_RUN`，不能据此声明动态剔除或训练质量通过。

## 9. 当前阶段记录：PR-07

### 问题现象

旧流程只把 LiDAR 点写入没有 observation track 的 COLMAP `points3D.bin`，gsplat 示例 Trainer 的 track-based `depth_loss` 因而拿不到有效深度监督。仓库没有逐图 KB4 深度投影、前表面 z-buffer、confidence、combined mask 剔除、缓存身份或并行生成器，也无法证明缓存能与 PR-05 的 factor/crop 同时使用。

### 修改文件

- `cloudstudio_3dgs/geometry/lidar_projection.py`
- `cloudstudio_3dgs/data/depth_cache.py`
- `cloudstudio_3dgs/data/image_sample.py`
- `tools/build_depth_cache.py`
- `tests/test_lidar_projection.py`
- `tests/test_depth_cache.py`
- `baselines/gs2_depth_cache.baseline.json`
- `README.md`

### 修改内容

- 将局部坐标点变换到逐图相机坐标后使用 KB4 投影，深度语义固定为 Euclidean ray range；按四舍五入像素执行确定性 z-buffer，range 最小的前表面获胜，相同 range 再按源点索引稳定决胜。
- confidence 由投影点到像素中心的亚像素误差和同像素支持点数共同确定；缓存同时保留 source index、support count、range 和 confidence，便于后续诊断。
- combined mask 在缓存写出前应用，因此凡是 PR-06 标成动态/天空或 PR-07 标成 depth-invalid 的像素都不会留下监督。当前真实 mask 仍只有几何 valid，不能把合成剔除测试升级为真实动态剔除证据。
- 缓存使用确定性稀疏 NPZ，只保存有效 pixel index，避免为 2912² 图写入大量零；ZIP 时间戳、数组顺序、dtype 和压缩参数固定。`image_sample` 可直接把稀疏 range/confidence 还原，再与 image/mask 共用 factor 1/2/4 和 full-resolution crop。
- 全局 cache key 绑定算法版本、数据 Manifest、mask Manifest、点云 SHA256、实际点数、投影配置和选中图片 ID；任一输入或配置变化都会产生新 key。
- CPU 生成支持线程并行，但 worker 数不进入内容身份；不同 worker 数必须逐字节生成同一 Manifest 和 NPZ。
- 支持 PR-04 规范 PLY、NPY/NPZ、LAS，以及安装了 laspy 压缩后端时的 LAZ。超过 500 万点的 LAS 默认拒绝无界加载；可显式 `--max-points` 做确定性诊断抽样，正式路径优先使用 PR-04 voxel PLY，避免无界内存和 LAS 写入顺序重新影响产品缓存。超过 100,000 的全局/ECEF 尺度坐标会失败，必须先转换到 S1 局部坐标。

### 验证方式与当前状态

合成球面和 z=4 m 平面的 ray-range 最大误差均低于 1 mm；同一像素的 3/5/7 m 表面只留下 3 m；combined mask 为 false 的动态像素不产生缓存；稀疏缓存还原后在 factor 1/2/4 与 crop 下，image、mask、depth、confidence 尺寸完全一致。小型端到端测试还覆盖 PLY/LAS 输入、并行 1/2 worker 逐字节确定性、缓存读取、partial 状态和配置变更导致 cache key 变化。

真实 gs2 验证使用 PR-04 的 376,906 点 PLY，对 1,238 张图片中均匀选出的 12 张生成缓存。有效深度像素 min/p50/p95/max 为 66,384 / 170,916.5 / 225,255.8 / 226,248，12 个 NPZ 共 22,829,361 字节。4 worker 与 2 worker 两次运行的全部 NPZ 和 Manifest 逐字节一致；cache key 为 `aa6262d8382edf0c88ef3333237afacc81eae1a4af835ec2818d0d6d128122b6`，Manifest 文件 SHA256 为 `f6fa9565f46812d863b43fb95b449737bcb2e92692e42d85c75d46320a64dd34`。真实缓存首帧在 2704² crop 下以 factor 1/2/4 得到 2704/1352/676 方形的 image、mask、depth、confidence，空间尺寸一致。

本次真实结果明确为 `complete_dataset=false`：它只覆盖 12 帧，并使用 PR-04 voxel PLY 而不是 2,843 万点原始 LAS。全量 1,238 帧缓存、全分辨率 LAS 密度对比、真实动态/天空 mask 回放和 Trainer depth loss 均为 `NOT_RUN`；因此 PR-07 只能声明“深度缓存源码与部分真实数据闭环”，不能声明深度训练或画质验收通过。

## 10. 当前阶段记录：PR-08

### 问题现象

旧 gsplat 示例按排序后的单张图片序号执行 `index % test_every` 切分，不理解固定双鱼眼 Rig，也没有空间泄漏告警。2026-08-20 的 666 图历史 GPU 训练中，333 对近同步左右图有 84 对被拆到训练/验证两侧，且没有一对左右图同时进入验证集，因此其 PSNR/SSIM/LPIPS 只能作为旧流程阶段性参考，不能升级为正式 PR-08 评估。旧流程还会把鱼眼黑边计入指标，缺少统一的 LiDAR 深度误差、资源统计、固定视角和可审计 HTML 报告。

### 修改文件

- `cloudstudio_3dgs/evaluation/splits.py`
- `cloudstudio_3dgs/evaluation/image_metrics.py`
- `cloudstudio_3dgs/evaluation/quality_report.py`
- `tools/build_split_manifest.py`
- `tools/evaluate_run.py`
- `tests/test_splits.py`
- `tests/test_quality_metrics.py`
- `tests/test_quality_report.py`
- `baselines/gs2_evaluation.baseline.json`
- `NOTICE.md`
- `README.md`

### 修改内容

- 以完整 Rig Frame 为最小单元提供 deterministic temporal block、spatial block 和严格 manual 三种切分；每张图片必须属于唯一完整左右对，否则正式切分失败。
- split Manifest 绑定数据 Manifest SHA256，固定 golden Rig Frames，并统计每个验证位姿到最近训练位姿的距离；低于配置阈值时显式告警，不把相邻帧泄漏伪装成泛化能力。
- masked PSNR/SSIM/LPIPS 只聚合 combined mask 内像素，空 mask 失败；LPIPS 是可选依赖，未安装或未执行时保留 `NOT_RUN`。LiDAR 深度指标只接受 Euclidean ray range，并输出 confidence 加权 MAE/RMSE 和绝对误差 p95。
- 签名 run Manifest 必须完整且仅覆盖 validation 图片，禁止只挑高分图或混入训练图。所有 artifact 只能用 run root 下的安全相对路径，Windows 反斜杠路径穿越同样失败。
- 每次评估原子写出签名 `quality_report.json` 和自包含 `quality_report.html`，并为固定 golden views 生成 reference/render/masked-render 对比图；训练时长、显存峰值、Gaussian 数量和模型大小/哈希缺失时逐项标记 `NOT_RUN`，报告状态保持 `PARTIAL`。

### 验证方式与当前状态

合成测试覆盖左右同 split、temporal/spatial/manual 模式、切分篡改、训练/验证重叠、近邻泄漏告警、黑边不影响 PSNR/SSIM、masked LPIPS 空间聚合、LiDAR ray-range MAE/RMSE、路径越界、验证集挑图和 run/split 身份不一致。两次报告生成的 JSON、HTML 和 golden assets 逐字节一致。

真实 gs2 当前完整 Manifest SHA256 为 `54c01abedb8be28d839d4ee9685de63c88059efc129eceacd501fa9943563ee3`，包含 619 个 Rig Frame / 1,238 张有位姿图片。默认 temporal block 切分两次生成的文件 SHA256 均为 `b9886ad4c0bb4d3dfc1503e93722d75ec6a93c7772a32f8b0e85860851d3f99e`：train 为 557 Rig / 1,114 图，validation 为 62 Rig / 124 图，固定 8 个 golden Rig。最近训练位姿距离 min/p50/p95 为 0.0973 / 0.3516 / 0.9434 m，0.25 m 阈值下有 3 个显式泄漏告警。

现有 3,000 步 GPU 结果来自另一份 666 图数据集和旧单图切分，不能与本次 split Manifest 绑定。按新 split 重训、124 张 validation 的 masked LPIPS、全量 LiDAR depth、训练峰值显存和正式 `quality_report.html` 均为 `NOT_RUN`；因此 PR-08 只声明“正式评估契约、真实 Rig 切分和合成报告闭环”，不声明 GPU 画质验收通过。

## 11. 当前阶段记录：PR-09

### 问题现象

`ImgPose.txt` 覆盖全时间轴，但 `transforms.json` 只包含 solver 选择并优化过的关键帧。旧流程二选一：只用 transforms 会丢掉大量图片，只用 ImgPose 又会丢掉关键帧修正；普通逐相机插值还可能破坏固定双鱼眼基线。仓库没有同名匹配、OpenGL/OpenCV 轴向对齐、SE(3) 修正、鲁棒异常过滤、Rig 级时间插值、非破坏性 pose set、曲线可视化或“画质未改善则继续使用 ImgPose”的默认位姿门。

### 修改文件

- `cloudstudio_3dgs/poses/__init__.py`
- `cloudstudio_3dgs/poses/keyframe_correction.py`
- `tools/build_corrected_pose_set.py`
- `tests/test_keyframe_correction.py`
- `baselines/gs2_pose_correction.baseline.json`
- `README.md`

### 修改内容

- 将 transforms 的 c2w/OpenGL 关键帧转成 c2w/OpenCV，与 Manifest 中同名 ImgPose 严格匹配；未知、重复、未配对或只含单眼的关键帧失败。
- 对同一关键 Rig 的左右两眼分别计算 `target_c2w @ inverse(imgpose_c2w)`，检查两份修正的一致性后进行确定性 quaternion/translation 融合。
- 用相邻关键帧 SE(3) 插值残差、MAD 和绝对下限过滤异常 anchor；通过的 Rig 修正沿完整时间轴执行 translation 线性插值和 rotation SLERP，首尾采用常量外推。
- 同一个左乘 SE(3) 修正同时应用于一个 Rig Frame 的左右图片，因此逐帧 left/right 相对外参保持不变。输出独立、签名的 `pose_set_manifest.json`，不会覆写数据 Manifest、ImgPose 或 transforms。
- 同时输出 `pose_correction_curve.svg` 和 `pose_correction_report.html`，报告平移/旋转修正 p50/p95/max、逐帧步长、异常 anchor、Rig 基线漂移和默认位姿 Gate。
- 默认位姿只有在 LiDAR-edge 误差严格下降、低分辨率 LPIPS 严格下降、建筑双边指标不恶化、修正曲线无突跳四项全部 PASS 时才切到 `keyframe_corrected`；任一 `FAIL` 或 `NOT_RUN` 都自动保留 `imgpose`。

### 验证方式与当前状态

合成测试覆盖线性平移+旋转轨迹精确恢复、SLERP、中间异常 anchor 剔除、左右固定基线、缺单眼关键帧失败、签名篡改失败、报告逐字节确定性，以及四项门槛全部改善/单项退化时的默认位姿选择。

真实 gs2 的 transforms SHA256 为 `7ca0d7d6e6b28a55d90b615682869b2fd59690a5f69e4a848e39f4f7d986b869`。174 张关键帧全部同名匹配，形成 87 个完整关键 Rig；左右修正最大分歧为 0.03045 mm / 0.000318°。鲁棒过滤拒绝 2 个平移异常和 1 个旋转异常，保留 84 个 anchor，传播得到 619 Rig / 1,238 张修正位姿。修正平移 p50/p95/max 为 0.0302 / 0.1572 / 0.2374 m，旋转为 0.1556 / 0.3895 / 0.7456°；Rig 基线平移/旋转最大漂移仅 `2.26e-14 m` / `2.29e-14°`。两次运行的 JSON/SVG/HTML 均逐字节一致，Manifest 文件 SHA256 为 `4777af5c6bfc3cf9ac2c17432fd5d623da6d28786a80bde86fc635a008d0aaa0`。

当前最大逐帧修正步长为 0.053812 m，超过默认 0.05 m 无突跳阈值，曲线 Gate 为 `FAIL`；新位姿的 LiDAR-edge、低分辨率 LPIPS、建筑双边对比均为 `NOT_RUN`。因此默认位姿明确保持 `imgpose`，本阶段只声明“候选位姿生成、传播、诊断和自动回退闭环”，不声明修正位姿优于原位姿。

## 12. 当前阶段记录：PR-10

### 问题现象

现有 S1 管线有 POS 初值和固定双鱼眼标定，但没有只使用训练集的特征匹配图、许可与版本锁定、已知位姿三角化、固定 Rig BA 或可审计的候选发布门。若直接逐图 BA，左右物理基线可能漂移；若把 validation 图用于特征或约束，又会污染 PR-08 的正式评估。普通“优化成功”也不能证明重投影、尺度和内参变化在允许范围内。

### 修改文件

- `cloudstudio_3dgs/ba/match_graph.py`
- `cloudstudio_3dgs/ba/runtime_lock.py`
- `cloudstudio_3dgs/ba/pycolmap_adapter.py`
- `cloudstudio_3dgs/ba/report.py`
- `tools/build_ba_match_graph.py`
- `tools/run_hloc_aliked_lightglue.py`
- `tools/run_hloc_triangulation.py`
- `tools/run_rig_ba.py`
- `upstream/rig_ba.lock.json`
- `tests/test_ba_match_graph.py`
- `tests/test_ba_runtime_lock.py`
- `tests/test_pycolmap_ba.py`
- `tests/test_ba_report.py`
- `baselines/gs2_rig_ba.baseline.json`
- `README.md`、`NOTICE.md`

### 修改内容

- 由 PR-08 split 建立确定性、仅训练集的匹配图，包含同 Rig 左右双目、同侧时序邻居和有最小帧间隔的空间回环；validation 图片进入任一 pair 都立即失败。
- 锁定 HLoc、LightGlue、ALIKED 的精确上游提交和许可证，以及已测试 PyCOLMAP 版本。默认要求 VCS 安装能证明提交；普通 wheel 只有显式 `--allow-unverified-vcs` 才可运行，且证据保留为 `UNVERIFIED`。
- 特征入口固定使用 HLoc 官方 `aliked-n16` 与 `aliked+lightglue` 配置；三角化入口禁止跳过几何验证，并使用参考 POS 位姿做 epipolar 验证。
- 三角化前按 pairs 自动构建 train-only 已知位姿参考模型，避免 HLoc 因 validation 缺特征失败，也避免正式评估泄漏。
- 把两台物理相机安装为一个 PyCOLMAP Rig；所有 Rig frame 用 Manifest POS 初始化，并对每个训练图中心施加 Cartesian 位置先验。左相机为参考 sensor，右到左外参来自记录标定。
- BA 分三阶段：Stage 1 只优化 Rig frame 位姿；Stage 2 才允许 `fx/fy`；Stage 3 才允许尝试畸变优化。PyCOLMAP 对鱼眼额外参数只提供整体开关，因此发布门只接受 `k1/k2` 的有限变化，任何 `k3/k4` 变化都拒绝候选。sensor-from-rig 始终固定，首个 Rig frame 作为额外 gauge anchor，点云参与优化。
- PyCOLMAP 4.1.1 的 pose-prior 构造过程会先做 Sim(3) 对齐并可能触碰 Rig translation；运行器在构造前缓存标定外参，在构造后和求解后重新断言原始外参，最终 before/after 快照再独立核验基线与尺度。
- 候选发布要求 solver 可用、重投影 p50 改善至少 30%、Rig 平移/旋转漂移不超阈值、场景尺度漂移不超 0.5%、内参改动符合当前 stage。任一 Gate 失败，报告明确选择 `before`，不会把 candidate 当成发布模型。
- 原子写出带 SHA256 的运行 Manifest 与 JSON/HTML before/after 报告；特征 Manifest 绑定 pairs/features/matches 和运行时，三角化 Manifest 再绑定 train-only 模型，BA 最后核对模型目录哈希与匹配图生成的 pairs 哈希，避免跨运行串用产物。

### 验证方式与当前状态

合成 PyCOLMAP 测试实际调用 Ceres，并使用非共线双相机轨迹、位置先验和固定外参。Stage 1 的重投影 p50 从 7.668242 px 降到 0.010774 px，改善 99.8595%；Rig 外参最大平移/旋转漂移为 `1.11e-16 m` / `9.94e-17°`，场景尺度漂移为 `6.32e-04`，候选通过全部 Gate。另有测试覆盖 train-only 参考模型、匹配图确定性、validation 排除、未知 stage、尺度漂移、越权内参变化、报告签名和逐字节确定性。

真实 gs2 绑定 Manifest `54c01ab...3563ee3` 与 PR-08 split 后，得到 557 个训练 Rig / 1,114 张训练图，validation 使用数为 0。共生成 6,787 对：双目 557、左右时序各 2,218、左右空间回环各 897；签名图内同时固化完整 train/validation ID 集并由 verifier 复查。两次 JSON 和 HLoc pairs 输出均逐字节一致，文件 SHA256 分别为 `8e57cac4...f5e938`、`fba14a66...6a892`，内部匹配图 SHA256 为 `367e1d20...54ef6a`。

真实集成已在锁定 HLoc `c13273b...`、LightGlue `eb42fee...`、PyCOLMAP `4.1.1` 与 CUDA 运行时上完成。1114 张训练图和 6787 对匹配经已知位姿几何验证后，完整注册 1114 图并三角化得到 765,590 点、3,083,168 个观测。真实 Stage 2 BA 的重投影 p50 从 `1.592867 px` 降到 `1.078868 px`，改善 `32.2688%`，通过 30% 门；p95 从 `3.405874 px` 降到 `2.833360 px`，焦距最大相对变化 `0.0657%`，场景尺度漂移 `0.0053%`，Rig 外参漂移约 `1e-14` 量级，候选通过全部 Gate。Stage 3 虽继续小幅降低 p50，但改变了禁止发布的 `k3/k4`，因此 `camera_parameter_bounds=FAIL` 并选择 before；当前正式选择 Stage 2。

## 13. 当前阶段记录：PR-11

### 问题现象

历史 GPU 路径通过长期修改 gsplat 的 `examples/simple_trainer.py` 与 `examples/datasets/colmap.py` 保留鱼眼畸变，并以 `S1_KEEP_FISHEYE` 环境变量切换行为。该路径把产品数据契约耦合到上游示例代码；旧深度 loss 使用投影 z-depth，也不符合 PR-07 已锁定的 Euclidean ray range。原示例没有把 PR-05 逐图 mask、PR-08 Rig split、坐标身份、断点输入身份、完整 masked validation 和 peak VRAM 统一成一个可审计的自有运行 Manifest。

### 修改文件

- `cloudstudio_3dgs/training/dataset.py`
- `cloudstudio_3dgs/training/backend.py`
- `cloudstudio_3dgs/training/losses.py`
- `cloudstudio_3dgs/training/checkpoint.py`
- `cloudstudio_3dgs/training/contracts.py`
- `cloudstudio_3dgs/training/trainer.py`
- `tools/train_gsplat.py`
- `tools/run_synthetic_training_acceptance.py`
- `upstream/cloudstudio_trainer.lock.json`
- `tests/test_training.py`
- `baselines/gs2_trainer.baseline.json`
- `README.md`、`NOTICE.md`

### 修改内容

- 新 Dataset 只消费签名 dataset/mask/split/depth Manifest，原始图片和逐图 mask/depth artifact 首次读取时复核 SHA256；不 import gsplat 示例数据集。训练和验证必须来自显式 split，输入坐标必须是 `s1_local`。
- 原始 `OPENCV_FISHEYE` 的 `k1-k4` 直接传给 gsplat，渲染固定为 `camera_model=fisheye`、`with_ut=true`、`with_eval3d=true`。锁定版上游明确禁止 UT 与 packed 同时启用，因此自有适配器固定 `packed=false`，没有沿用示例默认。
- crop/factor 同步作用于 RGB、combined mask、range 和 confidence，裁剪后 `cx/cy` 平移、全部内参再按 factor 缩放。RGB mask 与稀疏 depth mask 分开保留，开启 LiDAR loss 不会把 RGB 监督错误缩窄到只有 LiDAR 像素。
- LiDAR 直接监督 `RGB-Ed` 的 expected hit distance，语义与 PR-07 的 Euclidean ray range 一致；loss 只聚合 combined mask、有效 range 和正 confidence 的交集。
- 自有 Trainer 直接创建 Gaussian 参数和逐参数 Adam，直接调用 MCMC Strategy；从未创建或 import Viewer。MCMC 细化和噪声调度进入签名配置，短合成收敛验收可显式关闭噪声，但正式默认仍保留上游完整 MCMC 调度。
- checkpoint 原子写出参数、逐参数 optimizer、MCMC state、采样器和 CPU/CUDA RNG；恢复前严格核对 dataset/mask/split/depth、crop/factor、坐标、训练配置、初始化 PLY 和 gsplat 运行时身份。MCMC 改变 Gaussian 数量后也会按 checkpoint shape 重建参数引用，再恢复 optimizer state。
- `coordinate_transform_manifest.json` 显式记录 `s1_local -> s1_local` 米制恒等变换和未做 normalization；训练结束保存完整 validation 的 reference/render/mask、可用时的 ray-range 与 LiDAR cache，签名 `run_manifest.json` 同时记录训练耗时、Gaussian 数、模型 hash 所需路径和 peak VRAM，可继续交给 PR-08 质量报告。
- 新 gsplat lock 不含 patch。运行时必须是精确版本的干净 VCS commit，普通 wheel 当前不作为可验证来源。新路径不读取 `S1_KEEP_FISHEYE`，也不依赖两份上游 example 文件。历史 patch 链仍保留为对照，不能充当 PR-11 新路径证据。

### 验证方式与当前状态

九项新增测试覆盖逐图原始 artifact 篡改失败、raw fisheye/crop/factor 内参、RGB 与稀疏 depth mask 分离、坐标 Manifest 签名、3DGUT/MCMC/无 Viewer 配置契约、无 patch lock、规范 PLY、基线开放门和 loss；checkpoint 测试还从不同 Gaussian shape 恢复参数，并拒绝身份变化。当前完整 CPU/可选 Torch 测试集通过，源码编译通过。

第一次远端 CPU CI 发现 Trainer lock 的基线使用了 Windows 工作区 CRLF 文件字节 SHA，而 Git checkout 在 CI 中是 LF，导致内容相同却错误失败。基线和测试现改为对解析后的规范 JSON 计算语义 SHA256，仍能检测字段篡改，同时不受 Git 换行策略影响；该修复不改变训练行为，也没有重跑 GPU 训练。

在锁定提交 `f2d1413...` 的独立干净 worktree 上，RTX 5070 Laptop GPU 实际执行 80 步 raw-fisheye 3DGUT CUDA 前向/反向与 `RGB-Ed` LiDAR loss。总 loss 从 `0.399251` 降到 `0.186986`，改善 `53.1658%`，best 为 `0.183097`；末步 LiDAR range L1 为 `0.131814 m`，训练 peak VRAM 为 `8,731,648 bytes`。完整两个 validation 视角生成 masked 质量报告，PSNR mean `13.0758 dB`、SSIM mean `0.891985`、LiDAR range MAE/RMSE mean `0.105848/0.106903 m`；LPIPS 未执行，所以报告诚实保持 `PARTIAL`。签名 run Manifest SHA256 为 `68f1061a...be91dd`，质量报告 SHA256 为 `0549481c...7b9211`，详细证据锁入 `baselines/gs2_trainer.baseline.json`。

当前 Windows 安装可执行 3DGUT 渲染核，但短验收若启用 MCMC 位置噪声会因 `quat_scale_to_covar_preci_fwd` 未注册失败；本次 80 步验收因此把噪声停止步设为 0，完整 MCMC 噪声/致密化仍为 `NOT_RUN`，不能据此声明长程 MCMC 已通过。真实 gs2 与历史 smoke 同配置回归、中断式 GPU checkpoint resume、真实完整 validation 的 LPIPS/画质和 8GB 长程显存门也均为 `NOT_RUN`。本阶段完成的是自有 Trainer 源码契约与真实 CUDA 小型收敛闭环，不升级为真实客户数据训练 GO。

### PR-11 真实集成前置修复：深度监督完整性

#### 问题现象

真实 gs2 基线当前只有 `12/1238` 张深度缓存。此前 Trainer 在 `lidar_range_weight > 0` 时仍允许完全不提供 depth Manifest；提供 partial depth Manifest 时，缺失图片也会静默退化为纯 RGB 训练。这会让配置声称启用了 LiDAR range loss，但实际只有部分或零帧接受深度监督，违反“缺深度不得静默跳过”的数据契约。

#### 修改文件与修改内容

- `cloudstudio_3dgs/training/trainer.py`：正 LiDAR loss 权重必须同时提供 depth Manifest 和 depth root。
- `cloudstudio_3dgs/training/dataset.py`：只要启用 depth 输入，其 image ID 必须与完整 dataset 一致；同时逐图核对 camera ID 和 combined-mask SHA，拒绝错相机或错 mask 的缓存。
- `baselines/gs2_trainer.baseline.json`：把两个 fail-closed 门加入受版本控制的 Trainer 验收基线。
- `tests/test_training.py`：增加“正 depth 权重但无缓存”和“partial depth Manifest”两个独立红测试。

#### 验证方式与当前状态

修复前两个测试均按预期失败，证明旧实现确实会静默接受；修复后 PR-11 定向测试 `11/11`、完整测试集 `91/91` 通过，Python 源码编译与本次修改文件乱码检查通过。该代码修复本身没有启动 GPU 训练；随后按下一节单独完成全量 1238 图缓存，仍不据此升级真实训练结论。

### 真实集成门 A：当前 Manifest 的全量 LiDAR Depth Cache

#### 问题现象

PR-05/PR-07 的历史 mask 与 12 图 partial depth 绑定早期 dataset Manifest `0e4d65b3...`，而 PR-08 split、PR-09 pose candidate 和 PR-10 match graph 绑定加入正式 Rig/诊断字段后的当前 Manifest `54c01abe...`。虽然两版均包含 1238 个相同图像 ID，但签名身份不同，不能把旧 mask/depth 与当前 split/pose/BA 直接混合作为训练输入。

#### 修改文件与处理内容

- 由当前 Manifest 重新生成 1238 个逐图 mask，签名 `86ae782a...`。
- 使用 376,906 点 voxel PLY `66fbe620...`、当前 Manifest 和新 mask，不带 `--max-images` 生成完整 `1238/1238` Euclidean ray-range cache。
- 分别以 4 worker 和 2 worker 独立重放；两次 signed Manifest SHA、1238 条记录和实际 1238 个 `.npz` 文件逐个 SHA 均完全一致。
- `baselines/gs2_depth_cache.baseline.json` 从 partial 证据升级为 full 集成基线；`tests/test_depth_cache.py` 继续把真实 Trainer depth loss 保持为 `NOT_RUN`。

#### 验证方式与当前状态

完整缓存 `complete_dataset=true`，cache key 为 `d3bbed76...`，signed Manifest SHA 为 `3c114dfd...`，Manifest 文件 SHA 为 `747a68a9...`；1238 个 depth artifact 共 `2,112,535,370` bytes。有效像素数 min/p50/p95/max 为 `19,817 / 158,693.5 / 224,370.45 / 256,660`。Trainer 数据契约已无 GPU 地加载 `1114 train + 124 val`，共享同一 depth 身份。该阶段只证明真实全量深度输入完整、可重复、可装载；真实 CUDA depth loss、画质改善和 MCMC 仍未运行。

### 真实集成门 C：HLoc 中断续跑前置修复

#### 问题现象

真实 1114 张训练图 ALIKED 特征已全部写入约 1.36 GB H5，6787 对 LightGlue 匹配在 2840 对时因会话中断停止。HLoc 上游在 `overwrite=false` 时可跳过已存在的特征和匹配，但项目包装器此前只允许空输出目录或 `--overwrite`，因此无法利用上游续跑能力，只能重算全部特征。

#### 修改文件与修改内容

- `cloudstudio_3dgs/ba/hloc_artifacts.py`：增加 fail-closed 输出模式，只允许两个已知未签名 H5 半成品进入 resume；拒绝未知文件、缺 feature 的 match、目录项和已有最终运行时签名。
- `tools/run_hloc_aliked_lightglue.py`：增加与 `--overwrite` 互斥的 `--resume`，续跑时把 `overwrite=false` 传给上游 HLoc。
- `tests/test_hloc_artifacts.py`：覆盖 fresh、overwrite、合法 partial resume、未知 artifact、已签名完成目录和冲突模式。
- `README.md`：记录真实中断后的续跑命令和边界。

#### 验证方式与当前状态

新增定向测试与完整 CPU 回归共 `95/95` 通过，GitHub Actions run `32399041352` 通过。相同 image root、pairs 与输出目录执行 `--resume --require-cuda` 后明确跳过 1114 张已完成特征，只补算剩余 3947 对；续跑约 44 分钟后完整得到 `1114/1114` 个 feature group 和 `6787/6787` 个 match group。

最终签名 `feature_runtime_manifest.json` 的签名为 `e16b15de...f6ad`，文件 SHA 为 `b1a1142f...2b27`；feature H5 为 `1,357,797,328` bytes、SHA `403c0d2b...7b41`，match H5 为 `140,048,836` bytes、SHA `5f901202...5fd5`。Manifest 证明 `cuda_used=true`，HLoc `c13273b...`、LightGlue `eb42fee...` 与 PyCOLMAP `4.1.1` 均为 `PASS`。该步完成时只升级真实 ALIKED/LightGlue 门，三角化与 BA 当时仍为 `NOT_RUN`；随后由下一节单独执行和验收。

### 真实集成门 D：Manifest 参考模型、三角化与分阶段 Rig BA

#### 问题现象

已有 COLMAP 转换器会把 `left/xxx.jpg`、`right/xxx.jpg` 扁平化为 `left_xxx.jpg`、`right_xxx.jpg`，而 HLoc pairs 保留目录名。两者名称不一致时，train-only 参考模型会把全部训练图判为缺失；直接依赖尚未提交的转换器变更也不能形成可复现证据。

#### 修改文件与修改内容

- `cloudstudio_3dgs/ba/pycolmap_adapter.py`：从已签名 dataset Manifest 原生构建 1238 图已知位姿 PyCOLMAP 模型，严格要求 `c2w_opencv`、有限刚体位姿、有效相机模型和唯一规范化路径，并原样保留 HLoc 的 `left/...`、`right/...` 名称。
- `tools/run_hloc_triangulation.py`：增加与 `--reference-model` 互斥的 `--dataset-manifest`；三角化 Manifest 绑定 dataset 身份、文件 SHA、物化参考模型与来源策略。
- `tests/test_pycolmap_ba.py`：覆盖名称保留、位姿中心、两相机模型与错误 pose convention 的硬失败。
- `baselines/gs2_rig_ba.baseline.json`、`tests/test_ba_runtime_lock.py`、`README.md`：记录真实三角化、三阶段 BA、选中 Stage 2 与 Stage 3 回退。

#### 验证方式与当前状态

真实 Manifest 在内存构建出 `1238/1238` 个有位姿图像和 2 个相机，pairs 涉及的 1114 个名称缺失数为 0。三角化签名 `3010f470...3ef7`、模型 SHA `a68ebb3c...4687`，注册 `1114/1114` 图、765,590 点和 3,083,168 个观测，平均 track length `4.02718`、平均重投影误差 `1.55971 px`。

Stage 1 签名 `b91b8b74...e07b`，p50 改善 `32.0293%` 并通过；Stage 2 签名 `ca6bc08b...dfc9`，p50 改善 `32.2688%`、p95 改善，全部 Gate PASS，选中模型 SHA `64282ec6...1c04`。Stage 3 签名 `558be49c...3d4c`，右相机禁止发布的 `k3/k4` 最大变化 `5.353e-4`，`camera_parameter_bounds=FAIL`、`candidate_accepted=false`、`published=before`。因此 Stage 2 是当前真实 BA 候选；这不等于真实 3DGS 训练或画质验收，训练仍暂缓。

## 14. 当前阶段记录：PR-12

### 问题现象

PR-11 Trainer 固定使用 Manifest 位姿，无法在图像监督下对残余 Rig pose 误差做小范围联合优化。若直接给左右图片各自独立的 pose 参数，会破坏物理双目基线；若只判断训练是否结束而不比较原始/候选位姿，零改善或过大修正也可能被错误发布。旋转若直接绕全局原点应用，在离原点较远的局部坐标中还会把微小角度放大成伪平移。

### 修改文件

- `cloudstudio_3dgs/training/rig_pose.py`
- `cloudstudio_3dgs/training/dataset.py`
- `cloudstudio_3dgs/training/backend.py`
- `cloudstudio_3dgs/training/checkpoint.py`
- `cloudstudio_3dgs/training/trainer.py`
- `tests/test_rig_pose_refinement.py`
- `tests/test_training.py`
- `baselines/gs2_trainer.baseline.json`
- `README.md`

### 修改内容

- PR-12 为显式 opt-in。每个训练 Rig Frame 只创建一个 `[tx, ty, tz, rx, ry, rz]` 可微增量；同一增量同时左乘左右 `c2w`，validation Rig 不创建参数也不参与候选比较。
- 每个 Rig 以左右原始相机中心的均值作为旋转枢轴，因此旋转不会绕全局原点制造与场景坐标大小相关的平移；增量中的 translation 直接等于 Rig 中心位移。
- 增加平移和轴角旋转 L2 先验、独立 Adam 学习率、最大平移/旋转边界和最小损失改善阈值；所有字段进入 Trainer 签名契约与 checkpoint 身份。
- checkpoint 可选保存和恢复 pose 参数及其 optimizer state，不把 pose optimizer 交给只管理 Gaussian 的 MCMC Strategy。
- 训练结束后冻结 Gaussian，在确定性的训练 Rig 子集上分别用原始位姿与候选位姿计算同口径 RGB/LiDAR 监督损失。只有改善达到阈值且修正不越界才发布 refined；否则 pose 参数清零，最终 checkpoint 和签名 run Manifest 明确发布 original。
- 报告保存逐 Rig 候选平移/轴角、p50/p95/max、比较损失与三个 Gate；固定基线由“同一世界左乘修正”在构造上保证。

### 验证方式与当前状态

先加入缺模块红测试，确认旧代码无法满足 PR-12；实现后 `tests.test_rig_pose_refinement + tests.test_training` 共 `17/17` 通过，提交前完整测试集 `103/103` 通过且 Python 源码编译、当前阶段 diff 与 UTF-8 乱码检查均通过。CPU Torch 合成优化从零参数恢复已知 6DoF 修正，最终矩阵损失低于初值的 `1e-4`；大坐标 Rig 的左右相对位姿在 float32 数值容差内不变，Rig 中心位移等于显式 translation。测试还覆盖每 Rig 去重共享、未知 Rig 硬失败、无改善回退、越界回退、嵌套配置验证，以及 pose 参数和 optimizer 的 checkpoint 恢复。

本阶段没有启动真实训练。真实 gs2 的 on/off 消融、masked PSNR/SSIM/LPIPS、LiDAR range、清晰度与接缝评审均保持 `NOT_RUN`；在这些证据完成前，只声明 PR-12 源码与 CPU 合成闭环，不声明画质提升。

## 15. 当前阶段记录：PR-06 人影动态 mask 与 BA 残差审计

### 问题现象

已有逐图 mask 主要表达鱼眼有效视野，深度缓存另有 depth-valid；两者都不能证明人物已经被排除。若只在某次训练命令中临时识别人影，原始 POS 与 Stage 2 POS 的 A/B 可能使用不同遮挡条件，画质差异就无法归因于位姿。现有 Stage 2 BA 虽已通过全局重投影门，也没有证据说明少量高残差是否集中在人物、反光或其他运动物体上。

### 修改文件

- `cloudstudio_3dgs/data/person_masks.py`
- `cloudstudio_3dgs/data/torchvision_person.py`
- `cloudstudio_3dgs/data/image_sample.py`
- `cloudstudio_3dgs/training/dataset.py`
- `cloudstudio_3dgs/training/trainer.py`
- `cloudstudio_3dgs/ba/person_residual_audit.py`
- `tools/build_person_masks.py`
- `tools/audit_ba_person_residuals.py`
- `tools/finalize_person_mask_review.py`
- `tools/build_person_review_contact_sheets.py`
- `tools/repair_person_review_selection.py`
- `upstream/person_mask.lock.json`
- `tests/test_person_masks.py`、`tests/test_training.py`
- `baselines/gs2_person_masks.baseline.json`
- `README.md`、`NOTICE.md`

### 修改内容

- 人影层使用独立 `person_mask_manifest.json`，逐图绑定源图 SHA、base mask Manifest SHA、模型架构、运行时版本、官方权重 URL 与完整 SHA256；不改写既有 valid mask，也不重建已完成的 2.1 GB depth cache。
- 使用 TorchVision `maskrcnn_resnet50_fpn_v2` COCO_V1 的 person 类。仓库只登记官方 URL/哈希，不分发约 177 MB 权重；正式配置采用 800 像素推理、score 0.65、mask 0.5，并在 2912² 原图空间外扩 12 px。这个选择偏向覆盖动态边缘，而不是追求人物轮廓像素级精确。
- Trainer 在同一个 full-resolution crop/factor 前执行 `base_rgb_mask & ~person_dynamic_mask`；最终 depth mask 再与 depth-valid 相交，因此 RGB L1、masked SSIM 与 LiDAR ray-range 都不学习人影。正式训练默认缺 person Manifest 即失败，只有合成验收可显式声明不需要人物层。
- person Manifest SHA 进入 Dataset identity、checkpoint identity 和签名 run Manifest。后续原始 POS / Stage 2 POS A/B 必须引用同一个 SHA，不能各自重算 mask。
- BA 审计逐图流式加载 mask，对 Stage 2 模型的全部 3D observation 重算像素残差；默认把 `>=5 px` 视为高残差，只有高残差至少 20 个且其中至少 30% 落在人影上，才建议重跑 masked BA。工具同时输出人影内/外高残差叠加图，供人工识别反光或其他未建模动态物。
- 所有生成器先校验 dataset/base/person 签名与逐文件 SHA；partial person Manifest、路径逃逸、错相机、错源图、篡改 mask 和生产训练缺 person mask 均 fail-closed。已发布的 person Manifest 不允许 `--force` 就地覆盖，复算必须使用新目录。

### 验证方式与当前状态

先加入缺模块红测试；完整回归覆盖独立/确定性签名、模型 lock、RGB/depth 同步排除、crop/factor、partial/tampered Manifest、训练身份和 BA 决策阈值。真实生成完成 `1238/1238` 个唯一 mask，签名 `1eb2284f...c7c8`，共 12,134,080 bytes；954 张图检测到人物，左/右分别 468/486 张，共 2,057 个实例。单图人影比例 min/p50/p95/max 为 `0 / 0.002657 / 0.052954 / 0.113413`。

初始抽检选择因“高占比”和“均匀”样本重合得到 49 张，未把它冒充 50；修复选择器去重补位后只新增 1 张 overlay，不重跑 segmentation，最终左右各 25 张。两页 25 宫格由 Codex 逐图视觉检查：人物主体和近/远边缘均被覆盖，少量把镜头底部操作员衣物、手套或遮挡布一起屏蔽，属于安全过剔除；未见大面积墙面、地面或建筑误删。签名 review 为 `de421b44...0e685`、状态 `PASS`，reviewer 明确为 `codex_visual`；外部人工复核仍诚实保持 `NOT_RUN`。

Stage 2 模型 SHA `64282ec6...1c04` 的全部 1,114 张训练图完成流式审计。3,083,156 个可投影 observation 中，13,995 个位于人影内；`>=5 px` 的高残差共 2,921 个，只有 44 个位于人影内，重叠率 `1.506333%`，远低于 `30%` 重跑门。24 张最高残差叠加图中，紫色人影外残差主要分散在鹅卵石/砖缝、建筑高对比边缘和植被，没有形成随人物聚集的证据。签名审计 `a35f9aa6...ee92` 给出 `RETAIN_CURRENT_BA`，所以不为了 mask 重跑已通过的 Stage 2；masked BA 状态是 `NOT_RUN_NOT_RECOMMENDED`，不是跳过失败门。

本阶段仍未启动 3DGS 训练。下一步原始 POS / Stage 2 POS A/B 必须共同引用 person Manifest SHA `1eb2284f...c7c8`；masked PSNR/SSIM/LPIPS、LiDAR range、清晰度和接缝结论保持 `NOT_RUN`，不能由 mask/BA 审计推导画质提升。

## 16. 当前阶段记录：行业级 Gate 计划与 Gate 1A MCMC 运行证据

### 问题现象

2026-08-21 的《CloudStudio 3DGS 行业级算法深度调研、现状复盘与 Codex 实施计划》把后续路线从“按功能数量推进”调整为“按证据 Gate 逐项关闭”。当前最先要关闭的是完整 MCMC Windows CUDA runtime；历史 80 步合成运行明确关闭了位置噪声，只证明 3DGUT 前向/反向和最小 Trainer 可运行。机器安装脚本声称完成 full kernel build 也不能证明 covariance、noise、relocation、sample add、rasterization backward 与 checkpoint resume 在锁定运行时中全部成立。

### 修改文件

- `cloudstudio_3dgs/training/runtime_evidence.py`
- `cloudstudio_3dgs/training/backend.py`
- `cloudstudio_3dgs/training/trainer.py`
- `tools/audit_mcmc_runtime.py`
- `tests/test_mcmc_runtime.py`
- `baselines/full_mcmc_runtime.baseline.json`
- `README.md`

### 修改内容

- 新增完整 MCMC 注册门，必须同时满足干净锁定 gsplat commit、CUDA 可用、`3dgs/3dgut/reloc` build feature、`sample_add` Python API，以及 covariance forward/backward、UT projection、world-space 3DGUT rasterization、relocation 和 fused MCMC perturb 六项 native op。注册通过只允许写成 `PASS_REGISTERED`，不能替代 kernel execution 或画质结论。
- MCMC Strategy 每次 refine 边界记录调用前后的 Gaussian 数量、dead 数量、opacity 与实际 scale 的 min/p50/p95/max，并累计 relocate/add、refine 次数和 noise 调用步数；非 refine 步只累计紧凑计数，避免在数十万 Gaussian 上逐步做昂贵 quantile。
- Trainer 每步检查 loss，并在 checkpoint/refine 边界检查 gradient、参数以及指数化后的实际 scale；异常时在覆盖最新良好 checkpoint 前失败。MCMC telemetry 写入 checkpoint training state 和签名 run Manifest，full-MCMC resume 若缺 telemetry 会 fail-closed。
- 新增机器可执行的 `audit_mcmc_runtime.py`，输出不含本机绝对路径的确定性 SHA256 证据，并把尚未真正执行的 covariance、noise、relocation、add、rasterization backward、resume 和真实训练逐项保持 `NOT_RUN`。

### 验证方式与当前状态

先加入缺模块红测试，旧代码因 `runtime_evidence` 不存在按预期失败。实现后 `tests.test_mcmc_runtime + tests.test_training + tests.test_rig_pose_refinement` 共 `25/25` 通过，完整测试集 `117/117` 通过。

本机 RTX 5070 Laptop、Torch `2.11.0+cu128`、CUDA `12.8` 的实际注册审计结果为 `FAIL`：当前加载扩展只有 `3dgs=false, 3dgut=true, reloc=true`，缺少 `quat_scale_to_covar_preci_fwd`、`quat_scale_to_covar_preci_bwd` 和 `mcmc_perturb_positions`；同时外部 gsplat checkout 正包含另一台机器尚未提交的构建修订，源码身份门也因 dirty worktree 失败。证据 SHA256 为 `73415f98...4d38d`。因此 Gate 1 尚未关闭，非零噪声、真实 relocate/add、forward/backward 执行和中断恢复全部仍为 `NOT_RUN`；本阶段没有启动真实数据训练，也不声明画质提升。

## 17. 当前阶段记录：Gate 1B/1C 跨机 full-MCMC 与恢复等价性

### 问题现象

澳洲机器提交 `6473258` 增加 `--full-mcmc`，并报告在 RTX 5070 Ti、干净 gsplat `f2d1413` 上完成 80 步非零噪声训练，loss 改善 `87.40%`、Gaussian `24→29` 且无 NaN。该提交正确发现：8 点初始化下 `int(1.05N)` 永远不增长，以及 `means_lr=1e-8` 会把 `lr×noise_lr` 缩放后的噪声压到近零。但当时尚未合入 Gate 1A 的统一算子清单和 telemetry，也没有强制制造 dead Gaussian，因此只证明 add 路径发生，不能证明 relocation、fused perturb 注册或中断恢复一致性。

### 修改文件

- `cloudstudio_3dgs/training/checkpoint.py`
- `cloudstudio_3dgs/training/runtime_evidence.py`
- `cloudstudio_3dgs/training/trainer.py`
- `tools/audit_mcmc_runtime.py`
- `tools/run_synthetic_training_acceptance.py`
- `tests/test_mcmc_runtime.py`
- `README.md`

### 修改内容

- Trainer 新增仅供验收工具调用的受控中断入口：在指定步完成 optimizer、MCMC、telemetry 和原子 checkpoint 后抛出带 checkpoint 路径与完成步数的明确异常，不生成伪完整 run Manifest。
- 新增 checkpoint 全状态比较：递归核对 Gaussian 参数及索引顺序、optimizer、MCMC strategy、采样器、训练 telemetry、Rig 辅助状态、CPU/CUDA RNG、step 和 identity；浮点状态使用显式 `atol/rtol`，整数与 RNG 必须逐值一致。
- `audit_mcmc_runtime.py --execute-kernels` 在注册检查后实际执行 covariance forward/backward 和 fused `mcmc_perturb_positions`，要求输出与梯度有限且位置确实发生非零变化，避免把“算子名字存在”误写成“kernel 已执行”。
- `--resume-equivalence` 先运行连续参考，再在同一总步数配置下于中点受控停止并从 checkpoint 恢复；最终比较两个 checkpoint，而不是比较包含运行时长的 run Manifest。
- full-MCMC 验收后端只在合成验收中把一个初始 opacity 设到 `0.001`，确保 refine 窗口必须真实执行 relocation；24 点初始化继续保证 add 至少为 1。最终 PASS 同时要求 native kernel smoke、`PASS_REGISTERED`、noise 步数等于总步数、refine/relocate/add 均非零、最终 Gaussian 状态有限、loss 改善至少 20%，且 resume 全状态比较通过。

### 验证方式与当前状态

先加入缺少 `compare_checkpoint_payloads` 的红测试，旧代码按预期无法导入；实现后 Gate 1/Trainer/Rig pose 定向测试 `26/26`、完整测试集 `118/118` 通过，CLI、Python 编译与当前阶段 diff/UTF-8 检查通过。提交前以 `--execute-kernels` 重跑本机严格审计，结果仍为 `FAIL`：native smoke 明确记为 `NOT_RUN_INCOMPLETE_RUNTIME`，证据 SHA256 为 `10a5180a...0379`。本机 gsplat checkout 仍含共享未提交构建改动且已加载扩展缺关键算子，所以没有绕过 clean-lock 运行新的 GPU 合成验收；`relocation` 与 `resume_equivalence` 保持 `NOT_RUN`。下一次澳洲机器同步当前分支后，应运行 README 的 `--full-mcmc --resume-equivalence` 命令并提交签名 JSON，只有全部条件通过才能关闭 Gate 1。

## 18. 当前阶段记录：Gate 1D 实际噪声与签名 Exit Gate 证据

### 问题现象

现有 telemetry 的 `noise_injection_step_count` 只证明配置使 MCMC noise 分支被调用，不能证明训练中的 Gaussian position 发生过非零位移；旧的 `synthetic_acceptance.json` 也没有把运行时来源、kernel smoke、refine 分布、resume 比较和每项 Exit Gate 绑定成一个可独立校验的签名证据。另一台 GPU 机器即使终端显示训练成功，仍可能因零幅度 noise、漏跑 resume 或 JSON 被修改而被误升级为 Gate 1 PASS。

### 修改文件

- `cloudstudio_3dgs/training/runtime_evidence.py`
- `cloudstudio_3dgs/training/backend.py`
- `tools/run_synthetic_training_acceptance.py`
- `tools/verify_full_mcmc_gate.py`
- `tests/test_mcmc_runtime.py`
- `README.md`

### 修改内容

- MCMC strategy 调用前后只对最多 256 个 Gaussian 做一次轻量 position 探针；探针位于 optimizer step 之后且避开 refine 边界，因此测得的非零 delta 来自实际 noise 路径，而不是 Adam、relocation 或 add。首次观察到非零位移后停止复制，避免在生产点数上逐步增加显存和同步成本。
- 探针完成标记写入 MCMC strategy state，而不是仅存在 Backend 内存中；checkpoint 会保存和恢复该标记，防止中断恢复后重复探测造成 telemetry 与 uninterrupted 参考不一致。
- full-MCMC acceptance 新增实际 noise probe 次数、非零次数、最大位移、完整 refine event、Gaussian count curve、总 steps 和明确 `gate_status`。仅配置调用 noise 但实际 delta 为零时，Gate 必须失败。
- 验收后端的已知 dead Gaussian 从 opacity `0.001` 调低到 `1e-8`，使其在首次 refine 前不会被 opacity 梯度抬过 `min_opacity`；Machine-B 若仍未观察到 relocation，签名 Gate 必须失败而不是把“调用过 relocation 分支但零个对象”当作 PASS。
- 新增 `cloudstudio_full_mcmc_gate` 签名 schema，把 GPU/Torch/CUDA、锁定 gsplat commit、clean runtime、native covariance/fused perturb smoke、真实训练、relocate/add、分布曲线和 checkpoint 全状态恢复比较绑定到一个 canonical SHA256。
- 新增独立验证 CLI。验证器要求六项 execution gate 全为 PASS、runtime 与 repo lock 完全一致、loss 至少改善 20%、Gaussian 数量与 add 计数一致、noise 实际非零、relocation 非零、最终状态有限、曲线单调一致，以及连续/恢复 step 和 Gaussian identity/count 无 mismatch。任何字段篡改都会先触发签名失败。
- checked-in baseline 测试兼容当前 fail-closed registration schema 和未来 Machine-B 提交的完整 PASS schema；在完整签名证据通过验证前，现有 FAIL baseline 不会被自动覆盖。

### 验证方式与当前状态

先加入缺少签名/验证函数的红测试，旧代码按预期 ImportError；实现后 Gate 1/Trainer 定向测试 `24/24`、完整测试集 `121/121` 通过。测试覆盖签名 PASS、字段篡改、配置 noise 但实际零位移、Backend 真实 position delta 探针、只探测一次，以及既有 checkpoint/Trainer 契约；Python 编译、两个 CLI help、当前阶段 diff 与 UTF-8 乱码检查均通过。

随后同步 Machine-B 分支 `8886ede`：RTX 5070 Ti 的干净锁定 runtime 已证明所有注册算子、covariance/fused perturb 执行、80 步 add `24→29`、kill-based resume、受控中断后的 parameters/optimizer/MCMC/sampler/telemetry/CPU-CUDA RNG 全状态比较，以及 3000 步真实数据 runtime。3DGUT CUDA float atomic 导致连续/恢复运行存在约 `1.13e-6` 的数值漂移，因此 comparator 使用有实测依据的 `atol=5e-6`；这不是 bit-exact 声明。该证据同时明确 `total_relocated=0`，且旧 telemetry 只有 noise branch 调用次数、没有 position delta 探针，却把总门写成 PASS。合并时已保留全部正向证据，并把 baseline 纠正为 `FAIL`，列出“实际 noise 未探测、零 relocation”两个 blocker 后重新计算 SHA；checkpoint 恢复还吸收了 Machine-B 实测发现的 CUDA RNG blob 必须转回 CPU `uint8` 才能传给 `set_rng_state_all` 的修复。

当前本机没有完整 CUDA runtime，因此没有本地重跑合成或真实 3DGS 训练。Gate 1 下一步是在 Machine-B 同步本分支后，以 opacity `1e-8` 的已知 dead Gaussian 和新 position-noise 探针重跑，生成 `full_mcmc_gate_evidence.json` 并通过独立 CLI；在此之前不能由 CPU 单元测试或已有 3000 步 runtime 关闭总门，且该 3000 步运行已明确因 metric-scale MCMC noise 失配不接受画质结论。

## 19. 当前阶段记录：Renderer log-scale/linear-scale 契约修复

### 问题现象

Machine-B 首次真实 1044 图训练的全部验证视图呈无结构灰糊；未训练的 LiDAR 初始化直接渲染同样灰糊，但把同一点云通过数据侧 KB4 做 CPU point-splat 可以看到正确场景结构。逐层排除位姿、点云和相机投影后，定位到 Trainer 参数按 upstream MCMC 约定存储 `log(scale_m)`，而 `gsplat.rasterization()` 接口要求线性米制 scale。旧代码把约 `log(0.05)=-3` 直接送入 covariance，等效把 5 cm Gaussian 画成约 3 m blob；2 m 微型合成场景仍可通过大 blob 的颜色平均降低 loss，所以旧的 80 步“收敛”没有发现该错误。

### 修改文件

- `cloudstudio_3dgs/training/backend.py`
- `cloudstudio_3dgs/training/runtime_evidence.py`
- `tests/test_training.py`
- `tests/test_mcmc_runtime.py`
- `tools/run_synthetic_training_acceptance.py`
- `baselines/full_mcmc_runtime.baseline.json`
- `README.md`
- `docs/IMPLEMENTATION_PLAN.zh-CN.md`

### 修改内容

- 只在 renderer 边界使用 `torch.exp(params["scales"])` 转回线性米制 scale；optimizer、MCMC strategy、checkpoint 和 telemetry 继续保存 log-scale，不引入双重指数或 checkpoint schema 迁移。
- 增加不依赖 gsplat/CUDA 的 CPU 合约测试，用捕获 rasterization 参数的 fake backend 明确断言存储的 `log(0.1)` 到边界变为 `0.1`；另保留真实 CUDA footprint 测试，四个 z=2 m、scale=0.1 m、f=100 px 的 Gaussian 覆盖像素必须落在有限区间，防止 log-scale 再次洗满整帧。
- Machine-B 在修复后重新运行 80 步 full-MCMC：loss 改善从旧错误路径的 `87.40%` 修正为 `35.88%`，Gaussian 仍为 `24→29`，full-state resume `mismatch_count=0`、最大绝对漂移约 `1.91e-6`。baseline 保存新 run SHA 和 render-scale 根因，但总门继续因实际 noise delta 未探测、relocation 为零而保持 `FAIL`。
- 修复前 3000 步真实运行仍只保留为 GPU runtime/资源证据；其渲染质量、历史 smoke PSNR 和任何清晰度结论全部作废，不能作为 Gate 2 或真实 A/B 基线。
- 将同一个真实 GPU footprint smoke 接入 `run_synthetic_training_acceptance.py` 和签名 `cloudstudio_full_mcmc_gate`：四个已知米制 Gaussian 的 alpha 必须有限，且覆盖像素落在 `40..4000`。独立 verifier 新增 `metric_scale_rasterization` 必过项；仅 loss 收敛、算子注册或源码测试通过，都不能代替 renderer 的实际尺度契约。

### 验证方式与当前状态

`tests.test_training` 共 `16/16` 执行，其中 15 通过；真实 CUDA footprint 因本机 gsplat checkout 非干净锁定 runtime 明确 `SKIPPED`，不能据此声称本机 GPU 验收。签名 verifier 另有红/绿测试证明缺少 `metric_scale_rasterization` 时旧实现会误 PASS、新实现会拒绝。全仓 `124` 项测试为 `123 PASS + 1 SKIPPED`，CPU renderer 合约、Gate 1 签名/恢复、BA、mask、depth、Rig 和既有训练契约均通过。Machine-B 已用完整 runtime 生成修复后的合成运行证据，但仍需同步最新 dead-Gaussian/noise-probe/metric-scale/签名 schema 再跑一次，才具备关闭 Gate 1 的条件。

## 20. 当前阶段记录：Gate 1 签名 baseline 原子升级

### 问题现象

此前 README 只要求先运行 verifier，再由操作者手工把 `full_mcmc_gate_evidence.json` 复制成 checked-in baseline。验证和复制是两个独立动作，存在选错运行目录、复制验证前文件、写到一半中断，以及无意覆盖已有 PASS 证据的风险；这些都会让“验证通过”与最终提交的 JSON 失去事务一致性。

### 修改文件

- `tools/promote_full_mcmc_gate.py`
- `tests/test_mcmc_runtime.py`
- `README.md`
- `docs/IMPLEMENTATION_PLAN.zh-CN.md`

### 修改内容

- 新增唯一 promotion 入口：先用 repo 的 `cloudstudio_trainer.lock.json` 对完整签名 evidence 做 fail-closed 验证，只有所有 execution gate、runtime provenance、实际 noise、relocation/add、metric-scale footprint 和 resume 全部 PASS 才允许写 baseline。
- 输出使用同目录临时文件、UTF-8/LF、flush、`fsync` 和 `os.replace` 原子替换；验证失败时目标文件不存在或保持原样。
- 当前 FAIL baseline 可以被新的 PASS evidence 升级；若目标已经是 PASS，默认抛出 `FileExistsError`，只有显式 `--replace-pass` 才允许经人工审查后的替换，避免较弱或错误运行静默覆盖已接受证据。

### 验证方式与当前状态

先加入缺少 promotion 模块的红测试，旧代码按预期 ImportError；实现后测试证明合法签名可原子落盘、字段篡改无法创建目标文件、已有 PASS 默认不可覆盖。全仓 `125` 项为 `124 PASS + 1 SKIPPED`，Python 编译、CLI help、diff 与 UTF-8 乱码检查通过；唯一跳过仍是本机缺少干净完整 gsplat runtime 的真实 GPU footprint。该工具不生成 GPU 证据，也不会把当前 FAIL baseline 自动升级；Machine-B 严格重跑仍是关闭 Gate 1 的必要条件。

## 21. 当前阶段记录：Gate 1 严格签名证据关闭

### 问题现象

Machine-B 已推送的 `5822147` 修复了 renderer 的 log-scale/linear-scale 契约，但远端分支尚未包含满足最新统一 schema 的 relocation、实际 position-noise、米制 footprint 与签名 promotion 证据。共享 `external/gsplat` 仍有其他机器的未提交构建修改，不能拿它绕过 clean-lock 门；同时，旧报告的 `2h05m → 16m` 应换算为约 `7.8×`，不是 `125×`，后续性能结论必须使用可复核的同配置计时。

### 修改文件

- `baselines/full_mcmc_runtime.baseline.json`
- `README.md`
- `docs/IMPLEMENTATION_PLAN.zh-CN.md`

### 修改与执行内容

- 从锁文件指向的 gsplat `f2d14131483644e9977451b6403f6f0b73e6637f` 建立独立干净 checkout，不修改共享 dirty runtime；环境为 Windows 11、Python `3.12.9`、Torch `2.11.0+cu128`、CUDA `12.8`、RTX 5070 Laptop。
- Windows JIT 构建中确认 `NVCC_CCBIN` 必须指向 MSVC 工具目录；PyTorch 2.11 的 Ninja host compile 仍调用 `cl`，而当前进程同时存在大小写不同的重复 `PATH/Path`。最终在隔离子进程中只保留一条规范化 `Path`，完成全量扩展加载并核对 required-op 缺失数为 `0`。这只是本机环境诊断，不改写仓库构建脚本，也不把 `/FORCE:UNRESOLVED` 的世界空间批处理入口冒充为已执行证据。
- 使用 `run_synthetic_training_acceptance.py --steps 80 --full-mcmc --resume-equivalence` 生成统一 evidence，再由 `verify_full_mcmc_gate.py` 独立校验，最后通过 `promote_full_mcmc_gate.py` 原子替换 FAIL baseline；没有手工复制或修改签名 JSON。

### 验证方式与当前状态

签名证据 SHA256 为 `6e88d380c15889707950da680fc5f5b53a50b4b5e9179b4e918a05d128f31aa7`，runtime 为干净锁定源码。covariance forward/backward、rasterization forward/backward、fused position perturb、实际 MCMC noise、relocation、sample add、米制 scale rasterization 和中断恢复八类检查全部通过：80/80 步进入 noise，实际非零 position 探针通过，relocate `1`、add `5`、Gaussian `24→29`，四个 0.1 m Gaussian 的 footprint 为 `368 px`，最终状态 finite；连续/恢复 checkpoint 的 parameters、optimizer、MCMC、sampler、telemetry 与 CPU/CUDA RNG 比较为 `0` 失配，最大绝对漂移 `1.9073486328125e-6`，低于有 CUDA 原子噪底依据的 `5e-6` 容差。

当前状态为 **PASS（Gate 1 执行与恢复证据）**。这不升级为真实场景画质验收：修复前全部渲染数值继续作废，Machine-B 的 30k 真实训练在产物和签名推送前保持外部进行中/未验收。本次合成运行的 position-noise 最大位移为 `2.755 m`，说明固定上游噪声对米制大场景可能过强；Gate 2 第一优先项因此是把 means LR、noise LR 与场景尺度/初始化间距绑定，并用一次一变量的真实数据短跑验证，不能直接沿用合成配置。

## 22. 当前阶段记录：factor 后 LiDAR 评估产物对齐

### 问题现象

Machine-B 的真实 UK 训练首次运行到质量报告时发现：训练与渲染使用 `factor=4` 后的 728×728 图像和 LiDAR supervision，但 `_save_evaluation_artifacts` 仍把 2912×2912 原始 depth cache 原样复制到运行目录。质量报告按设计要求 RGB mask、rendered range 和 LiDAR target 空间尺寸完全一致，因此真实运行在这里 fail-closed；历史 factor=1 合成 fixture 让这个错误潜伏。

### 修改文件

- `cloudstudio_3dgs/training/trainer.py`
- `tests/test_training.py`
- `docs/IMPLEMENTATION_PLAN.zh-CN.md`

### 修改内容

- 吸收 Machine-B 提交 `62dabe6`：评估产物不再复制 full-resolution 源 cache，而是从 Dataset 已完成 factor、crop、base/person/depth-valid mask 合成后的 `depth_range_m`、`depth_confidence` 和 `depth_mask` 重建确定性稀疏 NPZ。
- 运行 Manifest 的逐帧记录新增 `lidar_depth_cache_semantics=factor_crop_mask_adjusted_euclidean_ray_range_m` 和有效像素数，明确该产物是评估视图监督，不伪装成保留原始 point provenance 的投影 cache；重采样后不再成立的 `source_index/support_count` 使用 `-1/0` 哨兵，质量指标只消费 range、confidence 与 valid pixel。
- 新增 CPU 回归：源 cache 路径故意不存在，证明实现不能退回复制；同时断言输出 shape、被 mask/NaN/零 confidence 排除后的精确 pixel index、range/confidence、provenance 哨兵和两次输出字节确定性。

### 验证方式与当前状态

该修复只校正质量评估输入，不改变训练 loss、checkpoint 或 Gaussian 参数。新增定向测试 `1/1` 通过；全仓 `126` 项为 `125 PASS + 1 SKIPPED`，唯一跳过仍是默认环境未指向 clean locked gsplat 的真实 CUDA footprint，本轮 Gate 1 签名证据已在隔离 runtime 中实际覆盖该项。当前状态为 **PASS（源码/CPU 契约）**；Machine-B 当前 30k 运行若在 `62dabe6` 之前启动，旧进程不会自动获得修复，必须以其实际代码 commit 与最终 run Manifest 判断能否直接生成质量报告，不能仅凭“训练完成”升级画质 Gate。

## 23. 当前阶段记录：Gate 2A KNN 尺度与米制 MCMC 噪声定标

### 问题现象

上游 MCMC 的 position perturb 实现对 log-scale 取指数、形成 covariance，再乘 `means_lr × noise_lr`；各向同性近似下名义位移尺度为 `scale² × means_lr × noise_lr`。上游默认 `noise_lr=5e5` 依赖归一化场景，而 CloudStudio 明确使用不归一化的 S1 局部米制坐标。此前把固定 5 cm scale、`means_lr=1.6e-4` 和 `noise_lr=5e5` 直接组合，名义透明高斯单步扰动达到局部 scale 的约 4 倍；真实尸检中均值云从约 ±20 m 扩散到约 200 m，不能继续靠训练步数或 cap 临时压住。

### 修改文件

- `cloudstudio_3dgs/training/scale_calibration.py`
- `cloudstudio_3dgs/training/backend.py`
- `cloudstudio_3dgs/training/trainer.py`
- `tools/audit_metric_scale_calibration.py`
- `tools/run_synthetic_training_acceptance.py`
- `tools/run_mcmc_resume_equivalence.py`
- `tests/test_training.py`
- `baselines/gs2_metric_scale_calibration.baseline.json`
- `README.md`
- `docs/IMPLEMENTATION_PLAN.zh-CN.md`

### 修改内容

- 生产默认从每个 LiDAR 初始化点的 3 个最近邻距离计算 RMS KNN scale，与 upstream 初始化语义一致；以全体中位局部间距作 reference，并把异常稀密点限制在 reference 的 `0.25×..4×`。Backend 接受 `[N]` 或 `[N,3]` 的正有限米制尺度，在参数中继续保存 log-scale，renderer 边界继续只做一次指数。
- 默认 means LR 为 reference scale 的 `0.0032`，保持原固定 5 cm 配置对应的相对步长；MCMC noise LR 由目标名义扰动 `0.25×reference scale` 反解，不再硬编码 50 万。定标策略、原始参数、有效参数、尺度分布和 canonical SHA 写入 checkpoint identity 与签名 run Manifest，输入 PLY 或策略改变后旧 checkpoint 会 fail-closed。
- Gate 1 两个合成/恢复工具显式使用 `mode=fixed` 且两个 fraction 为 `null`，保留历史 `init_scale/means_lr/noise_lr`，避免 Gate 2 默认升级悄悄改变 Gate 1 证据含义。
- 新增只读审计 CLI，可在 GPU 训练前对任意初始化 PLY 输出不含绝对路径、原子写入的定标 evidence；已有输出拒绝覆盖。

### 验证方式与当前状态

合成立方体缩放 10 倍时，每点 KNN scale 与 effective means LR 都精确放大 10 倍，effective noise LR 缩小 100 倍，最终名义 noise 仍保持局部 scale 的 25%；固定显式模式保持 `0.05 / 1.6e-4 / 5e5`，名义比例 4.0，可重放 Gate 1。Backend CPU 测试验证逐点 scale 精确进入 log 参数；签名 baseline 测试会拒绝 effective noise 字段篡改。

真实 PR-04 初始化 PLY `66fbe620...42df` 的 376,906 点完成两份独立文件重放：两份 per-point scale 字节完全相同，报告 SHA 均为 `fcce9850d0683e29194aff76f3229a6663cf720004c59c272a2a2c35d4fc73d4`。定标耗时约 0.35 秒；reference/p50 为 `0.097916096 m`，p95 `0.132621631 m`，959 点（约 0.254%）被钳制；effective means LR 为 `0.0003133315`，effective noise LR 为 `8148.5784`，名义 noise 为 `0.024479024 m`。全仓 `130` 项为 `129 PASS + 1 SKIPPED`；唯一跳过是默认环境的 clean-runtime CUDA footprint，Gate 1 已有独立签名实跑覆盖。

当前状态为 **PASS（Gate 2A 源码、CPU 契约、真实点云定标）**，但真实 GPU 短 A/B 与画质仍为 `NOT_RUN`。下一项继续按 Work Package 推进 SH、local masked SSIM、robust log-range、opacity/scale regularization、周期 golden eval 和 best checkpoint；在这些质量地基完成前，不启动新的正式长训练。

## 24. 当前阶段记录：Gate 2B 局部 SSIM 与 robust log-range

### 问题现象

旧 `masked_rgb_ssim_loss` 把整张有效鱼眼区域压成每通道一组均值、方差和协方差，不能表达 11×11 邻域内的边缘、纹理和小结构；person/FoV mask 边缘也没有窗口覆盖率门。旧 LiDAR loss 直接优化 `confidence × |pred-target|` 米制绝对误差，远距离同等相对误差天然更大，深度边缘离群点保持线性影响。这两项都会让正式长训练把容量用于全局颜色统计或少量远距异常点。

### 修改文件

- `cloudstudio_3dgs/training/losses.py`
- `cloudstudio_3dgs/training/trainer.py`
- `tools/run_synthetic_training_acceptance.py`
- `tools/run_mcmc_resume_equivalence.py`
- `tests/test_training.py`
- `README.md`
- `docs/IMPLEMENTATION_PLAN.zh-CN.md`

### 修改内容

- 生产默认 RGB 结构项改为 mask-aware Gaussian-window SSIM：窗口默认 11×11、sigma 1.5，只在中心像素有效且加权有效覆盖率至少 0.8 时计入；卷积前用 `where` 把 mask 外值归零，再按有效 support 归一化局部一阶/二阶矩，避免 NaN 或黑边污染。无任何合格窗口时 fail-closed。原全局 masked SSIM 保留为 `global_masked_rgb_ssim_loss`，只作诊断。
- 生产默认 LiDAR 项改为 confidence-weighted log-range smooth L1/Huber，delta 默认 0.05；预测/目标继续要求正、有限且通过同一 depth/person mask。log residual 让等比例近/远误差同权，Huber 在大残差区保持有界梯度。配置可显式选择 `linear_l1` 兼容模式。
- Trainer contract schema v2 记录窗口、sigma、覆盖率、range 模式和 delta；step telemetry 区分 `lidar_range_loss` 与其 mode，不再把无量纲 log loss误写成米制 L1。两个 Gate 1 fixture 显式锁为 `linear_l1`，保留历史 `final_lidar_range_l1_m` 和签名证据语义。

### 验证方式与当前状态

红测试首先因新 local/log loss 不存在而 ImportError。实现后验证：mask 外预测即使写成 100 也不会改变 local SSIM；局部边缘平移会产生非零结构损失；有效覆盖不足时明确失败；预测和目标同时放大 10 倍时 log-range Huber 逐值相同，2× 比例误差符合解析 smooth-L1 值。Trainer/Gate 1 定向测试通过，全仓 `132` 项为 `131 PASS + 1 SKIPPED`，唯一跳过仍是默认环境的 clean-runtime CUDA footprint，已有 Gate 1 隔离实跑证据覆盖。

当前状态为 **PASS（Gate 2B 数学、mask 与配置契约）**。没有启动真实 GPU 训练，因此不能声称 PSNR/SSIM/深度或清晰度已经提升；后续受控 A/B 必须单独比较旧 global+linear 与新 local+log，且保持同一 person Manifest、split、位姿、初始化 PLY、步数和随机种子。

## 25. 当前阶段记录：吸收澳洲 UK 质量分支（优先实现）

### 同步结论

GitHub 重新联网后发现 `origin/machine-b/uk-quality` 已在共同祖先 `9651013` 之上完成 UK 场景的 Trainer 质量实验。按协作决策，该分支的当前 `backend.py` 与 `trainer.py` 为优先实现；本地此前独立的 SH 包装不再保留在生产路径，避免出现两套外观配置和两种训练语义。

### 吸收内容

- 每张训练图一个受限 log-gain 的 exposure compensation，只作用于 RGB loss；validation 固定 gain=1，checkpoint 通过 auxiliary parameter/optimizer 保存，避免用曝光补偿伪造验证指标。
- 上游风格 SH：DC 由点色初始化、其余系数零初始化并以 `colors_lr/20` 学习；step 纯函数按 1000 步逐阶解锁，means LR 以指数方式衰减到指定 final factor。调度不依赖独立 scheduler 状态，因此中断恢复从 step 可重算。
- 训练/验证均可使用显式背景合成。UK 实验定位默认黑底与过曝天空的 alpha 渗漏；白背景 P5 在同一 106 张 validation 上为 PSNR `16.21`、SSIM `0.5589`、PSNR P10 `15.78`，相对 P4 黑背景 PSNR `15.59` 提升 `0.62 dB`。
- 单视角“看起来更锐”的 `lidar_range_weight=0.15` 被完整 validation 否决：P4b PSNR `14.25` / SSIM `0.5308` / 深度 MAE `3.93 m`，P4 为 `15.59` / `0.5528` / `4.01 m`。因此保持 `0.05`，不能以约 2% 深度收益换取 `1.34 dB` 外观损失。

### 记录修复与验证

同步时发现 `experiments/runs.csv` 的旧表头 16 列而所有数据行为 12 列，属于不能直接用于筛选 preset 的证据记录缺陷。已改成列数固定的结构：明确写入可由 handoff 证实的全量指标，并将原来语义无法可靠还原的 6 个字段以 `legacy_unlabeled_values` 原样保留。新增 CPU 契约覆盖 progressive SH 边界、means decay、SH coefficient/renderer 参数、alpha 背景合成和 exposure clamp；原有 evaluation fake backend 因新 optional 参数不兼容的回归已修复。

当前状态为 **PASS（澳洲实现已吸收，源码/CPU 契约）**。这些 UK 指标是该机器的已签名实验输入，不自动升级为本机或原始 POS/Stage 2 POS 的画质结论；本机 clean locked CUDA 复验和之后的受控 A/B 仍为 `NOT_RUN`。

## 26. 当前阶段记录：Gate 2D 米制 geometry regularization

### 问题现象

MCMC 的 relocate/add 能控制低 opacity 对象的数量，但不会阻止 RGB loss 借助半透明雾、远大于局部 LiDAR 间距的 splat 或极端针状 scale 比例降低图像误差。此前该项目只有 strategy 的 prune 阈值，没有把这些退化模式作为可审计训练 loss。

### 修改内容

- 新增 `GeometryRegularizationConfig`，生产默认启用三项弱约束：mean opacity sparsity（引导无支持的低 opacity 雾进入既有 MCMC prune 路径）、相对 KNN reference scale 的上界 soft barrier（默认 `8×`）和 axis-ratio soft barrier（默认 `10×`）。正常尺度、正常各向异性区域的后两项为零，避免把真实梁柱或薄构件强行各向同性化。
- 每项在米制 `exp(log_scale)` 空间计算；配置、阈值和权重写入签名 trainer contract，step telemetry 保存 unweighted 三项与 total。Gate 1 synthetic/kill-resume 工具显式 `enabled=false`，保持已签名历史证据的 loss 语义。
- 单元测试构造正常、巨型和针状 Gaussian：正常 scale 不触发 upper/anisotropy，巨型和针状项非零，并验证两组参数梯度有限；disabled 配置严格返回零。

### 当前状态

**PASS（Gate 2D 源码/CPU 契约）**。未把任一真实质量提升归因于该正则；下一项是周期 golden eval 与 best checkpoint，并在本机 clean runtime 复验后才进行受控 A/B。

## 27. 当前阶段记录：Gate 2E 周期黄金视角评估与最佳检查点

### 问题现象

此前 Trainer 只按训练 batch loss 保存 `latest.pt`，结束后才全量生成 validation 渲染与质量报告。训练 loss 与未见视角质量并不等价：MCMC、曝光补偿、背景和正则可能让最后一步 batch 更低，却让固定 validation 视角更差；一旦训练超调，也没有可复核的“最佳模型”可用于正式质量评估。

### 修改文件

- `cloudstudio_3dgs/training/golden_eval.py`
- `cloudstudio_3dgs/training/trainer.py`
- `tests/test_golden_eval.py`
- `README.md`
- `docs/IMPLEMENTATION_PLAN.zh-CN.md`

### 修改内容

- 新增独立 `GoldenEvaluationConfig`：生产默认每 `1000` 步评估一次，选择指标固定为 masked RGB PSNR 的黄金视角均值，提升至少 `0.001 dB` 才 promotion。该配置进入 trainer contract 和 checkpoint identity；禁用、间隔或阈值改变都不能与旧 checkpoint 静默混用。
- 黄金图像不从文件名、随机采样或训练集推断，而是按已签名 split Manifest 的 `golden_views`、Rig Frame 和左右相机顺序读取；若图像不在 validation 或重复出现，立即失败。评估使用同一 RGB/person mask、同一渲染背景；同步记录 masked PSNR、SSIM 与可用时的 confidence-weighted LiDAR range MAE。
- 每次评估产出 `evaluation/golden_history.json`，其中含 canonical SHA256；训练状态把完整历史与当前最佳记录写进 `latest.pt`，中断恢复不会丢失 checkpoint 选择依据。只有客观提升才原子写入 `checkpoints/best_golden.pt`，不会用最后一个训练 batch 覆盖它。
- `run_manifest.json` 绑定 golden history SHA、次数、最佳评估记录和最佳 checkpoint 相对路径。黄金评估只是训练中选择器，不替代结束后覆盖完整 validation 的正式 `evaluate_run.py`。
- 正式质量报告读取该 history 时重新校验外层 SHA、每次评估 SHA、严格递增步号、配置和 promotion 规则，并核对 run Manifest 声明、评估次数、best record 与最佳 checkpoint 文件；任何已声明历史的篡改、丢失或替换均 fail-closed。旧 run 没有该层会明确标记 `golden_evaluation:NOT_RUN`，不会冒充完成过中程选择。

### 验证方式与当前状态

新增 CPU 回归构造顺序被打乱的 validation Dataset，断言仍严格按 signed golden 顺序评估、背景参数确实传到 renderer、PSNR 选择阈值不允许同分或不足 `0.001 dB` 的 checkpoint 覆盖最佳模型；同时验证非法黄金视图和零间隔 fail-closed。质量报告回归再篡改 history 中的 PSNR，验证签名立即失败。定向 `34` 项为 `33 PASS + 1 SKIPPED`，尚未启动新的真实 GPU 训练。

当前状态为 **PASS（Gate 2E 源码/CPU 契约）**。下一步是以合入的澳洲质量实现，在干净锁定 CUDA runtime 上跑短程真实基线，检查 golden history、best checkpoint、全量 validation 质量报告与 GPU 资源证据，再决定原始 POS / Stage 2 POS 的受控 A/B 是否可以启动。

## 28. 当前阶段记录：Gate 2F preset、周期完整验证与受控 A/B 证据链

### 问题现象

Gate 2E 虽然能生成 `best_golden.pt`，但旧 Trainer 没有可执行的兼容 preset，澳洲 P5 也仍靠手工配置；KNN、SH 和 local SSIM 无法保证只改变一个变量。周期评估只覆盖 golden views，正式完整 validation 仍只在终点渲染；更关键的是终点质量产物继续使用最后一步参数，`best_golden.pt` 并未真正成为被验收模型。因此仅凭“存在最佳 checkpoint”不能满足 Gate 2 Exit。

### 修改文件

- `cloudstudio_3dgs/training/presets.py`
- `cloudstudio_3dgs/training/ab_matrix.py`
- `cloudstudio_3dgs/training/ab_results.py`
- `cloudstudio_3dgs/training/golden_eval.py`
- `cloudstudio_3dgs/training/trainer.py`
- `cloudstudio_3dgs/evaluation/quality_report.py`
- `tools/build_trainer_ab_matrix.py`
- `tools/summarize_trainer_ab.py`
- `configs/trainer_gate2_ab_base.example.json`
- `tests/test_training_presets.py`
- `tests/test_ab_matrix.py`
- `tests/test_golden_eval.py`
- `tests/test_quality_report.py`
- `README.md`
- `docs/IMPLEMENTATION_PLAN.zh-CN.md`

### 修改内容

- 固化五个不可冒名 preset：`legacy_minimal_v1` 精确恢复 fixed-scale、RGB sigmoid、global masked moments、linear range、无 exposure/geometry regularization/means decay 的旧语义；三个单变量臂分别只启用 KNN、SH3 progressive 或 local SSIM；组合候选 `gate2_quality_australian_p5_v1` 以澳洲 P5 的 exposure、SH3、means decay `0.01`、白背景为优先外观基础，再叠加 KNN、local SSIM、robust log-range 和米制正则。preset 固定字段出现冲突时拒绝运行，直接构造 dataclass 也不能用不匹配参数冒充命名 preset。
- `GoldenEvaluationConfig` 默认每 `1000` 步生成 signed golden PSNR/SSIM/depth 与 reference/render/mask PNG 哈希证据，每 `4000` 步对完整 validation 计算相同指标；两类 history 都按严格递增 step 签名并进入 checkpoint。最终若最佳步早于终点，Trainer 会原子读取 `best_golden.pt` 后再生成正式完整 validation 产物，run Manifest 同时区分 final 与 selected step/Gaussian 数并绑定 selected model SHA；`latest.pt` 仍保留最终训练状态。
- 质量报告验证 golden/full history、每条记录、周期 PNG、最佳 checkpoint 和 selected model 的 SHA；缺失或篡改 fail-closed。旧 run 会明确产生 `golden_evaluation:NOT_RUN` / `periodic_full_evaluation:NOT_RUN`，不升级历史证据。
- A/B builder 只接收共享数据/资源/训练预算字段，禁止 base 混入 arm-specific 算法项；生成五份配置和签名 matrix，逐项绑定 dataset、base/person/depth mask、split、初始化 PLY、gsplat lock、seed、factor、步数和 Trainer contract。verifier 重新计算 contract diff，确保 KNN/SH/local SSIM 三臂只有声明的变量组变化。
- A/B 汇总器要求每臂训练完成、完整 validation、verified golden/best、至少两次 periodic full eval、相同输入身份、深度指标和默认 LPIPS；统一把“正值=更好”后报告单变量 IMPROVED/MIXED/REGRESSED，并对澳洲质量候选执行零容差、不低于 legacy reference 的 Gate 判定。工具只汇总真实产物，不启动训练，也不把缺失报告写成 PASS。

### 验证方式与当前状态

CPU 回归覆盖 preset 固定/冲突/冒名拒绝、legacy global SSIM 实际执行、单变量 contract diff、matrix/config/input SHA 篡改、周期 golden PNG、完整 validation 顺序与 P10、history/checkpoint/model SHA 以及结果分类/报告绑定。CLI help 和 Python 编译通过。当前源码/CPU 契约已 PASS，本机锁定 CUDA 合成重放随后由第 29 节关闭；factor4 五臂 A/B、LPIPS 和真实质量候选 Gate 仍保持 `NOT_RUN`，因此 Gate 2 尚未关闭。

## 29. 当前阶段记录：Windows CUDA 可复现入口与周期评估恢复等价闭环

### 问题现象

澳洲优先代码合并后，本机首次 JIT 构建曾成功输出 `GSPLAT_READY`，但普通 PowerShell/`cmd` 进程同时继承大小写不同的 `Path`/`PATH`；交互式 `where cl` 可通过，Python 启动的 NVCC 子进程却取到不含 MSVC 的另一份 PATH。另一次运行少带 `/FI...msvc_clzll.h`，PyTorch 因编译参数签名变化清空了已完成缓存。固定环境后 42 个对象全部编译成功，但 G: 上旧 `.ninja_log` 在记录 90 分钟大对象时返回 `Invalid argument`，导致最终链接未执行。直接链接并进入 80 步 GPU 验收后，参数/优化器/MCMC/RNG 最大差在容差内，但新增 golden/full evaluation 记录因 CUDA 原子累加带来的约 `1e-7` 指标微差产生不同自签名 SHA，旧比较器把 3 个有效签名差异误判为恢复状态不等价。

### 修改文件

- `train/run_gate2_synthetic_acceptance_local.ps1`
- `tools/run_with_prebuilt_gsplat.py`
- `cloudstudio_3dgs/training/checkpoint.py`
- `tests/test_mcmc_runtime.py`
- `README.md`
- `docs/IMPLEMENTATION_PLAN.zh-CN.md`

### 修改内容

- 新增 PowerShell 7 入口，从脚本位置解析仓库根目录，从 `vcvars64.bat` 重建大小写不敏感的环境字典，清空子进程继承环境后只写入一个 `Path`；显式固定 Python、CUDA 12.8、MSVC、Ninja、sm_120、JIT cache、`NVCC_FLAGS`、`INCLUDE` 和 `LINK`。每次先在真实 Python 子进程断言 `cl`/`ninja`/`nvcc` 可解析且只有一个 PATH key，构建与训练共享同一环境。巨量编译输出写入 `external/runtime-logs` 的 UTF-8 日志，终端只显示尾部和退出码。
- 对已完成 42 个对象但 Ninja 日志不可写的恢复场景，入口严格读取 `build.ninja` 的 link rule，要求 42 个对象全部存在后才允许显式 `-LinkExistingObjects`；直接调用同一 MSVC linker 生成 `gsplat_cuda.pyd`。两个 trainer 不调用的 world-space batch 符号仍按澳洲 Windows 运行边界以 `/FORCE:UNRESOLVED` 留下明确告警，实际单视图 3DGUT、MCMC、covariance 与尺度渲染路径由运行验收覆盖。
- 新增 prebuilt bootstrap，以编译名 `gsplat_cuda` 加载 `.pyd` 并显式注入 `gsplat.csrc`，使后续 probe/训练不再读取不可靠的 Ninja 日志；不存在扩展、脚本或对象时均 fail-closed。
- checkpoint comparator 不宽松忽略 SHA。它先对 golden/full evaluation 记录重新计算自签名；任一伪造签名立即 FAIL。只有连续/恢复两边自签名均有效时，才跳过自签名字节串本身，并继续用既有 `atol/rtol` 比较记录内全部浮点、结构和身份字段。数据集、mask、split、初始化、模型与其他身份 SHA 仍严格逐字节比较，报告列出被语义归一的精确路径。

### 验证方式

- 子环境 probe：`CL`、`NINJA`、`NVCC` 均解析到锁定工具链，`PATH_KEYS=['PATH']`。
- 直接链接扩展 `gsplat_cuda.pyd` 为 `107,542,528` bytes，文件 SHA256 `2d7ce65600808f3cf9f281e72773e589f13ed4d3e7d3614f551a921444e4adce`；bootstrap 输出 `GSPLAT_READY 1.5.3`。
- `run_synthetic_training_acceptance.py --steps 80 --full-mcmc --resume-equivalence` 在真实 RTX 5070 Laptop/CUDA 上完成：原生 covariance forward/backward 与 fused perturb 均 finite；render scale contract 以 `0.1 m` 线性尺度覆盖 `368` 像素并 PASS；24 个初始 Gaussian 经 5 次 refine 变为 29 个，add=5、relocation=1，80/80 步均调用 noise；loss 从 `0.0354085` 降至 `0.0227030`，改善 `35.88%`，无 NaN。
- 连续 80 步与 step 40 受控中断恢复到 80 步：Gaussian 数均为 29，参数/优化器/MCMC/sampler/telemetry/auxiliary/CPU+CUDA RNG 和 signed evaluation semantics 共 `0 mismatch`，`max_abs_error=1.9073486328125e-6 < atol 5e-6`；3 个 evaluation 自签名先验证有效后按语义比较。签名 evidence 内部 SHA 为 `c86f94163a1135b5cd260c466c8c2a821e17c9eaf037df34bd4b2006ece83189`，证据文件 SHA256 为 `51426f21c640e8ea2dcbab7f5f0bf67b87c84e7cc5c0d4f43c5b06a040215306`。
- 独立 `verify_full_mcmc_gate.py` 返回 `status=PASS`、`signature_valid=true`、`errors=[]`。新增回归同时证明容差内重新签名的 evaluation 记录可通过语义等价比较，而篡改自签名仍 FAIL。

### 当前状态

当前为 **PASS（锁定 Windows CUDA 运行时、完整 MCMC、物理尺度渲染与中断恢复等价）**。这关闭了合并澳洲最新代码后的本机短程 GPU 复验，不等于真实场景画质或 Gate 2 正式退出；下一步仍需重建真实 gs2 的签名 person/depth/split/PLY 输入，并执行同一 factor4 数据上的 legacy、KNN-only、SH-only、local-SSIM-only 与澳洲 P5 质量候选五臂 A/B，且至少产生两次完整 validation 和 LPIPS 后才能判定 Gate 2。

## 30. 当前阶段记录：澳洲 P8/P9 归因、软曝光锚与 PLY 导出吸收

### 问题现象

联网复核时澳洲 `machine-b/uk-quality` 先从 `b5bcb30` 前进到 `21235e4`，随后 P10 完成并推进到 `2113134`。P8 单变量证据表明 hard zero-mean exposure anchor 相对 P5 下降约 `0.8 dB`，geometry regularization 被排除为主因；P9 同时使用 soft anchor、decoupled SSIM、SH1 和 geometry regularization 后仍比 P5 低 `0.62 dB`，因此澳洲用 P10 在当前代码上重跑完全相同的 P5 control。P10 得到 PSNR `16.45`、SSIM `0.5587`、P10 PSNR `16.01`、depth MAE `4.156 m`，高于旧 P5 的 `16.21 dB`，排除了“合并后 trainer 代码整体回归”，并确认 hard/soft exposure anchoring 都应拒绝。合并新配置后，旧命名 preset 未声明 `mean_anchor_weight/beta`，fail-closed 校验报 exposure contract 不匹配。全仓回归还暴露 PPISP 梯度测试使用未设种子的全局随机扰动，偶尔把 CRF 参数推入饱和区而产生零梯度，造成非确定性红灯。

### 修改文件

- 澳洲原提交：`cloudstudio_3dgs/training/exposure.py`、`experiments/runs.csv`、`tests/test_training.py`、`tools/export_gaussian_ply.py`
- 本机兼容：`cloudstudio_3dgs/training/presets.py`、`tests/test_ppisp.py`
- `docs/IMPLEMENTATION_PLAN.zh-CN.md`

### 修改内容

- 完整吸收澳洲 soft per-camera mean anchor：对每个物理相机组的 mean log-gain 使用 SmoothL1 软约束，保留 hard projection 但不把它提升为推荐 preset；P8/P9 回归结果按原记录保留，不把未通过实验的旋钮默认启用。
- 合并标准 3DGS viewer PLY 导出器，支持从 checkpoint 输出 INRIA binary little-endian 字段布局，并明确保留 trainer 本地米制坐标与 coordinate transform 责任边界。
- 所有命名 preset 显式冻结 `mean_anchor_weight=0.0`、`mean_anchor_beta=0.1`，使 P5、legacy 和三个单变量臂不会随 dataclass 新默认漂移，也不能被不匹配配置冒名。
- PPISP 梯度夹具改用测试内部固定 `torch.Generator(seed=20260822)`；仍要求 exposure、vignetting、color 与 CRF 全部参数取得 finite 且非零梯度，没有放宽断言。

### 验证方式与当前状态

合并后 `export_gaussian_ply.py --help` 通过；preset/A-B/soft anchor/PPISP 定向 `18` 项为 `17 PASS + 1 SKIPPED`，当时全仓 `182` 项为 `181 PASS + 1 SKIPPED`。澳洲最新 `2113134` 已成为当前分支祖先。当前状态为 **PASS（澳洲代码与 P10 结论已吸收）**：P10 证明当前代码下原始 P5 仍是赢家，P8/P9 的 hard/soft anchor 均被拒绝；澳洲已用该配置启动 30k-gold，最终结果仍待同步。

## 31. 当前阶段记录：真实 factor4 P5 短基线与深度覆盖率评估修复

### 问题现象

重新联网确认澳洲 `machine-b/uk-quality@21235e4` 已完整成为当前分支祖先后，本机使用澳洲 P5 作为组合候选准备真实 Gate 2。完整输入逐文件预检通过，但 600 步短基线暴露出评估契约缺陷：训练 step telemetry 中 `lidar_range_loss=0.344568`，最终也保存了 728×728 rendered range；周期 golden 和完整 validation 却把 16/16 与 124/124 张深度全部标记为 `UNMEASURABLE`。抽查并统计 124 张产物后确认，预测在 LiDAR 监督像素的覆盖率实际为 min/p05/p50/mean `0.932899 / 0.964930 / 0.993464 / 0.989577`，旧指标因为每张图仅有少量未覆盖像素就否决整张图，掩盖了绝大多数可测深度。

### 修改文件

- `cloudstudio_3dgs/evaluation/image_metrics.py`
- `cloudstudio_3dgs/training/golden_eval.py`
- `cloudstudio_3dgs/evaluation/quality_report.py`
- `tests/test_quality_metrics.py`
- `tests/test_golden_eval.py`
- `train/run_gate2_synthetic_acceptance_local.ps1`
- `docs/IMPLEMENTATION_PLAN.zh-CN.md`

### 修改内容

- 深度指标继续以 target-valid LiDAR 像素为分母，同时显式统计预测正有限覆盖数、缺失数和覆盖率；默认 API 仍保持 `100%` 严格门，训练周期评估和正式质量报告采用固定 `90%` fail-closed 门。达到门限后只在正有限交集计算 confidence-weighted MAE/RMSE，并把 coverage 作为伴随指标写入逐帧与汇总证据；低于门限或完全无覆盖仍明确失败，不会用极少量命中像素伪造好指标。
- golden/full evaluation 算法版本升级为 `v3`，签名记录新增 coverage min/mean 和门限；最终质量报告同时汇总 124 张 coverage 分布。
- Windows 锁定 CUDA 启动器新增互斥的 `-TrainerConfig` 路径，复用同一个去重 PATH、MSVC/CUDA 环境和预编译 gsplat bootstrap 启动真实 Trainer；原 `Output` 合成验收入口保持不变。

### 验证方式与当前状态

- 真实输入：1238/1238 张、4952 个原图/base mask/person mask/depth 文件逐文件 SHA 与实际解码通过；内部签名为 dataset `54c01abe...`、mask `86ae782a...`、person `1eb2284f...`、depth `3c114dfd...`、split `dbb4cf46...`，初始化 PLY 为 `66fbe620...`。五臂 matrix 使用相同输入，SHA 为 `1382d7b7210ed7a4b328256106fea0c1c7bd88290a4f9d1bdc204885f05dd228`，组合候选明确为 `gate2_quality_australian_p5_v1`。
- 真实 GPU：RTX 5070 Laptop 上 factor4 P5 完成 600 步，耗时 `316.99 s`、峰值额外 VRAM `866,290,688 bytes`、376,906 个 Gaussian、无 NaN；loss 从 `0.276763` 降至 `0.200898`，改善 `27.41%`。golden PSNR 在 step 200/400/600 为 `15.17 / 15.75 / 16.15 dB`，终点完整 124 图为 PSNR `16.1195 dB`、SSIM `0.502887`。签名 run Manifest SHA 为 `5146c877d3ccf4e9d00e5aea831960e1dc99f225af302d153ea12c85cb35e4d2`。
- 修复后对同一签名 run 重新生成质量报告：124/124 张深度可测，coverage min/mean 为 `0.932899 / 0.989577`，depth MAE mean 为 `8.19595 m`；报告仅因 LPIPS 尚未运行而保持 `PARTIAL`。该 MAE 是 600 步短链结果，不能解释为正式质量通过。
- 定向回归 `14/14 PASS`：严格默认仍拒绝缺口，显式低门测试只在超过门限时测量，低于门限/全零预测仍 `UNMEASURABLE`。完整 CPU 套件为 `182 PASS + 1 SKIPPED + 3 subtests PASS`；唯一跳过是需要锁定 CUDA 扩展的物理足迹测试，本轮真实 600 步 GPU 运行和第 29 节签名尺度证据已分别覆盖运行链与足迹链。

当前为 **PASS（真实输入、短训练链、RGB 与深度评估可测性）**，但 Gate 2 仍未关闭：LPIPS、至少两次周期完整验证、legacy/KNN/SH/local-SSIM/P5 五臂正式 A/B 均未完成；澳洲 P10 已确认 P5，30k-gold 最终结果仍待同步。下一步先推送本修复与澳洲最新合并，再用新 `v3` 评估契约启动正式受控运行。

## 32. 当前阶段记录：澳洲误差加权 MCMC 吸收与恢复状态门禁

### 问题现象

GitHub 重新联网后，澳洲 `machine-b/uk-quality` 从 `2113134` 推进到 `aaf25b8`，新增按 `opacity * error_score^0.4` 为 relocation 和 densification 选择落点的误差加权 MCMC。实现默认关闭并带 CPU 算法测试，但首次审计发现两个门禁缺口：其一，每个 Gaussian 的误差 EMA 只保存在运行期 `ErrorScoreState`，checkpoint 没有保存该层；启用后若中断恢复，下一次 refine 可能走不同 multinomial 分支。其二，Trainer 为读取轻量配置在模块顶层导入完整 CUDA 策略，使本来不依赖 gsplat 的 CPU preset/A-B 测试在收集阶段尝试 JIT，默认关闭也无法保持轻量导入契约。

### 修改文件

- 澳洲原提交：`cloudstudio_3dgs/training/error_weighted_mcmc.py`、`cloudstudio_3dgs/training/backend.py`、`cloudstudio_3dgs/training/trainer.py`、`tests/test_error_weighted_mcmc.py`
- 本机恢复补强：`cloudstudio_3dgs/training/error_weighted_config.py`、`cloudstudio_3dgs/training/error_weighted_mcmc.py`、`cloudstudio_3dgs/training/trainer.py`、`tools/run_synthetic_training_acceptance.py`、`tests/test_error_weighted_mcmc.py`
- `docs/IMPLEMENTATION_PLAN.zh-CN.md`

### 修改内容

- 以澳洲 `aaf25b8` 为当前优先代码主线；算法保持默认关闭，不能在未声明的旧 P5/legacy 配置中改变采样语义。
- 对原提交引用的 arXiv:2508.12313 进行原文核对：论文的 EAS 使用 Laplacian 边缘权重、Gaussian 对像素的 alpha contribution，并以绝对坐标梯度筛选候选；当前 `opacity * center-RGB-residual^0.4` 没有实现这些量。因此保留澳洲功能代码，但明确改称 CloudStudio 实验启发式，不再冒充论文复现；`0.4` 也作为待 A/B 的本项目参数记录。
- `ErrorScoreState` 新增版本化 checkpoint payload，保存完整 EMA tensor；恢复时严格检查 schema、维度、Gaussian 数和 finite 值，再复制到当前 device/dtype。启用算法但 checkpoint 缺少该层时 fail-closed，不允许静默退化为 uniform/opacity-only 采样。
- Trainer 的 `training_state` 同步持久化该状态；载入模型和 optimizer 后按恢复后的 Gaussian 数校验并恢复 EMA，使全状态比较器能够逐元素覆盖它。默认关闭时写入 `null`，旧的默认关闭 checkpoint 仍可读取。
- 将 `ErrorScoreConfig` 拆到不导入 torch/gsplat 的轻量模块；Trainer、配置解析和合成验收只依赖该模块。真正构造 `GsplatBackend` 时才导入 CUDA 策略，恢复默认关闭与纯 CPU 工具的模块边界。
- 合成验收工具新增 `--error-weighted-sampling`，只允许与完整 MCMC 同时使用；可与 `--resume-equivalence` 组合，专门执行启用新算法后的连续训练与受控中断恢复比较，而不是用“默认关闭”结果代替新路径证据。

### 验证方式与当前状态

- 新增 checkpoint round-trip 回归，验证 EMA scores 精确恢复且不共享存储；缺失/错误 schema、数量不符和 NaN 均明确拒绝。
- 通过锁定的 prebuilt gsplat bootstrap 执行 `tests/test_error_weighted_mcmc.py`，结果为 `21 PASS + 4 subtests PASS`；Python 编译检查通过。全仓 CPU 套件在隐藏 CUDA 后为 `182 PASS + 22 SKIPPED + 3 subtests PASS`，新增跳过项是明确需要已加载 gsplat 的澳洲策略测试，不再发生 preset/A-B 收集期 JIT 或权限错误。
- 启用误差加权采样后，真实 CUDA 80 步完整 MCMC 连续训练与 step-40 受控中断恢复到 80 步均完成：Gaussian 数均为 29，恢复比较 `0 mismatch`、`max_abs_error=2.6226043701171875e-6 < atol 5e-6`；恢复 checkpoint 中 29 个 EMA score 全部 finite，范围 `0.411736–0.557433`，trainer contract 明确记录 `enabled=true`。签名 run Manifest 为 `d07e0e712b5cca0675c82ae5e721311d2cc03fd438fc32013394138757bcc5b2`，验收文件 SHA256 为 `5d6b19f173fd4dced1a29a9fa78cf95890ae8e13779ea2cbee58a197942080e4`。

本节当前为 **PASS（澳洲误差加权采样的加载、真实执行与中断恢复状态门禁）**；这仍不证明画质提升，也不把它升级为默认。真实短 A/B 必须继续使用相同输入、seed、预算、mask 与 `v3` 深度覆盖门。

## 33. 当前阶段记录：真实 Gate 2 legacy 与澳洲 P5 正式两臂结果

### 问题现象

600 步短链只能证明训练、mask 和深度评估可运行，不能回答长期外观、LPIPS、最佳 checkpoint 或巨型 Gaussian 的行为。为避免把澳洲 P5 的改进归因于不同输入，本机在同一个签名 matrix 下先完成 legacy reference 与 `gate2_quality_australian_p5_v1` 两个 8000 步正式臂；两者绑定相同 1238 图 dataset、1114/124 train/validation split、person/depth mask、初始化 PLY、factor4、seed 42、1M cap 和锁定 gsplat runtime。

### 验证结果

- Legacy reference：选择 step 6000，124 图 PSNR `17.834794`、SSIM `0.467080`、LPIPS-Alex `0.592530`、depth MAE `4.005335 m`，质量报告 SHA256 为 `edc604a1ecb9feb9d1cbd456d51678bd04f22607f7a351ef08f21675b81009b1`。
- 澳洲 P5：8000 步耗时 `3275.30 s`、峰值额外 VRAM `1,961,667,584 bytes`、最终 1M Gaussian、无 NaN；golden PSNR 在 step 6000 达到 `20.5556 dB`，step 7000 回落到 `20.4886`，因此最终正确选择 step 6000。签名 run Manifest 为 `dc8293128cfca9da1cf032c63b23782971212ad482112d1dab7cdc6cead57a11`。
- P5 选择模型的 124 图正式指标为 PSNR `20.520440`、SSIM `0.595492`、LPIPS-Alex `0.479350`、depth MAE `6.555521 m`、预测 coverage min `1.0`；质量报告 SHA256 为 `08fd4f3700ddc1b12cc0dbbaabd1747620791e85ae8eb76d5233f523084ed9bb`。
- 相对 legacy，P5 提升 `+2.68565 dB` PSNR、`+0.12841` SSIM，LPIPS 降低 `0.11318`，但绝对深度 MAE 恶化 `+2.55019 m`（约 `63.7%`）。因此正式结论是 **MIXED / Gate 2 未通过**，不能用外观优势掩盖米制几何回归。
- CPU 尸检同为 1M Gaussian 的最佳 checkpoint：legacy 的 scale p50/p95/p99/p999/max 为 `0.0693/0.2279/0.5536/1.8695/13.9442 m`，大于 10 m 的仅 13 个；P5 为 `0.1249/0.3375/0.8977/4.7247/142.2038 m`，大于 10 m 的有 396 个。P5 的 soft scale regularization 没有阻止极端巨型 splat，与前景/边缘模糊及深度回归方向一致，但仍属于待单变量验证的根因候选。

### 当前状态

当前为 **PASS（两臂正式证据完整）/ FAIL（澳洲组合候选的零回归 Gate）**。下一步不盲目加大所有正则，而是按同数据短臂分别验证：① P5 保持外观配置、只把 LiDAR loss 改回 `linear_l1`；② P5 保持 loss、只启用米制 world-scale fuse；③ 澳洲最新误差加权采样。只有单变量结果确定后才组合复验，并继续完成 KNN/SH/local-SSIM 三个正式臂。

## 34. 当前阶段记录：P5 深度回归单变量探针与平衡候选

### 问题现象

澳洲 P5 已被正式证据确认是当前最强外观基础，但相对 legacy 的深度 MAE 恶化 `2.55019 m`。需要优先保留澳洲版本的 SH、KNN、local SSIM、曝光、背景与 means decay，不用多变量改动掩盖归因；短探针只允许改变一个深度或采样变量，再决定是否值得执行完整 8000 步。

### 修改文件

- `cloudstudio_3dgs/training/trainer.py`
- `cloudstudio_3dgs/training/presets.py`
- `cloudstudio_3dgs/training/ab_matrix.py`
- `cloudstudio_3dgs/training/checkpoint.py`
- `tests/test_training.py`
- `tests/test_training_presets.py`
- `train/run_gate2_synthetic_acceptance_local.ps1`
- `README.md`
- `docs/IMPLEMENTATION_PLAN.zh-CN.md`

### 修改内容与探针结果

- 澳洲误差加权 MCMC 在相同真实输入上的 1000 步 probe，正式 124 图为 PSNR `17.37595`、SSIM `0.53350`、LPIPS `0.60239`、depth MAE `7.73792 m`；相对默认关闭 P5 同步点外观轻微回归、深度不改善，因此保持默认关闭，不进入正式候选。
- 将 P5 的 robust log-Huber 整体换成 `linear_l1` 后，golden step 1000 为 PSNR `14.1659`、SSIM `0.48135`、depth MAE `4.47748 m`，且完整验证存在 depth coverage `0.895791 < 0.9`。虽然深度回收明显，但外观损失过大并触发覆盖门，拒绝。
- 只启用 `1.5 m` world-scale fuse 后，golden step 1000 为 PSNR `16.7026`、SSIM `0.53030`、depth MAE `7.63966 m`；仅改善约 `0.06 m` 深度却损失约 `0.77 dB`，拒绝把硬裁剪提升为候选。
- 新增 `lidar_linear_aux_weight`：主损失仍为澳洲 P5 的 confidence-weighted robust log-Huber，仅叠加弱线性米制 L1；该权重进入签名 Trainer contract、preset/A-B 字段与逐步 telemetry，非 robust 主损失、负权重或缺少 depth 输入均 fail-closed，算法版本升级为 `cloudstudio_gsplat_trainer_v5`。
- 三个同输入、同 seed、同 1000 步探针呈单调 Pareto：权重 `0.0025` 的完整 PSNR/SSIM/LPIPS/depth 为 `17.21927 / 0.52850 / 0.60563 / 6.99518 m`；`0.005` 为 `16.92767 / 0.52367 / 0.60939 / 6.37880 m`；`0.01` 为 `16.35399 / 0.51503 / 0.61603 / 5.80396 m`。`0.01` 相对原 P5 同步点约损失 `1.08 dB`，但改善约 `1.93 m` 深度，最可能利用 P5 正式 `+2.69 dB` 的外观余量关闭 legacy 深度门。
- 因此新增命名候选 `gate2_quality_australian_p5_depth_balanced_v1`，与 `gate2_quality_australian_p5_v1` 的固定字段只允许 `lidar_linear_aux_weight: 0.0 → 0.01` 一处差异。它是待正式验证的澳洲优先候选，不是生产默认；测试会直接比较两个 preset，防止未来悄然漂移其他外观旋钮。

### 验证方式与当前状态

当前 CPU 全仓为 `183 PASS + 22 SKIPPED + 3 subtests PASS`；跳过项仍是需要预加载锁定 CUDA 扩展的运行测试。所有短 probe 都绑定相同真实 factor4 输入、person/depth mask、初始化 PLY、seed 42 和 1M cap，质量报告分别签名保存。

### 正式 8000 步结果

- 命名候选在 RTX 5070 Laptop 上完成 8000 步，耗时 `3269.10 s`、峰值额外显存 `1,932,662,272 bytes`、最终 1M Gaussian、无 NaN；内部签名 run Manifest SHA256 为 `085ca8e0900f8dc4cea96edcc05a2376c2aba539cb093174c41494f6ab2f8661`。
- golden PSNR 从 step 1000 到 6000 按 `16.3869 → 17.6076 → 18.6600 → 19.1409 → 19.3240 → 19.5010 dB` 提升，step 7000/8000 回落到 `19.4583 / 19.3989 dB`，因此正确选择 step 6000；step 5000 的 golden depth MAE `4.917 m` 最低，step 6000 回退到 `5.024 m`，明确暴露外观选择与米制几何并非同一目标。
- 124 图 selected-model 正式报告为 `COMPLETE`：PSNR `19.452818 dB`、SSIM `0.568499`、LPIPS-Alex `0.501412`、depth MAE `5.042365 m`、depth coverage min `0.979954`；内部签名 quality report SHA256 为 `04d33bfe8dde895361c7418a45a11134750f6c68c263f80507c8eaad33999209`。
- 相对澳洲 P5，深度改善 `1.513156 m`，但 PSNR 下降 `1.067622 dB`、SSIM 下降 `0.026992`、LPIPS 恶化 `0.022062`；相对 legacy，PSNR 仍提升 `1.618024 dB`、SSIM 提升 `0.101420`、LPIPS 改善 `0.091118`，但 depth MAE 仍恶化 `1.037031 m`。结论仍为 **MIXED / Gate 2 FAIL**，不能替代澳洲 P5，也不能进入 POS A/B。
- 尺度尸检表明该辅助项没有根治巨型 splat：selected step 6000 的 scale p50/p95/p99/p999/max 为 `0.1308/0.4445/1.2527/4.9981/42.7079 m`，大于 `1 m` 有 `14,246` 个、大于 `10 m` 有 `270` 个。最大值虽比 P5 的 `142.2038 m` 小，但大于 `1 m` 的数量比 P5 的 `8,442` 个更多；抽检同一验证视角也确认建筑结构仍可辨，但前景和图像边缘继续存在明显模糊，不能仅凭平均指标升级。
- 按预定停止条件追加的 `0.02` 单变量 1000 步探针耗时 `478.92 s`、峰值额外显存 `959,450,112 bytes`，内部签名 run Manifest SHA256 为 `a646ceb3332178ce226d5673430e9ae22dce372e6ff6266ea1dacdaeed5222e8`。step 1000 full history 为 PSNR `15.584667 dB`、SSIM `0.503213`、depth MAE `5.072061 m`、p10 `15.230879 dB`；随后 selected-model 质量报告在逐帧重算时检测到最低 depth coverage `0.8994976 < 0.9` 并 fail-closed，未生成 `COMPLETE` LPIPS 报告。该权重既继续大幅损伤外观，又跨过覆盖硬门，明确拒绝扩大到 8000 步。

当前为 **PASS（正式证据完整）/ FAIL（深度零回归门）**。澳洲 `gate2_quality_australian_p5_v1` 继续作为优先外观基线，`0.01` 只保留为已验证的 Pareto 点；线性辅助权重蛮力搜索在 `0.02` 按预定条件终止。下一步转向显式 scale/visibility 约束和多目标 checkpoint 选择，重点处理“最大 scale 下降但大于 1 m 的数量反而增加”以及 30k 在 20k 处离散断崖，而不是继续增大深度 loss。

## 35. 当前阶段记录：尺度尾部风险与深度护栏 checkpoint 选择

### 问题现象

正式 P5 尸检显示 1M Gaussian 中只有少数严重超限点，但现有 `scale_upper` 对全部 Gaussian 的 barrier 取均值，再乘 `1e-4`；正常点的海量零值会稀释最坏尾部。`0.01` 深度平衡正式运行又出现 step 5000 depth MAE `4.917 m` 最低、step 6000 PSNR 更高但 depth 回退到 `5.024 m`，旧选择器仍覆盖 `best_golden.pt`。澳洲 30k 运行在 20k 发生离散断崖并依靠早期 best checkpoint 免疫，进一步证明 checkpoint promotion 不能只看单一外观均值。

### 修改文件

- `cloudstudio_3dgs/training/regularization.py`
- `cloudstudio_3dgs/training/golden_eval.py`
- `cloudstudio_3dgs/training/trainer.py`
- `tests/test_training.py`
- `tests/test_golden_eval.py`
- `README.md`
- `docs/IMPLEMENTATION_PLAN.zh-CN.md`

### 修改内容

- `GeometryRegularizationConfig` 新增 opt-in `scale_upper_tail_fraction`。默认 `1.0` 时继续执行原始 `.mean()`，并从序列化 contract 省略该兼容值，保证澳洲 P5 的配置身份与数值语义不变；小于 `1.0` 时只对 barrier 最大的 `ceil(N*fraction)` 个 Gaussian 求均值，使同一弱权重集中约束最坏尺度尾部，不使用会直接破坏外观的硬 world fuse。
- 新增训练 telemetry：`scale_over_limit_fraction` 和 `scale_upper_tail_count`，使短 A/B 能同时检查损失、超限比例和实际参与风险聚合的数量，而不是只看最大值。
- `GoldenEvaluationConfig` 新增 opt-in `max_depth_regression_m`。默认省略并保持 PSNR-only 澳洲 P5 兼容；启用后，候选必须先满足原 PSNR 提升门，同时所有带深度的 golden 帧均为 `MEASURED`，且 depth MAE 不得比当前最佳 checkpoint 回退超过指定米数。深度不可测、NaN 或超限均拒绝 promotion。
- history verifier 使用签名配置重放同一多目标规则；不能通过手改 `best` 或恢复时退化为 PSNR-only 绕过深度护栏。旧 history 不含新字段时仍按原规则验证，不破坏已完成证据。

### 验证方式与当前状态

- tail-risk 单测构造 4 个 Gaussian、其中 2 个超过 `8×` 参考尺度：默认均值保持原值，`tail_fraction=0.25` 只选最坏 1 个并产生更强 penalty；超限比例精确为 `0.5`，反向梯度 finite，非法 `0` 比例 fail-closed。
- checkpoint 单测证明 `+0.2 dB / +0.04 m` 在 `0.05 m` 护栏内可晋级，`+0.3 dB / +0.12 m` 被拒绝，深度 `UNMEASURABLE` 也被拒绝；签名 history 含一个外观更高但深度超限的候选时，verifier 重放后仍认定前一个 checkpoint 为 best。
- 修改后全仓 CPU 套件共运行 `209` 项，为 `208 PASS + 1 SKIPPED`；唯一跳过仍是需要预加载锁定 CUDA 扩展的物理足迹测试。

### 真实 tail-risk successive-halving

- `tail_fraction=0.01` 1000 步运行完成，内部 run Manifest SHA256 `c5a7c8bd95ecd904dfc71ae7ca3455f916991a9b37cc8e50854083aee5e43c275`；124 图报告 `COMPLETE`，PSNR `17.496205 dB`、SSIM `0.534071`、LPIPS `0.602686`、depth MAE `7.881738 m`、coverage min `0.933729`，报告 SHA256 `e28175f10eff990bc836c74f42823af505c0ca9256edec04239ad523562b5366`。但最终 max scale `26.3902 m`、大于 `1 m` 410 个；top 1% 约 4582 个仍远多于约 735 个实际超限点，barrier 被零值稀释，拒绝该比例。
- 按 successive-halving 收紧到 `tail_fraction=0.001` 后，内部 run Manifest SHA256 `e5b7dcbb844220c284b5e3730f9dab5551b5588f4770f11b8768de9cf4326ccc`；124 图报告同为 `COMPLETE`，PSNR `17.383084 dB`、SSIM `0.533708`、LPIPS `0.601801`、depth MAE `7.771762 m`、coverage min `0.959218`，报告 SHA256 `06efbae6da83615d49729c4cce2dce304ac2010fd5f46938868b802a5c5ac89c`。
- 与原澳洲 P5 的相同 seed、step 900 MCMC snapshot 比较，Gaussian 数同为 458129、p50/p95 同为约 `0.10/0.17 m`，但 max scale 从 P5 `11.82 m` 降到 `9.64 m`（约 `18.4%`）；相对 1% tail 的最终 checkpoint，p999 从 `0.9518` 降到 `0.8827 m`，大于 `1 m` 从 410 降到 324。同步 golden PSNR 相对 P5 仅约 `-0.07 dB`，depth 近似，属于小幅外观代价换取可测尺度尾部改善。

### 正式 8000 步结果

- `tail_fraction=0.001` 与 `max_depth_regression_m=0.05` 在 RTX 5070 Laptop 上完成正式 8000 步，耗时 `3140.83 s`、峰值额外显存 `1,961,272,320 bytes`、最终 1M Gaussian、状态 `COMPLETE`；run Manifest SHA256 为 `417cedaf9d6c669471394796d6005baa7684585a2d41308704691243b9133f39`，签名验证通过。
- golden 曲线在 step 1000/2000/3000/4000/5000/6000 达到 `17.4256/18.8812/19.7063/19.8385/20.1029/20.4991 dB`，step 7000/8000 回落到 `20.4816/20.3128 dB`，因此清单正确绑定 `best_golden@6000`，模型 SHA256 为 `e9109b911ca3f909b537192336feb3e02bcb314d97dead6e7ca45ab19217533e`。各次 PSNR 晋级时 depth 均改善或相对当前 best 回退不超过 `0.05 m`；本次没有出现需要真实拦截的超限候选，history verifier 仍按签名规则重放并通过。
- 124 图 selected-model 报告为 PSNR `20.412882 dB`、SSIM `0.594863`、LPIPS-Alex `0.476925`、depth MAE `6.762314 m`、depth coverage min `0.999568`，quality report SHA256 为 `5c4e7079f6f74edfdadb8a1f3b0f0a81f8c266793e1b2b504f59a478d4828b81`，完整签名验证通过。
- 相对澳洲 P5 正式基线，PSNR 下降 `0.107558 dB`、SSIM 下降 `0.000629`、depth MAE 恶化 `0.206793 m`，仅 LPIPS 改善 `0.002425`。这不是综合质量提升，不能替代澳洲 `gate2_quality_australian_p5_v1`。
- selected step 6000 的 scale p50/p95/p99/p999/max 为 `0.1253/0.3492/0.9565/2.0031/154.8749 m`，大于 `1 m` 有 `9,223` 个；澳洲 P5 同为 step 6000 的对应值为 `0.1249/0.3375/0.8977/4.7247/142.2038 m`、大于 `1 m` 有 `8,442` 个。该方案只压低 p999 分位，没有改善 p95/p99、超限数量或单点最大值，证明弱 top-tail 均值仍允许极少数 Gaussian 逃逸。

当前为 **PASS（实现、CPU 回归、真实 1000/8000 步、完整签名与 LPIPS 证据）/ FAIL（正式综合晋级）**。澳洲 P5 保持优先版本；`tail_fraction=0.001 + scale_upper_weight=1e-4` 记录为已拒绝 Pareto 点。后续若继续尺度治理，必须将分位 CVaR 与极值/屏幕足迹保险拆成单变量短探针，不得直接把这条配置升级为默认或混入 POS A/B。

### 屏幕足迹保险单变量探针

- 为避免把两个机制混在一起，下一条 1000 步探针恢复澳洲 P5 的默认 `scale_upper_tail_fraction=1.0`，只将 `screen_clip_enabled` 从 `false` 改为 `true`；`max_screen_fraction=0.15`、hardness `1.5` 和 opacity bump `3.0` 保持已有默认参数。运行状态 `COMPLETE`，耗时 `477.91 s`、峰值额外显存 `922,298,880 bytes`，触发 `157,144` 次 screen clip、world clamp 为 `0`，run Manifest SHA256 为 `0f9a16a272f13c8cbf8bcbe8e8eaa51750261434ece5b079ca5a603233a09931`。
- 124 图报告为 PSNR `16.248149 dB`、SSIM `0.532618`、LPIPS-Alex `0.594732`、depth MAE `7.655046 m`、depth coverage min `0.913219`，quality report SHA256 为 `1c91b28d317df09fa4973d69c3316f0f7477f73e99829779166cd076809aa119`。相对 `tail_fraction=0.001` 的同步 1k 报告，LPIPS 改善 `0.007069`、depth 改善 `0.116716 m`，但 PSNR 下降 `1.134935 dB`、SSIM 下降 `0.001091`、最低覆盖下降 `0.045999`。
- selected step 1000 的 scale p50/p95/p99/p999/max 为 `0.1034/0.1816/0.2907/0.7788/5.7713 m`，大于 `1 m` 有 `262` 个；尺度尾部确实受控，证明该运行路径有效而非空配置，但 15% 足迹阈值在训练早期过度干预可见 Gaussian，外观代价远超晋级余量。因此该参数 **REJECTED**，不进入 8k；若再探索，只允许提高足迹阈值做单变量短探针，不能与 CVaR 同时开启。
