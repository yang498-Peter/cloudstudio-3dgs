# cloudstudio-3dgs 持续优化实施计划

更新时间：2026-08-20

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

当前机器只有 PyCOLMAP 4.1.1，尚未安装锁定的 HLoc/LightGlue 可选运行时，因此真实 ALIKED 特征、LightGlue 匹配、HLoc 三角化和真实 BA 均为 `NOT_RUN`。本机 PyCOLMAP/Ceres 合成测试实际执行；CPU CI 不安装该可选包时显式 `SKIPPED`，其余契约测试仍必须通过。真实重投影 p50 改善至少 30% 的验收门没有证据，不能声明 PR-10 真实候选已接受；本阶段当前完成的是源码契约、真实训练匹配图和合成求解闭环。
