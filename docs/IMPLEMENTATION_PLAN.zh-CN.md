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
