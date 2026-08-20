# 2026-08-21 新机器接力状态与持续优化路线

> 本文档记录 2026-08-21 在新机器(RTX 5070 Ti 16GB)上的接力现状、可执行的分阶段优化路线
> 和持续推进循环协议。逐阶段的正式验收记录仍写入 `docs/IMPLEMENTATION_PLAN.zh-CN.md`。

## 1. 本机现状盘点(2026-08-21 实测)

| 项目 | 状态 | 证据 |
|---|---|---|
| GPU | **RTX 5070 Ti 16GB**(Blackwell sm_120,较历史机器 8GB 翻倍) | `nvidia-smi`:Driver 595.97 / CUDA 13.2 |
| Python | 3.12.10 ✔(锁定要求 3.12) | `python --version` |
| VS2022 BuildTools | 已安装 ✔ | `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools` |
| CUDA Toolkit / nvcc | **缺失**;本机无管理员权限,官方安装器不可用 | `nvcc` not found;IsInRole(Administrator)=False |
| PyTorch | 初始缺失 → 按锁定 `2.11.0+cu128` 安装至 `.venv-train` | `upstream/*.lock.json` |
| CPU 工具环境 | `scripts/bootstrap.ps1` 成功;doctor `PASS` | 本文档同日记录 |
| CPU 测试 | `90 run / 0 fail / 7 skipped(可选依赖)` | `python -m unittest discover -s tests` |
| **真实 gs2 数据** | **不在本机**(75GB 原始记录在旧机器 G 盘) | 本机无 G 盘;`C:\Peter\testdata` 只有全景补充包与 LAZ |
| GitHub 远端 | `origin/master` 可读,SHA `92eabf8` 与本地一致 | `git ls-remote` |

关键结论:

1. 本机显存翻倍,是推进 GPU 训练门槛(合成验收、完整 MCMC、PR-13 粗到细)的最佳机会。
2. 无管理员权限 ⇒ CUDA Toolkit 走 **pip NVIDIA 组件 wheel**(`nvidia-cuda-nvcc-cu12==12.8.93` 等)
   或解包安装器的免管理员路线,`CUDA_HOME` 指向组装目录。
3. 所有"真实 gs2"验收门(全量深度 loss、同配置回归、mipmap 对标)在数据迁移前保持 `NOT_RUN`,
   不得用合成结果冒充。

## 2. 差距分析(对照 IMPLEMENTATION_PLAN 阶段表)

已完成(源码/合成/部分真实闭环):PR-00 ~ PR-05、PR-07 ~ PR-11。
未完成且**本机可推进**的缺口,按价值排序:

| 缺口 | 影响 | 本机可行性 |
|---|---|---|
| G1 训练环境未在本机建立 | 一切 GPU 门槛的前置 | ✔ 免管理员路线 |
| G2 `quat_scale_to_covar_preci_fwd` 未注册 ⇒ **完整 MCMC 噪声/致密化从未跑通** | MCMC 是画质核心;当前只能噪声=0 的退化训练 | ✔ 需查 gsplat 1.5.3 Windows 构建为何缺算子 |
| G3 中断式 GPU checkpoint resume 未验证 | 长训练可靠性 | ✔ 合成场景即可验证 |
| G4 PR-12 Rig pose refinement 未实现 | 位姿残差直接限制清晰度上限 | ✔ 合成测试先行 |
| G5 PR-13 粗到细原图训练未实现 | 2912² 原图直训是画质/显存关键 | ✔ 16GB 显存正好是目标环境 |
| G6 LPIPS 评测缺失(报告长期 PARTIAL) | 质量验收不完整 | ✔ 兼容环境安装 lpips |
| G7 真实 gs2 全链路回归 | 最终验收 | ✘ 阻塞:需数据迁移 |
| G8 PR-10 真实特征/BA(HLoc/LightGlue) | 位姿精化 | 部分:可装锁定运行时,真实数据仍缺 |

## 3. 分阶段优化路线(本轮 plan of record)

每阶段完成定义与 IMPLEMENTATION_PLAN §1 相同:最小闭环、合成测试先行、fail-closed、
记录验证结果、精确暂存提交并推送。

### 阶段 A:训练环境补齐(G1,当日)

1. `.venv-train`:`torch==2.11.0+cu128` + pip CUDA 12.8 组件(nvcc/cudart/cccl)。
2. 干净检出 gsplat `f2d1413`(v1.5.3,无补丁),`TORCH_CUDA_ARCH_LIST=12.0`、
   `VSLANG=1033`、`DISTUTILS_USE_SDK=1` 编译安装(editable,满足 clean_vcs_commit 契约)。
3. 验收:`verify_gsplat_runtime` 通过;`run_synthetic_training_acceptance.py --steps 80`
   收敛(loss 改善 ≥20%),与 `baselines/gs2_trainer.baseline.json` 记录量级一致。
4. 把本机环境结论(免管理员 CUDA 路线、坑位)写入 IMPLEMENTATION_PLAN 新阶段记录。

### 阶段 B:完整 MCMC Windows 算子修复(G2)

1. 诊断锁定构建中 `quat_scale_to_covar_preci_fwd` 缺失根因
   (预期:编译单元被排除 / 注册宏未走到 / 旧机器构建部分失败)。
2. 修复优先级:构建配置修复 > 上游已修 commit 对比 > 最小补丁(若必须打补丁,
   需回到带 patch 的锁定协议并更新 `cloudstudio_trainer.lock.json` + NOTICE)。
3. 验收:合成验收在 `noise_injection_stop_iter` 默认(不为 0)下全程通过,
   Gaussian 数量随 MCMC 致密化变化,无 NaN。

### 阶段 C:GPU 可靠性门(G3、G6)

1. 合成场景中断-恢复:训练至 N/2 步 kill,resume 后与不间断运行 loss 曲线一致性验证。
2. 安装 lpips(不动锁定 torch),对合成 validation 出完整 PSNR/SSIM/LPIPS 报告,
   报告状态由 `PARTIAL` 升级为完整。

### 阶段 D:PR-12 Rig pose refinement(G4)

按 IMPLEMENTATION_PLAN 阶段表:每 Rig frame 一个 SE(3) 增量、先验约束、
左右基线不变式测试、无改善自动回退。合成收敛测试先行;真实消融保持 `NOT_RUN`。

### 阶段 E:PR-13 粗到细原图训练(G5)

A/B/C 分辨率阶段、valid-aware crop、OOM 降级、checkpoint 跨分辨率。
16GB 显存下的目标:2912² 原图不 OOM(合成大图先验证机制)。

### 阶段 F:真实数据回归(G7,阻塞待数据)

数据迁移到本机后:全量 1238 图 + 完整深度缓存 + 新 split 重训,
124 张 validation 的 masked PSNR/SSIM/LPIPS + LiDAR depth 指标,对标 mipmap。
**数据迁移是外部依赖,需人工把 `G:\S1\2026-06-17_12-40-48gs2`(75GB)拷到本机。**

## 4. 持续推进循环协议

每轮循环(自动):

1. `git status --short` + `git log -3` 识别现场;读本文件与 IMPLEMENTATION_PLAN 尾部,
   确认当前阶段与未关闭门槛。
2. 推进当前阶段最小下一步;长任务(编译/训练)后台化,轮内验证产物。
3. 每完成一个可验收单元:跑相应测试车道 → 更新 IMPLEMENTATION_PLAN 阶段记录 →
   精确暂存本任务文件 → 提交并推送 → 核对 `origin/master` SHA 一致。
4. 阻塞项(如 G7 数据)明确写入本文件 §5 并继续推进非阻塞阶段,不空转。

## 5. 当前阻塞与外部依赖

- **G7 真实 gs2 数据迁移**:等待人工拷贝 75GB 原始记录 + `process/` 解算成品到本机
  (建议路径 `C:\Peter\testdata\S1\2026-06-17_12-40-48gs2`,ASCII 无空格)。
- mipmap 对标产物(`gs.ply` 4855 万高斯)同样在旧机器,对标阶段一并迁移。
