# 研究基线冻结（2026-09-11）

分支 `fix/loss-parity-and-split` 在本提交处冻结为研究基线，标签 `research-baseline-2026-09-11`。此后算法臂的结果继续追加到 `01_first_batch_ledger.zh-CN.md`，但配方、评估器与工具的接口以此为准；工程化在 `eng/cli-pipeline` 分支上进行，不改变这里的结论。

## 基线内容
- 评估器：`tools/sharpness_metrics._load_backend(config, sh_degree)` 以模型阶数覆盖配置；`tools/roundtrip_checkpoint_ply.py` 带四个阴性对照；`tools/score_compare_sharpness.py`、`tools/score_offtrajectory_strips.py`（`name=dir`）。
- 身份：`tools/freeze_run_identity.py` → `identity/*.json`；`cache_dependency_dag.json` 列出 23 个受保护缓存根。
- 生命周期：`_opacity_summary` 独立 generator；`lifecycle_capacity_cap / lifecycle_capacity_rejected` 遥测；`post_refine_cull_every / until`（默认关闭，本数据集上判负）。
- 最佳配方（Tile_0）：`run_configs/house0305_tiles/v9/tile0_R1_range0_20k.json` = G9 配方 + cap 15M + `lidar_range_weight 0`；训练视角锐度 0.330（参考 0.396），离轨迹 0.553。四切片版本 `tile{1,2,3}_R1d_20k.json`。
- 已判臂：R0、C1（赢）、D1、C2、K1、K2、A1、L1、R1（赢）；室内 tile1_C1d、tile1_R1d。全部读数在台账 §6。
- 事故记录：宿主进程退出（09-06）、等待脚本误触发导致双训练器 + 宿主重启（09-07 → 09-10）。

## 未关闭的门禁
- torch-cpu CI 通道未在 Actions 上验证；cuda 通道 4 个失败属他人 preset WIP。
- 室内切片只有参考的 19%，三种配方无差异；离轨迹全场景约 55%。
- 全场景 R1d 合并与同帧对比正在进行，结果追加到台账。

## 工程化分支目标（`eng/cli-pipeline`）
1. 编排入库：训练臂 / 交付 / 打分 / 队列的 Python CLI，可断点续跑，配置文件替代写死路径。
2. 去掉 `tools/audit_colocated_morphology.py`、`tools/build_offtrajectory_compare.py`、`tools/score_offtrajectory_strips.py` 中的本机路径。
3. 第二数据集（`0614_full_house_S1`）跑通预处理链到切片规划，再排训练。
4. 预编译 gsplat 扩展、显存自适应 cap、失败重试。
