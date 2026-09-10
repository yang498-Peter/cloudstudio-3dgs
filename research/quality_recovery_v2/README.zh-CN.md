# quality_recovery_v2 — 按《澳洲B机_3DGS全链路审计与Agent执行任务书_2026-09-11》执行

任务书于 2026-09-11 由用户在会话中提供（连同 `3DGS_B_machine_audit_agent_pack_2026-09-11.zip`）。执行开始时（2026-09-11 06:10 AEST）桌面上已找不到这两个文件，任务书正文以会话中读取的内容为准；随附的 `audit_gs_config.py` 未取得，WP02 的日程审计改由仓库内 `cloudstudio_3dgs/training/schedule_audit.py` + `tools/audit_training_schedule.py` 实现（同时满足任务书要求的 `resolved_schedule` 纯函数）。若需用原脚本复核，请重新提供该 zip。

执行顺序遵循任务书：WP00 → WP01/WP02 → WP03 → WP04/WP05 → 依诊断选择 WP06/WP07 → WP08/WP09；WP10 的 CPU 工作并行。不一次性排满整卡。

## 与任务书对应的落地状态

| 项 | 状态 | 文件 |
| --- | --- | --- |
| 正在跑的 tile3（R1d 交付）不终止，完成后归档为 R1d 候选，不宣布胜过 G9 | **tile3 在 step ~8800 崩溃**（2026-09-11 06:29，`torch.AcceleratorError: CUDA error: unknown error`，发生在 `_grow_mipmap` 的 duplicate 阶段，人口 12.48M、cap 15M、崩溃前显存 14.0 GiB）。旧链在崩溃后仍按"latest.pt 存在"继续评估 step-5000 的中间 checkpoint 并准备合并——正是任务书 P0-1 描述的路径，已在合并前手动终止。崩溃目录保留为 `tile3_R1d_20k.crashed_step8800/`（日志在，3 GB 中间 checkpoint 已删）。按"OOM 重试改变配置即新臂"：tile3 以 `tile3_R1d_cap13m_20k`（仅 cap 15M→13M）重跑，交付改走加固后的 CLI（tile1/2 复用已验证完成的 R1d checkpoint，tile3 新训，合并后回读 PLY 评分，输出到 candidate 目录） | `C:/Peter/3dgs-runs/house0305_sop/delivery_R1d/` |
| 队列里与任务书冲突的后续臂（B1 混杂、E1 仅加步数、P2 未做位姿依赖审计、tile1_T1a）已停用 | 已做：配置改名 `*.json.parked`，链在交付后自然结束 | — |
| WP00 运行身份 | 已出 | `00_runtime_identity.json`（含 gsplat commit/patch/扩展 hash、两个仓库 HEAD、正在运行的训练进程；注意 venv 跳板会让同一训练显示为父子两个 `train_gsplat` 进程） |
| WP00 CLI 四个 P0（完成状态机、配置冻结、最终 PLY 评分、原子 GPU 租约）+ 故障注入测试 | 已入库（`eng/cli-pipeline`，108 项测试）：`job_state.json` 状态机 RUNNING→CHECKPOINTED→TRAINING_COMPLETE→EVALUATED→QUALITY_ACCEPTED→PUBLISHED（FAILED / CONTROLLED_PAUSE 独立），完成判定 = checkpoint 可加载 + 步数达标 + 退出原因合法 + 时间戳新于作业；`config_frozen.json` 冻结，同名改配置拒绝；交付先评 merged.pt 再导出、按 0/0.01/0.05 出阈值对照、回读导出 PLY 评分并绑定其 sha，默认发布到 `exports/candidate_<TAG>/`；`gpu.lock` O_EXCL 租约（两进程竞争恰一个成功）。遗留臂（无 job_state）按日志+checkpoint 一次性"adopted"，这是有意的宽松，需要时可收紧；DA2/背景/AT 尚未接入租约 | — |
| WP02 日程审计（随附脚本） | 已跑：`audit_pack/` 8 项自测通过；对仓库配置与 as-run 各跑一次，**as-run sha 与仓库配置一致**（tile0 `40b4d71a…`、tile1 `7728ae57…`）。tile0 R1：135 次常规 refine、最后一次 13900、`late_prune_threshold_reachable=false`、末步名义 means LR 1.8454e-6 = 终值 1.6e-7 的 11.53×；告警 S001（20k 截断 42640 日程）、S002（21320 不可达）、G001（试 rgb_only 增殖信号）、E001（曝光无零均值/锚定）、A001（SH 从头全开） | `02_schedule_audit_pack_repo.json`、`02_schedule_audit_pack_asrun.json` |
| WP02 日程审计（仓库内 `resolved_schedule`） | 已出（`825c54c`）：与随附脚本交叉一致（末步 LR 1.845e-6 / 11.53×，tile1 1.290e-6 / 8.06×）。**新发现**：`learning_rates.means=1.6e-5` 不是优化器实际底数——`scale_calibration.means_step_fraction`（默认 0.0032 × 参考尺度）在 `trainer.py:3588` 覆盖它，实际底数 2.49e-5（tile0）/ 1.98e-5（tile1），末步 2.87e-6 / 1.59e-6。事件：135 次 grow/cull、45 次 reset，最后一次出生 13900，之后每张图只剩 2.9–3.3 次访问。每切片 3 项不一致（截断日程、LR 底数、晚期阈值不可达），仓库配置与 as-run 零差异。S1 联动字段清单见提交说明；因训练器预检要求 `max_steps = 20×视角数`、`prune_switch = max_steps//2`，S1 需要一个显式命名的研究日程合同（子任务进行中） | `02_schedule_audit.json`、`02_schedule_events.csv` |
| WP01 评估协议 | 子任务进行中（分组 split 提案、48 个 battery 视角是否留出、A/B/C RenderSpec、六类 ROI、阴性对照规范） | `01_*` |
| WP03 室内诊断 | 待 WP01 的 ROI 登记出来后启动，复用同一组 ROI | `03_*` |
| WP04 §7.1 `rgb_only` 增殖信号 | 已钉住：`densification_gradient.py` 纯 torch 助手（两次反向 = 一次总反向的叶梯度；means2d 增长梯度只含 RGB；absgrad 快照不被第二次反向覆盖；审计不推进 RNG；拓扑变化后梯度/Adam/血统对齐），11 项测试；梯度审计修正：LiDAR range/normal 项此前**漏乘阶段倍率**，现报告 raw/weight/stage/effective、各参数组与 means2d 范数、成对余弦，几何正则对 means2d 记 N/A。**G0/G1 对照必须放在 `post_optimizer_gsplat` 顺序上**（vendor 顺序在 `trainer.py:1287` 拒绝 rgb_only），即 G0/G1 同时也是矩阵里的 O0→O1；两臂都保持 `absgrad false`、`footprint_weighted`。另记：`footprint_weighted` 永远读 `.grad` 不读 `.absgrad`，O2（absgrad）臂设计时要处理 | `tests/test_densification_gradient_source.py` |
| WP10 house0614 子集 | 子集 manifest/masks/split 已生成（`eng` 分支 `eda063b`）；运行手册已出（`eng` 分支 `docs/2026-09-11_house0614子集运行手册.zh-CN.md`：23 步、8 步需 GPU、磁盘 57–75 GB）。**阻塞**：C: 仅 50 GB 空闲；Mask R-CNN 权重不在本机；时间同步审计需要一个已训练 checkpoint（GPU）且工具无 CPU 渲染路径（`time_sync/BLOCKED.json`）。GPU 全链验证等 house0305 质量门 | — |

台账 `../quality_recovery_v1/01_first_batch_ledger.zh-CN.md` 与 `02_research_loop_plan.zh-CN.md` 中"锐度涨过噪声带即赢、两赢即全场交付"的机械规则自本日起废止，判臂按任务书 §12 的多指标、室内外双控、配对/多 seed 口径执行。
