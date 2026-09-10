# quality_recovery_v2 — 按《澳洲B机_3DGS全链路审计与Agent执行任务书_2026-09-11》执行

任务书于 2026-09-11 由用户在会话中提供（连同 `3DGS_B_machine_audit_agent_pack_2026-09-11.zip`）。执行开始时（2026-09-11 06:10 AEST）桌面上已找不到这两个文件，任务书正文以会话中读取的内容为准；随附的 `audit_gs_config.py` 未取得，WP02 的日程审计改由仓库内 `cloudstudio_3dgs/training/schedule_audit.py` + `tools/audit_training_schedule.py` 实现（同时满足任务书要求的 `resolved_schedule` 纯函数）。若需用原脚本复核，请重新提供该 zip。

执行顺序遵循任务书：WP00 → WP01/WP02 → WP03 → WP04/WP05 → 依诊断选择 WP06/WP07 → WP08/WP09；WP10 的 CPU 工作并行。不一次性排满整卡。

## 与任务书对应的落地状态

| 项 | 状态 | 文件 |
| --- | --- | --- |
| 正在跑的 tile3（R1d 交付）不终止，完成后归档为 R1d 候选，不宣布胜过 G9 | 执行中 | `C:/Peter/3dgs-runs/house0305_sop/delivery_R1d/` |
| 队列里与任务书冲突的后续臂（B1 混杂、E1 仅加步数、P2 未做位姿依赖审计、tile1_T1a）已停用 | 已做：配置改名 `*.json.parked`，链在交付后自然结束 | — |
| WP00 运行身份 | 已出 | `00_runtime_identity.json`（含 gsplat commit/patch/扩展 hash、两个仓库 HEAD、正在运行的训练进程；注意 venv 跳板会让同一训练显示为父子两个 `train_gsplat` 进程） |
| WP00 CLI 四个 P0（完成状态机、配置冻结、最终 PLY 评分、原子 GPU 租约）+ 故障注入测试 | 子任务进行中（`eng/cli-pipeline`） | — |
| WP02 日程审计（随附脚本） | 已跑：`audit_pack/` 8 项自测通过；对仓库配置与 as-run 各跑一次，**as-run sha 与仓库配置一致**（tile0 `40b4d71a…`、tile1 `7728ae57…`）。tile0 R1：135 次常规 refine、最后一次 13900、`late_prune_threshold_reachable=false`、末步名义 means LR 1.8454e-6 = 终值 1.6e-7 的 11.53×；告警 S001（20k 截断 42640 日程）、S002（21320 不可达）、G001（试 rgb_only 增殖信号）、E001（曝光无零均值/锚定）、A001（SH 从头全开） | `02_schedule_audit_pack_repo.json`、`02_schedule_audit_pack_asrun.json` |
| WP02 日程审计（仓库内 `resolved_schedule`） | 子任务进行中，完成后与随附脚本数字交叉核对 | `02_schedule_audit.json`、`02_schedule_events.csv` |
| WP01 评估协议 / WP03 室内诊断 | 待 WP02 出结果后启动 | — |
| WP10 house0614 子集 | 子集 manifest/masks/split 已生成（`eng` 分支 `eda063b`）；运行手册已出（`eng` 分支 `docs/2026-09-11_house0614子集运行手册.zh-CN.md`：23 步、8 步需 GPU、磁盘 57–75 GB）。**阻塞**：C: 仅 50 GB 空闲；Mask R-CNN 权重不在本机；时间同步审计需要一个已训练 checkpoint（GPU）且工具无 CPU 渲染路径（`time_sync/BLOCKED.json`）。GPU 全链验证等 house0305 质量门 | — |

台账 `../quality_recovery_v1/01_first_batch_ledger.zh-CN.md` 与 `02_research_loop_plan.zh-CN.md` 中"锐度涨过噪声带即赢、两赢即全场交付"的机械规则自本日起废止，判臂按任务书 §12 的多指标、室内外双控、配对/多 seed 口径执行。
