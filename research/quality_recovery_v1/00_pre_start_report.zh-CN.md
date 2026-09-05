# 质量恢复 v1 — 开工前报告（第一批：WP00 / WP01 不训练验证 / WP02 小型测试）

依据：`CloudStudio_3DGS_质量提升研究与Agent执行方案_2026-09-05.md` 第 16 节启动指令。第一批不启动任何新的长训练、不扩 cap、不开全部标定自由度、不引入大型框架。本报告只写已核实的事实；未执行的门禁标 NOT_RUN。

## 1. 当前身份（2026-09-05 实测）

| 项 | 值 |
| --- | --- |
| 工作树 | `C:/Peter/cloudstudio-3dgs-work`，分支 `fix/loss-parity-and-split`，HEAD `7a2b9e9`（方案评审基点 `0fd4881` 之后一个提交） |
| B 机树 | `C:/Peter/cloudstudio-3dgs-gate1`，`machine-b/uk-quality` @ `34e5415`；含**他人未提交**改动 `cloudstudio_3dgs/training/presets.py`、`tests/test_training_presets.py`（`world_shrink_factor: None`），本批不触碰 |
| master 树 | `C:/Peter/cloudstudio-3dgs` 本地 `806c49f`，落后远端 `e284e363`（`e284e363..HEAD` 为空） |
| Python / torch | 3.12.10 / 2.11.0+cu128，CUDA 12.8，RTX 5070 Ti 16 GiB |
| gsplat | 1.5.3，检出 `external/gsplat-clean` @ `f2d14131`，工作树脏 5 文件；`git apply --check -R` 锁定补丁在该树上**干净反向应用**，即 树 == 锁定 commit + 锁定补丁（补丁 sha256 `85a800a9…` 与 `upstream/gsplat.lock.json` 一致） |
| 编译扩展 | `C:/Peter/cloudstudio-3dgs/.torch-ext/gsplat_cuda/gsplat_cuda.pyd`，sha256 `60e7a458…caaf738e`，107,533,824 B；`TORCH_EXTENSIONS_DIR` 由 `env_machine_b.cmd` 设定。不经 `env_machine_b.cmd` 启动的 venv 会去 `…/Claude_…/LocalCache/…/torch_extensions` 找扩展并失败——这是本地"CUDA 扩展不可用"错误的来源，不是代码缺陷 |
| 确定性开关 | `cudnn.deterministic=False`，`cudnn.benchmark=False`，`CUBLAS_WORKSPACE_CONFIG` 未设 |
| G9 实跑分支 | `config_as_run.json`：`default_strategy.exact_mipmap_lifecycle=True`、`lifecycle_execution_order=pre_optimizer_vendor`、`growth_metric=footprint_weighted`、`absgrad=False`、`sh_degree=1`、`cap_max=12,000,000`（G9d 为 10M）、`seed=42`、`max_steps=42640` + `controlled_stop_after_steps`（20k 压缩日程） |
| 训练进程 | 无 |
| 磁盘 | C: 剩余 65 GB |

## 2. 既有证据（本批复核后的状态）

### 2.1 方案第 2 节两条更正——均已用代码/遥测证实
- **cap_max 不是失效的。** `backend.py:204` 把 `cap_max` 写进 `settings["capacity_cap"]`；`step_post_backward` 在 `exact_mipmap_lifecycle=True` 时进 `_grow_mipmap`，`capacity_cap` 在 ~799 行参与 topk 预算、在 ~1485 行决定 `relaxed_cull_at_capacity`。G9/G9d 都走这条分支。但 `capacity_rejected_count` 算完没有进 `progress.jsonl`（遥测键只有 `lifecycle_growth_eligible / grad_only_candidates / clone_parents / split_parents`），所以"两次都没顶到 cap"只能从人口峰值（1075 万 / 897 万 < cap）间接推断。旧文档与 memory 中"cap 完全失效"的表述已更正。
- **`_opacity_summary` 的 `randperm` 确实污染训练 RNG，而且在 G9 真的触发了。** 阈值 100 万元素；G9 候选 tile0 `lifecycle_clone_parents` 峰值 1,847,926；G9d tile0 有 42 次 refine 事件父集 >1M，tile3 28 次，tile1 0 次。该调用位于 `_grow_mipmap` 相对第 280–281 行，`split_selected` 在相对第 307 行，gsplat `strategy/ops.py` 的 split 用无 generator 的 `torch.randn`——顺序上遥测先改流、split 后取样。新测试 `tests/test_lifecycle_rng_isolation.py` 在修复前三项失败（CPU 流前进、CUDA 流前进、两次读数不一致），符合预期。

### 2.2 CI 真实状态（本地复现）
| 通道 | 结果 |
| --- | --- |
| 系统 Python（无 torch，≈CI） | `unittest discover`：Ran 499，errors=37，skipped=199。错误来源：26 个模块导入 `torch` 失败、9 个文件顶层 `import pytest`、1 个 pycolmap、1 个 gsplat。跳过来源全部是 torch/pycolmap/CUDA 缺失 |
| 训练 venv，未加载 `env_machine_b` | Ran 716，errors=8：4 个 preset `geometry_regularization` 不匹配（他人 WIP 范围）+ 4 个"CUDA 扩展不可用"（环境问题，见上） |
| 训练 venv + `env_machine_b` | 上述 4 个 CUDA 用例通过；剩余仅 preset 4 项 |
| pytest（训练 venv） | 收集 772（764 passed + 8 failed）。**56 个函数式测试**分布在 9 个文件（`test_adaptive_tiling_and_sky`、`test_gaussian_residency_model`、`test_mipmap_tile_geometry`、`test_mipmap_type2_contract`、`test_shape_knob_plumbing`、`test_surface_only_route`、`test_tangent_isotropy`、`test_time_sync_selection`、`test_world_shrink`）没有 `TestCase`，`unittest discover` 静默收集 0 个；单独用 pytest 跑 56 passed |

### 2.3 保留的历史结论（未重测，按方案要求降级为"线索"）
- 密度（grow_grad2d 5e-5）是唯一移动过训练视角锐度的旋钮；G9 对 G1/G8（同 20k）+45%。
- 四切片 G9d 合并后对 F6 仅 +12%，低于复跑噪声；同配方复跑人口差 16%（现知其中一部分是上面的 RNG 污染，不全是 GPU 非确定性）。
- 5.9 px 相邻帧分歧探针不是闭环配准，只是不一致性的读数。

## 3. 缺失输入
- G9/G9d 的 `progress.jsonl` 里没有预算拒绝计数——无法回溯证明 cap 在那两次里是否切过 topk；只能靠新增的 `lifecycle_capacity_rejected` 在**下一次**训练里得到。
- 没有 CPU 版 torch 环境：torch-cpu 通道只能在 GitHub Actions 上验证，本地标 NOT_RUN。
- GitHub-hosted runner 没有 GPU：CUDA 通道只能在 B 机跑，脚本已提供。
- `research/` 目录此前不存在，本批新建 `research/quality_recovery_v1/`。
- uv 未安装在本机：不能改 `uv.lock`；CI 里 pytest / CPU torch 用 `uv pip install --python .venv` 在锁之外安装，不动锁文件。

## 4. 拟修改 / 新增文件
| 文件 | 变更 | 状态 |
| --- | --- | --- |
| `cloudstudio_3dgs/training/default_strategy_adapter.py` | `_opacity_summary` 用独立 `torch.Generator`（与 `_sampled_quantile` 同法）；`_last_growth_event` 增 `capacity_cap / capacity_available_before_growth / capacity_rejected_count` | 待落地（本报告写完后执行） |
| `cloudstudio_3dgs/training/trainer.py` | progress 指标增 `lifecycle_capacity_cap`、`lifecycle_capacity_rejected` | 同上 |
| `tests/test_lifecycle_rng_isolation.py` | 新增，4 用例（含 CUDA 变体） | 已写，修复前 3 失败 |
| `tests/test_default_strategy_adapter.py` | 追加 `CapacityBudgetOnSelectedPathTests`（`pre_optimizer_vendor` 路径上 cap=5、3 候选 → 拒绝 1；无 cap → 报 None/0） | 待追加并先看失败 |
| `tools/freeze_run_identity.py` | 新增：config_as_run、所有输入 manifest、初始化、checkpoint、编译扩展、gsplat 检出的哈希与 payload 摘要 | 已写，待运行 |
| `tools/roundtrip_checkpoint_ply.py` | 新增：checkpoint→PLY→checkpoint 张量比对 + 同 backend 渲染差分 + 四个阴性对照（重复渲染噪声底、DC-only SH、相机平移 3.5 cm、换背景） | 已写，待运行 |
| `tests/check_collection.py` | 新增：JUnit 通道策略审计（禁止整体跳过、零收集文件、未解释错误） | 已写 |
| `.github/workflows/unit-tests.yml` | 拆 `cpu-tests` / `torch-cpu-tests`，pytest 收集 + 策略审计 + 产物上传；`patch-applicability` 不变 | 已写，CI 未跑 |
| `tools/run_cuda_test_channel.cmd` | B 机 CUDA 通道脚本 | 已写 |
| `docs/2026-09-02_house0305根因分析与修复路线.zh-CN.md` | 第 464 行 cap 失效段落更正 | 已改，562→562 行，无乱码 |
| 不改 | `presets.py`、`test_training_presets.py`（他人 WIP）；vendor DLL；任何 cache | — |

## 5. 测试计划
1. `tests.test_lifecycle_rng_isolation` + `CapacityBudgetOnSelectedPathTests`：先失败后通过。
2. 回归：`test_default_strategy_adapter`、`test_vendor_parity_lifecycle`、`test_mipmap_gate` 全绿；训练 venv + `env_machine_b` 全量 `pytest tests`，预期只剩他人 WIP 的 4 个 preset 失败。
3. CPU 通道本地复现：临时 torch-free venv（pytest + 锁内四个包）跑 `pytest --continue-on-collection-errors` → `check_collection.py --channel cpu`，必须为 0 违规。
4. WP00：`freeze_run_identity.py` 对 G9 候选、G9d 四切片、`delivery_g9/merged.pt`+两份交付 PLY 出 6 份 identity JSON。
5. WP01：`roundtrip_checkpoint_ply.py` 对 `delivery_g9/merged.pt`（19.3M，SH1）用 `delivery_eval_v9.json` 4 帧：张量 max_abs ≤ 1e-6；渲染差分 ≤ 重复渲染噪声底；三个阴性对照都要大于往返差分。产出差分图与 report.json。
6. 证据台账 `evidence_registry.json`、`cache_dependency_dag.json`、`ci_failure_triage.md`，附 PASS/FAIL/NOT_RUN 清单与下一步最小 WP03/WP04 任务。

## 6. 资源与风险
- 计算：WP00 哈希约 13 GB 文件（分钟级）；WP01 单次 GPU 渲染 4 帧 × 6 次，显存需求与既有 compare 工具同量级；无训练。
- 风险：`torch==2.11.0` CPU wheel 在 PyTorch 索引上的可用性只有 CI 跑过才能确认；`--continue-on-collection-errors` 让 pytest 退出码非 0，通道绿红由 `check_collection.py` 决定，需在 CI 上验证一次。
- 共享工作树：gate1 有他人未提交改动，本批所有改动只在 work 树，提交前按 `--only` 路径限定；不 push、不 merge，等负责人授权。
- 本机 bash heredoc 会破坏含反斜杠转义的脚本，所有脚本通过 Write 工具落盘。
