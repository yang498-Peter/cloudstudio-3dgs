# 已知红灯测试基线（`eng/cli-pipeline`，2026-09-15）

无人值守流水线不能在一个"本来就有 19 条红灯"的测试集上判绿。本文件记录**进入本轮工作之前就已经失败**的用例，逐条给出已核实的性质。新增改动必须与这张表逐条一致：**多一条红灯就是回归**。

复核方法：把 `eng/cli-pipeline` 合并前的提交 `594f57a` 解到临时目录跑同一套测试，失败集合与当前完全相同。**通过数只增不减，失败集合必须逐条相同**——门禁看的是失败集合，不是通过数：

| 时点 | 通过 | 失败 |
| --- | --- | --- |
| 合并前 `594f57a` | 1168 | 19 |
| 合并 + SDK 两包 + 填充接线 | 1440 | 19 |
| + GPU 租约 CLI（7 条） | 1447 | 19 |
| + SDK 估算干跑 `discover`（33 条） | 1480 | 19 |
| + bundle 构建、adopt 遥测地板、参考模型缺席门控（2026-09-18） | **1547** | **19** |


```bash
cd C:\Peter\cloudstudio-3dgs-eng && set PYTHONPATH=. && python -m pytest tests/ -q
```

## 表

| 用例 | 条数 | 已核实的性质 |
| --- | --- | --- |
| `test_pipeline_state.py::CheckpointInspectionTests`（2 条） | 2 | **不是安全漏洞，是过期断言**。用例断言拒绝理由里含 `data.pkl`，实际 torch 现在报 `RuntimeError: [enforce fail at inline_container.cc:180] . file in archive is not in a subdirectory`。**fail-closed 行为本身正确**：`info.loadable` 仍为 `False`，坏检查点照样被拒。只是 torch 版本换了措辞。 |
| `test_training_presets.py`（3 条） | 3 | `trainer_preset` 与 `TRAINER_PRESETS` 在 `geometry_regularization` 上不一致，预设声明与当前默认值漂移。属于配置契约漂移，需要一次显式判定：是预设该更新，还是默认值改错了。 |
| `test_diagnostic_set.py::test_repo_diag_configs_satisfy_contract`（9 个子项） | 9 | 仓库里 9 份 diag 配置不再满足当前研究日程合同。都是**已判完的诊断臂配置**，不参与交付。 |
| `test_ab_matrix.py`（1 条） | 1 | A/B 矩阵的"共享输入已签名、每臂单一变量"检查失败，同属配置契约漂移。 |
| `test_contribution_attribution.py::RealRasterizerContributionTest`（3 条） | 3 | 需要真实光栅化器，**要 CUDA**。本次采集时 GPU 被 25k 先验训练占用。属于未执行，不是失败。 |
| `test_training.py::RenderScaleContractTests::test_rendered_footprint_matches_linear_metric_scale` | 1 | 同上，需要真实渲染。 |

## 结论

- **4 条（`contribution_attribution` 3 + `RenderScaleContract` 1）是 `NOT_RUN`**，不是红灯：GPU 空闲时必须重跑才能定性。
- **2 条是过期断言**，门禁语义完好，改测试即可，但不要顺手改——它属于别的任务，改动要单独判定。
- **13 条是配置契约漂移**（预设 3 + diag 9 + A/B 矩阵 1），需要一次显式判定，不能靠放宽断言消掉。

在这 19 条清掉之前，**SDK 的质量门禁不得把"测试全绿"写进放行条件**；能写的是"与本基线逐条相同"。
