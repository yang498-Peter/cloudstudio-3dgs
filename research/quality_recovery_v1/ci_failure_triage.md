# CI 失败分诊（WP00，2026-09-05）

对照 GitHub Actions run 33935708607（499 tests / 37 errors / 201 skipped）与本地三通道复现。

## 根因分类

| # | 现象 | 根因 | 归类 | 处置 |
| --- | --- | --- | --- | --- |
| 1 | 26 个模块 `ModuleNotFoundError: torch` | 锁文件只含 laspy/numpy/pillow/scipy，torch 不在 `uv sync --frozen` 内；这些模块在顶层 `import torch` 而不是守卫导入 | 通道设计 | 新增 torch-cpu 通道让它们真的跑；cpu 通道把它们按模块计为 NOT_RUN 并列出 |
| 2 | 9 个文件 `ModuleNotFoundError: pytest` | 9 个函数式测试文件顶层 `import pytest`，而 CI 用 `unittest discover`，pytest 未安装 | 收集缺陷 | 两个通道改用 pytest 收集；本地 pytest 收集 772 vs unittest 716，56 个函数式用例此前从未在 CI 执行 |
| 3 | 199–201 skipped | 全部是 torch / pycolmap / CUDA 缺失的守卫跳过 | 通道设计 | `tests/check_collection.py` 按通道白名单计数；torch-cpu 通道上任何 "torch missing" 跳过都是违规 |
| 4 | 1 个 `gsplat`、1 个 `pycolmap` 导入错误 | 可选运行时 | 通道设计 | cpu / torch-cpu 通道允许并计数；cuda 通道只允许 pycolmap 缺失 |
| 5 | 本地训练 venv 4 个 "CUDA extension not available" | 未经 `train/env_machine_b.cmd` 启动时 `TORCH_EXTENSIONS_DIR` 不指向 `.torch-ext`，JIT 扩展找不到 | 环境 | `tools/run_cuda_test_channel.cmd` 固定加载 env 后再跑；加载后 4 个用例通过 |
| 6 | 本地 4 个 preset 错误（`trainer_preset ... does not match fields: geometry_regularization`） | `_REGULARIZATION_DISABLED` 缺 `world_shrink_factor`；gate1 树里已有他人未提交修复（`presets.py` + `tests/test_training_presets.py`） | 代码缺陷，他人 WIP | 本批不改；三通道里记为已知失败，待该 WIP 提交后消失 |
| 7 | 10 个函数式用例在无 torch 时 failed 而非 skipped（`test_mono_depth_far_cutoff…`、`test_cli_roundtrip`、`test_empty_shn_gradient…`、`test_colmap4_binary_model…`、`test_invalid_policy_is_rejected` 等） | 在函数体内 `import torch/pycolmap/gsplat` | 测试卫生 | 审计脚本按 "NOT_RUN but reported as failure" 单列；后续改为 `pytest.importorskip` |

## 三通道设计

| 通道 | 依赖 | 收集 | 策略 | 状态 |
| --- | --- | --- | --- | --- |
| cpu | `uv sync --frozen` + pytest | `pytest --continue-on-collection-errors --junitxml` | torch/gsplat/pycolmap/CUDA 缺失允许并计数，其他一律违规，`--min-tests 400` | 本地复现：collected 535 / passed 302 / NOT_RUN torch 218、pycolmap 11、gsplat 4 / 违规 0 |
| torch-cpu | 上 + `torch==2.11.0`（PyTorch CPU 索引） | 同上 | 任何 torch 缺失跳过或导入错误即违规，`--min-tests 700` | **NOT_RUN**：本机无 CPU torch，只能在 Actions 验证；CPU wheel 可用性待 CI 确认 |
| cuda | B 机 `env_machine_b.cmd` | `tools/run_cuda_test_channel.cmd` | 只允许 pycolmap 缺失 | 见 `ci/collection-cuda.json` |

审计脚本的阴性对照：把 cpu 通道的 junit 用 torch-cpu 策略判定 → 219 条违规（红）。说明策略确实区分通道，而不是任何输入都放行。

## 未改动
- `uv.lock` 未动（本机无 uv）；pytest 与 CPU torch 在锁外用 `uv pip install --python .venv` 安装。
- 他人 WIP 文件未动。
