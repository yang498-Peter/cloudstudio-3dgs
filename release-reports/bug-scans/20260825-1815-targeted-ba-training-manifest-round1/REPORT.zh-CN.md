# CloudStudio 专业缺陷扫描（第 1 轮）

- 扫描协议：`v2`
- 扫描深度：`targeted-delta`
- 扫描基线：`edf52a7d278786d0b2235ac5b396a029a5533905`
- 提交窗口：`edf52a7^..edf52a7d278786d0b2235ac5b396a029a5533905`
- 扫描时间：`2026-08-25`
- 包含范围：接受 BA 候选、签名训练 Manifest、Trainer 数据身份、真实 snow GPU 训练、评估和 PLY 导出
- 排除范围：MCMC densification 路径、安装包、Browser/WebView2、发布 CI、外部 PLY Viewer 验收
- 结论：**GO (verified contract scope)**
- 判定适用范围：BA 训练输入连接、固定容量 3DGUT 真实数据运行及本地导出
- 修改边界：`verify-and-fix`

## 结论摘要

确认并修复 1 个既有 P1：Trainer 与已接受 BA 候选之间没有发布连接，会使后续训练继续使用原始 POS。修复前独立红测无法导入连接入口；修复后正式单元测试、篡改/拒绝负控、训练/验证隔离、真实 snow 数据加载、30k GPU 训练、36 帧完整评估和 PLY 导出均通过。Pass B 未发现新增或修复引入缺陷。完整 MCMC、安装包和外部 Viewer 不在本结论内。

## 覆盖模型与责任矩阵

### 生产者—消费者图

```text
原始签名 dataset + split + 已接受 BA report + candidate_model
-> BA training manifest publisher
-> 新签名 dataset + 重建 mask/split/depth/person-mask
-> Trainer 固定容量 3DGUT
-> signed run manifest + evaluation artifacts
-> COMPLETE quality report + standard 3DGS PLY
```

| 边 | 首次成功 | 重跑失败/迟到 | A→B/owner | 数值/格式边界 | 取消/恢复 | 状态 |
|---|---|---|---|---|---|---|
| BA -> training manifest | 真实 Stage 2 候选发布 | 拒绝报告 fail-closed | dataset/split/report/model 四重 SHA 绑定 | 模型集必须等于 train split | 非异步发布，不适用 | covered |
| manifest -> Trainer | 306 train / 36 val 真实加载 | 篡改候选哈希阻断 | train 使用 BA、val 保持原 POS | 刚体位姿和相机模型检查 | 非异步加载，不适用 | covered |
| Trainer -> evaluation/export | 30k GPU COMPLETE | checkpoint/run 签名核验 | run 绑定全部派生 Manifest | 36 帧完整指标、362470 高斯 | 断点续训未在本轮制造 | covered with omission |

### 未执行矩阵单元

- 训练中断后恢复等价性：本轮目标为 BA 连接和一次完整固定容量训练，未制造中断；不影响本轮完成，但不能证明 resume 等价。
- 完整 MCMC：运行时审计仍为 `FAIL/NOT_RUN`，不属于本次禁用 densification/noise 的固定容量配置。
- 外部 PLY Viewer、Browser/WebView2、安装包与 CI：未执行，不能升级为客户发布结论。

## 测试预言机审计

| 检查 | 结果 | 证据 |
|---|---|---|
| 源/中间/结果/重导入身份可区分 | yes | base/split/report/model/run/PLY 使用不同 SHA 和路径 |
| 协议与真实 start/poll/result/cancel 一致 | yes（同步边界） | 发布器为原子同步写；Trainer 真实读取并跑完 30k |
| 未主动执行用户 workaround | yes | Trainer 直接读取派生标准 Manifest，无私有注入 |
| 至少一条真实算法/运行时边界 | yes | snow 数据在 RTX 5070 Laptop GPU 完成 30k |
| 乱序、失败和取消被确定性制造 | not applicable/omitted | 本连接为同步发布；拒绝和篡改负控已覆盖 |

## 第一轮结果与独立复审

### Pass A：症状/差异扫描

- 红测确认 `cloudstudio_3dgs.ba.training_manifest` 不存在，已接受 BA 无法成为 Trainer 输入。
- 修复增加 fail-closed 发布器、CLI 和永久单元测试；真实 Stage 2 候选发布后 Trainer 首个训练位姿变化 3.82 cm，首个验证位姿约 1e-9 m 不变。

### Pass B：正交攻击

- 独立审查：完成同一审查者的正交 Pass B；受当前协作策略限制未启用独立子代理。
- 未复用的维度：拒绝报告、候选目录字节篡改、train/val 位姿隔离、Stage 2 全局焦距传递、真实 GPU 消费、完整 LPIPS 评估。
- 新发现：0。
- 重复/排除：MCMC 审计失败属于已知外部门禁，不是本固定容量路径的新缺陷。

## 新发现

### R1-P1-BA-TRAINING-BRIDGE：已接受 BA 候选未连接 Trainer

- 发现类别：`product`
- 来源分类：`pre-existing`
- 问题现象：BA 已签名接受，但 Trainer 继续读取原始数据 Manifest。
- 最小场景：完成 Stage 2 BA 后直接以原 Manifest 开始训练。
- 根因链：BA 产物只有 candidate/report，没有将 train poses 和接受后的相机写入标准签名数据 Manifest 的生产者。
- 客户影响：训练会静默忽略已接受的位姿优化和焦距优化。
- 代码位置：`cloudstudio_3dgs/ba/training_manifest.py`、`tools/build_ba_training_manifest.py`。
- 独立复现：`tests/codex-scan/bh-20260825-ba-training-manifest-round1.py` 修复前导入失败。
- 正向控制：正式单测和真实 snow Trainer 加载。
- 与历史问题的区别：不是 BA 求解失败，而是已接受 BA 与 Trainer 的产物血缘缺口。
- 修复边界或建议：只替换 train pose，val 保持原 POS；相机内参全局更新；重建全部派生资产。
- 当前状态：`fixed`。

## 修复中新风险

- 未发现。Pass B 已检查验证集泄漏、候选模型篡改、拒绝报告、相机模型/尺寸变化和派生 Manifest 身份。

## 门禁接线与强制执行

| 测试 | 聚焦 | static/numeric | 默认套件 | Browser/WebView2 | 发布预检 | CI |
|---|---|---|---|---|---|---|
| `tests/test_ba_training_manifest.py` | executed | not applicable | discovered by pytest | not applicable | not executed | absent |
| `tests/codex-scan/bh-20260825-ba-training-manifest-round1.py` | executed | not applicable | deliberate audit lane | not applicable | not executed | absent |

- 发现方式：正式回归使用 pytest 默认 `test_*.py` 递归发现；独立红测保留在审计目录。
- 失败短路检查：Browser/发布 lane 不适用且未执行。
- 未接线风险：仓库没有可证明本回归被 CI 或发布预检强制执行的证据。

## 验证证据

```text
python -m pytest -q tests/codex-scan/bh-20260825-ba-training-manifest-round1.py
1 passed

python -m pytest -q tests/test_ba_training_manifest.py tests/test_ba_match_graph.py tests/test_ba_report.py tests/test_ba_runtime_lock.py tests/test_dataset_manifest.py tests/test_training_presets.py
23 passed

python -m pytest -q tests/test_training.py -k "dataset or manifest or golden or full_evaluation"
5 passed, 31 deselected

real snow: Stage 2 / 0.02 m POS -> all BA gates PASS
real GPU: 30000 steps COMPLETE, selected best_golden step 30000
quality: COMPLETE, 36 frames, LPIPS measured on CUDA
PLY: 362470/362470 gaussians, SHA-256 ea06a85a...12b4584
```

## 已检查但未发现新缺陷

- BA report/candidate/split/base dataset 身份绑定与篡改阻断。
- train/val 位姿隔离及 Stage 2 相机内参传播。
- 重建 mask、split、depth、person-mask 后的 Trainer 实际消费。
- 30k 固定容量 3DGUT 收尾、最佳检查点选择、36 帧评估与 PLY 导出。

## 边界与未验证项

- Full MCMC runtime gate：`FAIL/NOT_RUN`，缺少 `mcmc_perturb_positions` 和 covariance fwd/bwd 原生算子；本次训练没有调用这些操作。
- 外部 Viewer 对 89.9 MB PLY 的视觉和交互验收：未执行。
- 安装包、干净机、Browser/WebView2、发布 CI：未执行。

## 并行修改说明

- 冻结后新增修改：工作树原有 README、Trainer、converter、训练 patch/脚本等并发改动均未覆盖；本轮只拥有新增 BA 发布器、CLI、测试和本记录。
- 受影响复现是否重跑：yes，最终聚焦回归 29/29 通过。
- 仓库级 `git diff --check`：被既有并发文件 `train/patches/gsplat-s1-fisheye-keep-distortion.patch` 的尾随空格阻断；本轮拥有文件的独立空白、UTF-8 与语法检查通过，未修改该并发文件。

## 建议修复顺序

1. 若下一步启用 MCMC，再单独补齐并通过 Full MCMC 原生算子、前后向、relocation/sample/noise 和 resume 等价门禁。
2. 在发布前加入外部 PLY Viewer、安装包和 CI 强制执行证据。

## 发布判定

- 源码与自动化：`GO (verified contract scope)`，BA 到固定容量 Trainer 的精确契约已通过。
- Browser/WebView2：`NOT_RUN`。
- 真实数据/外部软件：真实 snow 数据通过；外部 PLY Viewer `NOT_RUN`。
- 安装包/发布环境：`NOT_RUN`。
- 最终适用结论：仅对 BA 训练 Manifest、固定容量 3DGUT 本地 GPU 训练、评估和 PLY 产物成立；不是模块级或客户发布 GO。
