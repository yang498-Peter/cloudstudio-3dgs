# 2026-08-25 BA 训练 Manifest 连接与 snow 全流程验证

## 问题现象

固定双目 BA 能生成并签名 `candidate_model` 与验收报告，但 Trainer 只读取原始 `dataset_manifest.json`。即使 BA 报告显示候选已接受，训练仍会静默使用原始 POS；Stage 2 优化后的焦距也不会进入训练。若直接替换相机而继续使用旧 mask、split、depth 与 person-mask Manifest，还会造成派生资产身份和相机投影不一致。

## 修改文件

- `cloudstudio_3dgs/ba/training_manifest.py`
- `tools/build_ba_training_manifest.py`
- `tests/test_ba_training_manifest.py`
- `tests/codex-scan/bh-20260825-ba-training-manifest-round1.py`
- `cloudstudio_3dgs/training/trainer.py`
- `tests/test_training.py`

## 修改内容

新增 fail-closed 的 BA 训练 Manifest 发布连接：校验基础数据 Manifest、split Manifest、BA 报告签名、候选接受状态及候选模型目录哈希；只用候选模型替换训练集相机位姿，验证集继续保留原始 POS；Stage 2 相机内参按已接受模型更新；所有基础数据、split、BA 报告与候选模型身份写入新 Manifest 的签名血缘。Trainer 继续读取标准签名数据 Manifest，无需绕过已有数据身份校验。

## 验证方式

- 红测：修复前导入连接模块失败，证明连接不存在。
- 单元回归：验证训练位姿替换、验证位姿隔离、Stage 2 内参传递、拒绝报告阻断、模型哈希篡改阻断。
- 真实数据：用 `2026-02-24_16-21-11snow.mpl` 对应 342 图数据发布 Stage 2 / 2 cm BA 训练 Manifest，并重建 mask、split、depth 和 person mask。
- Trainer 数据加载：检查 306 张训练图与 36 张验证图均可加载，首个训练位姿相对原 POS 变化 3.82 cm，首个验证位姿保持不变，优化后焦距和深度缓存已进入训练样本。
- Stage 2 / 2 cm POS：306 个训练图像的笛卡尔位置先验标准差设为 0.02 m。候选通过全部验收门限：重投影误差 p50 从 1.6565 px 降至 1.0922 px（改善 34.07%），场景尺度漂移 0.3331%（门限 0.5%），固定双目平移/旋转漂移约为 `1e-14`，左右相机最大焦距相对变化分别为 0.1480% / 0.0241%（门限 5%）。
- 完整 GPU 训练：RTX 5070 Laptop GPU 上固定容量 3DGUT 路径完成 30,000 步，耗时 13,180.71 秒，峰值 CUDA 显存 827,535,360 字节，362,470 个高斯；最佳 golden 检查点为 step 30,000。
- 独立评估：36/36 验证帧报告为 `COMPLETE`，PSNR 21.6909 dB、SSIM 0.585254、LPIPS(Alex/GPU) 0.474170、深度 MAE 4.21517 m，深度预测覆盖率 100%。
- PLY：从 `best_golden.pt` 导出 362,470 / 362,470 个高斯、15 个 SH rest 系数，文件 89,894,091 字节。
- 自动化回归：最终聚焦回归 29/29 通过；此前 BA/Trainer 邻接套件 50 通过、1 跳过。UTF-8 乱码检查、Python 语法和本轮拥有文件空白检查通过。仓库级 `git diff --check` 只被既有并发文件 `train/patches/gsplat-s1-fisheye-keep-distortion.patch` 的尾随空格阻断，本轮没有修改该文件。

## 当前状态

`BA candidate_model -> 签名训练 Manifest -> Trainer` 已连接并通过真实 snow 数据验证；Stage 2 / 2 cm POS 先验已实际执行且通过当前全部 BA 门限。30k GPU 训练、36 帧完整评估和 PLY 导出均已完成。

关键身份与产物：

- BA 报告签名：`dfd58999b78df14b0e53dc08cacb9289df35e19b97c9b9550a77ad115b01bc32`
- BA 候选模型 SHA-256：`7fb148dfb0d668a5ac29ea2c097003482b77b20756f79167465d17feaff610b3`
- BA 训练数据 Manifest 签名：`55b5df41778f774f450201bdadaf6551b13144bb4c5b3feba6dcb7aa57127bbf`
- 训练 Run Manifest 签名：`08d584c23ae168b51be799b945b61be9a17dbefe96fe7f1f3eed704c35ddd1f4`
- 最佳检查点 SHA-256：`a65993d42e80f41f46d77ea8d24ea0d3fe2329afbc1ce03b151638c6206f2b0a`
- 质量报告签名：`bde9f4316bb7a37ddf235766773ce7233961881add2193d7d4a8a6a51e7b0da5`
- 导出 PLY 文件 SHA-256：`ea06a85af606569f74fd50c9baedf37ce95ad9906793450f51985018912b4584`

边界说明：本次实际训练配置禁用了 densification、MCMC noise 和训练期位姿再优化，避免已接受 BA 被二次漂移；因此它证明的是固定容量 3DGUT 实际 GPU 路径。完整 MCMC 运行时审计仍为 `FAIL/NOT_RUN`（当前 clean binary 缺少三项原生算子），不能据此宣称 MCMC 路径通过。

## 后续高分辨率精修与最终版本选择

为避免仅凭 factor 2 名称覆盖已验证模型，后续精修使用了受血缘校验的热启动：只在数据 Manifest、Mask、Person Mask、Depth、坐标变换、初始化点顺序和锁定 gsplat runtime 全部一致时复制 Gaussian、SH 与辅助曝光参数；优化器、采样器、策略状态、随机状态和评估历史全部重新开始，允许分辨率阶段切换而不伪造断点续训等价。

- factor 2 / 5 步冷启动探针完成，峰值 CUDA 显存 1,291,980,288 字节，证明更高分辨率可运行。
- factor 2 / 5 步热启动探针完整通过血缘检查，起始完整验证为 PSNR 20.6377 dB、SSIM 0.5183、深度 MAE 4.1926 m。
- 原学习率热启动到 1k 后退化为 PSNR 18.5530 dB、SSIM 0.4972、深度 MAE 4.5910 m，已安全停止，不参与最终选择。
- 全参数和曝光学习率降为原值的十分之一后，1k 完整验证改善到 PSNR 19.9021 dB、SSIM 0.5136、深度 MAE 4.1989 m，但仍低于其 5 步热启动起点；四视角清晰度采样能量/一致性为 0.207/0.213，也没有形成可证明的提升。

因此最终版本不采用两条验证退化的 factor 2 臂，而采用原 30k/step 30,000 的最佳 golden checkpoint。已将经过哈希复核的标准 3DGS PLY 固化为：

`outputs/snow-20260224-full-20260825/final/snow-20260224-stage2-ba-verified-final.ply`

文件大小 89,894,091 字节，SHA-256 为 `ea06a85af606569f74fd50c9baedf37ce95ad9906793450f51985018912b4584`，与原始导出 PLY 字节完全一致。
