# 第三方组件 License 台账

红线:**禁止引入 INRIA 原版 3DGS(graphdeco-inria/gaussian-splatting)任何代码** —— 非商用许可。
每引入一个新仓库/论文实现,先查 LICENSE,登记到本表后再动手。

| 组件 | 用途 | License | 状态 |
|---|---|---|---|
| gsplat (nerfstudio-project/gsplat) | 训练/渲染核心 + 3DGUT + MCMC | Apache-2.0 | 选定,未引入 |
| nerfstudio | 数据约定参考 | Apache-2.0 | 仅格式参考 |
| dn-splatter (maturk/dn-splatter) | 深度/法向监督 loss 设计参考 | 需核验(引入前查) | 未引入 |
| SAM2 (Meta) | 语义 mask(人/车) | Apache-2.0 | 未引入 |
| 3dgrut (nv-tlabs/3dgrut) | 3DGUT 官方参考 | 需核验(引入前查) | 仅阅读 |
| T-3DGS | 瞬态剔除备选 | 需核验 | 未引入 |
| laspy | LAS 点云读取(工具脚本) | MIT | 使用中 |
| numpy / Pillow | 数值/图像 | BSD / MIT-CMU | 使用中 |
| SciPy | QA 距离变换、近邻查询与可选点云局部 PCA | BSD-3-Clause | 使用中 |
| LPIPS (richzhang/PerceptualSimilarity) | masked 感知质量评估 | BSD-2-Clause | 可选，不锁入 CPU 环境 |
| SpotLessSplats | 瞬态剔除思路 | 无完整官方开源 → 自实现思路,不抄代码 | — |

CloudStudio 本体为 BSD-2-Clause(自家),本工程只读取其产生的数据文件,不链接其代码。
metacam SDK(mvps1 解算器)为供应商闭源二进制,本工程仅消费其输出文件。
