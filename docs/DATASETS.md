# 本机真实数据清单(2026-07-02 盘点)

原始 MCAP 总量约 75GB,18 组完整记录。设备:MVP-S1(固件 v1.0.2–v1.0.3,SN 87340035 / 87340078 等)。

## 推荐测试场景(按研究计划 Phase 0 的选型要求)

| 用途 | 路径 | 说明 |
|---|---|---|
| **首选开发样本** | `G:\S1\2026-06-17_12-40-48gs2` | 4.4GB raw,623 左目图,已有 4 个解算 run(`process/..._3` 含 transforms.json + ImgPose.txt + colorized.las 1GB) |
| **完整管线参照** | `D:\S1\PROCESS\2026-01-29_14-36-28-Finland` | 7.5GB raw → colorized.las 1.26GB + uncolorized.ply;含 `all_frames.json`、`undistort/`(1091 张/目)、QualityReport PDF |
| 大场景/室内 | `G:\S1\2026-06-21_13-29-13house2` | 6.3GB,977 图 |
| 行人干扰验证(Phase 3) | 待从 G:\S1 各记录中人工确认哪条含行人 |
| RTK 室外 | `F:\S1\2026-03-23_12-24-00RTK` | 3.8GB,427 图 |
| 混合场景 | `G:\2026-02-06_14-35-01-ATS-IN-OUT` | 室内外混合(天空 mask 问题验证) |

## mipmap(竞品)对标

- 成品软件:`G:\S1\3DGS\Tersus-GNSS-MVP-S1-3DGS-Processor-V1.0.0-alpha2-20260608B`
  (数据契约逆向整理 → `docs/references/MVP_S1_3DGS输入数据与SDK调用整理_20260702.zh-CN.md`)
- 成功输出:`G:\Tersus3DGSResults\20260609-234131-s1-3dgs\3D\`
  - `model-gs-ply/gs.ply` ≈ 2.53GB / 4855 万高斯(质量基准线),`ue/gs_full.ply` ≈ 11.2GB
  - `model-gs-sog-tile/`、`model-b3dm/` 等 Web/Tiles 输出
- mipmap 参考样例数据:`G:\S1\2026-06-21_13-29-13house2`(含 process run,LAS 带项目前缀)。

## 全部存储位置

| 位置 | 内容 |
|---|---|
| `G:\S1\` | 13 组记录,56GB(2026-06 为主,最新) |
| `D:\S1\PROCESS\` | Finland 完整管线 |
| `F:\S1\` | 4 组(2026-03,含 RTK / tw) |
| `G:\RAW\`、`G:\各种SLAM数据汇总\`、`G:\2026-*` 散档 | 备份/杂项 |

> 数据只读。本工程的一切输出写到本工程 `experiments/` 或数据盘新建的独立目录,
> 不往原始记录目录里写任何东西(CloudStudio 的 process/ 归 metacam_cli 管)。
