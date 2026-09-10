澳洲B机 3DGS 全链路审计与Agent任务包 — 2026-09-11

先阅读中文Markdown任务书。最终复核已纳入CLI新提交249c70a；算法源码基线为1ca1cd4。

内容：
- 中文任务书：证据分级、最新CLI风险、算法机制、WP00–WP10任务、实验矩阵、验收与文献。
- audit_gs_config.py：不依赖Torch的只读日程检查工具；不启动训练，不修改配置。
- test_audit_gs_config.py：该独立辅助脚本的8项单元测试。
- audit_helper_tests.txt：本分析环境的辅助脚本测试记录；不是仓库完整测试或B机测试。

测试：python -m unittest -v test_audit_gs_config
用法：python audit_gs_config.py --config path/to/run.json --output report.json
多个配置可重复 --config，也可使用glob。

风险告警只指出需核验的代码/配置机制，不宣称相关机制已被真实数据证明为画质主因。
任何参数变化先冻结旧模型并创建新arm，禁止原地覆盖正在运行的配置或未经验证清理缓存。

没有进行B机GPU实验，没有修改GitHub仓库；最新PLY及运行状态需由B机Agent核验。
