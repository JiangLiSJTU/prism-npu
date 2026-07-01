# 方法论文档

7 篇方法论原理文档，每篇覆盖一个核心模块的物理依据 + 公式推导 + 误差分析 + 已知局限：

| 文件 | 内容 |
|------|------|
| [01_overview.md](01_overview.md) | 工具目的 + 模块依赖图（mermaid）+ CLI 入口 + 阅读建议 |
| [02_three_layer_roofline.md](02_three_layer_roofline.md) | wall-clock 三层模型（T_aic + T_aiv + host_gap）+ Williams 2009 对比 + msprof β 校准 + pipe-aware 拆分公式 |
| [03_eta_real_model.md](03_eta_real_model.md) | Cube 真实利用率 physics-informed 5 参数公式（η_pipeline · η_tile · η_batch）+ Levenberg-Marquardt 拟合 + BMM/MM op_kind 区分 + 训练/验证 MAE 14.33 pp |
| [04_arch_sensitivity.md](04_arch_sensitivity.md) | 12 维双向 sweep + pipe-aware predict_wallclock_v3 公式 + TCO 代理 + 5 个解锁的细粒度敏感维度 + Hypothesis 规则化表 |
| [05_calibration.md](05_calibration.md) | msprof 数据采集 pipeline（ATC + ais_bench + analyze）+ 完整 Pipe 字段词典（AIC 5 + AIV 4）+ 失败模式 + regime gate 校验 |
| [06_assumptions_limits.md](06_assumptions_limits.md) | **first-class 公开假设清单**（4 类 18+ 假设：物理 / 公式 / 数据 / 外推）+ 风险等级 + 论文引用建议 |
| [07_optimization_ceiling.md](07_optimization_ceiling.md) | 5 情景算子/软件/硬件优化天花板预测 + 4 类瓶颈物理分类 + 与 sweep 的互补关系 |

**阅读建议**：

| 你是 | 起点 |
|------|------|
| 工具新人 | [01](01_overview.md) → [02](02_three_layer_roofline.md) → [04](04_arch_sensitivity.md) |
| 上机做实验 | [01](01_overview.md) → [05](05_calibration.md) → [tutorials/](../tutorials/) |
| 芯片架构师 | [02](02_three_layer_roofline.md) → [04](04_arch_sensitivity.md) → [07](07_optimization_ceiling.md) → [findings/](../findings/) |
| 算子优化 | [03](03_eta_real_model.md) → [05](05_calibration.md) → [07](07_optimization_ceiling.md) |
| 论文 / 评审 | [06](06_assumptions_limits.md) 全文 + 关心模块的方法论 |

历史 phase 叙述文档（21 篇）原样保留在 [`../../legacy/docs/`](../../legacy/docs/) 供追溯。
