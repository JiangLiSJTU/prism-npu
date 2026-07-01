# 教程

4 篇 step-by-step 教程，按典型工作流组织：

| 文件 | 内容 | 工时 |
|------|------|------|
| [01_quickstart.md](01_quickstart.md) | 30 分钟从 git clone 到看到第一份 finding 报告 | 30 min |
| [02_reproduce_arch_sweep.md](02_reproduce_arch_sweep.md) | 复现 49 变体 × 5 模型 sweep + 自定义 baseline arch | 1 h |
| [03_recalibrate_with_new_msprof.md](03_recalibrate_with_new_msprof.md) | 用新 msprof 数据重新拟合 η_real（含 BERT MAE < 15 pp 硬门槛）| 1-2 h |
| [04_add_new_model.md](04_add_new_model.md) | 添加新 transformer 模型到 sweep MODELS（含 Vector ops 计算 + ATC + msprof 端到端）| 半天 |

**完成顺序**：01 → 02 → 03 → 04（每一篇假设你已完成上一篇）。

**进阶**：方法论原理见 [../methodology/](../methodology/)；CLI 参数见 [../reference/cli.md](../reference/cli.md)。
