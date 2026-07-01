# 顶级 Findings 报告

本目录由 `reports/templates/*.md.j2` 通过 `scripts/render_reports.py` 渲染生成。**禁手编**，改动应改 templates 或 `data/experiment_variables.json`。

| 文件 | 模板源 | 内容 |
|------|--------|------|
| `主报告.md` | `reports/templates/NPU 架构设计_v2.md.j2` | 产品线总裁阅读对象，立项核心论证（最高优先级）|
| `roofline校准报告.md` | `reports/templates/910B4_roofline_校准报告_v3.md.j2` | wall-clock 三层模型 + Timeloop 验证 |
| `微架构探索报告.md` | `reports/templates/微架构探索_报告_v3.md.j2` | Timeloop sweep 结果 |
| `msprof分解报告.md` | `reports/templates/msprof_breakdown_summary.md.j2` | msprof 实测 wall-clock vs NPU compute gap |

渲染：
```bash
python3 scripts/render_reports.py
```

CI 检查（无漂移）：
```bash
python3 scripts/render_reports.py --check
```
