# 贡献指南

本工具是NPU立项论证的分析工具链，欢迎以下类型贡献：

- 添加新模型 / 工作负载（详见 `docs/tutorials/04_add_new_model.md`）
- 用新 msprof 数据校准 η_real 拟合（详见 `docs/tutorials/03_recalibrate_with_new_msprof.md`）
- 扩展架构 sweep 维度（详见 `docs/methodology/04_arch_sensitivity.md`）
- 添加新优化情景到天花板预测工具（详见 `docs/methodology/07_optimization_ceiling.md`）

---

## 开发环境

```bash
# 克隆 + 安装
git clone https://github.com/JiangLiSJTU/prism-npu.git
cd prism
python3 -m venv .venv
# Activate: Linux/macOS  → source .venv/bin/activate
#           Windows cmd  → .venv\Scripts\activate.bat
#           Windows PS   → .venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# 运行测试
pytest tests/

# Lint
ruff check src/ scripts/ tests/
```

---

## 提交规范

### 分支命名
- `feature/<topic>` 新功能
- `fix/<bug>` bug 修复
- `docs/<topic>` 文档改动
- `refactor/<topic>` 重构

### Commit message
- 中文为主，关键术语保留英文（如 `pipe-aware`、`η_real`）
- 一行 summary < 80 字符
- 如改动涉及方法论文档（`docs/methodology/`），在 body 注明影响哪些下游文档

### Pull Request
- 自验收清单：
  - [ ] `pytest tests/` 全绿
  - [ ] `prism-render --check` exit 0（或 `python3 scripts/prism_render.py --check`）
  - [ ] 涉及拟合的 PR：附 `data/calibration/eta_physics_fit.json` 训练 / 验证 MAE 截图
  - [ ] 涉及 sweep 公式的 PR：附 baseline 输出与 reference 一致的 diff（应为空）
  - [ ] 新增方法论：`docs/methodology/` 与 `docs/findings/` 索引同步更新

---

## 不可修改的文件（locked）

以下文件**仅 maintainer 可改**，普通贡献请走 issue 讨论：

| 文件 | 原因 |
|-----|------|
| `arch/ascend_910b4*.yaml` / `arch/ascend_310p*.yaml` | 实卡参数权威源，所有派生 yaml 都从此引出 |
| `models/*.yaml` | 与 `src/prism/sweep/runner.py` 的 `MODELS` dict 一一对应，改了影响校准结果 |
| `data/calibration/eta_physics_fit.json` | physics-informed 5 参数拟合输出；改之前需重跑 `prism-fit` 并通过 BERT MAE < 15 pp 门槛 |
| `data/calibration/pipe_baseline_per_model.json` | per-model msprof PipeUtilization 实测 baseline；sweep / ceiling 都依赖此 |

---

## 添加新优化情景到 ceiling 工具（示例）

`src/prism/ceiling/scenarios.py` 加新函数：

```python
def compute_kv_prefetcher_ceiling(pipe, ...):
    """新硬件假想：KV cache prefetcher，aic_mte2 → 50%。"""
    ...
    return ScenarioResult(scenario='S5_kv_prefetcher', ...)
```

然后注册到 `predict.py` 的 `ALL_SCENARIOS` 表，工具自动渲染新列。

---

## 联系

Issue / PR：通过 GitHub。  
方法论讨论：参考 `docs/methodology/` 6 篇 + `docs/findings/` 4 份顶级报告。
