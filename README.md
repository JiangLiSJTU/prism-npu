# PRISM — Pipeline-aware Roofline & Inference Sweep Model

> 通用 NPU 架构探索工具链。基于真机 msprof PipeUtilization 实测校准的 wall-clock + TCO 预测，回答**"针对目标推理工作负载，NPU 在 die area / 内存带宽 / Cube / Vector 维度上应当如何取舍？"** 的工程决策问题。
>
> 当前 calibration 锚点是 Ascend 910B4 NPU（公开硬件），方法论可推广到任何具备类似 pipe-level 性能计数器的 NPU。

---

## 30 秒速览

```bash
pip install -e ".[dev]"

prism-render --check    # 4 份 finding 报告与已 commit 一致 → exit 0
prism-sweep             # 11 维 46 变体架构 sweep
prism-ceiling           # 5 情景算子/软件/硬件优化天花板
pytest tests/           # 全绿
```

读 [docs/findings/主报告.md](docs/findings/主报告.md) 看 NPU 架构选型的核心论证。

---

## 工具能干什么

| 输入 | 输出 |
|------|------|
| 11 配置 msprof PipeUtilization 实测 | `pipe_baseline_per_model.json` baseline pipe time |
| baseline + 候选 arch_variant | wall-clock + ratio + TCO 预测 |
| baseline 单一 workload | 5 优化情景的 wall-clock 天花板 |
| arch yaml + model yaml | regime 判定（host-bound / compute-bound / memory-bound）|

→ 工具不直接给出"用什么芯片"答案，它生成**带物理依据的 wall-clock + TCO 预测**，让 NPU 架构师 + 业务方在量化数据上讨论权衡。

---

## 使用层级与依赖

工具按用法分 3 个层级，依赖递增。绝大多数用户只需 Tier 1。

| 层级 | 用途 | CLI / 脚本 | 硬件依赖 | 软件依赖 |
|------|------|-----------|---------|---------|
| **Tier 1**：预测 | 跑架构 sweep / 优化天花板 / 复现报告 / 新模型分析 | `prism-sweep` `prism-ceiling` `prism-render` `prism-regime` `prism-fit` `prism-extract` `prism-predict-pipe` | 任何笔记本 | Python 3.9+，`pip install` |
| **Tier 2**：Mapping 校验 | 用 Timeloop 验证 Cube cycles | `prism-mapping` | 任何笔记本 | + Docker（`accelergy/timeloop:latest`）|
| **Tier 3**：重新校准 | 加新模型 / 新硬件，重新采集 msprof | `benchmark/*` + `prism-extract` + `prism-fit` | **Ascend 910B/910B4 NPU** | + CANN 8.5+（含 ATC + msprof）|

### Tier 1 — 预测（90% 用户）

工具核心论证已 commit 在 `data/calibration/`，无需任何外部资源即可复现：

```bash
git clone https://github.com/JiangLiSJTU/prism-npu.git
cd prism
pip install -e ".[dev]"

prism-render --check    # 验证报告与 commit 一致
prism-sweep             # 11 维架构 sweep
prism-ceiling           # 5 情景优化天花板
pytest tests/           # 25 passed, 5 skipped
```

完整工作流见 [docs/tutorials/01_quickstart.md](docs/tutorials/01_quickstart.md)。

### Tier 2 — Mapping 校验（验证 Cube cycles）

如要验证某个 GEMM shape 在 910B4 Cube 上的真实 cycles（非纯解析估计），需 Timeloop Docker：

```bash
# 一次性 Docker 准备
docker pull accelergy/timeloop-accelergy-pytorch:latest

# 跑单 GEMM manual mapping
prism-mapping --workload-name bert_ffn1 \
              --M 4096 --N 3072 --K 1024 \
              --arch-yaml arch/ascend_910b4_for_mapping.yaml \
              --output-json data/outputs/manual_mapping_bert.json
```

详见 [docs/tutorials/02_reproduce_arch_sweep.md](docs/tutorials/02_reproduce_arch_sweep.md) 第 4 节，CLI 完整参数见 [docs/reference/cli.md#prism-mapping](docs/reference/cli.md#prism-mapping)。

### Tier 3 — 重新校准（贡献者 / 新模型）

加新模型或换新硬件时，需在 Ascend NPU 上重采集 msprof，然后接进拟合 pipeline：

```bash
# 1. NPU 服务器（CANN 8.5+ 环境）：
python benchmark/export_qwen3_prefill.py --S 4096 --batch 1   # 导出 ONNX
# atc --model=... --output=...                                  # ATC 转 OM（自写或参考 benchmark/）
# msprof --application="./run_inference"                        # 采集 4 metric

# 2. 本地：
rsync -avz user@npu:~/sim-experiment/msprof_<model>/ msprof_data/
prism-extract --output data/calibration/cube_util_extracted.json
prism-fit --output data/calibration/eta_physics_fit.json

# 3. 重跑 sweep + 渲染（关键不变量：BERT MAE < 15 pp）
prism-sweep && prism-render
```

完整流程含 ONNX 导出 / ATC 转换 / msprof 采集脚本见 [benchmark/README.md](benchmark/README.md)，本地拟合接入见 [docs/tutorials/03_recalibrate_with_new_msprof.md](docs/tutorials/03_recalibrate_with_new_msprof.md)。

---

## 设计原则

| 原则 | 实施 |
|------|------|
| 实测优先 | 所有系数来自 910B4 msprof 实测（含 ArithmeticUtil + PipeUtil）|
| 公式可解读 | wall-clock = T_aic + T_aiv + host_gap，每层 pipe 拆 |
| 架构敏感度分模块 | 11 维独立改变，每维度 ratio 单列 |
| 优化潜力可量化 | S0 baseline → S4 +HBM3 五情景累积 |
| 单一 calibration 锚点 | 当前以 Ascend 910B4 为实证基准；方法论可推广到任何具备 pipe-level 性能计数器的 NPU |

详见 [docs/methodology/01_overview.md §2](docs/methodology/01_overview.md#2-设计原则)。

---

## 安装

### 推荐路径（pip install）

```bash
git clone https://github.com/JiangLiSJTU/prism-npu.git
cd prism
python3 -m venv .venv

# Activate venv:
#   Linux / macOS:    source .venv/bin/activate
#   Windows (cmd):    .venv\Scripts\activate.bat
#   Windows (PowerShell):  .venv\Scripts\Activate.ps1

pip install -e ".[dev]"
```

→ 7 个 CLI 入点（`prism-extract`、`prism-fit`、`prism-regime`、`prism-sweep`、`prism-ceiling`、`prism-mapping`、`prism-render`）安装到 `.venv/bin/`。

### 无 install 路径（直接跑 source）

```bash
python3 scripts/prism_render.py --check
python3 scripts/prism_sweep.py
python3 scripts/prism_ceiling.py
```

`scripts/prism_*.py` 是薄包装，自动注入 `PYTHONPATH=src/`。

---

## 项目结构

```
prism/
├── README.md              ← 本文件
├── pyproject.toml         ← Python 包元信息 + 7 个 CLI entry points
├── setup.py               ← legacy pip (<23) 兼容 shim
├── CONTRIBUTING.md        ← 开发指南（提交规范、locked files、测试要求）
├── LICENSE                ← MIT
│
├── src/prism/    ← 核心 Python 包（M2 重构后）
│   ├── eta_real/          ← Cube 真实利用率拟合（extract / fit / match / predict）
│   ├── roofline/          ← 三层 wall-clock 模型（predict / regime）
│   ├── sweep/             ← 11 维架构 sweep（runner / timeloop_problem）
│   ├── mapper/            ← Timeloop manual mapping（generate / runner）
│   ├── ceiling/           ← 5 情景优化天花板（predict）
│   └── reports/           ← Jinja2 渲染（render）
│
├── scripts/               ← 7 个 thin CLI wrappers（每个 < 20 行）
│   prism_{ceiling,extract,fit,mapping,regime,render,sweep}.py
│
├── arch/   models/   mappings/   ← YAML 配置（baseline + 派生）
├── benchmark/             ← NPU 上做实验的脚本（ONNX export, ATC, msprof）
├── data/                  ← 输入 / 校准参数 / 输出（部分 .json gitignored，可重生）
│   ├── inputs/
│   ├── calibration/
│   └── outputs/
│
├── reports/templates/     ← 4 个 Jinja2 模板
│
├── docs/                  ← 文档体系（M3-M4 重写）
│   ├── methodology/       ← 7 篇方法论原理（论文级详细）
│   ├── tutorials/         ← 4 篇 step-by-step 教程
│   ├── reference/         ← 4 篇参考（YAML schema / API / CLI）
│   └── findings/          ← 4 份顶级 finding 报告（主报告 + 3 份子报告）+ 1 ceiling 分析
│
└── tests/                 ← pytest（5 unit + 1 E2E）
```

---

## CLI 速查

| CLI | 用途 | 详情 |
|-----|------|------|
| `prism-extract` | 从 msprof CSV 提取 per-op pipe time + cube util | [docs/reference/cli.md](docs/reference/cli.md#prism-extract) |
| `prism-fit` | physics-informed η_real 拟合 | [docs/reference/cli.md](docs/reference/cli.md#prism-fit) |
| `prism-regime` | regime gate（host/compute/memory-bound 判定）| [docs/reference/cli.md](docs/reference/cli.md#prism-regime) |
| `prism-predict-pipe` | 从 GEMM 规格预测 pipe baseline（无 msprof）| [docs/reference/cli.md](docs/reference/cli.md#prism-predict-pipe) |
| `prism-sweep` | 11 维架构 sweep（46 变体 × baseline 全部模型）| [docs/reference/cli.md](docs/reference/cli.md#prism-sweep) |
| `prism-ceiling` | 5 情景优化天花板预测 | [docs/reference/cli.md](docs/reference/cli.md#prism-ceiling) |
| `prism-mapping` | Timeloop manual mapping（Docker 后端）| [docs/reference/cli.md](docs/reference/cli.md#prism-mapping) |
| `prism-render` | Jinja2 渲染 4 份 finding 报告 | [docs/reference/cli.md](docs/reference/cli.md#prism-render) |

---

## 文档导航

按角色入口：

| 你是 | 起点 |
|------|------|
| 工具新人 | [docs/methodology/01_overview.md](docs/methodology/01_overview.md) → [02_three_layer_roofline.md](docs/methodology/02_three_layer_roofline.md) → [04_arch_sensitivity.md](docs/methodology/04_arch_sensitivity.md) |
| 上机做实验 | [docs/tutorials/01_quickstart.md](docs/tutorials/01_quickstart.md) → [docs/methodology/05_calibration.md](docs/methodology/05_calibration.md) |
| 芯片架构师 | [docs/findings/主报告.md](docs/findings/主报告.md) → [docs/methodology/04_arch_sensitivity.md](docs/methodology/04_arch_sensitivity.md) → [07_optimization_ceiling.md](docs/methodology/07_optimization_ceiling.md) |
| 算子优化 | [docs/findings/optimization_ceiling.md](docs/findings/optimization_ceiling.md) → [docs/methodology/03_eta_real_model.md](docs/methodology/03_eta_real_model.md) |
| 论文 / 评审 | [docs/methodology/06_assumptions_limits.md](docs/methodology/06_assumptions_limits.md) 全文（含论文引用建议）|
| 加新模型 | [docs/tutorials/04_add_new_model.md](docs/tutorials/04_add_new_model.md) |

完整导航见 [docs/methodology/01_overview.md §4](docs/methodology/01_overview.md#4-文档导航)。

---

## 关键发现速览

| 发现 | 数据 | 来源 |
|------|------|------|
| 固定网络业务 (BERT/GPT-2/Net-Trans) 在 49 个 sweep 变体下 ratio ≈ 1.0（v3 修订：仅 LPDDR4X / UB+L1 融合 / 长上下文场景下 ratio ≠ 1）| host_gap 主导（占 wall_clock 67-87%）；v3 pipe-aware 公式解锁了真实敏感维度 | [findings/主报告.md](docs/findings/主报告.md) |
| LLM 长上下文 prefill (S=4096) UB+L1 融合带来 **20% 加速** | aiv_mte2 73.2% baseline → 5% 残余 | [findings/optimization_ceiling.md](docs/findings/optimization_ceiling.md) |
| LLM serving decode HBM3 升级带来 **14% 加速** | aic_mte2 84.8% baseline → 49% (392/800) | [findings/optimization_ceiling.md](docs/findings/optimization_ceiling.md) |
| **910B4-mini sweet spot**：16 cores × LPDDR5X × TDP 200W = **TCO -55%**，固定网络业务零退化 | 11 维 sweep + TCO 代理 | [findings/主报告.md §6](docs/findings/主报告.md) |
| η_real BERT 验证 MAE 14.33 pp（< 15 pp 硬门槛）| 5 参数 physics-informed fit + 21 训练数据点 | [methodology/03_eta_real_model.md §6](docs/methodology/03_eta_real_model.md#6-训练--验证集设计) |

---

## 验证保证

`pip install -e ".[dev]"` 后任何人重跑应得相同结果：

```bash
prism-render --check    # exit 0（4 templates 与 commit 一致）
prism-fit               # BERT validation MAE 14.33 pp
prism-sweep             # 49 variants × baseline 全部 model 与 reference 一致
prism-ceiling           # 5 scenarios × 11 configs 与 reference 一致
pytest tests/         # 全绿
```

任意 PR 必须 5 项全过。详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

---

## 已知局限

工具有 18+ first-class 公开假设，**任何使用本工具结论的报告必须连同 [docs/methodology/06_assumptions_limits.md](docs/methodology/06_assumptions_limits.md) 一起阅读**。最关键的 3 个：

1. **β_layer 假设跨 arch_variant 不变**：减核 / 减 L2 时 host 调度路径会变，β 可能 ±20% 漂移。所有 sweep ratio 是上界估计。
2. **跨芯片外推未独立校准**：sweep 仅在 910B4 上做，310P 的 baseline 校准是 910B4 占位值。需 310P 独立 msprof 数据才能做跨芯片对比。
3. **5 模型测试集偏差**：sweep MODELS dict 仅 5 模型，"固定网络业务无架构杠杆"严格只在这 5 模型代表固定网络业务的前提下成立。

完整局限清单：[docs/methodology/06_assumptions_limits.md](docs/methodology/06_assumptions_limits.md)。

---

## 历史素材

工具从 Phase A 到 Phase N 跨 9+ 个研究阶段。所有 phase 叙述文档（21 篇）+ 14 个历史 timeloop_results sweep + 5 个已废 scripts 原样保留在 [legacy/](legacy/) 供追溯，**勿改**。详见 [legacy/README.md](legacy/README.md) 索引。

---

## 联系 / 贡献

- Issue / PR：通过 GitHub
- 提交规范、locked files、测试要求：[CONTRIBUTING.md](CONTRIBUTING.md)
- 方法论讨论：[docs/methodology/](docs/methodology/) + [docs/findings/](docs/findings/)

---

**LICENSE**: MIT

**主要 baseline**：昇腾 910B4 (CANN 8.5)

**支持的目标**：910B4 派生变体、未来自研NPU
