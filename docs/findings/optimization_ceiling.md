# 算子 / 软件 / 硬件优化天花板分析报告

**输出工具**：`scripts/calibration/predict_optimization_ceiling.py`
**输入数据**：`data/pipe_baseline_per_model.json`（Phase N 9 配置 PipeUtil 实测 + 2 占位/继承）
**输出 JSON**：`data/optimization_ceiling.json`

---

## 一、问题陈述

给定一个 workload 的 msprof PipeUtilization 实测数据（含 AIC/AIV 各 pipe 占比），
回答以下三类问题：

1. **算子库优化天花板**：完美 ping-pong + 完美指令调度后，wall-clock 还能降多少？
2. **CANN runtime 优化天花板**：再加 host_gap 优化（kernel launch + graph fusion + async dispatch）能降多少？
3. **硬件改动可叠加增益**：UB+L1 融合 / HBM3 升级在算子优化天花板基础上还能再降多少？

---

## 二、5 情景定义

| 情景 | 操作 | 物理依据 |
|------|------|---------|
| **S0 Baseline** | msprof 实测，无任何优化 | 现状 |
| **S1 Software ceiling** | AIC bubble → 0；AIV idle → 0；kernel_gap → 0 | 完美双缓冲 + 完美指令调度 + 完美 AIC/AIV 异步 |
| **S2 Software + Runtime ceiling** | S1 + host_gap → 10.0 μs/kernel | CANN graph fusion + ModelLoad 缓存 + async dispatch |
| **S3 + UB+L1 融合**（硬件） | S2 + aiv_mte2 × 0.05 | 融合内存池消除 UB↔L1 algorithmic traffic |
| **S4 + HBM3 800 GB/s**（硬件） | S3 + aic_mte2 × 0.490 | 与 Phase J v3 sweep 一致 |

每个情景的 wall_clock 公式：

```
wall_clock = max(active_aic_pipes) + aic_bubble
           + max(active_aiv_pipes) + aiv_idle
           + kernel_gap + host_gap

S1: 把 aic_bubble、aiv_idle、kernel_gap 置 0
S2: + host_gap 降到 n_kernels × host_gap_target
S3: + aiv_mte2 × 0.05
S4: + aic_mte2 × (392/800)
```

---

## 三、9 配置 + 占位 全 scenarios 结果（per inference μs）

| Workload | S0 baseline | S1 sw | -%| S2 sw+rt | -%| S3 +UB融 | -%| S4 +HBM3 | -%|
|---------|-----------:|-----:|----:|--------:|----:|--------:|----:|--------:|----:|
| BERT-base-S128-b1 |     16,210 | 14,812 |  8.6% |    4,112 | 74.6% |    4,013 | 75.2% |    3,750 | 76.9% |
| GPT-2-S512-b1 |     17,280 | 13,985 | 19.1% |    5,587 | 67.7% |    5,587 | 67.7% |    5,182 | 70.0% |
| Qwen3-prefill-S512-b4 |    301,880 | 210,326 | 30.3% |  210,326 | 30.3% |  193,339 | 36.0% |  158,352 | 47.5% |
| Qwen3-prefill-S512-b8 |    603,760 | 444,782 | 26.3% |  444,782 | 26.3% |  413,515 | 31.5% |  350,034 | 42.0% |
| Qwen3-prefill-S256-b1 |     78,000 | 44,046 | 43.5% |   44,046 | 43.5% |   37,606 | 51.8% |   29,205 | 62.6% |
| Qwen3-prefill-S256-b4 |    140,000 | 94,433 | 32.5% |   94,433 | 32.5% |   85,444 | 39.0% |   73,659 | 47.4% |
| Qwen3-prefill-S256-b8 |    225,000 | 162,749 | 27.7% |  162,749 | 27.7% |  150,180 | 33.2% |  133,741 | 40.6% |
| Qwen3-prefill-S4096-b1 |  3,050,000 | 1,963,103 | 35.6% | 1,963,103 | 35.6% | 1,338,829 | 56.1% | 1,338,829 | 56.1% |
| Qwen3-decode-Min4-Skv128-b1 |      7,690 |  3,111 | 59.5% |    3,111 | 59.5% |    2,597 | 66.2% |    1,520 | 80.2% |
| Net-Transformer-S256-L1-b1 |        196 |     82 | 58.2% |       82 | 58.2% |       67 | 65.9% |       41 | 79.2% |
| Qwen3-Embedding-S4096-b1 |  3,050,000 | 1,963,103 | 35.6% | 1,963,103 | 35.6% | 1,338,829 | 56.1% | 1,338,829 | 56.1% |
| BERT-base-S128-b4 |      2,463 |  1,378 | 44.1% |    1,378 | 44.1% |    1,143 | 53.6% |      908 | 63.1% |
| BERT-base-S128-b8 |      4,154 |  2,368 | 43.0% |    2,368 | 43.0% |    2,263 | 45.5% |    1,691 | 59.3% |
| BERT-base-S128-b16 |      5,532 |  3,366 | 39.1% |    3,366 | 39.1% |    3,366 | 39.1% |    3,018 | 45.4% |
| GPT-2-S512-b4 |     14,594 |  7,963 | 45.4% |    7,963 | 45.4% |    7,963 | 45.4% |    7,349 | 49.6% |
| GPT-2-S512-b8 |     28,385 | 15,818 | 44.3% |   15,818 | 44.3% |   15,818 | 44.3% |   14,682 | 48.3% |
| GPT-2-S512-b16 |     60,094 | 34,380 | 42.8% |   34,380 | 42.8% |   34,295 | 42.9% |   29,472 | 51.0% |
| HF-BERT-S128-b1 |        582 |    162 | 72.2% |      162 | 72.2% |      104 | 82.1% |       78 | 86.5% |
| HF-BERT-S128-b4 |        787 |    270 | 65.6% |      270 | 65.6% |      150 | 80.9% |      118 | 85.0% |
| HF-BERT-S128-b8 |        952 |    381 | 59.9% |      381 | 59.9% |      210 | 77.9% |      194 | 79.6% |
| HF-BERT-S128-b16 |      1,168 |    565 | 51.6% |      565 | 51.6% |      270 | 76.9% |      260 | 77.7% |
| Net-Transformer-S256-L1-b4 |        267 |    117 | 56.1% |      117 | 56.1% |       94 | 64.9% |       56 | 78.9% |
| Net-Transformer-S256-L1-b8 |        324 |    158 | 51.3% |      158 | 51.3% |      123 | 62.1% |       82 | 74.8% |
| Net-Transformer-S256-L1-b16 |        452 |    266 | 41.1% |      266 | 41.1% |      226 | 49.9% |      167 | 63.0% |
| ModernBERT-base-S4096-b1 |    323,232 | 269,720 | 16.6% |  269,720 | 16.6% |  216,071 | 33.1% |  216,071 | 33.1% |
| Llama-3.2-1B-prefill-S2048-b1 |    197,402 | 169,660 | 14.1% |  167,518 | 15.1% |  140,938 | 28.6% |  126,829 | 35.8% |
| Qwen2.5-0.5B-prefill-S2048-b1 |    162,255 | 141,572 | 12.8% |  114,633 | 29.4% |  100,535 | 38.0% |   89,001 | 45.1% |
| SmolLM2-360M-prefill-S2048-b1 |    210,617 | 182,177 | 13.5% |  145,712 | 30.8% |  130,718 | 37.9% |  120,662 | 42.7% |
| Qwen3-prefill-S4096-b1-sdpa |    452,736 | 397,019 | 12.3% |  397,019 | 12.3% |  291,668 | 35.6% |  242,164 | 46.5% |
| Qwen3-prefill-S256-b1-sdpa |     82,384 | 72,967 | 11.4% |   24,636 | 70.1% |   22,301 | 72.9% |   20,592 | 75.0% |
| Qwen3-prefill-S256-b4-sdpa |     94,930 | 82,786 | 12.8% |   36,578 | 61.5% |   32,941 | 65.3% |   30,487 | 67.9% |
| Qwen3-prefill-S512-b4-sdpa |    118,198 | 101,489 | 14.1% |   58,718 | 50.3% |   53,764 | 54.5% |   47,332 | 60.0% |
| Qwen3-prefill-S256-b8-sdpa |    110,971 | 96,284 | 13.2% |   52,341 | 52.8% |   48,232 | 56.5% |   44,222 | 60.1% |
| Qwen3-prefill-S512-b8-sdpa |    189,944 | 162,401 | 14.5% |  131,551 | 30.7% |  117,499 | 38.1% |  103,843 | 45.3% |
| ModernBERT-base-S4096-b1-sdpa |    260,623 | 220,028 | 15.6% |  215,354 | 17.4% |  154,205 | 40.8% |  142,268 | 45.4% |
| Qwen2.5-0.5B-prefill-S2048-b1-sdpa |    148,047 | 130,521 | 11.8% |  102,176 | 31.0% |   86,274 | 41.7% |   72,943 | 50.7% |
| SmolLM2-360M-prefill-S2048-b1-sdpa |    191,347 | 165,821 | 13.3% |  127,375 | 33.4% |  113,469 | 40.7% |  101,245 | 47.1% |
| Llama-3.2-1B-prefill-S2048-b1-sdpa |    173,434 | 152,532 | 12.1% |  146,763 | 15.4% |  123,131 | 29.0% |  105,575 | 39.1% |
| Phi-3-mini-prefill-S2048-b1-sdpa |    387,206 | 338,004 | 12.7% |  338,004 | 12.7% |  288,010 | 25.6% |  242,652 | 37.3% |

---

## 四、按 workload 类别分析

### 4.1 固定网络业务（BERT / GPT-2 / Net-Transformer）

**BERT-base-S128-b1**：
- S0 baseline = 16,210 μs（其中 host_gap = 14,079 μs，占 87%）
- 算子优化（S1）单独降 **8.6%**
- + CANN runtime 优化再降 **66.0%**（host_gap 优化是这一类的主要杠杆）
- + UB+L1 融合再降 **0.6%**（固定网络业务 aiv_mte2 占比小，硬件杠杆有限）

**GPT-2-S512-b1**：
- S0 baseline = 17,280 μs（其中 host_gap = 11,607 μs，占 67%）
- 算子优化（S1）单独降 **19.1%**
- + CANN runtime 优化再降 **48.6%**（host_gap 优化是这一类的主要杠杆）
- + UB+L1 融合再降 **0.0%**（固定网络业务 aiv_mte2 占比小，硬件杠杆有限）

**Net-Transformer-S256-L1-b1**：
- S0 baseline = 196 μs（其中 host_gap = 0 μs，占 0%）
- 算子优化（S1）单独降 **58.2%**
- + CANN runtime 优化再降 **0.0%**（host_gap 优化是这一类的主要杠杆）
- + UB+L1 融合再降 **7.8%**（固定网络业务 aiv_mte2 占比小，硬件杠杆有限）

### 4.2 LLM prefill（Qwen3-0.6B 短/中/长上下文）

**Qwen3-prefill-S256-b1**：
- S0 baseline = 78,000 μs
- 算子优化（S1）降 **43.5%**；S2 总降 43.5%
- UB+L1 融合再降 **8.3%**
- HBM3 再降 **10.8%**

**Qwen3-prefill-S256-b8**：
- S0 baseline = 225,000 μs
- 算子优化（S1）降 **27.7%**；S2 总降 27.7%
- UB+L1 融合再降 **5.6%**
- HBM3 再降 **7.3%**

**Qwen3-prefill-S512-b8**：
- S0 baseline = 603,760 μs
- 算子优化（S1）降 **26.3%**；S2 总降 26.3%
- UB+L1 融合再降 **5.2%**
- HBM3 再降 **10.5%**

**Qwen3-prefill-S4096-b1**：
- S0 baseline = 3,050,000 μs
- 算子优化（S1）降 **35.6%**；S2 总降 35.6%
- UB+L1 融合再降 **20.5%**
- HBM3 再降 **0.0%**

**Qwen3-Embedding-S4096-b1**：
- S0 baseline = 3,050,000 μs
- 算子优化（S1）降 **35.6%**；S2 总降 35.6%
- UB+L1 融合再降 **20.5%**
- HBM3 再降 **0.0%**

### 4.3 LLM decode

**Qwen3-decode-Min4-Skv128-b1**：
- S0 baseline = 7,690 μs
- S1 软件优化降 **59.5%**（aiv_idle 占 ~50%，软件可消除）
- S2 + CANN runtime 总降 59.5%
- S3 + UB+L1 融合总降 66.2%
- **S4 + HBM3 总降 80.2%（HBM 是 decode 的真瓶颈）**

---

## 五、核心结论

### 5.1 "现有固定网络场景仅靠算子优化能解决 vector 瓶颈吗？"

**部分能，部分不能**——按 workload 分：

| Workload 类别 | S1 算子优化 wall_clock 降幅 | S3-S2 = 硬件 UB+L1 融合再降幅 | 判定 |
|--------------|---------------------------:|----------------------------:|------|
| 固定网络短输入 (BERT b=1) | 8.6% | 0.6% | **算子优化为主**，硬件可忽略 |
| LLM 长上下文 (Qwen3-prefill-S4096) | 35.6% | **20.5%** | **硬件 UB+L1 融合是必经**，算子优化达上限 |
| LLM decode | 59.5% | HBM3 再降 14.0% | **HBM3 + 软件双轮**，UB 融合次要 |

### 5.2 "Vector 瓶颈"分两类的实测证据

- **流水线效率类**（aiv_idle）：算子优化 100% 可解决。BERT b=1 上 idle 占 aiv_time 的 46%，这部分是软件可消除的同步等待。
- **算法搬运量类**（aiv_mte2 = UB↔L1 字节数）：算子优化无法消除。Qwen3-prefill-S4096 上 aiv_mte2 占 73.2%，是 algorithmic memory traffic，**只有 UB+L1 融合等硬件改动能消除**。

### 5.3 立项含义

1. **固定网络业务自研芯片**：硬件 vector 改动 ROI 低（固定网络 workload 的 vector 瓶颈是 CANN runtime + 算子库已有覆盖范围）
2. **LLM-friendly 自研芯片**：UB+L1 融合是真正可叠加在 CANN 算子优化之上的、长上下文 prefill 上 ~20% 加速的硬件 lever
3. **LLM serving 主战场（decode）**：HBM3 是首要硬件投资；UB+L1 融合是次要

---

## 六、工具使用与扩展

### 6.1 添加新 workload
1. 在 NPU 上跑 msprof PipeUtilization 采集（见 `benchmark/run_phase_b.sh` 模板）
2. 用 `scripts/calibration/extract_cube_util.py`（含 PipeUtil 字段提取）抽取
3. 添加 entry 到 `data/pipe_baseline_per_model.json`
4. 重跑 `predict_optimization_ceiling.py` 即可

### 6.2 自定义优化情景
在 `scripts/calibration/predict_optimization_ceiling.py` 中可定义新的 `compute_*` 函数：
- 调整某个 pipe 的 scaling 系数
- 模拟不同硬件改动（如 fixpipe BW × 2）
- 模拟 KV cache prefetcher 等假想架构

### 6.3 工具发布后的位置（M1 后）
- API：`from prism.ceiling import predict_all_scenarios`
- CLI：`prism-ceiling --pipe-baseline ... --output-md ...`

---

## 七、产物索引

| 文件 | 内容 |
|------|------|
| 本文 (`docs/operator_optimization_ceiling.md`) | 5 情景分析 + 立项建议 |
| `data/optimization_ceiling.json` | 5 情景 × 11 配置 全结果 JSON |
| `scripts/calibration/predict_optimization_ceiling.py` | 主工具脚本 |
| `data/pipe_baseline_per_model.json` | Phase N 9 配置 PipeUtil 实测（输入）|
| `docs/overhead_decomposition_audit.md` | Phase N audit（pipe % 数据来源解释）|
| `docs/arch_hypothesis_rules.md` | Phase N pipe % → 架构推荐规则（互补阅读）|
