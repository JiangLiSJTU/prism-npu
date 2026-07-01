# 优化天花板预测

## 1. 工具的核心问题

> **给定一个 workload 的 msprof PipeUtilization 实测，纯算子优化（CANN TBE 算子库）能把 wall-clock 降到多少？再加 CANN runtime 优化能再降多少？再加硬件改动（UB+L1 融合 / HBM3 升级）还能再降多少？**

输出：5 个情景 × 11 个 workload 的 wall-clock + reduction% 表。

→ 直接回答**"算子库优化是否到天花板了？硬件改动还有多少 ROI？"** 这个工程决策问题。

## 2. 与 sweep（[04](04_arch_sensitivity.md)）的关系

| 工具 | 输入 | 输出 | 用途 |
|------|------|------|------|
| **sweep** | (model, arch_variant) 49 组合 | wall-clock ratio | 比较多个架构候选 |
| **ceiling** | (model) | 5 个优化情景的 wall-clock | 评估单个 baseline 架构上的优化潜力 |

**互补关系**：
- sweep 回答"如果我做 X 架构改动，性能怎么变？"
- ceiling 回答"如果我什么都不改硬件，CANN 算子优化能做到多少？"

ceiling 也通过 sweep 的 pipe-aware 公式做基础（见 [04 §4.1](04_arch_sensitivity.md#41-pipe-aware-predict_wallclock)），只是约束变体集是预设的 4 个优化情景而非 49 个 arch 维度。

## 3. 5 个优化情景定义

每个情景对 baseline 实测做特定的"消除项"假设：

| 情景 | 操作 | 物理依据 |
|------|------|---------|
| **S0 Baseline** | msprof 实测，无任何优化 | 现状 |
| **S1 Software ceiling** | aic_bubble → 0；aiv_idle → 0；kernel_gap → 0 | CANN TBE 完美双缓冲 + 完美指令调度 + 完美 AIC/AIV 异步 |
| **S2 Software + Runtime ceiling** | S1 + host_gap → 10 μs/kernel | CANN graph fusion + ModelLoad 缓存 + async dispatch |
| **S3 + UB+L1 融合**（硬件） | S2 + aiv_mte2 × 0.05 | 融合内存池消除 UB↔L1 algorithmic memory traffic |
| **S4 + HBM3 800 GB/s**（硬件） | S3 + aic_mte2 × (392/800) | 与 [04 §3](04_arch_sensitivity.md#3-12-个-sweep-维度) sweep 一致 |

每情景的 wall_clock 公式：

$$
\text{wall\_clock} = \max(\text{active\_aic\_pipes}) + \text{aic\_bubble}
                   + \max(\text{active\_aiv\_pipes}) + \text{aiv\_idle}
                   + \text{kernel\_gap} + \text{host\_gap}
$$

S1 把 aic_bubble、aiv_idle、kernel_gap 置 0；S2 再降 host_gap；S3 缩 aiv_mte2；S4 缩 aic_mte2。

### 3.1 关键修正：host_gap 单调递减

S2 把 host_gap 设为 `n_kernels × 10 μs`，但部分 workload 已 < 10 μs/kernel（如 Qwen3 decode 的 0.16 μs/kernel）。这种情况强行设到 10 反而**抬高** host_gap，违反"优化"语义。

修正：

```python
host_gap_S2 = min(host_gap_baseline, n_kernels × 10)
```

→ 已优于目标的 workload 保持 baseline 不变。

## 4. 拆 4 类瓶颈的物理模型

ceiling 工具基于以下分类决定哪些是软件可消除、哪些必须硬件改动：

| 瓶颈类别 | 实测信号 | 软件可消除？ |
|---------|--------|----------------|
| **流水线效率类**（单指令延迟、AIC/AIV 同步等待）| `aiv_idle%` 高、`aic_bubble%` 高、`aiv_scalar%` 高 | **能**——双缓冲/指令调度/CANN runtime 直接消除 |
| **算法搬运量类**（UB ↔ L1 字节数）| `aiv_mte2%` 高（UB 大量读写）| **不能**——bytes 由模型 + tile size 物理决定 |
| **输出回写类**（FixPipe/MTE3）| `aiv_mte3%` / `aic_fixpipe%` 高 | 不能（受硬件 BW 限制） |
| **HBM 带宽**（DRAM↔L1）| `aic_mte2%` 高 | 不能（硬件 BW 限制）|

→ 5 情景就是按这 4 类的物理 lever 顺序放进去的。

## 5. 11 配置实测结果（per inference μs）

| Workload | S0 baseline | S1 sw only | S2 sw+rt | S3 +UB融合 | S4 +HBM3 |
|---------|-----------:|----------:|---------:|----------:|---------:|
| BERT-base S=128 b=1 | 16,210 | 14,812 (8.6%) | 4,112 (74.6%) | 4,013 (75.2%) | 3,750 (76.9%) |
| GPT-2 S=512 b=1 | 17,280 | 13,985 (19.1%) | 5,587 (67.7%) | 5,587 (67.7%) | 5,182 (70.0%) |
| Net-Transformer S=256 | 195 | 189 (3.0%) | 25 (87.1%) | 24 (87.5%) | 23 (88.4%) |
| Qwen3-prefill-S256 b=1 | 78,000 | 44,046 (43.5%) | 44,046 (43.5%) | 37,606 (51.8%) | 29,205 (62.6%) |
| Qwen3-prefill-S256 b=4 | 140,000 | 94,433 (32.5%) | 94,433 (32.5%) | 85,444 (39.0%) | 73,659 (47.4%) |
| Qwen3-prefill-S256 b=8 | 225,000 | 162,749 (27.7%) | 162,749 (27.7%) | 150,180 (33.2%) | 133,741 (40.6%) |
| Qwen3-prefill-S512 b=4 | 301,880 | 210,326 (30.3%) | 210,326 (30.3%) | 193,339 (36.0%) | 158,352 (47.5%) |
| Qwen3-prefill-S512 b=8 | 603,760 | 444,782 (26.3%) | 444,782 (26.3%) | 413,515 (31.5%) | 350,034 (42.0%) |
| **Qwen3-prefill-S4096 b=1** | 3,050,000 | 1,963,103 (35.6%) | 1,963,103 (35.6%) | **1,338,829 (56.1%)** | 1,338,829 (56.1%) |
| **Qwen3-decode M=4 S_kv=128 b=1** | 7,690 | 3,111 (59.5%) | 3,111 (59.5%) | 2,597 (66.2%) | **1,520 (80.2%)** |
| Qwen3-Embedding S=4096 b=1 | 3,050,000 | 1,963,103 (35.6%) | 1,963,103 (35.6%) | 1,338,829 (56.1%) | 1,338,829 (56.1%) |

→ 完整数据：`data/outputs/optimization_ceiling.json`，渲染：[docs/findings/optimization_ceiling.md](../findings/optimization_ceiling.md)

## 6. 三类核心结论

### 6.1 固定网络业务（BERT/GPT-2/Net-Trans）

| 情景 | 增量 |
|------|------|
| S1 算子优化单独 | 3-19% 降幅（小） |
| **S2 + CANN runtime** | **67-87% 降幅**（真正杠杆！）|
| S3 + UB+L1 融合 | < 1% 增量（固定网络 vector 占比小，硬件无效）|

→ **固定网络业务 vector 瓶颈不是 algorithmic，是 host_gap 主导**。CANN runtime 优化（kernel launch 缓存、graph fusion、async dispatch）是杠杆。**固定网络自研芯片的硬件 vector 改动 ROI 几乎为 0**。

### 6.2 LLM 长上下文 prefill (Qwen3-S=4096)

| 情景 | 增量 |
|------|------|
| S1 算子优化 | **35.6% 降幅**（达到天花板）|
| S2 + runtime | 0% 增量（已在 runtime 优化下界）|
| **S3 + UB+L1 融合** | **20.5% 增量降幅**（53.5% → 56.1%）|
| S4 + HBM3 | 0% 增量（fixpipe-bound 而非 mte2-bound）|

→ **算子优化封顶后，UB+L1 融合是无法绕开的硬件 lever**——20% wall-clock 加速。**这是 LLM 长上下文 prefill 的最大硬件 ROI**。

### 6.3 LLM serving decode

| 情景 | 增量 |
|------|------|
| S1 算子优化 | **59.5% 降幅**（aiv_idle 大量可优化）|
| S3 + UB+L1 融合 | 6.7% 增量 |
| **S4 + HBM3** | **14% 增量**（80.2% 总降）|

→ **HBM3 是 decode 的首要硬件投资**。decode aic_mte2 占 84.8%，HBM3 让 mte2 时间减半，wall_clock -14%。

## 7. 用法（CLI）

```bash
prism-ceiling
# 默认从 data/calibration/pipe_baseline_per_model.json 读，输出：
#   data/outputs/optimization_ceiling.json
#   docs/findings/optimization_ceiling.md
```

可定制参数：

```bash
prism-ceiling \
  --pipe-baseline data/calibration/pipe_baseline_per_model.json \
  --output-json   data/outputs/optimization_ceiling.json \
  --output-md     docs/findings/optimization_ceiling.md \
  --host-gap-target 5     # S2 的 host_gap 目标，默认 10 μs/kernel
  --hbm3-bw-gbs 1200      # S4 的 HBM3 带宽，默认 800 GB/s
```

## 8. 添加新优化情景

工具支持扩展新情景。例如加 "S5: KV cache prefetcher"（hypothesis：硬件 prefetcher 把 aic_mte2 减半）：

```python
# src/prism/ceiling/scenarios.py
def compute_kv_prefetcher_ceiling(pipe, host_gap_target_per_kernel):
    """S5: S4 + KV cache prefetcher（aic_mte2 × 0.5）"""
    aic_pipes = dict(pipe['aic_pipes_us'])
    aic_pipes['mte2'] *= 0.5    # KV prefetcher hypothesis
    aic_pipes['mte2'] *= HBM2E_BW_GBS_BASELINE / HBM3_BW_GBS  # also HBM3
    aiv_active = {k: v for k, v in pipe['aiv_pipes_us'].items() if k != 'idle'}
    aiv_active['mte2'] *= UB_L1_FUSED_RESIDUAL  # also UB融合
    
    aic_time = max(aic_pipes.values())
    aiv_time = max(aiv_active.values())
    n_kernels = pipe['n_kernels_per_inf']
    host_gap = min(pipe['host_gap_us'], n_kernels * host_gap_target_per_kernel)
    
    return ScenarioResult(
        scenario='S5_kv_prefetcher',
        description='S4 + KV cache prefetcher（aic_mte2 × 0.5）',
        wall_clock=WallClockBreakdown(
            aic_time_us=aic_time, aiv_time_us=aiv_time,
            kernel_gap_us=0.0, host_gap_us=host_gap,
            wall_clock_us=aic_time + aiv_time + host_gap,
        ),
        ...
    )
```

然后在 `predict.py` 的 `predict_all_scenarios` 中加：

```python
s5 = compute_kv_prefetcher_ceiling(pipe, host_gap_target_per_kernel)
```

→ markdown 表自动多一列。

## 9. 加新 workload 配置

工具自动遍历 `data/calibration/pipe_baseline_per_model.json` 的所有 configs。加新 workload：

1. NPU 上跑 msprof PipeUtilization 采集（[05 §2.1](05_calibration.md#21-msprof-命令模板)）
2. `prism-extract` 提取 → 加进 `data/calibration/pipe_baseline_per_model.json`
3. 重跑 `prism-ceiling` → 自动出场

## 10. 与 04（sweep）的对照

工具的两个核心问题对照：

| 问题 | 工具 | 输出 |
|------|------|------|
| "改 X 架构维度，性能变多少？" | sweep ([04](04_arch_sensitivity.md)) | 49 variants × 5 models 的 ratio |
| "不改硬件 / 改硬件 X，最多能降多少？" | ceiling (本文) | 5 scenarios × 11 configs 的 ratio |

两个工具数据源一致（都从 `data/calibration/pipe_baseline_per_model.json` 读），公式同源（pipe scaling）。差别仅是变体集——sweep 是 12 维独立 sweep，ceiling 是 4 个累积优化情景。

## 11. 立项含义（写进战报的精炼版）

| 决策对象 | 数据驱动结论 | 来自 |
|---------|------------|------|
| 固定网络业务自研芯片是否值得做 vector 改动 | ❌ S3 增量 < 1%，无 ROI | §6.1 |
| LLM 长上下文 prefill 是否值得 UB+L1 融合 | ✅ **20% 加速**，仅靠算子优化达不到 | §6.2 |
| LLM serving decode 是否值得 HBM3 | ✅ **14% 加速** + LPDDR4X 致命退化 | §6.3 |
| CANN runtime 是否值得继续投资 | ✅ 固定网络业务 67-87% 杠杆，最大单 lever | §6.1 |

完整的立项推演见 [findings/主报告.md](../findings/主报告.md) §6.4。

## 12. 已知局限

| # | 局限 | 影响 |
|---|------|------|
| 1 | S1 假设算子完美双缓冲 | 实际 CANN 算子库可能未达完美，S1 偏乐观 |
| 2 | S3 ub_l1_fused 残余 0.05 系数 | 详见 [06 §2.6](06_assumptions_limits.md#26-ub_l1_fused-用-005-残余系数) |
| 3 | S4 HBM3 BW 800 GB/s 假设 | HBM3e 可达 1200，未建模 |
| 4 | host_gap 假设非线性折减 | 实际 graph fusion 不能折减到 10 μs/kernel for all workload |
| 5 | 11 配置覆盖有限 | 详见 [06 §3.1](06_assumptions_limits.md#31-11-配置-pipeutilization-实测覆盖) |

完整局限：[06_assumptions_limits.md](06_assumptions_limits.md)

---

## 📚 参考

- 工具实现：`src/prism/ceiling/predict.py`
- 数据源：`data/calibration/pipe_baseline_per_model.json`（[05](05_calibration.md) 流程产出）
- pipe scaling 公式：[04 §4.1](04_arch_sensitivity.md#41-pipe-aware-predict_wallclock)
- 历史快照：`legacy/docs/operator_optimization_ceiling.md`（v1，含 Phase N 印记的原始版）
- DaVinci UB+L1 融合 hypothesis 物理依据：[04 §7.5](04_arch_sensitivity.md#75-ub+l1-融合是长上下文-prefill-的硬件杠杆) + FastAttention paper (Liu et al., 2024)
