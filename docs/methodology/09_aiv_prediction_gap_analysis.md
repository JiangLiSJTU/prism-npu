# 09 — AIV 预测误差根因分析与改进路线

> **Issue #2 P1 延伸研究**：为什么 AIC 物理公式能泛化（未知模型 7% 误差），而 AIV 始终失败（51%+）？

## 1. 问题陈述

PRISM 的 `predict_pipe` 模块对新模型的预测精度呈现极端不对称：

| 组件 | 训练集 MAE | LOO CV MAE | 未知模型（ModernBERT）|
|------|-----------|------------|---------------------|
| AIC (T_aic) | ~5% | ~12% | **7%** |
| AIV (T_aiv) | ~12% | ~19% | **51%** |
| Wall-clock | ~8% | ~15% | ~25% |

AIC 用的是第一性原理公式（GEMM ops / (peak_MACs × η) × archetype_amplification），
AIV 用的是经验回归（aiv_time = α_archetype × aic_time），α 按 6 类模型分桶。

**核心问题**：为什么 AIC 的简单物理公式能泛化，而 AIV 需要经验分桶却仍然泛化失败？

---

## 2. 文献调研结论

### 2.1 Verrocchio — DaVinci 指令级离散事件模型

**Tang & Wang, JPDC 2023** — "Performance modeling on DaVinci AI core"

这是目前唯一公开的 DaVinci 架构白盒性能模型。核心发现：

1. **离散事件模型达到 2.62% 平均误差**：把每条 CCE 指令建模为离散事件（含 set_flag/wait_flag 二值信号量同步），跟踪每个硬件单元的执行时间线。

2. **Per-instruction 固定开销 = 40 ns**（Init 时间）；kernel launch = 2050 ns。

3. **MTE2/MTE3 Interconnect Bus 竞争**是最大误差源（18-34%）：当 MTE2（HBM→L1 权重加载）和 MTE3（UB→HBM 输出写回）同时执行时，带宽平均分享（half-duplex bus）。

4. **Vector Unit 和 Cube Unit 在指令级建模下误差很低**：单独测 0.69%（Vector）、1.17%（Cube）。但整个 kernel 的误差上升到 5%+ 是因为 MTE 调度和信号量同步引入的 idle 时间难以准确捕获。

5. **关键洞察**：论文明确指出 **"the most straightforward execution time computed by DataSize / Bandwidth"（即 PRISM 的 physics.py 方式）is ineffective**——因为它忽略了指令间的二值信号量依赖。同一组数据传输操作，仅改变 set_flag/wait_flag 的参数顺序，wall-clock 就差 1.26×。

### 2.2 Zhou et al. — Ascend 组件级 Roofline（ASPLOS 2025）

**"Squeezing Operator Performance Potential for the Ascend Architecture"**

对 Ascend 910 系列做了组件级 Roofline 分解：

1. **利用率分解**：U_component = E（效率）× R（时间占比）。

2. **Vector 利用率可低至 13.54%**：AvgPool 算子的 Vector 时间占比高达 83.98%，但效率仅 13.54%——原因是 CANN 编译器设置了 `repeat=1`（每次操作后重新 dispatch），每条 Vector 指令的 dispatch 开销远超实际计算。修正 `repeat` 从 1→98 后 **4.31× 加速**，利用率升到 59.07%。

3. **MTE 是终极瓶颈**：PanGu-alpha 100B 模型中 47.37% 的算子是 GM→UB 带宽受限（MTE-bound），论文明确说 "difficult to alleviate through software optimizations… emphasizing the need of next-generation chips"。

4. **层归一化（LayerNorm）不是独立建模的**——它只作为 element-wise 算子融合的目标出现。Softmax 论文中未提及。这意味着即使在 ASPLOS 2025 级别的分析中，Vector 上的非 GEMM 算子也没有独立的延迟模型。

### 2.3 NeuSight — Tile 粒度 GPU 预测器（ASPLOS 2025）

**Qi et al.** — 用 per-tile Roofline 上界 + ML 残差修正，把 GPT-3/H100 预测误差从 121% 降到 2.3%。

核心方法：
- 对每个 tile 算 Roofline 物理上界（compute-bound 或 memory-bound 中的 max）
- 训练轻量 ML 模型学习"peak vs effective"的 gap（derating factor）
- Gap 的主要来源：bank conflicts, cache miss patterns, instruction dispatch overhead

### 2.4 其他相关论文

| 论文 | 关键发现 | 与 AIV 预测的关系 |
|------|---------|-----------------|
| PENDRAM (arXiv 2408.02412, 2024) | DRAM 地址映射策略决定有效带宽：row-buffer hit rate 从 30%→96% | AIV 的 HBM 访存效率取决于 CANN tiling 决策的地址局部性 |
| "Pin or Fuse?" (CGO 2023) | Scratchpad 小负载下带宽严重衰减（-50%+） | UB 192KB 在小 tensor 上传输效率远低于峰值 |
| "Mind the Memory Gap" (IEEE CLOUD 2025) | 即使大 batch 下 attention kernel 仍 >50% cycle stall on DRAM | HBM 竞争使有效带宽远低于标称值 |
| Bandwidth-Aware Loop Tiling (PACT 2020) | DMA 传输效率随 tile 形状变化 4× | UB↔L1 的 MTE 效率取决于 tile 对齐和大小 |

---

## 3. 根因分析：AIC 为什么准、AIV 为什么不准

### 3.1 AIC 预测准确的 4 个原因

1. **GEMM 是确定性的大粒度操作**：16×16×16 MAC 流水线一旦启动，每个 cycle 固定消耗 4096 FP16 ops。指令 dispatch 开销被大 tile 数均摊。

2. **archetype_amplification 捕获了主要的非理想因素**：CANN tile 重加载的放大效应是 AIC 最大的非理想因素，它与 weight_proxy_mb（模型大小）和 S（序列长度）强相关——恰好是 amplification 函数的两个输入。

3. **AIC 的 pipe 结构简单**：mac + mte1 + mte2 + fixpipe 四条 pipe 中，mac 和 mte2 通常是瓶颈，它们的行为相对可预测（compute-bound 或 memory-bound，Roofline 两条线之一）。

4. **Cube 利用率在同一 archetype 内相对稳定**：msprof 实测 BERT 18.3%、GPT-2 28.7%、Qwen3 30.3%——同一 archetype 内变异系数（CV）< 30%，使得 η_compute=0.70 的经验值加 archetype 修正就够用。

### 3.2 AIV 预测失败的 6 个根因

#### 根因 1：Per-kernel 固定开销主导（Init time）

Verrocchio 实测：**每条指令 40 ns init + kernel launch 2050 ns**。

对于 BERT-base（338 个 kernel），Vector 部分的 kernel 约占 60%（~200 个），其中大量是小型 element-wise 操作（LayerNorm、GeLU、Add）。

粗算：200 kernels × (40 ns init + ~150 ns scalar dispatch) ≈ **38 μs 固定开销**。
PRISM 的 `aiv_vec()` 对 BERT 返回的理论 Vector ALU 时间大约 **0.3 μs**——相差 100×。

**这就是 physics.py 对 AIV 失效的首要原因**：公式只算了 `flops / throughput`，完全没有 per-kernel 固定项。

#### 根因 2：CANN 编译器的指令参数不可预测

Zhou et al. (ASPLOS 2025) 发现 Vector 利用率 13-59% 取决于 `repeat`、`mask` 等指令参数——这些是 **CANN 编译器的内部决策**，外部无法获知。同一个 LayerNorm 算子，CANN 可能生成 `repeat=1`（效率 13%）也可能生成 `repeat=98`（效率 59%）。

PRISM 的 physics 公式用 peak throughput 除，隐含 `repeat=max` 的乐观假设。

#### 根因 3：UB↔L1 带宽严重衰减

PRISM 用 `ub_l1_bw_gbs = 2048 GB/s`（峰值估计），但 CGO 2023 论文 "Pin or Fuse?" 证明小负载 scratchpad 带宽可衰减 50%+。

BERT d_model=768：LayerNorm 的单次 UB→L1 传输约 768×2 = 1.5 KB。在 192 KB UB 上，1.5 KB 传输的总线利用率远低于峰值（bus setup + address alignment 开销占比极高）。

#### 根因 4：HBM Interconnect Bus 竞争

Verrocchio 证实：MTE2 和 MTE3 **平均分享 Interconnect Bus 带宽**。当 AIC 的 mte2（权重加载）和 AIV 的 mte3（UB→HBM 输出写回）同时执行时，各自只得到 50% 峰值带宽。

PRISM 的 `aiv_mte3()` 用全速 fixpipe_bw 计算，没有 contention derating。

#### 根因 5：AIC/AIV 同步导致 idle 时间不可预测

Verrocchio 的核心发现：binary semaphore 的 set_flag/wait_flag 顺序决定 AIC 和 AIV 是并行还是串行。PRISM 假设 `T_aic + T_aiv`（完全串行），但实际上 CANN 可能让部分 Vector 操作与下一层的 Cube 操作 overlap，或者反过来——取决于 kernel 内的信号量编排。

这意味着 aiv_time 不是独立的——它与 aic 的调度有耦合。`aiv = α × aic` 模型假设了固定的耦合比例，但这个比例随编译器策略变化。

#### 根因 6：非 GEMM 算子异质性

LayerNorm、Softmax、GeLU、RMSNorm、Add 在 Vector Unit 上的指令模式完全不同：

| 算子 | 类型 | 主要 pipe | 特点 |
|------|------|----------|------|
| LayerNorm | reduction + element-wise | vec + mte2 | 需要两遍扫描（mean + variance），数据重读 |
| Softmax | reduction + transcendental | vec + mte2 | exp() 指令慢 5-10× vs add/mul |
| GeLU | element-wise | vec | tanh 近似需要多条指令 |
| RMSNorm | reduction + element-wise | vec + mte2 | 类似 LayerNorm 但少一遍 |
| Add/Mul | pure element-wise | vec | 最简单，throughput 接近峰值 |

PRISM 的 `compute_vector_ops()` 把所有非 GEMM ops 算成统一的 `flops`，用同一个 `aiv_vec()` 公式——完全忽略了算子类型差异。

---

## 4. 改进路线提案

### 方案 A（推荐）：Physics-informed 多因子模型

**核心思想**：保留物理公式框架，但补上 6 个缺失因子。

```python
def predict_aiv_time_v2(spec: ModelSpec, arch: dict, aic_time_us: float) -> float:
    """Physics-informed AIV prediction with per-kernel overhead + bandwidth derating."""

    # ── Factor 1: Per-kernel fixed cost (from Verrocchio: 40ns init + scalar) ──
    n_vec_kernels = estimate_n_vector_kernels(spec)
    T_init_us = n_vec_kernels * T_INIT_PER_KERNEL_US        # ~0.04 μs × n_kernels

    # ── Factor 2: Effective Vector throughput with repeat-parameter derating ──
    # Small tensors → low repeat → low utilization
    avg_tensor_elements = spec.S * spec.d_model               # typical LayerNorm size
    eta_repeat = min(1.0, avg_tensor_elements / REPEAT_SATURATION_THRESHOLD)
    T_vec_compute = physics.aiv_vec(vec_flops, arch) / eta_repeat

    # ── Factor 3: UB↔L1 bandwidth derating for small payloads ──
    avg_payload_bytes = spec.d_model * _FP16_BYTES             # single LayerNorm row
    eta_ub_bw = ub_bandwidth_derating(avg_payload_bytes)       # 0.3–1.0
    T_ub_transfer = physics.aiv_mte2(intermediate_bytes, arch) / eta_ub_bw

    # ── Factor 4: HBM contention (MTE2/MTE3 bus sharing) ──
    # When AIC mte2 is active, AIV mte3 gets ~50% bandwidth (Verrocchio finding)
    aic_mte2_active_fraction = estimate_aic_mte2_fraction(spec, arch)
    hbm_sharing_factor = 1.0 - 0.5 * aic_mte2_active_fraction
    T_mte3 = physics.aiv_mte3(output_bytes, arch) / hbm_sharing_factor

    # ── Factor 5: Scalar/dispatch overhead (proportional to kernel count) ──
    T_scalar = n_vec_kernels * T_SCALAR_PER_KERNEL_US

    # ── Combine: max of pipe bottlenecks + additive fixed costs ──
    T_pipe_bottleneck = max(T_vec_compute, T_ub_transfer, T_mte3)
    T_aiv = T_pipe_bottleneck + T_init_us + T_scalar

    return T_aiv
```

**需要从 msprof 拟合的 4 个参数**：
- `T_INIT_PER_KERNEL_US`：~0.04 μs（Verrocchio 先验 = 40 ns，验证用 23 configs）
- `REPEAT_SATURATION_THRESHOLD`：avg_tensor_elements 超过此值时 η_repeat → 1.0
- `ub_bandwidth_derating(payload_bytes)`：分段线性或 sigmoid，2-3 个参数
- `T_SCALAR_PER_KERNEL_US`：~0.1-0.2 μs（从 msprof scalar pipe 时间拟合）

**预期效果**：LOO CV MAE 从 19% → 10-12%；未知模型 MAE 从 51% → 15-25%。

### 方案 B：NeuSight 风格 Roofline + ML 残差

1. 对每类 Vector 算子（LN/Softmax/GeLU/Add）分别计算 Roofline 上界
2. 训练轻量 regressor 学习 actual/roofline 的 derating ratio
3. 需要 50+ 配置训练数据（当前只有 23 个），但泛化性可能更好

**代价**：需要 ModernBERT + SmolLM2 等新模型的 msprof 数据（P3 完成后）。

### 方案 C：Verrocchio 式指令级模拟（长期）

对 CANN 生成的 kernel 做指令级离散事件模拟——精度最高（2-5%），但需要：
1. 获取每个 kernel 的 CCE 指令序列（CANN 编译产物）
2. 实现完整的信号量同步仿真
3. 实现 MTE2/MTE3 bus 竞争模型

这是论文级工作量（3-6 个月），不适合当前项目节奏，但可作为长期方向。

---

## 5. 实施进展与最终方案

### 5.1 v3 实施（已落地，2026-05）

按方案 A 的简化版本落地：保留方案 A 的 cost-driver 结构，但用 3-bucket archetype amplification 替代复杂的 derating 子函数。

```python
aiv_time = (n_vk * C_kernel + data_MB * C_data) * archetype_amp
```

其中 archetype_amp 来自 3 个离散桶：small / large_prefill / decode。

**修复了的关键 bug**：`compute_vector_ops()` 中 softmax 项漏了一个 S 因子（`L*H*S*3` 而非 `L*H*S*S*3`），导致 O(S²) attention 数据量在 AIV 物理基线中被严重低估。

**v3 拟合结果**：6 configs Training MAE = **10.9%**，但 GPT-2 outlier 32.4%。

### 5.2 v4 实施（最终方案，2026-05-15）

v3 的 GPT-2 outlier 暴露了**离散 archetype 分辨率不足**：BERT (attn_frac=0.24) 和 GPT-2 (attn_frac=0.56) 同 weight_proxy=226 MB，被塞进同一个 "small" bucket，但实际 amp 需求差异巨大。

**v4 (Method B)**：用连续函数替代离散 bucket：

```python
if S == 1:
    amp = amp_decode_const    # ~1.5
else:
    # 关键：连续函数 of attn_frac + (w_proxy/1000)²
    amp = max(0.1, a0 + a1 * attn_frac + a2 * (w_proxy/1000)**2)
```

**Method B 的关键洞察**：`attn_frac = O(S²) attention bytes / total AIV data bytes` 是一个 model geometry 不变量，自然区分 attention-dominated（长序列）vs FFN-dominated（短序列）工作负载，分辨率比 weight_proxy 桶高得多。

**为什么 `(w_proxy/1000)²` 用平方而不是线性**：经验拟合发现线性 `w_proxy/1000` 只给 3.4× 动态范围（0.005 → 0.763），不足同时拟合 Qwen3（需 amp≈10）和 BERT（amp≈1.5）；平方后 11.4× 范围（0.000025 → 0.582）让联合拟合从 24% → 4.9% MAE。**这是经验值，不是 first-principles 推导**——可能但未验证的物理直觉是 tile 重取次数 ∝ n_tiles × L1-miss-rate（两者都随 weight_MB 增长）。

**v4 拟合结果**：6 configs Training MAE = **4.9%**，**6/6 < 10%**，无 outlier。

| Config | err | amp | attn_frac |
|---|---:|---:|---:|
| BERT-base-S128-b1 | 0.7% | 1.49 | 0.242 |
| GPT-2-S512-b1 | 6.7% | 2.76 | 0.561 |
| Qwen3-prefill-S512-b4 | 1.9% | 10.33 | 0.593 |
| Qwen3-prefill-S256-b1 | 2.5% | 9.64 | 0.421 |
| Qwen3-decode-Min4 | 8.5% | 1.50 | 0.000 |
| Net-Transformer-S256 | 9.4% | 1.36 | 0.390 |

**生产实现**：`src/prism/predict_pipe/predict.py::predict_aiv_v2`，参数化为 6 个 grid-fitted 常数 (`aiv_C_kernel_us`、`aiv_C_data_us`、`aiv_amp_a0/a1/a2`、`aiv_amp_decode`)。

详见 `docs/methodology/08_predict_pipe.md §3.5`。

### 5.3 v5 候选（未实施）

如果 OOS 验证（ModernBERT/SmolLM2 真机 msprof）发现 v4 的 `(w_proxy/1000)²` 在 weight_proxy ∈ [200, 600] MB 区间外推不准：

- 把 `(w_proxy/1000)^k` 中的 k 也加入 grid search
- 引入第三个连续特征（如 `seq_x_layers = S × layers / 1000`）
- 或回到方案 B：per-operator-class（LN/Softmax/GeLU）Roofline + 轻量回归

---

## 6. 引用

1. Y. Tang, C.-l. Wang. "Performance modeling on DaVinci AI core." JPDC 175 (2023): 134-149. — Verrocchio 指令级模型
2. Y. Zhou et al. "Squeezing Operator Performance Potential for the Ascend Architecture." ASPLOS 2025. DOI: 10.1145/3676641.3716243 — Component Roofline
3. T. Qi et al. "NeuSight: Forecasting GPU Performance for DNN Training." ASPLOS 2025. arXiv: 2407.13853
4. Z. Wang, Y. Zhang, F. Wei, et al. "Using Analytical Performance/Power Model and Fine-Grained DVFS to Enhance AI Accelerator Energy Efficiency." ASPLOS 2025. DOI: 10.1145/3669940.3707231. — 13 位作者，南京大学 LM-System Group。把每个算子 cycle vs freq 关系建模为 piecewise linear，5,000+ 算子 × 6 频率点 = 30K 数据点，operator-level MAE 1.96%，workload-level 4.62%。**Dataset/fitted models 未开源**（仅引用了两个第三方 Ascend 公共仓 ascend/pytorch、ascend/ModelLink），但方法论 reproducible（任何有 Ascend NPU + CANN profiler 的人可重采）。该 paper 与 PRISM 关注点正交（他们做 DVFS 能效，我们做固定频率下的 wall-clock 预测），但 piecewise-linear `cycle = f(freq)` 公式 (§4.3) 可作 Phase J 频率维 sweep 的解析基础。
5. PENDRAM. arXiv: 2408.02412 (2024) — DRAM data-mapping policy for CNN accelerators.
6. "Pin or Fuse?" CGO 2023 — Scratchpad pinning & layer-fusion optimization.
7. S. Rosenberg et al. "Mind the Memory Gap." IEEE CLOUD 2025. arXiv: 2503.08311
8. "Bandwidth-Aware Loop Tiling for DMA-Supported Scratchpad Memory." PACT 2020.
