# 假设与已知局限

> **本文是工具的 first-class 公开假设清单**。任何使用本工具结论的报告/论文/立项材料都应在引用时附本文链接，并在 conclusion 中引用相关 §。

工具按 4 类组织假设：
- §1 物理假设（来自 DaVinci 架构理解）
- §2 公式假设（建模简化）
- §3 数据假设（msprof 实测的覆盖范围）
- §4 外推假设（跨芯片、跨 batch、跨 model）

每个假设给出：①陈述；②来源；③影响范围；④如何验证 / 推翻。

---

## 1. 物理假设（来自 DaVinci 架构）

### 1.1 Cube 与 AIV 在单层关键路径上 serial

**陈述**：单 transformer layer 的关键路径 `Cube (Q proj) → UB → AIV (LayerNorm) → UB → Cube (K proj) → ...` 中 Cube 与 AIV serial 执行。`T_compute = T_aic + T_aiv` 简单加法成立。

**来源**：DaVinci HC31 架构图（Cube 输出经 FixPipe → UB；AIV 从 UB 读输入）+ msprof step_trace 实测（prefill 阶段 AIC/AIV 时间几乎完全 serial，重叠 < 5%）。

**影响范围**：所有 wall-clock 预测公式（[02_three_layer_roofline.md §7](02_three_layer_roofline.md#7-wall-clock-预测公式-v3)、[04_arch_sensitivity.md §4.1](04_arch_sensitivity.md#41-pipe-aware-predict_wallclock)）。

**如果不成立**：CANN runtime 跨 stream 的 layer-level 重叠会让 wall-clock 比预测 *略低*（最多 5%）。模型偏保守，可接受。

**验证手段**：实测 wall_clock vs 预测 (T_aic + T_aiv + host_gap) 比对。当前 9 配置 baseline 重现误差 < 5%（[02 §7.1](02_three_layer_roofline.md#71-baseline-重现验证)）。

### 1.2 AIC 各 pipe（mac/mte1/mte2/fixpipe/scalar）可并行

**陈述**：AIC 内 5 个 pipe 在硬件上彼此独立，可同时活跃。`T_aic = max(active pipes) + bubble`，**不**`T_aic = Σ pipes`。

**来源**：DaVinci HC31 PE 架构图 + msprof 实测 9 配置 Σ aic pipe ratio = 1.66-2.59（远 > 1，证明 pipe 重叠）。

**影响范围**：[02 §5](02_three_layer_roofline.md#5-t_aic-的-pipe-aware-拆分) AIC pipe 公式。

**如果不成立**：T_aic 应改为 Σ pipes（极端悲观），baseline 重现误差变 大幅升高。当前 baseline 误差 < 5% 反向证明此假设成立。

### 1.3 AIV pipes（vec/mte2/mte3/scalar）可并行

**陈述**：同 §1.2，AIV 内 4 个 pipe 也可并行。

**来源**：同 §1.2。

**影响范围**：[02 §6](02_three_layer_roofline.md#6-t_aiv-的-pipe-aware-拆分)。

### 1.4 Cube 16×16×16 spatial（含 K=16 spatial）

**陈述**：DaVinci Cube 是 16×16×16 = 4096 MAC 单元，K 维通过 adder tree 也是 spatial（不是 K=1 spatial + K=16 temporal）。

**来源**：DaVinci HC31 + Phase L#1 实证（用 K=16 spatial 的 manual mapping 让 BERT FFN1 cycles 从 49152 → 3072，util 6.25% → 100%，与理论 compute peak 完全一致）。

**影响范围**：[03_eta_real_model.md §3.2](03_eta_real_model.md#32-项-1η_pipeline-systolic-fill-drain) 公式中 16-block 数 $M_b = \lceil M/16 \rceil$ 等。

**如果不成立**：η_pipeline 公式分母会少一个量级，η_real 严重高估。

---

## 2. 公式假设（建模简化）

### 2.1 β_layer arch-invariant

**陈述**：减核 / 减 L2 / 加 cube spatial 时，host 调度路径不变化，`β_layer × L` 保持 baseline 910B4 实测值。

**来源**：实测仅在 910B4 上做（无法在不同 cores 数的硬件上独立校准）。

**影响范围**：所有 sweep ratio。`predict_wallclock_v3` 中 `host_gap_new = pipe.n_kernels_per_inf * arch_variant.beta_host_gap_us_per_kernel`，β 取 baseline 值。

**如果不成立**：实际 β 可能因 host 路径变化而 ±20% 漂移。所有 ratio 是上界估计。

**验证手段**：等到自研芯片 tape-out 后实测，才能与本工具预测交叉对照。无法预先验证。

**接受理由**：芯片立项决策不需要绝对 wall-clock，只需要 ratio（相对 baseline）。即使 ratio 有 ±20% 偏差，定性结论（"减半固定网络业务无杠杆 / LLM prefill UB 融合 20% 加速"）仍稳健。

### 2.2 aic_bubble 与 aiv_idle 跨 arch 不变

**陈述**：`aic_bubble = aicore_time - max(active pipes)` 是 baseline 实测值，sweep 时假设此值不随 arch 变。同理 aiv_idle。

**来源**：bubble 主要由 pipe 间同步（如 mte2 等待 mac 完成才启动下一轮）决定，**与具体 cores/cube/bw 无关**——但与 mapper 有关。

**影响范围**：`predict_wallclock_v3` 中 bubble 直接加在新 max 上。

**如果不成立**：改 pipe 重叠效率的硬件改动（如 prefetcher、mapping 优化器）让 bubble 减小，模型预测会偏保守。当前不建模这种改进。

**接受理由**：bubble 占 wall_clock 的小部分（5-15%），即使预测偏 保守也不显著改变 ratio 排名。

### 2.3 T_kernel_gap 暂归入 T_host_gap

**陈述**：当前 wall_clock 拆分中 `kernel_gap`（device-internal kernel 间 idle）与 `host_gap`（pure host scheduling）合并到一起。

**来源**：分离需要 task_time + step_trace 两份 msprof 数据联合解析，工具内部还未实现。

**影响范围**：[02 §3.2](02_three_layer_roofline.md#32-bert-base-b1-实例per-inference) 中固定网络场景模型的 kernel_gap ≈ 562 μs (3.5% wall_clock)。

**如果不成立**：kernel_gap 部分实际可能受 arch 影响（kernel 执行更快 → kernel 间 gap 更长？），但占比小，影响 < 5% 总 wall_clock。

### 2.4 host_gap_per_kernel 跨 model 不同

**陈述**：实测发现 BERT b=1 的 host_gap_per_kernel = 41.6 μs/op、GPT-2 b=1 = 36.2 μs/op、Qwen3-decode = 28 μs/op。Phase J sweep 取 **per-model 值**而非全局常量。

**来源**：n_kernels_per_inf 与 wall_clock_per_inf 实测除法，按 model 显然不同（不同 model 的 op 类型分布不同，CANN runtime overhead 不同）。

**影响范围**：sweep 中每个 model 用自己的 host_gap_per_kernel。Cross-model 类比（如"BERT 在 X arch 上的 wall-clock 缩短了多少%，类比 GPT-2"）不成立。

**接受理由**：单模型 ratio 是工具的核心输出，cross-model 类比仅用于直觉理解，不参与立项数字。

### 2.5 Vector op 用 analytical pipe model（非 Timeloop）

**陈述**：LayerNorm/Softmax/GeLU/RMSNorm 等非-MAC 算子用 `T_aiv_pipe.vec = vector_ops / (n_cores × aiv × lanes × clock)` 解析公式预测，不通过 Timeloop。

**来源**：Timeloop classic 不建模非-MAC 算子（problem.shape 仅支持 GEMM/CONV）。

**影响范围**：[03_eta_real_model.md §1](03_eta_real_model.md#1-问题timeloop-100-不等于硬件-100) + L#3a Vector 建模。

**如果不成立**：Vector 时间预测可能 ±30%。但 baseline pipe time 来自 msprof 实测（aiv_vec 实测值），所以 `aiv_pipe.vec` 部分是实测，scaling 才是公式。误差有界。

### 2.6 ub_l1_fused 用 0.05 残余系数

**陈述**：模拟"UB+L1 融合"硬件假想时，把 `aiv_pipe.mte2 *= 0.05`（保留 5% 残余作为控制流路径）。

**来源**：经验估值——融合后 UB↔L1 数据搬运消失，但仍有 register/writeback 等控制路径。FastAttention paper (Liu et al., 2024) 报告类似量级。

**影响范围**：[04 §7.5](04_arch_sensitivity.md#75-ub+l1-融合是长上下文-prefill-的硬件杠杆) UB 融合 hypothesis。

**如果不成立**：实际硅可能 0.03-0.10（更乐观或更悲观）。Qwen3-prefill-S4096 的预测 ratio 0.80 可能在 0.78-0.85 范围。**定性结论"UB 融合是长上下文最大硬件 lever"稳健**。

---

## 3. 数据假设（msprof 实测覆盖范围）

### 3.1 11 配置 PipeUtilization 实测覆盖

**陈述**：工具的 baseline 数据来自 11 个配置实测：

```
BERT-base S=128 b=1
GPT-2-small S=512 b=1
Net-Transformer S=256 b=1 (estimated by BERT scaling，无独立 PipeUtil)
Qwen3-prefill S=256 × b=1/4/8
Qwen3-prefill S=512 × b=4/8
Qwen3-prefill S=4096 × b=1
Qwen3-decode M=4 S_kv=128 b=1
Qwen3-Embedding S=4096 b=1 (substituted by Qwen3-prefill body)
```

**未覆盖**：
- Qwen3-7B / Qwen3-14B 等更大 LLM
- 极长上下文（S=8192/16384）
- 大 batch decode（M=128 etc.）
- 真实 8-bit 量化场景（INT8 cube_mac_int8_ratio）

**影响范围**：[03 §6](03_eta_real_model.md#6-训练--验证集设计) η_real fit + [04 §6](04_arch_sensitivity.md#6-测试-workload-集合5-个-model) sweep MODELS dict。

**如果不成立**：未覆盖 workload 上预测可能偏。例如 Qwen3-7B 的 ops 量级 10× Qwen3-0.6B，η_real 外推可能偏（拟合数据集大 GEMM 训练样本不足）。

**应对**：用户加新 model 后**必须**重 fit + 验证 BERT MAE < 15 pp 硬门槛。

### 3.2 Net-Transformer 占位

**陈述**：Net-Transformer 没有独立 PipeUtilization 实测，工具用 BERT-base 比例缩放（按 GEMM ops 数 × FFN_hidden 比例）作为占位。

**来源**：Net-Transformer 是 1-layer 固定网络场景模型，CANN 算子库未优化覆盖；用其它固定网络 encoder（BERT）比例缩放是次优近似。

**影响范围**：sweep 中 Net-Transformer 的 ratio 反映 BERT 的 pipe 分布而非真实。结论"Net-Transformer 在所有 sweep 维度上 ratio = 1.0"是因为占位继承了 BERT 的 host-bound 性质。

**如果不成立**：Net-Transformer 的真实 host_gap / aic_pipe 分布可能不同——但 1-layer + S=256 极小 workload，host_gap 主导是合理推断，结论稳健。

**应对**：未来用户用 Net-Transformer 实测 PipeUtilization 替换占位即可。

### 3.3 大 workload msprof analyze 失败

**陈述**：S=4096 b=8、S=8192 b=1 的 PipeUtilization 在 msprof analyze 阶段崩溃（已尝试 loop=5/2/1）。

**影响范围**：极端长上下文 + 大 batch 配置数据缺失。

**应对**：用 S=4096 b=1 + S=512 b=8 推测（不在拟合数据集中）。fit 时这些缺失配置不参与，误差按已有数据计。

### 3.4 计算每 op 的 baseline pipe time 用算术平均

**陈述**：msprof 一次跑 N inference 后，op_summary CSV 含 N 个 op 实例。工具按 op_type 取算术平均得 `aic_pipe.mac / aiv_time / etc.`。

**影响范围**：fit + sweep 数据。

**如果不成立**：极端首 inference vs 第 N inference 的 launch overhead 差异未被建模。但 warmup_count 已剔除前几个 inference，其余 N 个稳态运行差异 < 5%，可接受。

---

## 4. 外推假设（跨芯片、跨 batch、跨 model）

### 4.1 跨芯片外推

**陈述**：β_layer / η_real 拟合仅在 910B4 上做。310P 的 calib block 当前是 910B4 占位值，未独立校准。

**影响范围**：sweep 不直接对比 310P。Phase J **以 910B4 为锚点**，所有 ratio 相对 910B4 baseline。

**如果不成立**：直接做 310P sweep 会有双重不确定性（baseline 估计 + sweep 外推），不可信。

**应对**：未来需要 310P 时，**重新跑 11 配置 msprof on 310P** 独立校准。当前不做。

### 4.2 跨 batch 外推

**陈述**：fit 数据集中 b ∈ {1, 4, 8}。b ≥ 16 的预测靠 γ_B `log_2(B)` 外推。

**影响范围**：[03 §3.4](03_eta_real_model.md#34-项-3η_batch批次摊薄)。

**如果不成立**：b = 32/64 实测可能偏离 log 关系，γ_B 应重 fit。

**验证手段**：未来加 b=16 实测后建议重 fit + 比对外推预测。

### 4.3 跨 sequence length 外推

**陈述**：fit 数据集 S ∈ {128, 256, 512, 4096}。未含 1024 / 2048 / 8192 等。

**影响范围**：S 不在训练集时按公式预测，未实测验证。

**应对**：S=8192 b=1 当前数据缺失（msprof 崩）。如未来需要建议先采集再加进 fit。

### 4.4 5 模型测试集偏差

**陈述**：sweep MODELS dict 仅 5 模型。conclusion "固定网络业务无架构杠杆" 严格只在这 5 模型代表固定网络业务的前提下成立。

**应对**：用户加新固定网络业务模型（如 NetGPT-XL、流量大模型）到 MODELS 后再确认结论。

---

## 5. Timeloop 相关假设

### 5.1 Timeloop classic 4 个固有失效

**陈述**：Timeloop classic 不建模以下 4 类，是工具不在主预测路径中调用 Timeloop 的根因：

1. K-reduce 错用 spatial（mapper 把 K=1 当 spatial，与 DaVinci 的 K=16 spatial 不一致）→ 已在 L#1 通过 manual mapping 修正，但 auto mapping 仍失效
2. L2 容量在 timeloop-model 下不强制（depth 字段不影响 cycles）→ L#2 audit
3. DRAM BW 在 timeloop 下不进 cycles 公式（仅算 energy）→ G2 audit
4. Vector / MTE 不建模（problem.shape 仅 GEMM/CONV）→ 主预测路径完全绕开

**来源**：详见 `legacy/docs/timeloop_failure_analysis_and_replan.md`（v1.0 + 修订）。

**影响范围**：工具的所有预测公式均**不依赖 Timeloop**——`predict_wallclock_v3` 直接用 msprof 实测的 pipe time 做 scaling。

**Timeloop 仍在**：`prism-mapping` 子命令（GEMM mapping 探索 + manual mapping 验证），但**不参与 wall-clock / sweep / ceiling 预测**。

### 5.2 ParseTreeSpecs 单元素 subtree 断言

**陈述**：Timeloop `topology.cpp:1085` 的 `assert(subTrees.getLength() == 1)` 阻挡 AIV 写成 DaVinci_Core 并列 subtree。

**真根因**：即使绕过此断言（把 AIV 写成独立顶层 yaml），Timeloop 也算不出 Vector cycles（§5.1 第 4 项）。所以是表面阻碍，根因是 §5.1。

---

## 6. TCO 估算的局限

### 6.1 TCO 是相对代理，不是绝对 BOM

**陈述**：[04 §5](04_arch_sensitivity.md#5-tco-代理模型) 的 TCO_score 是 ① die area 代理 + ② TDP + ③ 内存成本 的加权和，权重 (0.4, 0.3, 0.3) 来自经验。

**用途**：仅 Pareto 排序（"哪个 variant 比 baseline 便宜多少"）。不可作为绝对 BOM 估算（自研芯片晶圆成本、HBM 单价等）。

### 6.2 die area 公式简化

**陈述**：`A_die = n_cores × (cube_macs + l0_kb_total) / 1024 + l2_mb × 0.5` 是线性叠加，**未考虑** wire routing、verification overhead、yield loss、edge cells 等。

**影响**：相对 die 代理 ≈ ±20% 偏差。Pareto 排名相对稳健。

### 6.3 内存成本简化

**陈述**：`C_mem = HBM bw / 100 vs LPDDR bw / 200`——HBM 单 GB/s 比 LPDDR 贵 2 倍是经验估值。实际市场价随容量、代际波动 50%+。

**用途**：定性结论"LPDDR5X TCO < HBM2e baseline" 稳健，定量数字需配市场询价校准。

---

## 7. 论文 / 评审引用建议

如本工具的某结论被论文 / 立项报告引用，**必须同时引用本文** + 相关 §：

| 结论 | 必引 § |
|------|------|
| "固定网络业务架构资源加减无杠杆" | §2.1 (β_layer arch-invariant), §3.4 (5 模型测试集偏差) |
| "LLM 长上下文 UB 融合 20% 加速" | §2.6 (0.05 残余系数), §3.1 (Qwen3-prefill-S4096 b=1 单点) |
| "η_real BERT 验证 MAE 14.33 pp" | §3.1 (训练 / 验证集), §4.2 (跨 batch 外推) |
| "wall-clock 模型 MAE 0.57%" | §1.1 (AIC/AIV serial), §1.2 (pipe 并行), §2.1 (β arch-invariant) |
| "工具不依赖 Timeloop" | §5.1 (Timeloop 4 失效) |
| "TCO -55% sweet spot" | §6.1 (TCO 是代理), §6.3 (内存成本简化) |

---

## 8. 局限的优先级（哪些可推翻 conclusion 哪些可接受）

按风险排序：

| 假设 | 风险等级 | 可推翻什么 conclusion | 验证 / 推翻成本 |
|------|--------|--------------------|---------------|
| §2.1 β_layer arch-invariant | **高** | 所有 sweep ratio | 极高（需自研芯片 tape-out 实测）|
| §3.4 5 模型测试集偏差 | 高 | "固定网络业务无架构杠杆" | 中（加新固定网络业务模型重测）|
| §4.1 跨芯片外推 | 高 | 不能直接对比 310P | 中（在 310P 重跑 11 配置 msprof）|
| §3.2 Net-Transformer 占位 | 中 | Net-Trans 单点结论 | 低（采集 Net-Trans PipeUtil）|
| §2.6 ub_l1_fused 残余系数 | 中 | UB 融合预测 ratio ±5% | 高（需硅 prototype）|
| §1.1 AIC/AIV serial | 低 | wall_clock baseline 重现 < 5% 已证明成立 | — |
| §3.3 大 workload 缺失 | 低 | 极限 S/B 外推 | 中（msprof 重试）|
| §6.x TCO 局限 | 低 | 绝对 BOM；不影响 Pareto 排名 | 高（市场询价）|

→ **风险等级"高"的 3 个假设是工具结论可信度的主要不确定来源**。任何 audit/评审应优先质询这 3 个。

---

## 📚 参考

- 历史失败 / 验证记录：`legacy/docs/`（21 篇 phase 叙述）
- Timeloop 4 失效论证：`legacy/docs/timeloop_failure_analysis_and_replan.md`
- 实测 baseline：`legacy/docs/cube_efficiency_calibration.md`、`legacy/docs/overhead_decomposition_audit.md`
- 公式实现：`src/prism/sweep/runner.py`、`src/prism/eta_real/fit.py`
