# 架构敏感度分析方法论

## 1. 工具的核心问题

> **给定 baseline 架构（910B4）+ 一个候选 workload 集合（5 个固定网络 + LLM 模型），如果改变某个架构维度（如 n_cores、HBM BW、AIV:AIC、UB+L1 融合），每个 workload 的 wall-clock 会怎么变？哪些维度对哪些 workload 有真实杠杆？**

输出形式：

```
                       BERT  GPT-2  Qwen3-prefill-S4096  Qwen3-decode  Net-Trans
n_cores=8            | 1.00  1.00     2.95                 1.85         1.00
n_cores=16           | 1.00  1.00     1.50                 1.20         1.00
n_cores=24 (baseline)| 1.00  1.00     1.00                 1.00         1.00
n_cores=48           | 1.00  1.00     0.84                 0.92         1.00
ub_l1_fused=True     | 0.99  1.00     0.80                 0.93         1.00
hbm_bw=800 (HBM3)    | 1.00  1.00     1.00                 0.65         1.00
...
```

→ 每个 workload 在 12 个维度 × 5 个变体 = 49 个唯一 arch_variant 上的 wall-clock ratio。

## 2. 设计原则

### 2.1 910B4-anchored 双向 sweep（不从 310P 起手加法外推）

历史方案曾考虑"从 310P 起点加法 sweep"（J1）和"910B4 锚点双向 sweep"（J2）。**最终选 J2-only**：

| 维度 | J1（310P + 加法） | J2（910B4 双向）|
|------|-----------------|---------------|
| baseline 实卡校准 | 310P calib block 是 910B4 占位值（未实测）| 910B4 calib MAE 0.57% 实测拟合 |
| 加法变体（如 +cm × 4）| 双重不确定性（基线本身估计 + 加法外推）| 单重（加法相对实测基线）|
| 减法变体（如 -L2/2）| 不可做（已在 310P baseline）| 双向都可做 |

→ J2 + bidirectional 是唯一保留单一外推不确定性来源的设计。

### 2.2 每维度独立改变（不做联合 sweep）

不做 `n_cores × cube_spatial × L2` 全组合（数千 variants），而是 **6 维度 × 5 变体 = 49 唯一 variants**，每个 variant 仅改一个维度。理由：

1. **可解读**：单维度变体的 ratio 直接反映该维度的杠杆
2. **避免 hidden interaction**：联合 sweep 找出的 sweet spot 经常是某两个维度的非线性 interaction，难以排查根因
3. **复杂度可控**：49 variants × 5 models = 245 预测点 + 49 TCO 估算，**5 秒跑完**（pipe-aware 公式 全 analytical，无 Docker）

如需联合 sweep，单维度结果已经给出了"哪两个维度有显著 interaction"的提示——再做 targeted 联合 sweep（详见 §sweep 维度扩展）。

## 3. 12 个 sweep 维度

| 维度 | baseline 值 | sweep 范围 | 物理影响 |
|------|----------:|----------|----------|
| `n_cores` | 24 | 8 / 12 / 16 / 24 / 32 / 48 | aic_pipe.mac × n_cores（线性）|
| `cube_spatial` | 16×16×16 | 8×8×16, 8×16×16, 16×16×16, 16×16×32, 32×32×16 | aic_pipe.mac × cube_macs（线性）|
| `clock_ghz` | 1.6 | 0.8, 1.2, 1.6, 2.0, 2.4 | 全 device 时间 × 1/clock |
| `aiv_per_aic` | 2 | 1, 2, 4 | aiv_pipe.vec × aiv_count |
| `aiv_lanes_per_aiv` | 128 | 64, 128, 256 | aiv_pipe.vec × lanes |
| `hbm_bw_gbs` | 392 (HBM2e) | 50 (LPDDR4X), 100 (LPDDR5X), 392, 800 (HBM3) | aic_pipe.mte2 × 1/bw |
| `l1_l0_bw_gbs` | 2048 | 1024, 2048, 4096 | aic_pipe.mte1 × 1/bw |
| `fixpipe_bw_gbs` | 4096 | 2048, 4096, 8192 | aic_pipe.fixpipe + aiv_pipe.mte3 × 1/bw |
| `ub_l1_bw_gbs` | 2048 | 1024, 2048, 4096 | aiv_pipe.mte2 × 1/bw |
| `l2_mb` | 96 | 8, 16, 32, 96, 192, 384 | （未直接进 cycles 公式，arch yaml 仅记录）|
| `l0a_kb` / `l0b_kb` / `l0c_kb` / `ub_kb` | 64/64/128/192 | 各 ½, ×1, ×2 | （cap 进 mapping，不影响公式）|
| **`ub_l1_fused`**（布尔）| False | True | aiv_pipe.mte2 × 0.05（融合后残余）|

**两个高优先级 hypothesis 维度**：
- `ub_l1_fused = True`（消除 UB↔L1 流量）— Phase N audit 表明长上下文 prefill 上 ~20% 加速
- `hbm_bw_gbs` 升 HBM3 — Phase N audit 表明 LLM decode 上 ~35% 加速

## 4. Wall-clock 预测公式 v3

### 4.1 Pipe-aware predict_wallclock

baseline pipe time（每个 model 9-11 配置实测）作为输入。新 arch_variant 上 cycle 是 baseline cycle 按 scaling 缩放后取 max + bubble：

```python
def predict_wallclock_v3(model_key, batch, arch_variant,
                         baseline_arch=ASCEND_910B4_V2,
                         pipe_baseline=PIPE_BASELINE):
    pipe = pipe_baseline[model_key]
    
    # AIC pipes scaled
    cube_scale    = (baseline_arch.cube_total_macs * baseline_arch.clock) \
                    / (arch_variant.cube_total_macs * arch_variant.clock)
    bw_scale      = baseline_arch.hbm_bw / arch_variant.hbm_bw
    l1_l0_scale   = baseline_arch.l1_l0_bw / arch_variant.l1_l0_bw
    fixpipe_scale = baseline_arch.fixpipe_bw / arch_variant.fixpipe_bw
    
    aic_pipes_new = {
        'mac':     pipe.aic.mac     * cube_scale,
        'mte1':    pipe.aic.mte1    * l1_l0_scale,
        'mte2':    pipe.aic.mte2    * bw_scale,
        'fixpipe': pipe.aic.fixpipe * fixpipe_scale,
        'scalar':  pipe.aic.scalar,                        # arch-invariant
    }
    aic_bubble = pipe.aic_time - max(pipe.aic.values())   # baseline idle, 假设 arch-invariant
    aic_time_new = max(aic_pipes_new.values()) + aic_bubble
    
    # AIV pipes (similar)
    aiv_throughput_scale = (baseline_arch.aiv_per_aic * baseline_arch.aiv_lanes * baseline_arch.clock) \
                           / (arch_variant.aiv_per_aic * arch_variant.aiv_lanes * arch_variant.clock)
    ub_l1_scale = baseline_arch.ub_l1_bw / arch_variant.ub_l1_bw
    
    aiv_pipes_new = {
        'vec':    pipe.aiv.vec    * aiv_throughput_scale,
        'mte2':   pipe.aiv.mte2   * ub_l1_scale * arch_variant.ub_l1_fused_residual,  # 默认 1.0
        'mte3':   pipe.aiv.mte3   * fixpipe_scale,
        'scalar': pipe.aiv.scalar,
    }
    aiv_idle = pipe.aiv.idle
    aiv_time_new = max(aiv_pipes_new.values()) + aiv_idle
    
    # host gap：β_layer 假设 arch-invariant
    host_gap_new = pipe.n_kernels_per_inf * arch_variant.beta_host_gap_us_per_kernel
    
    return {
        'aic_time': aic_time_new,
        'aiv_time': aiv_time_new,
        'host_gap': host_gap_new,
        'wall_clock': aic_time_new + aiv_time_new + host_gap_new,
    }
```

### 4.2 公式可解锁的敏感度维度

经典 `wall = max(T_compute, T_overhead)` 用 `max()` 把 device-side 细节信号都吞了。pipe-aware 公式让以下维度**首次产生 ratio ≠ 1.0**：

| 维度 | 经典公式 ratio | pipe-aware ratio | 解锁机制 |
|------|--------------:|----------------:|---------|
| hbm_bw=50 (LPDDR4X), BERT b=1 | 1.000 | **1.32** | mte2 占 79.4%（aic_time × 0.794），bw_scale 7.84 → mte2 dominate |
| fixpipe_bw=2048, Qwen3-prefill-S4096 | 1.000 | **1.51** | fixpipe 占 55.6% baseline，bw_scale 2 → fixpipe dominate |
| ub_l1_fused=True, Qwen3-prefill-S4096 | 1.000 | **0.80** | aiv_mte2 占 73% baseline → 减为 5% 残余 |
| aiv_per_aic=4, Qwen3-prefill-S4096 | 1.000 | **0.91** | aiv_vec 占 29% baseline，throughput_scale 0.5 |
| hbm_bw=800 (HBM3), Qwen3-decode | 1.000 | **0.65** | aic_mte2 占 84.8% baseline，bw_scale 0.49 |

→ pipe-aware 升级**首次让 5 个细粒度敏感维度对芯片架构师可见**。

## 5. TCO 代理模型

工具用一个简化 TCO 公式估算每 arch_variant 的相对成本（不是绝对 BOM 估算，仅供 Pareto 分析）：

$$
\text{TCO\_score} = w_\text{die} \cdot A_\text{die} + w_\text{power} \cdot P_\text{TDP} + w_\text{mem} \cdot C_\text{mem}
$$

其中：

| 项 | 公式 | 权重 |
|----|------|----:|
| $A_\text{die}$ | `n_cores × (cube_macs + l0_kb_total) / 1024 + l2_mb × 0.5` (mm²) | 0.4 |
| $P_\text{TDP}$ | arch_variant.tdp_w | 0.3 |
| $C_\text{mem}$ | HBM cost = bw / 100；LPDDR cost = bw / 200；× 100 后入 score | 0.3 |

权重由 BOM/工艺经验值确定，可在 `src/prism/sweep/cost_model.py` 调整（pluggable）。

### 5.1 TCO 实测数字（910B4 baseline = 100%）

| 变体 | TCO | 与 910B4 比 |
|------|----:|-----------:|
| 910B4 baseline | 100% | 1.00 |
| LPDDR5X (100 GB/s) | 61% | **-39%** |
| LPDDR4X (50 GB/s) | 59% | -41% |
| HBM3 (800 GB/s) | 146% | +46% |
| n_cores=16（裁 1/3 cores）| 95% | -5% |
| n_cores=8（裁 2/3 cores）| 90% | -10% |
| ub_l1_fused（融合 SRAM）| 100% | 0%（die 持平，BW 利好）|
| TDP=200W | 89% | -11% |
| 综合 sweet spot（16 cores + LPDDR5X + UB融 + TDP200W）| **45%** | **-55%** |

→ 不做联合 sweep 也可推测综合 sweet spot 节约 ~55% TCO。

## 6. 测试 workload 集合（5 个 model）

工具的 sweep MODELS dict 选了覆盖性 5 model：

| 模型 | 特征 | 代表场景 |
|------|------|---------|
| **BERT-base S=128 b=1** | encoder, 短输入 | 固定网络业务 baseline（流量分类、URL filtering）|
| **GPT-2-small S=512 b=1** | decoder-style, 中等 prompt | 固定网络深度推理 |
| **Net-Transformer S=256 L=1** | 1-layer attention | 固定网络最轻量 inference |
| **Qwen3-0.6B prefill S=4096 b=1** | LLM 长上下文 prefill | LLM serving prefill 主战场 |
| **Qwen3-0.6B decode M=4 S_kv=128 b=1** | LLM autoregressive decode | LLM serving decode 主战场 |

每个 model 有 baseline pipe time（msprof 实测）+ ops 算子表（用于计算 cycles 缩放）。

加新 model 见 [tutorials/04_add_new_model.md](../tutorials/04_add_new_model.md)。

## 7. Sweep 结果汇总

完整结果见 `data/outputs/phase_j_sweep.json`（49 variants × 5 models = 245 数据点）+ [docs/findings/微架构探索报告.md](../findings/微架构探索报告.md)。本文摘要 5 个最重要发现：

### 7.1 固定网络业务架构无杠杆

49 个变体（12 维：n_cores、cube、l2、bw、fixpipe_bw、ub_l1_fused、tdp、aiv_per_aic 等）下 BERT/GPT-2/Net-Trans 的 ratio ≈ 1.000（v3 pipe-aware 修订：少数维度如 LPDDR4X / UB+L1 融合 / 长上下文 prefill 解锁后，ratio 偏离 1）。host-bound 部分原因：

```
BERT-base wall_clock = 16,210 μs
                    = 651  (T_aic) + 918 (T_aiv) + 14,079 (T_host_gap)
```

T_host_gap 占 87%。**改任何 device-side 维度（cores/cube/bw/UB/L0），wall_clock 不变**。

→ **固定网络自研芯片若仅服务固定网络业务，可大胆做减法**：n_cores → 16、HBM → LPDDR5X、TDP → 200W、AIV → 1。预计 TCO -55%、性能不退化。

### 7.2 LLM prefill 在 12 个维度上有真实杠杆

Qwen3-prefill-S4096 b=1 的关键变体：

| 变体 | wall_clock ratio | 物理 |
|------|-----------------:|------|
| n_cores=8 | 2.95× 慢 | aic_mac 主导被裁 1/3 |
| cube spatial 8x8x16 | 3.87× 慢 | aic_mac 主导被裁 1/4 |
| TDP=100W (clock 1.11 GHz) | 1.42× 慢 | clock 直接影响 mac |
| n_cores=48 | 0.84× 快 | mac 受益 |
| **ub_l1_fused** | **0.80× 快** | aiv_mte2 73.2% 主导被消除 |
| HBM3 (800 GB/s) | **0.85× 快** | aic_mte2 54.2% 主导被加速 |

→ LLM prefill 是 **芯片设计的真战场**，每个维度都有可量化的杠杆。

### 7.3 LLM decode 是 HBM-BW 主导

Qwen3-decode M=4 S_kv=128 b=1：

| 变体 | wall_clock ratio | 物理 |
|------|-----------------:|------|
| HBM3 (800 GB/s) | **0.65× 快** | aic_mte2 84.8% 主导，bw_scale 0.49 |
| LPDDR4X (50 GB/s) | 7.85× 慢 | mte2 直接放 7.84 倍 |
| n_cores=48 | 0.92× 快 | mac 仅 6.6%，n_cores 杠杆有限 |
| ub_l1_fused | 0.93× 快 | aiv_mte2 27.6%，UB 融合次要 |

→ **LLM serving 主战场（decode）的硬件投资优先级是 HBM3 而非 Cube/UB**。

### 7.4 加 Cube/Vector 算力对固定网络无效

n_cores 8 → 48 在 BERT/GPT-2/Net-Trans 上都是 ratio = 1.000；aiv_per_aic 1 → 4 也都是 1.000。原因 §7.1：host_gap 主导。

→ **固定网络业务自研芯片应去掉 50% AIV 数量**（aiv_per_aic = 1 而非 2）—— die area 节约 0%（AIV 占 die 极少），但减少了 routing 与功耗。

### 7.5 UB+L1 融合是长上下文 prefill 的硬件杠杆

ub_l1_fused = True 是 **Phase N audit 推出的 first hypothesis**——把 UB（192 KB/AIV）和 L1（512 KB/AIC）合并为单一 SRAM 池，消除 UB↔L1 数据搬运。

模拟方法：sweep 中 `ub_l1_fused=True` 时把 `aiv_pipe.mte2 × 0.05`（保留 5% 残余作为控制流路径）。

实测 baseline aiv_mte2 占比：
- BERT b=1: 23.5% → 融合后影响 ~24% × 0.95 = 22% AIV time 减少 → wall_clock < 5% 改进（固定网络 host-bound 限制）
- Qwen3-prefill-S4096 b=1: **73.2% → 融合后 AIV time 减少 70%**，wall_clock **20% 加速**

→ UB+L1 融合的最大 ROI 在长上下文 prefill。**这是一个仅靠 CANN 算子优化无法实现的硬件级 lever**。

## 8. Hypothesis 规则化（pipe % → arch 推荐）

工具自动从 PipeUtil 实测推导 arch 改进 hypothesis。规则：

| 触发条件（任意 model 实测）| 推断 hypothesis | 建议派生 arch sweep 维度 |
|---------------------------|----------------|------------------------|
| aic_mte2_ratio > 70% | **mte2-bound** | hbm_bw_gbs↑ / l2_mb↑ |
| aic_fixpipe_ratio > 50% | **output-write-bound** | fixpipe_bw_gbs↑ |
| aic_mac_ratio < 15% | **Cube 严重过剩** | n_cores↓ / cube_macs↓ |
| aic_pipe_bubble > 25% | **pipe stall**（L0 bank/MTE 同步）| l0a/b/c_bank_count↑ |
| aiv_mte2_ratio > 50% | **UB↔L1 流量 dominant** | UB+L1 fusion candidate |
| aiv_vec_ratio < 20% | **AIV ALU 过剩** | aiv_per_aic↓ / aiv_lanes↓ |
| host_gap > 60% wall_clock | **host CANN runtime bound** | 软件优化优先（不动芯片）|

CLI：`prism-regime --arch ... --model ...` 输出 regime + hypothesis（详见 [reference/cli.md](../reference/cli.md)）。

## 9. 加新 sweep 维度

加新维度（如 `kv_prefetcher_factor`）3 步：

1. 在 `arch/ascend_910b4_for_sweep_v2.yaml` 加新字段：
   ```yaml
   kv_prefetcher_factor: 1.0   # baseline
   ```
2. 在 `src/prism/sweep/runner.py` 的 `SWEEP` dict 加新维度：
   ```python
   SWEEP = {
       ...,
       'kv_prefetcher_factor': [0.5, 1.0, 2.0, 4.0],
   }
   ```
3. 在 `predict_wallclock_v3` 加 scaling 公式（如 `aic_pipe.mte2 *= 1.0 / kv_prefetcher_factor`）

→ sweep 自动包含新维度。

## 10. 为什么不强依赖 Timeloop

工具的 wall-clock 公式**完全基于 msprof 实测 + analytical scaling**——不调用 Timeloop。理由：

| Timeloop classic 失效 | 影响 sweep |
|---------------------|-----------|
| K-reduce 错用 spatial（mapper 把 K 当 spatial）| Cube cycles 16× 高估，已被 L#1 修复 |
| 核利用率 9.4%（mapper 仅用 3 个 spatial 因子）| Cube cycles 不收敛真实利用率 |
| MAC scale 不收敛（cm × 2 与 cm × 1 cycles 几乎一致）| 加 Cube spatial 在 Timeloop 中无效 |
| L2 容量零效果 | L2 维度 sweep 不能用 Timeloop |
| DRAM BW 零效果 | bw_gbs 维度 sweep 不能用 Timeloop |
| Vector / MTE 不建模 | LayerNorm/Softmax/AIV 全部不建模 |

→ Timeloop 在 6 个失效中**修不了 4 个**（设计 choice 而非 budget 问题）。完整论证 [legacy/docs/timeloop_failure_analysis_and_replan.md](../../legacy/docs/timeloop_failure_analysis_and_replan.md)。

工具中 Timeloop 仅在 `prism-mapping` 子命令下被可选调用，用于：
- GEMM mapping 探索（cm sweep、bank sweep 等微架构 mapping 决策）
- 与 manual mapping 对照验证 Cube cycles

主预测路径不依赖。

## 11. 已知局限

| # | 局限 | 影响 |
|---|------|------|
| 1 | β_layer arch-invariant 假设 | 减核 / 减 L2 时 host 调度路径变化未反映；ratio 是上界估计 |
| 2 | aic_bubble / aiv_idle 假设 arch-invariant | 改 pipe 重叠效率的硬件改动（如 prefetcher）无法预测 |
| 3 | l0_kb / l2_mb 维度仅 cap 不进 cycles 公式 | timeloop-model 不强制 L2 容量约束（[legacy/docs/timeloop_l2_capacity_audit.md](../../legacy/docs/timeloop_l2_capacity_audit.md)）|
| 4 | clock × cube_spatial 联合 sweep 未做 | 单维度 sweep 已揭示 sweet spot，联合 sweep 仅对极端配置有意义 |
| 5 | Timeloop 模拟 ub_l1_fused 用 0.05 系数 | 系数来自经验估值（融合后保留控制流路径），实际硅可能 0.03-0.10 |

完整局限：[06_assumptions_limits.md](06_assumptions_limits.md)

---

## 📚 参考

- 实测数据：`legacy/docs/architecture_sweep_report.md` v3 (Phase J pipe-aware 重写版)
- Hypothesis 规则源：`legacy/docs/arch_hypothesis_rules.md` (Phase N N7)
- 失效审查：`legacy/docs/timeloop_failure_analysis_and_replan.md`
- 公式实现：`src/prism/sweep/runner.py` 函数 `predict_wallclock_v3()`
- TCO 代理模型：`src/prism/sweep/cost_model.py`
- DaVinci 架构：HotChips 31 (2019), "Da Vinci: A Scalable Architecture for Neural Network Computing"
