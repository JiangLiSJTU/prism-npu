# 三层 Roofline Wall-Clock 模型

## 1. 问题：标准 Roofline 不能解释昇腾实测

Williams 2009 提出的经典 Roofline 模型把单 op 性能预测为：

$$
T_\text{compute}^{\text{Williams}} = \max\left(\frac{\text{ops}}{\text{peak\_TFLOPS}},\ \frac{\text{bytes}}{\text{peak\_BW}}\right)
$$

直接套用到昇腾 910B4 上跑 BERT-base S=128 b=1：

| 量 | 解析预测 | msprof 实测 | 偏差 |
|----|--------:|-----------:|-----:|
| `ops_per_inference` | 22.7 GFLOP（1 inference 全模型）| — | — |
| `T_compute = ops / 280 TFLOPS` | 81 μs | — | — |
| `bytes / 392 GB/s` (HBM2e) | ~120 μs | — | — |
| `T_williams = max` | **120 μs** | — | — |
| **wall_clock 实测** | — | **16,210 μs** | **135× 低估** |

→ 经典 Roofline 在昇腾上**预测严重低估**。原因不是 Cube/HBM 出错，而是 wall-clock 中**有大量时间在 device-active 之外**——经典 Roofline 假设 host overhead 可忽略，这在昇腾的 graph + kernel-launch 调度路径下完全失效。

## 2. 三层模型

本工具使用扩展 Roofline 公式：

$$
T_\text{wall-clock} = \underbrace{T_\text{aic} + T_\text{aiv}}_\text{NPU device-active} + \underbrace{T_\text{kernel\_gap}}_\text{device-internal idle} + \underbrace{T_\text{host\_gap}}_\text{host scheduling}
$$

每层的物理含义：

| 层 | 物理含义 | 来源 |
|----|---------|-----|
| **T_aic** | AI Core 单元活跃总时间 = max(active aic pipes) + aic_bubble | msprof `aicore_time(us)` 实测 |
| **T_aiv** | AI Vector 单元活跃总时间 = max(active aiv pipes) + aiv_idle | msprof `aiv_time(us)` 实测 |
| **T_kernel_gap** | kernel 内 device idle（AIC/AIV 等待同步、event wait）| msprof `Task Duration - max(aic, aiv)` |
| **T_host_gap** | kernel 间 host 调度时间（kernel launch、graph dispatch、D2H sync）| `wall_clock - Σ Task Duration` |

注意 **T_aic 和 T_aiv 在单层关键路径上是 serial**（数据依赖：Cube → UB → AIV → UB → Cube），所以它们相加而非取 max。证据见 [03_eta_real_model.md](03_eta_real_model.md) §AIV 与 AIC 的 serial 关系实测。

### 2.1 BERT-base b=1 实例（per inference）

```
wall_clock           = 16,210 μs
─ T_aic              =    651 μs   (Cube 实际工作)
─ T_aiv              =    918 μs   (AIV 实际工作)
─ T_kernel_gap       =    562 μs   (device 内 pipe gap，3.5%)
─ T_host_gap         = 14,079 μs   (kernel 间 host gap，**87%**)
```

→ BERT b=1 上 wall-clock 的 **87% 是 host_gap**。这就是经典 Roofline 算 120 μs 而实测 16210 μs 的根因。

## 3. 经典 Roofline 与本模型的对比

| 维度 | 经典 Roofline (Williams 2009) | 本三层模型 |
|------|-----------------------------|------------|
| 假设 | host overhead 可忽略 | host overhead 显式建模为 β |
| `T_compute` 公式 | `ops / peak_TFLOPS` | `Σ_op (M·N·K / cube_throughput / η_real)` |
| `T_memory` 公式 | `bytes / peak_BW` | 隐含在 `T_aic_pipe.mte2` 中（pipe-aware）|
| 处理 op 间空隙 | 无 | `T_kernel_gap` + `T_host_gap` 显式拆 |
| 适用 workload | 大 GEMM 单 op | 整模型 inference（含 multi-op、multi-layer）|
| 校准来源 | 厂商峰值 | msprof 实测 |

经典 Roofline 在 device-bound workload 上仍然准确（如 Qwen3 prefill S=4096 b=1 device_active 占 wall_clock 的 95%），但对 host-bound workload（固定网络 BERT/GPT-2 b=1）误差 100×+。

## 4. β 校准方法

`T_host_gap` 不能解析推导，必须从 msprof 实测拟合。校准模型分两层：

$$
T_\text{host\_gap}(\text{model}, B) = \beta_\text{device} + \beta_\text{layer} \cdot L + \alpha \cdot (B - 1)
$$

| 参数 | 含义 | 单位 |
|------|------|----|
| **β_device** | 与具体 model 无关的 device 端固定开销（model load、初始化）| μs（一次性）|
| **β_layer** | 每层 transformer 加上的 host 调度开销（per encoder/decoder layer）| μs/layer |
| **α** | 每个额外 batch 加上的 D2H 同步开销 | μs/batch |
| **L** | 模型层数 | int |
| **B** | batch size | int |

### 4.1 拟合数据集

7 个模型 × 3-4 batch = 21 数据点：

| 模型 | L | hidden | S | 测试 batch |
|------|---|-------|---|----------|
| BERT-base | 12 | 768 | 128 | 1, 4, 8, 16 |
| GPT-2-small | 12 | 768 | 512 | 1, 4, 8, 16 |
| Qwen3-0.6B prefill | 28 | 1024 | 512 | 1, 4, 8 |
| Net-Transformer | 1 | 384 | 256 | 1 |
| ET-BERT | 12 | 768 | 128 | 1 |
| Kitsune | — | — | — | 1 |
| MalConv2 | — | — | — | 1 |

### 4.2 OLS 拟合结果（910B4 上）

按 model class 分组拟合（encoder vs decoder vs LLM prefill）：

| model_class | β_layer (μs) | 含义 |
|-------------|------------:|------|
| encoder（BERT、ET-BERT）| 138.1 | per encoder layer |
| encoder small d (d≤256) | 93.2 | small-hidden encoder |
| decoder S=256 d=384 (NetTrans) | 80.4 | small decoder |
| decoder S=512 d=768 (GPT-2) | 355.6 | medium decoder |
| decoder S=512 d=1024 (Qwen3 prefill) | 560.4 | LLM prefill |
| **β_device** (model-invariant) | 119.0 | 全模型共用 |

### 4.3 校准误差

7 模型 × 21 数据点的 OLS 拟合：

| metric | 值 |
|--------|---|
| MAE（wall-clock 预测）| **0.57%** |
| RMSE | 1.2% |
| 最差单点 | 3.2%（Qwen3 prefill b=8）|

→ wall-clock 模型 MAE 0.57% 已远低于其它误差来源（Cube η_real fit MAE 14 pp = ~5% × 5 models = 25% upper bound）。**β_layer 拟合不是误差瓶颈**。

## 5. T_aic 的 pipe-aware 拆分（本工具的关键升级）

经典 Roofline 把 T_compute 看成单一标量。msprof PipeUtilization 让我们更细：

`aicore_time` 是 AIC 单元活跃总时间。其中包含 5 个 sub-pipe，**这些 pipe 在硬件上可以并行执行**：

| Pipe | 物理含义 | 与 arch 维度的对应 |
|------|---------|------------------|
| `aic_mac` | Cube MAC 单元活跃 | `n_cores × cube_m × cube_n × cube_k` |
| `aic_mte1` | L1 → L0A/L0B 数据搬运 | L1 容量、L0A/L0B bank 数 |
| `aic_mte2` | DRAM/L2 → L1 数据搬运 | HBM/LPDDR 带宽、L2 容量 |
| `aic_fixpipe` | L0C → 输出回写（L1/UB **或 GM 直写**）| HBM 带宽 + FixPipe 带宽（按 `gm_frac` blend）|
| `aic_scalar` | AIC 内 scalar 控制流 | 微架构控制单元 |

→ **T_aic = max(active pipes) + aic_bubble**（关键路径 + 全 pipe 都 idle 的间隙）。

具体公式：

$$
\begin{aligned}
\text{aic\_bubble} &= \text{aicore\_time} - \max_\text{pipes}(\text{pipe\_time}) \\[4pt]
T_\text{aic} &= \max\bigl(t_\text{mac},\ t_\text{mte1},\ t_\text{mte2},\ t_\text{fixpipe},\ t_\text{scalar}\bigr) + \text{aic\_bubble}
\end{aligned}
$$

每个 pipe time 受不同 arch 维度影响。当架构变化时，每个 pipe 按 scaling 公式独立缩放：

| Pipe | scaling 公式 | 受影响的 arch 维度 |
|------|-------------|-------------------|
| mac | `× (cube_macs_old × clock_old) / (cube_macs_new × clock_new)` | n_cores、cube_spatial、clock |
| mte1 | `× (l1_l0_bw_old) / (l1_l0_bw_new)` | L1↔L0 带宽 |
| mte2 | `× (hbm_bw_old) / (hbm_bw_new)` | HBM/LPDDR 带宽 |
| fixpipe | `× blend(hbm_bw, fixpipe_bw)` 按 `gm_frac` | HBM 带宽 + FixPipe 带宽 |
| scalar | 不变（control logic）| — |

→ 这就是 [04_arch_sensitivity.md](04_arch_sensitivity.md) `predict_wallclock_v3` 的核心。

> **`aic_fixpipe` 是双带宽 blend（Issue #7）**：FixPipe 把 Cube 结果搬出 L0C，目的地
> 可为片上（L1/UB，`fixpipe_bw`）**或 GM 直写**（V200 FixPipe 增强，`hbm_bw`）。OLS
> 实测（`scripts/calib_fixpipe_mte3_bw.py` → `data/calibration/pipe_dest_bw.json`）显示
> **多数 config `aic_fixpipe` 有效带宽落在 HBM 量级（240–800 GB/s）、`gm_frac` 0.4–1.0**
> ——即 `aic_fixpipe` 主要是 HBM-write-bound，不是 FixPipe 单元带宽 bound。长 prefill
> attention 输出回写（如 Qwen3-S4096，`gm_frac≈0.8`）尤其如此。详见
> [05_calibration.md §3.2](05_calibration.md) 与 [08_predict_pipe.md §3.6](08_predict_pipe.md)。

### 5.1 9 配置 AIC pipe 占比实测

| 配置 | aic_us | mac% | mte1% | mte2% | fixpipe% | scalar% |
|------|------:|----:|----:|--------:|------:|------:|
| BERT-base S=128 b=1 | 651 | 29.5 | 34.4 | **79.4** | 11.6 | 11.7 |
| GPT-2-small S=512 b=1 | 1716 | 43.5 | 31.9 | 67.2 | 30.3 | 7.6 |
| Qwen3-prefill-S256 b=1 | 2164 | 48.0 | 44.0 | **86.8** | 22.2 | 10.5 |
| Qwen3-prefill-S256 b=4 | 5253 | 68.1 | 53.5 | **90.6** | 34.5 | 9.5 |
| Qwen3-prefill-S256 b=8 | 9746 | 74.9 | 60.5 | **91.8** | 37.4 | 9.1 |
| Qwen3-prefill-S512 b=4 | 17285 | 65.1 | 52.0 | **90.4** | 35.9 | 7.3 |
| Qwen3-prefill-S512 b=8 | 34364 | 67.5 | 54.2 | **90.6** | 40.3 | 7.1 |
| Qwen3-prefill-S4096 b=1 | 266264 | 33.6 | 28.5 | 54.2 | **55.6** | 3.5 |
| Qwen3-decode M=4 S_kv=128 b=1 | 2489 | **6.6** | 19.2 | **84.8** | 7.6 | 6.1 |

→ **8/9 配置 mte2 主导**（54-92%），唯例外 Qwen3-prefill-S4096 b=1 是 fixpipe 主导（attention 输出回写瓶颈）。

→ **Qwen3 decode aic_mac 仅 6.6%**——LLM serving 主战场上 Cube 算力严重过剩，HBM 带宽是真正瓶颈。

## 6. T_aiv 的 pipe-aware 拆分

类似 AIC，AIV (Vector) 端也有 4 个 pipe：

| Pipe | 物理含义 | 受 arch 影响维度 |
|------|---------|----------------|
| `aiv_vec` | Vector ALU SIMD 计算 | aiv_per_aic、vector_lanes、clock |
| `aiv_mte2` | UB ↔ L1 数据搬运 | UB↔L1 带宽 |
| `aiv_mte3` | UB → 输出 写回（MTE3 引擎）| HBM 带宽 + UB↔L1 带宽（按 `gm_frac` blend）|
| `aiv_scalar` | AIV 内 scalar 控制流 | — |

`T_aiv = max(active pipes) + aiv_idle`，公式与 AIC 同构。

> **`aiv_mte3` ≠ FixPipe，且是双带宽 blend**：`aiv_mte3` 由 MTE3 引擎承载，MTE3 的
> store 有两个目的地——`copy_ubuf_to_gm`（UB→GM，受 HBM 带宽限）与
> `copy_ubuf_to_cbuf`（UB→L1，受 UB↔L1 带宽限）。FixPipe 是 AIC 侧 L0C→输出的
> 专属单元，AIV 不经过它。每 config 的 GM 字节占比 `gm_frac` 由 msprof 实测校准
> （`data/calibration/pipe_dest_bw.json`，详见 [05_calibration.md §3.3](05_calibration.md#33-aivai-vector端字段)
> 与 [08_predict_pipe.md §3.6](08_predict_pipe.md)）。**实测结论**：长上下文大 prefill
> 模型 `gm_frac ≈ 0.55–0.75`（MTE3 主要写 GM，访存瓶颈），小短序列模型
> `gm_frac ≈ 0`（留在片上）。

### 6.1 9 配置 AIV pipe 占比实测

| 配置 | aiv_us | aiv_vec% | aiv_mte2% | aiv_mte3% | scalar% |
|------|------:|--------:|---------:|---------:|------:|
| BERT-base S=128 b=1 | 918 | **8.3** | 23.5 | 9.3 | 12.7 |
| GPT-2-small S=512 b=1 | 3376 | 36.3 | 32.0 | 16.7 | 12.2 |
| Qwen3-prefill-S256 b=1 | 3406 | 7.3 | 39.0 | 20.1 | — |
| Qwen3-prefill-S512 b=8 | 30544 | 26.4 | 60.9 | 48.1 | 5.1 |
| Qwen3-prefill-S4096 b=1 | 316770 | 29.1 | **73.2** | 33.8 | — |
| Qwen3-decode M=4 b=1 | 2877 | 8.6 | 27.6 | 7.1 | 9.8 |

→ **Vector ALU 仅占 aiv_time 的 7-36%**——其余 64-93% 是数据搬运。**Vector 是 memory-bound 而非 compute-bound 单元**。这直接驱动 [04 §UB+L1 融合 hypothesis](04_arch_sensitivity.md#ub-l1-融合-hypothesis)。

## 7. wall-clock 预测公式 v3（最终版）

把 §5 + §6 + §4 综合：

$$
\boxed{
T_\text{wall-clock}(\text{model}, \text{batch}, \text{arch}) = T_\text{aic}(\text{arch}) + T_\text{aiv}(\text{arch}) + T_\text{host\_gap}(\text{model}, \text{batch})
}
$$

其中：

```python
T_aic = max(
    pipe.aic.mac     × cube_scale,
    pipe.aic.mte1    × l1_l0_scale,
    pipe.aic.mte2    × bw_scale,
    pipe.aic.fixpipe × fixpipe_blend_scale,   # blend(hbm_scale, fixpipe_scale) 按 gm_frac
    pipe.aic.scalar
) + aic_bubble_baseline

T_aiv = max(
    pipe.aiv.vec     × aiv_throughput_scale,
    pipe.aiv.mte2    × ub_l1_scale,
    pipe.aiv.mte3    × mte3_blend_scale,   # blend(hbm_scale, ub_l1_scale) 按 gm_frac
    pipe.aiv.scalar
) + aiv_idle_baseline

T_host_gap = β_device + β_layer × L + α × (B-1)        # arch-invariant per assumption [06 §3]
```

baseline 数据 `pipe.aic`、`pipe.aiv` 来自 msprof 实测（详见 [05_calibration.md](05_calibration.md)）；scaling factor 来自 `arch.yaml`（详见 [reference/arch_yaml_schema.md](../reference/arch_yaml_schema.md)）；β 系数来自 [04.2](#42-ols-拟合结果910b4-上)。

### 7.1 baseline 重现验证

把 baseline 910B4 的 arch 喂给上面公式，应当复现实测 wall-clock：

| 配置 | 实测 wall_clock | v3 模型预测 | 误差 |
|------|---------------:|----------:|----:|
| BERT-base b=1 | 16,210 μs | 16,089 μs | -0.7% |
| GPT-2-small b=1 | 17,280 μs | 17,134 μs | -0.8% |
| Qwen3-prefill-S4096 b=1 | 3,050,000 μs | 3,041,200 μs | -0.3% |
| Qwen3-decode M=4 b=1 | 7,690 μs | 7,521 μs | -2.2% |

→ baseline 重现误差 < 5%（用户设的硬门槛）。

## 8. Regime 分类（Decision Gate）

工具自动判定每对 (model, arch) 落在哪一类 regime，决定下游优化优先级：

```
IF T_host_gap > 2 × max(T_aic, T_aiv)        → "host-bound"     (软件优化为主)
ELIF T_aic > 2 × T_aiv                        → "cube-bound"     (硬件 Cube 维度有效)
ELIF T_aiv > 2 × T_aic                        → "vector-bound"   (硬件 AIV 维度有效)
ELIF aic_pipe_dominated == 'mte2'             → "memory-bound"   (硬件 BW/L2 维度有效)
ELSE                                          → "balanced"       (混合)
```

CLI 调用：

```bash
prism-regime --arch arch/ascend_910b4_for_sweep_v2.yaml \
           --model models/qwen3_0.6b.yaml --batch 1
# 输出：{model: Qwen3-0.6B, arch: 910B4, batch: 1, regime: cube-bound, ...}
```

→ 详见 [05 §regime gate 校验流程](05_calibration.md#regime-gate-校验流程)。

## 9. 与 Timeloop 的关系

本三层模型**不依赖 Timeloop**——baseline pipe time 直接来自 msprof 实测。Timeloop 仅在以下场景被调用：

| 场景 | 是否需要 Timeloop |
|------|----------------|
| baseline wall-clock 预测（已有 msprof）| 不需要 |
| 已知 model 在新 arch_variant 上的 ratio 预测 | 不需要（用 pipe scaling）|
| 探索 GEMM 算子的 mapping 空间 | 需要（[mapper/](../../src/prism/mapper/) + Docker）|
| 用 manual mapping 验证 cycle 数与公式一致 | 可选 |

理由：Timeloop classic 在昇腾 4 个失效（Vector 不建模、MAC scale 不收敛、L2 容量零效果、DRAM BW 零效果）让它无法独立预测 wall-clock。本工具 sweep 的核心信号链是：**msprof 实测 pipe time** → **arch scaling factor** → **wall-clock 预测**，绕开 Timeloop classic 的所有失效点。

完整论证见 [legacy/docs/timeloop_failure_analysis_and_replan.md](../../legacy/docs/timeloop_failure_analysis_and_replan.md)。

## 10. 已知局限

| # | 局限 | 影响 | 缓解 |
|---|-----|------|-----|
| 1 | β_layer 假设跨 arch_variant 不变 | 减核 / 减 L2 时 host 调度路径会变，β ±20% 漂移 | 接受为上界估计，[06 §3.1](06_assumptions_limits.md#31-β_layer-arch-invariant-假设) 详述 |
| 2 | aic_bubble / aiv_idle 假设 arch-invariant | 改 pipe 重叠效率的硬件改动（如 prefetcher）无法预测 | 仅在 [03](03_eta_real_model.md) 提及，未量化 |
| 3 | T_kernel_gap 暂归入 host_gap | 真实分解需更深 msprof 解析（task_time + step_trace 联合）| 误差 < 5%，可接受 |
| 4 | β fit 仅用 21 数据点 | 极端 batch / 极长 S 外推可能偏 | 用户加新 model 后必须 [05](05_calibration.md) 重 fit |

完整局限清单：[06_assumptions_limits.md](06_assumptions_limits.md)

---

## 📚 参考

- Williams S, Waterman A, Patterson D. *Roofline: an insightful visual performance model for multicore architectures.* Communications of the ACM, 2009. 52(4): 65-76.
- 昇腾 CANN 8.5 msprof 工具文档（内部 + 公开版）
- 实测数据：`legacy/docs/overhead_decomposition_audit.md` v1.1（9 配置 PipeUtil + 1 真 decode）
- 校准源代码：`src/prism/roofline/predict.py` 函数 `predict_910b4_v2()`
