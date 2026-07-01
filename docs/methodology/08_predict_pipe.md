# 08 — PredictPipe：从 GEMM 规格预测 pipe 分解（无 msprof）

> **目标**：让 Tier 1 用户加新模型时**无需重新采集 msprof PipeUtilization**，
> 直接从模型 hyperparameters 预测出与 `pipe_baseline_per_model.json` 同 schema
> 的 pipe baseline，使 `prism-sweep` / `prism-ceiling` 可用。

---

## 1. 动机：能力不对称

PRISM v0.1.0 中三个 Tier 1 CLI 对新模型的可用性如下：

| 工具 | 新模型可用？ | 输入需求 |
|------|------------|---------|
| `prism-regime` | ✅ | 仅高层 YAML（`layers`、`ops_b1`、`bytes_total`）|
| `prism-ceiling` | ❌ | 依赖 `pipe_baseline_per_model.json`（msprof 实测）|
| `prism-sweep` | ❌ | 同上 |

要让 ceiling/sweep 也对新模型可用，必须能在**没有真机 msprof**的情况下，从模型规格推出 per-pipe baseline。这就是 PredictPipe。

---

## 2. 方法论：物理公式 + 实测校准的两层结构

PredictPipe 把 pipe baseline 拆为两层：

```
   pipe_baseline_per_model.json entry
   ────────────────────────────────────
   ┌── 第一层：纯物理公式（无校准），输入 = arch_dict + GEMM/Vector 规格
   │     aic_pipes_us:  mac / mte1 / mte2 / fixpipe / scalar
   │     aiv_pipes_us:  vec / mte2 / mte3 / scalar / idle
   │
   └── 第二层：interaction 拟合常数（K0、H_prefill、H_decode）
         kernel_gap_us = K0 × n_kernels
         host_gap_us   = H_prefill (或 H_decode，按 regime)
         wall_clock_us = max(aic_pipes) + sum(aiv_pipes) + kernel_gap + host_gap
```

**关键洞察**：物理层只需要 arch + model 信息（无需校准）；interaction 常数来自 msprof 实测一次拟合，**与具体新模型无关**，所以可以应用到任何新模型上。

---

## 3. 第一层：物理公式（src/prism/predict_pipe/physics.py）

七个公式，全部带单位（输出 μs），全部以 arch dict 参数化（不再硬编码 910B4）：

### 3.1 AIC（Cube 单元）侧

| Pipe | 公式 | 物理含义 |
|------|------|---------|
| **mac** | `Σ(2·M·N·K) / (cube_total_macs × clock_ghz) / η_compute` | Cube MAC throughput |
| **mte1** | `activation_read_bytes / l1_l0_bw_gbs` | L1↔L0 激活流量（**权重被 M-tile 复用**，仅算激活）|
| **mte2** | `(weight_overflow + activation_overflow) / hbm_bw_gbs` | HBM↔L1：权重溢出 + 激活溢出（L2 是 scratchpad 不是 cache）|
| **fixpipe** | `output_bytes / fixpipe_bw_gbs` | L0C→输出（v4 简化；实测多为 L0C→GM 直写，sweep 用 §3.6 双带宽 blend）|
| **scalar** | 0（不建模） | 控制开销 |

### 3.2 AIV（Vector 单元）侧

| Pipe | 公式 | 物理含义 |
|------|------|---------|
| **vec** | `vector_flops / (aiv_total_throughput × clock_ghz)` | SIMD ALU |
| **mte2** | `intermediate_bytes / ub_l1_bw_gbs` | UB↔L1 中间激活 |
| **mte3** | `output_bytes × 0.5 / ub_l1_bw_gbs` | UB→输出，由 MTE3 引擎承载（**非 FixPipe**，见下注）|
| **scalar / idle** | 0（不建模） | — |

> **Oracle 评审重点**（来自 v0.1 原型修订）：
> - `aic_mte1` 必须排除权重（M-tile 内复用 4-16×）
> - `aic_mte2` 必须算**全部**权重字节 + 激活溢出（不能假设 L2 持久缓存）
> - ~~`aiv_mte3` 走 `fixpipe_bw_gbs`~~ —— **2026-05-20 修正**：此条 v0.1 评审结论有误。
>   MTE3 是 UB-rooted 输出引擎，FixPipe 是 AIC 侧 L0C-rooted 输出单元，AIV 不经过
>   FixPipe。MTE3 的 store 实为 UB→GM 与 UB→L1 双目的地——见 §3.6 的实测拆分校准。
>   本节 v4 物理公式用 `ub_l1_bw_gbs` 是简化（仅 v4 旧路径使用；v5–v8 按经验
>   vec:mte2:mte3 = 1:7:5 拆分，不调用本公式）；`prism-sweep` 用 §3.6 的双带宽 blend。

### 3.3 已知物理层局限（更新：P1 后部分修正）

| 局限 | 影响 | 状态 |
|-----|------|-----|
| ~~Qwen3 大模型 `aic_mte2` 误差 78-81%~~ | 大模型 sweep ratio 偏移 | ✅ **已修复**：通过 §3.4 archetype amp |
| AIV 物理层漏建 idle / scalar / sync 开销 | aiv_time 50-15000× 偏低 | ✅ **已修复**：通过 §3.5 empirical multi-factor v3 模型 |
| `n_kernels` 简单 `(n_gemm+n_vec)×layers` 漏算 tile-level launches | kernel_gap 2.5-32× 偏低 | ✅ **已修复**：通过 §3.4 archetype × multiplier |
| Batch scaling 假设线性 | b>4 时可能漂移 | 暂用线性外推，标 medium confidence |
| 公式校准只有 5 个 measured 配置 + 2 个家族 | 外推到不同 archetype 不可靠 | 扩展到 ModernBERT/SmolLM2 msprof（Issue #3）|

### 3.4 经验 archetype 修正（P1，Issue #2）

v0.1 物理公式系统性漏算了 CANN tile-by-tile 执行细节。三个修正：

#### A. AIC pipe amplification

```python
# physics.py
def archetype_amplification(weight_mb_proxy, S):
    if S == 1:           return 0.85   # decode
    if weight_mb_proxy < 600:  return 1.15   # small model (BERT/GPT-2 class)
    return 5.5                                # large prefill (Qwen3 class)
```

校准数据（5 measured configs）：

| Config | weight_mb_proxy | observed ratio (meas/pred) | bucket |
|---|---|---|---|
| BERT-base-S128-b1 | 226 | 1.06 | small (1.15) |
| GPT-2-S512-b1 | 226 | 1.25 | small (1.15) |
| Qwen3-prefill-S256-b1 | 763 | 5.19 | large (5.5) |
| Qwen3-prefill-S512-b4 | 763 | 5.68 | large (5.5) |
| Qwen3-decode-Min4 | 763 | 0.82 | decode (0.85) |

`weight_mb_proxy = layers × (4·d_model² + 3·d_model·d_ff) × 2 / 1e6` — 不含 embed / LM-head。

#### B. AIV time 经验锚定 — v1（1.25× 常数，已弃用）

> ⚠️ **v1 已被 v2（§3.5）替代**。保留此节作为历史记录。

物理只算 vec/mte2/mte3，漏了 idle/scalar/sync。实测 `aiv_time / aic_time` 落在 [0.94, 1.57] 区间，均值 ≈ 1.25。

```python
# v1（已弃用）：单一经验常数
aiv_time = aic_time × 1.25
```

**v1 的失败模式**：HF-BERT 类小模型 AIV err 达 78%。1.25× 常数把"AIV 真正的瓶颈是 UB 数据搬运"和"小 kernel 有巨大 fixed overhead"这两个现象抹平了。

#### C. n_kernels archetype multiplier

`estimate_n_kernels` 漏算 tile-level launches：

```python
# model_spec.py
if weight_mb_proxy < 600:    multiplier = 2.5    # BERT/GPT-2
elif S == 1:                  multiplier = 4.7    # decode
else:                         multiplier = 28.0   # large prefill
```

#### D. 修正后预测精度（5 measured configs）

| Config | v0.1 wall_clock err | After P1 fix |
|---|---:|---:|
| BERT-base-S128-b1 | 11.8% | **3.6%** |
| GPT-2-S512-b1 | 12.8% | **1.5%** |
| Qwen3-prefill-S256-b1 | 76.7% | **2.0%** |
| Qwen3-prefill-S512-b4 | 87.2% | **9.1%** |
| Qwen3-decode-Min4-Skv128-b1 | 51.0% | **10.0%** |

→ 所有 wall_clock 误差 < 11%，Issue #2 P1 接受门槛 < 30% **远超达成**。
→ hard gate test：`tests/test_predict_pipe.py::test_p1_wall_clock_error_under_30pct_on_all_measured`

### 3.5 AIV continuous-amp 模型 v4 (Method B)（Issue #2 P2，替代 §3.4-B + 历代 archetype 模型）

> **历史**：v1 用 `aiv_time = 1.25 × aic_time`（MAE 38.3%）；v2 改为 6-archetype α 线性模型（LOO CV MAE 19%，OOS >51%）；v3 用 3-bucket archetype amplification（Training MAE 10.9%，GPT-2 outlier 32.4%）；**v4（本节）用 continuous `attn_frac` + `(w_proxy/1000)²` 替代离散 archetype，Training MAE 4.9%，所有 configs < 10%**。

**v3 为什么不够好**：3-bucket 离散划分（small / large / decode）把 GPT-2（`attn_frac=0.56`，需要 amp≈2.0）和 BERT（`attn_frac=0.24`，需要 amp≈0.9）塞进同一个 "small" 桶（两者 `w_proxy` 都是 226 MB），导致两个模型只能共用一个 amp 值——GPT-2 一定 outlier。根因不是公式结构，而是 archetype 分辨率不足。

**v4 关键洞察**：用一个连续特征 `attn_frac = O(S²) attention softmax bytes / total AIV data bytes` 替代离散 bucket，自然区分注意力主导（GPT-2 长序列）和 FFN 主导（BERT 短序列）的模型。

**v3 → v4 演化的根因分析**（详见 `docs/methodology/09_aiv_prediction_gap_analysis.md`）：

学术文献调研识别 6 个 root cause（v3 与 v4 共同基础）：
1. **Per-kernel 固定开销**被忽略（Verrocchio, JPDC 2023: init=40 ns，但 CANN 实际 5–40 μs/kernel）
2. **CANN repeat 参数**不可预测（Zhou et al., ASPLOS 2025: repeat=1 → 13.54% utilization）
3. **UB↔L1 小载荷带宽折减**
4. **HBM Interconnect Bus 竞争**（MTE2/MTE3 半双工共享）
5. **AIC/AIV 信号量同步空闲**
6. **非 GEMM 算子异构性**（LN/Softmax/GeLU 指令模式迥异）

v3 已经把这 6 个因子吸收到 `(n_vk × C_kernel + data_MB × C_data) × archetype_amp` 框架内（修复了 `compute_vector_ops()` 漏掉 O(S²) attention softmax 的 bug），但用**离散 archetype** 路由 amp 值——这是 v3 失败的根因。v4 保留同一个加法 cost driver 结构，仅把 archetype 替换为**连续函数**。

**v4 公式**（生产代码）：

```python
# predict.py: predict_aiv_v2()
# 两个 cost driver:
#   (1) n_vector_kernels × C_kernel  — per-kernel 固定开销
#   (2) data_MB × C_data            — 数据搬运与处理

# 关键修正 1: 加入 attention softmax 数据量 (O(S²))
attn_softmax_bytes = L * H * S * S * FP16 * 2 * batch
data_bytes = inter_bytes * batch + output_b * 0.5 * batch + attn_softmax_bytes
data_MB = data_bytes / 1e6

# 关键修正 2: continuous amplification (替代 v3 3-bucket archetype)
attn_frac = attn_softmax_bytes_per_batch / data_bytes_per_batch
w_proxy = weight_proxy_mb(layers, d_model, d_ff)  # MB

if S == 1:
    amp = amp_decode    # decode 与 prefill 物理行为差异太大，保留单独常数
else:
    # 连续函数: a0 截距 + a1 注意力分数 + a2 权重大小平方
    amp = max(0.1, a0 + a1 * attn_frac + a2 * (w_proxy / 1000)**2)

aiv_time = (n_vk * C_kernel + data_MB * C_data) * amp
```

**6 个拟合参数**（grid search，中心化的细网格 ~30,625 组合）：

| 参数 | 含义 | Grid 范围 | Best fit |
|---|---|---|---|
| `aiv_C_kernel_us` | per-kernel 固定成本 (μs) | [8, 12, 16, 22, 30] | **16.0** |
| `aiv_C_data_us` | 数据搬运成本 (μs/MB) | [1, 2, 3, 5, 8] | **3.0** |
| `aiv_amp_a0` | amp 截距 | [-0.5, -0.2, 0, 0.3, 0.6] | **-0.2** |
| `aiv_amp_a1` | attn_frac 系数 | [1, 2, 3, 4, 5, 6, 8] | **4.0** |
| `aiv_amp_a2` | (w_proxy/1000)² 系数 | [6, 9, 11, 14, 17, 20, 24] | **14.0** |
| `aiv_amp_decode` | decode 常数 | [0.5, 1.0, 1.5, 2.0, 3.0] | **1.5** |

**`(w_proxy/1000)²` 的诚实可解释性说明**：

平方项**是经验拟合**，不是 first-principles 推导。线性 `w_proxy/1000` 仅给出 3.4× 动态范围（Net-Trans 0.005 → Qwen3 0.763），不足以同时拟合 Qwen3（需要 amp≈10）和 BERT（amp≈1.5）；平方后扩到 11.4×（0.000025 → 0.582），联合拟合从 24% MAE 降到 4.9% MAE。

有一个可能但**未经验证**的物理直觉——tile 重取次数 ∝ n_tiles × L1-miss-rate，两者都随 weight_MB 线性增长，乘积超线性——但**未用 msprof tile-trace 数据验证**。该指数应视为 hyperparameter，未来加入 `weight_proxy ∈ [200, 600] MB` 区间的实测 config（如 ModernBERT、SmolLM2）后可能需要重新拟合。

**v4 拟合质量**（6 KNOWN_MODELS configs，全部 measured 实测）：

| Config | aiv_pred (μs) | aiv_meas (μs) | err | amp | attn_frac |
|---|---:|---:|---:|---:|---:|
| BERT-base-S128-b1 | 912 | 918 | **0.7%** | 1.49 | 0.242 |
| GPT-2-S512-b1 | 3,601 | 3,376 | 6.7% | 2.76 | 0.561 |
| Qwen3-prefill-S512-b4 | 127,006 | 129,458 | 1.9% | 10.33 | 0.593 |
| Qwen3-prefill-S256-b1 | 34,913 | 34,058 | 2.5% | 9.64 | 0.421 |
| Qwen3-decode-Min4 | 3,123 | 2,877 | 8.5% | 1.50 | 0.000 |
| Net-Transformer-S256 | 104 | 114 | **9.4%** | 1.36 | 0.390 |

→ **Training MAE = 4.9%，6/6 configs 全部 < 10%**，无 outlier。

**vs 历代精度对比**：

| 模型 | Train MAE | Max Err | 参数数 | archetype 分辨率 |
|---|---:|---:|---:|---|
| v1（1.25× 常数）| 38.3% | 78.7% | 1 | 无 |
| v2（6-archetype α）| 11.7% | 33.5% | 6 | 6 个离散 bucket |
| v3（3-archetype empirical）| 10.9% | 32.4% | 5 | 3 个离散 bucket |
| **v4（continuous amp Method B）**| **4.9%** | **9.4%** | **6** | 连续函数 |

**物理可解释性**：

- `C_kernel = 16 μs/kernel`：每个 Vector kernel 的 CANN runtime 开销（init + scalar dispatch + sync），比 Verrocchio 理论值（0.04 μs）高 400×。v3 拟合到 18 μs，v4 拟合到 16 μs——基本一致
- `C_data = 3 μs/MB`：UB↔L1↔HBM 数据搬运成本（v3 拟合 5 μs/MB；v4 更小是因为 amp 项现在显式吸收了大模型 amplification，cost driver 不再 double-count）
- `amp` 物理直觉：`a0=-0.2` 是 baseline 截距；`a1=4.0` 系数表示 attention-dominated 工作负载 amp 高（GPT-2 `attn_frac=0.56` → 该项贡献 2.24）；`a2=14.0` 与 `(w_proxy/1000)²` 相乘，大模型 prefill 该项主导（Qwen3 763 MB → 该项贡献 8.15）
- `amp_decode = 1.5`：decode 模式 `attn_frac=0`（仅 S=1，无 O(S²) attention），需要单独常数

**已知局限**：

- **训练集仍只有 6 个 measured configs**：MAE 4.9% 是 in-sample，不是 OOS。LOO CV 待 ModernBERT/SmolLM2 真机 msprof 数据落地后做（Issue #3）
- **`(w_proxy/1000)²` 指数是经验值**：`weight_proxy ∈ [200, 600] MB` 区间无 measured config，外推风险已在 confidence label 中标 medium
- **Batch scaling**：仅 b=1/4 验证；b>4 线性外推
- **encoder + GLU FFN（ModernBERT 类）OOS**：训练集无 encoder + GLU 组合，置信度标 low

**输出 schema 影响**：`spec_summary` 字段变更：
- `aiv_model = "continuous_amp_v4"`（替代旧 `"empirical_v3"`）
- 新增 `aiv_attn_frac`、`aiv_amp_computed`
- 移除 `aiv_archetype_amp`（语义已不再是 bucket label）

**测试**：
- `test_aiv_multifactor_differentiates_small_vs_large` — Qwen3 AIV > 5× Net-Transformer
- `test_aiv_multifactor_hfbert_regression` — HF-BERT AIV > 50 μs，检查 v4 schema 字段存在（`aiv_model="continuous_amp_v4"`、`aiv_amp_computed`、`aiv_attn_frac`）
- `test_p1_wall_clock_error_under_30pct_on_all_measured` — 6 configs wall_clock err < 30%

### 3.6 aic_fixpipe / aiv_mte3 目的地带宽校准（Issue #7，2026-05-21）

`aic_fixpipe`（Cube L0C→输出）与 `aiv_mte3`（Vector UB→输出）都把计算结果搬出，
瓶颈带宽**取决于目的地**：

| pipe | 片上目的地（on-chip ref）| 片外目的地 |
|---|---|---|
| `aic_fixpipe` | L0C→L1/UB（`fixpipe_bw` ~4096）| **L0C→GM 直写**（`hbm_bw` ~392）|
| `aiv_mte3` | UB→L1（`ub_l1_bw` ~2048，`copy_ubuf_to_cbuf`）| **UB→GM**（`hbm_bw`，`copy_ubuf_to_gm`）|

**校准方法**（`scripts/calib_fixpipe_mte3_bw.py` → `data/calibration/pipe_dest_bw.json`）：
msprof 只报聚合 `*_time`。用 **prior-based 2-cluster 分类**：逐 op 算 implied 带宽
`bytes/time`，按物理先验阈值 `sqrt(hbm·onchip)`（两区域几何中点）切分到 GM/片上簇；
`gm_frac` = GM 簇的**字节占比**。每簇再做 OLS 取斜率作为 sanity。

此方法对所有模型普世：真双峰 config 干净分簇；单峰 config 自然退化为 `1cluster`。
回避了两种朴素方法的失效——`Σ字节/Σ时间`（被 per-op 固定开销污染）与单 pooled OLS
（双峰数据给 leverage-weighted 混合斜率，反解 `gm_frac` 有偏）。

**实测结论**（39 config）：

| pipe | 模型类 | GM 簇 OLS 斜率 (GB/s) | gm_frac | 备注 |
|---|---|---:|---:|---|
| `aic_fixpipe` | 长上下文大 prefill（Qwen3-S256/S512/S4096 sdpa）| 240–480 | **0.83–0.98** | 主要 L0C→GM 直写 |
| `aic_fixpipe` | ModernBERT/Llama/Phi-3 sdpa | 318–797 | **1.00**（1cluster）| GM 簇主导 |
| `aic_fixpipe` | Qwen2.5/SmolLM2 大 prefill | 248–662 | 0.28–0.70 | 部分 L0C→片上 |
| `aic_fixpipe` | HF-BERT/小模型 | 500–820 | 0.78–1.00 | 单峰或弱双峰 |
| `aiv_mte3` | 大 prefill | 446–551 | **0.50–0.85** | 主要 UB→GM |
| `aiv_mte3` | 小短序列模型 | — | ≤ 0.15 | 留片上 |

**关键发现**：`aic_fixpipe` 在几乎所有大 prefill config 上 GM 簇斜率落在 HBM 量级
（远低于片上 `fixpipe_bw` 4096）—— FixPipe 输出主要是 GM 直写。Qwen3-prefill-S4096
`gm_frac=0.98`。

`prism-sweep` 的 `scale_aic_pipes` / `scale_aiv_pipes` 据 `gm_frac` 把这两条 pipe 在
`hbm_bw` 与片上带宽之间 blend 缩放（每个 arch variant 重算）。**影响**：`hbm_bw` 维度
sweep 中大 prefill 模型 ratio 显著上升修正；`fixpipe_bw` 维度对 Qwen3-S4096 的
ratio 从 1.24 修正到 **1.00**（FixPipe 单元带宽几乎不是杠杆，HBM 才是）。
详见主报告 §6.4.1 修正说明。

---

## 4. 第二层：interaction 常数拟合（src/prism/predict_pipe/fit.py）

两个常数来自 9 个已知 msprof config 的 OLS 拟合：

### 4.1 `kernel_gap = K0 × n_kernels`

OLS through origin。`n_kernels` 由 `estimate_n_kernels(spec)` 估出（每 GEMM + 每 vector op = 1 kernel）。

| 拟合值 | 训练 MAE |
|-------|---------|
| `K0 = 1.86 μs/kernel` | 14.3% |

LOO CV（按模型族留一）显示 CV 误差 2.5% - 23%。

### 4.2 `host_gap` 按 regime 取常数

实测数据显示 host_gap 在 prefill 内**几乎与模型/序列长度/核数无关**——它代表 CANN runtime 的稳态 host 调度开销。

| Regime | 拟合值 | 训练 MAE |
|--------|-------|---------|
| prefill | `H_prefill = 13,424 μs` | 8.4% |
| decode (S=1) | `H_decode = 204 μs` | 0% |

> **判定规则**：config 名含 "decode" 或 "Min" → decode；否则 prefill。

### 4.3 重要事实：interaction 常数与具体新模型无关

K0、H_prefill、H_decode 都是**CANN runtime × NPU 调度路径**的特征，**与具体新模型架构无关**（在相同 arch baseline 下）。因此一次拟合可重复用于任何新模型预测。

这就是 PredictPipe 能"无需 msprof 加新模型"的根本原因。

---

## 5. 置信度判定（src/prism/predict_pipe/predict.py:assign_confidence）

预测结果带置信度标签，告诉用户哪些预测可信、哪些是外推：

| 条件 | 标签 | 解释 |
|------|------|-----|
| `arch=encoder` | **low** | 训练集仅 BERT-S128，encoder 长序列 / 不同 FFN type 都是外推 |
| `ffn_type=glu` | **low** | GLU FFN 不在训练集中，`aic_mte2` 行为未测 |
| `S=1`（decode） | **medium** | 1 个 decode config 在训练集，`host_gap` 已确定 |
| `arch=decoder`, `S ∈ [256, 4096]`, FFN standard/swiglu | **high** | GPT-2 + Qwen3 充分覆盖 |
| `S=128` 或 `S=8192` | **medium** | 边界长度，训练覆盖薄 |
| 其他 | **low** | OOD |

> **使用规范**：`confidence=low` 的预测不应直接驱动芯片投资决策；建议先跑真机 msprof 验证。

---

## 6. 输出 schema（兼容 `pipe_baseline_per_model.json`）

预测结果的 JSON 与 msprof 实测 entry **逐字段兼容**，因此能直接喂入下游：

```json
{
  "baseline_arch_name": "ascend_910b4_for_sweep_v2",
  "configs": {
    "ModernBERT-base-S4096-b1": {
      "n_kernels_per_inf": 242,
      "aic_pipes_us": {"mac": 1287, "mte1": 421, "mte2": 51708, "fixpipe": 47, "scalar": 0},
      "aiv_pipes_us": {"vec": 23, "mte2": 338, "mte3": 24, "scalar": 0, "idle": 0},
      "wall_clock_us": 65920,
      "kernel_gap_us": 449,
      "host_gap_us": 13424,
      "aic_dominant_pipe": "mte2",
      "source": "predict_pipe_v2",
      "predicted": true,
      "confidence": "low (encoder: only BERT-S128 in training set)",
      "spec_summary": {
        "arch": "encoder", "layers": 22, "S": 4096, ...,
        "aiv_model": "continuous_amp_v4",
        "aiv_attn_frac": 0.92, "aiv_amp_computed": 4.43, ...
      }
    }
  }
}
```

新增字段 `predicted: true`、`confidence`、`spec_summary` 是**增量**——下游消费方（sweep、ceiling）即使不识别这些字段也能正常工作。

---

## 7. 使用入口

```bash
# 1. 拟合常数（一次性，只在新增 msprof 数据后才需要重跑）
prism-predict-pipe --refit-params

# 2. 预测新模型
prism-predict-pipe \
    --model models/regime/modernbert_base_prefill_S4096.yaml \
    --arch  arch/ascend_910b4_for_sweep_v2.yaml \
    --output data/calibration/predict_pipe_modernbert.json

# 3. 喂入 sweep（或合并入主 baseline）
prism-sweep --pipe-baseline data/calibration/predict_pipe_modernbert.json
# 或：
prism-predict-pipe ... --merge-into data/calibration/pipe_baseline_per_model.json
```

完整教程见 [`tutorials/05_predict_new_model.md`](../tutorials/05_predict_new_model.md)。

---

## 8. 与现有 PRISM 工具的关系

```
┌──────────────────────────────────────────────────────────────────┐
│                  Tier 1 工作流（新模型加入路径）                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   HF config → models/regime/X.yaml (with gemm_spec)              │
│        │                                                          │
│        ├──→ prism-regime ──→ regime 分类（host/compute/memory）    │
│        │                                                          │
│        └──→ prism-predict-pipe ──→ predict_pipe_X.json            │
│                                       │                           │
│                                       ├──→ prism-sweep            │
│                                       └──→ prism-ceiling          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**与 `prism-fit` 区别**：`prism-fit` 拟合 η_real(M,N,K,B)（Cube 利用率），需要 msprof
ArithmeticUtilization 数据；`prism-predict-pipe` 拟合 K0/H 这两个**与模型架构无关**的常数，
拟合一次后任何新模型都能直接预测，**不再需要 msprof**。

---

## 9. 不变量与门槛

| 项 | 期望 | 来源 |
|---|---|---|
| `fit_all_and_save` reproduce K0≈1.86, H_prefill≈13424, H_decode≈204 | training MAE < 12% / 20% | v0.1 原型 + `test_fit_reproduces_v01_prototype_constants` |
| 预测输出 schema 全字段覆盖 baseline entry | `set(ref) ⊆ set(predicted)` | `test_predict_output_schema_compatible_with_pipe_baseline` |
| 置信度标签按 archetype 正确路由 | encoder→low / GLU→low / decode→medium / prefill→high | `test_confidence_label_routing` |
| CLI e2e 输出 ModernBERT wall_clock 在 50 ms - 1 s 区间 | v4 with high attn_frac@S=4096 produces higher prediction than v3 | `test_cli_e2e_modernbert` |
| AIV Training MAE | < 10% on 6 measured configs | `predict_pipe_params.json::aiv_training.mae_pct` |
| AIV per-config max err | < 10% on all measured configs | manual check via `--refit-params` output |

---

## 10. 已知盲区（拆 Issue #3 / #4 / #5）

| 项 | 状态 | 计划 |
|---|---|---|
| Qwen3 大模型 `aic_mte2` 78-81% 误差 | 不在本版本范围 | Issue #3，等 ModernBERT msprof 落地后重新拟合 |
| 8 个新模型 YAML 扩展验证（bge / deberta / llama-3.2 / qwen2.5 等）| 不在本版本范围 | Issue #4，本工具落地后批量跑 |
| 910B 真机实测验证 | 不在本版本范围 | Issue #5，由 Windows 端用 `benchmark/run_new_models_msprof.sh` 触发 |
| `aic_bubble` / `aiv_idle` 通用拟合 | 不在本版本范围 | 需 ≥3 新模型族 msprof 数据 |
| Batch scaling > 4 验证 | 不在本版本范围 | 同上 |

---

## 12. v5：bounded extrapolation + 显式 train/val 拆分（Issue #2 v5）

### 12.1 触发条件

v4 在 P3 真机验证中暴露**灾难性外推失败**：

| OOS Config | w_proxy MB | v4 wall_err |
|---|---:|---:|
| ModernBERT-S4096 | 220 | −1.8% ✓ |
| Qwen2.5-0.5B-S2048 | 782 | +155% |
| SmolLM2-360M-S2048 | 708 | +137% |
| **Llama-3.2-1B-S2048** | **2147** | **+1156%** ❌❌ |

根因：v4 拟合 `aiv_amp = a0 + a1·attn + a2·(w/1000)²` 用 6 configs（全部 w_proxy ∈ [89, 230] MB）。平方项 in-range 拟合得好，但 Llama w=2147 时 a2·(2.147)² = 64.5 把 amp 推到 67.7×。

### 12.2 设计：三个公式替换

| 部件 | v4 | v5 |
|---|---|---|
| AIC archetype amp | 3-bucket `{1.15, 5.5, 14.16}` | 线性 cap `min(amp_max, 1+α·w/1000)` |
| AIV continuous amp | `(w/1000)²` 平方爆炸 | sigmoid `w/(W_sat+w)` 数学饱和 |
| n_kernels archetype mult | 3-bucket `{2.5, 4.7, 28}` | 饱和 `base+(max-base)·(1-exp(-w/W_sat))` |

**三个公式都从"多项式/离散桶"换成"数学有界的形式"**。`min(amp_max, ...)` 和 sigmoid 让 w_proxy → ∞ 时 amp **永远不会爆炸**，独立于 fit 选择。

### 12.3 显式 train/val 拆分（`src/prism/predict_pipe/splits.py`）

```python
TRAIN     (6 configs)  w_proxy ∈ [89, 763] MB     # b=1 + Qwen3-S512-b4
VAL_batch (3 configs)  Qwen3-prefill batch>1
VAL_size  (4 configs)  Qwen3-S4096, Llama, Qwen2.5, SmolLM2-360M  # ALL w>700 MB
```

`fit_v5.py` 用 `scipy.optimize.differential_evolution`，**objective 只看 TRAIN MAE**，永远不偷看 VAL。

### 12.4 结果

| | TRAIN MAE | VAL_size MAE | Llama wall_err |
|---|---:|---:|---:|
| v4 baseline | 4.9% | ~511% | **+1156%** |
| **v5** | 17.3% | **103.7%** | **+232%** |

**Llama 5× 改进**，VAL_size 5× 改进。TRAIN 接受 12.4 pp 退化换泛化。

### 12.5 残留：单一公式无法分桶

v5 用单一连续公式拟合所有 prefill 模型，**无法同时满足 Qwen3-prefill（amp ~14）和 Llama（amp ~2）在相似 w_proxy 下的差异**。这驱动 v6。

---

## 13. v6：按瓶颈分桶（Issue #2 v6，响应用户 2026-05-17 洞察）

### 13.1 用户洞察 + 数据验证

> "Qwen3-prefill family 是典型的 gemm 占绝对优势，CUBE（AIC）是性能瓶颈，与其他几类不同性能瓶颈的模型是不是应该放在不同的 bucket 进行分类和校准?"

用 13 个 measured configs 的 AIV/AIC 比率验证：

| AIV/AIC 区间 | n configs | Models |
|---|---:|---|
| **< 1.2 (AIC-bound)** | 6 | Qwen3-prefill batch>1, long-S, decode |
| 1.2-2.5 (Balanced) | 3 | BERT, GPT-2, Qwen3-b1 |
| **> 2.5 (AIV-bound)** | 4 | Llama, Qwen2.5, SmolLM2, ModernBERT |

三段分布**清晰可分**，验证假设。

### 13.2 设计：4 桶 + per-bucket fit

`physics_v6.py::classify_bottleneck` 用 spec 启发式分类：

```
spec.S == 1                                              → AIC_DECODE
layers ≥ 24 + d_model ∈ [1000,1300] + swiglu + decoder   → AIC_QWEN3
d_model ≥ 700 + S × batch ≥ 1024                         → AIV_BOUND
default                                                   → BALANCED
```

每桶 3 free params：`amp_aic`, `amp_aiv`, `nk_mult`。`fit_v6.py` 用 DE **独立拟合每桶在自己 TRAIN subset**（无 cross-bucket 泄漏）。

### 13.3 per-bucket fitted coefficients (v6 v1)

| Bucket | amp_aic | amp_aiv | nk_mult |
|---|---:|---:|---:|
| AIC_DECODE | 0.71 | 1.77 | 3.14 |
| **AIC_QWEN3** | **10.59** | 1.16 | **31.32** |
| AIV_BOUND | 2.84 | 2.82 | 4.37 |
| BALANCED | 1.00 | 1.00 | 6.37 |

`AIC_QWEN3` 的 amp_aic=10.6 + nk_mult=31 完美捕获了 Qwen3 prefill "258 kernels/layer + 低 Cube 利用率" 特征，**不污染其他桶**——这是 v5 单一连续公式做不到的。

### 13.4 结果

| | TRAIN MAE | VAL_batch MAE | VAL_size MAE | Llama wall_err |
|---|---:|---:|---:|---:|
| v4 | 4.9% | – | ~511% | **+1156%** |
| v5 | 17.3% | 40.6% | 103.7% | **+232%** |
| **v6** | **0.23%** | **15.8%** | **28.9%** | **+26.6%** |

### 13.5 LOMO（leave-one-model-out）验证（13 configs）

`lomo_v6.py`：每个 measured config 单独留出 + 重拟合所在 bucket → predict held-out → 报告 err。

| Bucket | n | LOMO MAE | max |
|---|---:|---:|---:|
| BALANCED | 2 | **2.9%** | 3.1% |
| AIC_DECODE | 1 | 8.6% | 8.6% |
| AIV_BOUND | 4 | **14.9%** | 18.8% (Llama) |
| AIC_QWEN3 | 6 | 21.2% | 52.3% (S=4096) |
| **Overall** | 13 | **15.45%** | – |

**Llama LOMO err 18.8%** 确认 cross-architecture generalization——AIV_BOUND 桶在 ModernBERT (encoder/GLU) 上拟合的参数泛化到 Llama (decoder/swiglu/S=2048) 仍 < 20%。

### 13.6 confidence labels（v6 aware）

`assign_confidence(spec, batch, params)` 根据 `params["v_model"]` 分发：

| Bucket | Confidence | 依据 |
|---|---|---|
| AIV_BOUND | **high** | 4 OOS configs 验证 12-26% err |
| AIC_QWEN3 (S ≤ 2048) | **high** | Qwen3 multi-batch/multi-S 验证 5-28% err |
| AIC_QWEN3 (S > 2048) | medium | within-bucket S 外推（LOMO err 52% 最高）|
| AIC_DECODE | medium | 1 训练 anchor |
| BALANCED | medium | 2 训练 anchors |

v4/v5 path（不传 `v_model=v6`）保留 legacy 标签。

### 13.7 关键洞察：generalization 在 bucket 边界

TRAIN MAE 0.23% 看似 over-fit（per-bucket 1-2 anchors 拟合 3 params），但 LOMO 15.45% 证明：**bucket 分类器是 generalization 的载体，per-bucket 参数只是 calibration**。AIV_BOUND 桶上 ModernBERT (encoder/GLU/S=4096) 拟合的 amp 直接 transfer 到 Llama (decoder/swiglu/S=2048) 给出 18.8% err——这种 cross-architecture transfer 是 v4/v5 单一公式做不到的。

### 13.8 工件清单

- `src/prism/predict_pipe/physics_v5.py` + `physics_v6.py` — 新公式
- `src/prism/predict_pipe/fit_v5.py` + `fit_v6.py` + `lomo_v6.py` — 拟合 + LOMO
- `src/prism/predict_pipe/splits.py` — TRAIN / VAL_batch / VAL_size 定义
- `src/prism/predict_pipe/predict.py` — `v_model` dispatch (v4 / v5 / v6 共存)
- `data/calibration/predict_pipe_params_v6.json` + `predict_pipe_v6_lomo.json`
- `tests/test_predict_pipe.py::test_v6_oos_llama_under_50pct` + `test_v6_bucket_classification` 硬门禁

### 13.9 v6.1 之后的演进路线（参见 §14 / §15）

v6 / v6.1 是 **eager attention baseline** 上的最佳 calibration，但 production 用 SDPA。这驱动了 v7 → v8 演进：

- **v7（§14）**：SDPA-aware 3-bucket，删 AIC_QWEN3（SDPA 让 Qwen3-prefill 不再是 outlier）
- **v8（§15）**：multi-objective fit — 解决 v6 component error cancellation 问题

→ **新用户首选 v8**（生产路径 SDPA + component-honest fit）。v6 保留作历史 baseline。

---

## 14. v7：SDPA-aware calibration（Issue #2 v7）

### 14.1 触发：v6 的隐藏假设是错的

v6 calibration baseline 是 `attn_implementation="eager"`（transformers 默认）—— 但 production 部署（ais_bench 默认 + ATC 编译）走 SDPA path。两者性能特征差异：

| Config | eager wall | SDPA wall | speedup | AIV/AIC ratio (eager → SDPA) |
|---|---:|---:|---:|---|
| Qwen3-prefill-S4096-b1 | 3050 ms | **453 ms** | **6.74×** | 1.19 → 3.48 |
| Qwen3-prefill-S256-b1 | 78 ms | 14 ms | 5.5× | 1.57 → 4.2 |
| ModernBERT-S4096 | 323 ms | 261 ms | 1.24× | 4.54 → 5.6 |
| Llama-3.2-1B-S2048 | 197 ms | 173 ms | 1.14× | 2.66 → 3.5 |

**关键**：Qwen3-prefill 在 eager 下被错归为 AIC-bound（v6 的 AIC_QWEN3 桶），实际在 SDPA 下是 AIV-bound。**整个 AIC_QWEN3 桶是 eager attention 的 dispatch artifact，不是 Qwen3 架构属性**。

### 14.2 v7 设计：3 桶（删 AIC_QWEN3）

`physics_v7.py::classify_bottleneck_v7` 简化分类：

```
spec.S == 1                                      → AIC_DECODE
spec.d_model >= 700 AND S × batch >= 1024         → AIV_BOUND
default                                           → BALANCED
```

每桶 3 free params（amp_aic, amp_aiv, nk_mult），无 amp_aic_S_alpha（v6.1 用于救 Qwen3-S4096 的 hack 不再需要）。

### 14.3 双 baseline 保留政策（用户 2026-05-18 mandate）

baseline JSON 同时保留 eager + SDPA：

```
Qwen3-prefill-S4096-b1        (eager, 3050 ms)   ← v6 calibration
Qwen3-prefill-S4096-b1-sdpa   (SDPA,  453 ms)    ← v7 calibration
```

未来不允许"清理冗余"——这是用户明确 mandate。`v_model` dispatch 让 v4/v5/v6/v7 四 path 并存。

### 14.4 v7 拟合结果（SDPA training）

```
TRAIN MAE = 11.28%  (7 configs, mostly Qwen3-sdpa + eager AIV_BOUND anchor)
VAL_size MAE = 15.69%  (3 eager OOS: Llama/Qwen2.5/SmolLM2)
LOMO MAE = 17.83%  (13 configs leave-one-out)

Per-bucket coefficients:
  AIC_DECODE : amp_aic=0.71  amp_aiv=1.77  nk_mult=3.14
  AIV_BOUND  : amp_aic=2.50  amp_aiv=3.07  nk_mult=11.98
  BALANCED   : amp_aic=0.66  amp_aiv=1.05  nk_mult=7.10
```

### 14.5 v7 残留 component 抵消（驱动 v8）

v7 TRAIN component breakdown 暴露问题：

| Metric | v7 TRAIN | v7 OOS |
|---|---:|---:|
| AIC MAE | **49.7%** | **129.7%** |
| AIV MAE | 47.5% | 25.1% |
| wall MAE | 18.0% | 11.8% |
| cancellation ratio | 2.8 | **11.0** ❌ |

OOS AIC 129.7% 被 AIV under-prediction 抵消才让 wall 11.8% — 是 fit objective 只看 wall 的问题。v8 解决。

---

## 15. v8：multi-objective fit（用户 2026-05-18 mandate）

### 15.1 触发：用户两次明确要求

1. **2026-05-15**："用验证集避免过拟合" → v5/v6 显式 train/val splits（已做）
2. **2026-05-18**："用 msprof 精确校准 AIC / AIV 等主要模块延迟，避免只校准最终延迟让互相抵消干扰拟合" → **v8 multi-objective fit**

### 15.2 公式：复合 loss

```python
# fit_v8.py::_multi_objective_loss
loss = wall_mae + 0.3 × aic_mae + 0.3 × aiv_mae + 0.2 × nk_mae
```

DE optimizer 现在被迫**同时**最小化 4 个 component 的误差，无法再用 `(amp_aic, amp_aiv)` 互相抵消的 trick。msprof 实测 `aic_time_us` / `aiv_time_us` / `n_kernels_per_inf` 直接进 loss。

### 15.3 v8 v.s. v6/v7（5 个 split 完整对比）

| Metric | v6 TRAIN | v7 TRAIN | **v8 TRAIN** |
|---|---:|---:|---:|
| AIC | 47.7% | 49.7% | **43.7%** |
| AIV | 45.7% | 47.5% | **25.6%** |
| n_kern | 67.4% | 94.8% | **37.6%** |
| wall | 0.2% | 18.0% | 20.9% |
| **cancel ratio** | **204** ❌ | 2.8 | **2.1** ✓ |

| Metric | v6 OOS | v7 OOS | **v8 OOS (eager)** |
|---|---:|---:|---:|
| AIC | 56.9% | 129.7% | **7.0%** |
| AIV | 6.4% | 25.1% | **8.0%** |
| n_kern | 11.4% | 95.9% | **2.8%** |
| wall | 10.1% | 11.8% | **8.4%** |
| cancel ratio | 5.6 | 11.0 | **1.0** ✓✓ |

| Metric | **v8 SDPA-OOS** (Phase 3，新双 OOS) |
|---|---:|
| AIC | 17.9% |
| AIV | 33.5% |
| n_kern | 1.6% |
| wall | **13.7%** |
| cancel ratio | 2.44 ✓ |

→ **v8 OOS 全 component < 10%**（eager）/ wall < 25%（SDPA double OOS）。Cancellation ratio 1.0-2.4 vs v6 的 204 ❌。

### 15.4 v8 per-bucket fitted theta

| Bucket | amp_aic | amp_aiv | nk_mult |
|---|---:|---:|---:|
| AIC_DECODE | 0.82 | 1.38 | 4.08 |
| AIV_BOUND | **1.04** | 4.43 | 5.95 |
| BALANCED | 1.06 | 1.50 | 3.00 |

注意：所有 amp_aic 都接近 1.0（"physics base 正确"）。v8 不再依赖大幅扭曲来对冲——AIV_BOUND amp_aiv=4.4 反映 AIV 实际需要 ~4× scaling（合理物理意义，不是 fit artifact）。

### 15.5 trade-off：v6 vs v8 选择

| 场景 | 推荐 |
|---|---|
| **新模型 wall_clock 预测**（用户主 use case）| **v8** — OOS 全 component < 10% |
| **架构 sweep**（改 HBM BW / cm scale / L2）| **v8** — AIC sub-pipe 比例诚实 |
| **Bottleneck 诊断** | **v8** — sub-pipe 不被 amp 扭曲 |
| 仅需 in-distribution wall + 模型在 TRAIN 集 | v6 (TRAIN wall 0.2% 完美但虚) |
| Issue #2 legacy 对比 | v4/v5/v6/v7 全保留 |

### 15.6 v8 工件

- `src/prism/predict_pipe/physics_v7.py` + `fit_v8.py` — v8 复用 v7 schema，只换 fit objective
- `data/calibration/predict_pipe_params_v8.json` — fitted multi-objective coefficients
- `tests/test_predict_pipe.py::test_v8_oos_all_components_under_30pct` — 硬门禁
- `tests/test_predict_pipe.py::test_v8_train_no_component_cancellation` — cancellation_ratio < 5 门禁
- `tests/test_predict_pipe.py::test_v8_sdpa_oos_under_30pct_wall` — Phase 3 SDPA OOS 门禁
- `docs/findings/predict_pipe_component_cancellation_audit.md` — 完整 cancellation 分析
- `docs/findings/predict_pipe_phase3_sdpa_oos.md` — SDPA double-OOS 验证

### 15.7 v9 候选

- SDPA OOS AIV MAE 33.5% — 加 SDPA-OOS 进 TRAIN，或加 SDPA-specific amp_aiv 项
- TRAIN AIC 43.7% / nk 37.6% 仍偏高 — 调 λ 权重（提升 λ_aic）
- 跨 family 实测（Gemma/Mistral/Phi）— 见 Issue #5

---

## 11. 引用与来源

- 原型实现：`.sisyphus/predict_pipe_v0.1.py`（Windows 端 opencode reviewer，2026-05-11）
- Oracle 评审：见 issue #2 描述节 §2.3
- 拟合数据集：`data/calibration/pipe_baseline_per_model.json` 中 9 个 `source=msprof_PipeUtilization_measured` 配置（其中 6 个含 `aiv_time_us > 0` 用于 AIV 拟合）
- AIV v4 (Method B) 根因分析：`docs/methodology/09_aiv_prediction_gap_analysis.md`
- 修正公式依据：DaVinci microarchitecture HC31 (2019) + CANN 8.5 doc + FastAttention paper (Liu et al., 2024) + Verrocchio (Tang & Wang, JPDC 2023) + Component Roofline (Zhou et al., ASPLOS 2025)
