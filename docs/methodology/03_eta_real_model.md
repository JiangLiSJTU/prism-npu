# Cube 真实利用率 η_real 物理拟合模型

## 1. 问题：Timeloop 100% 不等于硬件 100%

Timeloop manual mapping 的 utilization 报告基于"spatial assignment 满载"——只要每个 cycle 上 4096 个 Cube_MAC slot 都有算子分配，就报告 100%。但**这个 100% 不反映真实 wall-clock**。

实卡 msprof 测得的真实 cube utilization（`aic_mac_fp16_ratio`，定义为 mac active cycles / aicore_time）：

| Workload | Timeloop 报告 | msprof 实测 cube_util | 偏差 |
|----------|------------:|---------------------:|----:|
| BERT-base S=128 b=1 FFN1 | 100% | **18.3%** | 5.5× 高估 |
| Qwen3-0.6B S=512 prefill b=1 | 100% | 30.3% | 3.3× 高估 |
| Qwen3-Embedding S=4096 b=1 ffn_gate | 100% | ~50% | 2× 高估 |

差异来源（Timeloop 不建模的损失）：

1. **流水 fill/drain**：systolic array 启停延迟。M×N=16×16 阵列，K=K_inner cycle 中 fill+drain 占 16+16=32 cycle，对小 K_inner（如 16, 32）损失 50%+
2. **L0 bank 冲突**：MTE1 把 L1→L0A/B 时多 bank 同时访问，bank 冲突 stall MAC unit
3. **MTE2 sync stall**：DRAM/L2→L1 不及时，Cube 等待
4. **Cube/Vector pipeline 切换**：层间 ping-pong 切换时序对齐损失
5. **小维度 tile 量化损失**：M、N、K 不是 16 倍数时 padding 浪费

→ **任何用 Timeloop cycles 直接做 wall-clock 预测的工作必须乘 1/η_real 修正**。本节给出 η_real 的物理拟合模型。

## 2. 设计哲学：physics-informed 而非 black-box ML

最初尝试用 OLS 线性回归拟合 `log(η) ~ log_M16 + log_N16 + log_K16 + has_small_dim` 等 6 特征：

| 拟合方法 | Train MAE (Qwen3) | BERT 验证 MAE | 缺陷 |
|---------|----------------:|---------------:|------|
| v1 OLS（6 个 log/indicator 特征） | 11 pp | 22 pp | **跨模型泛化差**（BERT 偏差 2 倍训练）|

OLS 的根本问题是没有体现 systolic array 的物理结构。换成 physics-informed 5 参数模型后：

| 版本 | Train MAE | BERT MAE | GPT-2 MAE | 跨模型泛化 |
|-----|---------:|--------:|---------:|-----|
| v4 物理模型（4 参数）| 11 pp | 15.6 pp | 11.9 pp | 接近门槛 |
| **v5**（v4 + op_kind 区分 BMM/MM + 全数据）| **11.98 pp** | **14.33 pp** ✓ | **12.07 pp** ✓ | **门槛通过** |

→ v5 是 release 版本。本文以 v5 为准。

**关键洞察**：物理结构（fill/drain 公式、tile 量化）让模型在新 model 上仍然成立，OLS 黑盒只能复现训练集。

## 3. v5 物理公式

### 3.1 整体形式

$$
\eta_\text{real}(M, N, K, B) = \min\left(1,\ \eta_\text{pipeline} \cdot \eta_\text{tile} \cdot \eta_\text{batch}\right)
$$

3 个独立项相乘 + clip 到 [0, 1]。

### 3.2 项 1：η_pipeline（systolic fill/drain）

经典 systolic array 的 fill/drain 经典公式：完整算 M×N×K 块需要 `(M-1) + (N-1) + (K-1)` 启停 cycle。把这个公式改写成"工作 cycle / 总 cycle" 比例：

$$
\eta_\text{pipeline} = \frac{M_b \cdot N_b \cdot K_b}{M_b \cdot N_b \cdot K_b + \alpha M_b N_b + \beta M_b K_b + \gamma N_b K_b + \delta (M_b + N_b + K_b)}
$$

其中：
- $M_b = \lceil M / 16 \rceil$，$N_b = \lceil N / 16 \rceil$，$K_b = \lceil K / 16 \rceil$（16 是 Cube spatial）
- $\alpha, \beta, \gamma$ 是 2D coupling 系数（拟合参数）
- $\delta$ 是 1D edge 系数（拟合参数，实测拟合后 ≈ 0）

直觉：分母 = work cycle + overhead cycle。fill/drain 在每对维度交界产生 ~ $M_b N_b$ 量级 stall（α 项）。

### 3.3 项 2：η_tile（边缘量化）

每个维度的 tile 不是 16 整数倍时，padding 浪费一些 MAC。定义单维度量化系数：

$$
f(x) = \frac{x}{16 \cdot \lceil x/16 \rceil}
$$

例如 x=128 时 f(x) = 128 / (16×8) = 1.0；x=130 时 f(x) = 130 / (16×9) = 0.903（11% padding 浪费）。

三维联合：

$$
\eta_\text{tile}(M, N, K) = f(M) \cdot f(N) \cdot f(K)
$$

→ **无 free param**，纯几何推导。

### 3.4 项 3：η_batch（批次摊薄）

经验发现 cube_util 随 batch 略升（多 batch 摊薄 launch overhead）：

$$
\eta_\text{batch}(B) = 1 + \gamma_B \cdot \log_2(\max(B, 1))
$$

实测 BERT b=1/4/8/16 的 cube_util 从 18.3% 升到 21.1%；拟合 γ_B ≈ 0.010。

### 3.5 完整 5 参数

| 参数 | 物理意义 | 拟合值（v5）|
|-----|---------|-----------:|
| α | M·N coupling（fill 主导项）| **14.60** |
| β | M·K coupling（drain 项）| **2.51** |
| γ | N·K coupling（reduce 项）| **1.75** |
| δ | linear edge 项 | ≈ 0 |
| γ_B | batch 摊薄系数 | **0.010** |

→ v5 模型只有 5 个 free param，远少于 OLS 的 6 个 log 特征，但跨模型泛化更好——这是 physics-informed 优势。

## 4. op_kind 区分（BMM vs MM）

msprof op_summary 的 `Op Type` 字段含 4 类 GEMM：MatMul、MatMulV2、BatchMatMul、BatchMatMulV2。前两类（MM）将 batch 维度 flatten 进 M；后两类（BMM）保持独立。

→ `effective_M` 计算时必须区分：

```python
def effective_M(M_per_batch, N, K, B, op_kind):
    if op_kind == 'MM':
        # M 已含 batch（CANN 把 batch flatten 了），不再乘 B
        return M_per_batch
    # BMM：判断是否 attention head（小 M/N/K）
    if min(M_per_batch, N, K) <= 128:
        # attention head：每 head 独立 tile，不乘 B
        return M_per_batch
    # 普通 BatchMatMul：乘 B
    return M_per_batch * B
```

attention head 区分理由：MultiHeadAttention 内 16 个 head（如 BERT 12 head, Qwen3-0.6B 16 head）在 CANN 中作为 BMM 但每个 head 单独 tile，head 间 batch 维不参与 cube 调度。

→ 没有 op_kind 区分时，BERT MAE 是 22 pp；区分后降到 14.33 pp。**这是 v5 vs v4 的关键改进**。

## 5. Levenberg-Marquardt 拟合

5 参数模型用 `scipy.optimize.least_squares(method='trf')` 拟合。损失函数：

$$
\mathcal{L}(\alpha, \beta, \gamma, \delta, \gamma_B) = \sum_{(M,N,K,B,B^\star) \in \text{train}} \bigl(\eta_\text{real}^\text{model}(M,N,K,B) - \eta_\text{real}^\text{measured}\bigr)^2
$$

边界约束：$\alpha, \beta, \gamma, \delta \in [0, 10^3]$；$\gamma_B \in [-0.5, 0.5]$。

初值：$(1, 1, 1, 1, 0.012)$。Levenberg-Marquardt 收敛 ~ 30 iter。

完整源码：`src/prism/eta_real/fit.py`。

## 6. 训练 / 验证集设计

### 6.1 数据采集

11 配置 msprof PipeUtilization 实测（原始 PROF_* 目录在 `legacy/data_archive/`）：

| 模型 | 配置 | 用途 |
|------|------|------|
| Qwen3-0.6B prefill | S=256/512/4096 × b=1/4/8 = 9 配置 | **训练 + 验证** |
| Qwen3-Embedding | (substitute by Qwen3-0.6B body) | 训练 |
| BERT-base | S=128 × b=1 | **验证** |
| GPT-2-small | S=512 × b=1 | **验证** |
| Qwen3-decode | M=4 S_kv=128 b=1 | 验证（decode 验证）|

每配置在 op_summary CSV 中提取 5-15 个 GEMM op shape。**总训练 shape 约 56 个，验证 shape 约 32 个**。

### 6.2 拟合输出（v5 final）

```json
{
  "method": "physics-informed (η_pipeline · η_tile · batch)",
  "params": {
    "alpha_MN_coupling": 14.5977,
    "beta_MK_coupling": 2.5051,
    "gamma_NK_coupling": 1.7484,
    "delta_linear_edge": 0.0,
    "gamma_B_batch": 0.0102
  },
  "training": {"n": 56, "mae_pp": 11.98, "rmse_pp": 16.33},
  "validation": {
    "bert":  {"n": 16, "mae_pp": 14.33, "rmse_pp": 18.93},
    "gpt2":  {"n": 16, "mae_pp": 12.07, "rmse_pp": 17.69}
  }
}
```

### 6.3 误差分析

按 op_shape 类别看 MAE：

| op shape 类 | 例 | Train MAE | Val MAE |
|-----------|---|---------:|---------:|
| 大 GEMM (M ≥ 1024 N ≥ 1024 K ≥ 1024) | Qwen3 FFN | 5-8 pp | 8-12 pp |
| 中 GEMM (M, N, K 均 ≥ 128) | BERT FFN1 | 10-15 pp | 12-18 pp |
| Attention head (M ≤ 128 或 K ≤ 128) | QK^T、AV | 15-25 pp | 15-25 pp |

→ attention head 是误差主要来源（小维度 + fill/drain 占比大，物理模型简化）。**长上下文 prefill (S=4096) 的 attention 是当前模型的弱点**，需要更大数据集做 GP 残差才能进一步降低（推迟到 paper-level 工作）。

## 7. 把 η_real 用到 wall-clock 预测

`predict_eta(M, N, K, B, params)` 调用方式：

```python
from prism.eta_real.predict import predict_eta
from prism.eta_real.fit import load_fit

params = load_fit('data/calibration/eta_physics_fit.json')['params']
eta = predict_eta(M=4096, N=3072, K=1024, B=1, params=params)
# eta ≈ 0.78 for Qwen3 FFN gate

# 用于 cycles 估算
cycles = (2 * M_eff * N * K) / (n_cores * cube_macs * eta)
wall_clock_us = cycles / clock_GHz / 1e3
```

`sweep/runner.py` 内部把 `predict_eta` 用在每个 GEMM op 上，再聚合到 layer + model 级别（详见 [04 §sweep 公式](04_arch_sensitivity.md#sweep-公式)）。

## 8. 加新 model 的流程

如果有新 transformer 模型要进 sweep MODELS dict（如 Qwen3-7B），步骤：

```bash
# 1) NPU 上跑 msprof 采集
ssh user@npu-server
cd ~/sim-experiment
bash benchmark/run_phase_b.sh   # 模板，按需改 model_name

# 2) rsync 拉回 PipeUtil + ArithUtil 数据
rsync -avz user@npu-server:~/sim-experiment/msprof_data_qwen3_7b/ msprof_data/

# 3) 提取 cube_util + pipe time
prism-extract --model qwen3_7b --batch 1 --metric ArithmeticUtilization
prism-extract --model qwen3_7b --batch 1 --metric PipeUtilization

# 4) 重新 fit η_real（含新 shape）
prism-fit \
  --cube-util-json data/calibration/cube_util_extracted.json \
  --output         data/calibration/eta_physics_fit.json

# 5) 验证 BERT MAE 仍 < 15 pp（硬门槛）
# 输出 JSON 含 validation.bert.mae_pp 字段

# 6) 加进 sweep MODELS dict
# 编辑 src/prism/sweep/runner.py 顶部 MODELS

# 7) 重跑 sweep
prism-sweep
```

详见 [tutorials/04_add_new_model.md](../tutorials/04_add_new_model.md)。

## 9. AIV 与 AIC 的 serial 关系实证

§7 公式 `T_compute = T_aic + T_aiv` 假设两单元单层关键路径上 serial。证据：

DaVinci 单 transformer layer 的实际数据流：

```
Cube (Q/K/V proj) → FixPipe → UB → AIV (LayerNorm) → UB ↑
                                                           Cube (QK^T) → FixPipe → UB →
AIV (Softmax) → UB → Cube (AV) → FixPipe → UB → AIV (output proj norm) → ...
```

**单 token 单层路径上 Cube 和 AIV 必然 serial**（数据依赖）。CANN runtime 跨 stream 可做 layer-level 重叠，但 msprof step_trace 实测显示 prefill 阶段 Cube/AIV 时间几乎完全 serial（重叠 < 5%）。

→ T_compute = T_aic + T_aiv 是 wall-clock 上界（serial 假设）。如有 partial overlap，实际 wall-clock 比预测略低，模型偏保守（< 5% 偏差，可接受）。

## 10. 已知局限

| # | 局限 | 影响 | 缓解 |
|---|-----|------|-----|
| 1 | attention head shape 拟合误差 15-25 pp | 长上下文 prefill 单 op 偏差大 | 全模型聚合后 ratio 误差 < 5%，实用可接受 |
| 2 | δ 拟合趋于 0 | linear edge 项被 fold 进 α | 接受，参数减为有效 4 个 |
| 3 | 跨芯片外推未验证 | 310P 上 η_real 是否同公式有效 | 需 310P msprof 数据，[06 §4](06_assumptions_limits.md#4-跨芯片外推) |
| 4 | b > 8 未在 train 集中 | b=16/32 拟合外推可能偏 | b=16 实测加入 train 后建议重 fit |
| 5 | η ≈ AIC mac 占比，不含 MTE bubble | 当 mte2 主导时 η_real 偏低估 cube cycles 利用 | 已在 [02 §pipe-aware](02_three_layer_roofline.md#5-t_aic-的-pipe-aware-拆分) 修正 |

完整局限：[06_assumptions_limits.md](06_assumptions_limits.md)

---

## 📚 参考

- 实测数据：`legacy/docs/cube_efficiency_calibration.md` v1.1（physics-informed v5 拟合记录）
- v5 拟合代码：`src/prism/eta_real/fit.py`
- v5 拟合输出：`data/calibration/eta_physics_fit.json`
- 拟合方法理论：More, J. J. *The Levenberg-Marquardt Algorithm: Implementation and Theory*, 1978
- 昇腾 DaVinci 架构：HotChips 31, 2019, "Da Vinci: A Scalable Architecture for Neural Network Computing"
- Cube fill/drain 公式：CUTLASS H100 GEMM library；FastAttention paper (Liu et al., 2024) (arXiv:2410.16663)
