# PredictPipe v4 — P3 OOS 综合验证（4 配置）+ v5 refit 计划

> 日期：2026-05-15 | 输入 arch：`arch/ascend_910b4_for_sweep_v2.yaml`
> 训练集：6 measured configs（BERT-S128, GPT-2-S512, Qwen3-prefill-S256/S512-b4, Qwen3-decode, Net-Transformer-S256）
> OOS 验证：4 个新 prefill models（ModernBERT, Llama-3.2-1B, Qwen2.5-0.5B, SmolLM2-360M）
> P3 完成度：**4/4 完成**，Issue #3 主线任务全部落地

---

## 1. OOS 误差总览（最关键的一张表）

| 模型 | w_proxy MB | wall_clock 实测 | v4 预测 | wall_clock err | aiv err | aic err | n_kernels err |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ModernBERT-base-S4096-b1** | **220** | **323 ms** | 317 ms | **−1.8% ✓** | −3.6% | +7.0% | −59% |
| Qwen2.5-0.5B-prefill-S2048-b1 | 782 | 162 ms | 414 ms | **+155% ❌** | +198% | +329% | +358% |
| SmolLM2-360M-prefill-S2048-b1 | 708 | 211 ms | 499 ms | **+137% ❌** | +160% | +426% | +360% |
| **Llama-3.2-1B-prefill-S2048-b1** | **2,147** | **197 ms** | 2,479 ms | **+1156% ❌❌** | +1532% | +458% | +354% |

→ **w_proxy ≤ 250 MB：v4 极准（1.8%）；w_proxy ≥ 700 MB：v4 catastrophic（137-1156% 高估）**

---

## 2. v4 失效的 3 个独立维度

### 2.1 AIC archetype_amplification 桶过激（v3.1 over-fit Qwen3-S4096）

```python
def archetype_amplification(w_proxy_mb, S):
    if S == 1: return 0.85       # decode
    if w_proxy_mb < 600: return 1.15  # small
    if w_proxy_mb >= 1500 and S >= 4096: return 14.16  # large_long ← Qwen3-S4096 outlier 拟合
    return 5.5                    # mid/large prefill
```

实测推算的真实 aic amp：

| Config | v4 amp | 实测 amp (aic实测/base) | 偏差 |
|---|---:|---:|---|
| ModernBERT-S4096 | 1.15 | ~1.0 | 接近 ✓ |
| Qwen2.5-S2048 | 5.5 | ~1.3 | 4.2× 过激 |
| SmolLM2-360M-S2048 | 5.5 | ~1.4 | 3.9× 过激 |
| Llama-S2048 | 14.16 | ~1.8 | 7.9× 过激 |
| Qwen3-S4096-b1 | 14.16 | ~14× | 准（calibrate 点）|

**根因**：把 Qwen3-S4096 的"长上下文 attention 重叠拖累" amp 错误归因为"weight 大"，导致 Llama-S2048（weight 大但 S 不长）误命中 14.16× 桶。

### 2.2 AIV continuous amp `(w_proxy/1000)²` 平方外推爆炸

```python
amp = -0.2 + 4.0 * attn_frac + 14.0 * (w_proxy / 1000) ** 2
```

| Config | w_proxy MB | (w_proxy/1000)² × 14 | 实测 amp |
|---|---:|---:|---:|
| BERT-S128 | 89 | 0.11 | ~1.5 ✓ |
| ModernBERT-S4096 | 220 | 0.68 | ~5 ✓ |
| Qwen3-S512-b4 | 230 | 0.74 | ~10 ✓ |
| Qwen2.5-S2048 | 782 | 8.6 | ~5 ❌ |
| SmolLM2-360M-S2048 | 708 | 7.0 | ~6 ❌ |
| Llama-S2048 | 2147 | **64.5** | **~9 ❌** |

平方项在 [200, 600] MB 训练区间被 fit 得贴合，但外推到 700-2147 MB 时**爆炸式增长 7-90×**。

**根因**：6 个 measured configs 中 w_proxy 范围 [89, 230] MB，连续 amp 曲面在 230 MB 之外没有 data 约束，只能由公式形式外推。`x²` 是错误的外推假设——实际行为饱和（接近常数），不是指数式爆炸。

### 2.3 n_kernels archetype multiplier 过激（4× 系统性高估）

```python
def estimate_n_kernels(spec):
    base = ...  # 几何计算
    if spec.S == 1: multiplier = 4.7         # decode
    elif w_proxy < 600: multiplier = 2.5     # small prefill / encoder
    else: multiplier = 28                    # large prefill ← 这里全部高估 4×
```

实测 n_kernels：

| Config | v4 n_kernels (mult=28) | 实测 | 实际 multiplier |
|---|---:|---:|---:|
| Qwen2.5-S2048 | 6,720 | 1,467 | **~6** |
| SmolLM2-360M-S2048 | 8,960 | 1,947 | **~6** |
| Llama-1B-S2048 | 4,480 | 987 | **~6** |
| ModernBERT-S4096 | 605 (mult=2.5) | 1,478 | ~6 |

**所有大 prefill 模型实际 multiplier 都约 6**，不是 28。CANN ATC 对 SwiGLU/RMSNorm/RoPE 的 fusion 效率远超 Qwen3 prefill 训练集所示。

### 2.4 host_gap 不是常量

| Config | host_gap 实测 | per_kernel | 备注 |
|---|---:|---:|---|
| BERT-S128 | 14,079 μs | 41.66 | training |
| ModernBERT-S4096 | 12,420 μs | 8.41 | 很多 fusion |
| Llama-S2048 | 12,011 μs | 12.17 | |
| Qwen2.5-S2048 | **41,608 μs** | **28.36** | **3× 标准值** |
| SmolLM2-360M-S2048 | **55,935 μs** | **28.73** | **4× 标准值** |

`H_prefill = 13424` 常量假设不成立——Qwen2.5 / SmolLM2 host_gap 是 3-4× 标准值。值得注意：per_kernel 数值在 Qwen2.5 + SmolLM2 上几乎相同（28.3-28.7），可能是某种 model family 特征。

---

## 3. AIV 实测物理验证（fork session §3 直觉再确认）

每个 OOS 模型的 AIV pipe breakdown：

| Model | aiv_vec | aiv_mte2 | aiv_mte3 | aiv_idle | mte2+mte3 / vec |
|---|---:|---:|---:|---:|---:|
| ModernBERT-S4096 | 33,079 | 220,128 | 166,479 | 0 | **11.7×** |
| Qwen2.5-S2048 | 11,448 | 75,531 | 61,433 | 0 | 12.0× |
| SmolLM2-360M-S2048 | 15,489 | 103,181 | 88,188 | 0 | 12.4× |
| Llama-S2048 | 16,347 | 114,473 | 87,892 | 0 | 12.4× |

**4/4 模型 aiv_mte2+aiv_mte3 ≈ 12× aiv_vec**——证实 AIV 是数据搬运（UB↔L1）主导，不是 ALU 计算主导。**fork session §3 物理直觉完全验证**。

→ v5 的 AIV 公式应当显式分拆 vec / mte2 / mte3 三部分，而不是把整个 AIV 当 black-box 用 amp 缩放。

---

## 4. v5 refit 路径（具体到公式）

### 4.1 AIV 物理化（替代 `amp × black_box`）

```python
# 当前 v4
aiv_time = (n_vk * C_kernel + data_MB * C_data) * amp_v4

# 提议 v5
aiv_vec_time   = aiv_vec_ops * (1/aiv_throughput) * eta_vec
aiv_mte2_time  = data_read_MB / ub_l1_bw_gbs * 1e6     # UB↔L1
aiv_mte3_time  = data_write_MB / fixpipe_bw_gbs * 1e6  # UB↔L1
aiv_time = aiv_vec_time + aiv_mte2_time + aiv_mte3_time   # serial（实测 idle=0）
```

参数（拟合 4 OOS + 6 training = 10 configs）：
- `eta_vec` ≈ 1.5（vec 实际 throughput 比理论 peak 慢的系数）
- `ub_l1_bw_gbs` 已在 arch yaml = 2048
- `fixpipe_bw_gbs` 已在 arch yaml = 4096

### 4.2 AIC archetype amp → 物理化（pipe-aware）

```python
# v5: 不用 archetype 桶，直接按 pipe scaling
aic_mac     = total_gemm_ops * fp16_byte / fp16_tflops * 1e6 / eta_compute
aic_mte1    = weight_re_fetch_MB / l1_l0_bw_gbs * 1e6
aic_mte2    = total_hbm_traffic_MB / hbm_bw_gbs * 1e6
aic_fixpipe = output_MB / fixpipe_bw_gbs * 1e6
aic_time = max(aic_mac, aic_mte1, aic_mte2, aic_fixpipe) + aic_bubble
```

`aic_bubble` 占总 aic_time 的 7-15%（4 OOS 实测），可作 model-class 调整或固定 12%。

### 4.3 n_kernels archetype mult: 28 → 6

降低 large prefill multiplier。新桶：

| 桶 | 多重 |
|---|---:|
| decode (S=1) | 4.7（沿用） |
| encoder / small prefill (w_proxy < 250) | 2.5（沿用） |
| **mid prefill (250 ≤ w_proxy < 1000)** | **6（新）** |
| **large prefill (w_proxy ≥ 1000)** | **6（降低 28→6）** |

或者用连续函数：`mult = 2.5 + 3.5 × (1 - exp(-w_proxy/500))`，自然饱和到 6。

### 4.4 host_gap: 不再常量

新公式：

```python
# 三参数线性
host_gap = beta_0 + beta_kernel * n_kernels + beta_size * weight_proxy_MB / 1000
```

或者按 model family 分（BERT-family / Qwen-family / Llama-family）—— 但这反映 CANN runtime 行为，不是物理学。建议 v5 用线性，留 family-aware 给 v6。

---

## 5. 拟合数据规模（10 anchors）

v5 fit 数据：6 training + 4 OOS = **10 prefill anchors**，覆盖：

| 维度 | 范围 |
|---|---|
| w_proxy MB | [89, 2147]（24× 跨度，对数空间 4 个量级）|
| S | [128, 4096] |
| attn_frac | [0.0, 0.86] |
| arch | encoder, decoder（含 GQA/SwiGLU/GLU 三种）|
| measured wall_clock | 16 ms ~ 323 ms |

LOO CV：留一 model family 验证，OOS MAE 目标 < 25%（v4 当前 OOS 极端误差 1156%）。

---

## 6. 立即行动 / 推迟项

### 本 session 已落地（commit）

- [x] ModernBERT 实测 + finding doc (`6c86baa`)
- [x] Llama 实测 + finding doc (`604c50f`)
- [x] Qwen2.5 + SmolLM2-360M 实测 (`facee5d`)
- [x] 本综合 P3 finding doc（本文件）

### 下 session 推荐顺序

1. **v5 公式实现**：`src/prism/predict_pipe/physics.py` + `predict.py` 重写 AIV pipe-aware + AIC 物理化
2. **v5 拟合**：10 anchors grid search 或 SciPy minimize
3. **v5 LOO CV**：留一 model family，目标 OOS MAE < 25%
4. **v5 batch_p2 regenerate**：跑 `scripts/regenerate_predict_pipe_batch_p2.py`
5. **v5 测试固化**：把 LOO CV 加进 `tests/test_predict_pipe.py`
6. **Issue #2 close**：v4 → v5 全 pipeline 跑通，confidence label 修正

### 推迟到 v6 / Issue #4-5

- host_gap 按 model family 分（v5 用线性已足够）
- AIC pipe 内部 misprediction（mte2 vs fixpipe）—— 影响 dominant 标签但不影响 wall_clock max
- decode batch>1 / 多 S_kv 验证
- SmolLM2-135M decode export bug（transformers q_length=1）

---

## 7. 关键洞察总结（一句话版本）

> **v4 在小模型（w_proxy ≤ 250 MB）准到 1.8% 是真本事，但在中大模型（≥ 700 MB）灾难性高估 137-1156% 暴露了 archetype 桶 + 平方外推 + n_kernels 28× 三个系统性 over-fit。4 个 OOS 实测拿到的不仅是验证数据，而是 v5 物理化重写所需的完整 anchor 网。**

整个 P3 验证的"灾难"反而是最好的结果——证伪比验证更有信息量。
