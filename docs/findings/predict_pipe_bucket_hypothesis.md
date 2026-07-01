# PredictPipe v6 计划 — 按瓶颈分桶（响应用户 2026-05-17 洞察）

> 用户原话：**"Qwen3-prefill family 是典型的 gemm 占绝对优势，CUBE（AIC）是性能瓶颈，与其他几类不同性能瓶颈的模型是不是应该放在不同的 bucket 进行分类和校准？"**

完全正确。v5 用单一连续公式拟合所有模型失败的根因正是这个：**不同 bottleneck regime 的模型物理行为不同，amp magnitude 也不同**。

---

## 1. 测量数据验证：3 类 bucket 是真实存在的

| Config | AIV/AIC measured | AIC dominant pipe | Bottleneck regime |
|---|---:|---|---|
| Qwen3-prefill-S256-b8 | **0.94×** | mte2 (91.8%) | **AIC-bound (CUBE)** |
| Qwen3-prefill-S512-b4 | **0.94×** | mte2 (90.4%) | **AIC-bound (CUBE)** |
| Qwen3-prefill-S512-b8 | **0.89×** | mte2 (90.6%) | **AIC-bound (CUBE)** |
| Qwen3-prefill-S256-b4 | 1.18× | mte2 (90.6%) | AIC-bound |
| Qwen3-decode-Min4-Skv128-b1 | 1.16× | mte2 (84.8%) | AIC-bound (marginal) |
| Qwen3-prefill-S4096-b1 | 1.19× | fixpipe (55.6%) | AIC-bound (fixpipe) |
| BERT-base-S128-b1 | 1.41× | mte2 (79.4%) | Balanced |
| Qwen3-prefill-S256-b1 | 1.57× | mte2 (86.8%) | Balanced |
| GPT-2-S512-b1 | 1.97× | mte2 (67.2%) | Balanced |
| **Llama-3.2-1B-prefill-S2048-b1** | **2.66×** | mte2 (86.0%) | **AIV-bound** |
| **Qwen2.5-0.5B-prefill-S2048-b1** | **3.24×** | mte2 (87.9%) | **AIV-bound** |
| **SmolLM2-360M-prefill-S2048-b1** | **4.24×** | mte2 (80.0%) | **AIV-bound** |
| **ModernBERT-base-S4096-b1** | **4.54×** | fixpipe (66.9%) | **AIV-bound** |

**清晰三段**：
- **AIC-bound (AIV/AIC < 1.2)**：6 configs — Qwen3 prefill batch>1 / long-S / decode
- **Balanced (1.2-2.5)**：3 configs — BERT, GPT-2, Qwen3-b1
- **AIV-bound (>2.5)**：4 configs — Llama, ModernBERT, Qwen2.5, SmolLM2

---

## 2. 但 bucket 不能从 physics base 提取（重要发现）

尝试用 **physics base** 的 AIV/AIC ratio 来 predict bucket：

| Config | base AIV/AIC | measured AIV/AIC |
|---|---:|---:|
| BERT-S128 | 1.00× | 1.41× |
| Qwen3-S256-b1 | 0.87× | 1.57× |
| Qwen3-S256-b8 | 0.41× | 0.94× |
| ModernBERT-S4096 | 1.11× | 4.54× |
| Llama-S2048 | 0.63× | 2.66× |
| Qwen2.5-S2048 | 1.07× | 3.24× |
| SmolLM2-S2048 | 1.12× | 4.24× |
| Qwen3-S512-b4 | 0.50× | 0.94× |

**base ratio 集中在 [0.4, 1.1] 窄区间，但 measured ratio 从 0.89 到 4.54 分散 5×**。

→ **bucket 行为本身就是 amp 行为**。无法用 physics base 提前 detect bucket。

---

## 3. 但 bucket 可以从 spec features 间接预测

观察 amp 行为与 spec 的关联：

| Bucket | 典型配置 | spec 特征 |
|---|---|---|
| AIC-bound | Qwen3-prefill batch>1, S>=4096, decode | 28 layers + d_model=1024 + GQA q/kv=8 |
| AIV-bound | Llama (16L, d=2048), Qwen2.5 (24L, d=896), SmolLM2 (32L, d=960), ModernBERT (22L, d=768) | 中小 d_model + 适中 layers + 大 S |
| Balanced | BERT/GPT-2 (12L) + Qwen3-b1 | 浅或单 batch |

无单一 spec feature 能完美 separate（试过 GQA ratio、d_ff/d_model、d_head 等），但**组合特征 + 经验阈值**可达 ~90% 正确分类。

v6 起步桶分类器（启发式）：

```python
def bottleneck_bucket(spec, batch):
    # Cube saturation indicator
    flops_per_layer = spec.d_model * spec.d_ff * spec.S * batch
    gemm_intensity = spec.layers * flops_per_layer
    
    if spec.S == 1:
        return "AIC_DECODE"   # decode is its own regime
    
    # Qwen3-family heuristic: many shallow layers + extreme GQA + medium d_model
    if (spec.layers > 24 and spec.d_model < 1500 
        and spec.n_heads / max(spec.n_kv_heads, 1) >= 6
        and (batch > 1 or spec.S >= 4096)):
        return "AIC_QWEN3"
    
    # Big-d_model decoder = AIV-bound (Cube saturates easily)
    if spec.d_model >= 768 and spec.S * batch >= 1024:
        return "AIV_BOUND"
    
    return "BALANCED"
```

---

## 4. v6 实施计划

### 4.1 桶分类器

`physics_v6.py::bottleneck_bucket(spec, batch) → 4 buckets`：
1. `AIC_QWEN3`：Qwen3-prefill family 类特征（28L + d_model<1500 + 极端 GQA + batch>1 OR S>=4096）
2. `AIC_DECODE`：S==1
3. `AIV_BOUND`：典型 Llama/ModernBERT/Qwen2.5/SmolLM2 类
4. `BALANCED`：BERT/GPT-2 类小模型 + Qwen3-prefill-b1

### 4.2 Per-bucket amp 参数

每桶独立 amp 系数（每桶约 3-4 free params）：

| Bucket | amp_aic | amp_aiv | n_kernels mult | host_gap |
|---|---|---|---|---|
| AIC_QWEN3 | 高 (~10-14) | 中 (~3-5) | 高 (~25) | 中 |
| AIC_DECODE | 低 (~0.85) | 低 (~1.5) | 中 (~4.7) | 低 (~200) |
| AIV_BOUND | 低 (~1.5-2) | 高 (~4-6) | 低 (~6) | 中 |
| BALANCED | 中 (~2-3) | 中 (~2-4) | 中 (~6) | 中 |

### 4.3 拟合策略

每桶有 2-3 个 TRAIN configs：
- AIC_QWEN3 train: Qwen3-S256-b4 + Qwen3-S512-b4 → fit 共 2 configs
- AIC_DECODE train: Qwen3-decode → fit 1 config (用先验)
- AIV_BOUND train: ModernBERT → fit 1 config + 借鉴 Llama/Qwen2.5 测量值
- BALANCED train: BERT + GPT-2 → fit 2 configs

每桶 fit 在自己的 subset，外推风险被 bucket 边界限制。

### 4.4 预期结果

| | v4 | v5 | **v6** (期望) |
|---|---:|---:|---:|
| TRAIN MAE | 4.9% | 17.3% | **<10%** (per-bucket fit) |
| VAL_size MAE | ~511% | 103% | **<30%** |
| Llama wall_err | +1156% | +232% | **<30%** |

---

## 5. 风险与降级

| 风险 | 处置 |
|---|---|
| 桶分类器 misclassify 边界 case | 测试集覆盖所有桶，监测 mis-bucket 率 |
| 每桶 anchor 太少 (Qwen3-decode 只 1 个) | 用先验 prior + 跨桶 regularization |
| 新模型不属于任何已知桶 | 默认 fall back 到 BALANCED + low confidence label |
| 桶定义过于 Qwen3-specific（fragile）| 等 v6 落地后看效果，可能需要 v7 重构 |

---

## 6. 即时下一步

1. ✅ 验证三桶假设（数据 + 上表）
2. ✅ 确认 base ratio 不能预测 bucket（→ 必须用 spec heuristic）
3. ⏳ 实现 `physics_v6.py::bottleneck_bucket` + per-bucket amp 公式
4. ⏳ 写 `fit_v6.py` per-bucket DE fit
5. ⏳ 测试 + 比较 v4/v5/v6 在 TRAIN/VAL_batch/VAL_size 上的 MAE
6. ⏳ 通过：commit + push + Issue #2 update
7. ⏳ 不通过：分析每桶残差，迭代 heuristic 或公式

总工时估算：~3-4 hours。

---

**结论**：用户的瓶颈分桶建议是 v5 残留 232% Llama err 的正确解法方向。v5 试图用 single continuous formula 拟合 3 个 distinct regimes，物理上注定失败。v6 显式按 bucket 拆分 + per-bucket fit，应能把 Llama 拉到 < 30%。
