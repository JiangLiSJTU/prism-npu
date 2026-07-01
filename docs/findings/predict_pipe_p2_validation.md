# PredictPipe P2 — 10 模型批量预测验证报告（v4 Method B 落地后）

> Issue #2 P1 + P2 落地 | 输入 arch：`arch/ascend_910b4_for_sweep_v2.yaml`
> 拟合常数：K0=1.86 μs/kernel, H_prefill=13,424 μs, H_decode=204 μs
> AIV v4 Method B: `aiv_time = (n_vk·C_kernel + data_MB·C_data) × amp`
> 其中 `amp = a0 + a1·attn_frac + a2·(w_proxy/1000)²`，prefill 6 参数全网格拟合
> AIC archetype amp 桶（沿用 v3）：0.85 / 1.15 / 5.5 / 14.16（按 weight_proxy_mb + S=1）
> 历史：v0.1 → v3 (P1 fix) → v4 Method B（本次）

---

## 1. PredictPipe 历代精度对比（5 个 measured configs，wall_clock 误差）

| Config | v0.1 | v3 (P1, 3-bucket amp + aiv=aic×1.25) | v4 (continuous amp) |
|---|---:|---:|---:|
| BERT-base-S128-b1 | 11.8% | 3.6% | **3.4%** |
| GPT-2-S512-b1 | 12.8% | 1.5% | 10.9% |
| Qwen3-prefill-S256-b1 | 76.7% | 2.0% | 10.0% |
| Qwen3-prefill-S512-b4 | 87.2% | 9.1% | **4.3%** |
| Qwen3-decode-Min4-Skv128-b1 | 51.0% | 10.0% | **8.6%** |

→ 5/5 wall_clock 误差仍 < 11%；v4 较 v3 在 GPT-2 + Qwen3-S256 上轻微回退（v3 经验 `aiv=aic×1.25` 与其他组件偏差互相抵消，v4 AIV 更诚实后暴露出来），在 Qwen3-S512-b4 + decode 上改善。

**AIV-only 误差**（v4 关键指标）：6 configs（含 Net-Transformer）Training MAE **4.9%**，全部 < 10%，GPT-2 outlier (v3=32.4%) 消除 → 6.7%。详见 [`methodology/08_predict_pipe.md §3.5`](../methodology/08_predict_pipe.md#35-aiv-v4-method-b-continuous-amp)。

`test_p1_wall_clock_error_under_30pct_on_all_measured` 持续通过。

---

## 2. P2 — 10 模型批量预测（v4 数字）

| 模型 | wall_clock | aic_time | aiv_time | aic_amp | aiv_attn_frac | aiv_amp_v4 | n_kernels | confidence |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BERT-base-S128-b1 | 15,652 | 704 | 912 | 1.15 | 0.24 | 1.49 | 330 | low |
| BGE-small-en-v1.5-S512-b1 | 17,902 | 801 | 3,064 | 1.15 | 0.56 | 2.74 | 330 | low |
| DeBERTa-base-S512-b1 | 19,219 | 1,582 | 3,601 | 1.15 | 0.56 | 2.74 | 330 | low |
| Flan-T5-base-encoder-S512-b1 | 18,819 | 1,491 | 3,291 | 1.15 | 0.56 | 2.74 | 330 | low |
| **Llama-3.2-1B-prefill-S2048-b1** ⚠️ | **2,478,840** | 280,313 | 2,176,789 | 14.16 | 0.61 | **66.85** | 4,480 | **medium**(外推) |
| ModernBERT-base-S4096-b1 | 317,439 | 59,465 | 243,428 | 1.15 | 0.85 | 5.04 | 605 | low |
| Qwen2.5-0.5B-prefill-S2048-b1 | 413,822 | 119,093 | 268,833 | 5.50 | 0.61 | 10.91 | 6,720 | medium |
| SmolLM2-135M-decode-b1 | 3,957 | 583 | 1,777 | 0.85 | 0.00 | 1.50 | 750 | medium |
| SmolLM2-360M-prefill-S2048-b1 | 499,117 | 151,573 | 317,492 | 5.50 | 0.61 | 9.94 | 8,960 | medium |
| T5-small-encoder-S512-b1 | 15,260 | 437 | 1,092 | 1.15 | 0.56 | 2.74 | 165 | low |

单位：μs。`aiv_amp_v4` = `a0 + a1·attn_frac + a2·(w_proxy/1000)²` = `-0.2 + 4.0·attn_frac + 14.0·(w_proxy/1000)²`。

⚠️ **Llama-3.2-1B**：w_proxy=2147 MB 远超训练集 [200,600] MB 范围，`(w_proxy/1000)²·14` 在外推区非线性爆炸（贡献 64.5），导致 aiv_amp=66.85（vs 其他 prefill 模型约 10×）。**预测值是上界估计，可信度有限**——需 P3 真机数据校准 amp 在 w_proxy > 1000 MB 区间的曲率。

---

## 3. 关键观察

### 3.1 v4 → v3 大模型 aiv 显著扩大（高 attn_frac × 大 weight）

| 模型 | v3 aiv (aic×1.25) | v4 aiv (continuous) | Δ |
|---|---:|---:|---:|
| BERT-base-S128-b1 | 880 | 912 | +4% |
| ModernBERT-S4096 | 74,331 | 243,428 | **+228%** |
| Llama-3.2-1B-S2048 | 350,391 | 2,176,789 | **+521%**（外推）|
| Qwen2.5-0.5B-S2048 | 148,867 | 268,833 | +81% |
| SmolLM2-360M-S2048 | 189,466 | 317,492 | +68% |

v3 经验 `aiv=aic×1.25` 系统性低估了高 attn_frac（softmax 字节 ∝ S²）+ 大权重场景的 AIV cost。v4 captured this：

1. **attn_frac ∝ L·H·S²/data_bytes**：S=4096 时 attn_softmax 数据量主导，amp 显著大于 S=128
2. **`(w_proxy/1000)²`**：大权重模型 AIV vector ALU 需要 re-fetch 中间激活更频繁

### 3.2 dominant pipe 分布

- `mte2` dominant：大多数 prefill（HBM 带宽是 AIC bottleneck）
- `mac` dominant：仅 Llama-3.2-1B 长 prefill + 大 weight（reaches Cube saturation）

### 3.3 P2 待真机验证（P3：910B SSH 已授权）

| 模型 | v4 预测 wall (ms) | 验证状态 |
|---|---:|---|
| ModernBERT-base-S4096-b1 | 317 | 待 P3 实测 |
| **Llama-3.2-1B-prefill-S2048-b1** | **2,479** | **待 P3 实测**（外推风险高，最高优先级）|
| Qwen2.5-0.5B-prefill-S2048-b1 | 414 | 待 P3 实测 |
| SmolLM2-360M-prefill-S2048-b1 | 499 | 待 P3 实测 |
| SmolLM2-135M-decode-b1 | 4.0 | 待 P3 实测 |

---

## 4. 置信度修订

| 标签 | 模型数 | 条件 |
|---|---:|---|
| **low** | 6 | encoder（5）+ GLU FFN（ModernBERT 同时是 encoder+GLU）|
| **medium** | 4 | 大 decoder prefill（3，依赖 6D 拟合外推）+ decode（1） |
| **high** | 0 | 当前无：所有 "high" 候选要么是 encoder 要么是大模型 |

**v4 暂未在 confidence 中编码外推距离**——Llama-3.2-1B（w_proxy=2147 MB）目前仍标 "medium"，但外推距离 3.5× 训练集上界。后续应在 `assign_confidence` 加入 w_proxy 外推标记。

P3 真机 msprof 落地后，可：
- 把 ModernBERT 的 encoder + GLU 测出来 → encoder "low" 提到 "medium"
- 把 Llama/Qwen2.5/SmolLM2 prefill 测出来 → "medium" 提到 "high"，并在 w_proxy ∈ [600, 2200] MB 加 ≥3 个数据点重拟合 `a2` 曲率

---

## 5. 输出文件

- 完整预测 JSON：`data/calibration/predict_pipe_batch_p2.json`（可直接喂入 `prism-sweep --pipe-baseline ...`）
- 重生成脚本：`scripts/regenerate_predict_pipe_batch_p2.py`（v4 / 后续模型升级后一键刷新）
- 新增 YAML（7 个）：`models/regime/{bge_small,deberta_base,flan_t5_base_encoder,llama_3_2_1b_prefill,qwen2_5_0_5b_prefill,smollm2_360m_prefill,t5_small_encoder}.yaml`
- 复用 YAML（3 个）：`models/regime/{bert_base, modernbert_base_prefill_S4096, smollm2_135m_decode}.yaml`
- v4 AIV 公式：`src/prism/predict_pipe/physics.py:compute_attention_fraction` + `src/prism/predict_pipe/predict.py:predict_aiv_v2`
- v4 拟合常数：`data/calibration/predict_pipe_params.json::{aiv_C_kernel_us, aiv_C_data_us, aiv_amp_a0/a1/a2, aiv_amp_decode}`
