# PredictPipe v4 — Llama-3.2-1B OOS 验证（外推灾难性失败 + 关键校准信号）

> 日期：2026-05-15 | 数据：`msprof_data/msprof_llama_3_2_1b_prefill_S2048_b1_PipeUtilization/`（910B4 实测，LOOP=10，2 分钟 ATC + msprof）
> 模型：Llama-3.2-1B prefill S=2048 b=1（unsloth/Llama-3.2-1B 社区镜像，无 HF gate）
> 预测模型：v4 Method B continuous amp（commit `bf0b0c9`）
> 实测来源：unsloth 镜像 ONNX export → ATC → ais_bench --loop 10

---

## 1. 结果速览 —— 12 倍 over-prediction

| Field | v4 Pred | Measured | Err |
|---|---:|---:|---:|
| **wall_clock_us** | **2,478,840** | **197,401** | **+1,156%** ❌ |
| aic_time_us | 280,313 | 50,198 | +458% ❌ |
| aiv_time_us | 2,176,789 | 133,360 | +1,532% ❌ |
| n_kernels_per_inf | 4,480 | 987 | +354% ❌ |
| kernel_gap_us | 8,314 | 1,832 | +354% ❌（n_kernels 级联）|
| host_gap_us | 13,424 | 12,011 | **+11.8% ✓**（host model 普适）|

**v4 预测 Llama 跑 2.48 秒；真机跑 197 毫秒**。v4 在 w_proxy=2147 MB 区间外推完全失控——印证 `2026-05-12 predict_pipe_p2_validation.md` 第 35 行的 ⚠️ 外推警告。

---

## 2. 故障根因 —— 3 个独立维度都过激

### 2.1 AIC archetype_amplification = 14.16× （来自 v3.1 "large_long" 桶）

`physics.py::archetype_amplification` 当 `weight_proxy_mb > 1500 AND S > 1024` 返回 14.16×。这个值是为消除 Qwen3-prefill-S4096-b1 的 outlier 在 v3.1 拟合的，**Llama-1B-S2048 也落入此桶但实际不需要这么夸张**：

| Config | w_proxy | S | aic 实测 / base | v4 amp | 是否合适 |
|---|---:|---:|---:|---:|---|
| Qwen3-prefill-S4096-b1 | 230 MB | 4096 | ~14× | 14.16× | ✓（calibrate point）|
| **Llama-3.2-1B-S2048** | **2147 MB** | **2048** | **~2-3×** | **14.16×** | **❌ 5× 过激** |

桶分类粗糙：把 S=4096 长上下文 amp 错误地外推到 S=2048。

### 2.2 AIV continuous amp `(w_proxy/1000)²` 在 2147 MB 处爆炸

`predict_aiv_v2` 公式：
```
amp = -0.2 + 4.0 × attn_frac + 14.0 × (w_proxy/1000)²
```

w_proxy=2147 MB 时：`14.0 × 2.147² = 64.5`（占总 amp 的 96%）。
但训练集 w_proxy ∈ [200, 600] MB，最大点 Qwen3-S512-b4 也只到 230 MB。
**2147 MB 是训练集最大 3.5×，平方项 × 7× 外推 → 完全失控**。

| Config | w_proxy | (w_proxy/1000)² | 14·^ |
|---|---:|---:|---:|
| Qwen3-S512-b4 | 230 MB | 0.053 | 0.74 |
| ModernBERT-S4096 | 220 MB | 0.048 | 0.68 |
| **Llama-3.2-1B-S2048** | **2147 MB** | **4.61** | **64.5** |

平方关系在外推区被放大数十倍。

### 2.3 n_kernels archetype multiplier = 28×（"large prefill"）

`model_spec.py::estimate_n_kernels` 对大 decoder prefill 乘 28×，对 Llama 估出 4480 kernels。
**实测 987 kernels（4.5× 少）**。

Llama-1B 的 CANN 算子融合度比 Qwen3 高（更现代化的 SwiGLU + RMSNorm + RoPE 组合容易 fuse 成大 fused kernel）。28× 系数是 Qwen3-prefill 数据 fit 出来的——不适用 Llama 类。

n_kernels 偏差独立放大了 `kernel_gap = K0 · n_kernels`（误差 +354%），但因 K0=1.86 很小，仅 8 ms 偏差占 wall_clock 0.3%。

---

## 3. 实测数据本身价值极高（用作 v5 refit 的关键 anchor）

### Pipe breakdown（per inference）

```
AIC dominant: mte2 (43 ms) ← HBM read 主导
  mac:     29 ms
  mte1:    24 ms
  fixpipe: 12 ms
  scalar:   2.5 ms

AIV dominant: mte2 (114 ms) ← UB read 主导
  vec:      16 ms
  mte3:     88 ms
  scalar:    2 ms
  idle:      0 ms（完全 saturated）

aiv_time:    133 ms（AIC 2.66×）
aic_time:     50 ms
kernel_gap:  1.8 ms (987 kernels × K0=1.86)
host_gap:   12.0 ms
wall:       197 ms
```

### 关键观察

1. **AIV 是真瓶颈**（133 vs aic 50）—— Llama-1B-S2048 受限于 UB↔L1 数据搬运而非 Cube 计算
2. **aiv_mte2 + aiv_mte3 = 202 ms** vs aiv_vec=16 ms —— 进一步印证 fork session §3 物理直觉："AIV = MTE bound，不是 ALU bound"
3. **aic_dominant = mte2** —— HBM 带宽限制（2.37 GB weight + S=2048 activation）符合预期
4. **host_gap_per_kernel = 12.17 μs/kernel**（介于 BERT 41.66 与 ModernBERT 8.41 之间）—— K0=1.86 普适，H_prefill=13424 普适

### Llama 实际 amp 系数（待拟合）

倒推从实测推算应有的 amp：

| 量 | 实测 / 推算 |
|---|---|
| aic_mac 实测 | 29 ms |
| aic_mac base (FP16 GEMM theoretical) | ~16 ms |
| 实际 aic amp | **~1.8×**（不是 v4 的 14.16）|
| aiv_time 实测 | 133 ms |
| aiv_time base (vector ALU only) | ~15 ms |
| 实际 aiv amp | **~9×**（不是 v4 的 66.85）|

---

## 4. v5 修复路径（Issue #2 真闭环）

### 4.1 改 amp 函数从平方到 log 或 sigmoid 形

`(w_proxy/1000)²` 在外推区平方放大是失控之源。改用：

候选 A：log 形 `amp = a0 + a1·attn_frac + a2·log(1 + w_proxy/1000)`
候选 B：饱和形 `amp = a0 + a1·attn_frac + a2·(w_proxy/1000) / (1 + w_proxy/1000)`

候选 B（sigmoid-like）当 w_proxy → ∞ 时 amp → a0+a1+a2，自然 cap，杜绝外推灾难。

### 4.2 AIC archetype 桶细化（按 S 拆开）

当前桶：`(small/big) × (decode/prefill_short/prefill_long)` → 4 桶。
应加维度：`(S < 2048 / S ≥ 2048)`，让 Llama-S2048（w_proxy 大但 S 中等）落入合适桶而非 Qwen3-S4096 的极端桶。

### 4.3 n_kernels archetype multiplier 重拟合

把 Llama (987 kernels) + ModernBERT (1478) 加入 fit。当前 28× 应降到 ~6-8×（取决于桶）。

### 4.4 拟合优先级 / 数据规划

5 个 measured configs + ModernBERT + Llama = 7 个 prefill anchors（覆盖 w_proxy ∈ [70, 2147] MB，S ∈ [128, 4096]）。  
再补 Qwen2.5-0.5B-S2048 + SmolLM2-360M-S2048 = 9 个 anchor。
6D grid search 重跑，期望：
- a2 系数从 14 降到 1-3
- aic 桶细分后 large_long 14.16× 降到 2-4×
- n_kernels multiplier 28× 降到 5-8×
- 训练 MAE 微涨 6-8%（vs v4 的 4.9%），但 OOS Llama 误差降到 < 30%

---

## 5. 与 ModernBERT 验证结果对照

| | ModernBERT-S4096 | Llama-3.2-1B-S2048 |
|---|---|---|
| **wall err** | **−1.8%** ✓ | **+1156%** ❌ |
| aic err | +7% | +458% |
| aiv err | −4% | +1532% |
| n_kernels err | −59% | +354% |
| host_gap err | +8% | +12% |
| w_proxy MB | 220 | 2147（外推 9.8×）|
| 训练集落入 | bucket 内（小 prefill）| 远在外（极端 large）|

**结论**：v4 在 w_proxy ∈ [200, 600] MB 训练区间内非常好，**在 > 1000 MB 外推区间完全失控**。Llama 实测正好补上了大 w_proxy anchor，让 v5 refit 有数据。

---

## 6. 输出文件

- 实测数据：`data/calibration/pipe_baseline_per_model.json::configs["Llama-3.2-1B-prefill-S2048-b1"]`（26 configs total）
- 源数据：`msprof_data/msprof_llama_3_2_1b_prefill_S2048_b1_PipeUtilization/`（28 MB，13 iterations of ais_bench loop=10）
- export 脚本：`~/sim-experiment/benchmark/export_llama_3_2_1b_prefill.py`（已 push 到 910B，可复用）
- ATC + msprof 脚本：`~/sim-experiment/benchmark/run_llama_atc_msprof.sh`
- 本报告：`docs/findings/predict_pipe_llama_oos_critical.md`

---

## 7. 立即生效的影响

### 7.1 `predict_pipe_batch_p2.json` 中 Llama 行需重新生成 + 标"已知严重高估"

当前 v4 给出 wall=2479ms，但真机 197ms。`predict_pipe_p2_validation.md` 已标外推警告，但 ⚠️ 强度需提升到"已实测验证为灾难性误差"。

### 7.2 confidence label 必须修

当前 Llama 标 "medium"。改为 **"low"**：
```
low (heuristic amp extrapolated 9.8× beyond [200, 600] MB training range;
     measured w_proxy>1500 MB shows 12× wall_clock over-prediction;
     do NOT use without v5 refit)
```

### 7.3 后续步骤

1. **立即**：commit Llama 数据 + finding doc + 把 Llama confidence 降到 low（代码修改）
2. **本 session 内（如时间允许）**：跑 Qwen2.5 + SmolLM2-360M → 凑齐 9 anchors → v5 refit
3. **下 session**：v5 公式（log/sigmoid amp）+ 拟合 + LOO CV + 发布 v5 commit + close Issue #2

---

**TL;DR**：Llama OOS 验证完美达到目的——**证伪了 v4 的 large w_proxy 外推，给出 v5 修复所需的关键 anchor 数据**。host_gap 仍然 12% 误差证明 host 模型普适，所有"灾难性"误差集中在 amp 上——是定位明确、修复路径清晰的好结果。
