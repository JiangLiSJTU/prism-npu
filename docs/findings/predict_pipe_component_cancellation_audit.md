# PredictPipe — Component Error Cancellation Audit（用户 2026-05-18 mandate）

> 用户原话："我希望泛化能力强，同时各个部件的仿真误差尽量小。"
>
> 当前现状：**v4-v7 都不同程度地用部件误差互相抵消来过 wall_clock 闸门**。最严重的是 v6 TRAIN，AIC 47.7% + AIV 45.7% 互相抵消，wall_clock 仅 0.2%。这是 fit objective 设计的副作用，不是 bug，但需要明确审计。

---

## 1. 完整 Per-Component MAE 矩阵（v4 / v5 / v6 / v7 × TRAIN(6) / OOS(4)）

| Version/Split | AIC MAE | AIV MAE | n_kern MAE | wall MAE |
|---|---:|---:|---:|---:|
| **v4 TRAIN(6)** | **18.9%** | **4.9%** | 11.7% | 6.2% |
| v5 TRAIN(6) | 41.7% | 39.9% | 47.4% | 17.3% |
| v6 TRAIN(6) | 47.7% | 45.7% | 67.4% | **0.2%** ← 抵消 |
| v7 TRAIN(6) | 49.7% | 47.5% | 94.8% | 18.0% |
| **v8 TRAIN(6)** | 43.7% | **25.6%** | **37.6%** | 20.9% |
| v4 OOS(4) | 304.9% | 473.5% | 282.8% | 362.4% |
| v5 OOS(4) | 175.6% | 100.7% | 49.5% | 87.8% |
| **v6 OOS(4)** | 56.9% | **6.4%** | 11.4% | **10.1%** |
| v7 OOS(4) | 129.7% | 25.1% | 95.9% | 11.8% |
| **v8 OOS(4)** | **7.0%** | **8.0%** | **2.8%** | **8.4%** ← 全 component < 10% |

### 1.1 Cancellation Ratio（关键诊断指标）

`cancellation_ratio = max(AIC_err, AIV_err) / wall_err`

健康值 ≈ 1-3（部件误差 ≈ wall 误差，无显著抵消）。

| Version/Split | AIC | AIV | wall | **ratio** | 诊断 |
|---|---:|---:|---:|---:|---|
| v4 TRAIN | 18.9% | 4.9% | 6.2% | **3.0** ✓ | 部件诚实 |
| v5 TRAIN | 41.7% | 39.9% | 17.3% | 2.4 ✓ | 部件诚实 |
| **v6 TRAIN** | **47.7%** | **45.7%** | **0.2%** | **204.7** ❌❌ | **极端抵消** |
| v7 TRAIN | 49.7% | 47.5% | 18.0% | 2.8 ✓ | 部件诚实 |
| **v8 TRAIN** | 43.7% | 25.6% | 20.9% | **2.1** ✓ | **multi-objective 后部件诚实** |
| v4 OOS | 304.9% | 473.5% | 362.4% | 1.3 | 部件 + wall 都差 |
| v5 OOS | 175.6% | 100.7% | 87.8% | 2.0 | 一致差 |
| **v6 OOS** | 56.9% | 6.4% | 10.1% | **5.6** | AIC 单边偏差 |
| **v7 OOS** | 129.7% | 25.1% | 11.8% | **11.0** | AIC 严重偏差 |
| **v8 OOS** | **7.0%** | **8.0%** | **8.4%** | **1.0** ✓✓ | **完美 — 全 component < 10%** |

---

## 2. 抵消的具体证据（v6 in-training）

| Config | v6 AIC err | v6 AIV err | v6 wall err |
|---|---:|---:|---:|
| BERT-base-S128-b1 | +37% (over) | −33% (under) | 0.0% |
| GPT-2-S512-b1 | +16% (over) | −61% (under) | 1.4% |
| Qwen3-prefill-S256-b1 | +51% (over) | −56% (under) | 0.1% |
| Qwen3-prefill-S4096-b1 | +83% (over) | −90% (under)* | 0.0% |
| Qwen3-prefill-S512-b4 | +55% (over) | −40% (under) | 0.0% |
| ModernBERT-base-S4096-b1 | −34% (under) | −7% (under) | 0.0% |

*= 推测，v6 fit 把 wall 拟到 0% 时 AIV 大幅 under。

**模式**：AIC 系统性 over-predict、AIV 系统性 under-predict，**误差方向相反、量级相近**，让 wall_clock 看起来完美。

---

## 3. 根因：fit objective 只看 wall_clock

`fit_v6.py::_bucket_loss` 计算：

```python
def obj(theta):
    return mean([abs(pred_wall - meas_wall) / meas_wall for cfg in bucket])
```

DE optimizer 找的是 **wall MAE 最小**，**不是 (AIC MAE + AIV MAE + n_kern MAE) 最小**。所以 fit 可以选择：

```
[amp_aic=47.7% high, amp_aiv=45.7% low]
→ AIC over-predict + AIV under-predict
→ 相加给出 wall 误差 0%
✓ TRAIN MAE 闸门通过
```

而**对的 fit objective** 应该惩罚 component 偏离：

```python
def obj_v8(theta):
    aic_err = mean([err(pred_aic, meas_aic)])
    aiv_err = mean([err(pred_aiv, meas_aiv)])
    nk_err  = mean([err(pred_nk, meas_nk)])
    wall_err = mean([err(pred_wall, meas_wall)])
    return wall_err + 0.3 * (aic_err + aiv_err + nk_err)  # 加权惩罚
```

---

## 4. 这个抵消引发的实际问题

### 4.1 Architecture sweep 不可信

`prism-sweep` 改 HBM BW 时，会影响 `aic_mte2_us`（HBM 流量）→ 影响 `aic_time_us` → 影响 `wall_clock`。但 v6 的 AIC 部件本身被 fit 扭曲了 47.7%，所以 sweep 看到的"HBM BW × 2 → wall_clock 改变量"是**两层扭曲叠加**：

- 真物理：HBM BW × 2 → aic_mte2 减半 → aic_time 减少 X μs
- v6 模型：HBM BW × 2 → aic_mte2 base 减半 → × amp_aic 47% 偏差 → 实际 sweep 结果误差 ~50%

**结论：v6 的 sweep 数字对架构师而言不可信**，因为 AIC 内部各 sub-pipe 的相对比例被 amp 拉扯过。

### 4.2 Bottleneck 诊断错误

v6 输出 `aic_dominant_pipe: mte2` 或 `aic_dominant_pipe: fixpipe`。但因为整个 AIC 被 amp 扭曲 47%，sub-pipe 之间的相对值可能完全反转。

**例**：Qwen3-prefill-S4096-b1 在 baseline 测试中 dominant = fixpipe（FixPipe writeback bound）。v6 预测可能因为 amp 把 mte2 推高 → 错误诊断为 mte2 dominant。

### 4.3 Confidence label 误导

v6 的 confidence 基于 bucket + LOMO wall MAE，对 component 一无所知。一个 "high (AIV_BOUND bucket: 12-26% wall err)" 的 label 可能藏着 AIC 60% error，但用户看到 "high" 就信了。

---

## 5. v8 候选改进（用户 mandate 直接驱动）

### 5.1 Multi-objective fit

```python
def obj_v8(theta, λ_aic=0.3, λ_aiv=0.3, λ_nk=0.2):
    # All errors on percentage scale (0-100)
    return (
        wall_mae(theta)
        + λ_aic * aic_mae(theta)
        + λ_aiv * aiv_mae(theta)
        + λ_nk  * nkernels_mae(theta)
    )
```

`λ` 是 trade-off：高 λ → component 准但 wall 可能稍差，低 λ → 现状（wall 完美 component 任意）。建议先 λ=0.3 开始。

### 5.2 Component-MAE 硬门禁测试

新增测试（本 commit）：

```python
def test_component_mae_bounds():
    """Each version × split: per-component MAE within published bound.
    Catches future fit regressions that further exploit cancellation."""
    BOUNDS = {
        'v6_TRAIN': {'wall': 5,   'aic': 60,  'aiv': 60,  'nk': 80},
        'v6_OOS':   {'wall': 30,  'aic': 100, 'aiv': 30,  'nk': 50},
        # v8 target: all < 30%
    }
```

### 5.3 文档化"cancellation_ratio"指标

每次 fit 后输出该 ratio，> 10 即报警："警告：fit 可能依赖 component 误差互相抵消"。

---

## 6. v8 实现结果（用户 mandate 已达成）

`fit_v8.py` 落地 multi-objective loss：

```python
loss = wall_mae + 0.3 × aic_mae + 0.3 × aiv_mae + 0.2 × n_kern_mae
```

**v8 on OOS（4 个真机外推 configs）**：

| Component | v6 | v7 | **v8** | v8 vs v6 |
|---|---:|---:|---:|---|
| AIC | 56.9% | 129.7% | **7.0%** | 8.1× 改进 |
| AIV | 6.4% | 25.1% | **8.0%** | 持平 |
| n_kern | 11.4% | 95.9% | **2.8%** | 4× 改进 |
| wall | 10.1% | 11.8% | **8.4%** | 微改进 |
| **cancellation ratio** | 5.6 | 11.0 | **1.0** | **5.6× 改进** ✓ |

→ **v8 OOS 全 component < 10%**，cancellation_ratio = 1.0 = 完美。

**Trade-off on TRAIN**：v8 放弃了 v6 的 0.2% wall MAE（靠 47% AIC + 46% AIV 抵消），换来：
- TRAIN AIV 25.6%（v6: 45.7%）— 改进 1.8×
- TRAIN n_kern 37.6%（v6: 67.4%）— 改进 1.8×
- TRAIN cancellation_ratio 2.1（v6: 204.7）— **97× 改进**
- TRAIN wall 20.9%（vs v6: 0.2%）— 这是合理代价，因为 wall=0.2% 是假象

**v8 per-bucket fitted theta**：

| Bucket | amp_aic | amp_aiv | nk_mult |
|---|---:|---:|---:|
| AIC_DECODE | 0.82 | 1.38 | 4.08 |
| AIV_BOUND | 1.04 | 4.43 | 5.95 |
| BALANCED | 1.06 | 1.50 | 3.00 |

注意：v8 的 amp_aic / amp_aiv 大幅靠近 1.0（"正确的 base"），不再依赖 amp 大幅扭曲来抵消。AIV_BOUND amp_aiv=4.4 反映 AIV 实际需要 ~4× scaling（合理物理意义）。

---

## 7. 推荐立即动作（本 session）

1. ✅ 本 finding doc — **DONE** (更新到 v8)
2. ✅ `test_component_mae_regression_bounds_v6` + `_v7` — 防回归
3. ✅ `test_v6_cancellation_ratio_flagged` — 记录 v6 已知坏状态
4. ✅ `test_v8_oos_all_components_under_30pct` + `test_v8_train_no_component_cancellation` — 硬门禁
5. ✅ `fit_v8.py` multi-objective — **完成**

---

## 8. v8 何时该用 / v6 何时该用

| 场景 | 推荐 | 为何 |
|---|---|---|
| **新模型预测 wall_clock**（用户主要 use case）| v8 | OOS 全 component < 10%，wall 8.4% |
| 架构 sweep（改 HBM BW / cm scale / L2）| v8 | AIC sub-pipe 比例诚实，sweep 结果可信 |
| Bottleneck 诊断（`aic_dominant_pipe`）| v8 | AIC 各 sub-pipe 不被 amp 扭曲 |
| 仅需 wall_clock + 已知 model 在 TRAIN 集 | v6 | v6 TRAIN wall 0.2%（in-distribution 完美）|
| Issue #2 legacy 比对 | v4-v7 全部保留 | 向后兼容 |

---

## 7. TL;DR

**用户主张 "v6 减小了 AIV 误差" 在 OOS 上属实（473% → 6.4%），但 v6 TRAIN 的 AIV 实际更差（4.9% → 45.7%），靠 AIC 反向偏差 47.7% 抵消让 wall 看起来 0.2%**。

这是 fit objective 只优化 wall 的副作用。**v4 在 TRAIN component 上最诚实**（cancellation ratio=3.0），**v6 在 OOS wall 上最强**（vs Llama 1156% → 27%）。两者 trade-off 当前没有理论 winner。

**用户的诉求"泛化强 + 各 component 准"需要 v8 改 fit objective**（multi-objective），不是单纯调 amp。

测试会固化这个 audit 结果为可重复 regression gate。
