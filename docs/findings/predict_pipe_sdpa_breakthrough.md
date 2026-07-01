# PredictPipe — SDPA / FlashAttention 路径下的结构性发现

> 日期：2026-05-18 | 触发：Path B 验证 (Issue #2 follow-up)
> 输入：`Qwen3-prefill-S4096-b1` 用 `attn_implementation="sdpa"` re-export + ATC + msprof
> 工件：`models/qwen3_06b_prefill_S4096_b1_sdpa.onnx` + `om/qwen3_06b_prefill_S4096_b1_sdpa.om`
> msprof：`msprof_data/msprof_qwen3_06b_prefill_S4096_b1_sdpa_PipeUtilization/`

---

## 1. 6 Qwen3 prefill 配置 SDPA vs eager 全对比（2026-05-18 完整数据）

| Config | wall_eager | wall_sdpa | **speedup** | AIV/AIC eager | AIV/AIC sdpa |
|---|---:|---:|---:|---:|---:|
| Qwen3-prefill-S256-b1 | 78 ms | 82 ms | **0.95×** （慢 5%）| 1.57× | 2.30× |
| Qwen3-prefill-S256-b4 | 140 ms | 95 ms | 1.47× | 1.18× | 1.68× |
| Qwen3-prefill-S256-b8 | 225 ms | 111 ms | 2.03× | 0.94× | 1.32× |
| Qwen3-prefill-S512-b4 | 302 ms | 118 ms | 2.55× | 0.94× | 1.46× |
| Qwen3-prefill-S512-b8 | 604 ms | 190 ms | 3.18× | 0.89× | 1.95× |
| **Qwen3-prefill-S4096-b1** | **3050 ms** | **453 ms** | **6.74×** | 1.19× | **3.48×** |

### 1.1 speedup 与 S 强相关（典型 FlashAttention 曲线）

- **S=256**：speedup 0.95-2.03×。短上下文 attention 不主导，FA 微开销甚至略输 eager (b=1)
- **S=512**：speedup 2.55-3.18×
- **S=4096**：speedup **6.74×**（FA 收益最显著）

这正是 FlashAttention paper 报告的曲线特征：**FA 优化的是 O(S²) attention 部分**，S 越大收益越大。

### 1.2 详细单 config（Qwen3-S4096-b1）

| Metric | Eager | **SDPA** | 变化 |
|---|---:|---:|---:|
| n_kernels | 14,448 | **1,664** | **−88%** |
| aic_time | 1,331,319 μs | **101,130 μs** | **−92%** |
| aiv_time | 1,583,848 μs | **352,287 μs** | **−78%** |
| wall_clock | 3,050,000 μs | **~453,000 μs** | **−85%, 6.7× 加速** |
| AIV/AIC ratio | 1.19× | **3.48×** | shift |
| aic_dominant | fixpipe (55.6%) | **mte2 (HBM)** | 正常 prefill 形态 |

step_trace 实测：12 次 iteration latency 在 [452, 453] ms，stdev < 0.2 ms（极稳定）。

---

## 2. v6.1 桶分类的潜在 invalidation

v6.1 把 Qwen3-prefill 单独放在 `AIC_QWEN3` 桶（amp_aic=7.96, amp_aic_S_alpha=0.70, nk_mult=31）。

**新数据下 Qwen3-S4096-sdpa 的 AIV/AIC=3.48**，自动跑分类器：

```python
classify_bottleneck(qwen3_S4096_sdpa_spec, batch=1)
# Q: d_model=1024 ∈ [1000,1300] AND layers=28 AND swiglu AND decoder?
#    → 是 → 但 measured AIV/AIC=3.48 应该 → AIV_BOUND
```

→ **classifier 的 spec 启发式 still 把它放到 AIC_QWEN3**（因为 d_model 还是 1024），但**真实物理行为已经是 AIV_BOUND 类**。

这是个分类器 bug：spec 启发式过于 fragile，它在 measure 之前根本不知道 attention kernel 是 eager 还是 FA。

---

## 3. 6 configs 实测后：AIC_QWEN3 桶**部分**可去除

完整 SDPA 实测后的 AIV/AIC 分布：

| 配置 | AIV/AIC eager | AIV/AIC sdpa | 当前 v6.1 桶 | sdpa 下应归 |
|---|---:|---:|---|---|
| Qwen3-S256-b1 | 1.57 | 2.30 | AIC_QWEN3 | **BALANCED**（接近边界 2.5）|
| Qwen3-S256-b4 | 1.18 | 1.68 | AIC_QWEN3 | **BALANCED** |
| Qwen3-S256-b8 | 0.94 | 1.32 | AIC_QWEN3 | **BALANCED** |
| Qwen3-S512-b4 | 0.94 | 1.46 | AIC_QWEN3 | **BALANCED** |
| Qwen3-S512-b8 | 0.89 | 1.95 | AIC_QWEN3 | **BALANCED** |
| **Qwen3-S4096-b1** | 1.19 | **3.48** | AIC_QWEN3 | **AIV_BOUND** |

### 结论修订（vs Path B 初步推测）

之前推测 "AIC_QWEN3 桶整体可删，Qwen3 都漂到 AIV_BOUND" — **数据部分验证、部分反驳**：

- ✅ **AIC_QWEN3 桶仍可去除**：6 个 Qwen3-sdpa configs **自然散布到 BALANCED 和 AIV_BOUND**，没有一个需要"独立桶"
- ❌ 但不是都到 AIV_BOUND：短 S（256/512）Qwen3 落入 BALANCED（与 BERT/GPT-2 同桶），只有 S=4096 进 AIV_BOUND
- ✅ 仍验证："AIC_QWEN3 桶是 eager-CANN 病态产物" — SDPA 下 Qwen3 行为正常化，符合 (S, batch) 决定的瓶颈分布

### v7 简化路径

```python
def classify_bottleneck_v7(spec, batch, attn_impl="sdpa"):
    """v7: 默认假设 SDPA path (production-aligned)."""
    if spec.S == 1:
        return "AIC_DECODE"
    if attn_impl == "eager" and is_qwen3_family(spec):
        return "AIC_QWEN3"     # legacy fallback for non-SDPA Qwen3
    if spec.d_model >= 700 and spec.S * batch >= 1024:
        return "AIV_BOUND"
    return "BALANCED"
```

**默认 SDPA → 3 桶**（AIC_DECODE, AIV_BOUND, BALANCED）；只有显式声明 `attn_impl="eager"` + Qwen3-family 才走 AIC_QWEN3 路径（兼容旧 baseline）。

---

## 4. PRISM 建模的"层错位"问题

v6.1 在拟合一个**错误的 baseline**——它把 CANN 的低效 eager attention dispatch 当成了"Qwen3 的固有特性"，于是动用复杂参数（5 free params × 4 buckets）去拟合。

而 **production 部署都会用 FlashAttention / SDPA**（包括 厂商 ModelZoo 的 Qwen3 优化版）。所以 PRISM 应该建模在 SDPA path 上：

| 错位 | 后果 |
|---|---|
| 建模 eager wall=3050ms | 给 sweep 工具一个**严重悲观**的 baseline |
| AIC_QWEN3 桶 amp_aic=8 | 工具 sweep 时"加 cube"看似没用（因为模型本来不缺 cube） |
| 真实 AIC_QWEN3 应该用 SDPA wall=453ms | sweep 工具应得到"加 HBM bandwidth 有效"的正确指引 |

**对芯片架构 sweep 而言，PRISM v6.1 给出的 "Qwen3 不 cube-bound" 结论可能误导**：production Qwen3 跑在 FA path 上其实是 HBM/mte2 bound。

---

## 5. v7 plan（待用户确认）

### Phase 1（必做 — 验证假设）：4 小时 NPU

1. 把 SDPA export script 推广到剩下的 5 个 Qwen3 prefill 配置（S256-b1/b4/b8, S512-b4/b8）
2. ATC + msprof
3. 看是否 AIV/AIC 都 shift 到 [2.5, 5] 范围
4. 如果是 → AIC_QWEN3 桶消失，v7 简化为 3 桶

### Phase 2（如 Phase 1 验证）：refit v7

5. 把 7 个 Qwen3-sdpa 配置加入 baseline JSON
6. 删除 AIC_QWEN3 桶 + amp_aic_S_alpha / amp_aiv_S_alpha
7. refit v7：4 桶 → 3 桶，5 params → 3 params per bucket
8. LOMO 重跑

### Phase 3（建议但非阻塞）：扩展到所有 transformer family

9. 所有 ModernBERT / Llama / Qwen2.5 / SmolLM2 也用 sdpa re-export
10. 检查是否 baseline 全部下移（预计 ~2-3× 加速）
11. arch yaml + sweep tool baseline 全面 refresh

### Phase 4（可选 — v8 path）：用 `torch_npu.npu_fusion_attention` 直接 PyTorch-NPU 跑

如果 SDPA→ATC fuse 的效率不是 100%（比如只达到 60% FA paper speedup），Path C 可以再榨一波。

---

## 6. 立即决策点

| 选项 | 工作量 | 价值 |
|---|---|---|
| A. 只 commit 当前 finding + 把 SDPA-S4096 数据加进 baseline | 15 min | 文档化新发现，不动 v6.1 |
| B. **Phase 1 完整跑 SDPA 6 configs + 看是否 AIC_QWEN3 消失** | 4 h NPU | **强证据** AIC_QWEN3 桶真否需要 |
| C. A + B + Phase 2 (refit v7) | 6 h | **v7 完整落地** |
| D. C + Phase 3 (全 family sdpa) | 12 h | full production-accurate baseline |

---

## 7. 对照保留政策（重要——用户 2026-05-18 mandate）

**不覆盖 eager 结果**。所有 SDPA 实测以**新 key**追加进 baseline JSON，eager 原始数据完整保留：

| 实验前（eager 路径）| 实验后（SDPA 路径）| 关系 |
|---|---|---|
| `Qwen3-prefill-S4096-b1` | `Qwen3-prefill-S4096-b1-sdpa` | 并列 |
| `Qwen3-prefill-S256-b1` | `Qwen3-prefill-S256-b1-sdpa` | 并列 |
| `Qwen3-prefill-S256-b4` | `Qwen3-prefill-S256-b4-sdpa` | 并列 |
| `Qwen3-prefill-S256-b8` | `Qwen3-prefill-S256-b8-sdpa` | 并列 |
| `Qwen3-prefill-S512-b4` | `Qwen3-prefill-S512-b4-sdpa` | 并列 |
| `Qwen3-prefill-S512-b8` | `Qwen3-prefill-S512-b8-sdpa` | 并列 |

后续报告、commit message、对比表都**显式标注 eager / SDPA 两个数字**。理由：
- **审计可重现**：未来如有人质疑某个 number，能直接查到原始 msprof 来源
- **建模演化可追溯**：v7 refit 用 SDPA 数据，但 v6 / v6.1 拟合用的是 eager —— 两套数据各自的 finding doc 与实测对应，不会混乱
- **"production vs reference" 双 baseline**：SDPA 是 production-aligned 路径，eager 是 PyTorch-naive 路径；芯片架构 sweep 工具可根据 deployment target 选基线
- **不能事后推翻**：如果 ATC / CANN 版本更新后 SDPA fuse 行为变化，旧 eager 数据仍是不变的 reference

工件命名规范（所有 SDPA 文件）：
- ONNX：`models/qwen3_06b_prefill_S{S}_b{B}_sdpa.onnx`
- OM：`om/qwen3_06b_prefill_S{S}_b{B}_sdpa.om`
- msprof：`msprof_data/msprof_qwen3_06b_prefill_S{S}_b{B}_sdpa_PipeUtilization/`
- baseline key：`Qwen3-prefill-S{S}-b{B}-sdpa`
- baseline source：`msprof_PipeUtilization_measured_SDPA`（区别于 `..._measured`）

---

## 8. TL;DR

- ✅ Path B 成功：SDPA + opset 17 + ATC 给出 **6.7× 加速**（3050→453 ms）
- ❌ v6.1 的 AIC_QWEN3 桶可能是**建模在错误的 CANN 配置上**的伪 calibration
- 🚧 v7 plan：先验证假设（4h NPU），如成立则 refit 简化版（3 桶 vs 4 桶）
- 💡 更深层 insight：**PRISM 应该建模 production path（SDPA / FA），不是 PyTorch eager**
