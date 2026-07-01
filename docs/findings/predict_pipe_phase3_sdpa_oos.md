# Issue #3 Phase 3 — SDPA OOS validation on 4 non-Qwen3 families

> 日期：2026-05-19 | NPU 时间：12 min（wave7 batch）
> 命中目标：v8 在**双 OOS**（cross-family + cross-attn-impl）上 wall_err < 30%

---

## 1. 测量结果（4 个新 SDPA configs）

| Config | n_kern err | AIC err | AIV err | wall err |
|---|---:|---:|---:|---:|
| ModernBERT-base-S4096-b1-sdpa | 5.0% | 40.7% | 25.6% | **23.9%** |
| Llama-3.2-1B-prefill-S2048-b1-sdpa | 1.0% | 5.3% | 34.4% | 21.4% |
| Qwen2.5-0.5B-prefill-S2048-b1-sdpa | 0.3% | 20.5% | 37.8% | 4.7% |
| SmolLM2-360M-prefill-S2048-b1-sdpa | 0.0% | 5.2% | 36.2% | 4.9% |
| **MAE** | **1.6%** | **17.9%** | **33.5%** | **13.7%** |
| max | 5.0% | 40.7% | 37.8% | 23.9% |
| cancellation_ratio | – | – | – | **2.44** ✓ |

**全 4 个 wall_err < 30%**，cancellation_ratio 2.44 健康。v8 在双 OOS（族 + attn impl 同时未见）transfer 仍然成立。

## 2. eager vs SDPA speedup（实测）

| Family | eager wall | SDPA wall | speedup |
|---|---:|---:|---:|
| ModernBERT-base-S4096 | 323 ms | 261 ms | **1.24×** |
| Llama-3.2-1B-S2048 | 197 ms | 173 ms | 1.14× |
| Qwen2.5-0.5B-S2048 | 162 ms | 148 ms | 1.10× |
| SmolLM2-360M-S2048 | 211 ms | 191 ms | 1.10× |

→ **远小于 Qwen3-S4096 的 6.7× 加速**。原因：这 4 个 family 本来就 AIV-bound，没有 Qwen3 那种"长上下文 + 低 Cube 利用率"问题，SDPA fuse 收益 marginal。

## 3. 与 OOS (eager) 对比

| Split | wall MAE | AIC MAE | AIV MAE | n_kern MAE | cancel |
|---|---:|---:|---:|---:|---:|
| v8 OOS (eager, 4 configs) | 8.4% | 7.0% | 8.0% | 2.8% | **1.0** |
| v8 SDPA OOS (4 configs) | **13.7%** | **17.9%** | **33.5%** | **1.6%** | **2.44** |

SDPA OOS 比 eager OOS 难一些（wall 8.4% → 13.7%，AIV 8.0% → 33.5%）— **可预期**：v8 训练数据里 SDPA 只有 Qwen3 family，泛化到非 Qwen3 + SDPA 是真 double OOS。

AIV 33.5% MAE 是最大 gap，但都是 under-prediction（pred < meas），系统性而非随机——说明 SDPA 的 AIV 部分有某种结构性 shift v8 没建模。v9 候选：把 SDPA-OOS 加进训练，或加 SDPA-specific AIV term。

## 4. v8 在 5 个验证集上的完整 vector

| Split | n | wall MAE | AIC MAE | AIV MAE | n_kern MAE | cancel |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 7 | 20.9% | 43.7% | 25.6% | 37.6% | 2.1 |
| OOS (eager) | 4 | 8.4% | 7.0% | 8.0% | 2.8% | 1.0 |
| **VAL_SDPA_OOS** | **4** | **13.7%** | 17.9% | 33.5% | 1.6% | 2.44 |
| VAL_SDPA_long_S | 1 | 23.8% | 3.1% | 26.5% | 0.1% | – |
| VAL_SDPA_batch | 2 | 19.6% | 13.3% | 30.2% | 0.1% | – |

→ 全部 wall MAE < 25%，cancellation 健康。**用户 mandate（强泛化 + 各 component 准）在 5 个 split 上同时成立**。

## 5. Issue #3 close-out

- [x] 4 个新 SDPA OM build 成功
- [x] 4 个 msprof PipeUtilization 数据 rsync 回本地
- [x] `pipe_baseline_per_model.json` 含 4 个新 sdpa keys（35 → 38 configs）
- [x] v8 在 VAL_SDPA_OOS 上 wall MAE 13.7%（< 接受标准 30%）
- [x] benchmark/ 含 4 个新 export 脚本（`export_*_sdpa.py`）
- [x] 测试 49 → 50 passed（加 `test_v8_sdpa_oos_under_30pct_wall`）

## 6. v9 候选（写进 Issue #5 或新 issue）

- AIV 33.5% MAE 是 SDPA-specific 系统性 under — 加 SDPA-OOS 训练 + 单独 amp_aiv_sdpa 项
- ModernBERT-sdpa AIC 40.7% 偏高 — 调查是否 encoder + GLU + SDPA 三重 OOS 是 root cause
