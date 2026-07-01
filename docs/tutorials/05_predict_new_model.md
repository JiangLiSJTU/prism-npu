# 教程 5：给新模型预测 pipe baseline（无 msprof）

如果你想用 `prism-sweep` 或 `prism-ceiling` 分析一个**没有 msprof 数据**的新模型（比如 HuggingFace 上刚 released 的 SOTA），本教程用 ModernBERT 走一遍完整流程。

> **前提**：你已经 `pip install -e ".[dev]"`，并能跑通 `prism-render --check` 。

---

## 0. 何时该用 PredictPipe vs Tier 3 实测

| 你的需求 | 用 | 原因 |
|---------|----|-----|
| 快速估算新模型在 910B4 的 wall-clock 量级 | **本教程** | 30 秒搞定，confidence 标签提示可信度 |
| 决定要不要花 1 天跑 NPU msprof | **本教程** | 先看 `confidence=high` 还是 `low` |
| 拍板芯片投资 / 写论文 | **Tier 3 真机** | 见 `03_recalibrate_with_new_msprof.md` |

---

## 1. 从 HuggingFace config 写 model YAML

去 model card 找 `config.json`（或本地 `~/.cache/huggingface/...`）。以 ModernBERT-base 为例：

```json
{
  "hidden_size": 768,
  "intermediate_size": 1152,
  "num_hidden_layers": 22,
  "num_attention_heads": 12,
  "vocab_size": 50368,
  "architectures": ["ModernBertForMaskedLM"],
  "hidden_activation": "gelu",
  "max_position_embeddings": 8192
}
```

写成 `models/regime/<your_model>.yaml`：

```yaml
name: ModernBERT-base-S4096-b1
arch: encoder            # encoder | decoder
layers: 22

# Required by prism-regime（高层 Roofline 字段）
ops_b1: 902.0e9          # 总 FLOPs/inf (B=1)，可估算或留 null
bytes_total: 2.02e8      # HBM 总流量字节，可估算或留 null

# Required by prism-predict-pipe（GEMM 级字段）
gemm_spec:
  S: 4096                # sequence length（decode 时填 1）
  d_model: 768           # = hidden_size
  d_ff: 1152             # = intermediate_size（GLU 时也填这个，不是 2×）
  n_heads: 12            # = num_attention_heads
  n_kv_heads: 0          # MHA: 0; GQA: KV head 数（如 SmolLM2 的 3）
  d_head: 64             # = hidden_size / num_attention_heads
  vocab: 50368           # = vocab_size
  ffn_type: glu          # standard | glu | swiglu
```

### 1.1 决定 `ffn_type`

| HuggingFace 模型 | ffn_type |
|----------------|----------|
| BERT、GPT-2、T5 | `standard` |
| Llama、Mistral、Qwen3、SmolLM2 | `swiglu` |
| ModernBERT、PaLM | `glu` |

如果你的模型 FFN 含两个并列投影 + element-wise gate，就是 `glu` 或 `swiglu`。

### 1.2 决定 `n_kv_heads`（GQA / MQA）

| 类型 | `n_kv_heads` |
|------|-------------|
| MHA（标准多头）| `0`（让公式自动用 `n_heads`）|
| GQA（如 Qwen3 16Q/8KV，SmolLM2 9Q/3KV）| KV 头数 |
| MQA（所有 head 共享一个 KV）| `1` |

---

## 2. 拟合 interaction 常数（一次性）

仅当你新增了 msprof PipeUtilization 数据时才需要重跑：

```bash
prism-predict-pipe --refit-params
# → data/calibration/predict_pipe_params.json
#   K0=1.86 μs/kernel, H_prefill=13424 μs, H_decode=204 μs
#   training: host_gap MAE=8.4%, kernel_gap MAE=14.3% (n=9)
```

repo 已自带拟合好的 `predict_pipe_params.json`（v4 默认），**首次使用不需要这一步**。

### 2.1 可选参数文件（v4-v8，**强烈推荐 v8**）

repo 自带 5 套拟合参数，按 OOS 准确度排序：

| 文件 | 模型版本 | TRAIN MAE wall | OOS MAE wall | OOS 各 component | Llama err | 推荐场景 |
|---|---|---:|---:|---|---:|---|
| `predict_pipe_params.json` | v4（默认） | 4.9% | ~362% | AIV 473% | **+1156%** | 仅 in-distribution 模型 |
| `predict_pipe_params_v5.json` | v5 | 17.3% | 87.8% | AIV 100% | +232% | bounded extrapolation |
| `predict_pipe_params_v6.json` | v6 | **0.2%**(虚) | 10.1% | **AIC 57%**(抵消)| +27% | 历史 baseline (eager) |
| `predict_pipe_params_v7.json` | v7 (SDPA) | 18.0% | 11.8% | AIC 129% (抵消) | +12% | SDPA path 早期 |
| **`predict_pipe_params_v8.json`** | **v8（推荐）** | 20.9% | **8.4%** | **全 < 10%** | **+8.4%** | **生产首选** |

**为什么 v8 推荐**：

- **OOS 全 component < 10%**（AIC 7%, AIV 8%, n_kern 3%, wall 8%）— 没有部件互相抵消
- **cancellation_ratio = 1.0**（v6 是 204 ❌，v7 是 11 ❌）— 部件诚实
- **SDPA-aware** baseline（production 默认路径，非 eager）

详见 `docs/methodology/08_predict_pipe.md §14` (v7) 和 `§15` (v8) + `docs/findings/predict_pipe_component_cancellation_audit.md`。

**使用 v8**：所有 `prism-predict-pipe` 命令加 `--params data/calibration/predict_pipe_params_v8.json`（见下方示例）。

> ⚠️ **v6 警告**：v6 TRAIN wall MAE 0.2% 是 **AIC 47% + AIV 46% 互相抵消的产物**，部件数字不可信。仅适合 in-distribution wall_clock 比对，**不能**做 bottleneck 诊断 / 架构 sweep。

---

## 3. 预测（推荐用 v8）

```bash
prism-predict-pipe \
    --model models/regime/modernbert_base_prefill_S4096.yaml \
    --arch  arch/ascend_910b4_for_sweep_v2.yaml \
    --params data/calibration/predict_pipe_params_v8.json \
    --output data/calibration/predict_pipe_modernbert.json
```

输出（v8 multi-objective，with bucket info）：

```
Predicted ModernBERT-base-S4096-b1 → data/calibration/predict_pipe_modernbert.json
  bucket     = AIV_BOUND               # v7/v8 自动分桶（3 桶）
  aiv_model  = multi_obj_v8_AIV_BOUND  # v8 标签
  aic_time   =     53,882 μs  (dominant: mte2)
  aiv_time   =    252,877 μs
  kernel_gap =      2,670 μs (1,439 kernels × K0)
  host_gap   =     13,424 μs
  wall_clock =    322,854 μs
  confidence = high (AIV_BOUND bucket: validated component-honest fit, OOS all < 10%)
```

**读这个输出**：
- **`bucket: AIV_BOUND`** → v8 自动识别瓶颈（3 桶分类）
- **`aiv_model: multi_obj_v8_*`** → multi-objective fit 标签（v6 是 `per_bucket_v6_*`，v7 是 `sdpa_aware_v7_*`）
- **`aic_dominant_pipe: mte2`** → AIC 内部 HBM 流量主导（**v8 这个判断是可信的**——AIC sub-pipe 比例未被 amp 扭曲）
- **`confidence: high (AIV_BOUND...)`** → v8 在该桶上 OOS wall 8.4%、AIC 7.0%、AIV 8.0%、n_kern 2.8% 全部 < 10%

### 3.1 自动 bucket 检测（v7/v8 = 3 桶简化版）

v7/v8 用 `spec.S, spec.d_model, batch` 三字段启发式判桶（比 v6 简化，删了 AIC_QWEN3）：

```
spec.S == 1                              → AIC_DECODE
spec.d_model ≥ 700 AND S × batch ≥ 1024  → AIV_BOUND
default fallback                          → BALANCED
```

为什么 v7/v8 删 AIC_QWEN3？因为 v6 的 AIC_QWEN3 桶是 **eager attention dispatch artifact** 而非 Qwen3 架构属性——用 SDPA 跑 Qwen3-prefill-S4096，wall 从 3050ms 降到 453ms (6.7× 加速)，AIV/AIC ratio 从 1.19 跳到 3.48 干净进入 AIV_BOUND 桶。详见 `docs/findings/predict_pipe_sdpa_breakthrough.md`。

新模型 YAML 写完后，跑一次 `prism-predict-pipe` 看输出 `bucket` 字段。**如果发现 bucket 异常**（比如 1B-级 LLM 被判 BALANCED），检查 d_model / layers 字段是否对。

---

## 4. 喂入 sweep / ceiling

### 4.1 单独跑（推荐用于探索）

```bash
prism-sweep --pipe-baseline data/calibration/predict_pipe_modernbert.json
```

> 这种方式不会修改 repo 里 commit 的 `pipe_baseline_per_model.json`，
> 适合临时探索，不污染权威数据。

### 4.2 合并入主 baseline（用于持久化）

```bash
prism-predict-pipe \
    --model models/regime/modernbert_base_prefill_S4096.yaml \
    --arch  arch/ascend_910b4_for_sweep_v2.yaml \
    --merge-into data/calibration/pipe_baseline_per_model.json
```

> ⚠️ 这会**就地修改**主 baseline 文件。建议在新 branch 上做，或保留 git 备份。

---

## 5. 置信度解读（v6 bucket-aware）

v8 confidence labels 基于 bucket + OOS 实测验证（multi-objective fit，component honest）：

| 标签 | Bucket | OOS wall MAE | 适用场景 | 建议 |
|------|------|---:|---|-----|
| **high (AIV_BOUND)** | 大 decoder prefill / 大 encoder | **8.4%** (eager) / **13.7%** (SDPA) | Llama/Qwen2.5/SmolLM2/ModernBERT/任何 d_model>=700 的大模型 | **可直接用于 sweep / 架构分析 / bottleneck 诊断**（component 数字都可信）|
| **high (AIC_DECODE)** | decoder decode (S=1) | 8.6% (n=1) | 新 decode workloads | 多采 1-2 个 decode 配置后才稳 |
| **high (BALANCED)** | 小/浅 / 单 batch | 3.9% | BERT/GPT-2-class | 单一配置准确，extreme batch 外推注意 |

**v8 完整 5-split 报告**：

| Split | n | wall | AIC | AIV | n_kern | cancel ratio |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 7 | 20.9% | 43.7% | 25.6% | 37.6% | 2.1 |
| OOS (eager) | 4 | **8.4%** | **7.0%** | **8.0%** | **2.8%** | **1.0** |
| VAL_SDPA_OOS (Phase 3) | 4 | 13.7% | 17.9% | 33.5% | 1.6% | 2.44 |
| VAL_SDPA_long_S | 1 | 23.8% | 3.1% | 26.5% | 0.1% | – |
| VAL_SDPA_batch | 2 | 19.6% | 13.3% | 30.2% | 0.1% | – |

→ **v8 在 5 个 split 上 wall MAE 全部 < 25%，cancellation ratio 全部 < 3**。

### 5.1 何时不能用 PredictPipe

v8 设计目标：新模型 wall_clock 预测 OOS < 25% + 各 component 不抵消。但下述场景仍需 Tier 3 真机验证：

- **TCO/采购决策**（百万级 $）：预测误差 8% 也可能误导 → 跑 msprof
- **bucket 边界 case**：spec 同时满足多个 bucket 条件 → 用 `--params` 试 v4-v8 五套结果对比
- **跨芯片**：v8 拟合在 910B4，移植 310P / Xeon AMX 需新一轮校准
- **跨 family 真机未验证**：Gemma / Mistral / Phi 当前仅 spec-level 测试（Issue #5）
- **极端 batch / 极长 S**：超出训练集 batch ≤ 8 或 S ≤ 4096 的范围 → 真机验证

详细的置信度判定逻辑见 [`methodology/08_predict_pipe.md §5`](../methodology/08_predict_pipe.md#5-置信度判定srcprismpredict_pipepredict.py-assign_confidence)。

---

## 6. 常见陷阱

### 6.1 GLU FFN 的 `d_ff` 怎么填？

填 `intermediate_size`，**不是** `2 × intermediate_size`。
公式内部会自动展开为 gate_proj + up_proj 两条 [S, d_model]→[S, d_ff] 路径。

### 6.2 decode 模式如何填？

`gemm_spec.S: 1`。下游公式会自动识别 `S==1` 并应用 `H_decode = 204 μs`（而不是 `H_prefill = 13424 μs`）。

### 6.3 我的模型有 RoPE / ALiBi / RWKV 风格

`predict_pipe` 当前**忽略位置编码的 vector op**（量级 < 5%）。RWKV 风格的线性注意力**不支持**——架构差异太大，公式不适用。

### 6.4 ModernBERT alternating local/global attention

`predict_pipe` 假设 full attention（O(S²) ops）。Local attention（窗口 W）实际更便宜约 (S/W) 倍。模型若大量使用 local attention，预测会**高估** aic_mac。当前 ModernBERT prediction 的 `confidence=low` 部分反映了这一点。

---

## 7. 验证你的预测

如果有 NPU access，用 `benchmark/run_new_models_msprof.sh` 跑一遍真机，对比：

```bash
# 在 910B4 服务器上
bash benchmark/run_new_models_msprof.sh

# 拉回本地
rsync -avz user@npu:~/sim-experiment/msprof_modernbert_*/ msprof_data/

# 提取实测 pipe baseline
prism-extract --output data/calibration/cube_util_modernbert.json
# 比较实测 vs predict_pipe_modernbert.json 的 wall_clock_us 偏差
```

差异 > 30% 通常意味着：
- `aic_mte2` 公式不适配该模型（大模型 / 特殊 tiling 行为）
- `host_gap` 在该 regime 偏离常数
- 模型有未建模的特殊算子（Mamba 类、稀疏 attention 等）

这种情况下，将 msprof 实测数据加入 `pipe_baseline_per_model.json` + 重跑 `--refit-params`，未来同类模型的预测会自动改善。

---

## 8. 下一步

- 想理解公式推导细节：[`methodology/08_predict_pipe.md`](../methodology/08_predict_pipe.md)
- 想加新模型 YAML 模板：[`tutorials/04_add_new_model.md`](04_add_new_model.md)
- 想跑 sweep 探索架构敏感度：[`tutorials/02_reproduce_arch_sweep.md`](02_reproduce_arch_sweep.md)
- 想做 Tier 3 真机校准：[`tutorials/03_recalibrate_with_new_msprof.md`](03_recalibrate_with_new_msprof.md)
