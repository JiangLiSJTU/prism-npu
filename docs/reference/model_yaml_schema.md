# Model YAML Schema

`models/` 目录下每个 yaml 描述一个 transformer/CNN 模型的 Roofline / Timeloop 输入。当前覆盖：

| 模型 | 用途 | 派生 |
|------|------|------|
| `bert_base.yaml` | 固定网络业务 baseline encoder | HF BertConfig |
| `gpt2_small.yaml` | 固定网络业务 decoder-style | HF GPT2Config |
| `qwen3_0.6b.yaml` | LLM 长上下文 prefill 主战场 | HF Qwen3Config |
| `qwen3_embedding.yaml` | LLM Embedding 推理 | Qwen3-Embedding-0.6B |
| `net_transformer.yaml` | 固定网络最轻量 1-layer 模型 | 自定义 |
| `hf_bert.yaml` | HuggingFace 标准 BERT (4-layer 校准用) | HF |
| `dara_dqn.yaml` | 固定网络 RL agent | 自定义 |
| `regime/*.yaml` | regime gate 用的简化 model（仅含 ops/bytes 关键字段）| 派生 |

---

## 1. 完整字段表

### 1.1 模型架构（必填）

| 字段 | 类型 | 含义 |
|------|------|------|
| `name` | str | 模型名（与 sweep MODELS dict key 对应）|
| `hidden_size` | int | hidden dim |
| `intermediate_size` | int | FFN_hidden（SwiGLU 三路 / GeLU 双路 等）|
| `num_hidden_layers` | int | L 层数 |
| `num_attention_heads` | int | Q heads 数 |
| `num_key_value_heads` | int | KV heads 数（GQA 时 < num_attention_heads，否则相同）|
| `head_dim` | int | per head dim |
| `vocab_size` | int | 词表大小（影响 embed/lm_head MM）|

### 1.2 推理超参（必填）

| 字段 | 类型 | 含义 |
|------|------|------|
| `seq_len` | int | 默认 sequence length |
| `batch_size` | int | 默认 batch |
| `dtype` | str | "fp16"（默认）/ "fp32" / "int8" |

### 1.3 计算信息（roofline 关键）

| 字段 | 类型 | 单位 | 含义 |
|------|------|------|------|
| `ops_per_inference` | float | FLOPs | 单 inference 总 FLOPs |
| `bytes_per_inference` | float | Bytes | 单 inference 总访存量 |
| `arithmetic_intensity` | float | OPs/Byte | ops / bytes |

### 1.4 GEMM 算子描述（sweep / mapper 用）

```yaml
gemm_ops:
  - name: q_proj
    M: 4096        # per-batch
    N: 2048        # n_q_heads × head_dim
    K: 1024
    op_kind: BMM   # BMM 或 MM
    count_per_layer: 1
  - name: k_proj
    ...
```

`op_kind` 区分 BMM / MM 影响 effective_M 计算：BMM 内 batch 维独立 → M_eff = M_per_batch × B（除 attention head op 不乘）；MM 已 flatten batch → M_eff = M_per_batch。详见 [methodology/03_eta_real_model.md §4](../methodology/03_eta_real_model.md#4-op_kind-区分bmm-vs-mm)。

### 1.5 Vector 算子描述（pipe-aware sweep 用）

```yaml
vector_ops_per_layer: 1124073472   # 总 vector ops/layer (LayerNorm + Softmax + GeLU/SwiGLU)
vector_ops_breakdown:
  layernorm: 25165824   # 2 × S × hidden × 3 (mean+var+norm+scale, 这里取 3 ops/element)
  softmax: 1073741824   # heads × S² × 4 (max-sub+exp+sum+div)
  swiglu: 25165824      # S × FFN × 2 (silu × gate)
```

→ 工具实际只读 `vector_ops_per_layer`（标量）。`breakdown` 是文档化用，方便后续审视。

### 1.6 Roofline 校准（β 拟合输出，可选）

```yaml
roofline_calib:
  beta_layer_us: 560.4         # per-layer host gap (实测拟合)
  beta_device_us: 119          # device-level fixed overhead
  alpha_per_batch_us: 300      # per additional batch overhead
  eta_compute: 0.78            # eta_real for this model 的全模型平均
```

如未填，sweep 会用 model_class 的默认 β（详见 [methodology/02 §4.2](../methodology/02_three_layer_roofline.md#42-ols-拟合结果910b4-上)）。

---

## 2. 完整示例（Qwen3-0.6B）

```yaml
# Qwen3-0.6B 算子描述
name: Qwen3-0.6B

# 1. 架构
hidden_size: 1024
intermediate_size: 3072    # SwiGLU FFN
num_hidden_layers: 28
num_attention_heads: 16    # Q heads
num_key_value_heads: 8     # KV heads (GQA 2:1)
head_dim: 128
vocab_size: 151936

# 2. 推理超参
seq_len: 4096              # 默认 prefill
batch_size: 1
dtype: fp16

# 3. 计算信息
ops_per_inference: 137_600_000_000     # 137.6 G FLOPs/inf @ S=4096 b=1
bytes_per_inference: 1_192_000_000     # 1.192 GB/inf
arithmetic_intensity: 115.4

# 4. GEMM ops
gemm_ops:
  - name: q_proj
    M: 4096
    N: 2048    # 16 q_heads × 128
    K: 1024
    op_kind: BMM
    count_per_layer: 1
  - name: kv_proj
    M: 4096
    N: 1024    # 8 kv_heads × 128
    K: 1024
    op_kind: BMM
    count_per_layer: 2    # k 和 v 分别一次
  - name: o_proj
    M: 4096
    N: 1024
    K: 2048
    op_kind: BMM
    count_per_layer: 1
  - name: ffn_gate_up
    M: 4096
    N: 3072
    K: 1024
    op_kind: BMM
    count_per_layer: 2    # gate + up
  - name: ffn_down
    M: 4096
    N: 1024
    K: 3072
    op_kind: BMM
    count_per_layer: 1
  - name: attn_qk
    M: 4096
    N: 4096    # S
    K: 128     # head_dim
    op_kind: BMM
    count_per_layer: 1
  - name: attn_av
    M: 4096
    N: 128
    K: 4096
    op_kind: BMM
    count_per_layer: 1

# 5. Vector ops
vector_ops_per_layer: 1124073472
vector_ops_breakdown:
  rmsnorm: 25165824     # 2 × 4096 × 1024 × 3
  softmax: 1073741824   # 16 × 4096 × 4096 × 4
  swiglu: 25165824      # 4096 × 3072 × 2

# 6. Roofline 校准
roofline_calib:
  beta_layer_us: 560.4
  beta_device_us: 119
  alpha_per_batch_us: 300
  eta_compute: 0.78
```

---

## 3. 不要修改

| 文件 | 原因 |
|------|------|
| `models/*.yaml` 已 commit 版本 | sweep MODELS dict 与 yaml 一对一对应；改动需同步 sweep + 重 fit + 验证不变量 |

如要加新模型，**新建 yaml**，详见 [tutorials/04_add_new_model.md](../tutorials/04_add_new_model.md)。

---

## 4. 与 sweep MODELS dict 的关系

`src/prism/sweep/runner.py` 顶部 `MODELS` dict 是 yaml 的代码副本（避免运行时 yaml 解析开销）。两者不一致时：

- yaml 是 spec / 文档化版本（人读）
- code dict 是运行时版本（机器读）

理论上加新 model 后两者都要改，但当前 sweep 的 MODELS dict 是 source of truth（yaml 用于参考）。未来如要 driver-from-yaml，加 `prism.sweep.runner.load_models_from_yaml()`。

---

## 5. regime/ 子目录的简化版

`models/regime/*.yaml` 是 regime gate 工具用的极简模型描述（仅 layers / ops / bytes 关键字段）：

```yaml
name: BERT-base
layers: 12
ops_b1: 22_700_000_000      # 单 batch ops
bytes_total: 438_000_000    # 总字节
ops_decode_token: 0         # decode 模式 ops（encoder 无 decode）
```

→ `prism-regime --model models/regime/bert_base.yaml --arch ...` 用这个简化 schema。
