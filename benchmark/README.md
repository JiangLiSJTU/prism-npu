# benchmark/ — NPU 实验脚本

本目录是 **Tier 3（重新校准）** 的脚本集，用于在 Ascend 910B/910B4 NPU 上重新采集 msprof 数据。普通用户（Tier 1 预测）不需要进入此目录；calibration 数据已在 `data/calibration/` 中预置。

> **硬件依赖**：Ascend 910B 或 910B4 NPU + CANN 8.5+（含 ATC + msprof）+ HuggingFace 模型缓存
> **软件依赖**：Python 3.9+, `torch>=2.5`, `transformers>=4.51.0`, `ais_bench`

---

## 完整工作流（5 步）

```
Step 1: HuggingFace → ONNX        (本地或 NPU 服务器，纯 Python)
Step 2: ONNX → OM                 (NPU 服务器，CANN ATC)
Step 3: OM × msprof               (NPU 服务器，4 个 metric)
Step 4: msprof CSV → 本地         (rsync)
Step 5: CSV → calibration JSON    (本地，prism-extract / prism-fit)
```

---

## 脚本一览

### ONNX 导出（任何机器，纯 Python）

| 脚本 | 用途 | 备注 |
|------|------|------|
| [`export_models_onnx.py`](export_models_onnx.py) | 导出 ET-BERT / NetGPT / MalConv2 / Kitsune 等固定网络场景模型 | 单文件批量 |
| [`export_bert_gpt2_qwen.py`](export_bert_gpt2_qwen.py) | 导出 BERT-base / GPT-2-small / Qwen3-0.6B 标准 LLM | HuggingFace 模型 |
| [`export_qwen3_prefill.py`](export_qwen3_prefill.py) | Qwen3-0.6B prefill 模式，参数化 `--S` `--batch` | 含 BOOL→int32 cumsum 补丁 |
| [`export_qwen3_decode.py`](export_qwen3_decode.py) | Qwen3-0.6B decode 模式 | KV cache 输入 |
| [`export_net_transformer.py`](export_net_transformer.py) | Net-Transformer（固定网络自定义 1-layer） | — |

### ATC 转换（NPU 服务器，需 CANN）

| 脚本 | 用途 | 备注 |
|------|------|------|
| [`convert_onnx_to_om.sh`](convert_onnx_to_om.sh) | 通用 ATC 批量转换，覆盖 BERT/GPT-2/Qwen3 固定网络场景模型 | 用法：`bash convert_onnx_to_om.sh <onnx_dir> <om_dir>` |
| [`convert_qwen3_prefill_om.sh`](convert_qwen3_prefill_om.sh) | Qwen3 prefill 多 (S, B) 组合 ATC 转换 | 用法：`bash convert_qwen3_prefill_om.sh <onnx_dir> <om_dir> "<S_list>" "<B_list>"` |

### 推理 + msprof 采集（NPU 服务器）

| 脚本 | 用途 | 备注 |
|------|-------|------|
| [`run_inference_ascend.py`](run_inference_ascend.py) | OM 推理 runner（msprof `--application` 调用此脚本） | 基于 `ais_bench` |
| [`run_qwen3_msprof_full.sh`](run_qwen3_msprof_full.sh) | Qwen3-0.6B b=1/4/8 × 4 metric 全采集（已存在则跳过）| `LOOP=50 WARMUP=10`，标准 prefill 配置 |
| [`run_phase_b.sh`](run_phase_b.sh) | 批量 msprof 采集（10 轮）模板 | 按 model 调整 |
| [`run_pipeutil_supplement.sh`](run_pipeutil_supplement.sh) | 补采 PipeUtilization（8 个 prefill 配置） | 含 loop 数衰减策略（5/2/1）应对长上下文崩溃 |

### 解析（本地，纯 Python）

| 脚本 | 用途 | 备注 |
|------|-------|------|
| [`parse_msprof.py`](parse_msprof.py) | 解析 msprof CSV → per-model 物理分解 JSON | 也含 Cube/Vector/MTE 算子分解 |
| [`parse_timeloop_stats.py`](parse_timeloop_stats.py) | 解析 Timeloop `stats.txt`（Tier 2 用） | — |

---

## Tier 2 — Timeloop Docker 配置

`prism-mapping` 用 Timeloop 验证 Cube cycles。需 Docker 镜像：

```bash
# 一次性拉取
docker pull accelergy/timeloop-accelergy-pytorch:latest

# 验证可用
docker run --rm accelergy/timeloop-accelergy-pytorch:latest timeloop-mapper --version
```

**关键 gotcha**（来自 Phase G2 经验）：
- 启动容器需 `--entrypoint /bin/bash`（默认 entrypoint 是 jupyter，会吃掉参数）
- `arch/` 子目录需含 3 个 component yaml（`SRAM.yaml`, `MAC.yaml`, `CACTI.yaml`），否则 Accelergy 报 "Could not find area for smartbuffer_sram"
- 工具已封装这些，直接用 `prism-mapping` 即可

完整 Manual Mapping workflow 见 [docs/methodology/05_calibration.md §6](../docs/methodology/05_calibration.md#6-msprof-失败模式--应对)。

---

## Tier 3 — Ascend NPU 校准 workflow

### Step 1 — 导出 ONNX（任何机器）

固定网络场景模型：
```bash
python benchmark/export_models_onnx.py
# → models/{et_bert,netgpt,malconv2,kitsune}_*.onnx
```

Qwen3 prefill（参数化）：
```bash
python benchmark/export_qwen3_prefill.py --S 4096 --batch 1
python benchmark/export_qwen3_prefill.py --S 256  --batch 1 4
# → models/qwen3_06b_prefill_S{S}_b{B}.onnx
```

### Step 2 — ONNX → OM（NPU 服务器，CANN ATC）

通用批量转换（覆盖 BERT / GPT-2 / Qwen3 / 固定网络场景模型）：

```bash
# 在 NPU 服务器上
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash benchmark/convert_onnx_to_om.sh ~/sim-experiment/models ~/sim-experiment/om
# → ~/sim-experiment/om/{hf_bert_batch_*,gpt2_*,qwen3_06b_b*}.om
```

Qwen3 prefill 参数化转换（多 S 多 B 组合）：

```bash
bash benchmark/convert_qwen3_prefill_om.sh \
     ~/sim-experiment/models \
     ~/sim-experiment/om \
     "256 512 4096" \
     "1 4"
```

两个脚本默认 `--soc_version=Ascend910B4`，如要换硬件改 `SOC` 变量。

### Step 3 — msprof 采集（NPU 服务器，4 metric × N 配置）

**Qwen3-0.6B b=1/4/8 全套**（4 metric × 3 batch = 12 个目录，~25 min NPU）：

```bash
bash benchmark/run_qwen3_msprof_full.sh
# → msprof_qwen3_06b_b{1,4,8}_{PipeUtilization,ArithmeticUtilization,L2Cache,Memory}/
```

**Qwen3 prefill 长上下文补采**（loop 衰减策略，应对 S≥4096 + B=8 崩溃）：

```bash
bash benchmark/run_pipeutil_supplement.sh
# 8 个 prefill (S × B) 配置，loop 10/5/2/1 阶梯下降
```

**手工单跑**（参考 [`run_phase_b.sh`](run_phase_b.sh) 模板）：

```bash
BASE_FLAGS="--task-time=on --ai-core=on --aicpu=on --runtime-api=on --hccl=off"
msprof --application="python3 -m ais_bench --model om/qwen3_06b_prefill_S4096_b1.om --loop 5 --warmup_count 1" \
       --output="./msprof_qwen3_06b_prefill_S4096_b1_PipeUtilization" \
       $BASE_FLAGS \
       --aic-metrics=PipeUtilization --l2=on
```

**4 个 metric 必须都采**（否则 `prism-extract` 拼不齐）：
- `ArithmeticUtilization` — Cube 利用率
- `PipeUtilization` — pipe-level breakdown（mte1/mte2/mte3/fixpipe/scalar）
- `Memory` — DRAM 流量
- `L2Cache` — L2 hit/miss

**已知崩溃模式**（来自 Phase M+ 经验）：
- `loop > 5` 在长上下文（S≥4096）+ B=8 时 msprof analyze 会 OOM → 用 `loop=1 warmup=0` 试试
- Qwen3 export 时 BOOL cumsum 在 ATC 报错 → `export_qwen3_prefill.py` 已含 `_patch_cumsum_bool_cast()` 补丁

### Step 4 — rsync 拉回

```bash
rsync -avz --include='*/' --include='op_summary*.csv' --include='task_time*.csv' --exclude='*' \
      user@npu-server:~/sim-experiment/msprof_qwen3_06b_prefill_*/ \
      msprof_data/
```

> **注意**：原始 msprof 目录每个 ~50MB，含大量 .db / 大 JSON。`.gitignore` 已排除 `msprof_data/` 和 `*.db` `*.om` `*.onnx`。只拉回 CSV 即可（小文件）。

### Step 5 — 接进 prism 拟合 pipeline

```bash
prism-extract --output data/calibration/cube_util_extracted.json
prism-fit     --output data/calibration/eta_physics_fit.json

# 验证：BERT 验证集 MAE < 15 pp 是硬门槛
cat data/calibration/eta_physics_fit.json | jq '.validation.bert.mae_pp'
```

完整流程含模型添加见 [docs/tutorials/03_recalibrate_with_new_msprof.md](../docs/tutorials/03_recalibrate_with_new_msprof.md) 与 [docs/tutorials/04_add_new_model.md](../docs/tutorials/04_add_new_model.md)。

---

## ONNX / OM 文件分发策略

| 文件类型 | git 追踪？ | 为什么 | 如何获取 |
|---------|----------|--------|---------|
| `*.onnx` | ❌ | 大（40-330 MB），可重生 | `python benchmark/export_*.py` |
| `*.om` | ❌ | Ascend 专用，必须在 NPU 上 ATC 转 | Step 2 流程 |
| `*.db` | ❌ | msprof SQLite，~38 MB/文件 | 重跑 msprof 重生 |
| `msprof_data/**/op_summary*.csv` | ❌ | 单文件几百 KB，但累积过大 | rsync from NPU |
| `data/calibration/*.json` | ✅ | 拟合后产物，KB 级 | repo 已包含 |

如果你需要预先生成的 ONNX（比如不想跑 HuggingFace 下载），后续会通过 GitHub Releases 提供 artifact 包：
```bash
# 规划中
gh release download v0.1.0 --repo JiangLiSJTU/prism-npu --pattern '*.onnx.tar.gz'
tar -xzf onnx_models_v0.1.0.tar.gz -C models/
```

---

## 故障排查

| 现象 | 根因 | 解法 |
|------|------|------|
| ATC 报 `Cumsum<DT_BOOL> not supported` | Qwen3 attention_mask BOOL cumsum | 用 `export_qwen3_prefill.py`（已含补丁），不要用旧 export 脚本 |
| msprof analyze OOM/段错误 | loop 数过多 + 长上下文 | 降 `--loop` 到 5/2/1 阶梯重试 |
| `prism-extract` 报 metric 缺失 | 4 个 metric 没采全 | 确认 `msprof_data/<model>_<metric>/` 4 个目录都在 |
| BERT MAE > 15 pp | 新数据有偏 / 拟合不稳 | 检查 `cube_util_extracted.json` 是否含异常 op；尝试加正则化 |
| Timeloop "smartbuffer_sram area not found" | Accelergy 缺组件 yaml | 用 `arch/ascend_910b4_for_mapping.yaml` 派生版（已含 3 个组件） |

更多 msprof 失败模式见 [docs/methodology/05_calibration.md §6](../docs/methodology/05_calibration.md#6-msprof-失败模式--应对)。
