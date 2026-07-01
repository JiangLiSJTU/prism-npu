# 教程 4：添加新 transformer 模型

把新模型（例：Qwen3-7B）加进 sweep MODELS dict，让它出现在 49 变体 × N 模型的输出中。

---

## 0. 前置

- 已完成 [01_quickstart.md](01_quickstart.md)
- NPU 服务器可访问（采集 msprof 实测）

## 1. 模型分析（决定 ops 表）

在编辑代码之前，先弄清楚新模型的关键 GEMM 算子和 Vector op 量级。例如 Qwen3-7B：

| 参数 | 值 |
|------|---|
| L (layers) | 32 |
| d (hidden) | 4096 |
| FFN_hidden | 22016 (SwiGLU) |
| n_q_heads | 32 |
| n_kv_heads | 8 (GQA) |
| head_dim | 128 |
| 默认 prefill S | 4096 |

GEMM ops per layer（× L 层）：

| Op | M | N | K | count_per_layer |
|----|---|---|---|----------------|
| Q proj | S | n_q_heads × head_dim = 4096 | d = 4096 | 1 |
| K/V proj | S | n_kv_heads × head_dim = 1024 | d = 4096 | 2 |
| O proj | S | d = 4096 | n_q_heads × head_dim = 4096 | 1 |
| FFN gate/up | S | FFN_hidden = 22016 | d = 4096 | 2 |
| FFN down | S | d = 4096 | FFN_hidden = 22016 | 1 |
| Attn QK^T | S | S = 4096 | head_dim = 128 | 1（per head 但用 BMM）|
| Attn AV | S | head_dim = 128 | S = 4096 | 1 |

Vector ops per layer：
- 2 RMSNorm: 2 × S × d × 3 ops
- 1 Softmax: heads × S × S × 4 ops
- SwiGLU: S × FFN × 2 ops

→ 总 vector_ops_per_layer ≈ 2.5 × 10^9（Softmax 占主导）

## 2. NPU 上跑 msprof 采集

参考 [methodology/05_calibration.md §2.2](../methodology/05_calibration.md#22-推荐采集模板11-配置)。简化模板：

```bash
ssh user@npu-server
cd ~/sim-experiment

# 1) 导出 ONNX（修改 export_qwen3_prefill.py 适配 Qwen3-7B）
python3 benchmark/export_qwen3_prefill.py --model Qwen/Qwen3-7B --S 4096 --batch 1

# 2) ATC 转 OM
atc --model models/qwen3_7b_prefill_S4096_b1.onnx \
    --framework=5 \
    --output om/qwen3_7b_prefill_S4096_b1 \
    --input_shape="input_ids:1,4096" \
    --soc_version=Ascend910B4 --log=error

# 3) msprof 4 metric × 1 配置
for m in ArithmeticUtilization PipeUtilization Memory L2Cache; do
  outdir=msprof_qwen3_7b_S4096_b1_${m}
  rm -rf $outdir; mkdir -p $outdir; chmod 750 $outdir
  msprof --application="python3 -m ais_bench --model om/qwen3_7b_prefill_S4096_b1.om \
                                              --loop 3 --warmup_count 1" \
         --output=./$outdir --task-time=on --ai-core=on --aicpu=on --runtime-api=on --hccl=off \
         --aic-metrics=$m --l2=on
done
```

> ⚠️ Qwen3-7B 内存大（~14 GB），loop 数要小（3-5）；msprof analyze 大概率成功率低于 Qwen3-0.6B。详见 [methodology/05_calibration.md §6](../methodology/05_calibration.md#6-msprof-失败模式--应对)。

## 3. rsync 拉回 + 提取

```bash
# 本地
for m in ArithmeticUtilization PipeUtilization Memory L2Cache; do
  rsync -avz user@npu-server:~/sim-experiment/msprof_qwen3_7b_S4096_b1_${m}/ \
              msprof_data/msprof_qwen3_7b_S4096_b1_${m}/
done

# 提取
prism-extract \
  --output data/calibration/cube_util_extracted.json
```

## 4. 重新 fit η_real

```bash
prism-fit
# 验证 BERT MAE 仍 < 15 pp
```

详见 [03_recalibrate_with_new_msprof.md](03_recalibrate_with_new_msprof.md)。

## 5. 把模型加进 sweep MODELS dict

编辑 `src/prism/sweep/runner.py` 顶部 `MODELS` dict：

```python
MODELS = {
    # ... 已有 5 个 ...
    
    'Qwen3-7B-prefill-S4096-b1': {
        'L': 32,
        'ops': [
            (4096, 4096,  4096, 'BMM', 1),    # Q proj
            (4096, 1024,  4096, 'BMM', 2),    # K/V proj (GQA: 8 KV heads × 128)
            (4096, 4096,  4096, 'BMM', 1),    # O proj
            (4096, 22016, 4096, 'BMM', 2),    # FFN gate/up
            (4096, 4096, 22016, 'BMM', 1),    # FFN down
            (4096, 4096,   128, 'BMM', 1),    # Attn QK^T
            (4096,  128,  4096, 'BMM', 1),    # Attn AV
        ],
        # 2 RMSNorm + 1 Softmax + 1 SwiGLU ≈ 2.5e9
        'vector_ops_per_layer': 2_500_000_000,
        'beta_layer_us': 1500,    # 占位，待 fit 后更新
        'beta_device_us': 119,
        'alpha_per_batch_us': 800,
    },
}
```

## 6. 把模型加进 ceiling 数据源

`prism-extract` 已自动加进 `data/calibration/pipe_baseline_per_model.json`。验证：

```python
import json
data = json.load(open('data/calibration/pipe_baseline_per_model.json'))
assert 'Qwen3-7B-prefill-S4096-b1' in data['configs']
print(data['configs']['Qwen3-7B-prefill-S4096-b1']['aic_pipes_us'])
```

## 7. 重跑 sweep + ceiling

```bash
prism-sweep      # 49 variants × 6 models（多了 Qwen3-7B）
prism-ceiling    # 5 scenarios × 12 configs
```

## 8. 渲染报告 + commit

```bash
prism-render
prism-render --check    # exit 0

git add src/prism/sweep/runner.py
git add data/calibration/eta_physics_fit.json
git add data/calibration/pipe_baseline_per_model.json
git add docs/findings/
git commit -m "models: add Qwen3-7B prefill-S4096 to sweep MODELS"
```

---

## 故障排除

### ATC 转 OM 失败

→ 见 [methodology/05_calibration.md §6.1-6.2](../methodology/05_calibration.md#61-atc-convert-cumsum-bool-已修复) 的失败模式。

### β_layer 占位值离谱

→ 准确值应从 wall-clock 实测拟合：

```python
β_layer ≈ (wall_clock_avg - β_device - Σ_op (op_aicore_time × count × n_layers)) / n_layers
```

详见 [methodology/02_three_layer_roofline.md §4](../methodology/02_three_layer_roofline.md#4-β-校准方法)。

### 跨模型 ratio 解读异常

→ 不同模型 host_gap_per_kernel 不同（[methodology/06 §2.4](../methodology/06_assumptions_limits.md#24-host_gap_per_kernel-跨-model-不同)），不可直接 cross-model 比 ratio。**单模型 ratio 才是工具的核心输出**。
