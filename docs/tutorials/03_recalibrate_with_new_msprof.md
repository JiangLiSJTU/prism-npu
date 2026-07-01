# 教程 3：用新 msprof 数据重新校准 η_real

如果你 NPU 上跑了新模型 / 新配置的 msprof 采集，本教程把数据接进 η_real 拟合 pipeline。

---

## 0. 前置

- NPU 服务器上已跑完 msprof 采集（见 [methodology/05_calibration.md §2](../methodology/05_calibration.md#2-数据采集-pipeline)）
- 本地已 `pip install -e ".[dev]"`

## 1. rsync 拉回 msprof 数据

```bash
# 模板：根据实际 model_name 调整
rsync -avz user@npu-server:~/sim-experiment/msprof_<model>_b<B>_ArithmeticUtilization/ \
           msprof_data/msprof_<model>_b<B>_ArithmeticUtilization/

rsync -avz user@npu-server:~/sim-experiment/msprof_<model>_b<B>_PipeUtilization/ \
           msprof_data/msprof_<model>_b<B>_PipeUtilization/
```

例如新模型 `qwen3_7b` 在 b=1 上的 4 metric 全套：

```bash
for m in ArithmeticUtilization PipeUtilization Memory L2Cache; do
  rsync -avz user@npu-server:~/sim-experiment/msprof_qwen3_7b_b1_${m}/ \
              msprof_data/msprof_qwen3_7b_b1_${m}/
done
```

## 2. 验证 msprof 数据完整性

```bash
find msprof_data/msprof_qwen3_7b_b1_*/PROF_*/mindstudio_profiler_output/op_summary*.csv
```

每个 metric 目录应有 1 个 op_summary CSV。如某 metric 没有，可能 msprof analyze 失败（[methodology/05_calibration.md §6.3](../methodology/05_calibration.md#63-msprof-analyze-大-workload-崩溃)）。

## 3. 提取 η_real 训练数据

```bash
prism-extract \
  --output data/calibration/cube_util_extracted.json
```

工具会自动遍历 `msprof_data/` 中所有 `*_ArithmeticUtilization/` 目录提取 cube_util，按 (model, S, B, op_shape) 聚合。

验证新模型已加入：

```python
import json
data = json.load(open('data/calibration/cube_util_extracted.json'))
keys = [k for k in data if 'qwen3_7b' in k.lower()]
print(f"Qwen3-7B 提取出 {sum(len(data[k].get('top_shapes_by_aicore_time', [])) for k in keys)} 个 shape")
```

## 4. 重新拟合 η_real

```bash
prism-fit \
  --cube-util-json data/calibration/cube_util_extracted.json \
  --output         data/calibration/eta_physics_fit.json
```

期望终端输出（**关键看 BERT validation MAE 是否仍 < 15 pp**）：

```
=== 拟合参数 ===
  α (M·N coupling) = 14.5xxx
  β (M·K coupling) = 2.5xxx
  γ (N·K coupling) = 1.7xxx
  δ (linear edge)  = 0.0000
  γ_B (batch term) = 0.010x

=== Qwen3 训练集 (n=XX 个 shape) ===
  MAE = 11.xx pp,  RMSE = 16.xx pp
  
=== BERT-base 验证集 (n=XX 个 shape) ===
  MAE = 14.xx pp ✓ (< 15 pp 硬门槛)

=== GPT-2-small 验证集 (n=XX 个 shape) ===
  MAE = 12.xx pp ✓
```

→ 写入 `data/calibration/eta_physics_fit.json`

## 5. 硬门槛检查

如 BERT MAE > 15 pp：

| 可能原因 | 应对 |
|---------|------|
| 新加的 model shape 分布偏（如全是 small attention head）| 加更多 mid-large GEMM shape 进训练集 |
| count threshold 过滤掉数据 | 检查 `extract_cube_util.py` 的 `count >= 20` 阈值；如新数据 loop=2 则 count 不足 |
| 物理公式不适合新 workload（如 INT8 量化）| η_real 公式仅适合 FP16；INT8 需扩公式 |

不过门槛**禁止 commit**——这是工具准入的硬标准。详见 [methodology/03_eta_real_model.md §6](../methodology/03_eta_real_model.md#6-训练--验证集设计)。

## 6. 重新跑 sweep + ceiling

```bash
prism-sweep      # 用新 η_real 参数跑 sweep
prism-ceiling    # ceiling 也读 pipe_baseline，需独立扩
```

如 baseline pipe 有变化（新模型加进 sweep MODELS dict），还需：

```bash
# 7) 重新提取 pipe baseline（含新 model）
prism-extract --metric PipeUtilization \
            --output data/calibration/pipe_baseline_per_model.json
```

## 7. 重新渲染 finding 报告

```bash
prism-render
# 4 模板渲染到 docs/findings/
```

## 8. Commit 流程

```bash
git add data/calibration/eta_physics_fit.json    # 拟合参数变了
git add data/calibration/pipe_baseline_per_model.json  # 如 pipe 也变了
git add docs/findings/                           # 报告渲染输出变了
git commit -m "calibration: refit η_real with Qwen3-7B 实测 (BERT MAE 14.xx pp)"
```

`data/outputs/*.json` 是可重生的，被 .gitignore 排除。

PR 提交前必须 5 项验证（[CONTRIBUTING.md](../../CONTRIBUTING.md)）：
```bash
prism-render --check    # exit 0
prism-fit               # BERT MAE < 15 pp
prism-sweep             # 与 ref 一致（如改了 baseline 这条会变）
prism-ceiling           # 同上
pytest tests/         # 全绿
```

---

## 完成后你应该能

- 把新 msprof 数据接入工具的 fit pipeline
- 验证拟合质量（BERT < 15 pp 硬门槛）
- 处理 fit 失败的常见原因
- 重新生成 finding 报告

## 下一步

| 目的 | 教程 |
|-----|------|
| 加新 transformer 模型到 sweep | [04_add_new_model.md](04_add_new_model.md) |
| 跑 NPU 上的 msprof 采集 | [methodology/05_calibration.md §2](../methodology/05_calibration.md#2-数据采集-pipeline) |
| 调整 fit 公式假设 | [methodology/03_eta_real_model.md](../methodology/03_eta_real_model.md) + 06 |
