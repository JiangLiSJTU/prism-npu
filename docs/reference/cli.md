# CLI 参数参考

8 个命令行入口（`pip install -e .` 后可用）+ 8 个 thin wrapper（`scripts/prism_*.py`，无 install 也可用）。

---

## prism-extract

从 msprof CSV 提取 per-op pipe time + cube util。

```bash
prism-extract [--msprof-dir DIR] [--output JSON]
```

| 参数 | 类型 | 默认 | 含义 |
|------|------|-----|------|
| `--msprof-dir` | path | `msprof_data/` | msprof 顶层目录（含多个 `msprof_<model>_b<B>_<metric>/` 子目录）|
| `--output` | path | `data/calibration/cube_util_extracted.json` | 输出 JSON |
| `--metric` | str | `ArithmeticUtilization` | 提哪个 metric: ArithmeticUtilization / PipeUtilization / Memory / L2Cache |
| `--model-filter` | str | （无）| 仅提取匹配的 model 名（如 'qwen3'）|

输出 JSON 结构：

```json
{
  "msprof_qwen3_06b_b1_ArithmeticUtilization": {
    "top_shapes_by_aicore_time": [
      {"M": 4096, "N": 3072, "K": 1024, "B": 1, "op_kind": "BMM",
       "cube_util_pct": 78.0, "aicore_time_us": 1331319, "count": 56}
    ]
  },
  ...
}
```

详见 [methodology/05_calibration.md §4](../methodology/05_calibration.md#4-提取脚本prism-extract)。

---

## prism-fit

physics-informed η_real 拟合（Levenberg-Marquardt）。

```bash
prism-fit [--cube-util-json IN_JSON] [--output OUT_JSON]
```

| 参数 | 类型 | 默认 | 含义 |
|------|------|-----|------|
| `--cube-util-json` | path | `data/calibration/cube_util_extracted.json` | 输入：prism-extract 输出 |
| `--output` | path | `data/calibration/eta_physics_fit.json` | 输出：5 参数 + 训练/验证 MAE |

输出 JSON：

```json
{
  "method": "physics-informed (η_pipeline · η_tile · batch)",
  "params": {
    "alpha_MN_coupling": 14.5977,
    "beta_MK_coupling": 2.5051,
    "gamma_NK_coupling": 1.7484,
    "delta_linear_edge": 0.0,
    "gamma_B_batch": 0.0102
  },
  "training": {"n": 56, "mae_pp": 11.98, "rmse_pp": 16.33},
  "validation": {
    "bert": {"n": 16, "mae_pp": 14.33, "rmse_pp": 18.93},
    "gpt2": {"n": 16, "mae_pp": 12.07, "rmse_pp": 17.69}
  }
}
```

终端输出：

```
=== 拟合参数 ===
  α (M·N coupling) = 14.5977
  β (M·K coupling) = 2.5051
  γ (N·K coupling) = 1.7484
  δ (linear edge)  = 0.0000
  γ_B (batch term) = 0.0102

=== Qwen3 训练集 (56 个 shape) ===
  MAE = 11.98 pp,  RMSE = 16.33 pp,  max abs error = 35.21 pp

=== BERT-base 验证集 (16 个 shape) ===
  MAE = 14.33 pp ✓ (< 15 pp 硬门槛)
  ...
```

详见 [methodology/03_eta_real_model.md §5](../methodology/03_eta_real_model.md#5-levenberg-marquardt-拟合)。

---

## prism-regime

regime gate：判定 (model, arch) 是 host-bound / compute-bound / memory-bound。

```bash
prism-regime [--arch ARCH_YAML] [--model MODEL_YAML] [--batch B] [--sweep] [--output JSON]
```

| 参数 | 类型 | 默认 | 含义 |
|------|------|-----|------|
| `--arch` | path | （单点模式必填）| regime 专用 arch yaml（如 `arch/regime/ascend_910b4_with_calib.yaml`，含 `chip` + `calib` block）|
| `--model` | path | （必填，除非 --sweep）| model yaml |
| `--batch` | int | 1 | batch size |
| `--sweep` | flag | False | 跑全部 (model × arch_variant × batch) 组合 |
| `--output` | path | `data/regime_matrix.json`（仅 --sweep）| sweep 输出 JSON |

单点输出：

```json
{
  "model": "Qwen3-0.6B",
  "arch": "Ascend910B4",
  "batch": 1,
  "T_compute_us": 18753,
  "T_memory_us": 6620,
  "T_overhead_us": 15810,
  "regime": "compute-bound",
  "timeloop_needed": true,
  "dominant_aic_pipe": "mte2 (54.2%)"
}
```

regime 阈值（[methodology/02 §8](../methodology/02_three_layer_roofline.md#8-regime-分类decision-gate)）：

```
IF T_overhead > 2 × max(T_compute, T_memory)        → "host-bound"
ELIF T_compute > 2 × T_memory                       → "compute-bound"
ELIF T_memory > 2 × T_compute                       → "memory-bound"
ELSE                                                → "balanced"
```

---

## prism-predict-pipe

从模型 GEMM 规格 + 硬件参数解析预测 pipe 分解，输出与 `pipe_baseline_per_model.json` 同 schema 的 JSON，**无需 msprof PipeUtilization**。详见 [methodology/08_predict_pipe.md](../methodology/08_predict_pipe.md)。

```bash
prism-predict-pipe --model MODEL_YAML [--arch ARCH_YAML] [--batch B] \
                   [--output OUT_JSON] [--params PARAMS_JSON] \
                   [--merge-into BASELINE_JSON] [--refit-params] [--quiet]
```

| 参数 | 类型 | 默认 | 含义 |
|------|------|-----|------|
| `--model` | path | 必填（除非 `--refit-params`）| model YAML（须含 `gemm_spec:` 块，见 [tutorials/05](../tutorials/05_predict_new_model.md)）|
| `--arch` | path | `arch/ascend_910b4_for_sweep_v2.yaml` | arch YAML（提供 `cube_total_macs / hbm_bw_gbs / ...` 等参数）|
| `--batch` | int | 1 | batch size（线性 scale ops/activation/output）|
| `--output` | path | `data/calibration/predict_pipe_<name>.json` | 输出 JSON（schema 兼容 pipe_baseline）|
| `--params` | path | `data/calibration/predict_pipe_params.json` | 拟合常数 K0/H_prefill/H_decode（由 `--refit-params` 生成）。**5 套可选**（OOS 准确度排序）：`predict_pipe_params.json` (v4 默认)、`_v5.json`、`_v6.json`、`_v7.json`、`_v8.json` (**推荐**, multi-objective, OOS 全 component < 10%)。版本对比见 [tutorials/05 §2.1](../tutorials/05_predict_new_model.md#21-可选参数文件)、[methodology/08 §14-15](../methodology/08_predict_pipe.md) |
| `--refit-params` | flag | False | 仅重新拟合 K0/H/H 并保存（忽略 `--model` 等其他参数）|
| `--pipe-baseline` | path | `data/calibration/pipe_baseline_per_model.json` | 拟合用输入（`--refit-params` 时读）|
| `--merge-into` | path | （无）| 若设置，将预测结果合并进该 baseline JSON（**会修改文件**）|
| `--quiet` | flag | False | 抑制 stdout summary |

终端输出（预测模式）：

```
Predicted ModernBERT-base-S4096-b1 → data/calibration/predict_pipe_modernbert.json
  aic_time   =     51,708 μs  (dominant: mte2)
  aiv_time   =        338 μs
  kernel_gap =        449 μs (242 kernels × K0)
  host_gap   =     13,424 μs
  wall_clock =     65,920 μs
  confidence = low (encoder: only BERT-S128 in training set)
```

终端输出（`--refit-params`）：

```
Fit complete → data/calibration/predict_pipe_params.json
  K0       = 1.8558 μs/kernel
  H_prefill= 13424 μs
  H_decode = 204 μs
  training: host_gap MAE = 8.4%, kernel_gap MAE = 14.3% (n=9)
```

输出 JSON 结构（与 `pipe_baseline_per_model.json` 兼容 + 3 个增量字段 `predicted`/`confidence`/`spec_summary`）：

```json
{
  "baseline_arch_name": "ascend_910b4_for_sweep_v2",
  "configs": {
    "ModernBERT-base-S4096-b1": {
      "n_kernels_per_inf": 242,
      "aic_pipes_us": {"mac": ..., "mte1": ..., "mte2": ..., "fixpipe": ..., "scalar": 0},
      "aiv_pipes_us": {"vec": ..., "mte2": ..., "mte3": ..., "scalar": 0, "idle": 0},
      "wall_clock_us": 65920,
      "kernel_gap_us": 449,
      "host_gap_us": 13424,
      "aic_dominant_pipe": "mte2",
      "source": "predict_pipe_v1",
      "predicted": true,
      "confidence": "low (encoder: ...)",
      "spec_summary": {...}
    }
  }
}
```

完整教程见 [tutorials/05_predict_new_model.md](../tutorials/05_predict_new_model.md)。

---

## prism-sweep

12 维架构 sweep（49 唯一 variants × N models）。

```bash
prism-sweep [--pipe-baseline JSON] [--arch-baseline ARCH_YAML] [--output JSON]
```

| 参数 | 类型 | 默认 | 含义 |
|------|------|-----|------|
| `--pipe-baseline` | path | `data/calibration/pipe_baseline_per_model.json` | per-model PipeUtil 实测 baseline |
| `--arch-baseline` | path | `arch/ascend_910b4_for_sweep_v2.yaml` | sweep 参考的 baseline arch |
| `--output` | path | `data/outputs/phase_j_sweep.json` | sweep 输出 JSON |

输出（终端）：

```
                       BERT  GPT-2  Qwen3-pre  Qwen3-Emb  Qwen3-dec  NetT
n_cores=8            | 1.00  1.00     2.95       2.95       1.85      1.00
n_cores=16           | 1.00  1.00     1.50       1.50       1.20      1.00
n_cores=24 (baseline)| 1.00  1.00     1.00       1.00       1.00      1.00
n_cores=48           | 1.00  1.00     0.84       0.84       0.92      1.00
ub_l1_fused=True     | 0.99  1.00     0.80       0.80       0.93      1.00
hbm_bw=800 (HBM3)    | 1.00  1.00     0.85       0.85       0.65      1.00
...
```

输出 JSON 结构：

```json
{
  "baseline_arch": {...},
  "baseline_results": {
    "BERT-base-S128-b1": {"wall_clock_us": 16210, ...},
    ...
  },
  "variants": [
    {
      "variant": "n_cores=8",
      "ratio_per_model": {"BERT-base-S128-b1": 1.0, "Qwen3-prefill-S4096-b1": 2.95, ...}
    },
    ...
  ]
}
```

详见 [methodology/04_arch_sensitivity.md §7](../methodology/04_arch_sensitivity.md#7-sweep-结果汇总)。

---

## prism-ceiling

5 情景算子/软件/硬件优化天花板预测。

```bash
prism-ceiling [--pipe-baseline JSON] [--output-json JSON] [--output-md MD]
            [--host-gap-target FLOAT] [--hbm3-bw-gbs FLOAT]
```

| 参数 | 类型 | 默认 | 含义 |
|------|------|-----|------|
| `--pipe-baseline` | path | `data/calibration/pipe_baseline_per_model.json` | 输入：per-model pipe baseline |
| `--output-json` | path | `data/outputs/optimization_ceiling.json` | 5 情景 × N 配置 完整数据 |
| `--output-md` | path | `docs/findings/optimization_ceiling.md` | 人读 markdown 报告 |
| `--host-gap-target` | float | 10.0 | S2 情景下 host_gap 目标值（μs/kernel）|
| `--hbm3-bw-gbs` | float | 800 | S4 情景下 HBM3 带宽 |

终端输出（每行一 config）：

```
config                       S0    S1 (-%)   S2 (-%)   S3 (-%)   S4 (-%)
BERT-base-S128-b1           16,210 14,812(8.6%) 4,112(74.6%) 4,013(75.2%) 3,750(76.9%)
Qwen3-prefill-S4096-b1   3,050,000 1,963,103(35.6%) ... 1,338,829(56.1%) 1,338,829(56.1%)
Qwen3-decode-Min4-Skv128-b1  7,690 3,111(59.5%) 3,111(59.5%) 2,597(66.2%) 1,520(80.2%)
...
```

5 情景定义见 [methodology/07 §3](../methodology/07_optimization_ceiling.md#3-5-个优化情景定义)。

---

## prism-mapping

Timeloop manual mapping cycles 校准（Docker 后端）。

```bash
prism-mapping --workload-name NAME [options]
```

| 参数 | 类型 | 必填 | 含义 |
|------|------|----|------|
| `--workload-name` | str | ✅ | 标识符（用于 run dir 命名）|
| `--M` `--N` `--K` | int | ✅（除非 --workload-yaml）| GEMM 形状 |
| `--m-l2-spatial` `--n-l2-spatial` | int | （默认 4 × 6）| L2 spatial 切分 |
| `--m-cube-spatial` `--n-cube-spatial` | int | 默认 16 / 16 | Cube spatial |
| `--workload-yaml` | path | （可选）| 已有 workload yaml（跳过 auto-gen）|
| `--mapping-yaml` | path | （可选）| 已有 mapping yaml |
| `--arch-yaml` | path | ✅ | arch yaml（推荐 `arch/ascend_910b4_for_mapping.yaml`）|
| `--cube-k-correction` | int | 16 | Timeloop K-temporal 修正因子（cycles / 16）|
| `--clock-mhz` | int | 1000 | 时钟频率（用于 wall-clock μs 计算）|
| `--output-json` | path | （可选）| 结果 JSON 输出 |

输出（终端）：

```
=== Result (qwen_emb_S4096_ffn_gate) ===
  cycles (raw):                131,072
  utilization:                 100.00%
  fJ/Compute:                 5253.96
  energy:                    67696.76 uJ
  cycles (corrected):            8,192    (raw / 16)
  wall-clock raw:              131.07 μs @1000 MHz
  wall-clock corrected:          8.19 μs @1000 MHz
```

→ 需 Docker。详见 [legacy/scripts/mapper_README.md](../../legacy/scripts/mapper_README.md)（旧 README，本工具 release 版迁此 doc）。

---

## prism-render

Jinja2 渲染 4 份 finding 报告。

```bash
prism-render [--vars JSON] [--templates-dir DIR] [--output-dir DIR] [--check] [--dry-run]
```

| 参数 | 类型 | 默认 | 含义 |
|------|------|-----|------|
| `--vars` | path | `data/experiment_variables.json` | 宏变量 JSON |
| `--templates-dir` | path | `reports/templates/` | Jinja2 模板目录 |
| `--output-dir` | path | `docs/findings/` | 输出目录（含特殊路径映射）|
| `--check` | flag | False | CI 模式：仅 check 现有渲染与 templates 一致；exit 1 表 drift |
| `--dry-run` | flag | False | 仅打印 diff，不写文件 |

正常输出（不带 flag）：

```
WROTE: docs/findings/主报告.md
WROTE: docs/findings/roofline校准报告.md
WROTE: docs/findings/微架构探索报告.md
WROTE: docs/findings/msprof分解报告.md

--- Summary ---
OK: 4 templates rendered
```

`--check` 输出：

```
--- Summary ---
OK:  910B4_roofline_校准报告_v3.md.j2 (identical)
OK:  msprof_breakdown_summary.md.j2 (identical)
OK:  微架构探索_报告_v3.md.j2 (identical)
OK:  NPU 架构设计_v2.md.j2 (identical)
```

exit 0 = 一致；exit 1 = drift（CI 必须 fail）。

详见 [methodology/05_calibration.md §8](../methodology/05_calibration.md#8-复现性保证)。

---

## thin wrapper（无 install）

`scripts/prism_*.py` 是薄包装，自动 `sys.path.insert(src/)`。等价于上面 8 个 entry-point 命令：

```bash
python3 scripts/prism_extract.py       [args]
python3 scripts/prism_fit.py           [args]
python3 scripts/prism_regime.py        [args]
python3 scripts/prism_predict_pipe.py  [args]
python3 scripts/prism_sweep.py         [args]
python3 scripts/prism_ceiling.py       [args]
python3 scripts/prism_mapping.py       [args]
python3 scripts/prism_render.py        [args]
```

适用场景：未 `pip install -e .`、CI 测试、临时调用。

---

## 退出码约定

| 码 | 含义 |
|----|------|
| 0 | 成功 |
| 1 | 数据错误（如 fit 失败、render --check 报 drift）|
| 2 | 命令行参数错误 |

CI 中：

```bash
prism-render --check && pytest tests/ -q && echo "PR OK"
```

---

## 全局选项（暂未实现，规划中）

- `--config <yaml>` 从配置文件读所有参数（替代命令行）
- `--quiet` 仅输出错误
- `--verbose` 输出详细 debug 信息

详见 [CONTRIBUTING.md §贡献指南](../../CONTRIBUTING.md)。
