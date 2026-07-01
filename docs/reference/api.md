# Python API 参考

`prism` 包对外 API。复用 + 集成进上层工作流（如自动化 sweep loop、CI 验证等）时使用。

```python
import prism
```

---

## 1. eta_real 模块

### `prism.eta_real.fit`

```python
from prism.eta_real.fit import fit, evaluate, predict_eta

# 拟合
samples = collect_shapes('data/calibration/cube_util_extracted.json', 'qwen3')
params = fit(samples)   # 返回 (α, β, γ, δ, γ_B) 5-tuple

# 验证
val_samples = collect_shapes(..., 'bert_base')
errs, mae, rmse = evaluate(val_samples, params, 'BERT 验证集')

# 单次预测
sample = {'M_per_batch': 4096, 'N': 3072, 'K': 1024, 'B': 1, 'op_kind': 'BMM'}
eta = predict_eta(sample, *params)   # → ~0.78
```

### `prism.eta_real.extract`

```python
from prism.eta_real.extract import collect_cube_util

# 从 msprof 目录提取
util_data = collect_cube_util('msprof_data/')
# {'msprof_qwen3_06b_b1_ArithmeticUtilization': {'top_shapes_by_aicore_time': [...]}}
```

### `prism.eta_real.predict`

```python
from prism.eta_real.predict import predict_eta_for_workload

# 给定 workload 描述，预测 η_real
eta = predict_eta_for_workload(
    M=4096, N=3072, K=1024, B=1, op_kind='BMM',
    fit_json='data/calibration/eta_physics_fit.json',
)
```

---

## 2. roofline 模块

### `prism.roofline.predict`

```python
from prism.roofline.predict import predict_910b4_v2

# 单 model 单 batch wall-clock 预测
result = predict_910b4_v2(
    model_yaml='models/qwen3_0.6b.yaml',
    batch=1,
)
# {'T_compute': 18753, 'T_memory': 6620, 'T_overhead': 15810, 'wall_clock': 18753, ...}
```

### `prism.roofline.regime`

```python
from prism.roofline.regime import classify_regime

regime = classify_regime(
    arch_yaml='arch/ascend_910b4_for_sweep_v2.yaml',
    model_yaml='models/qwen3_0.6b.yaml',
    batch=1,
)
# {'regime': 'compute-bound', 'timeloop_needed': True, ...}
```

---

## 3. sweep 模块

### `prism.sweep.runner`

```python
from prism.sweep.runner import run_sweep, MODELS, SWEEP

# 跑完整 sweep
results = run_sweep(
    pipe_baseline_path='data/calibration/pipe_baseline_per_model.json',
    arch_baseline_path='arch/ascend_910b4_for_sweep_v2.yaml',
)
# {'baseline': {...}, 'variants': [{variant_name, ratio_per_model, ...}, ...]}

# 单 variant 预测
from prism.sweep.runner import predict_wallclock_v3
result = predict_wallclock_v3(
    model_key='Qwen3-prefill-S4096-b1',
    batch=1,
    arch_variant=arch_variant_dict,
    pipe_baseline=pipe_baseline_dict,
)
# {'aic_time_us': ..., 'aiv_time_us': ..., 'host_gap_us': ..., 'wall_clock_us': ...}
```

### MODELS dict（sweep 的 model spec source of truth）

```python
from prism.sweep.runner import MODELS

# 5 个 model 的 (ops 表, vector_ops, β) 完整描述
print(MODELS['BERT-base-S128-b1'])
# {'L': 12, 'ops': [(M, N, K, op_kind, count), ...], 'vector_ops_per_layer': ..., 'beta_layer_us': ...}
```

### SWEEP dict（sweep 维度定义）

```python
from prism.sweep.runner import SWEEP

# 12 个维度的 variant 列表
print(SWEEP['n_cores'])           # → [8, 12, 16, 24, 32, 48]
print(SWEEP['ub_l1_fused'])       # → [False, True]
```

---

## 4. ceiling 模块

### `prism.ceiling.predict`

```python
from prism.ceiling.predict import predict_all_scenarios, ScenarioResult

# 跑 5 情景全 11 配置
results = predict_all_scenarios(
    pipe_baseline_path='data/calibration/pipe_baseline_per_model.json',
    host_gap_target_per_kernel=10.0,    # S2 目标
    hbm3_bw_gbs=800,                    # S4 假设
)
# {'BERT-base-S128-b1': [ScenarioResult(scenario='S0', ...), ScenarioResult(scenario='S1', ...), ...]}

# 单 scenario 计算
from prism.ceiling.scenarios import compute_software_ceiling
pipe = pipe_baseline['BERT-base-S128-b1']
s1_result = compute_software_ceiling(pipe)
# ScenarioResult(scenario='S1_software_ceiling', wall_clock=..., reduction_pct=8.6)
```

---

## 5. mapper 模块

### `prism.mapper.generate`

```python
from prism.mapper.generate import generate_mapping

mapping_yaml_str, info = generate_mapping(
    M=4096, N=3072, K=1024,
    m_l2_spatial=4, n_l2_spatial=6,
    m_cube_spatial=16, n_cube_spatial=16,
)
# Path 写入 mapping yaml
```

### `prism.mapper.runner`

```python
from prism.mapper.runner import run_timeloop_model

result = run_timeloop_model(
    arch_yaml=Path('arch/ascend_910b4_for_mapping.yaml'),
    workload_yaml=Path('mapper/audit/<workload>.yaml'),
    mapping_yaml=Path('mapper/manual/<mapping>.yaml'),
    run_dir=Path('timeloop_results/manual_mapping/<run>'),
    timeout_sec=120,
)
# {'cycles': 131072, 'utilization_pct': 100, 'fj_per_compute': 5253, ...}
```

→ 需 Docker，详见 [methodology/05_calibration.md §6](../methodology/05_calibration.md#6-msprof-失败模式--应对) 关于 Docker 配置。

---

## 6. reports 模块

### `prism.reports.render`

```python
from prism.reports.render import render_all, render_template

# 全部 4 模板
render_all(
    vars_path=Path('data/experiment_variables.json'),
    templates_dir=Path('reports/templates/'),
    output_dir=Path('docs/findings/'),
)

# 单模板
render_template(
    template_name='NPU 架构设计_v2.md.j2',
    vars_dict={...},   # 已加载的 variables
)
# 返回 rendered string
```

### CI 检查模式

```python
from prism.reports.render import check_drift

drift = check_drift(
    vars_path=...,
    templates_dir=...,
    output_dir=...,
)
# 返回 list of (template_name, diff_lines)；空表示 no drift
```

---

## 7. predict_pipe 模块

### `prism.predict_pipe`

从 GEMM 规格预测 pipe baseline，使 ceiling/sweep 对新模型可用（无需 msprof）。
详见 [methodology/08_predict_pipe.md](../methodology/08_predict_pipe.md)。

```python
from prism.predict_pipe import (
    ModelSpec, KNOWN_MODELS,
    compute_gemm_ops, compute_vector_ops, estimate_n_kernels,
    predict_pipe_baseline, predict_for_model_yaml, assign_confidence,
    fit_host_gap, fit_kernel_gap, leave_one_model_out_cv, fit_all_and_save,
)
```

#### 单模型预测

```python
from prism.predict_pipe import ModelSpec, predict_pipe_baseline
from prism.predict_pipe.predict import _arch_dict_from_yaml

spec = ModelSpec.from_yaml('models/regime/modernbert_base_prefill_S4096.yaml')
arch = _arch_dict_from_yaml('arch/ascend_910b4_for_sweep_v2.yaml')
fitted = {'K0_us_per_kernel': 1.86, 'H_prefill_us': 13424, 'H_decode_us': 204}

entry = predict_pipe_baseline(spec, arch, fitted, batch=1)
# {'wall_clock_us': 65920, 'aic_pipes_us': {...}, 'confidence': 'low (encoder: ...)', ...}
```

#### 拟合 + 持久化常数

```python
from prism.predict_pipe import fit_all_and_save

result = fit_all_and_save(
    pipe_baseline_path='data/calibration/pipe_baseline_per_model.json',
    output_path='data/calibration/predict_pipe_params.json',
)
# K0_us_per_kernel ≈ 1.86; H_prefill_us ≈ 13424; H_decode_us ≈ 204
# training MAE: host_gap 8.4%, kernel_gap 14.3%（n=9）
# loo_cv: per-family hold-out errors
```

#### v5 / v6 / v7 / v8 进阶 fit（per-bucket / multi-objective）

```python
# v6 — per-bucket fit (eager baseline, deprecated for new code)
from prism.predict_pipe import fit_v6, lomo_v6
result_v6 = fit_v6.fit_v6(
    baseline_path='data/calibration/pipe_baseline_per_model.json',
    arch_yaml='arch/ascend_910b4_for_sweep_v2.yaml',
    v4_params_path='data/calibration/predict_pipe_params.json',
)
# data/calibration/predict_pipe_params_v6.json

# v7 — SDPA-aware 3-bucket (replaces v6 AIC_QWEN3 hack)
from prism.predict_pipe import fit_v7
result_v7 = fit_v7.fit_v7(...)

# v8 — multi-objective (RECOMMENDED): wall + 0.3·aic + 0.3·aiv + 0.2·n_kern
from prism.predict_pipe import fit_v8
result_v8 = fit_v8.fit_v8(...)
# data/calibration/predict_pipe_params_v8.json
# OOS: wall MAE 8.4%, AIC 7.0%, AIV 8.0%, n_kern 2.8%; cancellation_ratio 1.0

# Predict using a specific version
fitted_v8 = json.load(open('data/calibration/predict_pipe_params_v8.json'))
# fitted_v8 includes "v_model": "v8" marker → predict_pipe_baseline auto-dispatches
entry = predict_pipe_baseline(spec, arch, fitted_v8, batch=1)
```

| Module | Schema buckets | Fit objective | OOS wall MAE | OOS all-component MAE | Use case |
|---|---|---|---:|---|---|
| `physics` (v4) | archetype 3-bucket | wall only | ~362% | AIV 473% | in-distribution only |
| `physics_v5` | linear amp + saturating n_kern | wall only | 87.8% | partial improvement | deprecated |
| `physics_v6` | 4-bucket (AIC_QWEN3 hack) | wall only | 10.1% | AIC 57% (cancellation) | historical eager baseline |
| `physics_v7` | 3-bucket SDPA-aware | wall only | 11.8% | AIC 130% (cancellation) | SDPA path early |
| `physics_v7` + `fit_v8` | 3-bucket SDPA | wall + 0.3·aic + 0.3·aiv + 0.2·n_kern | **8.4%** | **all < 10%** | **production** |

#### Public dataclass / 配置

```python
@dataclass
class ModelSpec:
    name: str
    arch: str          # "encoder" | "decoder"
    layers: int
    S: int             # sequence length（decode 时填 1）
    d_model: int
    d_ff: int
    n_heads: int
    n_kv_heads: int    # 0 = MHA; > 0 = GQA
    d_head: int
    vocab: int
    ffn_type: str = "standard"   # "standard" | "glu" | "swiglu"
    note: str = ""

    @classmethod
    def from_yaml(cls, path) -> "ModelSpec": ...
```

`KNOWN_MODELS` 是 baseline 注册表，键对应 `pipe_baseline_per_model.json` 中
9 个有 msprof 实测的 config（leave-one-out CV 用）。

---

## 8. dataclass 类型

为类型安全，工具内部用 dataclass 替代 dict 传参。**外部 API 调用者建议也用 dataclass**：

```python
from prism.ceiling.predict import WallClockBreakdown, ScenarioResult

# WallClockBreakdown
class WallClockBreakdown:
    aic_time_us:    float
    aiv_time_us:    float
    kernel_gap_us:  float
    host_gap_us:    float
    wall_clock_us:  float   # authoritative total (scenario-aware, not just sum)

# ScenarioResult
class ScenarioResult:
    scenario:      str
    description:   str
    wall_clock:    WallClockBreakdown
    reduction_pct: float
    aic_pipes:     dict[str, float]   # baseline pipe times scaled
    aiv_pipes:     dict[str, float]
```

`asdict()` 把它们转 dict（[CLI](cli.md) JSON output 用）。

---

## 9. 异常处理

| 异常 | 抛出 | 含义 |
|------|------|------|
| `KeyError('xxx')` | sweep / ceiling | 输入 model key 不在 baseline 中 |
| `ValueError('K=xxx 没有可行的 k_inner')` | mapper.generate | 工作负载形状不能被合法 tile |
| `FileNotFoundError` | 任何 IO | 通常是 yaml/json 路径不对，或 data/calibration 缺失 |
| `jinja2.exceptions.UndefinedError` | reports.render | 模板里引用的 var 没在 JSON 中 |

---

## 10. 版本兼容性

工具版本与 schema 版本对齐：

| 工具版本 | arch yaml version | sweep API |
|---------|-----------------|-----------|
| 0.1.x | 2.0 | `predict_wallclock_v3` (pipe-aware) |
| 0.2.x（未来）| 3.0 | 可能加 KV cache prefetcher 等新 sub-module |

`pyproject.toml` 的 `version` 字段是工具版本。schema 版本变更时 README + CHANGELOG.md 同步更新。

---

## 11. 完整模块导入清单

```python
# eta_real
from prism.eta_real.extract import collect_cube_util
from prism.eta_real.fit import fit, evaluate, predict_eta, main
from prism.eta_real.match import match_timeloop_to_msprof
from prism.eta_real.predict import predict_eta_for_workload

# roofline
from prism.roofline.predict import predict_910b4_v2, validate_against_known_v2
from prism.roofline.regime import classify_regime, main

# predict_pipe (Issue #2)
from prism.predict_pipe import (
    ModelSpec, KNOWN_MODELS,
    compute_gemm_ops, compute_vector_ops, estimate_n_kernels,
    predict_pipe_baseline, predict_for_model_yaml, assign_confidence,
    fit_host_gap, fit_kernel_gap, leave_one_model_out_cv, fit_all_and_save,
)

# sweep
from prism.sweep.runner import run_sweep, predict_wallclock_v3, MODELS, SWEEP
from prism.sweep.timeloop_problem import convert_op_to_timeloop_problem

# mapper
from prism.mapper.generate import generate_mapping, ARCH_910B4
from prism.mapper.runner import run_timeloop_model

# ceiling
from prism.ceiling.predict import (
    predict_all_scenarios, WallClockBreakdown, ScenarioResult,
    compute_baseline, compute_software_ceiling, compute_software_runtime_ceiling,
    compute_hw_ub_l1_fused, compute_hw_ub_l1_fused_hbm3,
)

# reports
from prism.reports.render import render_all, render_template
```

CLI 入口对应 `:main` 函数（详见 [cli.md](cli.md)）。
