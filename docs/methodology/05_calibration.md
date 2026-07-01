# msprof 校准流程

## 1. 校准做什么

工具的所有预测系数（`β_device`、`β_layer`、η_real 5 参数、per-model pipe baseline）都来自昇腾 910B4 实卡 msprof 实测。本文描述：

1. msprof 数据采集 pipeline（NPU 侧）
2. msprof 字段词典（解析时的语义）
3. 数据提取脚本（`prism-extract`）
4. 拟合脚本（`prism-fit`）
5. 失败模式 + 应对（msprof analyze 大 workload 崩溃）
6. regime gate 校验

## 2. 数据采集 pipeline

```
模型 (HF)  →  ONNX export  →  ATC convert  →  OM file
                                                  │
                                                  ▼
                                            ais_bench (推理驱动)
                                                  │  + msprof 包裹
                                                  ▼
                                msprof PROF_*/op_summary*.csv
                                                  │
                                                  ▼ rsync 拉回本地
                                  Local msprof_data/
                                                  │
                                                  ▼ prism-extract
                                  data/calibration/{cube_util,pipe_baseline}.json
```

每一步对应的脚本：

| 步骤 | 脚本 | 输出 |
|-----|------|------|
| ONNX export | `benchmark/export_qwen3_prefill.py` (含 3 patches: sdpa_mask BC, torch.diff, cumsum BOOL) | `models/<model>.onnx` |
| ATC convert | `benchmark/convert_qwen3_prefill_om.sh` | `om/<model>.om` |
| msprof 采集 | `benchmark/run_phase_b.sh` (4 metrics × N config) | `msprof_<model>_b<B>_<metric>/PROF_*/` |
| 提取 | `prism-extract` | `data/calibration/cube_util_extracted.json` |
| 拟合 | `prism-fit` | `data/calibration/eta_physics_fit.json` |

### 2.1 msprof 命令模板

```bash
msprof --application="python3 -m ais_bench --model om/<model>_b<B>.om \
                                            --loop <L> --warmup_count <W>" \
       --output=./<outdir> \
       --task-time=on --ai-core=on --aicpu=on --runtime-api=on --hccl=off \
       --aic-metrics=<metric>            # 4 选 1：ArithmeticUtilization / PipeUtilization / Memory / L2Cache
       --l2=on
```

`--aic-metrics` 决定 op_summary CSV 含哪些字段：

| metric | 主要新增字段 | 用途 |
|--------|-----------|------|
| `ArithmeticUtilization` | `aic_mac_fp16_ratio` 等 | η_real 拟合（cube util 主要源）|
| `PipeUtilization` | `aic_mte1/2_ratio`, `aic_fixpipe_ratio`, `aiv_*_time` 等 | pipe-aware sweep + ceiling 工具 |
| `Memory` | DRAM read/write bytes | bytes 实测，验证 mte2 公式 |
| `L2Cache` | L2 hit rate | L2 容量与 mte2 关系 |

完整 metric 列表见 CANN 8.5 msprof 文档。

> **采集陷阱：`--output` 目录权限。** msprof 出于安全原因拒绝 group-writable 的 `--output`
> 目录。`mkdir -p` 在默认 umask 022 下创建的目录是 755（group 可写），会触发
> `[ERROR] profiling error Argument --output=... is writable by groups`。在 `mkdir`
> 之后立即 `chmod 750`：
>
> ```bash
> mkdir -p "$outdir"
> chmod 750 "$outdir"        # msprof 要求 not group-writable
> msprof --output="$outdir" ...
> ```

### 2.2 推荐采集模板（11 配置）

工具 baseline 用了 11 配置 PipeUtilization：

| 模型 | 配置 | loop / warmup |
|------|------|--------------|
| BERT-base | b=1 | 30 / 5 |
| GPT-2-small | b=1 | 20 / 5 |
| Qwen3-prefill | S=256 b=1/4/8 | 10 / 2 |
| Qwen3-prefill | S=512 b=4/8 | 8 / 3 |
| Qwen3-prefill | S=4096 b=1 | 5 / 1 |
| Qwen3-decode | M=4 S_kv=128 b=1 | 20 / 5 |

参考脚本 `benchmark/run_phase_b.sh`、`benchmark/run_pipeutil_supplement.sh`。

## 3. msprof Pipe 字段词典

每个 op 一行 CSV，46 个字段。本工具消费的关键字段：

### 3.1 通用字段

| 字段 | 类型 | 含义 |
|------|------|------|
| `Op Name` | str | op 实例名（如 `/encoder/layer_0/attention/self/key/MatMul`）|
| `OP Type` | str | op 类型（`MatMul`, `BatchMatMulV2`, `LayerNorm`, `Softmax`, ...）|
| `Task Duration(us)` | float | 单 op 总耗时（含 idle）|
| `Task Wait Time(us)` | float | op 在 launch queue 的等待时间 |
| `Block Dim` | int | spatial block 数（cube spatial）|
| `Input Shapes` / `Output Shapes` | str | 形如 `1,128,768;1,768,768;` 的 shape 串 |

### 3.2 AIC（AI Core）端字段

`aicore_time(us)` = AIC 单元活跃总时间，**不含 AIV**。

| 字段 | 含义 | 与 arch 维度的对应 |
|------|------|-------------------|
| `aicore_time(us)` | AIC 活跃总时间 | — |
| `aic_total_cycles` | AIC 总 cycle 数 | — |
| `aic_mac_time(us)` / `aic_mac_ratio` | Cube MAC 单元活跃 | n_cores × cube_macs |
| `aic_mac_fp16_ratio` / `aic_mac_int8_ratio` | MAC 按数据类型分 | — |
| `aic_scalar_time(us)` / `aic_scalar_ratio` | AIC scalar 控制流 | — |
| `aic_mte1_time(us)` / `aic_mte1_ratio` | L1 → L0A/L0B 数据搬运 | l1_l0_bw |
| `aic_mte2_time(us)` / `aic_mte2_ratio` | DRAM/L2 → L1 数据搬运 | hbm_bw、l2_mb |
| `aic_fixpipe_time(us)` / `aic_fixpipe_ratio` | L0C → 输出回写（L1/UB 或 GM 直写）| hbm_bw + fixpipe_bw（按 `gm_frac` blend，见 §3.3 注）|
| `aic_icache_miss_rate` | AIC instruction cache miss | （高阶研究）|

**关键理解**：`aic_*_ratio` = `aic_*_time / aicore_time`。这些 pipe **可以并行**，所以 Σ ratio 可以 > 1（多 pipe 重叠）。

`aic_bubble = aicore_time - max(active pipe time)` = 全 pipe 都 idle 的时间。

### 3.3 AIV（AI Vector）端字段

`aiv_time(us)` = AIV 单元活跃总时间，**独立于 aicore_time**。整 wall-clock 中 AIV 的贡献是 aiv_time，要单独建模（这是经典 Roofline 漏掉的一项）。

| 字段 | 含义 | 与 arch 维度的对应 |
|------|------|-------------------|
| `aiv_time(us)` | AIV 活跃总时间 | — |
| `aiv_vec_time(us)` / `aiv_vec_ratio` | Vector ALU SIMD 计算 | aiv_per_aic × vector_lanes |
| `aiv_scalar_time(us)` / `aiv_scalar_ratio` | AIV scalar 控制流 | — |
| `aiv_mte2_time(us)` / `aiv_mte2_ratio` | UB ↔ L1 数据搬运 | ub_l1_bw |
| `aiv_mte3_time(us)` / `aiv_mte3_ratio` | UB → 输出 写回（MTE3 引擎）| hbm_bw + ub_l1_bw（按 `gm_frac` blend，见下注）|
| `aiv_icache_miss_rate` | AIV instruction cache miss | — |

> **`aic_fixpipe` / `aiv_mte3`：双目的地，按 `gm_frac` blend 校准（Issue #7）。**
> 这两条 pipe 都把计算结果搬出，瓶颈带宽**取决于目的地**：
> - `aic_fixpipe`：L0C → {L1/UB 片上（`fixpipe_bw`）| **GM 直写**（`hbm_bw`）}
> - `aiv_mte3`：UB → {L1 片上（`ub_l1_bw`，`copy_ubuf_to_cbuf`）| **GM**（`hbm_bw`，`copy_ubuf_to_gm`）}
>
> FixPipe 是 AIC 侧 L0C→输出专属单元，AIV 不经过它（`aiv_mte3` 早期误用 `fixpipe_bw`，
> 已修正）。
>
> **校准方法**（`scripts/calib_fixpipe_mte3_bw.py` → `data/calibration/pipe_dest_bw.json`）：
> msprof 只报聚合 `*_time`。**Prior-based 2-cluster 分类**：逐 op 算 implied 带宽
> `bytes/time`，按物理先验阈值 `sqrt(hbm_bw · onchip_bw)`（两个带宽区域的几何中点）
> 切分——低于阈值划入 GM 簇、高于阈值划入片上簇；`gm_frac` = GM 簇的**字节占比**。
> 每簇再做一遍 OLS（斜率作为 sanity check，应分别落在 `hbm_bw` 与 onchip 量级）。
>
> 此方法**对所有模型普世**：真双峰 config 干净分簇；单峰 config 自然退化为 `1cluster`
> （一个簇为空）。回避了两种朴素方法的失效模式：
> - `Σ字节/Σtime` 聚合：被 per-op 固定开销污染（同 3 MB GatherV2 实测 mte3 时间差 11×）。
> - 单一 pooled OLS：双峰数据给出 leverage-weighted 混合斜率，反解的 `gm_frac` 有偏。
>
> **实测结论**（39 config）：
> - `aic_fixpipe`：**多数 config GM-bound**（GM 簇 OLS 斜率 240–820 GB/s，HBM 量级），
>   `gm_frac` 0.42–1.00 —— FixPipe 输出主要是 L0C→GM 直写，不是 FixPipe 单元带宽 bound。
> - `aiv_mte3`：长上下文大 prefill `gm_frac` 0.50–0.85；小短序列模型 `gm_frac` ≤ 0.15。
> - 个别 config（如 HF-BERT-b8 fixpipe）真双峰（GM 簇 ~363 GB/s + 片上簇 ~1000 GB/s），
>   单 OLS 会给出偏 0.21 的 `gm_frac`，2-cluster 修正为 0.78。
>
> **局限**：(1) 阈值是物理先验（HBM/onchip 的几何中点）；若实际 HBM 写带宽与
> 标称 392 显著偏离，可能 ±10% 边界误判（实测 GM 簇斜率多落 250–550，是合理范围）。
> (2) `confidence=low` 的 config（小 host-bound 模型，pipe 被固定开销主导、本就非瓶颈）
> `gm_frac` 估计稳健性较差，但其该 pipe 非杠杆。(3) 9 个早期 config 无本地 msprof，用
> 近邻同族继承（标在 JSON `source` 字段）。(4) `gm_frac` 视为 arch-invariant。

`cube_utilization(%)` 字段是 AIC 报告的官方 cube_util（≈ `aic_mac_ratio`）。

### 3.4 wall-clock 与 device-time 关系

```
wall_clock_per_inference =
    Σ_kernel( task_duration )                 # 单 kernel 总时间
  + kernel_间_host_gap                         # kernel launch / dispatch
  + 同步等待                                    # D2H sync / event wait

task_duration_per_kernel =
    max( aicore_time, aiv_time )               # AIC 与 AIV 在 kernel 内 serial
  + kernel_内_pipe_gap                          # AIC/AIV 之间等待

aicore_time_per_kernel =
    overlapped( mac, mte1, mte2, fixpipe, scalar )
```

→ 当前 `β_layer = wall_clock - aicore_time` 的拆分包含 4 类完全不同性质的开销：
1. 纯 host：kernel launch / graph dispatch / D2H 同步
2. NPU 流水线气泡（MTE1/MTE2/MTE3 stall）
3. Cube/Vector pipeline gap（互相等待）
4. Scalar / 控制开销

工具的 pipe-aware 公式（[02_three_layer_roofline.md §5](02_three_layer_roofline.md#5-t_aic-的-pipe-aware-拆分)）就是为了把这 4 类拆开。

## 4. 提取脚本（`prism-extract`）

```bash
prism-extract --msprof-dir msprof_data/msprof_qwen3_06b_b1_PipeUtilization \
            --output    data/calibration/pipe_baseline_per_model.json
```

输出 JSON 结构：

```json
{
  "configs": {
    "BERT-base-S128-b1": {
      "n_kernels_per_inf": 338,
      "task_dur_us": 2131,
      "aic_time_us": 651,
      "aiv_time_us": 918,
      "aic_pipes_us": {"mac": 192, "mte1": 224, "mte2": 517, "fixpipe": 76, "scalar": 76},
      "aiv_pipes_us": {"vec": 76, "mte2": 216, "mte3": 86, "scalar": 117, "idle": 423},
      "wall_clock_us": 16210,
      "host_gap_us": 14079
    },
    ...
  }
}
```

工具内部把 ArithmeticUtilization 与 PipeUtilization 两次采集**联合**：前者给 cube_util fp16 ratio（用于 η_real fit），后者给所有 pipe time（用于 sweep）。

### 4.1 op shape 解析

`Input Shapes` 字段的格式：`1,128,768;1,768,768;`（分号分隔的 input shape 列表）。GEMM 算子的解析：

```python
# MatMul: input0 = (B, M, K), input1 = (B, K, N) → 取出 M, N, K
# BatchMatMul: 类似
# 工具自动按 op_kind 区分（详见 03_eta_real_model.md §4）
```

### 4.2 失败配置过滤

`extract` 会跳过 `cube_util_pct <= 0` 或 `count < 20` 的 op（数据不可信）。配置完全无 op_summary CSV 时（msprof analyze 崩溃），整配置 skip。

## 5. 拟合脚本（`prism-fit`）

```bash
prism-fit --cube-util-json data/calibration/cube_util_extracted.json \
        --output         data/calibration/eta_physics_fit.json
```

5 参数 Levenberg-Marquardt 拟合（详见 [03_eta_real_model.md §5](03_eta_real_model.md#5-levenberg-marquardt-拟合)）。

输出含训练 + 验证 MAE：

```json
{
  "method": "physics-informed (η_pipeline · η_tile · batch)",
  "params": {"alpha_MN_coupling": 14.5977, ...},
  "training": {"n": 56, "mae_pp": 11.98},
  "validation": {
    "bert": {"n": 16, "mae_pp": 14.33},
    "gpt2": {"n": 16, "mae_pp": 12.07}
  }
}
```

**硬门槛**：BERT 验证 MAE 必须 < 15 pp 才算合格。该值由用户设定，CI 应阻断失败 PR。

## 6. msprof 失败模式 + 应对

实测过程中遇到 4 类崩溃：

### 6.1 ATC convert: Cumsum BOOL（已修复）

ATC 不支持 `Cumsum<DT_BOOL>`。`transformers/masking_utils.py:1002` 的 `attention_mask.cumsum(-1)` 触发。

**修复**：`benchmark/export_qwen3_prefill.py` 加 `_patch_cumsum_bool_cast()` monkey-patch `torch.Tensor.cumsum` 自动 cast bool → int32。

### 6.2 ATC convert: M=1 attention 融合崩溃

decode workload 用 `input_ids: (B, 1)` 时 ATC 在 `MatMul_to_tranpose_batch_matmul` 算子融合阶段崩。

**应对**：用 M=4（多 token decode 模拟）。仍能验证 KV cache reload 的 BW pattern，但严格 single-token 路径未覆盖。

### 6.3 msprof analyze: 大 workload 崩溃

S=4096 b=8、S=8192 b=1 的 PipeUtilization 在 analyze 阶段崩溃（OOM 或超时）。loop=5/2/1 都试过仍崩。

**应对**：
- 接受这两个配置数据缺失
- 用更小 S/B 配置外推（fit 时这两个配置不在数据集中）
- 后续如必要，可分步采集（loop=1 + 多次 trace 拼接）

### 6.4 msprof step_trace 缺失

decode workload 的 step_trace CSV 偶尔缺失（msprof bug）。`wall_clock` 来自 ais_bench 报告替代。

## 7. regime gate 校验流程

`prism-regime` 输入 (model.yaml, arch.yaml, batch)，输出 regime + Timeloop 必要性 flag：

```bash
prism-regime --arch arch/ascend_910b4_for_sweep_v2.yaml \
           --model models/qwen3_0.6b.yaml --batch 1
```

输出：

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

regime 4 类阈值（见 [02 §8](02_three_layer_roofline.md#8-regime-分类decision-gate)）：

```
IF T_overhead > 2 × max(T_compute, T_memory)        → "host-bound"
ELIF T_compute > 2 × T_memory                       → "compute-bound"
ELIF T_memory > 2 × T_compute                       → "memory-bound"
ELSE                                                → "balanced"
```

`--sweep` 选项可对所有 (model, arch_variant) 组合做 sweep，输出 `data/outputs/regime_matrix.json`：

```bash
prism-regime --sweep --output data/outputs/regime_matrix.json
```

→ 见 [docs/findings/微架构探索报告.md](../findings/微架构探索报告.md) §regime 矩阵。

## 8. 复现性保证

`pip install -e ".[dev]"` 后任何人重跑：

```bash
# 1) η_real 拟合
prism-fit
# 期望输出：BERT validation MAE 14.33 pp（与 reference 一致）

# 2) sweep
prism-sweep
# 期望输出：49 variants × baseline 全部 model 的 ratio 与 docs/findings/微架构探索报告.md 一致
#          （默认遍历 pipe_baseline 全部 config；可用 --test-models 限定子集）

# 3) ceiling
prism-ceiling
# 期望输出：5 scenarios × 11 configs 的 ratio 与 docs/findings/optimization_ceiling.md 一致

# 4) render
prism-render --check
# 期望：exit 0（4 templates identical）
```

任意 PR 必须 4 项全过。

## 9. 已知局限

| # | 局限 | 影响 |
|---|------|------|
| 1 | 大 workload (S=4096 b=8, S=8192 b=1) PipeUtilization 缺失 | 极限长上下文外推依赖 S=4096 b=1 |
| 2 | step_trace 偶尔缺失，wall_clock fallback ais_bench | wall_clock 误差 ~1% |
| 3 | msprof aic_mac_ratio 只反映 fp16 | int8 / fp32 量化场景需用 aic_mac_int8_ratio 等其它字段 |
| 4 | 仅在 910B4 上校准 | 310P 上 β_layer / η_real 是否同公式有效，需独立采集（[06 §4](06_assumptions_limits.md#4-跨芯片外推)）|

完整局限：[06_assumptions_limits.md](06_assumptions_limits.md)

---

## 📚 参考

- 实验环境：`legacy/docs/server_env_910b4.md`（910B4 NPU 服务器手册）
- Pipe 字段语义验证：`legacy/docs/overhead_decomposition_audit.md` v1.1 §1
- 失败模式记录：`legacy/docs/cube_efficiency_calibration.md` §6
- CANN msprof 文档：内部 + https://www.hiascend.com/document/redirect/CannCommercialDocs（公开版）
- 解析源代码：`src/prism/eta_real/extract.py`、`src/prism/eta_real/match.py`
