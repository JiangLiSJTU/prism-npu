# Arch YAML Schema

工具用 3 种 arch yaml，按用途派生：

| 文件 | 用途 | locked? |
|------|------|--------|
| `arch/ascend_910b4.yaml` | Timeloop+Accelergy 兼容（含 CACTI 限制 spec）| **是**（不修改）|
| `arch/ascend_910b4_for_mapping.yaml` | manual mapping cycles 校准（L2 depth=96 MB 真实值）| 是（派生于 baseline）|
| `arch/ascend_910b4_for_sweep_v2.yaml` | sweep + ceiling 工具的 baseline + 12 个细粒度字段 | 是 |

本文以 `_for_sweep_v2.yaml` 为参考。

---

## 1. 完整字段表

### 1.1 既有字段（baseline）

| 字段 | 类型 | 单位 | 默认 (910B4) | 含义 |
|------|------|------|------------:|------|
| `version` | int | - | 2.0 | schema 版本（与 sweep v3 对应）|
| `n_cores` | int | - | 24 | AICore 数 |
| `cube_m` / `cube_n` / `cube_k` | int | - | 16 / 16 / 16 | Cube spatial（每核 4096 MAC）|
| `hbm_bw_gbs` | float | GB/s | 392 | HBM/LPDDR 带宽 |
| `l2_mb` | float | MB | 96 | L2 共享缓存大小 |
| `clock_ghz` | float | GHz | 1.6 | 时钟频率 |
| `fp16_tflops` | float | TFLOPS | 280 | FP16 算力 |
| `tdp_w` | float | W | 300 | TDP |

### 1.2 N6a 新增字段（细粒度 sweep）

| 字段 | 类型 | 单位 | 默认 | 含义 | 来源 |
|------|------|------|-----:|------|------|
| `l0a_kb` / `l0b_kb` / `l0c_kb` | int | KB | 64 / 64 / 128 | L0 缓存大小（per AIC）| HC31 |
| `l0a_banks` / `l0b_banks` / `l0c_banks` | int | - | 4 / 4 / 8 | L0 bank 数 | HC31 |
| `l1_kb` | int | KB | 512 | L1 缓存大小（per AIC）| CANN .ini |
| `l1_l0_bw_gbs` | float | GB/s | 2048 | MTE1 throughput (L1 ↔ L0A/B) | 经验估值 |
| `fixpipe_bw_gbs` | float | GB/s | 4096 | FixPipe (L0C → L1/UB) | 经验估值 |
| `aiv_per_aic` | int | - | 2 | 每 AIC 配几个 AIV | HC31 |
| `ub_kb_per_aiv` | int | KB | 192 | UB scratchpad（per AIV）| HC31 |
| `ub_l1_bw_gbs` | float | GB/s | 2048 | AIV MTE2 (UB ↔ L1) | 经验估值 |
| `aiv_lanes_per_aiv` | int | - | 128 | FP16 SIMD lanes/AIV | HC31 |
| `aiv_mte3_bw_gbs` | float | GB/s | 1024 | AIV MTE3 (UB → output) | 经验估值 |
| `beta_host_gap_us_per_kernel` | float | μs | 41.6 | host scheduling per kernel（model-class 可覆盖）| BERT b=1 实测 |
| `pipe_overlap_factor` | float | - | 0.6 | 多 pipe 并行 effective speedup | 经验值 |

### 1.3 Sweep 维度 metadata（注释，不参与公式）

下面 `*_variants` 是 sweep 工具默认覆盖的 variant 集，写在 yaml 注释里供文档化（运行时由 `prism.sweep.runner.SWEEP` dict 决定）：

```yaml
# n_cores_variants:        [8, 12, 16, 24, 32, 48]
# cube_kdim_variants:      [(8,8,16), (8,16,16), (16,16,16), (16,16,32), (32,32,16)]
# l2_mb_variants:          [8, 16, 32, 96, 192, 384]
# hbm_bw_gbs_variants:     [50, 100, 392, 800]
# aiv_per_aic_variants:    [1, 2, 4]
# tdp_w_variants:          [100, 150, 200, 300, 400]
# l0a_kb_variants:         [32, 64, 128, 256]
# l1_kb_variants:          [256, 512, 1024]
# l1_l0_bw_gbs_variants:   [1024, 2048, 4096]
# fixpipe_bw_gbs_variants: [2048, 4096, 8192]
# ub_l1_fused_variants:    [false, true]
# beta_host_gap_us_per_kernel_variants: [10, 41.6, 100]
```

---

## 2. 经验估值字段的物理依据

| 字段 | 估值方法 | 不确定度 |
|------|---------|---------|
| `l1_l0_bw_gbs = 2048` | DaVinci HC31 推算 + msprof `bytes / aic_mte1_time` 反推 | ±20% |
| `fixpipe_bw_gbs = 4096` | HC31 推算 + msprof `bytes_l0c / aic_fixpipe_time` 反推 | ±20% |
| `ub_l1_bw_gbs = 2048` | HC31 推算 + msprof `aiv_mte2_time` 反推 | ±20% |
| `aiv_mte3_bw_gbs = 1024` | HC31 推算 | ±30% |
| `pipe_overlap_factor = 0.6` | 9 配置 aic_bubble% 经验值 (10-33%) | ±0.1 |

→ 这些估值的不确定度直接传入 sweep ratio。详见 [methodology/06_assumptions_limits.md §2.5](../methodology/06_assumptions_limits.md#25-vector-op-用-analytical-pipe-model非-timeloop)。

---

## 3. 派生新 arch yaml

### 3.1 用途场景

| 场景 | 推荐文件名 |
|-----|----------|
| 假想 chip 变体（如 LPDDR5X baseline）| `ascend_910b4_lpddr5x.yaml` |
| 假想 chip 加 prefetcher | `ascend_910b4_with_prefetcher.yaml` |
| 跨芯片对比（实测后）| `ascend_310p_for_sweep_v2.yaml` |

### 3.2 派生模板

```yaml
# 1. 声明派生关系
# 派生自：arch/ascend_910b4_for_sweep_v2.yaml
# 创建日期：2026-XX-XX
# 用途：测试 LPDDR5X 替换 HBM2e 的 sweet spot

architecture:
  version: 2.0    # 与 sweep v3 一对一对应

# 2. 既有字段（保持 baseline 同等）
n_cores: 24
cube_m: 16
cube_n: 16
cube_k: 16
clock_ghz: 1.6
fp16_tflops: 280

# 3. 改动字段（重点）
hbm_bw_gbs: 100      # ← 改：LPDDR5X 100 GB/s（vs HBM2e 392）
l2_mb: 32            # ← 改：缩 L2 配合 LPDDR
tdp_w: 200           # ← 改：低功耗版

# 4. 其它细粒度字段（保持 baseline）
l0a_kb: 64
... (与 baseline 相同)
```

→ sweep 用 `--arch-baseline arch/ascend_910b4_lpddr5x.yaml`（详见 [tutorials/02 §7](../tutorials/02_reproduce_arch_sweep.md#7-自定义-baseline-arch如换-lpddr5x-试一组对照)）。

---

## 4. 规则

### 4.1 字段必须全列出

`predict_wallclock_v3` 假设所有字段都在 yaml 中。缺失会触发 KeyError。如要省略某字段（如不关心 fixpipe），用 baseline 默认值显式列出。

### 4.2 类型必须正确

`int` 字段不能用 float（如 `l0a_kb: 64.0` ❌；应当 `64`）。yaml 解析后类型保留。

### 4.3 BW 单位统一为 GB/s

所有 BW 类字段（`hbm_bw_gbs`、`l1_l0_bw_gbs` 等）都以 GB/s 为单位。**禁用** GB·s 或 byte/cycle。

### 4.4 derived field 在 runner 内计算

某些"组合"字段不在 yaml 内，由 `runner.py` 计算：

```python
arch.cube_total_macs = n_cores × cube_m × cube_n × cube_k       # 总 MAC 数
arch.aiv_throughput_ops_per_cycle = n_cores × aiv_per_aic × aiv_lanes_per_aiv
```

如未来需要把这些当 first-class 字段，应在 `runner.py` 加 dataclass 转换。

---

## 5. 不要修改

| 文件 | 原因 |
|------|------|
| `arch/ascend_910b4.yaml` | Timeloop+Accelergy 兼容标准（CACTI 限制 + 锁定）|
| `arch/ascend_910b4_for_mapping.yaml` | manual mapping 校准用，L2 depth 不可改 |

如要改架构假设，**派生新 yaml**（参见 §3.2）。

---

## 6. 与 [methodology/04_arch_sensitivity.md] 的对应

| arch 字段 | sweep 维度 | 影响的 pipe |
|----------|-----------|-----------|
| `n_cores` | n_cores | aic_pipe.mac |
| `cube_m/n/k` | cube_spatial | aic_pipe.mac |
| `clock_ghz` | clock | 全 device 时间 |
| `hbm_bw_gbs` | hbm_bw_gbs | aic_pipe.mte2 |
| `l1_l0_bw_gbs` | l1_l0_bw_gbs | aic_pipe.mte1 |
| `fixpipe_bw_gbs` | fixpipe_bw_gbs | aic_pipe.fixpipe + aiv_pipe.mte3 |
| `aiv_per_aic` × `aiv_lanes_per_aiv` × `clock_ghz` | aiv_per_aic | aiv_pipe.vec |
| `ub_l1_bw_gbs` | ub_l1_bw_gbs | aiv_pipe.mte2 |
| `(virtual) ub_l1_fused` | ub_l1_fused | aiv_pipe.mte2 × 0.05 |

详见 [methodology/04_arch_sensitivity.md §3](../methodology/04_arch_sensitivity.md#3-12-个-sweep-维度) + [§4.1](../methodology/04_arch_sensitivity.md#41-pipe-aware-predict_wallclock)。
