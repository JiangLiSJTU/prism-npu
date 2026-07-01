#!/usr/bin/env python3
"""
算子 / 软件 / 硬件优化天花板预测工具（Phase N 工具组件）

用 msprof PipeUtilization 实测的 pipe breakdown 作为输入，预测在不同优化层级下
该 workload 的 wall-clock 上限（"理论最优"），帮助回答：
  - "现有 CANN TBE 算子库还有多少优化空间？"（S1 ceiling）
  - "算子 + CANN runtime 一起优化能到什么程度？"（S2 ceiling）
  - "如果再加硬件改动（UB+L1 融合、HBM3）还能再降多少？"（S3/S4 ceiling）

输入：data/pipe_baseline_per_model.json（Phase N 9 配置实测 + 占位）
输出：
  data/optimization_ceiling.json — 5 情景 × 11 配置 × 详细分解
  docs/operator_optimization_ceiling.md — 人读分析报告

Author note (M1 集成路径)：本工具是发布前阻塞工作 Phase N 的产物。
工具发布 M1 阶段会把核心逻辑迁到 src/prism/ceiling/predict.py，
此处脚本改为薄 CLI 包装。当前位置临时放在 scripts/calibration/。

Usage:
    python3 scripts/calibration/predict_optimization_ceiling.py \\
        --pipe-baseline data/pipe_baseline_per_model.json \\
        --output-json   data/optimization_ceiling.json \\
        --output-md     docs/operator_optimization_ceiling.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent.parent.parent

# ─────────────────────────────────────────────────────────────────────
# 优化情景定义
# ─────────────────────────────────────────────────────────────────────
HOST_GAP_TARGET_US_PER_KERNEL = 10.0   # CANN runtime 优化目标（来自工业经验）
UB_L1_FUSED_RESIDUAL = 0.05            # UB+L1 融合后保留 5% 残余（控制流路径）
HBM3_BW_GBS = 800
HBM2E_BW_GBS_BASELINE = 392


@dataclass
class WallClockBreakdown:
    """一个情景下 per-inference wall-clock 的完整分解。

    Note: ``wall_clock_us`` is the authoritative total (already computed by the
    scenario function with possible bottleneck reasoning, not just additive sum).
    Removed the ``total`` property in favor of using ``wall_clock_us`` directly.
    """
    aic_time_us:    float
    aiv_time_us:    float
    kernel_gap_us:  float
    host_gap_us:    float
    wall_clock_us:  float


@dataclass
class ScenarioResult:
    """一个 (model, scenario) 元组的预测结果。"""
    scenario:      str
    description:   str
    wall_clock:    WallClockBreakdown
    reduction_pct: float = 0.0           # 相对 baseline 的降幅
    aic_pipes:     dict[str, float] = field(default_factory=dict)
    aiv_pipes:     dict[str, float] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────
# Per-scenario 计算函数
# ─────────────────────────────────────────────────────────────────────
def compute_baseline(pipe: dict[str, Any]) -> ScenarioResult:
    """S0: msprof 实测。"""
    aic_us = pipe['aic_time_us']
    aiv_us = pipe['aiv_time_us']
    kernel_gap = max(0.0, pipe.get('kernel_gap_us', 0))
    host_gap = pipe.get('host_gap_us', 0)
    return ScenarioResult(
        scenario='S0_baseline',
        description='msprof 实测，无任何优化',
        wall_clock=WallClockBreakdown(
            aic_time_us=aic_us,
            aiv_time_us=aiv_us,
            kernel_gap_us=kernel_gap,
            host_gap_us=host_gap,
            wall_clock_us=aic_us + aiv_us + kernel_gap + host_gap,
        ),
        aic_pipes=dict(pipe['aic_pipes_us']),
        aiv_pipes=dict(pipe['aiv_pipes_us']),
    )


def compute_software_ceiling(pipe: dict[str, Any]) -> ScenarioResult:
    """S1: 算子完美 ping-pong + 完美指令调度。

    操作：
      - AIC pipes: 完美 overlap → aic_time = max(active pipes) (无 bubble)
      - AIV pipes: 完美 overlap → aiv_time = max(active pipes) (无 idle)
      - kernel_gap → 0 (AIC/AIV 完美异步)
      - host_gap 不变（属 runtime 范畴，S2 才动）
    """
    aic_pipes = pipe['aic_pipes_us']
    aiv_active = {k: v for k, v in pipe['aiv_pipes_us'].items() if k != 'idle'}

    aic_time_new = max(aic_pipes.values())
    aiv_time_new = max(aiv_active.values()) if aiv_active else 0.0
    return ScenarioResult(
        scenario='S1_software_ceiling',
        description='CANN TBE 算子完美双缓冲 + 完美指令调度（消除 bubble/idle/kernel_gap）',
        wall_clock=WallClockBreakdown(
            aic_time_us=aic_time_new,
            aiv_time_us=aiv_time_new,
            kernel_gap_us=0.0,
            host_gap_us=pipe.get('host_gap_us', 0),
            wall_clock_us=aic_time_new + aiv_time_new + pipe.get('host_gap_us', 0),
        ),
        aic_pipes=dict(aic_pipes),
        aiv_pipes=dict(aiv_active),
    )


def compute_software_runtime_ceiling(pipe: dict[str, Any],
                                      host_gap_target_per_kernel: float = HOST_GAP_TARGET_US_PER_KERNEL
                                      ) -> ScenarioResult:
    """S2: S1 + CANN graph optim + async dispatch（host_gap 降到 target，只降不升）。

    关键修正：min(baseline, target) — 如某 workload 已在 target 下（如 Qwen3 decode
    的 0.16 μs/kernel），保持 baseline 不变；不能反向"优化"出更差的 host gap。
    """
    n_kernels = pipe['n_kernels_per_inf']
    s1 = compute_software_ceiling(pipe)
    baseline_host_gap = pipe.get('host_gap_us', 0)
    target_host_gap = n_kernels * host_gap_target_per_kernel
    new_host_gap = min(baseline_host_gap, target_host_gap)
    return ScenarioResult(
        scenario='S2_software_runtime_ceiling',
        description=f'S1 + CANN runtime 优化（host_gap 降到 {host_gap_target_per_kernel} μs/kernel，已低于该值则保持）',
        wall_clock=WallClockBreakdown(
            aic_time_us=s1.wall_clock.aic_time_us,
            aiv_time_us=s1.wall_clock.aiv_time_us,
            kernel_gap_us=0.0,
            host_gap_us=new_host_gap,
            wall_clock_us=s1.wall_clock.aic_time_us + s1.wall_clock.aiv_time_us + new_host_gap,
        ),
        aic_pipes=s1.aic_pipes,
        aiv_pipes=s1.aiv_pipes,
    )


def compute_hw_ub_l1_fused(pipe: dict[str, Any],
                            host_gap_target_per_kernel: float = HOST_GAP_TARGET_US_PER_KERNEL
                            ) -> ScenarioResult:
    """S3: S2 + 硬件 UB+L1 融合（aiv_mte2 → 5% 残余）。"""
    aic_pipes = pipe['aic_pipes_us']
    aiv_active_orig = {k: v for k, v in pipe['aiv_pipes_us'].items() if k != 'idle'}
    aiv_active_fused = dict(aiv_active_orig)
    aiv_active_fused['mte2'] = aiv_active_orig.get('mte2', 0) * UB_L1_FUSED_RESIDUAL

    aic_time = max(aic_pipes.values())
    aiv_time = max(aiv_active_fused.values()) if aiv_active_fused else 0.0
    n_kernels = pipe['n_kernels_per_inf']
    baseline_host_gap = pipe.get('host_gap_us', 0)
    host_gap = min(baseline_host_gap, n_kernels * host_gap_target_per_kernel)
    return ScenarioResult(
        scenario='S3_hw_ub_l1_fused',
        description=f'S2 + 硬件 UB+L1 融合（aiv_mte2 × {UB_L1_FUSED_RESIDUAL}）',
        wall_clock=WallClockBreakdown(
            aic_time_us=aic_time,
            aiv_time_us=aiv_time,
            kernel_gap_us=0.0,
            host_gap_us=host_gap,
            wall_clock_us=aic_time + aiv_time + host_gap,
        ),
        aic_pipes=dict(aic_pipes),
        aiv_pipes=aiv_active_fused,
    )


def compute_hw_ub_l1_fused_hbm3(pipe: dict[str, Any],
                                 host_gap_target_per_kernel: float = HOST_GAP_TARGET_US_PER_KERNEL,
                                 hbm3_bw_gbs: float = HBM3_BW_GBS
                                 ) -> ScenarioResult:
    """S4: S3 + HBM3 800 GB/s（aic_mte2 × 392/800）。"""
    bw_scale = HBM2E_BW_GBS_BASELINE / hbm3_bw_gbs
    aic_pipes_orig = pipe['aic_pipes_us']
    aic_pipes_hbm3 = dict(aic_pipes_orig)
    aic_pipes_hbm3['mte2'] = aic_pipes_orig.get('mte2', 0) * bw_scale

    aiv_active_orig = {k: v for k, v in pipe['aiv_pipes_us'].items() if k != 'idle'}
    aiv_active_fused = dict(aiv_active_orig)
    aiv_active_fused['mte2'] = aiv_active_orig.get('mte2', 0) * UB_L1_FUSED_RESIDUAL

    aic_time = max(aic_pipes_hbm3.values())
    aiv_time = max(aiv_active_fused.values()) if aiv_active_fused else 0.0
    n_kernels = pipe['n_kernels_per_inf']
    baseline_host_gap = pipe.get('host_gap_us', 0)
    host_gap = min(baseline_host_gap, n_kernels * host_gap_target_per_kernel)
    return ScenarioResult(
        scenario='S4_hw_ub_l1_fused_hbm3',
        description=f'S3 + HBM3 {hbm3_bw_gbs} GB/s（aic_mte2 × {bw_scale:.3f}）',
        wall_clock=WallClockBreakdown(
            aic_time_us=aic_time,
            aiv_time_us=aiv_time,
            kernel_gap_us=0.0,
            host_gap_us=host_gap,
            wall_clock_us=aic_time + aiv_time + host_gap,
        ),
        aic_pipes=aic_pipes_hbm3,
        aiv_pipes=aiv_active_fused,
    )


# ─────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────
def predict_all_scenarios(pipe_baseline: dict[str, Any],
                          host_gap_target_per_kernel: float = HOST_GAP_TARGET_US_PER_KERNEL,
                          hbm3_bw_gbs: float = HBM3_BW_GBS) -> dict[str, dict[str, ScenarioResult]]:
    """对每个 model 跑全部 5 情景。"""
    results = {}
    for model_key, pipe in pipe_baseline.items():
        s0 = compute_baseline(pipe)
        s1 = compute_software_ceiling(pipe)
        s2 = compute_software_runtime_ceiling(pipe, host_gap_target_per_kernel)
        s3 = compute_hw_ub_l1_fused(pipe, host_gap_target_per_kernel)
        s4 = compute_hw_ub_l1_fused_hbm3(pipe, host_gap_target_per_kernel, hbm3_bw_gbs)

        # 计算 reduction_pct
        baseline_wall = s0.wall_clock.wall_clock_us
        for s in (s1, s2, s3, s4):
            s.reduction_pct = round(100 * (1 - s.wall_clock.wall_clock_us / baseline_wall), 2) if baseline_wall > 0 else 0

        results[model_key] = {
            'S0_baseline':                  s0,
            'S1_software_ceiling':          s1,
            'S2_software_runtime_ceiling':  s2,
            'S3_hw_ub_l1_fused':            s3,
            'S4_hw_ub_l1_fused_hbm3':       s4,
        }
    return results


def serialize_results(results: dict[str, dict[str, ScenarioResult]]) -> dict[str, Any]:
    """打包成 JSON 友好结构。"""
    out = {}
    for model_key, scenarios in results.items():
        out[model_key] = {
            scen_key: {
                'scenario':    scen.scenario,
                'description': scen.description,
                'wall_clock':  asdict(scen.wall_clock),
                'reduction_pct_vs_baseline': scen.reduction_pct,
                'aic_pipes_us': {k: round(v, 2) for k, v in scen.aic_pipes.items()},
                'aiv_pipes_us': {k: round(v, 2) for k, v in scen.aiv_pipes.items()},
            }
            for scen_key, scen in scenarios.items()
        }
    return out


def render_markdown_report(results: dict[str, dict[str, ScenarioResult]]) -> str:
    """生成人读 Markdown 报告。"""
    lines = []
    lines.append('# 算子 / 软件 / 硬件优化天花板分析报告')
    lines.append('')
    lines.append('**输出工具**：`scripts/calibration/predict_optimization_ceiling.py`')
    lines.append('**输入数据**：`data/pipe_baseline_per_model.json`（Phase N 9 配置 PipeUtil 实测 + 2 占位/继承）')
    lines.append('**输出 JSON**：`data/optimization_ceiling.json`')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 一、问题陈述')
    lines.append('')
    lines.append('给定一个 workload 的 msprof PipeUtilization 实测数据（含 AIC/AIV 各 pipe 占比），')
    lines.append('回答以下三类问题：')
    lines.append('')
    lines.append('1. **算子库优化天花板**：完美 ping-pong + 完美指令调度后，wall-clock 还能降多少？')
    lines.append('2. **CANN runtime 优化天花板**：再加 host_gap 优化（kernel launch + graph fusion + async dispatch）能降多少？')
    lines.append('3. **硬件改动可叠加增益**：UB+L1 融合 / HBM3 升级在算子优化天花板基础上还能再降多少？')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 二、5 情景定义')
    lines.append('')
    lines.append('| 情景 | 操作 | 物理依据 |')
    lines.append('|------|------|---------|')
    lines.append('| **S0 Baseline** | msprof 实测，无任何优化 | 现状 |')
    lines.append('| **S1 Software ceiling** | AIC bubble → 0；AIV idle → 0；kernel_gap → 0 | 完美双缓冲 + 完美指令调度 + 完美 AIC/AIV 异步 |')
    lines.append(f'| **S2 Software + Runtime ceiling** | S1 + host_gap → {HOST_GAP_TARGET_US_PER_KERNEL} μs/kernel | CANN graph fusion + ModelLoad 缓存 + async dispatch |')
    lines.append(f'| **S3 + UB+L1 融合**（硬件） | S2 + aiv_mte2 × {UB_L1_FUSED_RESIDUAL} | 融合内存池消除 UB↔L1 algorithmic traffic |')
    lines.append(f'| **S4 + HBM3 800 GB/s**（硬件） | S3 + aic_mte2 × {HBM2E_BW_GBS_BASELINE/HBM3_BW_GBS:.3f} | 与 Phase J v3 sweep 一致 |')
    lines.append('')
    lines.append('每个情景的 wall_clock 公式：')
    lines.append('')
    lines.append('```')
    lines.append('wall_clock = max(active_aic_pipes) + aic_bubble')
    lines.append('           + max(active_aiv_pipes) + aiv_idle')
    lines.append('           + kernel_gap + host_gap')
    lines.append('')
    lines.append('S1: 把 aic_bubble、aiv_idle、kernel_gap 置 0')
    lines.append('S2: + host_gap 降到 n_kernels × host_gap_target')
    lines.append('S3: + aiv_mte2 × 0.05')
    lines.append('S4: + aic_mte2 × (392/800)')
    lines.append('```')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 三、9 配置 + 占位 全 scenarios 结果（per inference μs）')
    lines.append('')
    lines.append('| Workload | S0 baseline | S1 sw | -%| S2 sw+rt | -%| S3 +UB融 | -%| S4 +HBM3 | -%|')
    lines.append('|---------|-----------:|-----:|----:|--------:|----:|--------:|----:|--------:|----:|')
    for model_key, scenarios in results.items():
        s0 = scenarios['S0_baseline'].wall_clock.wall_clock_us
        s1 = scenarios['S1_software_ceiling']
        s2 = scenarios['S2_software_runtime_ceiling']
        s3 = scenarios['S3_hw_ub_l1_fused']
        s4 = scenarios['S4_hw_ub_l1_fused_hbm3']
        lines.append(f"| {model_key} | {s0:>10,.0f} | "
                     f"{s1.wall_clock.wall_clock_us:>6,.0f} | {s1.reduction_pct:>4.1f}% | "
                     f"{s2.wall_clock.wall_clock_us:>8,.0f} | {s2.reduction_pct:>4.1f}% | "
                     f"{s3.wall_clock.wall_clock_us:>8,.0f} | {s3.reduction_pct:>4.1f}% | "
                     f"{s4.wall_clock.wall_clock_us:>8,.0f} | {s4.reduction_pct:>4.1f}% |")
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 四、按 workload 类别分析')
    lines.append('')

    # 固定网络 workload
    lines.append('### 4.1 固定网络业务（BERT / GPT-2 / Net-Transformer）')
    lines.append('')
    for key in ['BERT-base-S128-b1', 'GPT-2-S512-b1', 'Net-Transformer-S256-L1-b1']:
        if key not in results:
            continue
        scen = results[key]
        s0 = scen['S0_baseline'].wall_clock
        s1 = scen['S1_software_ceiling']
        s2 = scen['S2_software_runtime_ceiling']
        s3 = scen['S3_hw_ub_l1_fused']
        sw_only_gain = s1.reduction_pct
        runtime_gain = s2.reduction_pct - s1.reduction_pct
        hw_extra_gain = s3.reduction_pct - s2.reduction_pct
        lines.append(f'**{key}**：')
        lines.append(f'- S0 baseline = {s0.wall_clock_us:,.0f} μs（其中 host_gap = {s0.host_gap_us:,.0f} μs，占 {100*s0.host_gap_us/s0.wall_clock_us:.0f}%）')
        lines.append(f'- 算子优化（S1）单独降 **{sw_only_gain:.1f}%**')
        lines.append(f'- + CANN runtime 优化再降 **{runtime_gain:.1f}%**（host_gap 优化是这一类的主要杠杆）')
        lines.append(f'- + UB+L1 融合再降 **{hw_extra_gain:.1f}%**（固定网络业务 aiv_mte2 占比小，硬件杠杆有限）')
        lines.append('')

    # LLM prefill
    lines.append('### 4.2 LLM prefill（Qwen3-0.6B 短/中/长上下文）')
    lines.append('')
    for key in ['Qwen3-prefill-S256-b1', 'Qwen3-prefill-S256-b8', 'Qwen3-prefill-S512-b8',
                'Qwen3-prefill-S4096-b1', 'Qwen3-Embedding-S4096-b1']:
        if key not in results:
            continue
        scen = results[key]
        s0 = scen['S0_baseline'].wall_clock
        s1 = scen['S1_software_ceiling']
        s2 = scen['S2_software_runtime_ceiling']
        s3 = scen['S3_hw_ub_l1_fused']
        s4 = scen['S4_hw_ub_l1_fused_hbm3']
        sw_only_gain = s1.reduction_pct
        hw_fused_extra = s3.reduction_pct - s2.reduction_pct
        hw_hbm3_extra = s4.reduction_pct - s3.reduction_pct
        lines.append(f'**{key}**：')
        lines.append(f'- S0 baseline = {s0.wall_clock_us:,.0f} μs')
        lines.append(f'- 算子优化（S1）降 **{sw_only_gain:.1f}%**；S2 总降 {s2.reduction_pct:.1f}%')
        lines.append(f'- UB+L1 融合再降 **{hw_fused_extra:.1f}%**')
        lines.append(f'- HBM3 再降 **{hw_hbm3_extra:.1f}%**')
        lines.append('')

    # decode
    lines.append('### 4.3 LLM decode')
    lines.append('')
    if 'Qwen3-decode-Min4-Skv128-b1' in results:
        scen = results['Qwen3-decode-Min4-Skv128-b1']
        s0 = scen['S0_baseline'].wall_clock
        s1 = scen['S1_software_ceiling']
        s2 = scen['S2_software_runtime_ceiling']
        s3 = scen['S3_hw_ub_l1_fused']
        s4 = scen['S4_hw_ub_l1_fused_hbm3']
        lines.append(f'**Qwen3-decode-Min4-Skv128-b1**：')
        lines.append(f'- S0 baseline = {s0.wall_clock_us:,.0f} μs')
        lines.append(f'- S1 软件优化降 **{s1.reduction_pct:.1f}%**（aiv_idle 占 ~50%，软件可消除）')
        lines.append(f'- S2 + CANN runtime 总降 {s2.reduction_pct:.1f}%')
        lines.append(f'- S3 + UB+L1 融合总降 {s3.reduction_pct:.1f}%')
        lines.append(f'- **S4 + HBM3 总降 {s4.reduction_pct:.1f}%（HBM 是 decode 的真瓶颈）**')
        lines.append('')

    lines.append('---')
    lines.append('')
    lines.append('## 五、核心结论')
    lines.append('')
    lines.append('### 5.1 "现有固定网络场景仅靠算子优化能解决 vector 瓶颈吗？"')
    lines.append('')
    lines.append('**部分能，部分不能**——按 workload 分：')
    lines.append('')
    lines.append('| Workload 类别 | S1 算子优化 wall_clock 降幅 | S3-S2 = 硬件 UB+L1 融合再降幅 | 判定 |')
    lines.append('|--------------|---------------------------:|----------------------------:|------|')
    bert = results.get('BERT-base-S128-b1')
    if bert:
        s1pct = bert['S1_software_ceiling'].reduction_pct
        s3vs2 = bert['S3_hw_ub_l1_fused'].reduction_pct - bert['S2_software_runtime_ceiling'].reduction_pct
        lines.append(f'| 固定网络短输入 (BERT b=1) | {s1pct:.1f}% | {s3vs2:.1f}% | **算子优化为主**，硬件可忽略 |')
    qwen_long = results.get('Qwen3-prefill-S4096-b1')
    if qwen_long:
        s1pct = qwen_long['S1_software_ceiling'].reduction_pct
        s3vs2 = qwen_long['S3_hw_ub_l1_fused'].reduction_pct - qwen_long['S2_software_runtime_ceiling'].reduction_pct
        lines.append(f'| LLM 长上下文 (Qwen3-prefill-S4096) | {s1pct:.1f}% | **{s3vs2:.1f}%** | **硬件 UB+L1 融合是必经**，算子优化达上限 |')
    decode = results.get('Qwen3-decode-Min4-Skv128-b1')
    if decode:
        s1pct = decode['S1_software_ceiling'].reduction_pct
        s4vs3 = decode['S4_hw_ub_l1_fused_hbm3'].reduction_pct - decode['S3_hw_ub_l1_fused'].reduction_pct
        lines.append(f'| LLM decode | {s1pct:.1f}% | HBM3 再降 {s4vs3:.1f}% | **HBM3 + 软件双轮**，UB 融合次要 |')
    lines.append('')
    lines.append('### 5.2 "Vector 瓶颈"分两类的实测证据')
    lines.append('')
    lines.append('- **流水线效率类**（aiv_idle）：算子优化 100% 可解决。BERT b=1 上 idle 占 aiv_time 的 46%，这部分是软件可消除的同步等待。')
    lines.append('- **算法搬运量类**（aiv_mte2 = UB↔L1 字节数）：算子优化无法消除。Qwen3-prefill-S4096 上 aiv_mte2 占 73.2%，是 algorithmic memory traffic，**只有 UB+L1 融合等硬件改动能消除**。')
    lines.append('')
    lines.append('### 5.3 立项含义')
    lines.append('')
    lines.append('1. **固定网络业务自研芯片**：硬件 vector 改动 ROI 低（固定网络 workload 的 vector 瓶颈是 CANN runtime + 算子库已有覆盖范围）')
    lines.append('2. **LLM-friendly 自研芯片**：UB+L1 融合是真正可叠加在 CANN 算子优化之上的、长上下文 prefill 上 ~20% 加速的硬件 lever')
    lines.append('3. **LLM serving 主战场（decode）**：HBM3 是首要硬件投资；UB+L1 融合是次要')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 六、工具使用与扩展')
    lines.append('')
    lines.append('### 6.1 添加新 workload')
    lines.append('1. 在 NPU 上跑 msprof PipeUtilization 采集（见 `benchmark/run_phase_b.sh` 模板）')
    lines.append('2. 用 `scripts/calibration/extract_cube_util.py`（含 PipeUtil 字段提取）抽取')
    lines.append('3. 添加 entry 到 `data/pipe_baseline_per_model.json`')
    lines.append('4. 重跑 `predict_optimization_ceiling.py` 即可')
    lines.append('')
    lines.append('### 6.2 自定义优化情景')
    lines.append('在 `scripts/calibration/predict_optimization_ceiling.py` 中可定义新的 `compute_*` 函数：')
    lines.append('- 调整某个 pipe 的 scaling 系数')
    lines.append('- 模拟不同硬件改动（如 fixpipe BW × 2）')
    lines.append('- 模拟 KV cache prefetcher 等假想架构')
    lines.append('')
    lines.append('### 6.3 工具发布后的位置（M1 后）')
    lines.append('- API：`from prism.ceiling import predict_all_scenarios`')
    lines.append('- CLI：`prism-ceiling --pipe-baseline ... --output-md ...`')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 七、产物索引')
    lines.append('')
    lines.append('| 文件 | 内容 |')
    lines.append('|------|------|')
    lines.append('| 本文 (`docs/operator_optimization_ceiling.md`) | 5 情景分析 + 立项建议 |')
    lines.append('| `data/optimization_ceiling.json` | 5 情景 × 11 配置 全结果 JSON |')
    lines.append('| `scripts/calibration/predict_optimization_ceiling.py` | 主工具脚本 |')
    lines.append('| `data/pipe_baseline_per_model.json` | Phase N 9 配置 PipeUtil 实测（输入）|')
    lines.append('| `docs/overhead_decomposition_audit.md` | Phase N audit（pipe % 数据来源解释）|')
    lines.append('| `docs/arch_hypothesis_rules.md` | Phase N pipe % → 架构推荐规则（互补阅读）|')
    lines.append('')
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pipe-baseline',
                        default=str(REPO / 'data' / 'calibration' / 'pipe_baseline_per_model.json'))
    parser.add_argument('--output-json',
                        default=str(REPO / 'data' / 'outputs' / 'optimization_ceiling.json'))
    parser.add_argument('--output-md',
                        default=str(REPO / 'docs' / 'findings' / 'optimization_ceiling.md'))
    parser.add_argument('--host-gap-target', type=float,
                        default=HOST_GAP_TARGET_US_PER_KERNEL,
                        help='S2 情景下 host_gap 目标值（μs/kernel）')
    parser.add_argument('--hbm3-bw-gbs', type=float,
                        default=HBM3_BW_GBS,
                        help='S4 情景下 HBM3 带宽 (GB/s)')
    args = parser.parse_args()

    # 加载输入
    with open(args.pipe_baseline, encoding='utf-8') as f:
        baseline_data = json.load(f)
    pipe_baseline = baseline_data['configs']

    # 计算 5 情景
    results = predict_all_scenarios(pipe_baseline, args.host_gap_target, args.hbm3_bw_gbs)

    # 保存 JSON
    serialized = serialize_results(results)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump({
            'version':         'v1',
            'host_gap_target_per_kernel_us': args.host_gap_target,
            'ub_l1_fused_residual':         UB_L1_FUSED_RESIDUAL,
            'hbm3_bw_gbs':                  args.hbm3_bw_gbs,
            'hbm2e_bw_gbs_baseline':        HBM2E_BW_GBS_BASELINE,
            'configs':                      serialized,
        }, f, indent=2, ensure_ascii=False)
    print(f'=> JSON: {args.output_json}')

    # 渲染 Markdown
    md = render_markdown_report(results)
    with open(args.output_md, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'=> MD:   {args.output_md}')

    # 终端速览
    print()
    print(f"{'config':<32}{'S0':>10}{'S1 (-%)':>14}{'S2 (-%)':>14}{'S3 (-%)':>14}{'S4 (-%)':>14}")
    for model_key, scenarios in results.items():
        s0 = scenarios['S0_baseline'].wall_clock.wall_clock_us
        s1 = scenarios['S1_software_ceiling']
        s2 = scenarios['S2_software_runtime_ceiling']
        s3 = scenarios['S3_hw_ub_l1_fused']
        s4 = scenarios['S4_hw_ub_l1_fused_hbm3']
        print(f"{model_key:<32}{s0:>10,.0f}"
              f"{s1.wall_clock.wall_clock_us:>8,.0f}({s1.reduction_pct:>4.1f}%)"
              f"{s2.wall_clock.wall_clock_us:>8,.0f}({s2.reduction_pct:>4.1f}%)"
              f"{s3.wall_clock.wall_clock_us:>8,.0f}({s3.reduction_pct:>4.1f}%)"
              f"{s4.wall_clock.wall_clock_us:>8,.0f}({s4.reduction_pct:>4.1f}%)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
