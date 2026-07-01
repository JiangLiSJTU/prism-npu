"""Test prism.ceiling module (5 优化情景预测).

Verifies:
- S0 baseline reduction = 0
- S1 software ceiling 消除 idle/bubble/kernel_gap
- S2 host_gap 单调递减（host_gap min(target, baseline) bug fix verification）
- S3/S4 在 baseline 之上单调递减
- ScenarioResult dataclass 字段完整
"""
from __future__ import annotations

import pytest


def test_ceiling_module_imports():
    from prism.ceiling import predict
    assert hasattr(predict, "predict_all_scenarios")
    assert hasattr(predict, "ScenarioResult")
    assert hasattr(predict, "WallClockBreakdown")


def test_baseline_zero_reduction(pipe_baseline):
    """S0 baseline reduction_pct 应为 0。"""
    from prism.ceiling.predict import compute_baseline

    pipe = pipe_baseline["configs"]["BERT-base-S128-b1"]
    s0 = compute_baseline(pipe)

    assert s0.reduction_pct == 0.0, f"S0 baseline reduction should be 0, got {s0.reduction_pct}"
    assert s0.scenario == "S0_baseline"


def test_software_ceiling_consumes_idle(pipe_baseline):
    """S1 应当消除 aiv_idle，aiv_time 减小到 max(active pipes)。"""
    from prism.ceiling.predict import compute_baseline, compute_software_ceiling

    pipe = pipe_baseline["configs"]["BERT-base-S128-b1"]
    s0 = compute_baseline(pipe)
    s1 = compute_software_ceiling(pipe)

    # S1 wall_clock 应 ≤ S0
    assert s1.wall_clock.wall_clock_us <= s0.wall_clock.wall_clock_us, (
        f"S1 ({s1.wall_clock.wall_clock_us}) should ≤ S0 ({s0.wall_clock.wall_clock_us})"
    )
    assert s1.scenario == "S1_software_ceiling"


def test_host_gap_only_decreases(pipe_baseline):
    """S2 host_gap 应当 ≤ baseline（修了 min(target, baseline) bug 的回归测试）。

    历史 bug：S2 把 host_gap 硬设到 n_kernels × 10 μs，
    但 Qwen3-decode 的 baseline 已经是 0.16 μs/kernel < 10 → S2 反而抬高 host_gap。
    """
    from prism.ceiling.predict import compute_baseline, compute_software_runtime_ceiling

    for cfg in pipe_baseline["configs"].keys():
        pipe = pipe_baseline["configs"][cfg]
        s0 = compute_baseline(pipe)
        s2 = compute_software_runtime_ceiling(pipe, host_gap_target_per_kernel=10.0)

        assert s2.wall_clock.host_gap_us <= s0.wall_clock.host_gap_us + 1e-3, (
            f"{cfg}: S2 host_gap ({s2.wall_clock.host_gap_us}) > "
            f"S0 ({s0.wall_clock.host_gap_us}) — `min(baseline, target)` regression!"
        )


def test_scenarios_monotonic_decreasing(pipe_baseline):
    """S0 ≥ S1 ≥ S2 ≥ S3 ≥ S4 wall_clock 单调递减（每加一层优化只能更好）。"""
    from prism.ceiling.predict import predict_all_scenarios

    results = predict_all_scenarios(pipe_baseline["configs"])

    for cfg, scenarios in results.items():
        s0 = scenarios["S0_baseline"].wall_clock.wall_clock_us
        s1 = scenarios["S1_software_ceiling"].wall_clock.wall_clock_us
        s2 = scenarios["S2_software_runtime_ceiling"].wall_clock.wall_clock_us
        s3 = scenarios["S3_hw_ub_l1_fused"].wall_clock.wall_clock_us
        s4 = scenarios["S4_hw_ub_l1_fused_hbm3"].wall_clock.wall_clock_us

        assert s1 <= s0 + 1e-3, f"{cfg}: S1 > S0"
        assert s2 <= s1 + 1e-3, f"{cfg}: S2 > S1"
        assert s3 <= s2 + 1e-3, f"{cfg}: S3 > S2"
        assert s4 <= s3 + 1e-3, f"{cfg}: S4 > S3"


def test_known_qwen3_prefill_ub_fusion_gain(pipe_baseline):
    """Qwen3-prefill-S4096 的 UB+L1 融合应带来 ≥ 15% wall-clock 加速 (S3 vs S2)。"""
    from prism.ceiling.predict import (
        compute_software_runtime_ceiling,
        compute_hw_ub_l1_fused,
    )

    pipe = pipe_baseline["configs"]["Qwen3-prefill-S4096-b1"]
    s2 = compute_software_runtime_ceiling(pipe)
    s3 = compute_hw_ub_l1_fused(pipe)

    extra_gain_pct = 100 * (1 - s3.wall_clock.wall_clock_us / s2.wall_clock.wall_clock_us)
    assert extra_gain_pct >= 15.0, (
        f"Qwen3-prefill-S4096 UB+L1 fusion 增量 only {extra_gain_pct:.1f}% (< 15% expected)"
    )


def test_known_qwen3_decode_hbm3_gain(pipe_baseline):
    """Qwen3-decode 的 HBM3 应带来 ≥ 10% 增量加速 (S4 vs S3)。"""
    from prism.ceiling.predict import (
        compute_hw_ub_l1_fused,
        compute_hw_ub_l1_fused_hbm3,
    )

    cfg = "Qwen3-decode-Min4-Skv128-b1"
    if cfg not in pipe_baseline["configs"]:
        pytest.skip(f"{cfg} not in baseline data")

    pipe = pipe_baseline["configs"][cfg]
    s3 = compute_hw_ub_l1_fused(pipe)
    s4 = compute_hw_ub_l1_fused_hbm3(pipe)

    extra_gain_pct = 100 * (1 - s4.wall_clock.wall_clock_us / s3.wall_clock.wall_clock_us)
    assert extra_gain_pct >= 10.0, (
        f"Qwen3-decode HBM3 增量 only {extra_gain_pct:.1f}% (< 10% expected)"
    )
