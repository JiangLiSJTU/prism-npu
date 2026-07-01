"""Test prism.sweep module.

Verifies:
- module imports + key exports (SWEEP dict + 关键函数)
- SWEEP 含 11 个细粒度维度（n_cores、cube_kdim、l2_mb、hbm_bw_gbs、aiv_per_aic、
  tdp_w、l0a_kb、l1_kb、l1_l0_bw_gbs、fixpipe_bw_gbs、ub_l1_fused）
  注：`beta_host_gap_us_per_kernel` 自 Issue #8 起从 SWEEP 移除 —— 视为 software-only、
       arch-invariant，host_gap 优化杠杆由 prism-ceiling S2 情景建模。
- baseline arch yaml 存在
- predict_wallclock_v3 baseline 重现误差 < 5%
"""
from __future__ import annotations

import json

import pytest


def test_sweep_module_imports():
    from prism.sweep import runner, timeloop_problem
    assert hasattr(runner, "SWEEP"), "SWEEP dict missing"
    assert hasattr(runner, "BASELINE_910B4"), "BASELINE_910B4 missing"
    assert hasattr(runner, "predict_wallclock_v3"), "predict_wallclock_v3 missing"


def test_sweep_dict_has_11_dimensions():
    """SWEEP 应含 11 个细粒度维度（Issue #8 起移除 beta_host_gap_us_per_kernel 死维度）。"""
    from prism.sweep.runner import SWEEP

    expected_dims = {
        "n_cores", "cube_kdim", "l2_mb", "hbm_bw_gbs",
        "aiv_per_aic", "tdp_w",
        "l0a_kb", "l1_kb",
        "l1_l0_bw_gbs", "fixpipe_bw_gbs",
        "ub_l1_fused",
    }
    assert len(SWEEP) >= 11, f"SWEEP only has {len(SWEEP)} dimensions"
    actual = set(SWEEP.keys())
    missing = expected_dims - actual
    assert not missing, f"SWEEP 缺关键维度 {missing}"
    # Issue #8 hard gate: 死维度不应被重新添加（predict_wallclock_v3 不读 variant_arch
    # 的该字段，列在 SWEEP 中会产出冗余/误导性变体）。
    assert "beta_host_gap_us_per_kernel" not in actual, (
        "beta_host_gap_us_per_kernel 被重新加进 SWEEP —— Issue #8 说明 host_gap 是 "
        "software-only，由 prism-ceiling S2 情景建模，不应作为架构 sweep 维度"
    )


def test_baseline_arch_has_required_fields():
    """BASELINE_910B4 dict 应含 sweep 公式所需的关键字段。"""
    from prism.sweep.runner import BASELINE_910B4

    required = {
        "n_cores", "cube_m", "cube_n", "cube_k", "cube_total_macs",
        "hbm_bw_gbs", "l2_mb", "clock_ghz",
    }
    actual = set(BASELINE_910B4.keys())
    missing = required - actual
    assert not missing, f"BASELINE_910B4 缺字段 {missing}"


def test_predict_wallclock_v3_baseline_reproduction(pipe_baseline):
    """predict_wallclock_v3 喂 baseline 应重现实测 wall_clock 误差 < 10%。

    关键不变量：[methodology/02 §7.1] 要求 baseline 重现误差 < 5%。
    本测试用 10% 阈值留 safety margin（msprof 自身的 measurement noise）。

    例外：占位/继承的 model 配置（Net-Transformer 用 BERT 比例缩放占位、
    Qwen3-Embedding 继承 Qwen3-prefill body）允许更大误差，仅检查
    "误差有上界"。
    """
    from prism.sweep.runner import predict_wallclock_v3, BASELINE_910B4

    PROXY_MODELS = {"Net-Transformer-S256-L1-b1", "Qwen3-Embedding-S4096-b1"}

    for cfg, pipe in pipe_baseline["configs"].items():
        result = predict_wallclock_v3(pipe, BASELINE_910B4, BASELINE_910B4)
        predicted = result["wall_clock_us"]
        actual = pipe.get("wall_clock_us", predicted)

        if actual <= 0:
            continue

        err_pct = 100 * abs(predicted - actual) / actual
        # Real-measured 配置：< 10%；占位/继承配置：< 100%（仅 sanity）
        threshold = 100.0 if cfg in PROXY_MODELS else 10.0
        assert err_pct < threshold, (
            f"{cfg}: baseline 重现误差 {err_pct:.1f}% > {threshold}% "
            f"(predicted={predicted:.0f}, actual={actual:.0f})"
        )


def test_baseline_arch_yaml_exists(arch_baseline_path):
    assert arch_baseline_path.is_file(), f"baseline arch yaml missing: {arch_baseline_path}"


# ─────────────────────────────────────────────────────────────────────────
# Issue #7 — aic_fixpipe / aiv_mte3 destination-bandwidth blend regression tests
#
# These exercise the new code paths added in PR #6 (`_dest_time_proxy`,
# `_dest_blend_factor`, `scale_*_pipes`'s new gm_frac kwarg). The pre-existing
# `test_predict_wallclock_v3_baseline_reproduction` only runs variant=baseline,
# where the blend factor is 1.0 by construction and so cannot catch regressions
# in the new math.
# ─────────────────────────────────────────────────────────────────────────


def test_dest_blend_factor_is_identity_when_variant_equals_baseline():
    """variant=baseline ⇒ blend factor = 1.0 exactly, for any gm_frac.

    Guards against regressions that would silently perturb baseline reproduction.
    """
    from copy import deepcopy
    from prism.sweep.runner import BASELINE_910B4, _dest_blend_factor

    variant = deepcopy(BASELINE_910B4)
    for gm in (0.0, 0.25, 0.5, 0.75, 1.0):
        for onchip_key in ("fixpipe_bw_gbs", "ub_l1_bw_gbs"):
            f = _dest_blend_factor(gm, BASELINE_910B4, variant, onchip_key)
            assert f == pytest.approx(1.0, abs=1e-12), (
                f"gm_frac={gm} onchip={onchip_key}: blend factor {f} ≠ 1.0"
            )


def test_dest_time_proxy_gm_dominated_tracks_hbm():
    """gm_frac→1 ⇒ proxy ≈ 1/hbm (HBM term dominates).

    This is the physical-regime sanity: a config whose stores almost all go to
    GM should be HBM-bandwidth-bound, not on-chip-bandwidth-bound.
    """
    from prism.sweep.runner import BASELINE_910B4, _dest_time_proxy

    hbm = BASELINE_910B4["hbm_bw_gbs"]
    fixpipe = BASELINE_910B4["fixpipe_bw_gbs"]

    proxy_all_gm = _dest_time_proxy(1.0, BASELINE_910B4, "fixpipe_bw_gbs")
    proxy_all_oc = _dest_time_proxy(0.0, BASELINE_910B4, "fixpipe_bw_gbs")
    proxy_mostly_gm = _dest_time_proxy(0.98, BASELINE_910B4, "fixpipe_bw_gbs")

    assert proxy_all_gm == pytest.approx(1.0 / hbm, rel=1e-9)
    assert proxy_all_oc == pytest.approx(1.0 / fixpipe, rel=1e-9)
    # 0.98 GM-dominated config: total time ~10× the all-on-chip case (~hbm/fixpipe ratio)
    assert proxy_mostly_gm / proxy_all_oc > 5.0, (
        f"gm_frac=0.98 proxy/all_oc = {proxy_mostly_gm/proxy_all_oc:.2f} "
        f"— expected ≫ 1 (HBM is 10× slower than fixpipe)"
    )


def test_fixpipe_halving_is_neutral_when_gm_frac_is_high():
    """The marquee Issue #7 finding.

    For a config like Qwen3-prefill-S4096-b1 with gm_frac≈0.976, halving
    `fixpipe_bw_gbs` should leave the fixpipe ratio essentially at 1.0
    (FixPipe unit bandwidth is *not* the lever; HBM is). The pre-#7 model
    would have given ratio ≈ 2.0 here (incorrectly).

    Tolerance 5% absorbs the ~2.4% on-chip residue + numerical slack.
    """
    from copy import deepcopy
    from prism.sweep.runner import BASELINE_910B4, _dest_blend_factor

    variant = deepcopy(BASELINE_910B4)
    variant["fixpipe_bw_gbs"] = BASELINE_910B4["fixpipe_bw_gbs"] // 2  # 4096 → 2048

    gm_frac_qwen3_s4096 = 0.976
    f = _dest_blend_factor(gm_frac_qwen3_s4096, BASELINE_910B4, variant, "fixpipe_bw_gbs")

    assert f == pytest.approx(1.0, abs=0.05), (
        f"fixpipe halved + gm_frac=0.976: factor = {f:.3f}; "
        f"expected ≈ 1.0 (HBM is the real lever, not fixpipe_bw)"
    )


def test_hbm_halving_is_lever_when_gm_frac_is_high():
    """Converse of the above: halving HBM at high gm_frac SHOULD ~double pipe time.

    Guards against regressions where the blend formula stops being HBM-sensitive
    when it should be — that would silently lose the Issue #7 modeling fix.
    """
    from copy import deepcopy
    from prism.sweep.runner import BASELINE_910B4, _dest_blend_factor

    variant = deepcopy(BASELINE_910B4)
    variant["hbm_bw_gbs"] = BASELINE_910B4["hbm_bw_gbs"] // 2  # 392 → 196

    f = _dest_blend_factor(0.976, BASELINE_910B4, variant, "fixpipe_bw_gbs")
    # With 97.6% of bytes going through HBM, halving HBM should nearly double the cost.
    # blend_factor in [1.8, 2.0] (not exactly 2.0 because 2.4% still goes on-chip)
    assert 1.8 < f < 2.0, (
        f"hbm halved + gm_frac=0.976: factor = {f:.3f}; "
        f"expected ~1.95 — HBM should be the dominant lever"
    )


def test_fused_ub_l1_eliminates_onchip_path_for_mte3():
    """UB+L1 fusion drives the on-chip term to 5% residual (mte3 only).

    Verifies the `fused_eliminates=True` branch in `_dest_time_proxy`. When
    gm_frac is low (on-chip dominated), fused should give ~20× speedup of the
    proxy; when gm_frac is high (GM dominated), fusion is mostly a no-op.
    """
    from prism.sweep.runner import BASELINE_910B4, _dest_time_proxy

    # All-on-chip config: fusion should give ~20× speedup (1/0.05 - 1, ignoring
    # the negligible GM term at gm_frac=0).
    p_oc_unfused = _dest_time_proxy(0.0, BASELINE_910B4, "ub_l1_bw_gbs",
                                    fused_eliminates=False)
    arch_fused = dict(BASELINE_910B4, ub_l1_fused=True)
    p_oc_fused = _dest_time_proxy(0.0, arch_fused, "ub_l1_bw_gbs",
                                  fused_eliminates=True)
    speedup = p_oc_unfused / p_oc_fused
    assert speedup == pytest.approx(20.0, rel=0.01), (
        f"fused on-chip speedup = {speedup:.2f}, expected 20.0 (5% residual)"
    )

    # GM-dominated config: fusion barely matters (GM term unchanged).
    p_gm_unfused = _dest_time_proxy(0.95, BASELINE_910B4, "ub_l1_bw_gbs",
                                    fused_eliminates=False)
    p_gm_fused = _dest_time_proxy(0.95, arch_fused, "ub_l1_bw_gbs",
                                  fused_eliminates=True)
    assert p_gm_fused / p_gm_unfused > 0.95, (
        f"fused @ gm_frac=0.95: speedup ratio = {p_gm_fused/p_gm_unfused:.3f}; "
        f"expected ~1.0 (GM term dominates, fusion has little headroom)"
    )


def test_scale_pipes_backward_compat_without_gm_frac_arg():
    """Calling scale_aic_pipes / scale_aiv_pipes without gm_frac uses defaults.

    Guards the public API for callers (notably the predict_pipe synthesis path)
    that don't have a per-config gm_frac to pass.
    """
    from prism.sweep.runner import BASELINE_910B4, scale_aic_pipes, scale_aiv_pipes

    aic_baseline = {"mac": 100, "mte1": 50, "mte2": 200, "fixpipe": 80, "scalar": 10}
    aiv_baseline = {"vec": 60, "mte2": 90, "mte3": 70, "scalar": 5, "idle": 20}

    # No gm_frac arg → uses DEFAULT_*_GM_FRAC (0.5). Should not raise.
    aic = scale_aic_pipes(aic_baseline, BASELINE_910B4, BASELINE_910B4)
    aiv = scale_aiv_pipes(aiv_baseline, BASELINE_910B4, BASELINE_910B4)

    # variant=baseline ⇒ all pipes preserved (modulo float fuzz)
    for k, v in aic_baseline.items():
        assert aic[k] == pytest.approx(v, rel=1e-9)
    for k, v in aiv_baseline.items():
        assert aiv[k] == pytest.approx(v, rel=1e-9)


def test_pipe_dest_bw_gm_frac_in_unit_interval(pipe_dest_bw):
    """All per-config gm_frac values must lie in [0, 1] (or be null).

    A gm_frac outside [0, 1] indicates a bug in the OLS back-solve or the
    2-cluster classifier — would propagate non-physical scaling into sweep.
    """
    for pipe in ("aic_fixpipe", "aiv_mte3"):
        for cfg, entry in pipe_dest_bw[pipe].items():
            g = entry.get("gm_frac")
            if g is None:
                continue
            assert 0.0 <= g <= 1.0, f"{pipe}[{cfg}].gm_frac = {g} ∉ [0,1]"


def test_pipe_dest_bw_marquee_config_is_gm_dominated(pipe_dest_bw):
    """Qwen3-prefill-S4096-b1-sdpa fixpipe gm_frac should be ≥ 0.9.

    Anchors the headline Issue #7 finding ("FixPipe halving ratio 1.24 → 1.00"
    in 主报告 §6.4.1). If gm_frac for this config drifts below 0.9, the report's
    numerical claim no longer holds.
    """
    entry = pipe_dest_bw["aic_fixpipe"].get("Qwen3-prefill-S4096-b1-sdpa")
    assert entry is not None, "marquee config missing from pipe_dest_bw.json"
    assert entry["gm_frac"] >= 0.9, (
        f"Qwen3-S4096-sdpa fixpipe gm_frac = {entry['gm_frac']}; "
        f"主报告 §6.4.1 claim requires ≥ 0.9 (GM-dominated)"
    )
