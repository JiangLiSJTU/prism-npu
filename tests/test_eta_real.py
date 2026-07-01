"""Test prism.eta_real.fit + predict modules.

Verifies:
- predict_eta returns sensible values (∈ [0, 1])
- predict_eta of large GEMM > predict_eta of attention head
- fitted params are within expected ranges
- BERT validation MAE硬门槛 < 15 pp
"""
from __future__ import annotations

import math


def test_predict_eta_in_range(eta_fit):
    """η_real ∈ [0, 1] for any sensible shape."""
    from prism.eta_real.fit import predict_eta

    params = eta_fit["params"]
    for (M, N, K, B) in [
        (4096, 3072, 1024, 1),
        (128, 768, 768, 1),
        (4096, 4096, 128, 1),     # attention head
    ]:
        sample = {"M_per_batch": M, "N": N, "K": K, "B": B, "op_kind": "BMM"}
        eta = predict_eta(
            M, N, K, B,
            list(params.values()),
            cube_m=16, cube_n=16, cube_k=16,
        ) if False else _wrap_predict(sample, params)
        assert 0 <= eta <= 1, f"eta={eta} out of [0, 1] for shape {M, N, K, B}"


def _wrap_predict(sample, params):
    """call signature compat — wraps fit.predict_eta whose API takes positional alpha/beta/gamma/delta/gamma_B."""
    from prism.eta_real.fit import predict_eta as _pe
    M = sample["M_per_batch"] * (sample["B"] if sample.get("op_kind") == "BMM" and min(sample["M_per_batch"], sample["N"], sample["K"]) > 128 else 1)
    return _pe(
        sample,
        params["alpha_MN_coupling"],
        params["beta_MK_coupling"],
        params["gamma_NK_coupling"],
        params["delta_linear_edge"],
        params["gamma_B_batch"],
    )


def test_large_gemm_beats_attention_head(eta_fit):
    """大 GEMM (M=N=K=1024+) η_real 应显著高于 attention head (small dim 128)."""
    params = eta_fit["params"]

    large = _wrap_predict({"M_per_batch": 4096, "N": 3072, "K": 1024, "B": 1, "op_kind": "BMM"}, params)
    head  = _wrap_predict({"M_per_batch": 128, "N": 128, "K": 64, "B": 1, "op_kind": "BMM"}, params)

    assert large > head, f"large GEMM η ({large:.3f}) should > attention head ({head:.3f})"


def test_fit_params_in_expected_range(eta_fit):
    """5 参数应在合理范围（avoid degenerate fits）。"""
    p = eta_fit["params"]

    assert 1.0 <= p["alpha_MN_coupling"] <= 50.0, f"alpha={p['alpha_MN_coupling']} out of expected [1, 50]"
    assert 0.5 <= p["beta_MK_coupling"]  <= 20.0
    assert 0.5 <= p["gamma_NK_coupling"] <= 20.0
    assert -0.1 <= p["delta_linear_edge"] <= 10.0
    assert -0.1 <= p["gamma_B_batch"]     <= 0.1


def test_bert_validation_mae_under_hard_gate(eta_fit):
    """硬门槛：BERT 验证 MAE < 15 pp（用户设定，进 sweep 的入门条件）。"""
    val = eta_fit.get("validation", {}).get("bert", {})
    assert val, "BERT validation block missing in eta_fit JSON"

    # 兼容两种 key 命名（mae 或 mae_pp）
    mae_pp = val.get("mae_pp", val.get("mae"))
    assert mae_pp is not None, f"BERT validation has no 'mae' or 'mae_pp' field: {val}"
    assert mae_pp < 15.0, f"BERT validation MAE {mae_pp:.2f} pp >= 15 pp 硬门槛"


def test_batch_factor_monotonic(eta_fit):
    """同 GEMM 形状下 η_batch 应随 B 单调递增（log scale）。"""
    params = eta_fit["params"]

    eta_b1 = _wrap_predict({"M_per_batch": 4096, "N": 3072, "K": 1024, "B": 1, "op_kind": "BMM"}, params)
    eta_b8 = _wrap_predict({"M_per_batch": 4096, "N": 3072, "K": 1024, "B": 8, "op_kind": "BMM"}, params)
    eta_b16 = _wrap_predict({"M_per_batch": 4096, "N": 3072, "K": 1024, "B": 16, "op_kind": "BMM"}, params)

    # Allow tiny numerical noise + monotonic non-decreasing
    assert eta_b8 >= eta_b1 - 1e-6
    assert eta_b16 >= eta_b8 - 1e-6
