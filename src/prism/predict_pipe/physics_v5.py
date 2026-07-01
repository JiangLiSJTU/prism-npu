"""v5 physics formulas (Issue #2 v5 — overfit-safe replacements for v4).

v4 failure modes (per Llama OOS validation, +1156% wall_clock err):

1. AIC archetype_amplification 3-bucket {1.15, 5.5, 14.16} over-fit to
   Qwen3 prefill — fails Llama/Qwen2.5/SmolLM2 at 4-8× excess
2. AIV continuous amp `(w_proxy/1000)²` explodes outside training [200, 600] MB
3. n_kernels archetype mult 28× over-fit Qwen3 — actual mult on big-prefill
   is ~6× across Llama/Qwen2.5/SmolLM2/ModernBERT

v5 replacements — saturating/linear, bounded extrapolation:

1. `archetype_amp_v5`: linear in w_proxy_mb, capped at amp_max
2. `predict_aiv_v5`: explicit serial sum aiv_vec + aiv_mte2 + aiv_mte3 with
   per-pipe efficiency params (no black-box amp), matching the observed
   mte2+mte3 ≈ 12× vec ratio across all 4 OOS configs
3. `n_kernels_mult_v5`: `mult = mult_base + (mult_max - mult_base) ×
   (1 - exp(-w_proxy / W_sat))` — saturates to ~6 instead of 28

All v5 free params live in `fitted_params` dict alongside v4 keys (no
breaking changes — v5 markers gate dispatch in `predict.py`).
"""
from __future__ import annotations
import math
from typing import Mapping


# ─────────────────────────────────────────────────────────────────────────
# AIC amp v5 — linear in w_proxy, saturates
# ─────────────────────────────────────────────────────────────────────────
def archetype_amp_v5(weight_mb_proxy: float, S: int,
                     params: Mapping[str, float]) -> float:
    """Linear AIC amp: `amp = 1 + alpha × w_proxy / 1000`, capped at amp_max.

    Calibrated from measured `aic_time / sum(physics aic_pipes)` ratios:
      ModernBERT (w=220):  ratio ≈ 1.08
      Qwen2.5     (w=782):  ratio ≈ 1.30
      Llama       (w=2147): ratio ≈ 2.53
      Qwen3-S4096 (w=763):  ratio ≈ 14   ← outlier, attention-heavy
      decode:                 ratio ≈ 0.85

    Linear `1 + 0.5·w_proxy/1000` gives {1.11, 1.39, 2.07} — close to first 3.
    The Qwen3-S4096 outlier needs `attn_frac` correction (handled in
    `predict.py` as an additional amp factor). v5 lets fit decide alpha.

    Free params:
      aic_amp_alpha  (default 0.5): slope per 1000 MB
      aic_amp_max    (default 3.0): hard cap
      aic_amp_decode (default 0.85): decode-regime constant
    """
    if S == 1:
        return float(params.get("aic_amp_decode", 0.85))
    alpha = float(params.get("aic_amp_alpha", 0.5))
    amp_max = float(params.get("aic_amp_max", 3.0))
    amp = 1.0 + alpha * weight_mb_proxy / 1000.0
    return min(amp, amp_max)


# ─────────────────────────────────────────────────────────────────────────
# AIV v5 — physics serial sum vec + mte2 + mte3 (no black-box amp)
# ─────────────────────────────────────────────────────────────────────────
def predict_aiv_v5(spec, arch: Mapping[str, float], batch: int,
                   params: Mapping[str, float],
                   aiv_data) -> tuple[float, dict, dict]:
    """v5 AIV: v4's base formula × bounded sigmoid amp (NO quadratic).

    Per-config aiv_meas / aiv_base ratio analysis (2026-05-15):
      "normal" prefill (BERT/GPT-2/ModernBERT/Qwen2.5/SmolLM2/Llama):
          ratio in [26-48]× — surprisingly tight cluster
      Qwen3-prefill family is an outlier (ratio 86-247×) — likely due to
          CANN tile dispatch behavior unique to that model class
      decode is unique: tiny base + per-kernel overhead → ratio 524×

    v4 fit `amp = a0 + a1·attn_frac + a2·(w_proxy/1000)²` got the in-range
    data right but quadratic blows up at Llama (w=2147). v5 replaces the
    quadratic with bounded sigmoid:

        amp = max(amp_floor, a0 + a1·attn_frac + a2 · w_sat_fn(w_proxy))
        w_sat_fn(w) = w / (W_sat + w)     ← sigmoid, saturates to 1

    so amp asymptotes to (a0 + a1 + a2) regardless of how large w_proxy gets.

    Free params (kept compatible with v4 for back-compat):
      aiv_amp_a0, aiv_amp_a1, aiv_amp_a2 — sigmoid coefficients
      aiv_amp_W_sat                       — sigmoid half-saturation (MB)
      aiv_amp_decode                      — decode-specific constant
      aiv_amp_floor                       — minimum amp (default 0.5)

    Returns: (aiv_time_us, aiv_pipes_dict, telemetry_dict)
    """
    from . import physics
    from .predict import predict_aiv_v2

    # Use v4's base formula via predict_aiv_v2 with amp turned OFF (set a2=0, a1=0, a0=1)
    # to get aiv_base, then apply v5's bounded amp.
    inter_bytes = aiv_data["inter_bytes"]
    output_bytes = aiv_data["output_bytes"]
    vec_ops = aiv_data["vec_ops"]
    n_vk = aiv_data["n_vector_kernels"]
    attn_frac = aiv_data["attn_frac"]

    # ── v4 base: C_kernel × n_vk + C_data × data_MB ──
    C_kernel = float(params.get("aiv_C_kernel_us", 16.0))
    C_data = float(params.get("aiv_C_data_us", 3.0))
    data_MB = (inter_bytes + output_bytes) / 1e6

    aiv_base = C_kernel * n_vk + C_data * data_MB

    # ── v5 bounded amp ──
    w_proxy = physics.weight_proxy_mb(spec.layers, spec.d_model, spec.d_ff)
    if spec.S == 1:
        amp = float(params.get("aiv_amp_decode", 1.5))
    else:
        a0 = float(params.get("aiv_amp_a0", -0.2))
        a1 = float(params.get("aiv_amp_a1", 4.0))
        a2 = float(params.get("aiv_amp_a2", 14.0))
        W_sat = float(params.get("aiv_amp_W_sat", 500.0))
        floor = float(params.get("aiv_amp_floor", 0.5))
        # Sigmoid (saturates to 1 at large w_proxy)
        w_factor = w_proxy / (W_sat + w_proxy)
        amp = max(floor, a0 + a1 * attn_frac + a2 * w_factor)

    aiv_time = aiv_base * amp

    # Split into pseudo-pipes (proportional to v4 active_frac convention)
    active_frac = float(params.get("aiv_active_frac", 0.85))
    aiv_active = aiv_time * active_frac
    aiv_scalar = max(0.0, n_vk * 0.04)
    aiv_idle = max(0.0, aiv_time - aiv_active - aiv_scalar)

    # vec : mte2 : mte3 ≈ 1 : 7 : 5 (measured average across 4 OOS configs)
    vec_share = 1.0 / 13.0
    mte2_share = 7.0 / 13.0
    mte3_share = 5.0 / 13.0
    pipes = {
        "vec":    round(aiv_active * vec_share, 1),
        "mte2":   round(aiv_active * mte2_share, 1),
        "mte3":   round(aiv_active * mte3_share, 1),
        "scalar": round(aiv_scalar, 1),
        "idle":   round(aiv_idle, 1),
    }
    telemetry = {
        "aiv_w_proxy_mb": round(w_proxy, 1),
        "aiv_amp_v5":     round(amp, 3),
        "aiv_base_us":    round(aiv_base, 1),
    }
    return aiv_time, pipes, telemetry


# ─────────────────────────────────────────────────────────────────────────
# n_kernels v5 — saturating multiplier
# ─────────────────────────────────────────────────────────────────────────
def n_kernels_mult_v5(weight_mb_proxy: float, S: int,
                      params: Mapping[str, float]) -> float:
    """Saturating multiplier for n_kernels archetype estimate.

    v4 used 3-bucket {decode=4.7, small=2.5, large=28}. The 28× for
    large prefill was wrong — measured mult on Llama/Qwen2.5/SmolLM2 was
    all ≈ 6, and ModernBERT (small bucket) was actually 6 too. So:

      mult(w) = mult_base + (mult_max - mult_base) × (1 - exp(-w / W_sat))

    Default: mult_base=2.5, mult_max=6, W_sat=300 MB → saturates to 6 by
    1500 MB. Bounded by mult_max so extrapolation can't explode.

    Free params:
      nk_mult_base   (default 2.5)
      nk_mult_max    (default 6.0)
      nk_W_sat       (default 300.0)
      nk_mult_decode (default 4.7)
    """
    if S == 1:
        return float(params.get("nk_mult_decode", 4.7))
    mult_base = float(params.get("nk_mult_base", 2.5))
    mult_max = float(params.get("nk_mult_max", 6.0))
    W_sat = float(params.get("nk_W_sat", 300.0))
    return mult_base + (mult_max - mult_base) * (1.0 - math.exp(-weight_mb_proxy / W_sat))


# ─────────────────────────────────────────────────────────────────────────
# Convenience: list of all v5 free params with defaults (for grid search)
# ─────────────────────────────────────────────────────────────────────────
V5_PARAM_DEFAULTS = {
    # AIC linear amp (capped)
    "aic_amp_alpha":   0.5,
    "aic_amp_max":     3.0,
    "aic_amp_decode":  0.85,
    # AIV bounded sigmoid amp (replaces v4 quadratic)
    "aiv_amp_a0":      -0.2,
    "aiv_amp_a1":      4.0,
    "aiv_amp_a2":      14.0,
    "aiv_amp_W_sat":   500.0,
    "aiv_amp_floor":   0.5,
    "aiv_amp_decode":  1.5,
    "aiv_C_kernel_us": 16.0,
    "aiv_C_data_us":   3.0,
    "aiv_active_frac": 0.85,
    # n_kernels saturating multiplier
    "nk_mult_base":    2.5,
    "nk_mult_max":     6.0,
    "nk_W_sat":        300.0,
    "nk_mult_decode":  4.7,
}

V5_PARAM_BOUNDS = {
    "aic_amp_alpha":   (0.0, 2.0),
    "aic_amp_max":     (1.5, 8.0),
    "aic_amp_decode":  (0.5, 1.2),
    "aiv_amp_a0":      (-2.0, 5.0),
    "aiv_amp_a1":      (0.0, 8.0),
    "aiv_amp_a2":      (0.0, 15.0),    # tighter — prevents Llama-style explosion
    "aiv_amp_W_sat":   (100.0, 2000.0),
    "aiv_amp_floor":   (0.1, 2.0),
    "aiv_amp_decode":  (0.5, 5.0),
    "aiv_C_kernel_us": (5.0, 40.0),
    "aiv_C_data_us":   (0.5, 10.0),
    "aiv_active_frac": (0.5, 1.0),
    "nk_mult_base":    (1.5, 6.0),
    "nk_mult_max":     (5.0, 30.0),     # widened — Qwen3 needs ~25, Llama ~6
    "nk_W_sat":        (100.0, 2000.0),
    "nk_mult_decode":  (3.0, 6.0),
}
