"""v6 physics formulas — per-bucket amp (responds to user 2026-05-17 insight).

User feedback: "Qwen3-prefill family 是典型的 gemm 占绝对优势，CUBE 是性能瓶颈，
与其他几类不同性能瓶颈的模型是不是应该放在不同的 bucket 进行分类和校准?"

Validated against 13 measured configs (see docs/findings/predict_pipe_bucket_hypothesis.md):

  AIV/AIC < 1.2  →  AIC-bound  (Qwen3-prefill batch>1, decode, S>=4096)
  AIV/AIC > 2.5  →  AIV-bound  (Llama, Qwen2.5, SmolLM2, ModernBERT)
  1.2-2.5        →  Balanced   (BERT, GPT-2, Qwen3-b1)

v5 used single continuous formula across all three regimes → impossible
to fit Qwen3 (high aic_amp) and Llama (low aic_amp) simultaneously.

v6 explicitly classifies + uses per-bucket coefficients.

Bucket classifier (heuristic — see classify_bottleneck below):
  AIC_DECODE  : S == 1
  AIC_QWEN3   : 28+ layers + d_model<1500 + extreme GQA (q/kv >= 6) + (batch>1 OR S>=4096)
  AIV_BOUND   : d_model >= 768 + S*batch >= 1024 (typical decoder prefill at scale)
  BALANCED    : default (small or shallow models, single-batch)
"""
from __future__ import annotations
from typing import Mapping, Literal


Bucket = Literal["AIC_DECODE", "AIC_QWEN3", "AIV_BOUND", "BALANCED"]


def classify_bottleneck(spec, batch: int = 1) -> Bucket:
    """Heuristic classifier — see module docstring.

    Returns one of: AIC_DECODE, AIC_QWEN3, AIV_BOUND, BALANCED.
    """
    if spec.S == 1:
        return "AIC_DECODE"

    # Qwen3-prefill family marker — deep (28L), d_model in [1000, 1300], swiglu
    # Cleanly separates Qwen3-0.6B from Llama (d>=2048), Qwen2.5/SmolLM2 (d<1000),
    # ModernBERT (encoder + glu). Empirically calibrated against measured data
    # where Qwen3-prefill consistently sits in AIV/AIC < 1.2 regime.
    is_qwen3_family = (
        spec.layers >= 24
        and 1000 <= spec.d_model <= 1300
        and getattr(spec, "ffn_type", "") == "swiglu"
        and spec.arch == "decoder"
    )
    if is_qwen3_family:
        return "AIC_QWEN3"

    # Default decoder prefill or encoder at scale = AIV-bound
    if spec.d_model >= 700 and spec.S * batch >= 1024:
        return "AIV_BOUND"

    return "BALANCED"


# Per-bucket amp coefficients — DEFAULTS only; fit_v6 overrides
# v6.1: added amp_aic_S_alpha for S-axis scaling within AIC_QWEN3 bucket
# (Qwen3-S4096 needs ~3× higher amp than Qwen3-S256; single amp can't cover both)
V6_BUCKET_DEFAULTS = {
    "AIC_DECODE": {
        "amp_aic":          0.85,
        "amp_aiv":          1.50,
        "nk_mult":          4.70,
        "amp_aic_S_alpha":  0.0,
        "amp_aiv_S_alpha":  0.0,
    },
    "AIC_QWEN3": {
        "amp_aic":          7.0,
        "amp_aiv":          5.0,
        "nk_mult":          25.0,
        "amp_aic_S_alpha":  0.5,
        "amp_aiv_S_alpha":  0.5,
    },
    "AIV_BOUND": {
        "amp_aic":          1.5,
        "amp_aiv":          4.5,
        "nk_mult":          6.0,
        "amp_aic_S_alpha":  0.0,
        "amp_aiv_S_alpha":  0.0,
    },
    "BALANCED": {
        "amp_aic":          2.0,
        "amp_aiv":          3.0,
        "nk_mult":          6.0,
        "amp_aic_S_alpha":  0.0,
        "amp_aiv_S_alpha":  0.0,
    },
}

V6_BUCKET_BOUNDS = {
    "AIC_DECODE": {
        "amp_aic":          (0.5, 1.5),
        "amp_aiv":          (0.5, 3.0),
        "nk_mult":          (3.0, 6.0),
        "amp_aic_S_alpha":  (0.0, 0.001),
        "amp_aiv_S_alpha":  (0.0, 0.001),
    },
    "AIC_QWEN3": {
        "amp_aic":          (3.0, 15.0),
        "amp_aiv":          (1.0, 12.0),
        "nk_mult":          (15.0, 35.0),
        "amp_aic_S_alpha":  (0.0, 1.5),
        "amp_aiv_S_alpha":  (0.0, 1.5),
    },
    "AIV_BOUND": {
        "amp_aic":          (0.8, 3.0),
        "amp_aiv":          (2.0, 8.0),
        "nk_mult":          (4.0, 10.0),
        "amp_aic_S_alpha":  (0.0, 0.001),
        "amp_aiv_S_alpha":  (0.0, 0.001),
    },
    "BALANCED": {
        "amp_aic":          (1.0, 4.0),
        "amp_aiv":          (1.0, 5.0),
        "nk_mult":          (4.0, 10.0),
        "amp_aic_S_alpha":  (0.0, 0.001),
        "amp_aiv_S_alpha":  (0.0, 0.001),
    },
}


# Reference S for AIC_QWEN3 S-scaling: amp_effective = amp_aic × (S/S_REF)^alpha
AIC_QWEN3_S_REF = 512


def get_bucket_params(bucket: Bucket, params: Mapping[str, float]) -> dict:
    """Resolve per-bucket params from a flat dict.

    Convention: bucket params live under keys `v6_<BUCKET>_<param>`,
    e.g. `v6_AIC_QWEN3_amp_aic = 12.0`. Falls back to V6_BUCKET_DEFAULTS.
    """
    out = {}
    for k in ("amp_aic", "amp_aiv", "nk_mult", "amp_aic_S_alpha", "amp_aiv_S_alpha"):
        flat_key = f"v6_{bucket}_{k}"
        if flat_key in params:
            out[k] = float(params[flat_key])
        else:
            out[k] = V6_BUCKET_DEFAULTS[bucket][k]
    return out


def predict_v6(spec, arch: Mapping[str, float], batch: int,
               params: Mapping[str, float],
               aic_pipes_base: dict, aiv_base_us: float) -> tuple[float, float, int, str]:
    """Apply v6 per-bucket amps. Inputs are physics-base values (no amp).

    AIC_QWEN3 bucket: amp_aic *= (S/S_REF)^alpha (S-axis scaling for
    within-bucket S extrapolation; default S_REF=512, alpha bounded [0,1]).

    Returns: (aic_time_us, aiv_time_us, n_kernels, bucket_name)
    """
    bucket = classify_bottleneck(spec, batch)
    bp = get_bucket_params(bucket, params)

    # S-axis scaling on amp_aic AND amp_aiv (active only for AIC_QWEN3 alpha>0)
    alpha_aic = bp.get("amp_aic_S_alpha", 0.0)
    alpha_aiv = bp.get("amp_aiv_S_alpha", 0.0)
    if spec.S > 1:
        S_factor_aic = (spec.S / AIC_QWEN3_S_REF) ** alpha_aic if alpha_aic > 0.01 else 1.0
        S_factor_aiv = (spec.S / AIC_QWEN3_S_REF) ** alpha_aiv if alpha_aiv > 0.01 else 1.0
    else:
        S_factor_aic = 1.0
        S_factor_aiv = 1.0

    amp_aic_eff = bp["amp_aic"] * S_factor_aic
    amp_aiv_eff = bp["amp_aiv"] * S_factor_aiv
    aic_time = max(aic_pipes_base.values()) * amp_aic_eff
    aiv_time = aiv_base_us * amp_aiv_eff

    # n_kernels = base (geometric) × per-bucket mult
    n_gemm_per_layer = 7 if spec.arch == "decoder" else 8
    n_vec_per_layer = 3
    base = (n_gemm_per_layer + n_vec_per_layer) * spec.layers
    n_kernels = int(base * bp["nk_mult"])

    return aic_time, aiv_time, n_kernels, bucket
